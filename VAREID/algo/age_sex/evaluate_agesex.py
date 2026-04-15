import os
import json
import cv2
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import timm

from albumentations import Compose, Normalize, Resize
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ==========================================
# CONSTANTS & MAPPINGS
# ==========================================
AGE_MAP = {'0-2': 0, '3-5': 1, '6-11': 2, '12-23': 3, '24-35': 4, '36+': 5}
SEX_MAP = {'Female': 0, 'Male': 1}
REV_AGE = {v: k for k, v in AGE_MAP.items()}
REV_SEX = {v: k for k, v in SEX_MAP.items()}
IMG_SIZE = 224

# ==========================================
# DATASET & MODEL
# ==========================================
def load_coco_to_df(json_path):
    print(f"Loading {os.path.basename(json_path)}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    images_df = pd.DataFrame(data['images'])
    annots_df = pd.DataFrame(data['annotations'])
    df = pd.merge(annots_df, images_df, left_on='image_uuid', right_on='uuid', suffixes=('', '_img'))
    return df

class GrevyMultiTaskDataset(Dataset):
    def __init__(self, df, cache_dir, transforms=None):
        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.cache_dir = cache_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.loc[index]
        img_path = os.path.join(self.cache_dir, f"{row['uuid']}.jpg")
        img = cv2.imread(img_path)
        
        if img is None:
            img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        else:
            img = img[:, :, ::-1] 
            
        if self.transforms:
            img = self.transforms(image=img)["image"]
            
        targets = {
            'age': torch.tensor(AGE_MAP[row['age']], dtype=torch.long),
            'sex': torch.tensor(SEX_MAP[row['sex']], dtype=torch.long)
        }
        return img, targets

class ZebraMultiTaskNet(nn.Module):
    def __init__(self, model_arch='efficientnet_b0', num_ages=6, num_sexes=2):
        super().__init__()
        self.backbone = timm.create_model(model_arch, pretrained=False, num_classes=0)
        n_features = self.backbone.num_features
        self.age_head = nn.Linear(n_features, num_ages)
        self.sex_head = nn.Linear(n_features, num_sexes)

    def forward(self, x):
        features = self.backbone(x)
        return self.age_head(features), self.sex_head(features)

# ==========================================
# EVALUATION ENGINES
# ==========================================
@torch.no_grad()
def generate_prediction_df(model, dataloader, df, split_name, device):
    results = []
    start_idx = 0 
    
    for imgs, targets in tqdm(dataloader, total=len(dataloader), desc=f"Inference: {split_name}"):
        imgs = imgs.to(device).float()
        
        age_logits, sex_logits = model(imgs)
        age_probs = F.softmax(age_logits, dim=1).cpu().numpy()
        sex_probs = F.softmax(sex_logits, dim=1).cpu().numpy()
        
        age_preds = np.argmax(age_probs, axis=1)
        sex_preds = np.argmax(sex_probs, axis=1)
        
        batch_len = imgs.size(0)
        batch_df = df.iloc[start_idx : start_idx + batch_len]
        
        for i in range(batch_len):
            row = batch_df.iloc[i]
            
            qual = row.get('quality', 1.0)
            if pd.isna(qual): qual = 1.0
                
            res = {
                'uuid': row['uuid'],
                'cluster_id': row['cluster_id'],
                'split': split_name,
                'true_age': targets['age'][i].item(),
                'true_sex': targets['sex'][i].item(),
                'pred_age': age_preds[i],
                'pred_sex': sex_preds[i],
                'quality': qual 
            }
            for a in range(6): res[f'age_prob_{a}'] = age_probs[i][a]
            for s in range(2): res[f'sex_prob_{s}'] = sex_probs[i][s]
            results.append(res)
            
        start_idx += batch_len
            
    return pd.DataFrame(results)

def calculate_best_chip_selection(df, title):
    print(f"\n{'='*60}\n{title.upper()} (MAX CONFIDENCE SELECTION)\n{'='*60}")
    
    # 1. Base Annotation Output
    print("\n--- PER-ANNOTATION METRICS ---")
    print("\n[AGE] Classification Report:")
    print(classification_report(df['true_age'], df['pred_age'], target_names=list(AGE_MAP.keys()), zero_division=0))
    print("\n[SEX] Classification Report:")
    print(classification_report(df['true_sex'], df['pred_sex'], target_names=list(SEX_MAP.keys()), zero_division=0))

    # 2. Cluster Level Logic (Max Score)
    working_df = df.copy()
    
    age_cols = [f'age_prob_{i}' for i in range(6)]
    sex_cols = [f'sex_prob_{i}' for i in range(2)]
    
    working_df['age_conf'] = working_df[age_cols].max(axis=1)
    working_df['sex_conf'] = working_df[sex_cols].max(axis=1)
    
    working_df['age_score'] = working_df['age_conf'] * working_df['quality']
    working_df['sex_score'] = working_df['sex_conf'] * working_df['quality']
    
    best_age_idx = working_df.groupby('cluster_id')['age_score'].idxmax()
    best_sex_idx = working_df.groupby('cluster_id')['sex_score'].idxmax()
    
    cluster_df = pd.DataFrame({
        'cluster_id': working_df.loc[best_age_idx, 'cluster_id'].values,
        'true_age': working_df.loc[best_age_idx, 'true_age'].values,
        'true_sex': working_df.loc[best_age_idx, 'true_sex'].values, 
        'vote_pred_age': working_df.loc[best_age_idx, 'pred_age'].values, 
        'vote_pred_sex': working_df.loc[best_sex_idx, 'pred_sex'].values  
    })
    
    print(f"\n--- PER-CLUSTER METRICS (N={len(cluster_df)}) ---")
    print("\n[AGE] Classification Report:")
    print(classification_report(cluster_df['true_age'], cluster_df['vote_pred_age'], target_names=list(AGE_MAP.keys()), zero_division=0))
    print("\n[SEX] Classification Report:")
    print(classification_report(cluster_df['true_sex'], cluster_df['vote_pred_sex'], target_names=list(SEX_MAP.keys()), zero_division=0))
    
    return cluster_df

# ==========================================
# VISUALIZATION UTILS
# ==========================================
def plot_matrices(df, title_prefix, out_dir, is_cluster=False):
    age_labels = list(AGE_MAP.keys())
    sex_labels = list(SEX_MAP.keys())
    
    p_age = 'vote_pred_age' if is_cluster else 'pred_age'
    p_sex = 'vote_pred_sex' if is_cluster else 'pred_sex'
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    cm_age = confusion_matrix(df['true_age'], df[p_age])
    sns.heatmap(cm_age, annot=True, fmt='d', cmap='Blues', xticklabels=age_labels, yticklabels=age_labels, ax=axes[0])
    axes[0].set_title(f'{title_prefix} - Age Confusion')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')
    
    cm_sex = confusion_matrix(df['true_sex'], df[p_sex])
    sns.heatmap(cm_sex, annot=True, fmt='d', cmap='Oranges', xticklabels=sex_labels, yticklabels=sex_labels, ax=axes[1])
    axes[1].set_title(f'{title_prefix} - Sex Confusion')
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    save_path = os.path.join(out_dir, f"{title_prefix.replace(' ', '_')}_Confusion.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved Confusion Matrix to {save_path}")

def plot_tp_tn_fp_fn_examples(df, task, cache_dir, out_dir):
    """
    Finds one TP, TN, FP, and FN for every class and plots them in a 1x4 grid.
    Saves one image per class.
    """
    mapping = AGE_MAP if task == 'age' else SEX_MAP
    inv_map = REV_AGE if task == 'age' else REV_SEX
    true_col = f'true_{task}'
    pred_col = f'pred_{task}'
    
    for class_name, class_idx in mapping.items():
        # Condition definitions
        conditions = {
            'True Positive (TP)': (df[true_col] == class_idx) & (df[pred_col] == class_idx),
            'True Negative (TN)': (df[true_col] != class_idx) & (df[pred_col] != class_idx),
            'False Positive (FP)': (df[true_col] != class_idx) & (df[pred_col] == class_idx), # Predicted class, but it isn't
            'False Negative (FN)': (df[true_col] == class_idx) & (df[pred_col] != class_idx)  # Is class, but predicted otherwise
        }
        
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        fig.suptitle(f"Task: {task.upper()} | Focus Class: {class_name}", fontsize=16, weight='bold')
        
        for ax, (cond_name, cond_mask) in zip(axes, conditions.items()):
            subset = df[cond_mask]
            
            if len(subset) == 0:
                # No examples exist (e.g. model never predicted this class falsely)
                ax.text(0.5, 0.5, f"No {cond_name.split(' ')[0]}\nFound", ha='center', va='center', fontsize=12)
                ax.axis('off')
                ax.set_title(cond_name)
                continue
                
            # Sample 1 random row
            sample = subset.sample(1).iloc[0]
            
            # Load Image
            img_path = os.path.join(cache_dir, f"{sample['uuid']}.jpg")
            img = cv2.imread(img_path)
            img = img[:, :, ::-1] if img is not None else np.zeros((224, 224, 3), dtype=np.uint8)
            
            ax.imshow(img)
            ax.axis('off')
            
            # Formatting labels
            t_label = inv_map[sample[true_col]]
            p_label = inv_map[sample[pred_col]]
            
            color = 'green' if t_label == p_label else 'red'
            ax.set_title(f"{cond_name}\nTrue: {t_label} | Pred: {p_label}", color=color)

        plt.tight_layout()
        save_path = os.path.join(out_dir, f"Examples_{task.upper()}_{class_name}.png")
        plt.savefig(save_path)
        plt.close()

# ==========================================
# MAIN
# ==========================================
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(args.out_dir, exist_ok=True)
    cache_dir = os.environ.get('TMPDIR', '/tmp')

    # Load Model
    model = ZebraMultiTaskNet().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"\nModel loaded successfully from Epoch {checkpoint.get('epoch', 'Unknown')}.")

    # Load Data
    test_df = load_coco_to_df(args.test_json)
    
    eval_transforms = Compose([
        Resize(IMG_SIZE, IMG_SIZE),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    test_loader = DataLoader(
        GrevyMultiTaskDataset(test_df, cache_dir, eval_transforms), 
        batch_size=64, shuffle=False, num_workers=12, pin_memory=True
    )

    # 1. Run Inference
    test_res = generate_prediction_df(model, test_loader, test_df, "Test", device)

    # 2. Evaluate & Print Metrics
    test_clusters = calculate_best_chip_selection(test_res, "Test Set")

    # 3. Generate Confusion Matrices
    print("\nGenerating Plots...")
    plot_matrices(test_res, "Test (Per-Annotation)", args.out_dir, is_cluster=False)
    plot_matrices(test_clusters, "Test (Cluster Voted)", args.out_dir, is_cluster=True)

    # 4. Generate TP/TN/FP/FN Examples
    plot_tp_tn_fp_fn_examples(test_res, task='age', cache_dir=cache_dir, out_dir=args.out_dir)
    plot_tp_tn_fp_fn_examples(test_res, task='sex', cache_dir=cache_dir, out_dir=args.out_dir)
    print(f"Evaluation complete. All results and plots saved to: {args.out_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Age and Sex Classification')
    parser.add_argument('--test_json', type=str, required=True, help='Path to test JSON split')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to best_baseline.pth')
    parser.add_argument('--out_dir', type=str, default='./eval_results', help='Directory to save output plots')
    
    args = parser.parse_args()
    main(args)