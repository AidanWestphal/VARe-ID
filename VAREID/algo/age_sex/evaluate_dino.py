import os
import json
import cv2
import argparse
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

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
# DATASET
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

# ==========================================
# DINO ARCHITECTURES
# ==========================================
class DINOProbeCLS(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.fc = nn.Sequential(
            nn.Linear(self.backbone.embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        with torch.no_grad():
            features = self.backbone.forward_features(x)
            cls_token = features['x_norm_clstoken'] 
        return self.fc(cls_token)

class LightweightTransformerPool(nn.Module):
    def __init__(self, in_dim, embed_dim=256, num_heads=4, depth=1):
        super().__init__()
        self.proj = nn.Linear(in_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, 
            dim_feedforward=embed_dim * 2, dropout=0.1, 
            batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x):
        x = self.proj(x)
        refined_tokens = self.transformer(x)
        return refined_tokens.mean(dim=1)

class DINOProbePatch(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        in_dim = self.backbone.embed_dim
        bottleneck_dim = 256
        self.pooler = LightweightTransformerPool(in_dim=in_dim, embed_dim=bottleneck_dim, num_heads=4, depth=1)
        self.fc = nn.Sequential(
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(), # <--- ADD THIS BACK IN FOR OLD CHECKPOINTS
            nn.Dropout(p=0.4), # <--- Ensure this matches what you trained with (likely 0.4 based on your previous message)
            nn.Linear(bottleneck_dim, num_classes)
        )

    def forward(self, x):
        with torch.no_grad():
            features = self.backbone.forward_features(x)
            patch_tokens = features['x_norm_patchtokens'] 
        pooled_features = self.pooler(patch_tokens)
        return self.fc(pooled_features)

# ==========================================
# EVALUATION ENGINES
# ==========================================
@torch.no_grad()
def generate_prediction_df(age_model, sex_model, dataloader, df, split_name, device):
    results = []
    start_idx = 0 
    
    for imgs, targets in tqdm(dataloader, total=len(dataloader), desc=f"Inference: {split_name}"):
        imgs = imgs.to(device).float()
        
        age_logits = age_model(imgs)
        sex_logits = sex_model(imgs)
        
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
    
    print("\n--- PER-ANNOTATION METRICS ---")
    print("\n[AGE] Classification Report:")
    print(classification_report(df['true_age'], df['pred_age'], target_names=list(AGE_MAP.keys()), zero_division=0))
    print("\n[SEX] Classification Report:")
    print(classification_report(df['true_sex'], df['pred_sex'], target_names=list(SEX_MAP.keys()), zero_division=0))

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

def plot_tp_tn_fp_fn_examples(df, task, cache_dir, out_dir, prefix="Test"):
    mapping = AGE_MAP if task == 'age' else SEX_MAP
    inv_map = REV_AGE if task == 'age' else REV_SEX
    true_col = f'true_{task}'
    pred_col = f'pred_{task}'
    
    for class_name, class_idx in mapping.items():
        conditions = {
            'True Positive (TP)': (df[true_col] == class_idx) & (df[pred_col] == class_idx),
            'True Negative (TN)': (df[true_col] != class_idx) & (df[pred_col] != class_idx),
            'False Positive (FP)': (df[true_col] != class_idx) & (df[pred_col] == class_idx), 
            'False Negative (FN)': (df[true_col] == class_idx) & (df[pred_col] != class_idx)  
        }
        
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        fig.suptitle(f"[{prefix}] Task: {task.upper()} | Focus Class: {class_name}", fontsize=16, weight='bold')
        
        for ax, (cond_name, cond_mask) in zip(axes, conditions.items()):
            subset = df[cond_mask]
            
            if len(subset) == 0:
                ax.text(0.5, 0.5, f"No {cond_name.split(' ')[0]}\nFound", ha='center', va='center', fontsize=12)
                ax.axis('off')
                ax.set_title(cond_name)
                continue
                
            sample = subset.sample(1).iloc[0]
            
            img_path = os.path.join(cache_dir, f"{sample['uuid']}.jpg")
            img = cv2.imread(img_path)
            img = img[:, :, ::-1] if img is not None else np.zeros((224, 224, 3), dtype=np.uint8)
            
            ax.imshow(img)
            ax.axis('off')
            
            t_label = inv_map[sample[true_col]]
            p_label = inv_map[sample[pred_col]]
            
            color = 'green' if t_label == p_label else 'red'
            ax.set_title(f"{cond_name}\nTrue: {t_label} | Pred: {p_label}", color=color)

        plt.tight_layout()
        save_path = os.path.join(out_dir, f"{prefix}_Examples_{task.upper()}_{class_name}.png")
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

    # Load Config
    script_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(script_dir, args.config)
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    dino_model_name = config.get('dino_model', 'dinov3_vitb16')
    dino_repo = config.get('dino_repo', '/fs/ess/PAS2136/ggr_data/dinov3')
    dino_weights = config.get('dino_weights', {}).get(dino_model_name, None)

    # Load DINO Backbone
    print(f"Loading {dino_model_name} backbone from {dino_repo}...")
    dinov3 = torch.hub.load(dino_repo, dino_model_name, source='local', weights=dino_weights).to(device)
    dinov3.eval()

    # Load Head Models based on Type
    if args.model_type == 'cls':
        age_model = DINOProbeCLS(dinov3, num_classes=6).to(device)
        sex_model = DINOProbeCLS(dinov3, num_classes=2).to(device)
    else:
        age_model = DINOProbePatch(dinov3, num_classes=6).to(device)
        sex_model = DINOProbePatch(dinov3, num_classes=2).to(device)

    # Load Checkpoints
    age_checkpoint = torch.load(args.age_checkpoint, map_location=device)
    age_model.load_state_dict(age_checkpoint['model_state_dict'])
    age_model.eval()
    print(f"Age Model loaded successfully (Epoch {age_checkpoint.get('epoch', 'Unknown')}).")

    sex_checkpoint = torch.load(args.sex_checkpoint, map_location=device)
    sex_model.load_state_dict(sex_checkpoint['model_state_dict'])
    sex_model.eval()
    print(f"Sex Model loaded successfully (Epoch {sex_checkpoint.get('epoch', 'Unknown')}).")

    # Load Data
    test_df = load_coco_to_df(args.test_json)
    train_df = load_coco_to_df(args.train_json)
    
    # We use EVAL transforms for both, because we do NOT want random augmentations 
    # to artificially lower the training accuracy during evaluation.
    eval_transforms = Compose([
        Resize(IMG_SIZE, IMG_SIZE),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    loader_kwargs = {'batch_size': config.get('batch_size', 32), 'shuffle': False, 'num_workers': 12, 'pin_memory': True}
    
    test_loader = DataLoader(GrevyMultiTaskDataset(test_df, cache_dir, eval_transforms), **loader_kwargs)
    train_loader = DataLoader(GrevyMultiTaskDataset(train_df, cache_dir, eval_transforms), **loader_kwargs)

    # ==========================================
    # EVALUATE TEST SET
    # ==========================================
    test_res = generate_prediction_df(age_model, sex_model, test_loader, test_df, "Test", device)
    test_clusters = calculate_best_chip_selection(test_res, "Test Set")

    print("\nGenerating Test Plots...")
    plot_matrices(test_res, "Test (Per-Annotation)", args.out_dir, is_cluster=False)
    plot_matrices(test_clusters, "Test (Cluster Voted)", args.out_dir, is_cluster=True)
    plot_tp_tn_fp_fn_examples(test_res, task='age', cache_dir=cache_dir, out_dir=args.out_dir, prefix="Test")
    plot_tp_tn_fp_fn_examples(test_res, task='sex', cache_dir=cache_dir, out_dir=args.out_dir, prefix="Test")

    # ==========================================
    # EVALUATE TRAIN SET
    # ==========================================
    train_res = generate_prediction_df(age_model, sex_model, train_loader, train_df, "Train", device)
    train_clusters = calculate_best_chip_selection(train_res, "Train Set")

    print("\nGenerating Train Plots...")
    plot_matrices(train_res, "Train (Per-Annotation)", args.out_dir, is_cluster=False)
    plot_matrices(train_clusters, "Train (Cluster Voted)", args.out_dir, is_cluster=True)
    plot_tp_tn_fp_fn_examples(train_res, task='age', cache_dir=cache_dir, out_dir=args.out_dir, prefix="Train")
    plot_tp_tn_fp_fn_examples(train_res, task='sex', cache_dir=cache_dir, out_dir=args.out_dir, prefix="Train")

    print(f"\nEvaluation complete. All results and plots saved to: {args.out_dir}/")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate DINO Age and Sex Classification')
    parser.add_argument('--config', type=str, default='agesex_dino.yaml', help='Path to YAML config file')
    parser.add_argument('--train_json', type=str, required=True, help='Path to train JSON split')
    parser.add_argument('--test_json', type=str, required=True, help='Path to test JSON split')
    parser.add_argument('--age_checkpoint', type=str, required=True, help='Path to best_age_model.pth')
    parser.add_argument('--sex_checkpoint', type=str, required=True, help='Path to best_sex_model.pth')
    parser.add_argument('--model_type', type=str, choices=['cls', 'patch'], required=True, help='Which DINO architecture to evaluate')
    parser.add_argument('--out_dir', type=str, default='./eval_results', help='Directory to save output plots')
    
    args = parser.parse_args()
    main(args)