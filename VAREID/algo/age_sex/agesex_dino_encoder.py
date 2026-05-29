import os
import time
import json
import argparse
import subprocess
import pandas as pd
import numpy as np
import cv2
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from albumentations import Compose, Normalize, Resize, HorizontalFlip, ShiftScaleRotate, ColorJitter
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from sklearn.metrics import classification_report

# ==========================================
# CONSTANTS & MAPPINGS
# ==========================================
AGE_MAP = {'0-2': 0, '3-5': 1, '6-11': 2, '12-23': 3, '24-35': 4, '36+': 5}
SEX_MAP = {'Female': 0, 'Male': 1}
IMG_SIZE = 224

# ==========================================
# DATA LOADING & DATASET
# ==========================================
# (IDENTICAL TO BASELINE - OMITTED FOR SPACE, COPY/PASTE load_coco_to_df AND DINOZebraDataset FROM ABOVE)
def load_coco_to_df(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return pd.merge(pd.DataFrame(data['annotations']), pd.DataFrame(data['images']), left_on='image_uuid', right_on='uuid', suffixes=('', '_img'))

class DINOZebraDataset(Dataset):
    def __init__(self, df, cache_dir, task='age', transforms=None):
        self.df = df.reset_index(drop=True).copy(); self.cache_dir = cache_dir; self.task = task; self.transforms = transforms; self.map = AGE_MAP if task == 'age' else SEX_MAP
    def __len__(self): return len(self.df)
    def __getitem__(self, index):
        row = self.df.loc[index]; img_path = os.path.join(self.cache_dir, f"{row['uuid']}.jpg"); img = cv2.imread(img_path); img = img[:, :, ::-1] if img is not None else np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        if self.transforms: img = self.transforms(image=img)["image"]
        return img, torch.tensor(self.map[row[self.task]], dtype=torch.long)

# ==========================================
# MODEL ARCHITECTURE (TRANSFORMER POOL)
# ==========================================

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss
    

class LightweightTransformerPool(nn.Module):
    def __init__(self, in_dim, embed_dim=128, num_heads=4, depth=1):
        super().__init__()
        self.proj = nn.Linear(in_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, 
            dim_feedforward=embed_dim * 2, dropout=0.3, 
            batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x):
        x = self.proj(x)
        refined_tokens = self.transformer(x)
        pooled_features = refined_tokens.mean(dim=1) 
        return pooled_features

class DINOProbe(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        in_dim = self.backbone.embed_dim
        bottleneck_dim = 128
        
        self.pooler = LightweightTransformerPool(in_dim=in_dim, embed_dim=bottleneck_dim)
        
        self.fc = nn.Sequential(
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(p=0.4), 
            nn.Linear(bottleneck_dim, num_classes)
        )

    def forward(self, x):
        with torch.no_grad():
            features = self.backbone.forward_features(x)
            patch_tokens = features['x_norm_patchtokens'] 
            
        pooled_features = self.pooler(patch_tokens)
        return self.fc(pooled_features)

# ==========================================
# TRAINING & EVALUATION LOOPS
# ==========================================
# (IDENTICAL TO BASELINE - COPY/PASTE create_sampler, train, evaluate, train_epochs FROM ABOVE)
def create_sampler(df, target_col):
    class_counts = df[target_col].value_counts()
    weights = df[target_col].map(1.0 / class_counts).values
    return WeightedRandomSampler(weights=weights, num_samples=5000, replacement=True)


def train(model, optim, criterion, dataloader, device, verbose=True):
    model.train()
    running_loss = 0.0
    pbar = tqdm(dataloader, desc="Training", disable=not verbose)
    for imgs, targets in pbar:
        imgs, targets = imgs.to(device).float(), targets.to(device)
        optim.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, targets)
        loss.backward()
        
        # ADDED: Gradient clipping to stabilize the learning curves
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optim.step()
        running_loss += loss.item()
        if verbose:
            pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
    return running_loss / len(dataloader)


@torch.no_grad()
def evaluate(model, dataloader, criterion, task_name, device, verbose=True):
    model.eval()
    running_loss = 0.0
    all_preds, all_targets = [], []
    for imgs, targets in tqdm(dataloader, desc="Evaluating", disable=not verbose):
        imgs, targets = imgs.to(device).float(), targets.to(device)
        logits = model(imgs)
        preds = torch.argmax(logits, dim=1)
        loss = criterion(logits, targets)
        running_loss += loss.item()
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
            
    avg_loss = running_loss / len(dataloader)
    target_names = list(AGE_MAP.keys()) if task_name == 'age' else list(SEX_MAP.keys())
    cr_dict = classification_report(all_targets, all_preds, target_names=target_names, zero_division=0, output_dict=True)
    if verbose:
        print("\n" + "="*50 + f"\n{task_name.upper()} CLASSIFICATION REPORT\n" + "="*50)
        print(classification_report(all_targets, all_preds, target_names=target_names, zero_division=0))
        print(f"AVERAGE LOSS: {avg_loss:.4f}")
    
    return cr_dict['macro avg']['f1-score'], avg_loss


def train_epochs(model, optim, train_dataloader, val_dataloader, criterion, task_name, device, checkpoint_dir, run_name, start_epoch=0, epochs=10, best_metric=0.0, verbose=True):
    best_model_path = os.path.join(checkpoint_dir, f"best_{task_name}_{run_name}.pth")
    last_model_path = os.path.join(checkpoint_dir, f"last_{task_name}_{run_name}.pth")
    history_path = os.path.join(checkpoint_dir, f"history_{task_name}_{run_name}.json")

    history = {'train_loss': [], 'val_loss': [], 'val_f1': []}
    if start_epoch > 0 and os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history = json.load(f)

    for epoch in range(start_epoch, epochs):
        print(f"\n--- {task_name.upper()} PROBE: Epoch {epoch+1}/{epochs} ---")
        train_loss = train(model, optim, criterion, train_dataloader, device, verbose=verbose)
        val_macro_f1, val_loss = evaluate(model, val_dataloader, criterion, task_name, device, verbose=verbose)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_macro_f1)
        
        if val_macro_f1 > best_metric:
            print(f"Validation F1 improved to {val_macro_f1:.4f}. Saving best model...")
            best_metric = val_macro_f1
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optim.state_dict(), 'val_f1': best_metric}, best_model_path)
            
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optim.state_dict(), 'val_f1': best_metric}, last_model_path)
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=4)
        
    print(f"Finished training {task_name.upper()}. Best Macro F1: {best_metric:.4f}\n")


# ==========================================
# MAIN EXECUTION
# ==========================================
def main(args):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(script_dir, 'agesex_dino.yaml')
    with open(yaml_path, 'r') as f: config = yaml.safe_load(f)
        
    run_name = "encoder_" + config.get('run_name', 'patch_encoder')
    epochs = config.get('epochs', 50); batch_size = config.get('batch_size', 32); lr = config.get('lr', 1e-4)
    verbose = config.get('verbose', True); resume = config.get('resume', False) or args.resume

    model_name = config.get('dino_model', 'dinov3_vit7b16')
    model_dir = config.get('dino_repo', './dino')
    if not os.path.exists(model_dir): subprocess.run(["git", "clone", config.get('dino_github_url', ''), model_dir], check=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading DINOv3 backbone...")
    dinov3 = torch.hub.load(model_dir, model_name, source='local', weights=config.get('dino_weights')[model_name]).to(device)
    for param in dinov3.parameters(): param.requires_grad = False
    dinov3.eval()

    slurm_cores = os.environ.get('SLURM_CPUS_PER_TASK')
    num_workers = int(slurm_cores) if slurm_cores else os.cpu_count()
    prefetch_factor = num_workers if num_workers > 0 else None
    tmpdir = os.environ.get('TMPDIR', '/tmp')
    
    print(f"\nExtracting {args.tar_path} to {tmpdir}...")
    subprocess.run(['tar', '-xf', args.tar_path, '-C', tmpdir], check=True)

    train_df = load_coco_to_df(args.train_json)
    val_df = load_coco_to_df(args.val_json)

    train_df = train_df.groupby('cluster_id').apply(lambda x: x.sample(min(len(x), 5), random_state=42), include_groups=False).reset_index(drop=True)

    train_transforms = Compose([
        ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
        HorizontalFlip(p=0.5), 
        # Replaced CoarseDropout with mild ColorJitter to simulate lighting
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.05, hue=0.0, p=0.5),
        Resize(IMG_SIZE, IMG_SIZE),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    eval_transforms = Compose([
        Resize(IMG_SIZE, IMG_SIZE),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    loader_kwargs = {'batch_size': batch_size, 'num_workers': num_workers, 'pin_memory': True}
    if num_workers > 0: loader_kwargs['prefetch_factor'] = prefetch_factor
        
    age_train_loader = DataLoader(DINOZebraDataset(train_df, tmpdir, task='age', transforms=train_transforms), sampler=create_sampler(train_df, 'age'), **loader_kwargs)
    sex_train_loader = DataLoader(DINOZebraDataset(train_df, tmpdir, task='sex', transforms=train_transforms), sampler=create_sampler(train_df, 'sex'), **loader_kwargs)
    age_val_loader = DataLoader(DINOZebraDataset(val_df, tmpdir, task='age', transforms=eval_transforms), shuffle=False, **loader_kwargs)
    sex_val_loader = DataLoader(DINOZebraDataset(val_df, tmpdir, task='sex', transforms=eval_transforms), shuffle=False, **loader_kwargs)

    age_model = DINOProbe(dinov3, num_classes=6).to(device)
    sex_model = DINOProbe(dinov3, num_classes=2).to(device)

    age_optimizer = torch.optim.AdamW(list(age_model.pooler.parameters()) + list(age_model.fc.parameters()), lr=lr)
    sex_optimizer = torch.optim.AdamW(list(sex_model.pooler.parameters()) + list(sex_model.fc.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()

    start_epoch_age, start_epoch_sex, best_f1_age, best_f1_sex = 0, 0, 0.0, 0.0
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    last_age_model_path = os.path.join(args.checkpoint_dir, f"last_age_{run_name}.pth")
    last_sex_model_path = os.path.join(args.checkpoint_dir, f"last_sex_{run_name}.pth")
    
    if resume:
        if os.path.exists(last_age_model_path):
            checkpoint = torch.load(last_age_model_path, map_location=device)
            age_model.load_state_dict(checkpoint['model_state_dict'])
            age_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch_age = checkpoint['epoch'] + 1
            best_f1_age = checkpoint.get('val_f1', 0.0) 
            print(f"Resumed age model successfully. Starting at epoch {start_epoch_age+1}.\n")
            
        if os.path.exists(last_sex_model_path):
            checkpoint = torch.load(last_sex_model_path, map_location=device)
            sex_model.load_state_dict(checkpoint['model_state_dict'])
            sex_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch_sex = checkpoint['epoch'] + 1
            best_f1_sex = checkpoint.get('val_f1', 0.0) 
            print(f"Resumed sex model successfully. Starting at epoch {start_epoch_sex+1}.\n")

    print("\n" + "="*40 + "\nTRAINING AGE PROBE\n" + "="*40)
    train_epochs(age_model, age_optimizer, age_train_loader, age_val_loader, criterion, 'age', device, args.checkpoint_dir, run_name, start_epoch_age, epochs, best_f1_age, verbose)

    print("\n" + "="*40 + "\nTRAINING SEX PROBE\n" + "="*40)
    train_epochs(sex_model, sex_optimizer, sex_train_loader, sex_val_loader, criterion, 'sex', device, args.checkpoint_dir, run_name, start_epoch_sex, epochs, best_f1_sex, verbose)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_json', type=str, required=True)
    parser.add_argument('--val_json', type=str, required=True)
    parser.add_argument('--tar_path', type=str, required=True)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    main(args)