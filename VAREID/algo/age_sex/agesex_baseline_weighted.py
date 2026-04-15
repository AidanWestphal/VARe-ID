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
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.models import efficientnet_b0
import timm

from albumentations import Compose, Normalize, Resize, RandomBrightnessContrast, Rotate, HorizontalFlip, ShiftScaleRotate
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
def load_coco_to_df(json_path):
    print(f"Loading {os.path.basename(json_path)}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    images_df = pd.DataFrame(data['images'])
    annots_df = pd.DataFrame(data['annotations'])
    
    df = pd.merge(annots_df, images_df, left_on='image_uuid', right_on='uuid', suffixes=('', '_img'))
    return df

class GrevyMultiTaskDataset(Dataset):
    def __init__(self, df, cache_dir, transforms=None, is_inference=False):
        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.cache_dir = cache_dir
        self.transforms = transforms
        self.is_inference = is_inference

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
            
        if self.is_inference:
            return img
            
        targets = {
            'age': torch.tensor(AGE_MAP[row['age']], dtype=torch.long),
            'sex': torch.tensor(SEX_MAP[row['sex']], dtype=torch.long)
        }
        
        return img, targets

# ==========================================
# MODEL ARCHITECTURE
# ==========================================
class ZebraMultiTaskNet(nn.Module):
    def __init__(self, model_arch='efficientnet_b0', num_ages=6, num_sexes=2, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(model_arch, pretrained=pretrained, num_classes=0)
        n_features = self.backbone.num_features
        
        self.age_head = nn.Linear(n_features, num_ages)
        self.sex_head = nn.Linear(n_features, num_sexes)

    def forward(self, x):
        features = self.backbone(x)
        age_logits = self.age_head(features)
        sex_logits = self.sex_head(features)
        return age_logits, sex_logits

# ==========================================
# TRAINING & EVALUATION LOOPS
# ==========================================
def train_one_epoch_profiled(model, dataloader, optimizer, criterion_age, criterion_sex, device, verbose=True):
    model.train()
    running_loss = 0.0
    
    total_data_time, total_forward_time, total_backward_time = 0.0, 0.0, 0.0
    pbar = tqdm(dataloader, desc="Training", disable=not verbose)
    
    start_time = time.time() 
    
    for imgs, targets in pbar:
        data_time = time.time() - start_time
        total_data_time += data_time
        
        imgs = imgs.to(device).float()
        age_targets = targets['age'].to(device)
        sex_targets = targets['sex'].to(device)
        
        fwd_start = time.time()
        optimizer.zero_grad()
        age_logits, sex_logits = model(imgs)
        loss_age = criterion_age(age_logits, age_targets)
        loss_sex = criterion_sex(sex_logits, sex_targets)
        loss = loss_age + loss_sex
        forward_time = time.time() - fwd_start
        total_forward_time += forward_time
        
        bwd_start = time.time()
        loss.backward()
        optimizer.step()
        backward_time = time.time() - bwd_start
        total_backward_time += backward_time
        
        running_loss += loss.item()
        
        if verbose:
            pbar.set_postfix({
                'Data(s)': f"{data_time:.2f}", 
                'Fwd(s)': f"{forward_time:.2f}", 
                'Bwd(s)': f"{backward_time:.2f}"
            })
            
        start_time = time.time()
        
    return running_loss / len(dataloader)

@torch.no_grad()
def evaluate(model, dataloader, criterion_age, criterion_sex, device, verbose=True):
    model.eval()
    running_loss = 0.0
    
    all_age_preds, all_age_targets = [], []
    all_sex_preds, all_sex_targets = [], []
    
    for imgs, targets in tqdm(dataloader, desc="Evaluating", disable=not verbose):
        imgs = imgs.to(device).float()
        age_targets = targets['age'].to(device)
        sex_targets = targets['sex'].to(device)
        
        age_logits, sex_logits = model(imgs)
        
        loss_age = criterion_age(age_logits, age_targets)
        loss_sex = criterion_sex(sex_logits, sex_targets)
        running_loss += (loss_age + loss_sex).item()
        
        age_preds = torch.argmax(age_logits, dim=1)
        sex_preds = torch.argmax(sex_logits, dim=1)
        
        all_age_preds.extend(age_preds.cpu().numpy())
        all_age_targets.extend(age_targets.cpu().numpy())
        all_sex_preds.extend(sex_preds.cpu().numpy())
        all_sex_targets.extend(sex_targets.cpu().numpy())
        
    avg_loss = running_loss / len(dataloader)
    
    print("\n" + "="*50)
    print("AGE CLASSIFICATION REPORT")
    print("="*50)
    print(classification_report(all_age_targets, all_age_preds, target_names=list(AGE_MAP.keys()), zero_division=0))
    
    print("\n" + "="*50)
    print("SEX CLASSIFICATION REPORT")
    print("="*50)
    print(classification_report(all_sex_targets, all_sex_preds, target_names=list(SEX_MAP.keys()), zero_division=0))
    
    # We return avg_loss to drive the checkpoint saving
    # Can return macro F1 scores here if you want to save based on F1 instead of loss
    return avg_loss

# ==========================================
# MAIN EXECUTION
# ==========================================
def main(args):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(script_dir, 'agesex_baseline.yaml')
    
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
        
    run_name = config.get('run_name', 'default_model')
    epochs = config.get('epochs', 50)
    batch_size = config.get('batch_size', 32)
    lr = config.get('lr', 1e-4)
    verbose = config.get('verbose', True)
    resume = config.get('resume', False)
    
    if args.resume: resume = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    slurm_cores = os.environ.get('SLURM_CPUS_PER_TASK')
    num_workers = int(slurm_cores) if slurm_cores else os.cpu_count()
    prefetch_factor = num_workers if num_workers > 0 else None

    tmpdir = os.environ.get('TMPDIR', '/tmp')
    cache_dir = tmpdir 
    
    print(f"\nExtracting {args.tar_path} to {tmpdir}...")
    subprocess.run(['tar', '-xf', args.tar_path, '-C', tmpdir], check=True)

    # --- 3. Data Loading & Sub-sampling ---
    train_df = load_coco_to_df(args.train_json)
    val_df = load_coco_to_df(args.val_json)
    
    # Randomly sample max 5 chips per cluster to prevent identity shortcut
    train_df = train_df.groupby('cluster_id', group_keys=False).apply(
        lambda x: x.sample(min(len(x), 5), random_state=42)
    ).reset_index(drop=True)
    
    # Create Composite Class for Multi-Task Balancing
    train_df['balance_class'] = train_df['age'] + "_" + train_df['sex']
    class_counts = train_df['balance_class'].value_counts()
    print(f"\nTraining Class Distribution (Max 5 chips/cluster):\n{class_counts}\n")
    
    # Calculate inverted weights for the sampler
    sample_weights = train_df['balance_class'].map(1.0 / class_counts).values
    
    train_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=5000, 
        replacement=True 
    )
    
    train_transforms = Compose([
        ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5), # Soft cropping & rotation
        HorizontalFlip(p=0.5), 
        RandomBrightnessContrast(p=0.2),
        Resize(IMG_SIZE, IMG_SIZE),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    valid_transforms = Compose([
        Resize(IMG_SIZE, IMG_SIZE),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    loader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': True
    }
    if num_workers > 0:
        loader_kwargs['prefetch_factor'] = prefetch_factor
        loader_kwargs['persistent_workers'] = True

    train_ds = GrevyMultiTaskDataset(train_df, cache_dir, transforms=train_transforms)
    val_ds = GrevyMultiTaskDataset(val_df, cache_dir, transforms=valid_transforms)

    # SHUFFLE MUST BE FALSE WHEN USING A SAMPLER
    train_loader = DataLoader(train_ds, sampler=train_sampler, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    # --- 4. Model & Optimizers ---
    model = ZebraMultiTaskNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    criterion_age = nn.CrossEntropyLoss()
    criterion_sex = nn.CrossEntropyLoss() 

    # --- 5. Checkpoint Resumption ---
    start_epoch = 0
    best_val_loss = float('inf')
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(args.checkpoint_dir, f"best_{run_name}.pth")
    last_model_path = os.path.join(args.checkpoint_dir, f"last_{run_name}.pth")

    if resume and os.path.exists(last_model_path):
        checkpoint = torch.load(last_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('val_loss', float('inf')) 
        print(f"Resumed successfully. Starting at epoch {start_epoch+1}.\n")

    # --- 6. Training Loop ---
    for epoch in range(start_epoch, epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        train_loss = train_one_epoch_profiled(
            model, train_loader, optimizer, criterion_age, criterion_sex, device, verbose=verbose
        )
        val_loss = evaluate(
            model, val_loader, criterion_age, criterion_sex, device, verbose=verbose
        )
        
        if val_loss < best_val_loss:
            print(f"Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}. Saving best model...")
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss
            }, best_model_path)
            
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': best_val_loss 
        }, last_model_path)

    print("\nTraining Complete!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-Task Age and Sex Classifier Training')
    parser.add_argument('--train_json', type=str, required=True, help='Path to train JSON split')
    parser.add_argument('--val_json', type=str, required=True, help='Path to validation JSON split')
    parser.add_argument('--tar_path', type=str, required=True, help='Path to the network .tar file containing image chips')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Directory to save/load model states')
    parser.add_argument('--resume', action='store_true', help='Override YAML to force resume from checkpoint_dir')
    args = parser.parse_args()
    main(args)