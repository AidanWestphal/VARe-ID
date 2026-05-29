import os
import cv2
import json
import yaml
import torch
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from albumentations import Compose, Normalize, Resize
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# ==========================================
# CONSTANTS & MAPPINGS
# ==========================================
AGE_MAP = {'0-2': 0, '3-5': 1, '6-11': 2, '12-23': 3, '24-35': 4, '36+': 5}
SEX_MAP = {'Female': 0, 'Male': 1}
REV_AGE = {v: k for k, v in AGE_MAP.items()}
REV_SEX = {v: k for k, v in SEX_MAP.items()}
IMG_SIZE = 224

# ==========================================
# DATASET (Modified to return raw image for plotting)
# ==========================================
class GrevyVisualDataset(Dataset):
    def __init__(self, json_path, cache_dir, transforms=None):
        with open(json_path, 'r') as f:
            data = json.load(f)
        images_df = pd.DataFrame(data['images'])
        annots_df = pd.DataFrame(data['annotations'])
        self.df = pd.merge(annots_df, images_df, left_on='image_uuid', right_on='uuid', suffixes=('', '_img')).reset_index(drop=True)
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
            img = img[:, :, ::-1] # BGR to RGB
            
        # Keep a copy of the raw image for clean plotting
        raw_img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            
        if self.transforms:
            tensor_img = self.transforms(image=img)["image"]
        else:
            tensor_img = img
            
        targets = {
            'age': AGE_MAP[row['age']],
            'sex': SEX_MAP[row['sex']],
            'uuid': row['uuid']
        }
        return tensor_img, raw_img, targets

# ==========================================
# MODEL ARCHITECTURE
# ==========================================
class AttentionMILPool(nn.Module):
    def __init__(self, in_dim, embed_dim=256):
        super().__init__()
        self.proj = nn.Linear(in_dim, embed_dim)
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        h = self.proj(x)  
        attn_scores = self.attention(h)  
        attn_weights = torch.softmax(attn_scores, dim=1)  
        pooled_features = torch.sum(h * attn_weights, dim=1)  
        return pooled_features, attn_weights

class DINOProbeMIL(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        in_dim = self.backbone.embed_dim
        bottleneck_dim = 256
        
        self.pooler = AttentionMILPool(in_dim=in_dim, embed_dim=bottleneck_dim)
        self.fc = nn.Sequential(
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2), 
            nn.Linear(bottleneck_dim, num_classes)
        )

    def forward(self, x, return_attention=False):
        with torch.no_grad():
            features = self.backbone.forward_features(x)
            patch_tokens = features['x_norm_patchtokens'] 
            
        pooled_features, attn_weights = self.pooler(patch_tokens)
        logits = self.fc(pooled_features)
        
        if return_attention:
            return logits, attn_weights
        return logits

# ==========================================
# VISUALIZATION ENGINE
# ==========================================
def overlay_attention(raw_img, attn_weights):
    """
    Dynamically reshapes the 1D attention sequence into a 2D grid, 
    resizes it to match the image, and applies a colormap.
    """
    # 1. Strip batch and feature dims -> Shape: (Num_Patches,)
    attn_1d = attn_weights.squeeze() 
    
    # 2. Dynamically calculate the spatial grid size (e.g., sqrt(196) = 14)
    grid_size = int(np.sqrt(len(attn_1d)))
    attn_2d = attn_1d.reshape(grid_size, grid_size)
    
    # 3. Min-Max normalization to make the heatmap visually pop
    attn_2d = (attn_2d - attn_2d.min()) / (attn_2d.max() - attn_2d.min())
    attn_2d = np.uint8(255 * attn_2d)
    
    # 4. Resize to match the original image size
    attn_resized = cv2.resize(attn_2d, (raw_img.shape[1], raw_img.shape[0]), interpolation=cv2.INTER_CUBIC)
    
    # 5. Apply colormap and blend
    heatmap = cv2.applyColorMap(attn_resized, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Blend image and heatmap (0.6 image, 0.4 heatmap)
    overlay = cv2.addWeighted(raw_img, 0.6, heatmap, 0.4, 0)
    return overlay

def generate_visuals(model, dataloader, device, task, out_dir, num_samples=10):
    model.eval()
    os.makedirs(out_dir, exist_ok=True)
    
    inv_map = REV_AGE if task == 'age' else REV_SEX
    samples_processed = 0

    print(f"\nGenerating {task.upper()} attention visuals...")
    
    with torch.no_grad():
        for tensor_imgs, raw_imgs, targets in dataloader:
            tensor_imgs = tensor_imgs.to(device).float()
            
            # Forward pass requesting attention weights
            logits, attn_weights = model(tensor_imgs, return_attention=True)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            attn_weights = attn_weights.cpu().numpy()
            raw_imgs = raw_imgs.numpy()
            true_labels = targets[task].numpy()
            uuids = targets['uuid']

            for i in range(len(tensor_imgs)):
                if samples_processed >= num_samples:
                    return

                raw_img = raw_imgs[i]
                attn = attn_weights[i]
                true_val = inv_map[true_labels[i]]
                pred_val = inv_map[preds[i]]
                img_uuid = uuids[i]

                # Generate the overlay
                overlay = overlay_attention(raw_img, attn)

                # Plot Side-by-Side
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                fig.suptitle(f"Task: {task.upper()} | True: {true_val} | Pred: {pred_val}", fontsize=14, weight='bold')
                
                axes[0].imshow(raw_img)
                axes[0].set_title("Original Image")
                axes[0].axis('off')
                
                axes[1].imshow(overlay)
                axes[1].set_title("Attention Map")
                axes[1].axis('off')
                
                # Save as high-DPI publication-ready format
                plt.tight_layout()
                save_path = os.path.join(out_dir, f"Attn_{task.upper()}_{img_uuid}.png")
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                samples_processed += 1
                
    print(f"Saved {samples_processed} attention maps to {out_dir}/")

# ==========================================
# MAIN
# ==========================================
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    cache_dir = os.environ.get('TMPDIR', '/tmp')

    # Config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    dino_model_name = config.get('dino_model', 'dinov3_vitb16')
    dino_repo = config.get('dino_repo', './dino')
    dino_weights = config.get('dino_weights', {}).get(dino_model_name, None)

    print(f"Loading {dino_model_name} backbone...")
    dinov3 = torch.hub.load(dino_repo, dino_model_name, source='local', weights=dino_weights).to(device)
    dinov3.eval()

    # Load MIL Models
    age_model = DINOProbeMIL(dinov3, num_classes=6).to(device)
    sex_model = DINOProbeMIL(dinov3, num_classes=2).to(device)

    # Load Checkpoints
    print("Loading checkpoints...")
    age_checkpoint = torch.load(args.age_checkpoint, map_location=device)
    age_model.load_state_dict(age_checkpoint['model_state_dict'])
    
    sex_checkpoint = torch.load(args.sex_checkpoint, map_location=device)
    sex_model.load_state_dict(sex_checkpoint['model_state_dict'])

    # Transforms (Evaluation only)
    eval_transforms = Compose([
        Resize(IMG_SIZE, IMG_SIZE),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    dataset = GrevyVisualDataset(args.test_json, cache_dir, eval_transforms)
    
    # Shuffle set to True to get a random variety of samples for the paper
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

    # Generate Visuals
    generate_visuals(age_model, dataloader, device, task='age', out_dir=args.out_dir, num_samples=args.num_samples)
    generate_visuals(sex_model, dataloader, device, task='sex', out_dir=args.out_dir, num_samples=args.num_samples)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate DINO Attention Map Visuals')
    parser.add_argument('--config', type=str, default='agesex_dino.yaml', help='Path to YAML config')
    parser.add_argument('--test_json', type=str, required=True, help='Path to test JSON split')
    parser.add_argument('--age_checkpoint', type=str, required=True, help='Path to best_age_model.pth')
    parser.add_argument('--sex_checkpoint', type=str, required=True, help='Path to best_sex_model.pth')
    parser.add_argument('--out_dir', type=str, default='./attention_visuals', help='Directory to save plots')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of image pairs to generate per task')
    
    args = parser.parse_args()
    main(args)