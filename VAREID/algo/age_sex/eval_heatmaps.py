import os
import argparse
import yaml
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2

from agesex_baseline import load_coco_to_df, AGE_MAP, SEX_MAP

class TransformerAttentionExtractor:
    def __init__(self, model):
        self.attn_weights = None
        self.mha_module = model.pooler.transformer.layers[0].self_attn
        self.mha_module.register_forward_pre_hook(self._pre_hook, with_kwargs=True)
        self.mha_module.register_forward_hook(self._forward_hook)

    def _pre_hook(self, module, args, kwargs):
        kwargs['need_weights'] = True
        kwargs['average_attn_weights'] = True 
        return args, kwargs

    def _forward_hook(self, module, inputs, outputs):
        self.attn_weights = outputs[1].detach().cpu()

def overlay_attention(raw_img, attn_1d):
    """Overlays the attention map on the original, unwarped image dimensions."""
    grid_size = int(np.sqrt(len(attn_1d)))
    attn_2d = attn_1d.reshape(grid_size, grid_size)
    
    # Normalize safely
    attn_2d = (attn_2d - attn_2d.min()) / (attn_2d.max() - attn_2d.min() + 1e-8)
    attn_2d = np.uint8(255 * attn_2d)
    
    # Scale back up to original image size (NO WARPING)
    attn_resized = cv2.resize(attn_2d, (raw_img.shape[1], raw_img.shape[0]), interpolation=cv2.INTER_CUBIC)
    
    heatmap = cv2.applyColorMap(attn_resized, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Adjusted weights: 0.4 img, 0.6 heatmap makes the colors darker and pop more
    overlay = cv2.addWeighted(raw_img, 0.4, heatmap, 0.6, 0)
    return overlay

def main(args):
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    run_name = config.get('run_name', 'default')
    exp_out_dir = os.path.join(args.output_dir, run_name, f'heatmaps_{args.task}')
    os.makedirs(exp_out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 6 if args.task == 'age' else 2

    # 1. Load Backbone Once
    print("Loading DINOv3 backbone...")
    dino_model = config.get('dino_model', 'dinov3_vitb16')
    dino_weights = config.get('dino_weights', {}).get(dino_model, None)
    dinov3 = torch.hub.load(config.get('dino_repo'), dino_model, source='local', weights=dino_weights).to(device)

    # 2. Load MIL Probe
    from agesex_dino_mil import DINOProbe as MILProbe
    mil_model = MILProbe(dinov3, num_classes=num_classes).to(device)
    mil_ckpt = torch.load(os.path.join(args.checkpoint_dir, f"best_{args.task}_mil_{run_name}.pth"), map_location=device, weights_only=True)
    mil_model.load_state_dict(mil_ckpt['model_state_dict'], strict=False)
    mil_model.eval()

    # 3. Load Encoder Probe
    from agesex_dino_encoder import DINOProbe as EncoderProbe
    encoder_model = EncoderProbe(dinov3, num_classes=num_classes).to(device)
    encoder_ckpt = torch.load(os.path.join(args.checkpoint_dir, f"best_{args.task}_encoder_{run_name}.pth"), map_location=device, weights_only=True)
    encoder_model.load_state_dict(encoder_ckpt['model_state_dict'], strict=False)
    encoder_model.eval()

    extractor = TransformerAttentionExtractor(encoder_model)

    # 4. Load Data & Sample
    print(f"Sampling {args.n_samples} images per class from {args.eval_json}...")
    df = load_coco_to_df(args.eval_json)
    sampled_df = df.groupby(args.task, group_keys=False).apply(lambda x: x.sample(n=min(args.n_samples, len(x)), random_state=42)).reset_index(drop=True)

    transform = Compose([Resize(224, 224), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()])

    for _, row in sampled_df.iterrows():
        img_uuid = row['uuid']
        class_label = row[args.task]
        
        img_path = os.path.join(args.cache_dir, f"{img_uuid}.jpg")
        raw_img = cv2.imread(img_path)
        if raw_img is None:
            continue
            
        raw_img = raw_img[:, :, ::-1] # BGR to RGB (Original Aspect Ratio)
        
        # Warp purely for the network input
        tensor_img = transform(image=raw_img)["image"].unsqueeze(0).to(device)

        with torch.no_grad():
            features = dinov3.forward_features(tensor_img)
            patch_tokens = features['x_norm_patchtokens']
            
            # MIL Attention
            _, mil_attn = mil_model.pooler(patch_tokens)
            mil_attn = mil_attn.squeeze().cpu().numpy()
            
            # Encoder Attention (Hook capture)
            _ = encoder_model(tensor_img)
            encoder_attn = extractor.attn_weights.mean(dim=1).squeeze().numpy()

        # Build Unwarped Overlays
        mil_overlay = overlay_attention(raw_img, mil_attn)
        encoder_overlay = overlay_attention(raw_img, encoder_attn)

        # Plotting
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(raw_img); axes[0].set_title(f"Original ({class_label})", fontweight='bold'); axes[0].axis('off')
        axes[1].imshow(mil_overlay); axes[1].set_title("AB-MIL Spatial Filter", fontweight='bold'); axes[1].axis('off')
        axes[2].imshow(encoder_overlay); axes[2].set_title("Transformer Encoder", fontweight='bold'); axes[2].axis('off')
        
        plt.tight_layout()
        class_dir = os.path.join(exp_out_dir, str(class_label).replace('+', '_plus'))
        os.makedirs(class_dir, exist_ok=True)
        plt.savefig(os.path.join(class_dir, f"{img_uuid}.pdf"), dpi=300, bbox_inches='tight')
        plt.close()

    print(f"Saved unwarped heatmaps to {exp_out_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='agesex_dino.yaml')
    parser.add_argument('--eval_json', type=str, required=True, help="Path to json (test set)")
    parser.add_argument('--cache_dir', type=str, required=True)
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./eval_results')
    parser.add_argument('--task', type=str, choices=['age', 'sex'], required=True)
    parser.add_argument('--n_samples', type=int, default=5)
    args = parser.parse_args()
    main(args)