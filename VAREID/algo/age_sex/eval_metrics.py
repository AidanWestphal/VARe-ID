import os
import argparse
import yaml
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.utils import resample
from tqdm import tqdm

from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2
from agesex_baseline import load_coco_to_df, DINOZebraDataset, AGE_MAP, SEX_MAP

def bootstrap_metrics(y_true, y_pred, n_iterations=1000, alpha=0.05):
    acc_scores, f1_scores = [], []
    n_size = len(y_true)
    for _ in range(n_iterations):
        indices = resample(range(n_size), n_samples=n_size)
        y_true_boot, y_pred_boot = y_true[indices], y_pred[indices]
        acc_scores.append(accuracy_score(y_true_boot, y_pred_boot))
        f1_scores.append(f1_score(y_true_boot, y_pred_boot, average='macro', zero_division=0))
    acc_ci = (np.percentile(acc_scores, alpha/2 * 100), np.percentile(acc_scores, (1-alpha/2) * 100))
    f1_ci = (np.percentile(f1_scores, alpha/2 * 100), np.percentile(f1_scores, (1-alpha/2) * 100))
    return np.mean(acc_scores), acc_ci, np.mean(f1_scores), f1_ci

def load_evaluation_model(model_type, task, config, checkpoint_dir, device):
    """Clean factory function to instantiate and load the requested architecture."""
    num_classes = 6 if task == 'age' else 2
    run_name = config.get('run_name', 'default')
    ckpt_path = os.path.join(checkpoint_dir, f"best_{task}_{model_type}_{run_name}.pth")
    
    print(f"Loading {model_type.upper()} architecture from: {ckpt_path}")
    
    if model_type == 'base':
        from agesex_baseline import EfficientNetProbe
        model = EfficientNetProbe(num_classes=num_classes).to(device)
    else:
        # Load backbone ONCE for DINO variants
        dino_model = config.get('dino_model', 'dinov3_vitb16')
        dino_weights = config.get('dino_weights', {}).get(dino_model, None)
        dinov3 = torch.hub.load(config.get('dino_repo'), dino_model, source='local', weights=dino_weights).to(device)
        
        if model_type == 'global':
            from agesex_dino_global import DINOGlobalProbe as Probe
        elif model_type == 'mil':
            from agesex_dino_mil import DINOProbe as Probe
        elif model_type == 'encoder':
            from agesex_dino_encoder import DINOProbe as Probe
            
        model = Probe(dinov3, num_classes=num_classes).to(device)
        
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    return model

def main(args):
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    run_name = config.get('run_name', 'default')
    exp_out_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(exp_out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Data
    val_df = load_coco_to_df(args.val_json)
    eval_transforms = Compose([
        Resize(224, 224),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    val_loader = DataLoader(
        DINOZebraDataset(val_df, args.cache_dir, task=args.task, transforms=eval_transforms),
        batch_size=32, shuffle=False, num_workers=4, pin_memory=True
    )
    
    class_names = list(AGE_MAP.keys()) if args.task == 'age' else list(SEX_MAP.keys())

    # 2. Load Model
    model = load_evaluation_model(args.model_type, args.task, config, args.checkpoint_dir, device)

    # 3. Inference
    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, targets in tqdm(val_loader, desc=f"Evaluating {args.model_type}"):
            logits = model(imgs.to(device).float())
            preds = torch.argmax(logits, dim=1)
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    # 4. Reports & Bootstrapping
    report_str = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    print(f"\n=== {args.model_type.upper()} ({args.task.upper()}) ===")
    print(report_str)
    
    print("Calculating Bootstrapped Confidence Intervals (1000 iterations)...")
    mean_acc, acc_ci, mean_f1, f1_ci = bootstrap_metrics(y_true, y_pred)
    
    # Save Report to Text
    txt_path = os.path.join(exp_out_dir, f"report_{args.task}_{args.model_type}.txt")
    with open(txt_path, 'w') as f:
        f.write(f"=== {args.model_type.upper()} ({args.task.upper()}) ===\n")
        f.write(report_str + "\n\n")
        f.write(f"Accuracy: {mean_acc:.3f} (95% CI: {acc_ci[0]:.3f} - {acc_ci[1]:.3f})\n")
        f.write(f"Macro F1: {mean_f1:.3f} (95% CI: {f1_ci[0]:.3f} - {f1_ci[1]:.3f})\n")
        
    print(f"Accuracy: {mean_acc:.3f} (95% CI: {acc_ci[0]:.3f} - {acc_ci[1]:.3f})")
    print(f"Macro F1: {mean_f1:.3f} (95% CI: {f1_ci[0]:.3f} - {f1_ci[1]:.3f})")

    # 5. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(f"{args.model_type.upper()} Confusion Matrix ({args.task.capitalize()})", fontweight='bold')
    plt.ylabel("True Class")
    plt.xlabel("Predicted Class")
    plt.tight_layout()
    cm_path = os.path.join(exp_out_dir, f"cm_{args.task}_{args.model_type}.pdf")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved evaluation artifacts to {exp_out_dir}/")

    # ADD THIS AT THE VERY BOTTOM OF eval_metrics.py
    np.savez(os.path.join(exp_out_dir, f"preds_{args.task}_{args.model_type}.npz"), y_true=y_true, y_pred=y_pred)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='agesex_dino.yaml')
    parser.add_argument('--val_json', type=str, required=True)
    parser.add_argument('--cache_dir', type=str, required=True, help="Path to extracted images")
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./eval_results')
    parser.add_argument('--model_type', type=str, choices=['base', 'global', 'mil', 'encoder'], required=True)
    parser.add_argument('--task', type=str, choices=['age', 'sex'], required=True)
    args = parser.parse_args()
    main(args)