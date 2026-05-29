import os
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, recall_score

def plot_1x4_cm(output_dir, task, models, class_names):
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    sns.set_theme(style="white")
    
    # MAKE CMS
    cms = []
    v_min = 1
    v_max = 0

    for model in models:
        data_path = os.path.join(output_dir, f"preds_{task}_{model}.npz")
        if not os.path.exists(data_path):
            print(f"Missing {data_path}, skipping {model} CM.")
            continue
            
        data = np.load(data_path)
        cm = confusion_matrix(data['y_true'], data['y_pred'])
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cms.append(cm_norm)

        v_min = min(np.min(cm_norm), v_min)
        v_max = max(np.max(cm_norm), v_max)


    # MAKE GRAPHS OFF CMS
    for i, cm_norm in enumerate(cms):
        # Get model again
        model = models[i]
        # Grounding
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", vmin=v_min, vmax=v_max,
                    xticklabels=class_names, yticklabels=class_names, ax=axes[i], cbar=False)
        axes[i].set_title(model.upper(), fontweight='bold', fontsize=14)
        if i == 0:
            axes[i].set_ylabel("True Class", fontsize=12)
        axes[i].set_xlabel("Predicted Class", fontsize=12)
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"combined_cm_{task}.pdf"), dpi=300)
    plt.close()
    print(f"Saved 1x4 Confusion Matrix for {task}.")

def plot_distribution_vs_recall(output_dir, task, class_names, target_model='mil'):
    data_path = os.path.join(output_dir, f"preds_{task}_{target_model}.npz")
    if not os.path.exists(data_path):
        return

    data = np.load(data_path)
    y_true, y_pred = data['y_true'], data['y_pred']
    
    # Calculate counts and recall
    unique, counts = np.unique(y_true, return_counts=True)
    recalls = recall_score(y_true, y_pred, average=None, zero_division=0)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    sns.set_theme(style="whitegrid")

    # Bar chart for data distribution
    bars = ax1.bar(class_names, counts, color='lightgray', edgecolor='gray', label='Test Samples')
    ax1.set_ylabel('Number of Annotations (Support)', color='gray', fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='gray')
    ax1.set_xlabel(f'{task.capitalize()} Class', fontweight='bold')

    # Line chart for Recall
    ax2 = ax1.twinx()
    line = ax2.plot(class_names, recalls, color='#e63946', marker='o', linewidth=3, markersize=8, label=f'{target_model.upper()} Recall')
    ax2.set_ylabel('Class Recall', color='#e63946', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#e63946')
    ax2.set_ylim(0, 1.05)

    plt.title(f"{task.upper()}: Class Distribution vs. AB-MIL Recall", fontweight='bold', fontsize=14)
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, f"dist_vs_recall_{task}.pdf"), dpi=300)
    plt.close()
    print(f"Saved Distribution vs Recall for {task}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    
    models = ['base', 'global', 'mil', 'encoder']
    age_classes = ['0-2', '3-5', '6-11', '12-23', '24-35', '36+']
    sex_classes = ['Female', 'Male']
    
    plot_1x4_cm(args.output_dir, 'age', models, age_classes)
    plot_1x4_cm(args.output_dir, 'sex', models, sex_classes)
    
    plot_distribution_vs_recall(args.output_dir, 'age', age_classes)
    plot_distribution_vs_recall(args.output_dir, 'sex', sex_classes)