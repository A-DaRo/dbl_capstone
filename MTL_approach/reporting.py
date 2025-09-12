# reporting.py
# Dependencies: matplotlib, seaborn, numpy, torch

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from typing import Dict, List, Tuple

# Helper to denormalize image for visualization
def denormalize(tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = tensor.clone() * std + mean
    return tensor.permute(1, 2, 0).numpy().clip(0, 1)

def plot_primary_performance(val_history: Dict[str, List[float]], save_path: str = None):
    """Plots mIoU for primary tasks and H-Mean vs. epoch, highlighting the best epoch."""
    epochs = range(1, len(val_history.get('H-Mean', [])) + 1)
    h_mean = val_history.get('H-Mean', [])
    miou_genus = val_history.get('mIoU_genus', [])
    miou_health = val_history.get('mIoU_health', [])

    best_epoch = np.argmax(h_mean) + 1
    best_score = np.max(h_mean)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(epochs, h_mean, label=f'H-Mean (Best: {best_score:.4f})', color='b', linewidth=2.5, marker='o')
    ax.plot(epochs, miou_genus, label='mIoU Genus', color='g', linestyle='--')
    ax.plot(epochs, miou_health, label='mIoU Health', color='orange', linestyle='--')

    ax.axvline(x=best_epoch, color='r', linestyle=':', label=f'Best Epoch: {best_epoch}')

    ax.set_title('Primary Task Performance on Validation Set', fontsize=16)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('mIoU / H-Mean', fontsize=12)
    ax.legend(fontsize=10)
    ax.set_ylim(bottom=0)

    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_per_class_iou_bar_chart(class_iou: Dict[str, float], task_name: str, save_path: str = None):
    """Generates a bar chart of IoU for each class of a given task."""
    class_names = list(class_iou.keys())
    iou_scores = [class_iou[name] for name in class_names]

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8))
    bars = ax.bar(class_names, iou_scores, color=plt.cm.viridis(np.linspace(0, 1, len(class_names))))

    ax.set_title(f'Per-Class IoU for {task_name.title()} Task', fontsize=16)
    ax.set_xlabel('Class Name', fontsize=12)
    ax.set_ylabel('Intersection over Union (IoU)', fontsize=12)
    ax.set_ylim(0, 1.0)
    plt.xticks(rotation=45, ha='right')

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{yval:.2f}', ha='center', va='bottom')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], task_name: str, save_path: str = None):
    """Plots a normalized confusion matrix heatmap."""
    # --- FIX 1: Handle division by zero for classes not present in the validation set ---
    # We use a context manager to suppress the warning and np.nan_to_num to replace NaNs with 0.
    with np.errstate(divide='ignore', invalid='ignore'):
        row_sums = cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm.astype('float') / row_sums)

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)

    ax.set_title(f'Normalized Confusion Matrix for {task_name.title()} Task', fontsize=16)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_qualitative_grid(images: torch.Tensor, gt_masks: Dict[str, torch.Tensor], pred_masks: Dict[str, torch.Tensor], save_path: str = None):
    """
    --- FIX 2: A flexible grid that adapts to the provided tasks ---
    Generates a grid comparing input images, ground truths, and predictions.
    It now dynamically determines the number of rows based on the keys in the mask dictionaries.
    """
    num_samples = images.shape[0]
    tasks = list(gt_masks.keys())  # Dynamically get task names from the dictionary keys
    num_tasks = len(tasks)

    # 1 row for the input image, plus 2 rows for each task (GT and Prediction)
    num_rows = 1 + (num_tasks * 2)

    fig, axes = plt.subplots(num_rows, num_samples, figsize=(4 * num_samples, 4 * num_rows), squeeze=False)

    # --- Row 0: Input Images ---
    for i in range(num_samples):
        axes[0, i].imshow(denormalize(images[i]))
        axes[0, i].set_title(f'Sample {i+1}', fontsize=14)
    axes[0, 0].set_ylabel('Input Image', fontsize=14) # Label only the first column

    # --- Subsequent Rows: GT and Predictions for each task ---
    for task_idx, task_name in enumerate(tasks):
        gt_row = 1 + (task_idx * 2)
        pred_row = 2 + (task_idx * 2)
        
        # Use different colormaps for visual distinction between tasks
        cmap = 'viridis' if task_idx % 2 == 0 else 'inferno'

        for i in range(num_samples):
            # Plot Ground Truth
            axes[gt_row, i].imshow(gt_masks[task_name][i], cmap=cmap)
            # Plot Prediction
            axes[pred_row, i].imshow(pred_masks[task_name][i], cmap=cmap)

        # Set row labels
        axes[gt_row, 0].set_ylabel(f'GT {task_name}', fontsize=14)
        axes[pred_row, 0].set_ylabel(f'Pred {task_name}', fontsize=14)

    # General formatting for all axes
    for r in range(num_rows):
        for c in range(num_samples):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def print_results_table(results: Dict[str, Dict[str, float]]):
    """Prints a formatted final results table to the console."""
    header = f"{'Model':<20} | {'mIoU Genus':<12} | {'BIoU Genus':<12} | {'mIoU Health':<12} | {'H-Mean':<10}"
    print(header)
    print('-' * len(header))
    
    for model_name, metrics in results.items():
        miou_g = f"{metrics.get('mIoU_genus', 0) * 100:.2f}%"
        biou_g = f"{metrics.get('BIoU_genus', 0) * 100:.2f}%"
        miou_h = f"{metrics.get('mIoU_health', 0) * 100:.2f}%"
        h_mean = f"{metrics.get('H-Mean', 0) * 100:.2f}%"
        print(f"{model_name:<20} | {miou_g:<12} | {biou_g:<12} | {miou_h:<12} | {h_mean:<10}")


if __name__ == '__main__':
    print("--- Running Demo of Reporting Plots & Tables ---")

    # --- 1. Dummy Validation History (10 epochs) ---
    mock_val_history = {
        'H-Mean': [0.45, 0.55, 0.62, 0.68, 0.71, 0.73, 0.74, 0.735, 0.72, 0.715],
        'mIoU_genus': [0.40, 0.50, 0.58, 0.65, 0.69, 0.70, 0.72, 0.71, 0.70, 0.69],
        'mIoU_health': [0.50, 0.60, 0.66, 0.71, 0.73, 0.76, 0.76, 0.76, 0.74, 0.74]
    }
    print("\n1. Plotting Primary Performance vs. Epoch...")
    plot_primary_performance(mock_val_history)

    # --- 2. Dummy Per-Class IoU for Genus ---
    genus_names = ["background", "other", "massive", "branching", "acropora", "table", "pocillopora", "meandering", "stylophora"]
    mock_genus_iou = {name: np.random.uniform(0.3, 0.95) for name in genus_names}
    mock_genus_iou['pocillopora'] = 0.25 # Simulate a rare, difficult class
    print("\n2. Plotting Per-Class IoU Bar Chart...")
    plot_per_class_iou_bar_chart(mock_genus_iou, "Genus")

    # --- 3. Dummy Confusion Matrix ---
    num_genus_classes = len(genus_names)
    mock_cm = np.random.randint(0, 50, size=(num_genus_classes, num_genus_classes))
    np.fill_diagonal(mock_cm, np.random.randint(200, 500, size=num_genus_classes))
    mock_cm[4, 8] = 150 # Simulate confusing acropora with stylophora
    print("\n3. Plotting Confusion Matrix...")
    plot_confusion_matrix(mock_cm, genus_names, "Genus")

    # --- 4. Dummy Qualitative Data ---
    B, C, H, W = 2, 3, 256, 256
    mock_images = torch.rand(B, C, H, W)
    mock_gt_masks = {
        'genus': torch.randint(0, num_genus_classes, (B, H, W)),
        'health': torch.randint(0, 4, (B, H, W)),
    }
    mock_pred_masks = { # Predictions are similar to GT but with some errors
        'genus': mock_gt_masks['genus'].clone(),
        'health': mock_gt_masks['health'].clone(),
    }
    mock_pred_masks['genus'][:, 100:150, 100:150] = 5 # Add a block of error
    print("\n4. Plotting Qualitative Grid...")
    plot_qualitative_grid(mock_images, mock_gt_masks, mock_pred_masks)

    # --- 5. Dummy Final Results Table ---
    mock_final_results = {
        "Single-Task (Genus)": {"mIoU_genus": 0.68, "BIoU_genus": 0.55, "mIoU_health": 0, "H-Mean": 0},
        "Single-Task (Health)": {"mIoU_genus": 0, "BIoU_genus": 0, "mIoU_health": 0.72, "H-Mean": 0},
        "Coral-MTL (Ours)": {"mIoU_genus": 0.74, "BIoU_genus": 0.65, "mIoU_health": 0.76, "H-Mean": 0.75}
    }
    print("\n5. Printing Final Results Table...")
    print_results_table(mock_final_results)