import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch


def _save_plot_data(data: Dict, save_path: str):
    """Helper function to save plot data to a JSON file if a path is provided."""
    if save_path:
        json_path = Path(save_path).with_suffix('.json')
        json_path.parent.mkdir(parents=True, exist_ok=True)
        # Convert numpy arrays/tensors to lists for JSON serialization
        serializable_data = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                serializable_data[key] = value.tolist()
            elif isinstance(value, torch.Tensor):
                serializable_data[key] = value.tolist()
            elif isinstance(value, list) and value and isinstance(value[0], torch.Tensor):
                serializable_data[key] = [v.item() for v in value]
            else:
                serializable_data[key] = value
        
        with open(json_path, 'w') as f:
            json.dump(serializable_data, f, indent=4)

# --- Plotting Functions for In-Training Monitoring ---

def plot_composite_losses(log_history: Dict[str, List[float]], save_path: str = None):
    """Plots total, primary, and auxiliary losses vs. training steps."""
    _save_plot_data(log_history, save_path)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))
    
    steps = range(len(log_history.get('total_loss', [])))
    if not steps: return
        
    ax.plot(steps, log_history.get('total_loss', []), label='Total Loss', color='b', linewidth=2)
    ax.plot(steps, log_history.get('primary_loss', []), label='Primary Loss (Spec 6.2)', color='g', linestyle='--')
    ax.plot(steps, log_history.get('auxiliary_loss', []), label='Auxiliary Loss (Spec 6.3)', color='r', linestyle=':')
    
    ax.set_title('Composite Loss Trends During Training', fontsize=16)
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Loss Value', fontsize=12)
    ax.legend(fontsize=10)
    ax.set_yscale('log')
    
    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close(fig)

def plot_individual_task_losses(log_history: Dict[str, List[float]], save_path: str = None):
    """Creates a grid of plots for each individual task loss vs. training steps."""
    task_losses = {k: v for k, v in log_history.items() if '_loss' in k and k not in ['total_loss', 'primary_loss', 'auxiliary_loss']}
    if not task_losses: return
    
    _save_plot_data(task_losses, save_path)
        
    num_tasks = len(task_losses)
    fig, axes = plt.subplots(num_tasks, 1, figsize=(10, 4 * num_tasks), sharex=True)
    if num_tasks == 1: axes = [axes]
        
    steps = range(len(list(task_losses.values())[0]))
    for ax, (name, values) in zip(axes, task_losses.items()):
        ax.plot(steps, values, label=name)
        ax.set_title(f'Loss Trend for: {name}')
        ax.set_ylabel('Loss')
        ax.legend()

    axes[-1].set_xlabel('Training Steps')
    plt.suptitle('Individual Task Loss Diagnostics', fontsize=18, y=0.99)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close(fig)

def plot_uncertainty_weights(log_history: Dict[str, List[float]], save_path: str = None):
    """Plots the learned uncertainty weights (sigma^2) vs. training steps."""
    log_vars = {key: val for key, val in log_history.items() if 'log_var' in key}
    if not log_vars: return

    _save_plot_data(log_vars, save_path)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for key, log_var_values in log_vars.items():
        # Ensure values are tensors for torch.exp
        if not isinstance(log_var_values, torch.Tensor):
            log_var_values = torch.tensor(log_var_values, dtype=torch.float32)
        
        sigma_sq = torch.exp(log_var_values).numpy()
        steps = range(len(sigma_sq))
        ax.plot(steps, sigma_sq, label=f'σ² {key.replace("log_var_", "")}')
    
    ax.set_title('Learned Task Uncertainty (σ²) - Spec Section 6.2', fontsize=16)
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('σ² Value', fontsize=12)
    ax.legend(fontsize=10)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close(fig)

def plot_learning_rate(log_history: Dict[str, List[float]], warmup_steps: int, save_path: str = None):
    """Plots the learning rate schedule vs. training steps."""
    lr_values = log_history.get('lr', [])
    if len(lr_values) == 0: return
    
    _save_plot_data({'lr': lr_values, 'warmup_steps': warmup_steps}, save_path)
    
    steps = range(len(lr_values))
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(steps, lr_values, label='Learning Rate', color='teal')
    if warmup_steps > 0:
        ax.axvline(x=warmup_steps, color='r', linestyle='--', label='End of Warmup')
    
    ax.set_title('Learning Rate Schedule (Poly w/ Warmup) - Spec Section 6.3', fontsize=16)
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.legend(fontsize=10)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close(fig)

# --- Plotting Functions for Post-Hoc Analysis and Reporting ---

def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Helper to denormalize an image tensor for visualization."""
    # Denormalization based on ImageNet stats from Spec Section 5.2
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = tensor.cpu().clone() * std + mean
    return tensor.permute(1, 2, 0).numpy().clip(0, 1)

def plot_primary_performance(val_history: Dict[str, List[float]], save_path: str = None):
    """Plots mIoU for primary tasks and H-Mean vs. epoch."""
    h_mean = val_history.get('H-Mean', [])
    if len(h_mean) == 0: return
    
    _save_plot_data(val_history, save_path)
    
    epochs = range(1, len(h_mean) + 1)
    best_epoch = np.argmax(h_mean) + 1
    best_score = np.max(h_mean)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(epochs, h_mean, label=f'H-Mean (Best: {best_score:.4f})', color='b', linewidth=2.5, marker='o')
    if 'mIoU_genus' in val_history:
        ax.plot(epochs, val_history['mIoU_genus'], label='mIoU Genus', color='g', linestyle='--')
    if 'mIoU_health' in val_history:
        ax.plot(epochs, val_history['mIoU_health'], label='mIoU Health', color='orange', linestyle='--')
    ax.axvline(x=best_epoch, color='r', linestyle=':', label=f'Best Epoch: {best_epoch}')
    ax.set_title('Primary Task Performance on Validation Set', fontsize=16)
    ax.set_xlabel('Epoch', fontsize=12); ax.set_ylabel('mIoU / H-Mean (Spec 7.3.1)', fontsize=12)
    ax.legend(fontsize=10); ax.set_ylim(bottom=0)
    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close(fig)

def plot_per_class_iou_bar_chart(class_iou: Dict[str, float], task_name: str, save_path: str = None):
    """Generates a bar chart of IoU for each class of a given task."""
    _save_plot_data(class_iou, save_path)

    class_names, iou_scores = list(class_iou.keys()), list(class_iou.values())
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8))
    bars = ax.bar(class_names, iou_scores, color=plt.cm.viridis(np.linspace(0, 1, len(class_names))))
    ax.set_title(f'Per-Class IoU for {task_name.title()} Task', fontsize=16)
    ax.set_xlabel('Class Name', fontsize=12); ax.set_ylabel('IoU', fontsize=12)
    ax.set_ylim(0, 1.0); plt.xticks(rotation=45, ha='right')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{yval:.2f}', ha='center', va='bottom')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close(fig)

def _plot_confusion_heatmap(cm: np.ndarray, class_names: List[str], task_name: str, save_path: str):
    """Internal function to plot a standard normalized confusion matrix heatmap."""
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_normalized = np.nan_to_num(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis])
    
    fig, ax = plt.subplots(figsize=(max(8, len(class_names)*0.8), max(6, len(class_names)*0.7)))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title(f'Normalized Confusion Matrix for {task_name.title()} Task', fontsize=16)
    ax.set_xlabel('Predicted Label', fontsize=12); ax.set_ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close(fig)

def _plot_top_k_misclassifications(cm: np.ndarray, class_names: List[str], task_name: str, k: int, save_path: str):
    """Internal function to plot top-k misclassifications for each class as a stacked bar chart."""
    # This visualization focuses on the most significant errors, which is more
    # useful for tasks with many classes like Genus Segmentation (Spec Section 1.1).
    np.fill_diagonal(cm, 0)
    
    y_labels, all_misclass_counts, all_misclass_names = [], [], []
    
    for i, class_name in enumerate(class_names):
        row = cm[i, :]
        top_k_indices = np.argsort(row)[::-1][:k]
        top_k_counts = row[top_k_indices]
        
        if np.sum(top_k_counts) > 0:
            y_labels.append(class_name)
            all_misclass_counts.append(top_k_counts)
            all_misclass_names.append([class_names[j] for j in top_k_indices])

    if not y_labels:
        print(f"No misclassifications to plot for task {task_name}.")
        return

    y_pos = np.arange(len(y_labels))
    fig, ax = plt.subplots(figsize=(12, 1 + len(y_labels) * 0.5))
    
    unique_confused = sorted(list(set(n for names in all_misclass_names for n in names)))
    colors = plt.cm.get_cmap('tab20', len(unique_confused))
    color_map = {name: colors(i) for i, name in enumerate(unique_confused)}

    for i, (counts, names) in enumerate(zip(all_misclass_counts, all_misclass_names)):
        left = 0
        for count, name in zip(counts, names):
            if count > 0:
                ax.barh(y_pos[i], count, left=left, color=color_map.get(name), edgecolor='white', label=name)
                left += count
    
    ax.set_yticks(y_pos); ax.set_yticklabels(y_labels); ax.invert_yaxis()
    ax.set_xlabel('Misclassification Count (e.g., Pixel Count)'); ax.set_title(f'Top {k} Misclassifications for {task_name.title()}')
    
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), title='Confused with', bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close(fig)

def plot_confusion_analysis(cm: np.ndarray, class_names: List[str], task_name: str, save_path: str = None, top_k: int = 3, threshold: int = 10):
    """
    Analyzes and plots a confusion matrix, choosing the best plot type based on class count.
    - Saves the raw confusion matrix data to a JSON file.
    - For matrices with classes <= threshold, plots a standard normalized heatmap.
    - For matrices > threshold, plots a bar chart of top-k misclassifications for each class.
    """
    if save_path:
        json_path = Path(save_path).with_suffix('.json')
        json_path.parent.mkdir(parents=True, exist_ok=True)
        cm_data = {'task_name': task_name, 'class_names': class_names, 'confusion_matrix': cm.tolist()}
        with open(json_path, 'w') as f:
            json.dump(cm_data, f, indent=4)

    if len(class_names) <= threshold:
        _plot_confusion_heatmap(cm, class_names, task_name, save_path)
    else:
        _plot_top_k_misclassifications(cm, class_names, task_name, top_k, save_path)

def plot_qualitative_grid(images: torch.Tensor, gt_masks: Dict[str, torch.Tensor], pred_masks: Dict[str, torch.Tensor], save_path: str = None):
    """Generates a flexible grid comparing input images, ground truths, and predictions."""
    num_samples = images.shape[0]
    gt_task_names, pred_task_names = list(gt_masks.keys()), list(pred_masks.keys())
    assert len(gt_task_names) == len(pred_task_names), "Mismatch in number of GT vs Pred tasks."
    num_tasks = len(gt_task_names)
    num_rows = 1 + (num_tasks * 2)
    fig, axes = plt.subplots(num_rows, num_samples, figsize=(4 * num_samples, 4 * num_rows), squeeze=False)

    for i in range(num_samples):
        axes[0, i].imshow(denormalize(images[i]))
        axes[0, i].set_title(f'Sample {i+1}', fontsize=14)
    axes[0, 0].set_ylabel('Input Image', fontsize=14)

    for task_idx in range(num_tasks):
        gt_key, pred_key = gt_task_names[task_idx], pred_task_names[task_idx]
        gt_row, pred_row = 1 + (task_idx * 2), 2 + (task_idx * 2)
        cmap = 'viridis' if task_idx % 2 == 0 else 'inferno'
        for i in range(num_samples):
            axes[gt_row, i].imshow(gt_masks[gt_key][i].cpu(), cmap=cmap)
            axes[pred_row, i].imshow(pred_masks[pred_key][i].cpu(), cmap=cmap)
        axes[gt_row, 0].set_ylabel(f'GT {gt_key}', fontsize=14)
        axes[pred_row, 0].set_ylabel(f'Pred {pred_key}', fontsize=14)

    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close(fig)

def print_results_table(results: Dict[str, Dict[str, float]]):
    """Prints a formatted final results table to the console."""
    header = f"{'Model':<20} | {'mIoU Genus':<12} | {'BIoU Genus':<12} | {'mIoU Health':<12} | {'H-Mean':<10}"
    print(header); print('-' * len(header))
    for model_name, metrics in results.items():
        miou_g = f"{metrics.get('mIoU_genus', 0) * 100:.2f}%"
        biou_g = f"{metrics.get('BIoU_genus', 0) * 100:.2f}%"
        miou_h = f"{metrics.get('mIoU_health', 0) * 100:.2f}%"
        h_mean = f"{metrics.get('H-Mean', 0) * 100:.2f}%"
        print(f"{model_name:<20} | {miou_g:<12} | {biou_g:<12} | {miou_h:<12} | {h_mean:<10}")