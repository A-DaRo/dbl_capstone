import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

class Visualizer:
    """
    A unified class for generating all visualizations for the Coral-MTL project.
    It handles plotting and saving of training logs, validation metrics, confusion matrices,
    and qualitative results to a specified output directory.
    """

    def __init__(self, output_dir: str, task_info: Dict = None, style: str = 'seaborn-v0_8-whitegrid'):
        """
        Initializes the Visualizer.

        Args:
            output_dir (str): The root directory where all plots and data will be saved.
            task_info (Dict, optional): A dictionary containing task definitions,
                                        specifically the 'id2label' mappings.
            style (str): The matplotlib style to use for all plots.
        """
        self.output_dir = Path(output_dir)
        self.task_info = task_info or {}
        self.output_dir.mkdir(parents=True, exist_ok=True)
        plt.style.use(style)

    def _get_class_names(self, task_name: str) -> List[str]:
        """Safely retrieves class names for a given task from the task_info."""
        if not self.task_info:
            return []
        
        id2label = self.task_info.get('id2label', {}).get(task_name, {})
        if not id2label:
            return []
            
        # Sort by ID to ensure consistent order
        return [name for _, name in sorted(id2label.items())]

    def _save_plot_data(self, data: Dict, filename: str):
        """Helper to save plot data to a JSON file in the output directory."""
        json_path = self.output_dir / Path(filename).with_suffix('.json')
        serializable_data = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                serializable_data[key] = value.tolist()
            elif isinstance(value, torch.Tensor):
                serializable_data[key] = value.tolist()
            elif isinstance(value, (list, tuple)) and value and isinstance(value[0], torch.Tensor):
                serializable_data[key] = [v.item() for v in value]
            else:
                serializable_data[key] = value
        
        with open(json_path, 'w') as f:
            json.dump(serializable_data, f, indent=4)

    @staticmethod
    def denormalize(tensor: torch.Tensor) -> np.ndarray:
        """
        Helper to denormalize an image tensor for visualization.
        Denormalization is based on ImageNet stats from Spec Section 5.2.
        """
        mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(3, 1, 1)
        tensor = tensor.clone() * std + mean
        return tensor.permute(1, 2, 0).cpu().numpy().clip(0, 1)

    # --- High-Level Performance and Comparison Plots ---

    def plot_model_comparison(self, results: Dict[str, Dict[str, float]], filename: str = "model_comparison.png"):
        """Generates a grouped bar chart to compare key performance metrics across models."""
        self._save_plot_data(results, filename)
        
        df = pd.DataFrame(results).T.reset_index().rename(columns={'index': 'Model'})
        df_melted = df.melt(id_vars='Model', var_name='Metric', value_name='Score')

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(data=df_melted, x='Metric', y='Score', hue='Model', ax=ax, palette=['#1f77b4', '#999999'])
        
        ax.set_title('High-Level Model Performance Comparison', fontsize=18, pad=20)
        ax.set_xlabel('Key Performance Metric', fontsize=12)
        ax.set_ylabel('Score (e.g., mIoU)', fontsize=12)
        ax.set_ylim(0, max(1.0, df_melted['Score'].max() * 1.1))
        ax.legend(title='Model', fontsize=10)
        
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.3f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                        textcoords='offset points')
                        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, bbox_inches='tight')
        plt.close(fig)

    def plot_validation_performance(self, val_history: Dict[str, List[float]], filename: str = "validation_performance.png"):
        """Plots mIoU for primary tasks and H-Mean vs. epoch from validation history."""
        h_mean = val_history.get('H-Mean', [])
        if not h_mean: return
        
        self._save_plot_data(val_history, filename)
        
        epochs = range(1, len(h_mean) + 1)
        best_epoch = np.argmax(h_mean) + 1
        best_score = np.max(h_mean)

        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(epochs, h_mean, label=f'H-Mean (Best: {best_score:.4f})', color='b', linewidth=2.5, marker='o')
        if 'mIoU_genus' in val_history:
            ax.plot(epochs, val_history['mIoU_genus'], label='mIoU Genus', color='g', linestyle='--')
        if 'mIoU_health' in val_history:
            ax.plot(epochs, val_history['mIoU_health'], label='mIoU Health', color='orange', linestyle='--')
        
        ax.axvline(x=best_epoch, color='r', linestyle=':', label=f'Best Epoch: {best_epoch}')
        ax.set_title('Primary Task Performance on Validation Set', fontsize=16)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('mIoU / H-Mean (Spec 7.3.1)', fontsize=12)
        ax.legend(fontsize=10)
        ax.set_ylim(bottom=0)
        
        plt.savefig(self.output_dir / filename)
        plt.close(fig)

    # --- In-Training Monitoring Plots ---

    def plot_training_losses(self, log_history: Dict[str, List[float]], filename: str = "training_losses.png"):
        """Plots total, primary, and auxiliary losses vs. training steps."""
        self._save_plot_data(log_history, filename)
        
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
        
        plt.savefig(self.output_dir / filename)
        plt.close(fig)

    def plot_learning_rate(self, log_history: Dict[str, List[float]], warmup_steps: int = 0, filename: str = "learning_rate.png"):
        """Plots the learning rate schedule vs. training steps."""
        lr_values = log_history.get('lr', [])
        if len(lr_values) == 0: return
        
        self._save_plot_data({'lr': lr_values, 'warmup_steps': warmup_steps}, filename)
        
        steps = range(len(lr_values))
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(steps, lr_values, label='Learning Rate', color='teal')
        if warmup_steps > 0:
            ax.axvline(x=warmup_steps, color='r', linestyle='--', label='End of Warmup')
        
        ax.set_title('Learning Rate Schedule (Poly w/ Warmup) - Spec Section 6.3', fontsize=16)
        ax.set_xlabel('Training Steps', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.legend(fontsize=10)
        
        plt.savefig(self.output_dir / filename)
        plt.close(fig)

    def plot_uncertainty_weights(self, log_history: Dict[str, List[float]], filename: str = "uncertainty_weights.png"):
        """Plots the learned uncertainty weights (sigma^2) vs. training steps."""
        log_vars = {key: val for key, val in log_history.items() if 'log_var' in key}
        if not log_vars: return

        self._save_plot_data(log_vars, filename)

        fig, ax = plt.subplots(figsize=(12, 7))
        for key, log_var_values in log_vars.items():
            sigma_sq = torch.exp(torch.tensor(log_var_values, dtype=torch.float32)).numpy()
            steps = range(len(sigma_sq))
            ax.plot(steps, sigma_sq, label=f'σ² {key.replace("log_var_", "")}')
        
        ax.set_title('Learned Task Uncertainty (σ²) - Spec Section 6.2', fontsize=16)
        ax.set_xlabel('Training Steps', fontsize=12)
        ax.set_ylabel('σ² Value', fontsize=12)
        ax.legend(fontsize=10)
        
        plt.savefig(self.output_dir / filename)
        plt.close(fig)

    # --- Detailed Error Analysis and Qualitative Plots ---

    def plot_qualitative_results(self, images: torch.Tensor, gt_masks: torch.Tensor, pred_masks: torch.Tensor, task_name: str, filename: str = "qualitative_results.png", num_samples: int = 4):
        """Generates a grid comparing input images, GTs, predictions, and an error map."""
        num_samples = min(num_samples, images.shape[0])
        fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples), squeeze=False)

        for i in range(num_samples):
            axes[i, 0].imshow(self.denormalize(images[i]))
            axes[i, 1].imshow(gt_masks[i].cpu().numpy(), cmap='viridis')
            axes[i, 2].imshow(pred_masks[i].cpu().numpy(), cmap='viridis')
            
            gt_mask, pred_mask = gt_masks[i].cpu().numpy(), pred_masks[i].cpu().numpy()
            error_map = np.zeros_like(gt_mask)
            error_map[(gt_mask == 0) & (pred_mask != 0)] = 1  # False Positive (FP)
            error_map[(gt_mask != 0) & (pred_mask == 0)] = 2  # False Negative (FN)
            
            cmap_err = plt.cm.colors.ListedColormap(['#d3d3d3', '#ff7f0e', '#1f77b4']) # Gray, Orange, Blue
            axes[i, 3].imshow(error_map, cmap=cmap_err, vmin=0, vmax=2)

        col_titles = ['Input Image', 'Ground Truth', 'Prediction', 'Error Map (FP/FN)']
        for ax, title in zip(axes[0], col_titles):
            ax.set_title(title, fontsize=14)
        for i in range(num_samples):
            axes[i, 0].set_ylabel(f'Sample {i+1}', fontsize=14)
        for ax in axes.flat:
            ax.set_xticks([]); ax.set_yticks([])
            
        plt.suptitle(f'Qualitative Results for {task_name.title()} Task', fontsize=20, y=1.02)
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, bbox_inches='tight')
        plt.close(fig)

    @staticmethod
    def _calculate_error_rates(cm: np.ndarray, bg_class_idx: int = 0) -> Dict[str, float]:
        """Calculates TIDE-inspired error rates from a confusion matrix as per Spec 7.4."""
        total_pixels = cm.sum()
        if total_pixels == 0: return {'Classification': 0, 'Background': 0, 'Missed': 0}
        
        fg_indices = [i for i in range(cm.shape[0]) if i != bg_class_idx]
        
        class_error = sum(cm[r, c] for r in fg_indices for c in fg_indices if r != c)
        bg_error = cm[bg_class_idx, fg_indices].sum() # False Positives
        missed_error = cm[fg_indices, bg_class_idx].sum() # False Negatives

        total_error = class_error + bg_error + missed_error
        if total_error == 0: return {'Classification': 0, 'Background': 0, 'Missed': 0}

        return {
            'Classification Error': class_error / total_error,
            'Background Error (FP)': bg_error / total_error,
            'Missed Error (FN)': missed_error / total_error
        }

    def plot_diagnostic_error_breakdown(self, results: Dict[str, Tuple[np.ndarray, List[str]]], task_name: str, filename: str = "error_breakdown.png"):
        """Creates a stacked bar chart of the TIDE-inspired error decomposition."""
        error_data = {model: self._calculate_error_rates(cm, class_names.index('Background'))
                      for model, (cm, class_names) in results.items() if 'Background' in class_names}
        if not error_data:
            print("Warning: Could not compute error breakdown. 'Background' class not found.")
            return

        self._save_plot_data(error_data, filename)
        df = pd.DataFrame(error_data).T
        
        fig, ax = plt.subplots(figsize=(10, 7))
        df.plot(kind='barh', stacked=True, ax=ax, colormap='viridis')
        
        ax.set_title(f'Diagnostic Error Breakdown for {task_name.title()} Task', fontsize=18, pad=20)
        ax.set_xlabel('Proportion of Total Error', fontsize=12)
        ax.set_ylabel('Model', fontsize=12)
        ax.set_xlim(0, 1.0)
        ax.legend(title='Error Type', bbox_to_anchor=(1.02, 1), loc='upper left')
        
        for c in ax.containers:
            labels = [f'{v.get_width()*100:.1f}%' if v.get_width() > 0.02 else '' for v in c]
            ax.bar_label(c, labels=labels, label_type='center', color='white', weight='bold')
            
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.savefig(self.output_dir / filename, bbox_inches='tight')
        plt.close(fig)

    def plot_confusion_analysis(self, cm: np.ndarray, class_names: List[str], task_name: str, filename: str = "confusion_matrix.png", top_k: int = 3, threshold: int = 10):
        """Analyzes and plots a confusion matrix, choosing the best plot type."""
        json_path = self.output_dir / Path(filename).with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump({'task_name': task_name, 'class_names': class_names, 'confusion_matrix': cm.tolist()}, f, indent=4)

        if len(class_names) <= threshold:
            self._plot_confusion_heatmap(cm, class_names, task_name, filename)
        else:
            self._plot_top_k_misclassifications(cm, class_names, task_name, top_k, filename)

    def _plot_confusion_heatmap(self, cm: np.ndarray, class_names: List[str], task_name: str, filename: str):
        """Internal function to plot a standard normalized confusion matrix heatmap."""
        with np.errstate(divide='ignore', invalid='ignore'):
            cm_normalized = np.nan_to_num(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis])
        
        fig, ax = plt.subplots(figsize=(max(8, len(class_names)*0.8), max(6, len(class_names)*0.7)))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_title(f'Normalized Confusion Matrix for {task_name.title()}', fontsize=16)
        ax.set_xlabel('Predicted Label', fontsize=12); ax.set_ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(self.output_dir / filename)
        plt.close(fig)

    def _plot_top_k_misclassifications(self, cm: np.ndarray, class_names: List[str], task_name: str, k: int, filename: str):
        """Internal function to plot top-k misclassifications for each class."""
        np.fill_diagonal(cm, 0)
        plot_data = [{'True Class': true_class, 'Predicted Class': class_names[j], 'Count': cm[i, j]}
                     for i, true_class in enumerate(class_names) if true_class != 'Background'
                     for j in np.argsort(cm[i, :])[::-1][:k] if cm[i, j] > 0]
        
        if not plot_data:
            print(f"No misclassifications to plot for task {task_name}.")
            return

        df = pd.DataFrame(plot_data)
        df_pivot = df.pivot(index='True Class', columns='Predicted Class', values='Count').fillna(0)
        
        fig, ax = plt.subplots(figsize=(12, 1 + len(df_pivot) * 0.4))
        df_pivot.plot(kind='barh', stacked=True, ax=ax, colormap='tab20')
        
        ax.invert_yaxis()
        ax.set_xlabel('Misclassification Count', fontsize=12)
        ax.set_ylabel('True Class', fontsize=12)
        ax.set_title(f'Top {k} Misclassifications for {task_name.title()} Task', fontsize=18, pad=20)
        ax.legend(title='Predicted As', bbox_to_anchor=(1.02, 1), loc='upper left')
        
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.savefig(self.output_dir / filename)
        plt.close(fig)
        
    @staticmethod
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