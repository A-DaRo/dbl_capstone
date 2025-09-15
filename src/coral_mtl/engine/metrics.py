import torch
import numpy as np
from typing import Dict, List
from scipy.ndimage import binary_dilation

EPS = 1e-6

class CoralMTLMetrics:
    """
    A comprehensive metrics calculator for the Coral-MTL project.

    Accumulates statistics over multiple batches and computes mIoU, mPA, Boundary IoU,
    and a final H-Mean score.
    """
    def __init__(self,
                 num_classes: Dict[str, int],
                 device: torch.device,
                 boundary_thickness: int = 2,
                 ignore_index: int = 255,
                 primary_tasks: List[str] = ['genus', 'health'],
                 aux_tasks: List[str] = ['fish', 'human_artifacts', 'substrate']):
        self.num_classes = num_classes
        self.device = device
        self.boundary_thickness = boundary_thickness
        self.ignore_index = ignore_index

        self.primary_tasks = primary_tasks
        self.aux_tasks = aux_tasks
        self.all_tasks = self.primary_tasks + self.aux_tasks
        
        self.reset()

    def reset(self):
        """Resets all accumulated statistics to zero."""
        self.confusion_matrices = {
            task: torch.zeros((n, n), dtype=torch.int64, device=self.device)
            for task, n in self.num_classes.items()
        }
        self.biou_stats = {
            task: {'intersection': 0.0, 'union': 0.0} for task in self.primary_tasks
        }

    def update(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]):
        """Updates internal statistics with a new batch of predictions and targets."""
        preds = {task: torch.argmax(logits, dim=1) for task, logits in predictions.items()}
        
        for task in self.all_tasks:
            pred_task, target_task = preds[task], targets[task]
            n_cls = self.num_classes[task]

            mask = (target_task != self.ignore_index)
            cm_update = torch.bincount(
                n_cls * target_task[mask].long() + pred_task[mask].long(),
                minlength=n_cls**2
            ).reshape(n_cls, n_cls)
            self.confusion_matrices[task] += cm_update.to(self.device)

            if task in self.primary_tasks:
                self._update_biou_stats(pred_task.cpu().numpy(), target_task.cpu().numpy(), task)

    def _update_biou_stats(self, pred_np: np.ndarray, target_np: np.ndarray, task_name: str):
        """Helper to compute and accumulate boundary stats for a batch."""
        for i in range(pred_np.shape[0]):
            for c in range(1, self.num_classes[task_name]):
                gt_c, pred_c = (target_np[i] == c), (pred_np[i] == c)
                if not gt_c.any(): continue

                gt_boundary = binary_dilation(gt_c, iterations=self.boundary_thickness) & ~gt_c
                pred_boundary = binary_dilation(pred_c, iterations=self.boundary_thickness) & ~pred_c
                
                self.biou_stats[task_name]['intersection'] += np.sum(gt_boundary & pred_boundary)
                self.biou_stats[task_name]['union'] += np.sum(gt_boundary | pred_boundary)

    def compute(self) -> Dict[str, float]:
        """Computes all final metrics from the accumulated statistics."""
        results = {}
        for task in self.all_tasks:
            cm = self.confusion_matrices[task].cpu().numpy()
            tp = np.diag(cm)
            fp, fn = cm.sum(axis=0) - tp, cm.sum(axis=1) - tp
            
            iou = tp / (tp + fp + fn + EPS)
            pa = tp / (tp + fn + EPS)
            
            results[f'mIoU_{task}'] = np.nanmean(iou)
            results[f'mPA_{task}'] = np.nanmean(pa)
            
            if task in self.aux_tasks:
                for c in range(1, self.num_classes[task]):
                    results[f'IoU_{task}_class_{c}'] = iou[c]
        
        for task in self.primary_tasks:
            stats = self.biou_stats[task]
            results[f'BIoU_{task}'] = stats['intersection'] / (stats['union'] + EPS)

        results['H-Mean'] = (results['mIoU_genus'] + results['mIoU_health']) / 2.0
        
        return results