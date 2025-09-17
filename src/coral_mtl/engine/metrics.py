import torch
import numpy as np
import yaml
from typing import Dict, List, Union
from scipy.ndimage import binary_dilation
from abc import ABC, abstractmethod

EPS = 1e-6

class AbstractCoralMetrics(ABC):
    """
    Abstract base class for calculating coral segmentation metrics.

    This class provides the core, shared logic for computing mIoU, mPA, Boundary IoU,
    and TIDE-style error analysis from accumulated confusion matrices. Subclasses
    must implement the `__init__` and `update` methods to handle their specific
    input formats (e.g., MTL dictionaries vs. single non-MTL tensors).
    """
    def __init__(self,
                 device: torch.device,
                 primary_tasks: List[str],
                 boundary_thickness: int = 2,
                 ignore_index: int = 255):
        self.device = device
        self.primary_tasks = primary_tasks
        self.boundary_thickness = boundary_thickness
        self.ignore_index = ignore_index

        # These must be populated by the subclass's __init__ method
        self.num_classes: Dict[str, int] = {}
        self.all_tasks: List[str] = []

    def reset(self):
        """Resets all accumulated statistics to zero."""
        if not self.num_classes:
            raise RuntimeError("Subclass must set self.num_classes before reset() is called.")
        
        self.confusion_matrices = {
            task: torch.zeros((n, n), dtype=torch.int64, device=self.device)
            for task, n in self.num_classes.items()
        }
        self.biou_stats = {
            task: {'intersection': 0.0, 'union': 0.0} for task in self.primary_tasks
        }

    @abstractmethod
    def update(self, predictions: Union[torch.Tensor, Dict[str, torch.Tensor]],
               targets: Union[torch.Tensor, Dict[str, torch.Tensor]]):
        """
        Abstract method to update internal statistics with a new batch.
        Must be implemented by subclasses.
        """
        pass

    def _update_biou_stats(self, pred_np: np.ndarray, target_np: np.ndarray, task_name: str):
        """Helper to compute and accumulate boundary stats for a batch."""
        for i in range(pred_np.shape[0]):
            for c in range(1, self.num_classes[task_name]): # Ignore background class
                gt_c, pred_c = (target_np[i] == c), (pred_np[i] == c)
                if not gt_c.any():
                    continue

                gt_boundary = binary_dilation(gt_c, iterations=self.boundary_thickness) & ~gt_c
                pred_boundary = binary_dilation(pred_c, iterations=self.boundary_thickness) & ~pred_c
                
                self.biou_stats[task_name]['intersection'] += np.sum(gt_boundary & pred_boundary)
                self.biou_stats[task_name]['union'] += np.sum(gt_boundary | pred_boundary)

    def compute(self) -> Dict[str, float]:
        """
        Computes all final metrics from the accumulated statistics.
        This implementation is shared across all subclasses.
        """
        results = {}
        for task in self.all_tasks:
            cm = self.confusion_matrices[task].cpu().numpy()
            n_cls = self.num_classes[task]
            
            # Standard Metrics
            tp = np.diag(cm)
            fp = cm.sum(axis=0) - tp
            fn = cm.sum(axis=1) - tp
            
            iou = tp / (tp + fp + fn + EPS)
            pa = tp / (cm.sum(axis=1) + EPS)
            
            results[f'mIoU_{task}'] = np.nanmean(iou)
            results[f'mPA_{task}'] = np.nanmean(pa)
            
            # Per-class IoU
            for i in range(n_cls):
                results[f'IoU_{task}_class_{i}'] = iou[i]
            
            # TIDE Error Analysis (implements spec for TIDE errors)
            total_pixels = cm.sum() + EPS
            if n_cls > 1:
                foreground_cm = cm[1:, 1:]
                cls_err = foreground_cm.sum() - np.diag(foreground_cm).sum()
                bkg_err = cm[0, 1:].sum()
                miss_err = cm[1:, 0].sum()

                results[f'TIDE_ClsError_{task}'] = cls_err / total_pixels
                results[f'TIDE_BkgError_{task}'] = bkg_err / total_pixels
                results[f'TIDE_MissError_{task}'] = miss_err / total_pixels

        for task in self.primary_tasks:
            if task in self.biou_stats:
                stats = self.biou_stats[task]
                results[f'BIoU_{task}'] = stats['intersection'] / (stats['union'] + EPS)

        # H-Mean Calculation (implements spec section 7.3.1)
        h_mean_components = [results[f'mIoU_{task}'] for task in self.primary_tasks if f'mIoU_{task}' in results]
        if len(h_mean_components) > 0:
            results['H-Mean'] = sum(h_mean_components) / len(h_mean_components)
        
        return results

class CoralMTLMetrics(AbstractCoralMetrics):
    """
    Metrics calculator for Multi-Task Learning models.
    Expects predictions and targets as dictionaries mapping task names to tensors.
    """
    def __init__(self,
                 num_classes: Dict[str, int],
                 device: torch.device,
                 primary_tasks: List[str],
                 **kwargs):
        super().__init__(device=device, primary_tasks=primary_tasks, **kwargs)
        self.num_classes = num_classes
        self.all_tasks = list(num_classes.keys())
        self.reset()

    def update(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]):
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

class CoralMetrics(AbstractCoralMetrics):
    """
    Metrics calculator for single-task (non-MTL) models.
    
    This class takes a single prediction tensor and "un-flattens" it into a
    multi-task structure using a provided task definitions dictionary, allowing for
    a direct, apples-to-apples comparison with MTL models.
    """
    def __init__(self,
                 task_definitions: Dict,
                 device: torch.device,
                 primary_tasks: List[str],
                 **kwargs):
        super().__init__(device=device, primary_tasks=primary_tasks, **kwargs)
        
        self.num_classes = {}
        self.per_task_luts = {}
        self.all_tasks = list(task_definitions.keys())
        
        # Find the max global class ID to determine LUT size
        max_global_id = 0
        for details in task_definitions.values():
            # The keys of the 'mapping' are the new IDs for the flattened mask.
            # We need to find the highest value among them to size the LUT correctly.
            # Note: these are stored as strings in the YAML, so we must cast to int.
            all_new_ids = [int(k) for k in details.get('mapping', {}).keys()]
            if all_new_ids:
                max_global_id = max(max_global_id, max(all_new_ids))

        for task_name, details in task_definitions.items():
            self.num_classes[task_name] = len(details['id2label'])
            
            # Create a lookup table (LUT) to map global IDs to task-local IDs
            lut = torch.zeros(max_global_id + 1, dtype=torch.long, device=self.device)
            for new_id_str, old_ids_list in details.get('mapping', {}).items():
                local_id = int(new_id_str)
                # Map all original class IDs to the new task-local ID
                for old_id in old_ids_list:
                   if old_id < len(lut): lut[old_id] = local_id

            self.per_task_luts[task_name] = lut
            
        self.reset()

    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        preds = torch.argmax(predictions, dim=1)
        
        for task_name in self.all_tasks:
            n_cls = self.num_classes[task_name]
            lut = self.per_task_luts[task_name]
            
            # Apply LUT to "un-flatten" the global masks to task-specific masks
            pred_task = lut[preds]
            target_task = lut[targets]

            mask = (targets != self.ignore_index) # Use original target for ignore mask
            cm_update = torch.bincount(
                n_cls * target_task[mask].long() + pred_task[mask].long(),
                minlength=n_cls**2
            ).reshape(n_cls, n_cls)
            self.confusion_matrices[task_name] += cm_update.to(self.device)

            if task_name in self.primary_tasks:
                self._update_biou_stats(pred_task.cpu().numpy(), target_task.cpu().numpy(), task_name)