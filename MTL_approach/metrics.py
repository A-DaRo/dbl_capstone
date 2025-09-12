import torch
import numpy as np
from typing import Dict, List, Any
from scipy.ndimage import binary_dilation

# A small epsilon value to avoid division by zero
EPS = 1e-6

class CoralMTLMetrics:
    """
    A comprehensive metrics calculator for the Coral-MTL project.

    This class accumulates statistics over multiple batches and computes a suite of
    metrics for primary and auxiliary tasks, including mIoU, mPA, Boundary IoU,
    and a final H-Mean score for model selection.

    It is designed to be used in a standard training/validation loop:
    1. Instantiate the class.
    2. In the loop, for each batch: `metrics.update(predictions, targets)`.
    3. At the end of the epoch: `results = metrics.compute()`.
    4. Before the next epoch: `metrics.reset()`.
    """
    def __init__(self,
                 num_classes: Dict[str, int],
                 device: torch.device,
                 boundary_thickness: int = 2,
                 ignore_index: int = 255):
        """
        Args:
            num_classes (Dict[str, int]): Dictionary mapping task names to their number of classes.
            device (torch.device): The device to store confusion matrices on (e.g., 'cuda:0').
            boundary_thickness (int): The thickness in pixels for the BIoU calculation.
            ignore_index (int): The label index to ignore during metric computation.
        """
        self.num_classes = num_classes
        self.device = device
        self.boundary_thickness = boundary_thickness
        self.ignore_index = ignore_index

        self.primary_tasks = ['genus', 'health']
        self.aux_tasks = ['fish', 'human_artifacts', 'substrate']
        self.all_tasks = self.primary_tasks + self.aux_tasks
        
        self.reset()

    def reset(self):
        """Resets all accumulated statistics to zero."""
        # Confusion matrices are the most efficient way to track stats for mIoU and mPA
        self.confusion_matrices = {
            task: torch.zeros((n, n), dtype=torch.int64, device=self.device)
            for task, n in self.num_classes.items()
        }
        # BIoU requires separate tracking of intersection and union of boundaries
        self.biou_stats = {
            task: {'intersection': 0.0, 'union': 0.0}
            for task in self.primary_tasks
        }

    def update(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]):
        """
        Updates the internal statistics with a new batch of predictions and targets.

        Args:
            predictions (Dict[str, torch.Tensor]): A dictionary of raw logits from the model.
            targets (Dict[str, torch.Tensor]): A dictionary of ground truth label tensors.
        """
        # Get class predictions from logits
        preds = {task: torch.argmax(logits, dim=1) for task, logits in predictions.items()}
        
        for task in self.all_tasks:
            pred_task = preds[task]
            target_task = targets[task]
            n_cls = self.num_classes[task]

            # --- Update Confusion Matrix ---
            mask = (target_task != self.ignore_index)
            pred_flat = pred_task[mask]
            target_flat = target_task[mask]
            
            cm_update = torch.bincount(
                n_cls * target_flat.long() + pred_flat.long(),
                minlength=n_cls**2
            ).reshape(n_cls, n_cls)
            self.confusion_matrices[task] += cm_update.to(self.device)

            # --- Update BIoU stats (for primary tasks only) ---
            if task in self.primary_tasks:
                self._update_biou_stats(pred_task, target_task, task)

    def _update_biou_stats(self, pred: torch.Tensor, target: torch.Tensor, task_name: str):
        """Helper to compute and accumulate boundary stats for a batch."""
        # Move to CPU for numpy/scipy operations
        pred_np = pred.cpu().numpy()
        target_np = target.cpu().numpy()
        
        for i in range(pred_np.shape[0]): # Iterate over batch
            for c in range(1, self.num_classes[task_name]): # Iterate over each class (skip background)
                gt_c = (target_np[i] == c)
                pred_c = (pred_np[i] == c)

                # Skip if class not in ground truth
                if not gt_c.any():
                    continue

                # Create boundary bands using dilation
                gt_boundary = binary_dilation(gt_c, iterations=self.boundary_thickness) & ~gt_c
                pred_boundary = binary_dilation(pred_c, iterations=self.boundary_thickness) & ~pred_c
                
                intersection = np.sum(gt_boundary & pred_boundary)
                union = np.sum(gt_boundary | pred_boundary)
                
                self.biou_stats[task_name]['intersection'] += intersection
                self.biou_stats[task_name]['union'] += union

    def compute(self) -> Dict[str, float]:
        """
        Computes all the final metrics from the accumulated statistics.

        Returns:
            Dict[str, float]: A dictionary containing all computed metrics.
        """
        results = {}
        
        # --- Calculate metrics from confusion matrices ---
        for task in self.all_tasks:
            cm = self.confusion_matrices[task].cpu().numpy()
            
            tp = np.diag(cm)
            fp = cm.sum(axis=0) - tp
            fn = cm.sum(axis=1) - tp
            
            # IoU and Pixel Accuracy per class
            iou = tp / (tp + fp + fn + EPS)
            pa = tp / (tp + fn + EPS)
            
            # Mean scores (ignoring NaNs for classes not in the dataset)
            results[f'mIoU_{task}'] = np.nanmean(iou)
            results[f'mPA_{task}'] = np.nanmean(pa)
            
            # Class-specific IoU for auxiliary tasks
            if task in self.aux_tasks:
                for c in range(1, self.num_classes[task]): # Skip background
                    results[f'IoU_{task}_class_{c}'] = iou[c]
        
        # --- Calculate Boundary IoU for primary tasks ---
        for task in self.primary_tasks:
            stats = self.biou_stats[task]
            results[f'BIoU_{task}'] = stats['intersection'] / (stats['union'] + EPS)

        # --- Calculate Overall H-Mean ---
        results['H-Mean'] = (results['mIoU_genus'] + results['mIoU_health']) / 2.0
        
        return results


if __name__ == '__main__':
    print("--- Running Sanity Check for CoralMTLMetrics ---")

    # 1. Setup parameters
    B, H, W = 4, 128, 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = {
        'genus': 9, 'health': 4, 'fish': 2, 'human_artifacts': 2, 'substrate': 4
    }
    
    # 2. Instantiate metrics calculator
    metrics = CoralMTLMetrics(num_classes=num_classes, device=device)
    
    # 3. Create dummy data
    # Predictions (logits)
    dummy_preds = {
        task: torch.randn(B, n_cls, H, W).to(device) for task, n_cls in num_classes.items()
    }
    # Targets (labels) - create a perfect prediction case for one batch
    dummy_targets = {
        task: torch.argmax(logits, dim=1) for task, logits in dummy_preds.items()
    }
    # Add some noise to the second half of the batch for more realistic metrics
    for task, n_cls in num_classes.items():
        noise = torch.randint(0, n_cls, (B//2, H, W), device=device)
        dummy_targets[task][B//2:] = noise

    print(f"Using device: {device}")
    print(f"Simulating update with batch size {B}...")

    # 4. Update metrics with the dummy batch
    metrics.update(dummy_preds, dummy_targets)

    # 5. Compute the final metrics
    final_metrics = metrics.compute()

    # 6. Print and verify the results
    print("\n--- Computed Metrics ---")
    for name, value in final_metrics.items():
        print(f"  - {name:<25}: {value:.4f}")
    
    assert 'H-Mean' in final_metrics
    assert 'mIoU_genus' in final_metrics and 0 < final_metrics['mIoU_genus'] <= 1.0
    assert 'BIoU_health' in final_metrics
    assert 'IoU_fish_class_1' in final_metrics
    assert 'IoU_substrate_class_3' in final_metrics

    # 7. Test reset functionality
    metrics.reset()
    for task, cm in metrics.confusion_matrices.items():
        assert torch.all(cm == 0)
    for task, stats in metrics.biou_stats.items():
        assert stats['intersection'] == 0.0 and stats['union'] == 0.0
    print("\nReset functionality confirmed.")

    print("\nSanity check passed! Metrics calculator is working as expected.")