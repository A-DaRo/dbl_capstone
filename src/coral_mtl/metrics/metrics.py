"""
Metrics calculation module for coral segmentation tasks.
This module provides:
    A hierarchical metrics calculation system that computes detailed per-class,
    per-task (at grouped and ungrouped levels), and global metrics.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Tuple
from abc import ABC, abstractmethod
from coral_mtl.utils.task_splitter import TaskSplitter
from .metrics_storer import MetricsStorer, AsyncMetricsStorer, AdvancedMetricsProcessor


EPS = 1e-6


def gpu_binary_dilation(binary_mask: torch.Tensor, iterations: int) -> torch.Tensor:
    """Performs binary dilation on a GPU tensor using a 3x3 kernel."""
    kernel = torch.ones(1, 1, 3, 3, device=binary_mask.device, dtype=torch.float32)
    
    # Handle both single images and batches
    if binary_mask.dim() == 2:
        binary_mask = binary_mask.unsqueeze(0)  # Add batch dimension
        squeeze_batch = True
    else:
        squeeze_batch = False
    
    dilated = binary_mask.float().unsqueeze(1)  # Add channel dimension
    for _ in range(iterations):
        dilated = (F.conv2d(dilated, kernel, padding=1) > 0).float()
    
    result = dilated.squeeze(1).bool()  # Remove channel dimension
    
    if squeeze_batch:
        result = result.squeeze(0)  # Remove batch dimension if we added it
    
    return result


class AbstractCoralMetrics(ABC):
    """
    Abstract base class for calculating coral segmentation metrics.
    """
    def __init__(self,
                 splitter: TaskSplitter,
                 storer: MetricsStorer,
                 device: torch.device,
                 boundary_thickness: int = 2,
                 ignore_index: int = 255,
                 use_async_storage: bool = True):
        self.splitter = splitter
        self.storer = storer
        # Create async storer if requested and regular storer is provided
        if use_async_storage and isinstance(storer, MetricsStorer):
            self.async_storer = AsyncMetricsStorer(storer.output_dir)
        else:
            self.async_storer = None
        self.device = device
        self.boundary_thickness = boundary_thickness
        self.ignore_index = ignore_index
        self.all_tasks = list(self.splitter.hierarchical_definitions.keys())

    def reset(self):
        """Resets all accumulated statistics for a new epoch or run."""
        self.task_cms = {
            task: torch.zeros((len(details['ungrouped']['id2label']), len(details['ungrouped']['id2label'])),
                              dtype=torch.int64, device=self.device)
            for task, details in self.splitter.hierarchical_definitions.items()
        }
        self.global_cm = torch.zeros((self.splitter.num_global_classes, self.splitter.num_global_classes),
                                     dtype=torch.int64, device=self.device)
        self.biou_stats = {
            task: {'ungrouped': {'intersection': 0.0, 'union': 0.0},
                   'grouped': {'intersection': 0.0, 'union': 0.0} if details.get('is_grouped') else None}
            for task, details in self.splitter.hierarchical_definitions.items()
        }
        # Initialize global BIoU stats with explicit float values
        self.global_biou_stats = {'intersection': 0.0, 'union': 0.0}
        
        # Tier 1 - New GPU-based accumulators for advanced metrics
        # Boundary statistics (GPU tensors for efficient accumulation)
        self.boundary_tp_global = torch.zeros(1, dtype=torch.float32, device=self.device)
        self.boundary_fp_global = torch.zeros(1, dtype=torch.float32, device=self.device)
        self.boundary_fn_global = torch.zeros(1, dtype=torch.float32, device=self.device)
        
        # Probabilistic statistics (for ECE, NLL, Brier)
        self.total_nll = torch.zeros(1, dtype=torch.float32, device=self.device)
        self.total_brier = torch.zeros(1, dtype=torch.float32, device=self.device)
        self.total_pixels = torch.zeros(1, dtype=torch.int64, device=self.device)
        
        # ECE bins (10 bins for confidence calibration)
        num_bins = 10
        self.ece_bin_boundaries = torch.linspace(0, 1, num_bins + 1, device=self.device)
        self.ece_bin_confidences = torch.zeros(num_bins, dtype=torch.float32, device=self.device)
        self.ece_bin_accuracies = torch.zeros(num_bins, dtype=torch.float32, device=self.device)
        self.ece_bin_counts = torch.zeros(num_bins, dtype=torch.int64, device=self.device)

    @abstractmethod
    def update(self, predictions: Any, original_targets: torch.Tensor, image_ids: List[str], 
              epoch: int, predictions_logits: Any = None, store_per_image: bool = True, 
              is_testing: bool = False):
        """
        Update metrics with batch predictions and targets.
        
        Args:
            predictions: Model predictions (argmax of logits)
            original_targets: Ground truth targets
            image_ids: List of image identifiers
            epoch: Current training epoch
            predictions_logits: Raw model logits for probabilistic metrics (Tier 1)
            store_per_image: Whether to store per-image data (Tier 2)
            is_testing: Whether this is test evaluation
        """
        pass

    def _update_biou_stats_gpu(self, pred_mask: torch.Tensor, target_mask: torch.Tensor, task_name: str, level: str):
        """Calculates and accumulates BIoU stats on the GPU."""
        num_classes = len(self.splitter.hierarchical_definitions[task_name][level]['id2label'])
        for c in range(1, num_classes): # Skip background
            gt_c = (target_mask == c)
            pred_c = (pred_mask == c)
            
            # Process only if there is ground truth for this class
            if not torch.any(gt_c):
                continue

            gt_boundary = gpu_binary_dilation(gt_c, self.boundary_thickness) & ~gt_c
            pred_boundary = gpu_binary_dilation(pred_c, self.boundary_thickness) & ~pred_c
            
            # Handle both single images and batches
            if gt_boundary.dim() == 2:  # Single image
                intersection = torch.sum(gt_boundary & pred_boundary)
                union = torch.sum(gt_boundary | pred_boundary)
            else:  # Batch
                intersection = torch.sum(gt_boundary & pred_boundary, dim=[1, 2])
                union = torch.sum(gt_boundary | pred_boundary, dim=[1, 2])
                intersection = torch.sum(intersection)
                union = torch.sum(union)
            
            self.biou_stats[task_name][level]['intersection'] += intersection.item()
            self.biou_stats[task_name][level]['union'] += union.item()

    def _update_global_biou_stats_gpu(self, global_pred_mask: torch.Tensor, global_target_mask: torch.Tensor):
        """Update global BIoU statistics on the GPU."""        
        total_intersection = 0.0
        total_union = 0.0
        
        for c in range(1, self.splitter.num_global_classes):  # Skip background
            gt_c = (global_target_mask == c)
            pred_c = (global_pred_mask == c)

            if not torch.any(gt_c):
                continue

            gt_boundary = gpu_binary_dilation(gt_c, self.boundary_thickness) & ~gt_c
            pred_boundary = gpu_binary_dilation(pred_c, self.boundary_thickness) & ~pred_c

            # Handle both single images and batches
            if gt_boundary.dim() == 2:  # Single image
                intersection = torch.sum(gt_boundary & pred_boundary)
                union = torch.sum(gt_boundary | pred_boundary)
            else:  # Batch
                intersection = torch.sum(gt_boundary & pred_boundary, dim=[1, 2])
                union = torch.sum(gt_boundary | pred_boundary, dim=[1, 2])
                intersection = torch.sum(intersection)
                union = torch.sum(union)

            total_intersection += intersection.item()
            total_union += union.item()

        self.global_biou_stats['intersection'] += total_intersection
        self.global_biou_stats['union'] += total_union

    def _update_boundary_stats_gpu(self, pred_mask: torch.Tensor, target_mask: torch.Tensor):
        """Update boundary statistics for global Boundary F1 metric on GPU."""
        # Process all foreground classes (skip background class 0)
        for c in range(1, self.splitter.num_global_classes):
            gt_c = (target_mask == c)
            pred_c = (pred_mask == c)
            
            # Skip if no ground truth for this class
            if not torch.any(gt_c):
                continue
                
            # Create boundary masks
            gt_boundary = gpu_binary_dilation(gt_c, self.boundary_thickness) & ~gt_c
            pred_boundary = gpu_binary_dilation(pred_c, self.boundary_thickness) & ~pred_c
            
            # Calculate boundary metrics
            tp = torch.sum(gt_boundary & pred_boundary).float()
            fp = torch.sum(pred_boundary & ~gt_boundary).float()  
            fn = torch.sum(gt_boundary & ~pred_boundary).float()
            
            # Accumulate stats
            self.boundary_tp_global += tp
            self.boundary_fp_global += fp
            self.boundary_fn_global += fn

    def _update_probabilistic_stats_gpu(self, logits: torch.Tensor, target_mask: torch.Tensor):
        """Update probabilistic statistics (NLL, Brier, ECE) on GPU."""
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=1)  # Shape: [B, C, H, W]
        
        # Flatten for easier processing
        probs_flat = probs.view(probs.shape[0], probs.shape[1], -1)  # [B, C, H*W]
        targets_flat = target_mask.view(target_mask.shape[0], -1)    # [B, H*W]
        
        batch_size, num_classes, num_pixels = probs_flat.shape
        
        # Create valid mask (exclude ignore_index)
        valid_mask = (targets_flat != self.ignore_index)
        
        for b in range(batch_size):
            valid_pixels = valid_mask[b]
            if not torch.any(valid_pixels):
                continue
                
            batch_probs = probs_flat[b, :, valid_pixels]  # [C, valid_pixels]
            batch_targets = targets_flat[b, valid_pixels]  # [valid_pixels]
            
            # NLL calculation
            log_probs = torch.log(batch_probs + 1e-8)
            nll = F.nll_loss(log_probs.t(), batch_targets, reduction='sum')
            self.total_nll += nll
            
            # Brier Score calculation  
            # Create one-hot targets
            targets_one_hot = F.one_hot(batch_targets, num_classes).float()  # [valid_pixels, C]
            brier_scores = torch.sum((batch_probs.t() - targets_one_hot) ** 2, dim=1)
            self.total_brier += torch.sum(brier_scores)
            
            # ECE calculation
            # Get confidence (max probability) and predictions
            confidences, predictions = torch.max(batch_probs, dim=0)
            correct = (predictions == batch_targets)
            
            # Assign to bins
            bin_indices = torch.searchsorted(self.ece_bin_boundaries[1:], confidences)
            bin_indices = torch.clamp(bin_indices, 0, len(self.ece_bin_boundaries) - 2)
            
            # Accumulate per bin
            for bin_idx in range(len(self.ece_bin_boundaries) - 1):
                bin_mask = (bin_indices == bin_idx)
                if torch.any(bin_mask):
                    self.ece_bin_confidences[bin_idx] += torch.sum(confidences[bin_mask])
                    self.ece_bin_accuracies[bin_idx] += torch.sum(correct[bin_mask].float())
                    self.ece_bin_counts[bin_idx] += torch.sum(bin_mask)
            
            # Track total pixels processed
            self.total_pixels += torch.sum(valid_pixels)

    def _compute_metrics_from_cm(self, cm: np.ndarray, class_names: List[str]) -> Dict[str, Any]:
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp
        iou = tp / (tp + fp + fn + EPS)
        precision = tp / (tp + fp + EPS)
        recall = tp / (tp + fn + EPS)
        f1_score = 2 * (precision * recall) / (precision + recall + EPS)
        support = cm.sum(axis=1).astype(int)
        
        task_summary = {
            'mIoU': np.nanmean(iou), 'mPrecision': np.nanmean(precision),
            'mRecall': np.nanmean(recall), 'mF1-Score': np.nanmean(f1_score),
            'pixel_accuracy': tp.sum() / (cm.sum() + EPS)
        }
        per_class = {name: {'IoU': iou[i], 'Precision': precision[i], 'Recall': recall[i],
                            'F1-Score': f1_score[i], 'support': support[i]}
                     for i, name in enumerate(class_names)}
        
        total_pixels = cm.sum() + EPS
        tide = {}
        if cm.shape[0] > 1:
            foreground_cm = cm[1:, 1:]
            tide = {
                'classification_error': (foreground_cm.sum() - np.diag(foreground_cm).sum()) / total_pixels,
                'background_error': cm[0, 1:].sum() / total_pixels,
                'missed_error': cm[1:, 0].sum() / total_pixels
            }
        return {'task_summary': task_summary, 'per_class': per_class, 'TIDE_errors': tide}

    def _aggregate_cm(self, fine_cm: np.ndarray, mapping: np.ndarray, num_grouped: int) -> np.ndarray:
        grouped_cm = np.zeros((num_grouped, num_grouped), dtype=fine_cm.dtype)
        for i in range(fine_cm.shape[0]):
            for j in range(fine_cm.shape[1]):
                grouped_cm[mapping[i], mapping[j]] += fine_cm[i, j]
        return grouped_cm

    def compute(self) -> Dict[str, Any]:
        """Computes all final metrics from accumulated statistics and returns a structured report."""
        report = {'tasks': {}, 'global_summary': {}, 'optimization_metrics': {}}
        
        # Per-Task Metrics
        for task, details in self.splitter.hierarchical_definitions.items():
            report['tasks'][task] = {}
            cm_np = self.task_cms[task].cpu().numpy()
            
            for level in ['ungrouped', 'grouped']:
                if level == 'grouped' and not details.get('is_grouped'): continue
                
                level_cm = self._aggregate_cm(cm_np, details['ungrouped_to_grouped_map'], len(details[level]['id2label'])) if level == 'grouped' else cm_np
                level_report = self._compute_metrics_from_cm(level_cm, details[level]['class_names'])
                stats = self.biou_stats[task][level]
                level_report['BIoU'] = stats['intersection'] / (stats['union'] + EPS)
                report['tasks'][task][level] = level_report
                report['optimization_metrics'][f'tasks.{task}.{level}.mIoU'] = level_report['task_summary']['mIoU']
                report['optimization_metrics'][f'tasks.{task}.{level}.BIoU'] = level_report['BIoU']
        
        # Global Metrics
        global_cm_np = self.global_cm.cpu().numpy()
        global_summary = self._compute_metrics_from_cm(global_cm_np, self.splitter.global_class_names)
        report['global_summary'] = global_summary
        
        # Add global metrics to optimization_metrics
        report['optimization_metrics']['global.mIoU'] = global_summary['task_summary']['mIoU']
        
        # Calculate global BIoU with better handling and debugging
        global_biou_intersection = self.global_biou_stats['intersection']
        global_biou_union = self.global_biou_stats['union']
        
        if global_biou_union > 0:
            global_biou = global_biou_intersection / global_biou_union
        else:
            # If no boundaries detected, set BIoU to 0
            global_biou = 0.0
        
        report['optimization_metrics']['global.BIoU'] = global_biou
        
        # Add global TIDE metrics to optimization_metrics
        if global_summary['TIDE_errors']:
            report['optimization_metrics']['global.classification_error'] = global_summary['TIDE_errors']['classification_error']
            report['optimization_metrics']['global.background_error'] = global_summary['TIDE_errors']['background_error'] 
            report['optimization_metrics']['global.missed_error'] = global_summary['TIDE_errors']['missed_error']

        # Tier 1 Advanced Metrics (GPU-aggregated)
        # Boundary F1 (from boundary TP, FP, FN)
        boundary_tp = self.boundary_tp_global.item()
        boundary_fp = self.boundary_fp_global.item()
        boundary_fn = self.boundary_fn_global.item()
        
        boundary_precision = boundary_tp / (boundary_tp + boundary_fp + EPS)
        boundary_recall = boundary_tp / (boundary_tp + boundary_fn + EPS)
        boundary_f1 = 2 * (boundary_precision * boundary_recall) / (boundary_precision + boundary_recall + EPS)
        
        report['optimization_metrics']['global.Boundary_F1'] = boundary_f1
        report['optimization_metrics']['global.Boundary_Precision'] = boundary_precision
        report['optimization_metrics']['global.Boundary_Recall'] = boundary_recall
        
        # Probabilistic Metrics (NLL, Brier, ECE)
        total_pixels = self.total_pixels.item()
        if total_pixels > 0:
            # NLL and Brier (normalized by total pixels)
            nll = (self.total_nll / total_pixels).item()
            brier = (self.total_brier / total_pixels).item()
            
            report['optimization_metrics']['global.NLL'] = nll
            report['optimization_metrics']['global.Brier_Score'] = brier
            
            # ECE calculation
            ece = 0.0
            for bin_idx in range(len(self.ece_bin_boundaries) - 1):
                bin_count = self.ece_bin_counts[bin_idx].item()
                if bin_count > 0:
                    bin_accuracy = (self.ece_bin_accuracies[bin_idx] / bin_count).item()
                    bin_confidence = (self.ece_bin_confidences[bin_idx] / bin_count).item()
                    ece += (bin_count / total_pixels) * abs(bin_accuracy - bin_confidence)
            
            report['optimization_metrics']['global.ECE'] = ece
        else:
            report['optimization_metrics']['global.NLL'] = 0.0
            report['optimization_metrics']['global.Brier_Score'] = 0.0
            report['optimization_metrics']['global.ECE'] = 0.0

        return report

class CoralMTLMetrics(AbstractCoralMetrics):
    """Metrics calculator for Multi-Task Learning models."""
    def update(self, predictions: Dict[str, torch.Tensor], original_targets: torch.Tensor, image_ids: List[str], 
              epoch: int, predictions_logits: Dict[str, torch.Tensor] = None, store_per_image: bool = True, 
              is_testing: bool = False):
        # Use provided logits if available, otherwise predictions are assumed to be logits  
        if predictions_logits is not None:
            preds = {task: torch.argmax(logits, dim=1) for task, logits in predictions_logits.items()}
            logits = predictions_logits
        else:
            # Assume predictions are logits for backwards compatibility
            preds = {task: torch.argmax(logits, dim=1) for task, logits in predictions.items()}
            logits = predictions
            
        mask = (original_targets != self.ignore_index)
        batch_size = original_targets.shape[0]
        global_target = self.splitter.global_mapping_torch.to(self.device)[original_targets]

        # Calculate global predictions for BIoU and other metrics
        pred_fg = torch.zeros_like(original_targets, dtype=torch.long)
        for p in preds.values():
            pred_fg |= (p > 0)
        global_pred = pred_fg * self.splitter.global_mapping_torch.to(self.device)[original_targets]

        # Tier 1 GPU Updates (Fast aggregation)
        self._update_global_biou_stats_gpu(global_pred, global_target)
        self._update_boundary_stats_gpu(global_pred, global_target)
        
        # Update probabilistic stats if logits are available (for primary task)
        if logits and len(logits) > 0:
            # Use the first task's logits as representative for global metrics
            first_task_logits = next(iter(logits.values()))
            if first_task_logits is not None:
                self._update_probabilistic_stats_gpu(first_task_logits, global_target)

        for i in range(batch_size):
            img_id = image_ids[i]
            img_mask = mask[i]
            per_image_cms = {}
            per_image_predictions = {}

            # Update per-task CMs and BIoU stats
            for task, details in self.splitter.hierarchical_definitions.items():
                mapping = torch.from_numpy(details['ungrouped']['mapping_array']).to(self.device)
                target_ungrouped = mapping[original_targets[i]]
                pred_ungrouped = preds[task][i]
                
                n_cls = len(details['ungrouped']['id2label'])
                cm_update = torch.bincount(
                    n_cls * target_ungrouped[img_mask].long() + pred_ungrouped[img_mask].long(),
                    minlength=n_cls**2).reshape(n_cls, n_cls)
                self.task_cms[task] += cm_update
                
                if store_per_image:
                    per_image_cms[task] = cm_update.cpu().numpy()
                    per_image_predictions[task] = pred_ungrouped.cpu().numpy()

                # Update BIoU stats for this task on GPU
                self._update_biou_stats_gpu(
                    pred_ungrouped.unsqueeze(0), 
                    target_ungrouped.unsqueeze(0), 
                    task, 'ungrouped'
                )
                if details.get('is_grouped'):
                    grouped_pred = torch.from_numpy(details['ungrouped_to_grouped_map']).to(self.device)[pred_ungrouped]
                    grouped_target = torch.from_numpy(details['ungrouped_to_grouped_map']).to(self.device)[target_ungrouped]
                    self._update_biou_stats_gpu(
                        grouped_pred.unsqueeze(0), 
                        grouped_target.unsqueeze(0), 
                        task, 'grouped'
                    )

            # Update Global CM
            pred_fg_img = torch.zeros_like(original_targets[i], dtype=torch.long)
            for p_task in preds.values():
                pred_fg_img |= (p_task[i] > 0)
            global_pred_img = pred_fg_img * self.splitter.global_mapping_torch.to(self.device)[original_targets[i]]

            n_cls_global = self.splitter.num_global_classes
            global_cm_update = torch.bincount(
                n_cls_global * global_target[i][img_mask].long() + global_pred_img[img_mask].long(),
                minlength=n_cls_global**2).reshape(n_cls_global, n_cls_global)
            self.global_cm += global_cm_update
            
            if store_per_image:
                per_image_cms['global'] = global_cm_update.cpu().numpy()
                per_image_predictions['global'] = global_pred_img.cpu().numpy()
                
                # Get task definitions for class counts (needed for BIoU computation)
                num_classes_per_task = {}
                target_masks_for_biou = {}
                
                for task, details in self.splitter.hierarchical_definitions.items():
                    num_classes_per_task[task] = len(details['ungrouped']['id2label'])
                    # Get target mask for this task
                    mapping = torch.from_numpy(details['ungrouped']['mapping_array']).to(self.device)
                    target_masks_for_biou[task] = mapping[original_targets[i]].cpu().numpy()
                
                num_classes_per_task['global'] = self.splitter.num_global_classes
                target_masks_for_biou['global'] = global_target[i].cpu().numpy()
                
                # Use new storage method that computes per-image metrics
                if self.async_storer:
                    self.async_storer.store_per_image_cms_with_metrics(
                        img_id, per_image_cms, per_image_predictions, target_masks_for_biou,
                        num_classes_per_task, is_testing=is_testing, epoch=epoch
                    )
                else:
                    self.storer.store_per_image_cms_with_metrics(
                        img_id, per_image_cms, per_image_predictions, target_masks_for_biou,
                        num_classes_per_task, is_testing=is_testing, epoch=epoch
                    )

class CoralMetrics(AbstractCoralMetrics):
    """Metrics calculator for baseline (single-head) models."""
    def update(self, predictions: torch.Tensor, original_targets: torch.Tensor, image_ids: List[str], 
              epoch: int, predictions_logits: torch.Tensor = None, store_per_image: bool = True, 
              is_testing: bool = False):
        # Use provided logits if available, otherwise assume predictions are logits
        if predictions_logits is not None:
            logits = predictions_logits
            flat_preds = torch.argmax(logits, dim=1)
        else:
            # Assume predictions are logits for backwards compatibility
            logits = predictions
            flat_preds = torch.argmax(predictions, dim=1)
            
        mask = (original_targets != self.ignore_index)
        
        original_preds = self.splitter.flat_to_original_mapping_torch.to(self.device)[flat_preds]
        global_target = self.splitter.global_mapping_torch.to(self.device)[original_targets]
        global_pred = self.splitter.global_mapping_torch.to(self.device)[original_preds]
        
        # Tier 1 GPU Updates (Fast aggregation)
        self._update_global_biou_stats_gpu(global_pred, global_target)
        self._update_boundary_stats_gpu(global_pred, global_target)
        
        # Update probabilistic stats if logits are available
        if logits is not None:
            self._update_probabilistic_stats_gpu(logits, global_target)
        
        batch_size = original_targets.shape[0]
        for i in range(batch_size):
            img_id = image_ids[i]
            img_mask = mask[i]
            per_image_cms = {}
            per_image_predictions = {}

            # Update per-task CMs and BIoU stats
            for task, details in self.splitter.hierarchical_definitions.items():
                mapping = torch.from_numpy(details['ungrouped']['mapping_array']).to(self.device)
                target_ungrouped = mapping[original_targets[i]]
                pred_ungrouped = mapping[original_preds[i]]
                
                n_cls = len(details['ungrouped']['id2label'])
                cm_update = torch.bincount(
                    n_cls * target_ungrouped[img_mask].long() + pred_ungrouped[img_mask].long(),
                    minlength=n_cls**2).reshape(n_cls, n_cls)
                self.task_cms[task] += cm_update
                
                if store_per_image:
                    per_image_cms[task] = cm_update.cpu().numpy()
                    per_image_predictions[task] = pred_ungrouped.cpu().numpy()

                # Update BIoU stats for this task on GPU
                self._update_biou_stats_gpu(
                    pred_ungrouped.unsqueeze(0), 
                    target_ungrouped.unsqueeze(0), 
                    task, 'ungrouped'
                )
                if details.get('is_grouped'):
                    grouped_pred = torch.from_numpy(details['ungrouped_to_grouped_map']).to(self.device)[pred_ungrouped]
                    grouped_target = torch.from_numpy(details['ungrouped_to_grouped_map']).to(self.device)[target_ungrouped]
                    self._update_biou_stats_gpu(
                        grouped_pred.unsqueeze(0), 
                        grouped_target.unsqueeze(0), 
                        task, 'grouped'
                    )

            # Update Global CM
            n_cls_global = self.splitter.num_global_classes
            global_cm_update = torch.bincount(
                n_cls_global * global_target[i][img_mask].long() + global_pred[i][img_mask].long(),
                minlength=n_cls_global**2).reshape(n_cls_global, n_cls_global)
            self.global_cm += global_cm_update
            
            if store_per_image:
                per_image_cms['global'] = global_cm_update.cpu().numpy()
                per_image_predictions['global'] = global_pred[i].cpu().numpy()
                
                # Get task definitions for class counts (needed for BIoU computation)
                num_classes_per_task = {}
                target_masks_for_biou = {}
                
                for task, details in self.splitter.hierarchical_definitions.items():
                    num_classes_per_task[task] = len(details['ungrouped']['id2label'])
                    # Get target mask for this task
                    mapping = torch.from_numpy(details['ungrouped']['mapping_array']).to(self.device)
                    target_masks_for_biou[task] = mapping[original_targets[i]].cpu().numpy()
                
                num_classes_per_task['global'] = self.splitter.num_global_classes
                target_masks_for_biou['global'] = global_target[i].cpu().numpy()
                
                # Use new storage method that computes per-image metrics
                if self.async_storer:
                    self.async_storer.store_per_image_cms_with_metrics(
                        img_id, per_image_cms, per_image_predictions, target_masks_for_biou,
                        num_classes_per_task, is_testing=is_testing, epoch=epoch
                    )
                else:
                    self.storer.store_per_image_cms_with_metrics(
                        img_id, per_image_cms, per_image_predictions, target_masks_for_biou,
                        num_classes_per_task, is_testing=is_testing, epoch=epoch
                    )