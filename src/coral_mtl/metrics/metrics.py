"""
Metrics calculation module for coral segmentation tasks.
This module provides:
    A hierarchical metrics calculation system that computes detailed per-class,
    per-task (at grouped and ungrouped levels), and global metrics.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any
from abc import ABC, abstractmethod
from coral_mtl.utils.task_splitter import TaskSplitter
from .metrics_storer import MetricsStorer, AsyncMetricsStorer, AdvancedMetricsProcessor  # noqa: F401


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
            task: torch.zeros((len(details['ungrouped']['id2label']),
                               len(details['ungrouped']['id2label'])),
                              dtype=torch.int64, device=self.device)
            for task, details in self.splitter.hierarchical_definitions.items()
        }
        self.global_cm = torch.zeros((self.splitter.num_global_classes,
                                      self.splitter.num_global_classes),
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

    def _safe_lookup(self, mapping_tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Safely maps indices through a lookup tensor while preserving ignore_index."""
        result = torch.full_like(indices, self.ignore_index)
        valid = indices != self.ignore_index
        if torch.any(valid):
            clipped = torch.clamp(indices[valid], min=0, max=mapping_tensor.numel() - 1)
            result[valid] = mapping_tensor[clipped]
        return result

    @abstractmethod
    def update(self, predictions: Any, original_targets: torch.Tensor, image_ids: List[str],
               epoch: int, predictions_logits: Any = None, store_per_image: bool = True,
               is_testing: bool = False):
        """
        Update metrics with batch predictions and targets.

        Args:
            predictions: Model predictions (argmax of logits)
            original_targets: Ground truth targets (ORIGINAL space)
            image_ids: List of image identifiers
            epoch: Current training epoch
            predictions_logits: Raw model logits for probabilistic metrics (Tier 1)
            store_per_image: Whether to store per-image data (Tier 2)
            is_testing: Whether this is test evaluation
        """
        pass

    def _update_biou_stats_gpu(self, pred_mask: torch.Tensor, target_mask: torch.Tensor,
                               task_name: str, level: str):
        """Calculates and accumulates BIoU stats on the GPU."""
        num_classes = len(self.splitter.hierarchical_definitions[task_name][level]['id2label'])
        for c in range(1, num_classes):  # Skip background
            gt_c = (target_mask == c)
            pred_c = (pred_mask == c)

            # Process only if there is ground truth for this class
            if not torch.any(gt_c):
                continue

            gt_boundary = gpu_binary_dilation(gt_c, self.boundary_thickness) & ~gt_c
            pred_boundary = gpu_binary_dilation(pred_c, self.boundary_thickness) & ~pred_c

            # Compute intersection / union
            if gt_boundary.dim() == 2:
                intersection = torch.sum(gt_boundary & pred_boundary)
                union = torch.sum(gt_boundary | pred_boundary)
            else:
                intersection = torch.sum(gt_boundary & pred_boundary, dim=[1, 2])
                union = torch.sum(gt_boundary | pred_boundary, dim=[1, 2])
                intersection = torch.sum(intersection)
                union = torch.sum(union)

            # Skip if no boundary union (no perimeter present)
            if union.item() == 0:
                continue

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

            if gt_boundary.dim() == 2:
                intersection = torch.sum(gt_boundary & pred_boundary)
                union = torch.sum(gt_boundary | pred_boundary)
            else:
                intersection = torch.sum(gt_boundary & pred_boundary, dim=[1, 2])
                union = torch.sum(gt_boundary | pred_boundary, dim=[1, 2])
                intersection = torch.sum(intersection)
                union = torch.sum(union)

            # Skip if no boundary union
            if union.item() == 0:
                continue

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

            if not torch.any(gt_c):
                continue

            gt_boundary = gpu_binary_dilation(gt_c, self.boundary_thickness) & ~gt_c
            pred_boundary = gpu_binary_dilation(pred_c, self.boundary_thickness) & ~pred_c

            tp = torch.sum(gt_boundary & pred_boundary).float()
            fp = torch.sum(pred_boundary & ~gt_boundary).float()
            fn = torch.sum(gt_boundary & ~pred_boundary).float()

            # Accumulate stats
            self.boundary_tp_global += tp
            self.boundary_fp_global += fp
            self.boundary_fn_global += fn

    def _update_probabilistic_stats_gpu(self, logits: torch.Tensor, target_mask: torch.Tensor):
        """Update probabilistic statistics (NLL, Brier, ECE) on GPU."""
        probs = F.softmax(logits, dim=1)  # [B, C, H, W]

        probs_flat = probs.view(probs.shape[0], probs.shape[1], -1)  # [B, C, H*W]
        targets_flat = target_mask.view(target_mask.shape[0], -1)    # [B, H*W]

        batch_size, num_classes, _ = probs_flat.shape

        valid_mask = (targets_flat != self.ignore_index)

        for b in range(batch_size):
            valid_pixels = valid_mask[b]
            if not torch.any(valid_pixels):
                continue

            batch_probs = probs_flat[b, :, valid_pixels]  # [C, valid_pixels]
            batch_targets = targets_flat[b, valid_pixels]  # [valid_pixels]

            # NLL
            log_probs = torch.log(batch_probs + 1e-8)
            nll = F.nll_loss(log_probs.t(), batch_targets, reduction='sum')
            self.total_nll += nll

            # Brier
            targets_one_hot = F.one_hot(batch_targets, num_classes).float()
            brier_scores = torch.sum((batch_probs.t() - targets_one_hot) ** 2, dim=1)
            self.total_brier += torch.sum(brier_scores)

            # ECE
            confidences, predictions = torch.max(batch_probs, dim=0)
            correct = (predictions == batch_targets)

            bin_indices = torch.searchsorted(self.ece_bin_boundaries[1:], confidences)
            bin_indices = torch.clamp(bin_indices, 0, len(self.ece_bin_boundaries) - 2)

            for bin_idx in range(len(self.ece_bin_boundaries) - 1):
                bin_mask = (bin_indices == bin_idx)
                if torch.any(bin_mask):
                    self.ece_bin_confidences[bin_idx] += torch.sum(confidences[bin_mask])
                    self.ece_bin_accuracies[bin_idx] += torch.sum(correct[bin_mask].float())
                    self.ece_bin_counts[bin_idx] += torch.sum(bin_mask)

            self.total_pixels += torch.sum(valid_pixels)

    def _compute_metrics_from_cm(self, cm: np.ndarray, class_names: List[str]) -> Dict[str, Any]:
        tp = np.diag(cm).astype(np.float64)
        fp = (cm.sum(axis=0) - np.diag(cm)).astype(np.float64)
        fn = (cm.sum(axis=1) - np.diag(cm)).astype(np.float64)

        def _nanmean_safe(values: np.ndarray) -> float:
            return float(np.nanmean(values)) if not np.all(np.isnan(values)) else 0.0

        denom_iou = tp + fp + fn
        iou = np.divide(tp, denom_iou, out=np.full(tp.shape, np.nan, dtype=np.float64), where=denom_iou > 0)

        denom_precision = tp + fp
        precision = np.divide(tp, denom_precision, out=np.full(tp.shape, np.nan, dtype=np.float64),
                              where=denom_precision > 0)

        denom_recall = tp + fn
        recall = np.divide(tp, denom_recall, out=np.full(tp.shape, np.nan, dtype=np.float64),
                           where=denom_recall > 0)

        sum_pr = precision + recall
        f1_score = np.divide(2 * (precision * recall), sum_pr,
                             out=np.full(tp.shape, np.nan, dtype=np.float64), where=sum_pr > 0)

        support = cm.sum(axis=1).astype(int)

        task_summary = {
            'mIoU': _nanmean_safe(iou),
            'mPrecision': _nanmean_safe(precision),
            'mRecall': _nanmean_safe(recall),
            'mF1-Score': _nanmean_safe(f1_score),
            'pixel_accuracy': tp.sum() / (cm.sum() + EPS)
        }
        per_class = {
            name: {'IoU': iou[i], 'Precision': precision[i], 'Recall': recall[i],
                   'F1-Score': f1_score[i], 'support': support[i]}
            for i, name in enumerate(class_names)
        }

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
                if level == 'grouped' and not details.get('is_grouped'):
                    continue

                level_cm = (self._aggregate_cm(cm_np, details['ungrouped_to_grouped_map'],
                                               len(details[level]['id2label']))
                            if level == 'grouped' else cm_np)

                level_report = self._compute_metrics_from_cm(level_cm, details[level]['class_names'])

                stats = self.biou_stats[task][level]
                union = stats['union'] + EPS
                biou = stats['intersection'] / union
                level_report['BIoU'] = biou

                report['tasks'][task][level] = level_report
                report['optimization_metrics'][f'tasks.{task}.{level}.mIoU'] = level_report['task_summary']['mIoU']
                report['optimization_metrics'][f'tasks.{task}.{level}.BIoU'] = level_report['BIoU']

        # Global Metrics
        global_cm_np = self.global_cm.cpu().numpy()
        global_summary = self._compute_metrics_from_cm(global_cm_np, self.splitter.global_class_names)
        report['global_summary'] = global_summary

        # Add global metrics to optimization_metrics
        report['optimization_metrics']['global.mIoU'] = global_summary['task_summary']['mIoU']

        # Global BIoU
        global_biou_intersection = self.global_biou_stats['intersection']
        global_biou_union = self.global_biou_stats['union']
        global_biou = (global_biou_intersection / (global_biou_union + EPS)) if global_biou_union > 0 else 0.0
        report['optimization_metrics']['global.BIoU'] = global_biou

        # Add global TIDE metrics
        if global_summary['TIDE_errors']:
            report['optimization_metrics']['global.classification_error'] = global_summary['TIDE_errors']['classification_error']
            report['optimization_metrics']['global.background_error'] = global_summary['TIDE_errors']['background_error']
            report['optimization_metrics']['global.missed_error'] = global_summary['TIDE_errors']['missed_error']

        # Tier 1 Advanced Metrics (Boundary F1 derived from TP/FP/FN)
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
            nll = (self.total_nll / total_pixels).item()
            brier = (self.total_brier / total_pixels).item()
            report['optimization_metrics']['global.NLL'] = nll
            report['optimization_metrics']['global.Brier_Score'] = brier

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

    def __init__(self,
                 splitter: TaskSplitter,
                 storer: MetricsStorer,
                 device: torch.device,
                 boundary_thickness: int = 2,
                 ignore_index: int = 255,
                 use_async_storage: bool = True,
                 global_reference_task: str | None = None):
        super().__init__(
            splitter=splitter,
            storer=storer,
            device=device,
            boundary_thickness=boundary_thickness,
            ignore_index=ignore_index,
            use_async_storage=use_async_storage,
        )
        if global_reference_task and global_reference_task in self.splitter.hierarchical_definitions:
            self.global_reference_task = global_reference_task
        else:
            preferred = 'genus' if 'genus' in self.splitter.hierarchical_definitions else None
            self.global_reference_task = preferred or next(iter(self.splitter.hierarchical_definitions.keys()))

        global_mapping_np = np.asarray(self.splitter.global_mapping_array, dtype=np.int64)

        ref_details = self.splitter.hierarchical_definitions[self.global_reference_task]
        inverse_mapping = ref_details['ungrouped'].get('inverse_mapping_array')
        if inverse_mapping is None:
            self._global_reference_inverse = None
            self._global_reference_to_global = None
        else:
            inverse_np = np.asarray(inverse_mapping, dtype=np.int64)
            self._global_reference_inverse = torch.from_numpy(inverse_np).long()
            clipped_inverse = np.clip(inverse_np, 0, global_mapping_np.shape[0] - 1)
            reference_to_global = global_mapping_np[clipped_inverse]
            self._global_reference_to_global = torch.from_numpy(reference_to_global).long()

        self._global_background_id = int(global_mapping_np[0]) if global_mapping_np.size > 0 else 0

        # --- Precompute original->task index for each head and grouped proxies ---
        self._orig2task_idx: Dict[str, torch.Tensor] = {}
        self._grouped_proto_ungrouped_idx: Dict[str, torch.Tensor] = {}

        for task_name, det in self.splitter.hierarchical_definitions.items():
            # original -> ungrouped indices
            ung_map_np = np.asarray(det['ungrouped']['mapping_array'], dtype=np.int64)

            if det.get('is_grouped'):
                # Compose original->grouped via (original->ungrouped)->(ungrouped->grouped)
                ung2grp_np = np.asarray(det['ungrouped_to_grouped_map'], dtype=np.int64)
                orig2task_np = ung2grp_np[ung_map_np]  # original -> grouped

                # grouped -> representative ungrouped index (deterministic non-background proxy when possible)
                num_grouped = len(det['grouped']['id2label'])
                proto = np.zeros(num_grouped, dtype=np.int64)  # keep group 0 -> ungrouped 0 (background)
                for u, g in enumerate(ung2grp_np):
                    if g == 0:
                        continue  # don't override group 0 proxy
                    if proto[g] == 0 and u != 0:  # choose first non-background ungrouped id for this group
                        proto[g] = u
                # If any non-background group remained 0, fallback to 0 is acceptable.
                self._grouped_proto_ungrouped_idx[task_name] = torch.from_numpy(proto).long()
            else:
                # Head predicts ungrouped directly
                orig2task_np = ung_map_np

            self._orig2task_idx[task_name] = torch.from_numpy(orig2task_np).long()

    def update(self, predictions: Dict[str, torch.Tensor], original_targets: torch.Tensor, image_ids: List[str],
               epoch: int, predictions_logits: Dict[str, torch.Tensor] = None, store_per_image: bool = True,
               is_testing: bool = False):
        # Use provided logits if available, otherwise predictions are assumed to be logits
        if predictions_logits is not None:
            preds = {task: torch.argmax(lgts, dim=1) for task, lgts in predictions_logits.items()}
            logits = predictions_logits
        else:
            preds = {task: torch.argmax(lgts, dim=1) for task, lgts in predictions.items()}
            logits = predictions

        mask = (original_targets != self.ignore_index)
        batch_size = original_targets.shape[0]
        global_mapping = self.splitter.global_mapping_torch.to(self.device)
        global_target = self._safe_lookup(global_mapping, original_targets)

        background_global_id = self._global_background_id
        global_pred = torch.full_like(original_targets, background_global_id, dtype=torch.long, device=self.device)

        global_logits_for_metrics = None
        if logits is not None and len(logits) > 0:
            # Fused global logits from all available heads (no target leakage)
            global_logits_for_metrics = self._fuse_heads_to_global_logits(
                logits, self.splitter.global_mapping_torch.to(self.device)
            )
            global_pred = torch.argmax(global_logits_for_metrics, dim=1)
        else:
            # Fallback: map reference head to global if present; no use of targets
            reference_preds = preds.get(self.global_reference_task)
            if reference_preds is not None and self._global_reference_inverse is not None:
                inv_tensor = self._global_reference_inverse.to(self.device)
                if inv_tensor.numel() > 0:
                    clamped = torch.clamp(reference_preds.to(self.device), 0, inv_tensor.numel() - 1)
                    original_ids = inv_tensor[clamped]
                    global_pred = self._safe_lookup(global_mapping, original_ids)

        # Tier 1 GPU Updates
        self._update_global_biou_stats_gpu(global_pred, global_target)
        self._update_boundary_stats_gpu(global_pred, global_target)

        # Probabilistic stats if fused logits are available
        if global_logits_for_metrics is not None:
            self._update_probabilistic_stats_gpu(global_logits_for_metrics, global_target)

        for i in range(batch_size):
            img_id = image_ids[i]
            img_mask = mask[i]
            per_image_cms: Dict[str, np.ndarray] = {}
            per_image_predictions: Dict[str, np.ndarray] = {}

            # Per-task CMs & BIoU
            for task, details in self.splitter.hierarchical_definitions.items():
                mapping = torch.from_numpy(details['ungrouped']['mapping_array']).to(self.device).long()
                target_ungrouped = self._safe_lookup(mapping, original_targets[i])

                # Build UNGROUPED prediction even for grouped heads (use representative proxy)
                if details.get('is_grouped'):
                    proto = self._grouped_proto_ungrouped_idx[task].to(self.device)  # [num_grouped] -> ungrouped id
                    pred_grouped = preds[task][i]
                    pred_ungrouped = self._safe_lookup(proto, pred_grouped)
                else:
                    pred_ungrouped = preds[task][i]

                n_cls = len(details['ungrouped']['id2label'])
                cm_update = torch.bincount(
                    n_cls * target_ungrouped[img_mask].long() + pred_ungrouped[img_mask].long(),
                    minlength=n_cls ** 2
                ).reshape(n_cls, n_cls)
                self.task_cms[task] += cm_update

                if store_per_image:
                    per_image_cms[task] = cm_update.cpu().numpy()
                    per_image_predictions[task] = pred_ungrouped.cpu().numpy()

                # BIoU (ungrouped)
                self._update_biou_stats_gpu(
                    pred_ungrouped.unsqueeze(0),
                    target_ungrouped.unsqueeze(0),
                    task, 'ungrouped'
                )
                # BIoU (grouped) via ungrouped->grouped map
                if details.get('is_grouped'):
                    grouped_map = torch.from_numpy(details['ungrouped_to_grouped_map']).to(self.device).long()
                    grouped_pred = self._safe_lookup(grouped_map, pred_ungrouped)
                    grouped_target = self._safe_lookup(grouped_map, target_ungrouped)
                    self._update_biou_stats_gpu(
                        grouped_pred.unsqueeze(0),
                        grouped_target.unsqueeze(0),
                        task, 'grouped'
                    )

            # Global CM
            global_pred_img = global_pred[i]
            n_cls_global = self.splitter.num_global_classes
            global_cm_update = torch.bincount(
                n_cls_global * global_target[i][img_mask].long() + global_pred_img[img_mask].long(),
                minlength=n_cls_global ** 2
            ).reshape(n_cls_global, n_cls_global)
            self.global_cm += global_cm_update

            if store_per_image:
                per_image_cms['global'] = global_cm_update.cpu().numpy()
                per_image_predictions['global'] = global_pred_img.cpu().numpy()

                # Prepare per-image BIoU inputs (targets only; predictions already stored)
                num_classes_per_task = {}
                target_masks_for_biou = {}

                for task, details in self.splitter.hierarchical_definitions.items():
                    num_classes_per_task[task] = len(details['ungrouped']['id2label'])
                    mapping = torch.from_numpy(details['ungrouped']['mapping_array']).to(self.device).long()
                    target_masks_for_biou[task] = self._safe_lookup(mapping, original_targets[i]).cpu().numpy()

                num_classes_per_task['global'] = self.splitter.num_global_classes
                target_masks_for_biou['global'] = global_target[i].cpu().numpy()

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

    def _fuse_heads_to_global_logits(
        self,
        logits_dict: Dict[str, torch.Tensor],
        orig2global: torch.Tensor
    ) -> torch.Tensor:
        """
        Build fused global logits by (a) summing per-head log-probs at original-class level,
        then (b) marginalizing originals -> global via scatter-add in probability space,
        and (c) returning log(prob) so downstream softmax recovers the same distribution.
        """
        any_head = next(iter(logits_dict.values())).to(self.device)
        B, _, H, W = any_head.shape

        num_original = int(orig2global.numel())
        num_global = self.splitter.num_global_classes

        # Accumulate log-prob over heads at ORIGINAL space
        fused_logp_orig = torch.zeros((B, num_original, H, W), device=self.device)

        for task, head_logits in logits_dict.items():
            lp = F.log_softmax(head_logits.to(self.device), dim=1)  # [B, C_task, H, W]
            idx = self._orig2task_idx[task].to(self.device)         # [num_original]
            sel = lp.gather(1, idx.view(1, -1, 1, 1).expand(B, -1, H, W))  # [B, num_original, H, W]
            fused_logp_orig += sel

        # Marginalize originals -> global in prob space, then back to log
        fused_prob_orig = torch.exp(fused_logp_orig)  # [B, num_original, H, W]
        fused_prob_global = torch.zeros((B, num_global, H, W), device=self.device)

        fused_prob_global.scatter_add_(
            1,  # along class dim
            orig2global.view(1, -1, 1, 1).expand(B, -1, H, W),
            fused_prob_orig
        )

        fused_prob_global = torch.clamp(fused_prob_global, min=1e-12)
        fused_prob_global = fused_prob_global / fused_prob_global.sum(dim=1, keepdim=True)
        fused_global_logits = torch.log(fused_prob_global)

        return fused_global_logits


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
            logits = predictions
            flat_preds = torch.argmax(predictions, dim=1)

        mask = (original_targets != self.ignore_index)

        mapping_flat_to_original = self.splitter.flat_to_original_mapping_torch.to(self.device)
        original_preds = self._safe_lookup(mapping_flat_to_original, flat_preds)

        global_mapping = self.splitter.global_mapping_torch.to(self.device)
        global_target = self._safe_lookup(global_mapping, original_targets)
        global_pred = self._safe_lookup(global_mapping, original_preds)

        # Tier 1 GPU Updates
        self._update_global_biou_stats_gpu(global_pred, global_target)
        self._update_boundary_stats_gpu(global_pred, global_target)

        # Probabilistic stats if logits are available
        if logits is not None:
            self._update_probabilistic_stats_gpu(logits, global_target)

        batch_size = original_targets.shape[0]
        for i in range(batch_size):
            img_id = image_ids[i]
            img_mask = mask[i]
            per_image_cms: Dict[str, np.ndarray] = {}
            per_image_predictions: Dict[str, np.ndarray] = {}

            # Per-task CMs & BIoU
            for task, details in self.splitter.hierarchical_definitions.items():
                mapping = torch.from_numpy(details['ungrouped']['mapping_array']).to(self.device).long()
                target_ungrouped = self._safe_lookup(mapping, original_targets[i])

                # ORIGINAL preds -> this task’s UNGROUPED space
                pred_ungrouped = self._safe_lookup(mapping, original_preds[i])

                n_cls = len(details['ungrouped']['id2label'])
                cm_update = torch.bincount(
                    n_cls * target_ungrouped[img_mask].long() + pred_ungrouped[img_mask].long(),
                    minlength=n_cls ** 2
                ).reshape(n_cls, n_cls)
                self.task_cms[task] += cm_update

                if store_per_image:
                    per_image_cms[task] = cm_update.cpu().numpy()
                    per_image_predictions[task] = pred_ungrouped.cpu().numpy()

                # BIoU (ungrouped)
                self._update_biou_stats_gpu(
                    pred_ungrouped.unsqueeze(0),
                    target_ungrouped.unsqueeze(0),
                    task, 'ungrouped'
                )
                # BIoU (grouped) if applicable
                if details.get('is_grouped'):
                    grouped_map = torch.from_numpy(details['ungrouped_to_grouped_map']).to(self.device).long()
                    grouped_pred = self._safe_lookup(grouped_map, pred_ungrouped)
                    grouped_target = self._safe_lookup(grouped_map, target_ungrouped)
                    self._update_biou_stats_gpu(
                        grouped_pred.unsqueeze(0),
                        grouped_target.unsqueeze(0),
                        task, 'grouped'
                    )

            # Global CM
            n_cls_global = self.splitter.num_global_classes
            global_cm_update = torch.bincount(
                n_cls_global * global_target[i][img_mask].long() + global_pred[i][img_mask].long(),
                minlength=n_cls_global ** 2
            ).reshape(n_cls_global, n_cls_global)
            self.global_cm += global_cm_update

            if store_per_image:
                per_image_cms['global'] = global_cm_update.cpu().numpy()
                per_image_predictions['global'] = global_pred[i].cpu().numpy()

                # Prepare per-image BIoU inputs
                num_classes_per_task = {}
                target_masks_for_biou = {}

                for task, details in self.splitter.hierarchical_definitions.items():
                    num_classes_per_task[task] = len(details['ungrouped']['id2label'])
                    mapping = torch.from_numpy(details['ungrouped']['mapping_array']).to(self.device).long()
                    target_masks_for_biou[task] = self._safe_lookup(mapping, original_targets[i]).cpu().numpy()

                num_classes_per_task['global'] = self.splitter.num_global_classes
                target_masks_for_biou['global'] = global_target[i].cpu().numpy()

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
