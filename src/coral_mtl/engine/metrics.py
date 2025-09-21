"""
Metrics calculation module for coral segmentation tasks.
This module provides:
    A hierarchical metrics calculation system that computes detailed per-class,
    per-task (at grouped and ungrouped levels), and global metrics.
"""

import torch
import numpy as np
from typing import Dict, List, Any, Tuple
from scipy.ndimage import binary_dilation
from abc import ABC, abstractmethod
from coral_mtl.utils.task_splitter import TaskSplitter


EPS = 1e-6


class AbstractCoralMetrics(ABC):
    """
    Abstract base class for calculating coral segmentation metrics.
    """
    def __init__(self,
                 splitter: TaskSplitter,
                 device: torch.device,
                 boundary_thickness: int = 2,
                 ignore_index: int = 255):
        self.splitter = splitter
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
        self.per_image_cms_buffer: List[Tuple[str, Dict[str, np.ndarray]]] = []

    @abstractmethod
    def update(self, predictions: Any, original_targets: torch.Tensor, image_ids: List[str]):
        pass

    def _update_biou_stats(self, pred_np: np.ndarray, target_np: np.ndarray, task_name: str, level: str):
        num_classes = len(self.splitter.hierarchical_definitions[task_name][level]['id2label'])
        for i in range(pred_np.shape[0]):
            for c in range(1, num_classes):
                gt_c, pred_c = (target_np[i] == c), (pred_np[i] == c)
                if not np.any(gt_c): continue
                gt_boundary = binary_dilation(gt_c, iterations=self.boundary_thickness) & ~gt_c
                pred_boundary = binary_dilation(pred_c, iterations=self.boundary_thickness) & ~pred_c
                self.biou_stats[task_name][level]['intersection'] += np.sum(gt_boundary & pred_boundary)
                self.biou_stats[task_name][level]['union'] += np.sum(gt_boundary | pred_boundary)

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
        report['global_summary'] = self._compute_metrics_from_cm(global_cm_np, self.splitter.global_class_names)
        report['optimization_metrics']['global.mIoU'] = report['global_summary']['task_summary']['mIoU']

        return report

class CoralMTLMetrics(AbstractCoralMetrics):
    """Metrics calculator for Multi-Task Learning models."""
    def update(self, predictions: Dict[str, torch.Tensor], original_targets: torch.Tensor, image_ids: List[str]):
        preds = {task: torch.argmax(logits, dim=1) for task, logits in predictions.items()}
        mask = (original_targets != self.ignore_index)
        
        batch_size = original_targets.shape[0]
        global_target = self.splitter.global_mapping_torch.to(self.device)[original_targets]

        for i in range(batch_size):
            img_id = image_ids[i]
            img_mask = mask[i]
            per_image_cms = {}

            # Update per-task CMs
            for task, details in self.splitter.hierarchical_definitions.items():
                mapping = torch.from_numpy(details['ungrouped']['mapping_array']).to(self.device)
                target_ungrouped = mapping[original_targets[i]]
                pred_ungrouped = preds[task][i]
                
                n_cls = len(details['ungrouped']['id2label'])
                cm_update = torch.bincount(
                    n_cls * target_ungrouped[img_mask].long() + pred_ungrouped[img_mask].long(),
                    minlength=n_cls**2).reshape(n_cls, n_cls)
                self.task_cms[task] += cm_update
                per_image_cms[task] = cm_update.cpu().numpy()

            # Update Global CM
            pred_fg = torch.zeros_like(original_targets[i], dtype=torch.long)
            for p in preds.values():
                pred_fg |= (p[i] > 0)
            global_pred = pred_fg * self.splitter.global_mapping_torch.to(self.device)[original_targets[i]]

            n_cls_global = self.splitter.num_global_classes
            global_cm_update = torch.bincount(
                n_cls_global * global_target[i][img_mask].long() + global_pred[img_mask].long(),
                minlength=n_cls_global**2).reshape(n_cls_global, n_cls_global)
            self.global_cm += global_cm_update
            per_image_cms['global'] = global_cm_update.cpu().numpy()
            
            self.per_image_cms_buffer.append((img_id, per_image_cms))

class CoralMetrics(AbstractCoralMetrics):
    """Metrics calculator for baseline (single-head) models."""
    def update(self, predictions: torch.Tensor, original_targets: torch.Tensor, image_ids: List[str]):
        flat_preds = torch.argmax(predictions, dim=1)
        mask = (original_targets != self.ignore_index)
        
        original_preds = self.splitter.flat_to_original_mapping_torch.to(self.device)[flat_preds]
        global_target = self.splitter.global_mapping_torch.to(self.device)[original_targets]
        global_pred = self.splitter.global_mapping_torch.to(self.device)[original_preds]
        
        batch_size = original_targets.shape[0]
        for i in range(batch_size):
            img_id = image_ids[i]
            img_mask = mask[i]
            per_image_cms = {}

            # Update per-task CMs
            for task, details in self.splitter.hierarchical_definitions.items():
                mapping = torch.from_numpy(details['ungrouped']['mapping_array']).to(self.device)
                target_ungrouped = mapping[original_targets[i]]
                pred_ungrouped = mapping[original_preds[i]]
                n_cls = len(details['ungrouped']['id2label'])
                cm_update = torch.bincount(
                    n_cls * target_ungrouped[img_mask].long() + pred_ungrouped[img_mask].long(),
                    minlength=n_cls**2).reshape(n_cls, n_cls)
                self.task_cms[task] += cm_update
                per_image_cms[task] = cm_update.cpu().numpy()

            # Update Global CM
            n_cls_global = self.splitter.num_global_classes
            global_cm_update = torch.bincount(
                n_cls_global * global_target[i][img_mask].long() + global_pred[i][img_mask].long(),
                minlength=n_cls_global**2).reshape(n_cls_global, n_cls_global)
            self.global_cm += global_cm_update
            per_image_cms['global'] = global_cm_update.cpu().numpy()
            
            self.per_image_cms_buffer.append((img_id, per_image_cms))