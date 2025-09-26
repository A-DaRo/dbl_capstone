"""
Metrics storage utilities for coral segmentation tasks.
This module provides:
    A MetricsStorer class to handle persistent storage of epoch-wise history
    and per-image, raw confusion matrices for detailed analysis.
    An AsyncMetricsStorer for background, non-blocking storage operations.
    An AdvancedMetricsProcessor for the three-tier system with multi-process CPU workers.
"""


import numpy as np
import json
import os
import threading
import queue
import atexit
import multiprocessing
import time
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
from multiprocessing.synchronize import Event as EventType
from multiprocessing.managers import SyncManager


EPS = 1e-6


def compute_per_image_miou(confusion_matrix: np.ndarray) -> float:
    """Compute mIoU from a confusion matrix."""
    tp = np.diag(confusion_matrix)
    fp = confusion_matrix.sum(axis=0) - tp
    fn = confusion_matrix.sum(axis=1) - tp
    iou = tp / (tp + fp + fn + EPS)
    return np.nanmean(iou)


def compute_per_image_biou(pred_mask: np.ndarray, target_mask: np.ndarray, num_classes: int, boundary_thickness: int = 2) -> float:
    """
    Compute Boundary IoU (BIoU) for a single image.
    
    Args:
        pred_mask: Predicted segmentation mask
        target_mask: Ground truth segmentation mask  
        num_classes: Number of classes
        boundary_thickness: Thickness of boundary region in pixels
        
    Returns:
        BIoU score for this image
    """
    try:
        from scipy.ndimage import binary_dilation
    except ImportError:
        # Fallback to simple boundary detection if scipy is not available
        return 0.0
    
    total_intersection = 0.0
    total_union = 0.0
    
    for c in range(1, num_classes):  # Skip background
        gt_c = (target_mask == c)
        pred_c = (pred_mask == c)
        
        # Skip if no ground truth for this class
        if not np.any(gt_c):
            continue
            
        # Create boundary masks using binary dilation
        struct = np.ones((3, 3))
        gt_boundary = binary_dilation(gt_c, structure=struct, iterations=boundary_thickness) & ~gt_c
        pred_boundary = binary_dilation(pred_c, structure=struct, iterations=boundary_thickness) & ~pred_c
        
        intersection = np.sum(gt_boundary & pred_boundary)
        union = np.sum(gt_boundary | pred_boundary)
        
        total_intersection += intersection
        total_union += union
    
    if total_union > 0:
        return total_intersection / total_union
    else:
        return 0.0



class MetricsStorer:
    """
    Handles the persistent storage of metrics and raw data.
    """
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.history_path = os.path.join(self.output_dir, "history.json")
        self.val_cm_path = os.path.join(self.output_dir, "validation_cms.jsonl")
        self.test_cm_path = os.path.join(self.output_dir, "test_cms.jsonl")
        self.loss_diagnostics_path = os.path.join(self.output_dir, "loss_diagnostics.jsonl")

        self._history_data: Dict[str, List] = {}
        self._val_cm_file = None
        self._test_cm_file = None
        self._loss_diagnostics_file = None

    @staticmethod
    def _sanitize_for_json(obj: Any) -> Any:
        """Recursively convert numpy/scalar objects into JSON-serializable types."""

        if isinstance(obj, dict):
            return {key: MetricsStorer._sanitize_for_json(value) for key, value in obj.items()}

        if isinstance(obj, (list, tuple, set)):
            return [MetricsStorer._sanitize_for_json(item) for item in obj]

        if isinstance(obj, np.generic):
            return obj.item()

        if isinstance(obj, np.ndarray):
            return obj.tolist()

        return obj

    def open_for_run(self, is_testing: bool = False):
        """Opens file handles for a validation or testing run."""
        path = self.test_cm_path if is_testing else self.val_cm_path
        # Open in append mode. The file is now a stream; we don't clear it.
        # Clearing, if needed, should be an explicit action outside this class.
        file_handle = open(path, 'a')
        if is_testing:
            self._test_cm_file = file_handle
        else:
            self._val_cm_file = file_handle
            if self._loss_diagnostics_file is None:
                self._loss_diagnostics_file = open(self.loss_diagnostics_path, 'a')
    
    def close(self):
        """Closes any open file handles."""
        if self._val_cm_file:
            self._val_cm_file.close()
            self._val_cm_file = None
        if self._test_cm_file:
            self._test_cm_file.close()
            self._test_cm_file = None
        if self._loss_diagnostics_file:
            self._loss_diagnostics_file.close()
            self._loss_diagnostics_file = None

    def store_loss_diagnostics(self, step: int, epoch: int, diagnostics: Dict[str, Any]) -> None:
        """Persist loss/strategy diagnostics for the given training step."""
        if not self._loss_diagnostics_file:
            return
        record = {
            "step": int(step),
            "epoch": int(epoch),
        }
        record.update(self._sanitize_for_json(diagnostics))
        self._loss_diagnostics_file.write(json.dumps(record) + '\n')
        self._loss_diagnostics_file.flush()

    @staticmethod
    def _flatten_dict(d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
        """Flattens a nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, dict):
                items.extend(MetricsStorer._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def store_epoch_history(self, metrics_report: Dict, epoch: int):
        """Updates and saves the training history with metrics from a new epoch."""
        flat_optim_metrics = self._flatten_dict(metrics_report.get('optimization_metrics', {}))
        
        if not self._history_data: # First epoch
            self._history_data = {key: [] for key in flat_optim_metrics.keys()}
            self._history_data['epoch'] = []
        
        self._history_data['epoch'].append(epoch)
        for key, value in flat_optim_metrics.items():
            self._history_data.setdefault(key, []).append(value)
        
        # Safe write: write to temp file then rename
        temp_path = self.history_path + ".tmp"
        with open(temp_path, 'w') as f:
            json.dump(self._history_data, f, indent=2)
        os.replace(temp_path, self.history_path)

    def store_per_image_cms(self, image_id: str, confusion_matrices: Dict[str, np.ndarray], predicted_masks: Dict[str, np.ndarray] = None, is_testing: bool = False, epoch: int = None):
        """Stores the raw confusion matrices and computed per-image metrics for a single image to a JSONL file."""
        file_handle = self._test_cm_file if is_testing else self._val_cm_file
        if not file_handle:
            raise IOError("File handle is not open. Call `open_for_run()` before storing per-image data.")
        
        serializable_cms = {
            task: cm.tolist() for task, cm in confusion_matrices.items()
        }
        record = {"image_id": image_id, "confusion_matrices": serializable_cms}
        
        # Compute and store per-image metrics instead of full prediction masks
        if predicted_masks is not None:
            per_image_metrics = {}
            for task, pred_mask in predicted_masks.items():
                if isinstance(pred_mask, np.ndarray):
                    # Compute mIoU from confusion matrix
                    cm = confusion_matrices.get(task)
                    if cm is not None:
                        miou = compute_per_image_miou(cm)
                        per_image_metrics[task] = {
                            'mIoU': miou,
                            'mask_shape': pred_mask.shape,
                            'unique_predictions': np.unique(pred_mask).tolist()
                        }
                        
                        # If we have target mask information, compute BIoU as well
                        # Note: This requires access to target mask, which we'll need to pass separately
                        # For now, we'll store the shape and unique values info
                        
            record["per_image_metrics"] = per_image_metrics
        
        if epoch is not None:
            record["epoch"] = epoch
            
        file_handle.write(json.dumps(record) + '\n')

    def store_per_image_cms_with_metrics(self, image_id: str, confusion_matrices: Dict[str, np.ndarray], 
                                       predicted_masks: Dict[str, np.ndarray] = None, 
                                       target_masks: Dict[str, np.ndarray] = None,
                                       num_classes_per_task: Dict[str, int] = None,
                                       is_testing: bool = False, epoch: int = None):
        """
        Stores confusion matrices and computes per-image BIoU and mIoU metrics.
        
        Args:
            image_id: Unique identifier for the image
            confusion_matrices: Dict mapping task names to confusion matrices
            predicted_masks: Dict mapping task names to predicted segmentation masks
            target_masks: Dict mapping task names to ground truth masks (needed for BIoU)
            num_classes_per_task: Dict mapping task names to number of classes
            is_testing: Whether this is for test set (vs validation)
            epoch: Current epoch number (optional)
        """
        file_handle = self._test_cm_file if is_testing else self._val_cm_file
        if not file_handle:
            raise IOError("File handle is not open. Call `open_for_run()` before storing per-image data.")
        
        serializable_cms = {
            task: cm.tolist() for task, cm in confusion_matrices.items()
        }
        record = {"image_id": image_id, "confusion_matrices": serializable_cms}
        
        # Compute and store per-image metrics
        if predicted_masks is not None:
            per_image_metrics = {}
            for task, pred_mask in predicted_masks.items():
                if isinstance(pred_mask, np.ndarray):
                    # Compute mIoU from confusion matrix
                    cm = confusion_matrices.get(task)
                    metrics = {}
                    
                    if cm is not None:
                        metrics['mIoU'] = compute_per_image_miou(cm)
                    
                    # Compute BIoU if we have target masks
                    if target_masks and task in target_masks and num_classes_per_task and task in num_classes_per_task:
                        target_mask = target_masks[task]
                        num_classes = num_classes_per_task[task]
                        metrics['BIoU'] = compute_per_image_biou(pred_mask, target_mask, num_classes)
                    
                    # Store additional info
                    metrics.update({
                        'mask_shape': pred_mask.shape,
                        'unique_predictions': np.unique(pred_mask).tolist(),
                        'num_pixels': pred_mask.size
                    })
                    
                    per_image_metrics[task] = metrics
                        
            record["per_image_metrics"] = per_image_metrics
        
        if epoch is not None:
            record["epoch"] = epoch
            
        file_handle.write(json.dumps(record) + '\n')

    def save_final_report(self, metrics_report: Dict[str, Any], filename: str):
        """Saves the final metrics report as a JSON file."""
        report_path = os.path.join(self.output_dir, filename)
        sanitized_report = self._sanitize_for_json(metrics_report)
        with open(report_path, 'w') as f:
            json.dump(sanitized_report, f, indent=2)

    # ---------------- New API: Test Loss Report Persistence -----------------
    def save_test_loss_report(self, loss_metrics: Dict[str, float]) -> None:
        """Persist aggregated test-phase loss metrics.

        Args:
            loss_metrics: Mapping of namespaced loss component -> scalar float (already averaged).

        Writes atomic JSON file 'test_loss_metrics.json' under output_dir.
        """
        # Sanitize and convert tensors / numpy types if present
        sanitized = self._sanitize_for_json(loss_metrics)
        path = os.path.join(self.output_dir, 'test_loss_metrics.json')
        tmp = path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(sanitized, f, indent=2)
        os.replace(tmp, path)


class AsyncMetricsStorer:
    """
    Asynchronous metrics storage that performs I/O operations in a background thread.
    This prevents blocking during validation loops while maintaining per-image storage capability.
    """
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Storage queue and worker thread
        self._storage_queue = queue.Queue()
        self._worker_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()
        self._active_files: Dict[str, Any] = {}
        self._loss_diagnostics_path = os.path.join(self.output_dir, "loss_diagnostics.jsonl")
        
        # Register cleanup on exit
        atexit.register(self.shutdown)
    
    def _start_worker_if_needed(self):
        """Start the background worker thread if not already running."""
        if self._worker_thread is None or not self._worker_thread.is_alive():
            self._shutdown_event.clear()
            self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self._worker_thread.start()
    
    def _worker_loop(self):
        """Background worker thread that processes storage operations."""
        while not self._shutdown_event.is_set():
            try:
                # Wait for storage tasks with timeout to check shutdown periodically
                task = self._storage_queue.get(timeout=1.0)
                if task is None:  # Shutdown signal
                    self._storage_queue.task_done()
                    break
                
                # Process the storage task
                self._process_storage_task(task)
                self._storage_queue.task_done()
                
            except queue.Empty:
                continue  # Check shutdown and continue
            except Exception as e:
                print(f"AsyncMetricsStorer worker error: {e}")
                # Continue processing other tasks
    
    def _process_storage_task(self, task: Dict[str, Any]):
        """Process a single storage task."""
        task_type = task['type']
        
        if task_type == 'open_file':
            path = task['path']
            is_testing = task['is_testing']
            file_handle = open(path, 'a')
            key = 'test' if is_testing else 'val'
            self._active_files[key] = file_handle
            if not is_testing and 'loss' not in self._active_files:
                self._active_files['loss'] = open(self._loss_diagnostics_path, 'a')
            
        elif task_type == 'store_per_image':
            is_testing = task['is_testing']
            key = 'test' if is_testing else 'val'
            file_handle = self._active_files.get(key)
            
            if file_handle:
                # Convert numpy arrays to lists and compute per-image metrics in background thread
                confusion_matrices = task['confusion_matrices']
                predicted_masks = task.get('predicted_masks')
                
                serializable_cms = {
                    task_name: cm.tolist() if isinstance(cm, np.ndarray) else cm
                    for task_name, cm in confusion_matrices.items()
                }
                
                record = {
                    "image_id": task['image_id'],
                    "confusion_matrices": serializable_cms,
                    "epoch": task.get('epoch')
                }
                
                # Compute per-image metrics instead of storing full masks
                if predicted_masks:
                    per_image_metrics = {}
                    for task_name, pred_mask in predicted_masks.items():
                        if isinstance(pred_mask, np.ndarray):
                            cm = confusion_matrices.get(task_name)
                            metrics = {}
                            
                            if cm is not None:
                                metrics['mIoU'] = compute_per_image_miou(cm)
                            
                            # Store additional lightweight info
                            metrics.update({
                                'mask_shape': pred_mask.shape,
                                'unique_predictions': np.unique(pred_mask).tolist(),
                                'num_pixels': pred_mask.size
                            })
                            
                            per_image_metrics[task_name] = metrics
                    
                    record["per_image_metrics"] = per_image_metrics
                
                file_handle.write(json.dumps(record) + '\n')
                file_handle.flush()  # Ensure data is written
        
        elif task_type == 'store_per_image_with_metrics':
            is_testing = task['is_testing']
            key = 'test' if is_testing else 'val'
            file_handle = self._active_files.get(key)
            
            if file_handle:
                confusion_matrices = task['confusion_matrices']
                predicted_masks = task.get('predicted_masks')
                target_masks = task.get('target_masks')
                num_classes_per_task = task.get('num_classes_per_task')
                
                serializable_cms = {
                    task_name: cm.tolist() if isinstance(cm, np.ndarray) else cm
                    for task_name, cm in confusion_matrices.items()
                }
                
                record = {
                    "image_id": task['image_id'],
                    "confusion_matrices": serializable_cms,
                    "epoch": task.get('epoch')
                }
                
                # Compute comprehensive per-image metrics
                if predicted_masks:
                    per_image_metrics = {}
                    for task_name, pred_mask in predicted_masks.items():
                        if isinstance(pred_mask, np.ndarray):
                            cm = confusion_matrices.get(task_name)
                            metrics = {}
                            
                            if cm is not None:
                                metrics['mIoU'] = compute_per_image_miou(cm)
                            
                            # Compute BIoU if we have target masks
                            if (target_masks and task_name in target_masks and 
                                num_classes_per_task and task_name in num_classes_per_task):
                                target_mask = target_masks[task_name]
                                num_classes = num_classes_per_task[task_name]
                                metrics['BIoU'] = compute_per_image_biou(pred_mask, target_mask, num_classes)
                            
                            # Store additional info
                            metrics.update({
                                'mask_shape': pred_mask.shape,
                                'unique_predictions': np.unique(pred_mask).tolist(),
                                'num_pixels': pred_mask.size
                            })
                            
                            per_image_metrics[task_name] = metrics
                    
                    record["per_image_metrics"] = per_image_metrics
                
                file_handle.write(json.dumps(record) + '\n')
                file_handle.flush()
                
        elif task_type == 'close_files':
            for file_handle in self._active_files.values():
                if file_handle:
                    file_handle.close()
            self._active_files.clear()

        elif task_type == 'store_loss_diagnostics':
            file_handle = self._active_files.get('loss')
            if file_handle:
                diagnostics = MetricsStorer._sanitize_for_json(task['diagnostics'])
                record = {
                    'step': int(task['step']),
                    'epoch': int(task['epoch'])
                }
                record.update(diagnostics)
                file_handle.write(json.dumps(record) + '\n')
                file_handle.flush()
            
        elif task_type == 'save_report':
            report_path = os.path.join(self.output_dir, task['filename'])
            temp_path = report_path + ".tmp"
            with open(temp_path, 'w') as f:
                json.dump(task['metrics_report'], f, indent=2)
            os.replace(temp_path, report_path)
    
    def open_for_run(self, is_testing: bool = False):
        """Opens file handles for a validation or testing run asynchronously."""
        self._start_worker_if_needed()
        
        path = os.path.join(self.output_dir, "test_cms.jsonl" if is_testing else "validation_cms.jsonl")
        task = {
            'type': 'open_file',
            'path': path,
            'is_testing': is_testing
        }
        self._storage_queue.put(task)
    
    def store_per_image_cms(self, image_id: str, confusion_matrices: Dict[str, np.ndarray], 
                           predicted_masks: Dict[str, np.ndarray] = None, 
                           is_testing: bool = False, epoch: int = None):
        """Stores per-image confusion matrices asynchronously with computed metrics (non-blocking)."""
        self._start_worker_if_needed()
        
        # Defer computation to background thread to avoid blocking
        task = {
            'type': 'store_per_image',
            'image_id': image_id,
            'confusion_matrices': confusion_matrices,  # Keep as numpy arrays
            'predicted_masks': predicted_masks,        # Keep as numpy arrays
            'is_testing': is_testing,
            'epoch': epoch
        }
        self._storage_queue.put(task)

    def store_per_image_cms_with_metrics(self, image_id: str, confusion_matrices: Dict[str, np.ndarray], 
                                       predicted_masks: Dict[str, np.ndarray] = None, 
                                       target_masks: Dict[str, np.ndarray] = None,
                                       num_classes_per_task: Dict[str, int] = None,
                                       is_testing: bool = False, epoch: int = None):
        """
        Stores confusion matrices and computes per-image BIoU and mIoU metrics asynchronously.
        
        Args:
            image_id: Unique identifier for the image
            confusion_matrices: Dict mapping task names to confusion matrices
            predicted_masks: Dict mapping task names to predicted segmentation masks
            target_masks: Dict mapping task names to ground truth masks (needed for BIoU)
            num_classes_per_task: Dict mapping task names to number of classes
            is_testing: Whether this is for test set (vs validation)
            epoch: Current epoch number (optional)
        """
        self._start_worker_if_needed()
        
        task = {
            'type': 'store_per_image_with_metrics',
            'image_id': image_id,
            'confusion_matrices': confusion_matrices,
            'predicted_masks': predicted_masks,
            'target_masks': target_masks,
            'num_classes_per_task': num_classes_per_task,
            'is_testing': is_testing,
            'epoch': epoch
        }
        self._storage_queue.put(task)

    def store_loss_diagnostics(self, step: int, epoch: int, diagnostics: Dict[str, Any]) -> None:
        """Queues loss diagnostics for asynchronous persistence."""
        self._start_worker_if_needed()
        task = {
            'type': 'store_loss_diagnostics',
            'step': step,
            'epoch': epoch,
            'diagnostics': diagnostics,
        }
        self._storage_queue.put(task)
    
    def save_final_report(self, metrics_report: Dict[str, Any], filename: str):
        """Saves the final metrics report asynchronously."""
        self._start_worker_if_needed()
        
        task = {
            'type': 'save_report',
            'metrics_report': metrics_report,
            'filename': filename
        }
        self._storage_queue.put(task)
    
    def close(self):
        """Close file handles asynchronously."""
        if self._worker_thread and self._worker_thread.is_alive():
            task = {'type': 'close_files'}
            self._storage_queue.put(task)
    
    def shutdown(self):
        """Shutdown the async storer and wait for all tasks to complete."""
        if self._worker_thread and self._worker_thread.is_alive():
            if self._active_files:
                self._storage_queue.put({'type': 'close_files'})
            self._storage_queue.put(None)
            self._storage_queue.join()
            self._shutdown_event.set()

            try:
                self._worker_thread.join(timeout=5.0)
            finally:
                self._worker_thread = None
            self._active_files.clear()
    
    def wait_for_completion(self):
        """Wait for all pending storage operations to complete."""
        if self._worker_thread and self._worker_thread.is_alive():
            self._storage_queue.join()


def run_metric_gauntlet(job_queue: multiprocessing.JoinableQueue,
                       results_queue: multiprocessing.Queue,
                       enabled_tasks: List[str]):
    """Worker process entry point for computing advanced metrics."""

    enabled = set(enabled_tasks)

    while True:
        try:
            job = job_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        if job is None:
            job_queue.task_done()
            break

        image_id, pred_mask, target_mask = job
        pred_mask = np.asarray(pred_mask, dtype=np.uint8)
        target_mask = np.asarray(target_mask, dtype=np.uint8)

        metrics_result = {
            "image_id": image_id,
            "timestamp": time.time()
        }

        try:
            if {"ASSD", "HD95"} & enabled:
                distances = compute_surface_distances(pred_mask, target_mask)
                if "ASSD" in enabled:
                    metrics_result["ASSD"] = distances["mean"]
                if "HD95" in enabled:
                    metrics_result["HD95"] = distances["hd95"]

            if {"ARI", "VI"} & enabled:
                clustering = compute_clustering_metrics(pred_mask, target_mask)
                if "ARI" in enabled:
                    metrics_result["ARI"] = clustering.get("ARI", 0.0)
                if "VI" in enabled:
                    metrics_result["VI"] = clustering.get("VI", 0.0)

            if "PanopticQuality" in enabled:
                metrics_result["PanopticQuality"] = compute_panoptic_quality(pred_mask, target_mask)

        except Exception as exc:
            # Defensive: never let a worker crash; missing metrics default to zero.
            for task in enabled:
                metrics_result.setdefault(task, 0.0)

        results_queue.put(metrics_result)
        job_queue.task_done()

def compute_surface_distances(pred_mask: np.ndarray, target_mask: np.ndarray) -> Dict[str, float]:
    """Compute Average Symmetric Surface Distance and 95th percentile Hausdorff Distance."""
    try:
        from scipy.ndimage import distance_transform_edt
        
        # Convert to binary masks for each class
        distances = []
        
        for class_id in np.unique(target_mask):
            if class_id == 0:  # Skip background
                continue
                
            pred_binary = (pred_mask == class_id)
            target_binary = (target_mask == class_id)
            
            if not np.any(pred_binary) or not np.any(target_binary):
                continue
                
            # Compute distance transforms
            pred_dist = distance_transform_edt(~pred_binary)
            target_dist = distance_transform_edt(~target_binary)
            
            # Surface distances
            pred_surface_dist = pred_dist[target_binary]
            target_surface_dist = target_dist[pred_binary]
            
            if len(pred_surface_dist) > 0 and len(target_surface_dist) > 0:
                all_distances = np.concatenate([pred_surface_dist, target_surface_dist])
                distances.append(all_distances)
        
        if distances:
            all_surf_distances = np.concatenate(distances)
            return {
                'mean': float(np.mean(all_surf_distances)),
                'hd95': float(np.percentile(all_surf_distances, 95))
            }
        else:
            return {'mean': 0.0, 'hd95': 0.0}
            
    except Exception:
        return {'mean': 0.0, 'hd95': 0.0}

def compute_clustering_metrics(pred_mask: np.ndarray, target_mask: np.ndarray) -> Dict[str, float]:
    """Compute Adjusted Rand Index and Variation of Information."""
    try:
        from sklearn.metrics import adjusted_rand_score
        
        # Flatten masks
        pred_flat = pred_mask.flatten()
        target_flat = target_mask.flatten()
        
        # Compute ARI
        ari = adjusted_rand_score(target_flat, pred_flat)
        
        # Compute VI (simplified version)
        # For a complete VI implementation, you'd need mutual_info_score
        vi = 1.0 - ari  # Simplified approximation
        
        return {
            'ARI': float(ari),
            'VI': float(vi)
        }
        
    except Exception:
        return {'ARI': 0.0, 'VI': 0.0}

def compute_panoptic_quality(pred_mask: np.ndarray, target_mask: np.ndarray) -> Dict[str, Any]:
    """Compute a lightweight panoptic quality summary."""
    try:
        thing_classes = np.unique(pred_mask)[1:]
        tp = fp = fn = 0
        iou_sum = 0.0

        for class_id in thing_classes:
            pred_class = pred_mask == class_id
            tgt_class = target_mask == class_id

            if np.any(pred_class) and np.any(tgt_class):
                intersection = float(np.sum(pred_class & tgt_class))
                union = float(np.sum(pred_class | tgt_class))
                iou = intersection / union if union > 0 else 0.0
                if iou > 0.5:
                    tp += 1
                    iou_sum += iou
                else:
                    fp += 1
                    fn += 1
            elif np.any(pred_class):
                fp += 1
            elif np.any(tgt_class):
                fn += 1

        denom = tp + 0.5 * fp + 0.5 * fn
        pq_value = iou_sum / denom if denom > 0 else 0.0

        return {
            "pq": pq_value,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "iou_sum": iou_sum
        }
    except Exception:
        return {"pq": 0.0, "tp": 0, "fp": 0, "fn": 0, "iou_sum": 0.0}
    


class AdvancedMetricsProcessor:
    """
    Three-tier system processor for advanced metrics computation.
    Manages job queues, CPU worker pool, and dedicated I/O writer process.
    """
    
    def __init__(self, output_dir: str, num_cpu_workers: int = 30,
                 enabled_tasks: List[str] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.num_cpu_workers = max(1, min(int(num_cpu_workers), 8))
        self.enabled_tasks = enabled_tasks or ["ASSD", "HD95", "PanopticQuality", "ARI"]
        self._enabled_task_set = list(dict.fromkeys(self.enabled_tasks))

        self.job_queue: Optional[multiprocessing.JoinableQueue] = None
        self.results_queue: Optional[multiprocessing.Queue] = None
        self.worker_pool: List[multiprocessing.Process] = []
        self.io_writer_process: Optional[multiprocessing.Process] = None
        self.shutdown_event: Optional[EventType] = None

        self._active = False

    def start(self):
        """Initialize queues, workers, and the writer process."""
        if self._active:
            return

        print(f"Starting AdvancedMetricsProcessor with {self.num_cpu_workers} workers")

        self.job_queue = multiprocessing.JoinableQueue()
        self.results_queue = multiprocessing.Queue()
        self.shutdown_event = multiprocessing.Event()

        self.worker_pool = []
        for idx in range(self.num_cpu_workers):
            worker = multiprocessing.Process(
                target=run_metric_gauntlet,
                args=(self.job_queue, self.results_queue, self._enabled_task_set),
                name=f"MetricsWorker-{idx}"
            )
            worker.daemon = False
            worker.start()
            self.worker_pool.append(worker)

        self.io_writer_process = multiprocessing.Process(
            target=self._io_writer_loop,
            args=(self.results_queue, str(self.output_dir), self._enabled_task_set, self.shutdown_event),
            name="MetricsIOWriter"
        )
        self.io_writer_process.daemon = False
        self.io_writer_process.start()

        self._active = True
        print("AdvancedMetricsProcessor started successfully")
    
    def dispatch_image_job(self, image_id: str, pred_mask_tensor: np.ndarray, 
                          target_mask_tensor: np.ndarray):
        """
        Dispatch a single image job to the worker pool.
        Non-blocking operation.
        """
        if not self._active or self.job_queue is None:
            raise RuntimeError("AdvancedMetricsProcessor not started")

        if hasattr(pred_mask_tensor, "cpu"):
            pred_mask_np = pred_mask_tensor.cpu().numpy().astype(np.uint8)
        else:
            pred_mask_np = np.asarray(pred_mask_tensor, dtype=np.uint8)

        if hasattr(target_mask_tensor, "cpu"):
            target_mask_np = target_mask_tensor.cpu().numpy().astype(np.uint8)
        else:
            target_mask_np = np.asarray(target_mask_tensor, dtype=np.uint8)

        job = (image_id, pred_mask_np, target_mask_np)

        try:
            self.job_queue.put(job, block=True, timeout=5.0)
        except queue.Full:
            # As a last resort, block until space is available.
            self.job_queue.put(job, block=True)
    
    def shutdown(self):
        """Gracefully shutdown all workers and I/O process."""
        if not self._active:
            return
            
        print("Shutting down AdvancedMetricsProcessor...")

        # Drain all pending jobs before closing workers.
        if self.job_queue is not None:
            self.job_queue.join()
            for _ in self.worker_pool:
                self.job_queue.put(None)

        if self.results_queue is not None:
            self.results_queue.put(None)

        if self.shutdown_event:
            self.shutdown_event.set()

        for worker in self.worker_pool:
            worker.join(timeout=5.0)
            if worker.is_alive():
                worker.terminate()
                worker.join()

        if self.io_writer_process:
            self.io_writer_process.join(timeout=5.0)
            if self.io_writer_process.is_alive():
                self.io_writer_process.terminate()
                self.io_writer_process.join()

        if self.job_queue is not None:
            self.job_queue.close()
            self.job_queue = None

        if self.results_queue is not None:
            self.results_queue.close()
            self.results_queue = None

        self.worker_pool = []
        self.io_writer_process = None
        self.shutdown_event = None
        self._active = False
        print("AdvancedMetricsProcessor shutdown complete")
    
    @staticmethod
    def _io_writer_loop(results_queue: multiprocessing.Queue,
                       output_dir: str,
                       enabled_tasks: List[str],
                       shutdown_event: EventType):
        """Dedicated process that serialises metric results to disk."""

        output_file = Path(output_dir) / "advanced_metrics.jsonl"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        enabled = set(enabled_tasks)

        try:
            with open(output_file, "w", encoding="utf-8") as handle:
                while True:
                    try:
                        result = results_queue.get(timeout=0.5)
                    except queue.Empty:
                        if shutdown_event.is_set():
                            continue
                        continue

                    if result is None:
                        break

                    payload = {
                        "image_id": result.get("image_id"),
                        "timestamp": result.get("timestamp")
                    }

                    for task in enabled:
                        if task in result:
                            payload[task] = result[task]

                    handle.write(json.dumps(payload) + "\n")
                    handle.flush()
        finally:
            shutdown_event.set()
            print("I/O Writer process finished")