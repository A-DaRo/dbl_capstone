"""
Metrics storage utilities for coral segmentation tasks.
This module provides:
    A MetricsStorer class to handle persistent storage of epoch-wise history
    and per-image, raw confusion matrices for detailed analysis.
    An AsyncMetricsStorer for background, non-blocking storage operations.
"""


import numpy as np
import json
import os
import threading
import queue
import atexit
from typing import Dict, List, Any, Tuple, Optional



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

        self._history_data: Dict[str, List] = {}
        self._val_cm_file = None
        self._test_cm_file = None

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
    
    def close(self):
        """Closes any open file handles."""
        if self._val_cm_file:
            self._val_cm_file.close()
            self._val_cm_file = None
        if self._test_cm_file:
            self._test_cm_file.close()
            self._test_cm_file = None

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
        """Stores the raw confusion matrices and predicted masks for a single image to a JSONL file."""
        file_handle = self._test_cm_file if is_testing else self._val_cm_file
        if not file_handle:
            raise IOError("File handle is not open. Call `open_for_run()` before storing per-image data.")
        
        serializable_cms = {
            task: cm.tolist() for task, cm in confusion_matrices.items()
        }
        record = {"image_id": image_id, "confusion_matrices": serializable_cms}
        
        # Add predicted masks if provided
        if predicted_masks is not None:
            serializable_predictions = {
                task: pred.tolist() if isinstance(pred, np.ndarray) else pred
                for task, pred in predicted_masks.items()
            }
            record["predicted_masks"] = serializable_predictions
        
        if epoch is not None:
            record["epoch"] = epoch
            
        file_handle.write(json.dumps(record) + '\n')

    def save_final_report(self, metrics_report: Dict[str, Any], filename: str):
        """Saves the final metrics report as a JSON file."""
        report_path = os.path.join(self.output_dir, filename)
        with open(report_path, 'w') as f:
            json.dump(metrics_report, f, indent=2)


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
            
        elif task_type == 'store_per_image':
            is_testing = task['is_testing']
            key = 'test' if is_testing else 'val'
            file_handle = self._active_files.get(key)
            
            if file_handle:
                # Convert numpy arrays to lists in background thread
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
                
                if predicted_masks:
                    serializable_predictions = {
                        task_name: pred.tolist() if isinstance(pred, np.ndarray) else pred
                        for task_name, pred in predicted_masks.items()
                    }
                    record["predicted_masks"] = serializable_predictions
                
                file_handle.write(json.dumps(record) + '\n')
                file_handle.flush()  # Ensure data is written
                
        elif task_type == 'close_files':
            for file_handle in self._active_files.values():
                if file_handle:
                    file_handle.close()
            self._active_files.clear()
            
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
        """Stores per-image confusion matrices asynchronously (non-blocking)."""
        self._start_worker_if_needed()
        
        # Defer numpy array conversion to background thread to avoid blocking
        task = {
            'type': 'store_per_image',
            'image_id': image_id,
            'confusion_matrices': confusion_matrices,  # Keep as numpy arrays
            'predicted_masks': predicted_masks,        # Keep as numpy arrays
            'is_testing': is_testing,
            'epoch': epoch
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
            # Signal shutdown and wait for queue to empty
            self._storage_queue.put(None)  # Shutdown signal
            self._shutdown_event.set()
            
            # Wait for remaining tasks to complete with timeout
            try:
                self._worker_thread.join(timeout=5.0)
            except:
                pass
    
    def wait_for_completion(self):
        """Wait for all pending storage operations to complete."""
        if self._worker_thread and self._worker_thread.is_alive():
            self._storage_queue.join()