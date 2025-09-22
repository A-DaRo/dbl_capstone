"""
Metrics storage utilities for coral segmentation tasks.
This module provides:
    A MetricsStorer class to handle persistent storage of epoch-wise history
    and per-image, raw confusion matrices for detailed analysis.
"""


import numpy as np
import json
import os
from typing import Dict, List, Any, Tuple



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
        # Open in write mode to clear previous run's results
        if os.path.exists(path):
            os.remove(path)
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