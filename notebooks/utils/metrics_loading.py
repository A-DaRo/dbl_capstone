"""Helper utilities to load experiment metrics into structured pandas objects."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from .experiment_registry import ExperimentMetadata


def load_history_frame(metadata: ExperimentMetadata) -> pd.DataFrame:
    """Return the experiment history as a tidy DataFrame."""
    history_path = metadata.artifacts.get("history")
    if not history_path:
        raise FileNotFoundError(f"Missing history.json for experiment {metadata.name}")

    with history_path.open("r", encoding="utf-8") as handle:
        history_dict = json.load(handle)

    frame = pd.DataFrame(history_dict)
    if "epoch" in frame.columns:
        frame = frame.rename(columns={"epoch": "metric_epoch"})
    frame.index.name = "epoch"
    tidy = frame.reset_index().melt(id_vars="epoch", var_name="metric", value_name="value")
    tidy["experiment"] = metadata.name
    return tidy


def load_test_metrics(metadata: ExperimentMetadata) -> Dict:
    """Load the comprehensive test metrics JSON structure."""
    path = metadata.artifacts.get("test_metrics")
    if not path:
        raise FileNotFoundError(f"Missing test_metrics_full_report.json for {metadata.name}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _aggregate_confusion_matrices(fields: Iterable[np.ndarray]) -> np.ndarray:
    iterator = iter(fields)
    total = None
    for matrix in iterator:
        matrix = np.asarray(matrix, dtype=np.int64)
        if total is None:
            total = matrix.copy()
        else:
            total += matrix
    if total is None:
        raise ValueError("No confusion matrices provided for aggregation.")
    return total


def load_confusion_matrices(metadata: ExperimentMetadata) -> Dict[str, np.ndarray]:
    """Load aggregated confusion matrices for the experiment."""
    path = metadata.artifacts.get("test_cms")
    if not path:
        raise FileNotFoundError(f"Missing test_cms.jsonl for {metadata.name}")

    matrices: Dict[str, List[np.ndarray]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            confs = payload.get("confusion_matrices", {})
            for task, matrix in confs.items():
                matrices.setdefault(task, []).append(np.asarray(matrix, dtype=np.int64))
    return {task: _aggregate_confusion_matrices(entries) for task, entries in matrices.items()}


def load_advanced_metrics(metadata: ExperimentMetadata) -> pd.DataFrame:
    """Load advanced per-image metrics into a DataFrame if available."""
    path = metadata.artifacts.get("advanced_metrics")
    if not path:
        raise FileNotFoundError(f"Missing advanced_metrics.jsonl for {metadata.name}")
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def extract_per_class_rows(metrics_dict: Dict, task: str, level: str = "ungrouped") -> pd.DataFrame:
    """Extract per-class metrics for a given task and hierarchy level."""
    tasks = metrics_dict.get("tasks", {})
    task_block = tasks.get(task, {})
    level_block = task_block.get(level, {})
    per_class = level_block.get("per_class", {})
    summary_rows = []
    for class_name, metrics in per_class.items():
        row = {"class_name": class_name}
        row.update(metrics)
        summary_rows.append(row)
    frame = pd.DataFrame(summary_rows)
    return frame


def extract_task_summary(metrics_dict: Dict, task: str, level: str = "ungrouped") -> pd.Series:
    """Return the task summary metrics as a pandas Series."""
    tasks = metrics_dict.get("tasks", {})
    task_block = tasks.get(task, {})
    level_block = task_block.get(level, {})
    summary = level_block.get("task_summary", {})
    return pd.Series(summary)
