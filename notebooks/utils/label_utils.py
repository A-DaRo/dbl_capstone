"""Utilities for retrieving task-specific label metadata from the task splitter."""

from __future__ import annotations

from typing import Dict, List

from coral_mtl.utils.task_splitter import TaskSplitter


def get_grouped_label_names(splitter: TaskSplitter, task_name: str) -> List[str]:
    """Return grouped label names for a task, preserving index order."""
    task_def = splitter.hierarchical_definitions[task_name]
    if not task_def["is_grouped"]:
        return task_def["ungrouped"]["class_names"]
    grouped = task_def["grouped"]["id2label"]
    max_idx = max(grouped.keys()) if grouped else -1
    return [grouped.get(i, f"class_{i}") for i in range(max_idx + 1)]


def get_ungrouped_label_names(splitter: TaskSplitter, task_name: str) -> List[str]:
    """Return ungrouped label names for a task."""
    task_def = splitter.hierarchical_definitions[task_name]
    mapping = task_def["ungrouped"]["id2label"]
    max_idx = max(mapping.keys()) if mapping else -1
    return [mapping.get(i, f"class_{i}") for i in range(max_idx + 1)]
