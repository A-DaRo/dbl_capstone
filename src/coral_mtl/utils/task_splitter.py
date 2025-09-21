"""
Task handling utilities for Coralscapes datasets.

This module contains a class-based system for parsing and preparing task
definitions. It creates hierarchical, per-task structures and a unified,
non-overlapping global label space for comprehensive metrics.
"""

import numpy as np
import torch
from typing import Dict, List, Any, Set, Tuple
from abc import ABC, abstractmethod


class TaskSplitter(ABC):
    """
    Abstract Base Class for parsing and structuring task definitions.
    """

    def __init__(self, task_definitions: Dict[str, Dict[str, Any]]):
        if not isinstance(task_definitions, dict) or not task_definitions:
            raise ValueError("Task definitions must be a non-empty dictionary.")
        
        self.raw_definitions = task_definitions
        self._validate_initial_structure()
        
        # Calculate max_original_id first as it's needed by _parse_tasks
        self.max_original_id = self._find_max_original_id()
        
        # --- Core Parsed Properties ---
        self.hierarchical_definitions: Dict[str, Dict[str, Any]] = self._parse_tasks()

        # --- Global Space Properties ---
        self.global_mapping_array: np.ndarray
        self.global_id2label: Dict[int, str]
        self.global_class_names: List[str]
        self.num_global_classes: int
        (
            self.global_mapping_array,
            self.global_id2label
        ) = self._create_global_space()
        self.global_class_names = list(self.global_id2label.values())
        self.num_global_classes = len(self.global_id2label)
        self.global_mapping_torch = torch.from_numpy(self.global_mapping_array)
        
        print(f"Initialized TaskSplitter with {len(self.hierarchical_definitions)} tasks "
              f"and a unified global space of {self.num_global_classes} classes.")

    def _validate_initial_structure(self):
        """Performs a basic validation of the raw task definitions dictionary."""
        for task_name, task_info in self.raw_definitions.items():
            if not isinstance(task_info, dict):
                raise ValueError(f"Task '{task_name}' info must be a dictionary.")
            if 'id2label' not in task_info:
                raise ValueError(f"Task '{task_name}' missing required 'id2label' field.")
            if 'groupby' in task_info and ('mapping' not in task_info['groupby'] or 'id2label' not in task_info['groupby']):
                raise ValueError(f"Task '{task_name}' has incomplete 'groupby' section.")

    def _parse_tasks(self) -> Dict[str, Dict[str, Any]]:
        """
        Parses the raw task definitions, processing 'groupby' directives to create
        a clean, hierarchical structure for each task.
        """
        parsed = {}
        for task_name, task_info in self.raw_definitions.items():
            ungrouped_id2label = {int(k): v for k, v in task_info['id2label'].items()}
            ungrouped_map = {orig_id: i for i, orig_id in enumerate(sorted(ungrouped_id2label.keys()))}
            
            task_data = {
                'is_grouped': 'groupby' in task_info,
                'ungrouped': {
                    'id2label': {i: ungrouped_id2label[orig_id] for orig_id, i in ungrouped_map.items()},
                    'class_names': [ungrouped_id2label[k] for k in sorted(ungrouped_id2label.keys())],
                    'mapping_array': self._create_mapping_array(ungrouped_map)
                }
            }
            
            if task_data['is_grouped']:
                grouped_id2label = {int(k): v for k, v in task_info['groupby']['id2label'].items()}
                raw_group_map = task_info['groupby']['mapping']
                
                ungrouped_to_grouped_map = np.zeros(len(ungrouped_id2label), dtype=np.int64)
                for new_id, old_ids in raw_group_map.items():
                    ids_to_map = [old_ids] if isinstance(old_ids, int) else old_ids
                    for old_id in ids_to_map:
                        if old_id in ungrouped_map:
                            ungrouped_idx = ungrouped_map[old_id]
                            ungrouped_to_grouped_map[ungrouped_idx] = int(new_id)

                task_data['grouped'] = {
                    'id2label': grouped_id2label,
                    'class_names': list(grouped_id2label.values())
                }
                task_data['ungrouped_to_grouped_map'] = ungrouped_to_grouped_map

            parsed[task_name] = task_data
        return parsed

    def _create_mapping_array(self, mapping_dict: Dict[int, int]) -> np.ndarray:
        """Helper to create a LUT array from a {original_id: new_id} dict."""
        array = np.zeros(self.max_original_id + 1, dtype=np.int64)
        for original_id, new_id in mapping_dict.items():
            if original_id <= self.max_original_id:
                array[original_id] = new_id
        return array

    def _find_max_original_id(self) -> int:
        """Finds the maximum original label ID across all tasks."""
        max_id = 0
        for task_info in self.raw_definitions.values():
            max_id = max(max_id, max(int(k) for k in task_info['id2label'].keys()))
        return max_id

    def _create_global_space(self) -> Tuple[np.ndarray, Dict[int, str]]:
        """
        Creates a unified, non-overlapping label space from all original class IDs.
        """
        all_original_ids: Set[int] = set()
        original_id2label: Dict[int, str] = {}
        for task_info in self.raw_definitions.values():
            for k, v in task_info['id2label'].items():
                k_int = int(k)
                if k_int != 0: # Exclude background/unlabeled from the unique set
                    all_original_ids.add(k_int)
                original_id2label[k_int] = v

        global_id2label = {0: "background"}
        mapping_array = np.zeros(self.max_original_id + 1, dtype=np.int64)
        
        current_global_id = 1
        for original_id in sorted(list(all_original_ids)):
            global_id2label[current_global_id] = original_id2label[original_id]
            mapping_array[original_id] = current_global_id
            current_global_id += 1
            
        return mapping_array, global_id2label


class MTLTaskSplitter(TaskSplitter):
    """
    A TaskSplitter for Multi-Task Learning models. It prepares per-task
    mappings for generating multi-channel ground truth masks.
    """
    pass # No extra methods needed, inherits all functionality


class BaseTaskSplitter(TaskSplitter):
    """
    A TaskSplitter for baseline (single-head) models. It creates a "flattened"
    mapping for training, and an inverse mapping for evaluation.
    """
    def __init__(self, task_definitions: Dict[str, Dict[str, Any]]):
        super().__init__(task_definitions)
        self.flat_mapping_array, self.flat_id2label = self._create_flattened_space()
        self.flat_to_original_mapping_array = self._create_inverse_mapping()
        self.flat_to_original_mapping_torch = torch.from_numpy(self.flat_to_original_mapping_array)
        print(f"BaseTaskSplitter created a flattened training space of {len(self.flat_id2label)} classes.")

    def _create_flattened_space(self) -> Tuple[np.ndarray, Dict[int, str]]:
        """Creates a single mapping array for training the baseline model."""
        # The global space is the perfect flattened space.
        return self.global_mapping_array, self.global_id2label
    
    def _create_inverse_mapping(self) -> np.ndarray:
        """Creates a LUT to map from the flattened space back to original IDs."""
        max_flat_id = len(self.flat_id2label) - 1
        inverse_map = np.zeros(max_flat_id + 1, dtype=np.int64)
        for original_id, flat_id in enumerate(self.flat_mapping_array):
            if flat_id > 0:
                inverse_map[flat_id] = original_id
        return inverse_map