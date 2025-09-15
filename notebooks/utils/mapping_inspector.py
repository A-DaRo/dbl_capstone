import yaml
import json
from huggingface_hub import hf_hub_download
from pathlib import Path
from typing import Dict, Set

# This utility module provides functions to inspect the label transformation mappings
# defined in 'configs/task_definitions.yaml'. It is intended for use in notebooks
# to aid in data exploration and verification of the Label Transformation Pipeline (Section 2.1).

def _get_all_raw_labels(repo_id: str = "EPFL-ECEO/coralscapes") -> Dict[int, str]:
    """
    (Internal) Downloads and loads the official id-to-label mapping from the Hugging Face Hub.
    """
    print(f"--- Fetching official id2label mapping from '{repo_id}' ---")
    try:
        id2label_path = hf_hub_download(repo_id=repo_id, filename="id2label.json", repo_type="dataset")
        with open(id2label_path, "r") as f:
            id2label_str_keys = json.load(f)
            id2label = {int(k): v for k, v in id2label_str_keys.items()}
        id2label[0] = "unlabeled"
        print("Successfully loaded raw class labels.\n")
        return id2label
    except Exception as e:
        print(f"Error fetching or parsing official labels: {e}")
        return {}

def report_unmapped_classes(task_definitions_path: Path):
    """
    Verifies the completeness of the task mappings and prints a report of any
    raw Coralscapes class IDs that have not been assigned to a task.
    
    Args:
        task_definitions_path (Path): The path to the 'task_definitions.yaml' config file.
    """
    all_raw_labels = _get_all_raw_labels()
    if not all_raw_labels:
        return
    
    all_raw_ids = set(all_raw_labels.keys())
    
    # Load mapped IDs
    if not Path(task_definitions_path).exists():
        print(f"Error: Task definitions file not found at '{task_definitions_path}'")
        return
        
    with open(task_definitions_path, 'r') as f:
        task_definitions = yaml.safe_load(f)

    if not task_definitions:
        print(f"Error: The task definitions file at '{task_definitions_path}' is empty or could not be parsed.")
        return
        
    mapped_ids = set()
    for task_info in task_definitions.values():
        if "mapping" in task_info and task_info["mapping"] is not None:
            for old_ids_list in task_info["mapping"].values():
                mapped_ids.update(old_ids_list)

    unmapped_ids = all_raw_ids - mapped_ids
    
    print("--- Unmapped Classes Verification Report ---")
    if not unmapped_ids or unmapped_ids == {0}:
        print("✅ All raw class IDs (except for 'unlabeled') are successfully mapped.")
    else:
        print("⚠️ The following raw class IDs from the dataset are NOT mapped to any task:")
        for missing_id in sorted([uid for uid in unmapped_ids if uid != 0]):
            label_name = all_raw_labels.get(missing_id, "Unknown Label")
            print(f"  - ID: {missing_id:<3} | Label: {label_name}")

def report_current_mappings(task_definitions_path: Path):
    """
    Prints a human-readable report of all current mappings, grouped by task,
    showing which original classes are combined to form each new class.

    Args:
        task_definitions_path (Path): The path to the 'task_definitions.yaml' config file.
    """
    all_raw_labels = _get_all_raw_labels()
    if not all_raw_labels:
        return

    if not Path(task_definitions_path).exists():
        print(f"Error: Task definitions file not found at '{task_definitions_path}'")
        return

    with open(task_definitions_path, 'r') as f:
        task_definitions = yaml.safe_load(f)

    if not task_definitions:
        print(f"Error: The task definitions file at '{task_definitions_path}' is empty or could not be parsed.")
        return

    print("\n--- Current Mappings Report (Grouped by Task) ---")
    for task_name, task_info in task_definitions.items():
        print(f"\n==============================================")
        print(f" TASK: {task_name.upper()}")
        print(f"==============================================")
        
        if "mapping" not in task_info or not task_info["mapping"]:
            print("No mappings defined for this task.")
            continue

        task_id2label = {int(k): v for k, v in task_info["id2label"].items()}
        
        for new_id, old_ids_list in sorted(task_info["mapping"].items()):
            new_label_name = task_id2label.get(int(new_id), "Unknown")
            print(f"\n  New Class: '{new_label_name}' (ID: {new_id}) is composed of:")
            
            if not old_ids_list:
                print("    - (No raw classes assigned)")
                continue

            for old_id in sorted(old_ids_list):
                original_label = all_raw_labels.get(old_id, "Unknown Raw Label")
                print(f"    - Raw ID: {old_id:<3} | Original Label: '{original_label}'")