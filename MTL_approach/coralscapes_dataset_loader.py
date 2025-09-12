import numpy as np
from datasets import load_dataset
from PIL import Image
import time

# --- 1. Definition of Multi-Task Learning (MTL) Mappings ---
# Based on the analysis of functional relevance, semantic cohesion, and data-driven insights,
# we define the mappings from the 39 original classes to our 2 primary and 3 auxiliary tasks.

# The background class is 0 for all tasks. Other classes start from 1.

# A dictionary to hold all task definitions for clarity and scalability.
# Each task has a 'mapping' (new_id -> [old_ids]) and an 'id2label' map.
TASK_DEFINITIONS = {
    "genus": {
        "id2label": {
            0: "background", 1: "other_coral", 2: "massive_meandering", 3: "branching",
            4: "acropora", 5: "table_acropora", 6: "pocillopora", 7: "meandering", 8: "stylophora",
        },
        "mapping": {
            1: [6],             # other_coral
            2: [16, 17, 23],    # massive_meandering
            3: [19, 20, 22],    # branching
            4: [25],            # acropora
            5: [28, 32],        # table_acropora
            6: [31],            # pocillopora
            7: [33, 36, 37],    # meandering
            8: [34],            # stylophora
        },
    },
    "health": {
        "id2label": {0: "background", 1: "alive", 2: "bleached", 3: "dead"},
        "mapping": {
            1: [6, 17, 22, 25, 28, 31, 34, 36],  # alive
            2: [16, 19, 33],                   # bleached
            3: [20, 23, 32, 37],                   # dead
        },
    },
    "fish": {
        "id2label": {0: "background", 1: "fish"},
        "mapping": {
            1: [9], # fish
        },
    },
    "human_artifacts": {
        "id2label": {0: "background", 1: "artifact"},
        "mapping": {
            1: [7, 8, 15], # human, transect tools, transect line
        },
    },
    "substrate": {
        "id2label": {0: "background", 1: "sand", 2: "rock_rubble", 3: "algae_covered"},
        "mapping": {
            1: [5],         # sand
            2: [12, 18],    # unknown hard substrate, rubble
            3: [10],        # algae covered substrate
        },
    },
}

# --- 2. Helper Function to Create Efficient Lookup Tables ---

def create_lookup_table(task_mapping, num_raw_classes=40):
    """
    Creates a NumPy array to efficiently map raw class IDs to new task-specific IDs.

    Args:
        task_mapping (dict): A dictionary mapping new class IDs to a list of raw class IDs.
        num_raw_classes (int): The total number of raw classes (default is 40 for 0-39).

    Returns:
        np.ndarray: A 1D array where the index is the raw class ID and the value
                    is the new task-specific class ID.
    """
    # Initialize a table with zeros (background class)
    lookup_table = np.zeros(num_raw_classes, dtype=np.int64)
    # Populate the table based on the mapping
    for new_id, old_ids_list in task_mapping.items():
        lookup_table[old_ids_list] = new_id
    return lookup_table

# --- 3. Main Demonstration ---

if __name__ == "__main__":
    # Load the dataset from the Hugging Face Hub
    # This creates a DatasetDict object containing all data splits.
    print("--- Loading CoralScapes Dataset ---")
    dataset = load_dataset("EPFL-ECEO/coralscapes")

    # Access the training split
    train_dataset = dataset["train"]
    print(f"Loaded train split with {len(train_dataset)} examples.")

    # Get the first example for demonstration
    print("\n--- Processing First Example from Training Set ---")
    first_example = train_dataset[0]
    original_image = first_example["image"]
    original_label_mask = np.array(first_example["label"])

    print(f"Original Image Size: {original_image.size}, Mode: {original_image.mode}")
    print(f"Original Label Mask Shape: {original_label_mask.shape}")
    print(f"Unique Raw IDs in Mask: {np.unique(original_label_mask)}")

    # Create lookup tables for all defined tasks
    print("\n--- Creating Lookup Tables for All MTL Tasks ---")
    lookup_tables = {
        task_name: create_lookup_table(task_info["mapping"])
        for task_name, task_info in TASK_DEFINITIONS.items()
    }
    print(f"Successfully created {len(lookup_tables)} lookup tables.")

    # Apply the transformations to generate the new masks
    print("\n--- Applying Mappings to Generate New Task-Specific Masks ---")
    start_time = time.time()

    # The vectorized mapping is extremely fast.
    # new_mask = lookup_table[original_mask]
    transformed_masks = {
        task_name: table[original_label_mask]
        for task_name, table in lookup_tables.items()
    }
    end_time = time.time()
    print(f"Transformation took: {end_time - start_time:.6f} seconds.")


    # --- 4. Verification and Output ---
    print("\n--- Verification of Transformed Masks ---")
    for task_name, new_mask in transformed_masks.items():
        task_id2label = TASK_DEFINITIONS[task_name]["id2label"]
        unique_ids = np.unique(new_mask)

        print(f"\n--- Task: {task_name.upper()} ---")
        print(f"Mask Shape: {new_mask.shape}, Dtype: {new_mask.dtype}")
        print("Unique IDs Found:", unique_ids)
        print("Decoded Labels Present:")
        for label_id in unique_ids:
            print(f"  ID {label_id}: {task_id2label.get(label_id, 'N/A')}")