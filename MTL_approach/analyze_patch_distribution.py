import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# Imports from your project files to get the class names
from dataset import TASK_DEFINITIONS

def analyze_distribution(args):
    """
    Analyzes the class distribution of a pre-processed patch dataset.
    """
    patch_dir = Path(args.patch_dir)
    mask_dir = patch_dir / "masks"

    if not mask_dir.exists():
        print(f"Error: Mask directory not found at '{mask_dir}'")
        print("Please ensure you provide the correct path to the PDS output directory.")
        return

    mask_files = list(mask_dir.glob("*.npz"))
    if not mask_files:
        print(f"Error: No .npz mask files found in '{mask_dir}'.")
        return

    print(f"Analyzing {len(mask_files)} patch mask files...")

    # Initialize a dictionary to hold pixel counts for each class in each task
    # e.g., {'genus': {0: 1234, 1: 567, ...}, 'health': {...}}
    pixel_counts = {
        task: defaultdict(int) for task in TASK_DEFINITIONS.keys()
    }

    # Iterate through all .npz files and accumulate counts
    for mask_path in tqdm(mask_files, desc="Analyzing masks"):
        with np.load(mask_path) as data:
            for task_name in data.files:
                if task_name in pixel_counts:
                    mask_array = data[task_name]
                    # Get unique class IDs and their counts for this patch
                    class_ids, counts = np.unique(mask_array, return_counts=True)
                    # Add these counts to the running total
                    for class_id, count in zip(class_ids, counts):
                        pixel_counts[task_name][class_id] += count

    print("\n--- Final Class Distribution Report ---\n")

    # Print the results in a formatted table
    for task_name, counts_dict in pixel_counts.items():
        print(f"--- Task: {task_name.upper()} ---")
        id2label = TASK_DEFINITIONS[task_name]['id2label']
        total_pixels = sum(counts_dict.values())

        if total_pixels == 0:
            print("No pixels found for this task.")
            continue
        
        # Sort by class ID for consistent ordering
        sorted_class_ids = sorted(counts_dict.keys())

        print(f"{'ID':<5} | {'Class Name':<25} | {'Pixel Count':<20} | {'Percentage':<15}")
        print("-" * 70)

        for class_id in sorted_class_ids:
            count = counts_dict[class_id]
            label = id2label.get(class_id, "Unknown")
            percentage = (count / total_pixels) * 100
            print(f"{class_id:<5} | {label:<25} | {count:<20,} | {percentage:.3f}%")
        
        print("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze the pixel class distribution of the generated patch dataset."
    )
    parser.add_argument(
        "--patch_dir", type=str, required=True,
        help="Path to the root of the PDS-generated patch directory (e.g., './pds_training_patches')."
    )
    
    args = parser.parse_args()
    analyze_distribution(args)