import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import json
from PIL import Image
from typing import Dict

def analyze_distribution(patch_dir: Path, output_dir: Path):
    """
    Analyzes the class distribution of a pre-processed patch dataset and saves a
    Markdown report to the specified output directory.

    Args:
        patch_dir (Path): Path to the root of the PDS-generated patch directory.
        output_dir (Path): Directory where the output report 'distribution_report.md' will be saved.
    """
    mask_dir = patch_dir / "masks"
    id2label_path = patch_dir / "id2label.json"

    # --- Validation ---
    if not mask_dir.exists():
        print(f"Error: Mask directory not found at '{mask_dir}'")
        return
    if not id2label_path.exists():
        print(f"Error: Metadata file 'id2label.json' not found in '{patch_dir}'.")
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load Metadata ---
    print(f"Loading class definitions from: {id2label_path}")
    with open(id2label_path, 'r') as f:
        id2label: Dict[int, str] = {int(k): v for k, v in json.load(f).items()}

    mask_files = sorted(list(mask_dir.glob("*.png")))
    if not mask_files:
        print(f"Error: No .png mask files found in '{mask_dir}'.")
        return

    # --- Pixel Counting ---
    print(f"Analyzing {len(mask_files)} patch mask files...")
    pixel_counts = defaultdict(int)
    for mask_path in tqdm(mask_files, desc="Analyzing masks"):
        mask_array = np.array(Image.open(mask_path))
        class_ids, counts = np.unique(mask_array, return_counts=True)
        for class_id, count in zip(class_ids, counts):
            pixel_counts[class_id] += count

    # --- Report Generation ---
    report_lines = ["# Patch Dataset Class Distribution Report", ""]
    total_pixels = sum(pixel_counts.values())
    
    if total_pixels == 0:
        report_lines.append("No pixels found in any masks.")
    else:
        # Create Markdown Table
        report_lines.append("| ID  | Class Name                   | Pixel Count          | Percentage |")
        report_lines.append("|:----|:-----------------------------|:---------------------|:-----------|")
        sorted_class_ids = sorted(pixel_counts.keys())
        for class_id in sorted_class_ids:
            count = pixel_counts[class_id]
            label = id2label.get(class_id, "Unknown ID")
            percentage = (count / total_pixels) * 100
            report_lines.append(f"| {class_id:<3} | {label:<28} | {count:<20,} | {percentage:.3f}%   |")

    # --- Save Report ---
    report_content = "\n".join(report_lines)
    report_path = output_dir / "distribution_report.md"
    with open(report_path, "w") as f:
        f.write(report_content)
    
    print(f"\n--- Analysis Complete ---")
    print(f"Report saved to: {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze the pixel class distribution of a patch dataset.")
    parser.add_argument("--patch_dir", type=str, required=True, help="Path to the PDS-generated patch directory.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the Markdown report.")
    
    args = parser.parse_args()
    analyze_distribution(Path(args.patch_dir), Path(args.output_dir))