import argparse
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import json
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from typing import Optional, Dict
import math

# This constant defines the maximum number of classes per plot, as requested.
# 20 classes * 2 bars/class = 40 bars total.
MAX_CLASSES_PER_PLOT = 20

def _create_remapping_lut_from_yaml(task_definition_path: Path, num_original_classes: int = 40) -> Optional[np.ndarray]:
    """Creates a numpy lookup table from a YAML task definition file."""
    if not task_definition_path.exists():
        print(f"Warning: Task definition YAML not found at {task_definition_path}. Cannot remap original labels.")
        return None
        
    with open(task_definition_path, 'r') as f:
        task_definitions = yaml.safe_load(f)
    
    remapping_lut = np.zeros(num_original_classes, dtype=np.uint8)
    for task_group, details in task_definitions.items():
        for new_id_str, old_ids in details.get('mapping', {}).items():
            new_id = int(new_id_str)
            for old_id in old_ids:
                if old_id < num_original_classes:
                    remapping_lut[old_id] = new_id
    return remapping_lut

def _calculate_pixel_counts(mask_dir: Path, remapping_lut: Optional[np.ndarray] = None) -> Dict[int, int]:
    """Calculates total pixel counts for all .png masks in a directory, applying a remapping LUT if provided."""
    pixel_counts = defaultdict(int)
    mask_files = sorted(list(mask_dir.glob("**/*.png")))
    if not mask_files:
        raise FileNotFoundError(f"No .png mask files found in {mask_dir}")
        
    for mask_path in tqdm(mask_files, desc=f"Scanning {mask_dir.parent.name} masks"):
        mask_array = np.array(Image.open(mask_path))
        if remapping_lut is not None:
            mask_array = remapping_lut[mask_array]
        
        class_ids, counts = np.unique(mask_array, return_counts=True)
        for class_id, count in zip(class_ids, counts):
            pixel_counts[class_id] += count
    return dict(pixel_counts)

def _generate_plot_chunk(df_chunk: pd.DataFrame, output_path: Path):
    """Generates a single plot image for a chunk of the data."""
    num_classes_in_chunk = len(df_chunk.index)
    x = np.arange(num_classes_in_chunk)
    
    # Setup plot aesthetics
    fig, ax = plt.subplots(figsize=(max(12, num_classes_in_chunk * 0.8), 10))
    fig.suptitle('Dataset Distribution Comparison: Original vs. PDS-Sampled', fontsize=20, y=0.98)

    # Define bar properties for overlap and styling
    bar_width = 0.6
    overlap_offset = 0.15
    
    # Plot PDS bars (solid, pale) first so hatched bars are drawn on top
    ax.bar(x - overlap_offset, df_chunk['PDS'], width=bar_width, 
           label='PDS Sampled', color='lightskyblue', zorder=2)
    
    # Plot Original bars (hatched)
    ax.bar(x + overlap_offset, df_chunk['Original'], width=bar_width, 
           label='Original', hatch='//', facecolor='none', 
           edgecolor='dodgerblue', linewidth=1.5, zorder=3)
    
    # Find the maximum data value to determine the plot's upper bound
    # We need this to ensure annotations don't go off-screen.
    max_data_value = max(df_chunk['Original'].max(), df_chunk['PDS'].max())
    
    # Add % Change annotations above the bars
    for i, (label, row) in enumerate(df_chunk.iterrows()):
        # Position text 20% above the taller bar
        y_pos = max(row['Original'], row['PDS']) * 1.2 
        text = f"{row['%_Change']:+.1f}%"
        # Ensure y_pos is not zero or negative for log scale
        if y_pos > 0:
            ax.text(i, y_pos, text, ha='center', va='bottom', fontsize=9, color='darkgrey', zorder=4)

    # --- Axes and Grid Formatting ---
    ax.set_ylabel('Total Pixel Count', fontsize=14)
    ax.set_yscale('log')
    
    # We set the top limit to be 40% higher than the tallest bar, 
    # giving plenty of space for the y_pos = max * 1.2 annotation.
    # We get the bottom limit from whatever matplotlib decided initially.
    current_bottom_lim, _ = ax.get_ylim()
    ax.set_ylim(bottom=current_bottom_lim, top=max_data_value * 1.4)

    ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    
    ax.set_xticks(x)
    ax.set_xticklabels(df_chunk.index, rotation=45, ha='right', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
    ax.tick_params(axis='x', which='major', pad=5)

    # --- Final Touches ---
    ax.legend(fontsize=12)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for title and labels
    
    plt.savefig(output_path, dpi=300)
    print(f"Comparison plot saved to: {output_path}")
    plt.close()

def compare_and_visualize(
    original_dataset_root: Path,
    pds_patch_dir: Path,
    output_dir: Path,
    task_definition_path: Optional[Path] = None
):
    """
    Compares class distributions between the original and PDS-sampled datasets,
    saves a comparison table, and generates one or more visualization plots.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load PDS dataset info
    id2label_path = pds_patch_dir / "id2label.json"
    if not id2label_path.exists():
        raise FileNotFoundError(f"Metadata 'id2label.json' not found in '{pds_patch_dir}'")
    with open(id2label_path, 'r') as f:
        id2label: Dict[int, str] = {int(k): v for k, v in json.load(f).items()}

    print("Analyzing PDS-sampled dataset...")
    pds_counts = _calculate_pixel_counts(pds_patch_dir / "masks")

    # 2. Load and process Original dataset info
    remapping_lut = None
    if task_definition_path:
        print(f"Remapping original dataset labels using '{task_definition_path.name}'")
        remapping_lut = _create_remapping_lut_from_yaml(task_definition_path)

    print("Analyzing original dataset...")
    original_mask_dir = original_dataset_root / "gtFine" / "val"
    original_counts = _calculate_pixel_counts(original_mask_dir, remapping_lut)

    # 3. Combine data into a Pandas DataFrame for analysis
    df = pd.DataFrame([original_counts, pds_counts], index=['Original', 'PDS']).T.fillna(0).astype(np.int64)
    df['Label'] = df.index.map(id2label)
    # Filter out any classes not present in the final label map
    df = df.dropna(subset=['Label'])
    df.set_index('Label', inplace=True)
    # Exclude the background/unlabeled class from analysis and plotting
    df = df[df.index != id2label.get(0, "unlabeled")]

    df['Original_%'] = (df['Original'] / (df['Original'].sum() + 1e-9)) * 100
    df['PDS_%'] = (df['PDS'] / (df['PDS'].sum() + 1e-9)) * 100
    df['%_Change'] = ((df['PDS_%'] - df['Original_%']) / (df['Original_%'] + 1e-9)) * 100
    
    # Order decreasingly by the PDS dataset's pixel count, as requested
    df.sort_values(by="PDS", ascending=False, inplace=True)

    # 4. Save Markdown comparison table
    report_path = output_dir / "comparison_report.md"
    with open(report_path, 'w') as f:
        f.write("# Dataset Distribution Comparison Report\n\n")
        f.write("This report compares the class distribution of the original dataset against the PDS-sampled patch dataset.\n\n")
        report_df = df[['Original', 'PDS', 'Original_%', 'PDS_%', '%_Change']].copy()
        report_df.rename(columns={'Original': 'Original Pixels', 'PDS': 'PDS Pixels'}, inplace=True)
        f.write(report_df.to_markdown(floatfmt=(".0f", ".0f", ".3f", ".3f", ".2f")))
    print(f"Comparison report saved to: {report_path}")

    # 5. Handle plot splitting
    num_classes = len(df.index)
    num_plots = math.ceil(num_classes / MAX_CLASSES_PER_PLOT)

    if num_plots == 0:
        print("No data to plot.")
        return

    base_plot_path = output_dir / "distribution_comparison.png"
    for i in range(num_plots):
        start_idx = i * MAX_CLASSES_PER_PLOT
        end_idx = start_idx + MAX_CLASSES_PER_PLOT
        df_chunk = df.iloc[start_idx:end_idx]
        
        # Determine output path for the plot chunk
        if num_plots > 1:
            chunk_path = base_plot_path.with_name(f"{base_plot_path.stem}_part{i+1}{base_plot_path.suffix}")
        else:
            chunk_path = base_plot_path
            
        _generate_plot_chunk(df_chunk, chunk_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare class distributions between original and PDS-sampled datasets.")
    parser.add_argument("--original_dataset_root", type=str, required=True, help="Path to the root of the original, non-PDS dataset.")
    parser.add_argument("--pds_patch_dir", type=str, required=True, help="Path to the PDS-generated patch directory.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the report and plot(s).")
    parser.add_argument("--task_definition_path", type=str, default=None, help="Optional: Path to a YAML file to remap original labels for comparison.")
    
    args = parser.parse_args()
    
    task_path = Path(args.task_definition_path) if args.task_definition_path else None
    compare_and_visualize(Path(args.original_dataset_root), Path(args.pds_patch_dir), Path(args.output_dir), task_path)