"""
A simple launcher script for creating PDS patches and generating reports.

This script provides an easy way to launch the PDS patch and report generation
pipeline with predefined parameters. For customization, modify the parameters in
pds_config.py or provide command-line arguments.
"""

import os
import sys
import argparse
import traceback
from pathlib import Path

# Ensure the project root is in the Python path
try:
    from pds_launcher.pds_config import (
        DATASET_ROOT, PDS_OUTPUT_DIR, ANALYSIS_OUTPUT_DIR,
        PATCH_SIZE, PDS_RADIUS, NUM_WORKERS
    )
    from coral_mtl.scripts.create_pds_patches_and_report import create_pds_patches_and_report
except ImportError as e:
    # If imports fail, it might be because the project root is not in the path.
    # Let's add it and try again.
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    try:
        from pds_launcher.pds_config import (
            DATASET_ROOT, PDS_OUTPUT_DIR, ANALYSIS_OUTPUT_DIR,
            PATCH_SIZE, PDS_RADIUS, NUM_WORKERS
        )
        from coral_mtl.scripts.create_pds_patches_and_report import create_pds_patches_and_report
    except ImportError as e:
        print(f"ERROR: Could not import necessary modules: {e}")
        print("Please ensure that the script is run from the project's root directory and that all dependencies are installed.")
        sys.exit(1)

def get_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Run the PDS patch and report generation pipeline.")
    
    # Add arguments with defaults from pds_config.py
    parser.add_argument("--dataset_root", type=str, default=DATASET_ROOT,
                        help="Path to the root of the dataset.")
    parser.add_argument("--pds_output_dir", type=str, default=PDS_OUTPUT_DIR,
                        help="Directory to save PDS patches.")
    parser.add_argument("--analysis_output_dir", type=str, default=ANALYSIS_OUTPUT_DIR,
                        help="Directory to save analysis reports.")
    parser.add_argument("--patch_size", type=int, default=PATCH_SIZE,
                        help="The size of the patches to create.")
    parser.add_argument("--pds_radius", type=int, default=PDS_RADIUS,
                        help="The radius for PDS sampling.")
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS,
                        help="Number of worker processes for data loading.")
    
    return parser.parse_args()

def main():
    """Main function to run the pipeline."""
    args = get_args()

    # Convert paths to Path objects
    dataset_root = Path(args.dataset_root)
    pds_output_dir = Path(args.pds_output_dir)
    analysis_output_dir = Path(args.analysis_output_dir)

    # Print configuration for user reference
    print("="*50)
    print("Starting PDS Patch and Report Generation Pipeline")
    print("="*50)
    print(f"  - Dataset Root: {dataset_root}")
    print(f"  - PDS Output Dir: {pds_output_dir}")
    print(f"  - Analysis Output Dir: {analysis_output_dir}")
    print(f"  - Patch Size: {args.patch_size}")
    print(f"  - PDS Radius: {args.pds_radius}")
    print(f"  - Number of Workers: {args.num_workers}")
    print("="*50 + "\n")

    # Validate dataset root
    if not dataset_root.exists():
        if str(dataset_root) == "path/to/your/coralscapes/dataset":
            print("ERROR: Please update 'DATASET_ROOT' in pds_config.py or provide the path via command-line.")
        else:
            print(f"ERROR: Dataset root directory not found: {dataset_root}")
        return 1

    try:
        # Run the main pipeline function
        create_pds_patches_and_report(
            dataset_root=dataset_root,
            pds_output_dir=pds_output_dir,
            analysis_output_dir=analysis_output_dir,
            patch_size=args.patch_size,
            pds_radius=args.pds_radius,
            num_workers=args.num_workers,
            task_definition_path=None  # Kept as None for now
        )
        
        print("\n" + "="*50)
        print("Pipeline completed successfully!")
        return 0
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
        return 1
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    # Change to the project root directory to ensure relative paths work correctly
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    sys.exit(main())