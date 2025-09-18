#!/usr/bin/env python3
"""
Simple launcher script for create_pds_patches_and_report.py

This script provides an easy way to launch the PDS patches and report generation
with predefined parameters. Modify the parameters in pds_config.py as needed.
"""

import os
import sys
from pathlib import Path

def main():
    # Get the project root directory (where this script is located)
    project_root = Path(__file__).parent
    
    # Add the scripts directory to Python path to enable imports
    scripts_dir = project_root / "scripts"
    sys.path.insert(0, str(scripts_dir))
    
    # Import configuration
    try:
        from pds_config import (
            DATASET_ROOT, PDS_OUTPUT_DIR, ANALYSIS_OUTPUT_DIR,
            PATCH_SIZE, PDS_RADIUS, NUM_WORKERS #TASK_DEFINITION_PATH
        )
    except ImportError:
        print("ERROR: Could not import pds_config.py")
        print("Make sure the configuration file exists and is properly formatted.")
        return 1
    
    # Import the function from the script
    try:
        from scripts.create_pds_patches_and_report import create_pds_patches_and_report
    except ImportError as e:
        print(f"ERROR: Could not import create_pds_patches_and_report function: {e}")
        print("Make sure the scripts directory and all required modules are available.")
        return 1
    
    # Use configuration values
    dataset_root = Path(DATASET_ROOT)
    pds_output_dir = Path(PDS_OUTPUT_DIR)
    analysis_output_dir = Path(ANALYSIS_OUTPUT_DIR)
    patch_size = PATCH_SIZE
    pds_radius = PDS_RADIUS
    num_workers = NUM_WORKERS
    #task_definition_path = Path(TASK_DEFINITION_PATH) if TASK_DEFINITION_PATH else None
    task_definition_path = None
    
    # Validate task definition path
    if task_definition_path and task_definition_path.exists():
        print(f"Using task definition file: {task_definition_path}")
    elif task_definition_path:
        print(f"Task definition file not found at {task_definition_path}, proceeding without it")
        task_definition_path = None
    else:
        print("No task definition file specified, proceeding without it")
    
    # Print configuration for reference
    print("Launching create_pds_patches_and_report with the following configuration:")
    print(f"  Dataset Root: {dataset_root}")
    print(f"  PDS Output Dir: {pds_output_dir}")
    print(f"  Analysis Output Dir: {analysis_output_dir}")
    print(f"  Patch Size: {patch_size}")
    print(f"  PDS Radius: {pds_radius}")
    print(f"  Number of Workers: {num_workers}")
    print(f"  Task Definition Path: {task_definition_path}")
    print("\n" + "="*50 + "\n")
    
    # Check if dataset root exists
    if not dataset_root.exists():
        if str(dataset_root) == "path/to/your/coralscapes/dataset":
            print("ERROR: Please update the 'DATASET_ROOT' path in pds_config.py to point to your actual dataset!")
        else:
            print(f"ERROR: Dataset root directory does not exist: {dataset_root}")
        return 1
    
    # Change to project root directory to ensure relative paths work correctly
    os.chdir(project_root)
    
    try:
        # Call the function directly
        create_pds_patches_and_report(
            dataset_root=dataset_root,
            pds_output_dir=pds_output_dir,
            analysis_output_dir=analysis_output_dir,
            patch_size=patch_size,
            pds_radius=pds_radius,
            num_workers=num_workers,
            task_definition_path=task_definition_path
        )
        
        print("\n" + "="*50)
        print("Pipeline completed successfully!")
        return 0
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user")
        return 1
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())