"""
Configuration file for PDS patches and report generation.

Modify these parameters according to your dataset and requirements.
"""

import os
from pathlib import Path

project_root = Path(__file__).parent.parent
# =============================================================================
# DATASET CONFIGURATION
# =============================================================================

# Path to the root of the original 'coralscapes' directory
# This directory should contain 'leftImg8bit' and 'gtFine' subdirectories
DATASET_ROOT = project_root / "../coralscapes"  # TODO: Update this path!

# =============================================================================
# OUTPUT CONFIGURATION
# =============================================================================

# Directory where the PDS-generated patch dataset will be saved
PDS_OUTPUT_DIR = project_root / "./dataset/processed/pds_patches"

# Root directory for all analysis reports and plots
ANALYSIS_OUTPUT_DIR = project_root / "./experiments/pds/data_analysis/no_task_def"

# =============================================================================
# PATCH GENERATION PARAMETERS
# =============================================================================

# Width and height of the square patches to extract
PATCH_SIZE = 512

# Minimum distance between centers of any two patches for Poisson Disk Sampling
# Smaller values = more patches but more overlap
# Larger values = fewer patches but better distribution
PDS_RADIUS = 300

# Number of CPU cores to use for parallel processing
# Set to None to use all available cores
NUM_WORKERS = 32  # Will use os.cpu_count() if None

# =============================================================================
# TASK DEFINITION (OPTIONAL)
# =============================================================================

# Path to a YAML file defining the class remapping for flattening labels
# Set to None to skip task-specific label remapping
#TASK_DEFINITION_PATH = "./configs/task_definitions.yaml"

# =============================================================================
# DERIVED SETTINGS (DO NOT MODIFY)
# =============================================================================

if NUM_WORKERS is None:
    NUM_WORKERS = os.cpu_count()
