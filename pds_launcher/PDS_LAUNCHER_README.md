# PDS Patches and Report Generator - Simple Launcher

This directory contains simple launcher scripts for the `create_pds_patches_and_report.py` pipeline.

## Quick Start

### Method 1: Using Python Directly
1. Edit `pds_config.py` to configure your settings
2. Run: `python pds_simple_script.py`

### Method 2: Manual Command Line
Run the original script directly:
```bash
python scripts/create_pds_patches_and_report.py --dataset_root "path/to/dataset" [other options]
```

## How It Works

The simple launcher script (`pds_simple_script.py`) now directly imports and calls the `create_pds_patches_and_report` function instead of using subprocess. This approach is:
- More efficient (no subprocess overhead)
- Better error handling and debugging
- Cleaner integration with the existing codebase
- Preserves the same Python environment and imports

## Configuration

Edit `pds_config.py` to customize:

- **DATASET_ROOT**: Path to your coralscapes dataset directory
- **PDS_OUTPUT_DIR**: Where to save the generated patches
- **ANALYSIS_OUTPUT_DIR**: Where to save analysis reports and plots
- **PATCH_SIZE**: Size of extracted patches (default: 512)
- **PDS_RADIUS**: Minimum distance between patch centers (default: 300)
- **NUM_WORKERS**: Number of CPU cores to use (default: all available)
- **TASK_DEFINITION_PATH**: Optional YAML file for label remapping

## What the Pipeline Does

1. **Creates PDS Dataset**: Extracts patches using Poisson Disk Sampling
2. **Analyzes Distribution**: Generates analysis reports for the patch dataset
3. **Compares Distributions**: Compares the new dataset against the original

## Output Structure

```
data/processed/pds_patches/          # Generated patches
experiments/data_analysis/
├── pds_analysis/                    # Patch distribution analysis
└── comparison_results/              # Original vs PDS comparison
```

## Requirements

Make sure you have all dependencies installed:
```bash
pip install -r requirements.txt
```

## Troubleshooting

- Ensure your dataset path in `pds_config.py` is correct
- Check that the dataset contains `leftImg8bit` and `gtFine` directories
- Verify you have sufficient disk space for the output
- Make sure all required Python packages are installed