# Coral-MTL: Hierarchical Multi-Task Learning for Coral Reef Health Assessment

**Authors:** Group 21 - Alessandro Da Ros, Ergi Livanaj, Sushrut Patil, Alexandru Radu, Gabriel Merle, Mateusz Lotko  
**Course:** Capstone Data Challenge (JBG060)  
**Institution:** Eindhoven University of Technology  
**Stakeholder:** ReefSupport

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Problem Statement & Context](#problem-statement--context)
3. [Solution Approach](#solution-approach)
4. [Repository Structure](#repository-structure)
5. [Installation & Environment Setup](#installation--environment-setup)
6. [Data Pipeline](#data-pipeline)
7. [Model Training](#model-training)
8. [Evaluation & Results](#evaluation--results)
9. [Reproducing Results](#reproducing-results) **-> This is where you find the main Jupyter Notebook for the reproduction of results, analysis, and plots.**
10. [Testing & Quality Assurance](#testing--quality-assurance)
11. [Documentation](#documentation)
12. [Citation & References](#citation--references)

---

## Project Overview

Coral-MTL is a hierarchical multi-task learning framework for automated coral reef health assessment from underwater imagery. The system simultaneously predicts:

- **Genus identification** (8 coral genera)
- **Health status** (alive/bleached/dead)
- **Contextual information** (fish, human artifacts, substrate, background, biota)

### Key Achievements

- **+9.9% global mIoU** over baseline SegFormer
- **+32.6% boundary IoU** improvement (critical for cover estimation)
- **+28.9% boundary F1** enhancement (perimeter fidelity)
- **Honest uncertainty reporting** (ECE, NLL, Brier scores without post-hoc calibration)

### Three Model Variants

1. **Baseline SegFormer** - Single flat decoder (40 classes)
2. **MTL Focused** - 2 primary tasks (genus+health) + 5 auxiliary
3. **MTL Holistic** - All 7 tasks as primary with full cross-attention (**best performer**)

![alt text](/latex/Methodology/Result-figures/architecture_comparison.svg)

### Cross-Attention layer

Our MTL design is based on Explicit Feature Exchange, allowing for tasks to dynamically "query" each other for relevant context using a cross-attention mechanism. Given the necessity of making our model sustainable we develop a full cross-attention only for primary tasks (all MLP decoders) yet, to provide additional context we still enable feature-exchange of auxiliary tensors (1x1 convolutional blocks). In detail, primary decoders perform cross-attention: pooled Q from the task attends to concatenated K/V from all other tasks. Their enriched features are gated and fused with the original. Auxiliary heads (fish, human_artifacts, substrate) are lightweight; they provide K/V context but don’t attend, acting as regularizers.

![alt text](/latex/Methodology/Result-figures/feature_exchange_detail.svg)

---

## Problem Statement & Context

### Business Need

ReefSupport requires automated coral reef health assessment at the pace of image collection. Manual annotation cannot keep up with data acquisition rates, delaying management decisions. Current semi-automated tools:

- Miss colony boundaries (critical for cover estimation)
- Cannot jointly answer "what coral" and "how healthy"
- Lack calibrated confidence for expert triage

### Technical Challenge

Design a unified model that:
- Provides **dense, pixel-wise** genus and health segmentation
- Achieves **high boundary quality** (not just pixel accuracy)
- Reports **calibrated probabilities** for uncertainty-aware deployment
- Works robustly under **variable field conditions** (turbidity, lighting, depth)

### Dataset

**Coralscapes**: 2075 images, 35 sites, 5 Red Sea countries
- Dense pixel-level annotations (39 benthic classes)
- Live/bleached/dead labels for coral health
- Challenging conditions: variable depth, turbidity, illumination

---

## Solution Approach

### Architecture

**Encoder**: SegFormer-B2 backbone (25.4M parameters)

**Decoders**:
- **Primary tasks** (genus, health): Full MLP decoders with cross-attention feature exchange
- **Auxiliary tasks** (5 context heads): Lightweight regularizers for boundary sharpening

**Innovation**: Explicit feature exchange via cross-attention allows genus (morphology) and health (appearance) predictions to inform each other, while auxiliary heads prevent "coral-shaped background" false positives.

### Training Strategy

- **Loss**: Dice + Focal hybrid with IMGrad gradient balancing
- **Optimization**: AdamW with polynomial decay (6e-5 LR, 50 epochs)
- **Augmentation**: Physics-plausible underwater transformations (haze, color cast, blur)
- **Sampling**: Poisson Disk Sampling (PDS) to reduce spatial redundancy
- **Split**: Site-level hold-out (70% train, 15% val, 15% test)

### Evaluation Metrics

**Primary**: Global mIoU, Boundary IoU (BIoU), Boundary F1  
**Calibration**: ECE, NLL, Brier (reported as-is, no post-hoc adjustment)  
**Diagnostics**: TIDE-inspired error decomposition (classification/background/missed)

---

## Repository Structure

```
coral-mtl-project/
├── configs/                          # YAML experiment configurations
│   ├── baseline_comparisons/         # Production training configs
│   └── task_definitions.yaml         # Hierarchical class definitions
│
├── data/                             # Dataset storage (external)
│   ├── raw/coralscapes/              # Original images (not tracked)
│   └── processed/pds_patches/        # PDS-sampled patches (generated)
│
├── experiments/                      # Training outputs
│   └── baseline_comparisons/
│       ├── coral_baseline_b2_run/    # Baseline results
│       ├── coral_mtl_b2_focused_run/ # MTL Focused results
│       └── coral_mtl_b2_holistic_run/# MTL Holistic results (best)
│
├── notebooks/                        # Analysis & visualization
│   └── FINAL_NOTEBOOK.ipynb          # Complete results reproduction
│
├── latex/                            # Report & poster source
│   ├── Methodology/                  # Final report (LaTeX)
│   └── Poster_Data_shallange/        # Conference poster
│
├── pds_launcher/                     # Dataset preprocessing
│   ├── pds_simple_script.py          # PDS execution script
│   └── pds_config.py                 # Sampling parameters
│
├── src/coral_mtl/                    # Core library (installable)
│   ├── ExperimentFactory.py          # Central orchestrator
│   ├── data/                         # Dataloaders & augmentations
│   ├── model/                        # Architecture components
│   ├── engine/                       # Training, losses, optimizers
│   ├── metrics/                      # 3-tier metrics system
│   └── utils/                        # Task splitters & helpers
│
├── tests/                            # Pytest suite
│   └── coral_mtl_tests/              # Mirrors src/ structure
│
├── scripts/                          # Standalone utilities
│   └── experiments/baselines_comparison/
│       └── train_val_test_script.py  # HPC training orchestrator
│
├── requirements.txt                  # Python dependencies
├── pytest.ini                        # Test configuration
└── README.md                         # This file
```

---

## Installation & Environment Setup

### Prerequisites

- **Python 3.9+**
- **CUDA 11.8+** (for GPU training, optional for evaluation)
- **48GB+ VRAM** (for training, RTX 6000Ada or similar)
- **8GB+ RAM** (sufficient for evaluation/inference)

### Setup Steps

1. **Clone repository**
   ```bash
   git clone https://github.com/A-DaRo/dbl_capstone
   cd dbl_capstone
   ```

2. **Install dependencies**
   ```bash
   python.exe -m pip install --upgrade pip
   pip install -r requirements.txt        # Refer to `requirements.txt` if you want to check the specific version of each dependency
   ```

3. **Install project as editable package**
   ```bash
   pip install -e .
   ```
   This enables package-style imports (`from coral_mtl.ExperimentFactory import ...`)

   ### ***Make sure to have installed all dependencies before running the Jupyter Notebook!***

4. **Verify installation**
   ```bash
   pytest tests/coral_mtl_tests/data/ -v
   ```

5. **MOCK run on CPU with small dataset**
   ```bash
   python tests/trial_run_test.py
   ```

### Hardware Requirements

**Training** (HPC/Cloud):
- GPU: 48GB+ VRAM (RTX 6000Ada, A6000, A100)
- RAM: 64GB+ system memory
- Storage: 500GB+ for dataset + checkpoints
- Time: ~9 hours per model (50 epochs)

**Evaluation/Inference** (Laptop):
- GPU: Optional (CPU sufficient for evaluation)
- RAM: 8GB+ system memory
- Storage: 10GB for checkpoints + outputs

---

## Data Pipeline

### 1. Raw Dataset Download & Setup

**Dataset Source**: [Coralscapes on Zenodo](https://zenodo.org/records/15061505)

The Coralscapes dataset is distributed as a compressed `.7z` archive.

#### Linux/Mac Setup

```bash
# Install required utilities (Ubuntu/Debian)
sudo apt install p7zip-full curl

# Navigate to parent directory (one level above project root)
cd ..

# Download dataset
curl -L -o coralscapes.7z "https://zenodo.org/records/15061505/files/coralscapes.7z?download=1"

# Alternative: using wget
wget -O coralscapes.7z "https://zenodo.org/records/15061505/files/coralscapes.7z?download=1"

# Extract archive (creates coralscapes/ directory)
7z x coralscapes.7z

# Verify structure
ls coralscapes/
# Expected output: leftImg8bit/ gtFine/ README.md (or similar)

# Return to project directory
cd dbl_capstone
```

#### Windows Setup

**Manual (Recommended for Windows)**
1. Download from browser: [Direct Link](https://zenodo.org/records/15061505/files/coralscapes.7z?download=1)
2. Install [7-Zip](https://www.7-zip.org/download.html) if not present
3. Extract `coralscapes.7z` to the **parent directory** of `dbl_capstone/`
4. Final structure should be:
   ```
   your_workspace/
   ├── dbl_capstone/          # This project
   └── coralscapes/           # Dataset (extracted here)
       ├── leftImg8bit/
       └── gtFine/
   ```

### 2. Poisson Disk Sampling (PDS)

**Purpose**: Extract spatially distributed patches to reduce redundancy in overlapping orthomosaics.

**Configuration**: Edit `pds_launcher/pds_config.py`
```python
DATASET_ROOT = Path("../dataset/coralscapes")  # Update this in case the dataset is elsewhere
PATCH_SIZE = 512
PDS_RADIUS = 300  # Minimum distance between patch centers
```

**Execution**:
```bash
cd pds_launcher
python pds_simple_script.py
```

**Outputs**:
- `dataset/processed/pds_patches/` - ~15,000 patches (train/val/test)
- `experiments/pds/data_analysis/` - Distribution reports

### 3. Data Verification

Inspect distribution:
```bash
python scripts/analyze_patch_distribution.py \
  --dataset_root dataset/processed/pds_patches \
  --output experiments/pds/data_analysis
```

---

## Model Training

### Configuration

Training is controlled via YAML configs in `configs/baseline_comparisons/`:

- `baseline_config.yaml` - Single-task SegFormer
- `mtl_config.yaml` - Multi-task variants (Focused/Holistic)

**Key parameters**:
```yaml
model:
  type: "CoralMTL"  # or "SegFormerBaseline"
  params:
    backbone: "nvidia/mit-b2"
    decoder_channel: 256
    attention_dim: 128

data:
  batch_size: 4
  patch_size: 512
  pds_train_path: "./dataset/processed/pds_patches/"

loss:
  type: "CompositeHierarchical"
  weighting_strategy:
    type: "IMGrad"  # Gradient balancing

optimizer:
  params:
    lr: 6.0e-5
    weight_decay: 0.01

trainer:
  epochs: 50
  device: "cuda"
  output_dir: "experiments/baseline_comparisons/coral_mtl_b2_holistic_run"
```

### HPC Training Execution

**Script**: `experiments/baselines_comparison/train_val_test_script.py`

**Run all models**:
```bash
python experiments/baselines_comparison/train_val_test_script.py --mode both
```

**Run specific model**:
```bash
# Baseline only
python experiments/baselines_comparison/train_val_test_script.py --mode baseline

# MTL only
python experiments/baselines_comparison/train_val_test_script.py --mode mtl
```

**Evaluation only** (skip training):
```bash
python experiments/baselines_comparison/train_val_test_script.py --eval-only
```

### Training Outputs

Each run creates:
```
experiments/baseline_comparisons/<run_name>/
├── best_model.pth                    # Best checkpoint (model selection metric)
├── history.json                      # Per-epoch training/validation metrics
├── test_metrics_full_report.json     # Final test evaluation
├── test_cms.jsonl                    # Per-image confusion matrices
└── loss_diagnostics.jsonl            # Gradient norms, cosine similarity
```

### Monitoring Training

Track progress via `history.json`:
```python
import json
history = json.load(open('experiments/.../history.json'))
print(f"Epoch 50 mIoU: {history['global.mIoU'][-1]:.4f}")
```

---

## Evaluation & Results

### Test Set Evaluation

**Automatic** (part of training pipeline):
```bash
python experiments/baselines_comparison/train_val_test_script.py --mode mtl
```

**Manual** (specific checkpoint):
```python
from coral_mtl.ExperimentFactory import ExperimentFactory

factory = ExperimentFactory('configs/baseline_comparisons/mtl_config.yaml')
results = factory.run_evaluation(
    checkpoint_path='experiments/.../best_model.pth'
)
```

### Key Results (Test Set)

| Metric | Baseline | MTL Focused | **MTL Holistic** |
|--------|----------|-------------|------------------|
| Global mIoU ↑ | 0.3888 | 0.4039 | **0.4272** (+9.9%) |
| Global BIoU ↑ | 0.0937 | 0.1075 | **0.1243** (+32.6%) |
| Boundary F1 ↑ | 0.1714 | 0.1942 | **0.2211** (+28.9%) |
| ECE ↓ | **0.1014** | 0.1275 | 0.1423 |
| NLL ↓ | **1.2239** | 1.3995 | 1.5162 |
| Brier ↓ | 0.5016 | 0.4959 | **0.4937** |

**Interpretation**:
- MTL Holistic achieves best segmentation/boundary quality
- Baseline slightly better calibrated (lower ECE/NLL)
- Trade-off reflects conservative behavior in low-contrast scenes

### Error Analysis

TIDE-inspired decomposition shows:
- **Classification errors**: ↓ 15% (Baseline → Holistic)
- **Background FPs**: ↓ 28% (fewer "coral-shaped substrate" errors)
- **Missed regions**: ↑ 12% (conservative in low contrast)

**Actionable**: Target data curation for faint bleaching, modest focal loss rebalancing.

---

## Reproducing Results

### Complete Pipeline Reproduction

**Notebook**: `notebooks/FINAL_NOTEBOOK.ipynb`

### ***Make sure to have installed all dependencies before running the Jupyter Notebook!***
```bash
python.exe -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

**Execution** (laptop-friendly):
```bash
jupyter notebook notebooks/FINAL_NOTEBOOK.ipynb
```

**Generated artifacts**:
- All report figures → `latex/Methodology/Result-figures/`
- All poster figures → `latex/Poster_Data_shallange/Result-figures/`
- Statistics → `notebooks/extra_plots_stats/`

**Sections**:
1. Environment setup & dependency verification
2. Data pipeline documentation (PDS)
3. Training overview (references HPC execution)
4. Experiment artifact discovery
5. Training dynamics analysis
6. Per-class performance analysis
7. Test set evaluation & inference
8. Qualitative visualizations
9. Architecture diagrams (Graphviz)
10. Extra plots for report/poster

### Quick Start (Evaluation Only)

If training is already complete:

```python
# Load notebook and run from Section 3 onwards
# All pre-trained checkpoints expected in experiments/baseline_comparisons/
```

### From Scratch (Full Pipeline)

1. **Data preparation**: Run PDS sampling (see [Data Pipeline](#data-pipeline))
2. **Training**: Execute HPC script (see [Model Training](#model-training))
3. **Analysis**: Run complete notebook (see above)
4. **Report**: Compile LaTeX sources in `latex/`

**Total time**: ~12 hours

---

## Testing & Quality Assurance

### Test Suite

**Structure**: `tests/` mirrors `src/` package layout

**Coverage**: 85%+ on core modules (data, model, engine, metrics)

### Running Tests

**Full suite with coverage**:
```bash
pytest
```

**Targeted tests**:
```bash
# Data pipeline only
pytest tests/coral_mtl_tests/data/

# Loss functions only
pytest tests/coral_mtl_tests/engine/losses/

# Exclude integration tests
pytest -m "not integration"
```

**GPU tests** (if CUDA available):
```bash
set CUDA_VISIBLE_DEVICES=0
pytest -m gpu
```

**Coverage reports**:
```bash
# Terminal summary (automatic via pytest.ini)
pytest

# HTML report
pytest --cov-report=html
# Opens htmlcov/index.html
```

### Test Markers

- `@pytest.mark.gpu` - Requires CUDA device
- `@pytest.mark.integration` - Slower end-to-end tests
- `@pytest.mark.optdeps` - Requires optional dependencies
- `@pytest.mark.slow` - Long-running tests

### Continuous Validation

Before committing:
```bash
pytest -m "not integration"  # Fast unit tests
```

Before major changes:
```bash
pytest  # Full suite including integration
```

---

## Documentation

### Primary Documents

1. **Technical Specification** - `project_specification/technical_specification.md`
   - Complete API reference
   - Component interfaces
   - Factory orchestration

2. **Theoretical Specification** - `project_specification/theorethical_specification.md`
   - Design rationale
   - Multi-task learning theory
   - Metric justifications

3. **Loss & Optimization Guide** - `project_specification/loss_and_optim_specification.md`
   - Weighting strategies (Uncertainty, NashMTL, IMGrad)
   - Gradient manipulation techniques
   - Diagnostic-driven selection

4. **Configuration Guide** - `configs/CONFIGS_README.md`
   - All YAML parameters documented
   - Example configurations
   - Validation checklist

### Reports & Publications

- **Final Report**: `latex/Methodology/final-report.tex`
- **Conference Poster**: `latex/Poster_Data_shallange/poster1.tex`
- **Results Notebook**: `notebooks/FINAL_NOTEBOOK.ipynb`

### In-Code Documentation

- **Docstrings**: NumPy style throughout `src/`
- **Type Hints**: Full typing coverage in core modules

---

## Key References

- **Dataset**: Sauder et al. (2025) - Coralscapes: Densely annotated coral reef dataset
- **Baseline**: Xie et al. (2021) - SegFormer: Simple and Efficient Design
- **MTL Framework**: Liu et al. (2019) - End-to-End Multi-Task Learning with Attention
- **Gradient Balancing**: Zhou et al. (2025) - IMGrad: Balancing Gradient Magnitude
- **Boundary Metrics**: Cheng et al. (2021) - Boundary IoU
- **Error Decomposition**: Bolya et al. (2020) - TIDE: A General Toolbox

Full bibliography: See `latex/Methodology/references.bib`

---

## Contact & Support

**Group 21 Members**:
- Alessandro Da Ros - a.da.ros@student.tue.nl
- Ergi Livanaj - e.livanaj@student.tue.nl
- Sushrut Patil - s.patil@student.tue.nl
- Alexandru Radu - i.a.radu@student.tue.nl
- Gabriel Merle - g.merle@student.tue.nl
- Mateusz Lotko - m.lotko@student.tue.nl

**Institution**: Eindhoven University of Technology  
**Course**: JBG060 - Capstone Data Challenge  
**Stakeholder**: ReefSupport

**Issues**: Please use repository issue tracker for bug reports or questions.

---

## License

This project is developed for academic purposes as part of the Capstone Data Challenge course at TU/e. Code is provided as-is for educational and research use.

**Dataset License**: Coralscapes dataset follows its original license terms (see Hugging Face Hub).

---

## Acknowledgments

- **ReefSupport** for problem formulation and domain expertise
- **TU/e Faculty** for guidance and compute resources
- **Coralscapes Authors** for the high-quality annotated dataset
- **Open Source Community** for foundational libraries (PyTorch, Hugging Face, etc.)

---

**Version**: 1.0 (Final Submission)
