# Configuration Guide for Coral Segmentation Experiments

This comprehensive guide documents all configurable parameters for the `ExperimentFactory` system used in coral reef segmentation experiments. The factory supports both Multi-Task Learning (MTL) and baseline segmentation models with extensive customization options.

## Table of Contents

1. [Configuration Structure](#configuration-structure)
2. [Model Configuration](#model-configuration)
3. [Data Configuration](#data-configuration)
4. [Augmentations Configuration](#augmentations-configuration)
5. [Loss Configuration](#loss-configuration)
6. [Optimizer Configuration](#optimizer-configuration)
7. [Metrics Configuration](#metrics-configuration)
8. [Trainer Configuration](#trainer-configuration)
9. [Evaluator Configuration](#evaluator-configuration)
10. [Hyperparameter Study Configuration](#hyperparameter-study-configuration)
11. [Available Model Selection Metrics](#available-model-selection-metrics)
12. [Configuration Examples](#configuration-examples)
13. [Configuration Validation](#configuration-validation)

---

## Configuration Structure

All configurations are defined in YAML format and contain the following top-level sections accessible through the `ExperimentFactory`:

```yaml
model:              # Model architecture and parameters  
data:               # Dataset and dataloader configuration
augmentations:      # Data augmentation settings (optional)
loss:               # Loss function configuration
optimizer:          # Optimizer and scheduler settings
metrics:            # Metrics calculation parameters (optional)
metrics_processor:  # Advanced metrics processing (optional)
trainer:            # Training loop configuration
evaluator:          # Evaluation settings (optional)
study:              # Optuna hyperparameter optimization (optional)
```

**Required Sections**: `model`, `data`, `loss`, `optimizer`, `trainer`  
**Optional Sections**: `augmentations`, `metrics`, `metrics_processor`, `evaluator`, `study`

**Factory Methods Mapping:**
- `model` → `ExperimentFactory.get_model()`
- `data` → `ExperimentFactory.get_dataloaders()`  
- `loss` → `ExperimentFactory.get_loss_function()`
- `optimizer` → `ExperimentFactory.get_optimizer_and_scheduler()`
- `metrics` → `ExperimentFactory.get_metrics_calculator()`
- `trainer` → `ExperimentFactory.run_training()`
- `evaluator` → `ExperimentFactory.run_evaluation()`
- `study` → `ExperimentFactory.run_hyperparameter_study()`

---

## Model Configuration

The model configuration defines the architecture and task setup for your experiment. The `ExperimentFactory.get_model()` method uses this configuration to instantiate the appropriate model class.

### Required Parameters

```yaml
model:
  type: str  # REQUIRED: Model architecture type ("CoralMTL" or "SegFormerBaseline")
```

### Model Types

#### 1. CoralMTL (Multi-Task Learning)

**Required Configuration Structure:**
```yaml
model:
  type: "CoralMTL"                    # REQUIRED: Must be exactly "CoralMTL"
  params:
    backbone: str                     # REQUIRED: SegFormer backbone from Hugging Face
    decoder_channel: int              # REQUIRED: Unified decoder channel dimension
    attention_dim: int                # REQUIRED: Cross-attention dimension
    encoder_weights: str              # OPTIONAL: Pre-trained weights (default: "imagenet")
    encoder_depth: int                # OPTIONAL: Encoder depth (default: 5)
  tasks:
    primary: list[str]                # REQUIRED: Primary task names
    auxiliary: list[str]              # REQUIRED: Auxiliary task names
```

**Parameter Details:**
- `backbone`: Valid Hugging Face SegFormer model ID (see Available Backbones section)
- `decoder_channel`: Integer, typically 128-512, controls model capacity
- `attention_dim`: Integer, typically 64-256, dimension for cross-attention mechanism
- `encoder_weights`: String, "imagenet" or None for random initialization
- `encoder_depth`: Integer 1-5, number of encoder stages to use
- `tasks.primary`: List of task names that must exist in `task_definitions.yaml`
- `tasks.auxiliary`: List of auxiliary task names, also defined in task definitions

**Complete Example:**
```yaml
model:
  type: "CoralMTL"
  params:
    backbone: "nvidia/mit-b2"
    decoder_channel: 256
    attention_dim: 128
    encoder_weights: "imagenet"
    encoder_depth: 5
  tasks:
    primary: ["genus", "health"]
    auxiliary: ["fish", "human_artifacts", "substrate", "background", "biota"]
```

#### 2. SegFormerBaseline (Single-Task)

**Required Configuration Structure:**
```yaml
model:
  type: "SegFormerBaseline"           # REQUIRED: Must be exactly "SegFormerBaseline" 
  params:
    backbone: str                     # REQUIRED: SegFormer backbone from Hugging Face
    decoder_channel: int              # REQUIRED: MLP decoder channel dimension
    num_classes: int                  # REQUIRED: Total number of output classes
    encoder_weights: str              # OPTIONAL: Pre-trained weights (default: "imagenet")
    encoder_depth: int                # OPTIONAL: Encoder depth (default: 5)
```

**Parameter Details:**
- `num_classes`: Integer, total flattened classes from task definitions (e.g., 40)
- Other parameters same as CoralMTL model

**Complete Example:**
```yaml
model:
  type: "SegFormerBaseline"
  params:
    backbone: "nvidia/mit-b2"
    decoder_channel: 256
    num_classes: 40
    encoder_weights: "imagenet"
    encoder_depth: 5
```

### Available Backbones

- `nvidia/mit-b0` (3.7M parameters)
- `nvidia/mit-b1` (14.0M parameters)  
- `nvidia/mit-b2` (25.4M parameters)
- `nvidia/mit-b3` (45.2M parameters)
- `nvidia/mit-b4` (62.6M parameters)
- `nvidia/mit-b5` (82.0M parameters)

---

## Data Configuration

The data configuration controls dataset loading, preprocessing, and DataLoader creation via `ExperimentFactory.get_dataloaders()`.

### Required Parameters

```yaml
data:
  dataset_name: str              # REQUIRED: Local path or HuggingFace dataset ID
  task_definitions_path: str     # REQUIRED: Path to task definitions YAML file
  batch_size: int               # REQUIRED: Training batch size per GPU
  num_workers: int              # REQUIRED: DataLoader worker processes (0-16 typical)
  patch_size: int               # REQUIRED: Square patch size for training/inference
```

### Optional Parameters

```yaml
data:
  pds_train_path: str           # OPTIONAL: Path to PDS-sampled training patches
  data_root_path: str           # OPTIONAL: Path to raw dataset (fallback for val/test)
  ignore_index: int             # OPTIONAL: Class index to ignore in loss/metrics (default: 255)
```

### Parameter Details

**Required Parameters:**
- `dataset_name`: 
  - **Type**: String
  - **Values**: Local path (absolute/relative) or HuggingFace dataset ID
  - **Examples**: `"./dataset/coralscapes/"`, `"EPFL-ECEO/coralscapes"`
  - **Factory Usage**: Passed to `CoralscapesMTLDataset` or `CoralscapesDataset`

- `task_definitions_path`:
  - **Type**: String (path)
  - **Values**: Path to YAML file with class/task definitions
  - **Required**: Always, even for baseline models (used for evaluation)
  - **Factory Usage**: Loaded in `_initialize_task_splitter()` for class counts

- `batch_size`:
  - **Type**: Integer
  - **Range**: 1-64 (depends on GPU memory and patch_size)
  - **Factory Usage**: Used as `batch_size_per_gpu` in DataLoader creation

- `num_workers`:
  - **Type**: Integer  
  - **Range**: 0-16 (0=no multiprocessing, higher=faster loading but more RAM)
  - **Factory Usage**: Direct DataLoader parameter

- `patch_size`:
  - **Type**: Integer
  - **Range**: 256, 512, 768, 1024 (powers of 2 recommended)
  - **Factory Usage**: Controls spatial dimensions for training/augmentation

**Optional Parameters:**
- `pds_train_path`: Path to pre-processed training patches (prioritized over raw data)
- `data_root_path`: Fallback path for validation/test splits when using PDS training
- `ignore_index`: Class index ignored in loss computation (must match loss configuration)

### Complete Example

```yaml
data:
  dataset_name: "./dataset/coralscapes/"
  pds_train_path: "./dataset/processed/pds_patches/"
  data_root_path: "./dataset/coralscapes/"
  task_definitions_path: "configs/task_definitions.yaml"
  batch_size: 4
  num_workers: 4
  patch_size: 512
  ignore_index: 0
```

### Dataset Loading Priority

The factory follows this loading hierarchy:
1. **Training Split**: Uses `pds_train_path` if available, otherwise `dataset_name`
2. **Validation/Test**: Always uses `dataset_name` or `data_root_path` (full images)
3. **Path Resolution**: Relative paths resolved from config file location

---

## Augmentations Configuration

**Optional Section**: Controls data augmentation via `SegmentationAugmentation` class. If omitted, no augmentations are applied to training data.

### Configuration Structure

```yaml
augmentations:
  crop_scale: list[float]       # OPTIONAL: RandomResizedCrop scale range
  rotation_degrees: int         # OPTIONAL: Random rotation range in degrees  
  jitter_params:               # OPTIONAL: ColorJitter parameters (image-only)
    brightness: float          # OPTIONAL: Brightness variation
    contrast: float            # OPTIONAL: Contrast variation
    saturation: float          # OPTIONAL: Saturation variation  
    hue: float                # OPTIONAL: Hue variation
```

### Parameter Details

- `crop_scale`:
  - **Type**: List of two floats
  - **Range**: [0.1, 1.0] for each value, first ≤ second
  - **Default**: [0.5, 1.0]
  - **Usage**: Scale range for RandomResizedCrop before resize to patch_size

- `rotation_degrees`:
  - **Type**: Integer
  - **Range**: 0-180 degrees
  - **Default**: 15
  - **Usage**: Random rotation in range [-degrees, +degrees]

- `jitter_params.*`:
  - **Type**: Float for each parameter
  - **Range**: 0.0-1.0 (0.0 = no effect)
  - **Default**: brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
  - **Usage**: Applied only to images (not masks) for color variation

### Augmentation Pipeline

The factory applies augmentations in this order:
1. **Geometric transforms** (synchronized on image + all masks):
   - RandomResizedCrop with `crop_scale`
   - RandomHorizontalFlip (p=0.5)
   - RandomVerticalFlip (p=0.5) 
   - RandomRotation with `rotation_degrees`

2. **Color transforms** (image only):
   - ColorJitter with `jitter_params`
   - GaussianBlur (kernel_size=(5,9), sigma=(0.1,5.0))

3. **Normalization**: ImageNet mean/std applied to final image

### Example Configuration

```yaml
augmentations:
  crop_scale: [0.5, 1.0]
  rotation_degrees: 15
  jitter_params:
    brightness: 0.2
    contrast: 0.2
    saturation: 0.2
    hue: 0.1
```

### Factory Integration

- **Training**: Augmentations applied via `SegmentationAugmentation`
- **Validation/Test**: No augmentations (only normalization)
- **Dependency**: Requires `data.patch_size` for final crop dimensions

---

## Loss Configuration

The loss configuration defines the loss function used for training via `ExperimentFactory.get_loss_function()`. The factory automatically selects the appropriate loss class based on model type.

### Loss Types by Model

#### Multi-Task Loss (CoralMTL Models)

**Configuration Structure:**
```yaml
loss:
  type: "CompositeHierarchical"        # REQUIRED: Must be exactly "CompositeHierarchical"
  params:
    w_consistency: float               # OPTIONAL: Consistency regularizer weight (0.0-1.0, default: 0.1)
    hybrid_alpha: float                # REQUIRED: Primary loss weight (0.0-1.0)
    focal_gamma: float                 # REQUIRED: Focal loss focusing parameter (1.0-5.0)
    ignore_index: int                  # REQUIRED: Class index to ignore (must match data config)
  weighting_strategy:                  # OPTIONAL: Multi-task weighting strategy
    type: str                          # Strategy name (default: "Uncertainty")
    params: dict                       # Strategy-specific parameters
```

**Parameter Details:**
- `w_consistency`: Weight for logical consistency penalty between genus/health tasks
- `hybrid_alpha`: Balance between Focal/CE loss (hybrid_alpha) and Dice loss (1-hybrid_alpha)
- `focal_gamma`: Focusing parameter for Focal Loss (higher = more focus on hard examples)
- `ignore_index`: Class index excluded from loss calculation (typically background)

#### Single-Task Loss (SegFormerBaseline Models)

**Configuration Structure:**
```yaml
loss:
  type: "HybridLoss"                   # REQUIRED: Must be exactly "HybridLoss"
  params:
    primary_loss_type: str             # REQUIRED: "focal" or "cross_entropy"
    hybrid_alpha: float                # REQUIRED: Primary loss weight (0.0-1.0)  
    focal_gamma: float                 # REQUIRED: Focal loss gamma (if primary_loss_type="focal")
    dice_smooth: float                 # OPTIONAL: Dice loss smoothing (default: 1.0)
    ignore_index: int                  # REQUIRED: Class index to ignore
```

**Parameter Details:**
- `primary_loss_type`: Main classification loss ("focal" recommended for imbalanced data)
- `dice_smooth`: Smoothing factor for Dice loss to prevent division by zero
- Other parameters same as MTL loss

### Multi-Task Weighting Strategies

For CoralMTL models, you can specify how to balance losses from different tasks:

#### 1. Uncertainty Weighting (Default, Loss-based)

Learns task-specific uncertainty parameters to balance contributions automatically.

```yaml
loss:
  type: "CompositeHierarchical"
  params:
    # ... other params
  weighting_strategy:
    type: "Uncertainty"
    params:
      clamp_range: float               # OPTIONAL: Clamp log variance to [-range, +range] (default: 10.0)
      learnable_tasks: list[str]       # OPTIONAL: Tasks with learnable weights (default: primary tasks)
```

#### 2. IMGrad (Gradient-based)

Advanced gradient balancing combining MGDA and CAGrad for conflict resolution.

```yaml
loss:
  type: "CompositeHierarchical"
  params:
    # ... other params  
  weighting_strategy:
    type: "IMGrad"
    params:
      solver: str                      # OPTIONAL: "auto", "qp", or "pgd" (default: "auto")
      mgda_pg_steps: int              # OPTIONAL: Projected gradient steps (default: 25)
      mgda_lr: float                  # OPTIONAL: MGDA learning rate (default: 0.25)
```

#### 3. NashMTL (Game-theoretic)

Frames multi-task learning as a bargaining game for proportionally fair updates.

```yaml
loss:
  type: "CompositeHierarchical"
  params:
    # ... other params
  weighting_strategy:
    type: "NashMTL" 
    params:
      update_frequency: int            # OPTIONAL: Recompute weights every N steps (default: 25)
      solver: str                      # OPTIONAL: "auto", "ccp", or "iterative" (default: "auto")
      max_norm: float                  # OPTIONAL: Gradient clipping norm (default: 0.0, disabled)
      optim_niter: int                # OPTIONAL: Optimization iterations (default: 20)
```

#### 4. DWA (Dynamic Weight Averaging)

Adjusts task weights based on recent loss trends.

```yaml
loss:
  type: "CompositeHierarchical"
  params:
    # ... other params
  weighting_strategy:
    type: "DWA"
    params:
      temperature: float               # OPTIONAL: Softmax temperature (default: 2.0)
```

#### 5. GradNorm

Balances gradients to maintain similar learning rates across tasks.

```yaml
loss:
  type: "CompositeHierarchical"  
  params:
    # ... other params
  weighting_strategy:
    type: "GradNorm"
    params:
      alpha: float                     # OPTIONAL: Restoring force strength (default: 0.5)
      lr: float                       # OPTIONAL: Weight update learning rate (default: 0.025)
```

### Complete Examples

**MTL with Uncertainty Weighting:**
```yaml
loss:
  type: "CompositeHierarchical"
  params:
    w_consistency: 0.1
    hybrid_alpha: 0.5
    focal_gamma: 2.0
    ignore_index: 0
  weighting_strategy:
    type: "Uncertainty"
    params:
      clamp_range: 10.0
```

**Baseline with Focal Loss:**
```yaml
loss:
  type: "HybridLoss"
  params:
    primary_loss_type: "focal"
    hybrid_alpha: 0.5
    focal_gamma: 2.0
    dice_smooth: 1.0
    ignore_index: 0
```

---

## Optimizer Configuration

The optimizer configuration controls the training optimization process via `ExperimentFactory.get_optimizer_and_scheduler()`. Currently, only AdamW with polynomial decay scheduling is supported.

### Configuration Structure

```yaml
optimizer:
  type: "AdamWPolyDecay"              # REQUIRED: Must be exactly "AdamWPolyDecay"
  use_pcgrad_wrapper: bool            # OPTIONAL: Enable PCGrad wrapper (default: false)
  params:
    lr: float                         # REQUIRED: Base learning rate
    weight_decay: float               # REQUIRED: L2 regularization strength  
    adam_betas: list[float]           # REQUIRED: Adam momentum parameters [β₁, β₂]
    warmup_ratio: float               # REQUIRED: Fraction of training for warmup
    power: float                      # REQUIRED: Polynomial decay exponent
```

### Parameter Details

**Required Parameters:**
- `lr`:
  - **Type**: Float
  - **Range**: 1e-6 to 1e-2 (typically 1e-5 to 1e-4 for SegFormer)
  - **Usage**: Peak learning rate after warmup

- `weight_decay`:
  - **Type**: Float  
  - **Range**: 0.0-0.1 (typically 0.01-0.05)
  - **Usage**: L2 penalty on parameters (excludes biases and layer norms)

- `adam_betas`:
  - **Type**: List of two floats
  - **Range**: [0.9, 0.999] typical for first, [0.99, 0.9999] for second
  - **Usage**: Adam momentum parameters for gradient and squared gradient

- `warmup_ratio`:
  - **Type**: Float
  - **Range**: 0.0-0.5 (typically 0.05-0.1)
  - **Usage**: Linear warmup from 0 to `lr` over first `warmup_ratio * total_steps`

- `power`:
  - **Type**: Float  
  - **Range**: 0.5-2.0 (typically 1.0)
  - **Usage**: Polynomial decay exponent (1.0 = linear decay)

**Optional Parameters:**
- `use_pcgrad_wrapper`:
  - **Type**: Boolean
  - **Default**: false
  - **Usage**: Wraps optimizer with PCGrad for gradient conflict mitigation
  - **Compatible**: With any multi-task weighting strategy

### Learning Rate Schedule

The factory creates a polynomial decay schedule with warmup:

1. **Warmup Phase** (0 to `warmup_ratio * total_steps`):
   - Learning rate increases linearly from 0 to `lr`

2. **Decay Phase** (warmup end to total_steps):
   - Learning rate decays polynomially from `lr` to 1e-7
   - Formula: `lr * (1 - progress)^power` where progress ∈ [0,1]

### PCGrad Integration

PCGrad is a gradient surgery technique that projects conflicting gradients to reduce negative interference between tasks.

**When to Use:**
- Multi-task models with gradient conflicts
- Can be combined with any weighting strategy
- Adds computational overhead but improves convergence

**Configuration:**
```yaml
optimizer:
  type: "AdamWPolyDecay"
  use_pcgrad_wrapper: true
  params:
    lr: 6.0e-5
    weight_decay: 0.01
    adam_betas: [0.9, 0.999]
    warmup_ratio: 0.1
    power: 1.0
```

### Factory Dependencies

The optimizer factory method requires:
- **Model**: For parameter enumeration and decay grouping
- **DataLoaders**: To calculate `total_steps = len(train_loader) * epochs`
- **Trainer Config**: For `epochs` parameter

### Complete Example

```yaml
optimizer:
  type: "AdamWPolyDecay"
  use_pcgrad_wrapper: false
  params:
    lr: 6.0e-5
    weight_decay: 0.01
    adam_betas: [0.9, 0.999] 
    warmup_ratio: 0.1
    power: 1.0
```

### Parameter Recommendations

**For SegFormer Backbones:**
- `lr`: 6e-5 (B0/B1), 4e-5 (B2/B3), 2e-5 (B4/B5)
- `weight_decay`: 0.01-0.05
- `warmup_ratio`: 0.1 (10% of training)
- `power`: 1.0 (linear decay)
---


## Metrics Configuration

**Optional Section**: Controls metric calculation via `ExperimentFactory.get_metrics_calculator()`. If omitted, default values are used.

### Configuration Structure

```yaml
metrics:
  boundary_thickness: int       # OPTIONAL: BIoU boundary thickness in pixels
  ignore_index: int            # OPTIONAL: Class index to ignore in metrics
  primary_tasks: list[str]     # OPTIONAL: Tasks for H-Mean calculation (MTL only)
  use_async_storage: bool      # OPTIONAL: Enable async metrics storage
```

### Parameter Details

- `boundary_thickness`:
  - **Type**: Integer
  - **Range**: 1-10 pixels
  - **Default**: 2
  - **Usage**: Pixel width of boundary region for Boundary IoU calculation

- `ignore_index`:
  - **Type**: Integer  
  - **Default**: 255
  - **Usage**: Class index excluded from metric calculation (should match loss config)

- `primary_tasks`:
  - **Type**: List of strings
  - **Usage**: Task names used for H-Mean calculation in MTL models
  - **Requirement**: Must match task names in `task_definitions.yaml`

- `use_async_storage`:
  - **Type**: Boolean
  - **Default**: true
  - **Usage**: Enable asynchronous confusion matrix storage for better performance

### Advanced Metrics Processing

**Optional Section**: Controls Tier 2/3 advanced metrics computation.

```yaml
metrics_processor:
  enabled: bool                 # REQUIRED: Enable advanced metrics computation  
  num_cpu_workers: int         # OPTIONAL: Number of CPU worker processes
  tasks: list[str]             # OPTIONAL: Advanced metrics to compute
```

**Parameter Details:**
- `enabled`:
  - **Type**: Boolean
  - **Default**: false
  - **Usage**: Enables computationally expensive per-image metrics

- `num_cpu_workers`:
  - **Type**: Integer
  - **Range**: 1-64 (depends on system)
  - **Default**: 30
  - **Usage**: Parallel workers for advanced metric computation

- `tasks`:
  - **Type**: List of strings
  - **Default**: ["ASSD", "HD95", "PanopticQuality", "ARI"]
  - **Available**: "ASSD", "HD95", "PanopticQuality", "ARI"

### Metric Calculation Hierarchy

The factory uses a three-tier metrics system:

**Tier 1 (Real-time, GPU):**
- Confusion matrices
- IoU, mIoU per class/task
- Boundary IoU (BIoU)
- Calibration metrics (NLL, Brier, ECE)

**Tier 2/3 (Async, CPU):**
- Average Symmetric Surface Distance (ASSD)
- 95th Percentile Hausdorff Distance (HD95)  
- Panoptic Quality metrics
- Adjusted Rand Index (ARI)

### Factory Integration

- **Metrics Calculator**: Uses `CoralMTLMetrics` or `CoralMetrics` based on model type
- **Storage**: Integrates with `MetricsStorer` for persistence
- **Dependencies**: Requires task splitter and configured output directory

### Complete Examples

**Basic Metrics Configuration:**
```yaml
metrics:
  boundary_thickness: 4
  ignore_index: 0
  primary_tasks: ["genus", "health"]
  use_async_storage: true
```

**With Advanced Metrics Enabled:**
```yaml
metrics:
  boundary_thickness: 2
  ignore_index: 0
  primary_tasks: ["genus", "health"]

metrics_processor:
  enabled: true
  num_cpu_workers: 16
  tasks: ["ASSD", "HD95"]
```

**Minimal Configuration (uses defaults):**
```yaml
# metrics section can be omitted entirely for defaults
```

---

## Trainer Configuration

The trainer configuration controls the training loop execution via `ExperimentFactory.run_training()`. This section configures the `Trainer` class behavior.

### Required Parameters

```yaml
trainer:
  device: str                   # REQUIRED: Device for training ("cuda", "cpu", or "auto")
  epochs: int                   # REQUIRED: Total number of training epochs
  output_dir: str               # REQUIRED: Directory for saving checkpoints and logs
  model_selection_metric: str   # REQUIRED: Metric name for best model selection
```

### Optional Parameters

```yaml
trainer:
  gradient_accumulation_steps: int    # OPTIONAL: Steps to accumulate before optimizer update
  inference_stride: int              # OPTIONAL: Sliding window stride for validation
  inference_batch_size: int          # OPTIONAL: Batch size for validation inference
  val_frequency: int                 # OPTIONAL: Validation frequency in epochs  
  checkpoint_frequency: int          # OPTIONAL: Checkpoint saving frequency
  save_best_only: bool              # OPTIONAL: Save only best model vs all epochs
  early_stopping_patience: int      # OPTIONAL: Epochs to wait before stopping
  min_delta: float                  # OPTIONAL: Minimum improvement for early stopping
  use_mixed_precision: bool         # OPTIONAL: Enable FP16 training
  max_grad_norm: float              # OPTIONAL: Gradient clipping threshold
  log_frequency: int                # OPTIONAL: Training step logging frequency
```

### Parameter Details

**Required Parameters:**

- `device`:
  - **Type**: String
  - **Values**: "cuda", "cpu", "auto"
  - **Usage**: "auto" selects CUDA if available, otherwise CPU
  - **Factory**: Resolves device and moves model/loss to target device

- `epochs`:
  - **Type**: Integer
  - **Range**: 1-1000 (typical: 50-200)
  - **Usage**: Total training epochs, used to calculate LR schedule

- `output_dir`:
  - **Type**: String (path)
  - **Usage**: Base directory for all experiment outputs
  - **Factory**: Creates directory structure, stores checkpoints/logs/metrics

- `model_selection_metric`:
  - **Type**: String
  - **Values**: See "Available Model Selection Metrics" section
  - **Usage**: Metric name for saving best model checkpoint

**Optional Parameters:**

- `gradient_accumulation_steps`:
  - **Type**: Integer
  - **Range**: 1-64
  - **Default**: 1
  - **Usage**: Effective batch size = `data.batch_size * accumulation_steps`

- `inference_stride`:
  - **Type**: Integer or tuple of integers
  - **Default**: `patch_size // 2`  
  - **Usage**: Sliding window stride for validation inference (can be (H, W))

- `inference_batch_size`:
  - **Type**: Integer
  - **Range**: 1-64
  - **Default**: 16
  - **Usage**: Batch size for validation sliding window inference

- `val_frequency`:
  - **Type**: Integer
  - **Range**: 1-50
  - **Default**: 1
  - **Usage**: Validate every N epochs

- `checkpoint_frequency`:
  - **Type**: Integer  
  - **Default**: 10
  - **Usage**: Save checkpoint every N epochs (in addition to best model)

- `save_best_only`:
  - **Type**: Boolean
  - **Default**: true
  - **Usage**: If false, saves checkpoint every epoch

- `early_stopping_patience`:
  - **Type**: Integer
  - **Range**: 5-100
  - **Default**: 15
  - **Usage**: Stop training if no improvement for N epochs

- `min_delta`:
  - **Type**: Float
  - **Range**: 1e-6 to 1e-2
  - **Default**: 1e-4
  - **Usage**: Minimum improvement threshold for early stopping

- `use_mixed_precision`:
  - **Type**: Boolean
  - **Default**: false
  - **Usage**: Enable FP16 automatic mixed precision (reduces memory, may affect stability)

- `max_grad_norm`:
  - **Type**: Float
  - **Range**: 0.1-10.0
  - **Default**: 1.0
  - **Usage**: Gradient clipping threshold (0.0 disables)

- `log_frequency`:
  - **Type**: Integer
  - **Range**: 1-1000
  - **Default**: 100
  - **Usage**: Log training metrics every N steps

### Factory Integration

The trainer configuration is processed by the factory as follows:

1. **Device Resolution**: "auto" → CUDA if available, else CPU
2. **Path Resolution**: Relative `output_dir` resolved to absolute path
3. **Patch Size Injection**: Uses `data.patch_size` for inference configuration  
4. **Strategy Detection**: Determines gradient vs loss-based multi-task strategy
5. **Component Assembly**: Passes all components to `Trainer` class

### Complete Example

```yaml
trainer:
  device: "cuda"
  epochs: 100
  output_dir: "experiments/coral_mtl_b2_run"
  model_selection_metric: "H-Mean"
  gradient_accumulation_steps: 2
  inference_stride: 256
  inference_batch_size: 16
  val_frequency: 1
  checkpoint_frequency: 10
  save_best_only: true
  early_stopping_patience: 15
  min_delta: 1e-4
  use_mixed_precision: true
  max_grad_norm: 1.0
  log_frequency: 100
```

### Output Directory Structure

The factory creates this structure under `output_dir`:
```
output_dir/
├── best_model.pth              # Best model checkpoint
├── history.json                # Training/validation metrics per epoch
├── validation_cms.jsonl        # Per-image confusion matrices (validation)
├── loss_diagnostics.jsonl     # Loss component details per batch
└── evaluation/                 # Created by evaluator (if run)
    ├── test_cms.jsonl          # Test confusion matrices  
    ├── test_metrics_full_report.json
    └── advanced_metrics.jsonl  # If metrics_processor enabled
```

---

## Evaluator Configuration

**Optional Section**: Controls final evaluation execution via `ExperimentFactory.run_evaluation()`. If omitted, automatic settings are used.

### Configuration Structure

```yaml
evaluator:
  checkpoint_path: str          # OPTIONAL: Explicit path to model checkpoint
  output_dir: str              # OPTIONAL: Evaluation output directory  
  inference_stride: int         # OPTIONAL: Sliding window stride for test inference
  inference_batch_size: int     # OPTIONAL: Batch size for test inference
  num_visualizations: int       # OPTIONAL: Number of qualitative result images
```

### Parameter Details

**All Parameters Optional:**

- `checkpoint_path`:
  - **Type**: String (path) or null
  - **Default**: null (auto-detect)
  - **Usage**: Explicit checkpoint file path
  - **Auto-detection**: Uses `{trainer.output_dir}/best_model.pth`

- `output_dir`:
  - **Type**: String (path) or null  
  - **Default**: null (auto-create)
  - **Usage**: Directory for evaluation outputs
  - **Auto-creation**: Uses `{trainer.output_dir}/evaluation/`

- `inference_stride`:
  - **Type**: Integer or tuple
  - **Default**: 256
  - **Usage**: Sliding window stride for test set inference
  - **Range**: Typically `patch_size//4` to `patch_size//2`

- `inference_batch_size`:
  - **Type**: Integer
  - **Range**: 1-64
  - **Default**: 16
  - **Usage**: Batch size for test inference (can be larger than training)

- `num_visualizations`:
  - **Type**: Integer
  - **Range**: 0-50
  - **Default**: 8
  - **Usage**: Number of qualitative visualization images to generate

### Factory Integration

The evaluator factory method (`run_evaluation`) processes this configuration:

1. **Checkpoint Resolution**: 
   - Explicit path → use as-is
   - null → auto-detect from `trainer.output_dir/best_model.pth`
   - Relative paths → resolve to absolute

2. **Output Directory Creation**:
   - Explicit path → use as-is  
   - null → create `{trainer.output_dir}/evaluation/`

3. **Device Configuration**:
   - Uses same device as trainer configuration
   - Moves model and loss function to target device

4. **Component Assembly**:
   - Model, test dataloader, metrics calculator, loss function
   - Passes all to `Evaluator` class for execution

### Evaluation Outputs

The evaluator creates these files in the output directory:

```
evaluation/
├── test_cms.jsonl                    # Per-image confusion matrices
├── test_metrics_full_report.json     # Comprehensive metrics summary
├── qualitative_results_grid.png      # Visualization grid (if num_visualizations > 0)
└── advanced_metrics.jsonl            # Advanced metrics (if metrics_processor enabled)
```

### Example Configurations

**Minimal (all defaults):**
```yaml
evaluator: {}
```

**Explicit checkpoint:**
```yaml
evaluator:
  checkpoint_path: "experiments/my_run/epoch_50.pth"
  output_dir: "results/final_evaluation"
  num_visualizations: 12
```

**Custom inference settings:**
```yaml
evaluator:
  inference_stride: 128
  inference_batch_size: 32
  num_visualizations: 16
```

**Programmatic Usage:**
```python
# Can also be called programmatically with override
factory = ExperimentFactory("config.yaml")
results = factory.run_evaluation(
    checkpoint_path="custom/path/model.pth"
)
```

---

## Hyperparameter Study Configuration

**Optional Section**: Controls Optuna-based hyperparameter optimization via `ExperimentFactory.run_hyperparameter_study()`.

### Configuration Structure

```yaml
study:
  name: str                     # REQUIRED: Unique study identifier
  storage: str                  # REQUIRED: Optuna storage backend URL
  direction: str                # REQUIRED: "maximize" or "minimize"  
  n_trials: int                # REQUIRED: Total number of optimization trials
  config_path: str             # REQUIRED: Path to search space definition
  pruner:                      # OPTIONAL: Early trial termination
    type: str                  # Pruner algorithm
    params: dict               # Pruner-specific parameters
```

### Parameter Details

**Required Parameters:**

- `name`:
  - **Type**: String
  - **Usage**: Unique identifier for the study (allows resuming)
  - **Example**: "coral_mtl_lr_alpha_study"

- `storage`:
  - **Type**: String (URL)
  - **Format**: `"sqlite:///path/to/study.db"` 
  - **Usage**: Persistent storage for trial results and resumption

- `direction`:
  - **Type**: String
  - **Values**: "maximize" or "minimize"
  - **Usage**: Optimization direction for `trainer.model_selection_metric`

- `n_trials`:
  - **Type**: Integer
  - **Range**: 10-1000+
  - **Usage**: Total trials to execute (can resume if interrupted)

- `config_path`:
  - **Type**: String (path)
  - **Usage**: Path to search space YAML file
  - **Resolution**: Relative paths resolved from main config location

**Optional Parameters:**

- `pruner.type`:
  - **Type**: String
  - **Values**: "MedianPruner" (only supported type currently)
  - **Usage**: Early stopping for unpromising trials

- `pruner.params`:
  - **Type**: Dictionary
  - **Available**: `n_startup_trials`, `n_warmup_steps`, `interval_steps`

### Search Space Configuration

Create a separate YAML file defining the hyperparameter search space:

```yaml
# search_space.yaml - Nested parameter paths with Optuna suggest configs
optimizer.params.lr:
  type: "float"
  params:
    name: "lr"
    low: 1.0e-5
    high: 1.0e-3
    log: true

model.params.decoder_channel:
  type: "int"
  params:
    name: "decoder_channel"
    low: 128
    high: 512
    step: 64

loss.params.focal_gamma:
  type: "float"
  params:
    name: "focal_gamma"
    low: 1.0
    high: 5.0

augmentations.rotation_degrees:
  type: "categorical"
  params:
    name: "rotation_degrees"
    choices: [0, 15, 30, 45]

loss.weighting_strategy.type:
  type: "categorical"
  params:
    name: "weighting_strategy"
    choices: ["Uncertainty", "NashMTL", "IMGrad"]
```

### Supported Parameter Types

1. **Float Parameters:**
   - `type: "float"`
   - `params: {name, low, high, log: optional}`
   - `log: true` for log-uniform distribution

2. **Integer Parameters:**
   - `type: "int"`
   - `params: {name, low, high, step: optional}`

3. **Categorical Parameters:**
   - `type: "categorical"`
   - `params: {name, choices: list}`

### Factory Integration

The study workflow:

1. **Study Setup**: Creates/loads Optuna study with specified storage
2. **Objective Function**: For each trial:
   - Modifies config with sampled hyperparameters
   - Creates new factory instance with trial config
   - Runs complete training workflow
   - Returns final validation metric value
3. **Best Trial**: Reports optimal hyperparameters after completion

### Complete Example

```yaml
study:
  name: "coral_mtl_comprehensive_study"
  storage: "sqlite:///experiments/comprehensive_hyperopt.db"
  direction: "maximize"
  n_trials: 100
  config_path: "configs/studies/comprehensive_search_space.yaml"
  pruner:
    type: "MedianPruner"
    params:
      n_startup_trials: 10
      n_warmup_steps: 15
      interval_steps: 5
```

### Usage Pattern

```python
# Programmatic execution
factory = ExperimentFactory("config_with_study.yaml")
factory.run_hyperparameter_study()  # Runs complete optimization
```

### Output

Study results are persisted in the SQLite database and can be analyzed using Optuna's visualization tools or accessed programmatically for post-hoc analysis.

---

## Available Model Selection Metrics

The `model_selection_metric` parameter in the trainer configuration determines which metric is used for saving the best model checkpoint. Available metrics depend on the model type and current implementation:

### Global Metrics
- `global.mIoU` - Mean Intersection over Union across all classes
- `global.BIoU` - Boundary Intersection over Union across all classes
- `global.classification_error` - TIDE classification error rate
- `global.background_error` - TIDE background error rate  
- `global.missed_error` - TIDE missed detection error rate

### Per-Task Metrics (format: `tasks.{task_name}.{level}.{metric}`)

#### Levels
- `ungrouped` - Fine-grained class level
- `grouped` - Coarse-grained grouped classes (if task supports grouping)

#### Metrics
- `mIoU` - Mean Intersection over Union
- `BIoU` - Boundary Intersection over Union

#### Examples
- `tasks.genus.ungrouped.mIoU`
- `tasks.genus.ungrouped.BIoU`  
- `tasks.health.grouped.mIoU`
- `tasks.health.grouped.BIoU`

### Special MTL Metrics
- `H-Mean` - Harmonic mean of primary task mIoUs (for MTL models)

### Recommended Settings

**For MTL Models**:
```yaml
trainer:
  model_selection_metric: "H-Mean"  # Balances primary tasks
```

**For Baseline Models**:
```yaml
trainer:
  model_selection_metric: "global.BIoU"  # Now available - boundary-aware performance
```

**For Specific Task Focus**:
```yaml  
trainer:
  model_selection_metric: "tasks.genus.ungrouped.mIoU"  # Focus on genus classification
```

---

## Configuration Examples

### Minimal MTL Configuration

```yaml
# Minimum viable MTL configuration
model:
  type: "CoralMTL"
  params:
    backbone: "nvidia/mit-b2"
    decoder_channel: 256
    attention_dim: 128
  tasks:
    primary: ["genus", "health"]
    auxiliary: ["fish", "human_artifacts", "substrate"]

data:
  dataset_name: "./dataset/coralscapes/"
  task_definitions_path: "configs/task_definitions.yaml"
  batch_size: 4
  num_workers: 4
  patch_size: 512

loss:
  type: "CompositeHierarchical"
  params:
    hybrid_alpha: 0.5
    focal_gamma: 2.0
    ignore_index: 0

optimizer:
  type: "AdamWPolyDecay"
  params:
    lr: 6.0e-5
    weight_decay: 0.01
    adam_betas: [0.9, 0.999]
    warmup_ratio: 0.1
    power: 1.0

trainer:
  device: "cuda"
  epochs: 100
  output_dir: "experiments/mtl_run"
  model_selection_metric: "H-Mean"
```

### Minimal Baseline Configuration

```yaml
# Minimum viable baseline configuration
model:
  type: "SegFormerBaseline"
  params:
    backbone: "nvidia/mit-b2"
    decoder_channel: 256
    num_classes: 40

data:
  dataset_name: "./dataset/coralscapes/"
  task_definitions_path: "configs/task_definitions.yaml"
  batch_size: 4
  num_workers: 4
  patch_size: 512

loss:
  type: "HybridLoss"
  params:
    primary_loss_type: "focal"
    hybrid_alpha: 0.5
    focal_gamma: 2.0
    dice_smooth: 1.0
    ignore_index: 0

optimizer:
  type: "AdamWPolyDecay"
  params:
    lr: 6.0e-5
    weight_decay: 0.01
    adam_betas: [0.9, 0.999]
    warmup_ratio: 0.1
    power: 1.0

trainer:
  device: "cuda"
  epochs: 100
  output_dir: "experiments/baseline_run"
  model_selection_metric: "global.mIoU"
```

### Production MTL Configuration with Advanced Features

```yaml
# Production-ready MTL configuration with all features
model:
  type: "CoralMTL"
  params:
    backbone: "nvidia/mit-b2"
    decoder_channel: 256
    attention_dim: 128
    encoder_weights: "imagenet"
    encoder_depth: 5
  tasks:
    primary: ["genus", "health"]
    auxiliary: ["fish", "human_artifacts", "substrate", "background", "biota"]

data:
  dataset_name: "EPFL-ECEO/coralscapes"
  pds_train_path: "./dataset/processed/pds_patches/"
  data_root_path: "./dataset/raw/coralscapes/"
  task_definitions_path: "configs/task_definitions.yaml"
  batch_size: 4
  num_workers: 8
  patch_size: 512
  ignore_index: 0

augmentations:
  crop_scale: [0.5, 1.0]
  rotation_degrees: 15
  jitter_params:
    brightness: 0.2
    contrast: 0.2
    saturation: 0.2
    hue: 0.1

loss:
  type: "CompositeHierarchical"
  params:
    w_consistency: 0.1
    hybrid_alpha: 0.5
    focal_gamma: 2.0
    ignore_index: 0
  weighting_strategy:
    type: "NashMTL"
    params:
      update_frequency: 10
      solver: "auto"
      max_norm: 1.0

optimizer:
  type: "AdamWPolyDecay"
  use_pcgrad_wrapper: true
  params:
    lr: 6.0e-5
    weight_decay: 0.01
    adam_betas: [0.9, 0.999]
    warmup_ratio: 0.1
    power: 1.0

metrics:
  boundary_thickness: 2
  ignore_index: 0
  primary_tasks: ["genus", "health"]
  use_async_storage: true

metrics_processor:
  enabled: true
  num_cpu_workers: 16
  tasks: ["ASSD", "HD95", "PanopticQuality"]

trainer:
  device: "auto"
  epochs: 150
  output_dir: "experiments/production_mtl_run"
  model_selection_metric: "H-Mean"
  gradient_accumulation_steps: 2
  inference_stride: 256
  inference_batch_size: 16
  val_frequency: 1
  checkpoint_frequency: 10
  save_best_only: true
  early_stopping_patience: 20
  min_delta: 1e-4
  use_mixed_precision: true
  max_grad_norm: 1.0
  log_frequency: 50

evaluator:
  inference_stride: 128
  inference_batch_size: 32
  num_visualizations: 16

study:
  name: "production_coral_mtl_study"
  storage: "sqlite:///experiments/production_hyperopt.db"
  direction: "maximize"
  n_trials: 100
  config_path: "configs/advanced_search_space.yaml"
  pruner:
    type: "MedianPruner"
    params:
      n_startup_trials: 10
      n_warmup_steps: 15
      interval_steps: 5
```

### Baseline with Advanced Optimization

```yaml
# High-performance baseline configuration
model:
  type: "SegFormerBaseline"
  params:
    backbone: "nvidia/mit-b3"
    decoder_channel: 512
    num_classes: 40
    encoder_weights: "imagenet"

data:
  dataset_name: "EPFL-ECEO/coralscapes"
  pds_train_path: "./dataset/processed/pds_patches/"
  task_definitions_path: "configs/task_definitions.yaml"
  batch_size: 2
  num_workers: 8
  patch_size: 768
  ignore_index: 0

augmentations:
  crop_scale: [0.6, 1.0]
  rotation_degrees: 30
  jitter_params:
    brightness: 0.3
    contrast: 0.3
    saturation: 0.3
    hue: 0.15

loss:
  type: "HybridLoss"
  params:
    primary_loss_type: "focal"
    hybrid_alpha: 0.7
    focal_gamma: 3.0
    dice_smooth: 1.0
    ignore_index: 0

optimizer:
  type: "AdamWPolyDecay"
  params:
    lr: 4.0e-5
    weight_decay: 0.05
    adam_betas: [0.9, 0.999]
    warmup_ratio: 0.15
    power: 0.9

metrics:
  boundary_thickness: 3
  ignore_index: 0
  primary_tasks: ["genus", "health"]

trainer:
  device: "cuda"
  epochs: 200
  output_dir: "experiments/advanced_baseline"
  model_selection_metric: "global.BIoU"
  gradient_accumulation_steps: 4
  inference_stride: 192
  inference_batch_size: 8
  use_mixed_precision: true
  max_grad_norm: 0.5
  early_stopping_patience: 30
```

---

## Configuration Validation

### Common Configuration Errors

1. **Missing Required Sections**
   ```yaml
   # ❌ Missing required sections
   model:
     type: "CoralMTL"
   # Missing: data, loss, optimizer, trainer
   ```

2. **Invalid Model Selection Metric**
   ```yaml
   # ❌ This metric is not currently implemented
   trainer:
     model_selection_metric: "global.BIoU"  # Not available
   ```

3. **Mismatched ignore_index Values**
   ```yaml
   # ❌ Inconsistent ignore values
   data:
     ignore_index: 0
   loss:
     params:
       ignore_index: 255  # Should match data.ignore_index
   ```

4. **Invalid Hyperparameter Paths**
   ```yaml
   # ❌ Invalid nested path in search space
   invalid.path.that.does.not.exist:
     type: "float"
     params: {name: "test", low: 0.0, high: 1.0}
   ```

### Validation Checklist

Before running experiments, verify:

- [ ] All required sections are present
- [ ] Model type matches task configuration (MTL vs baseline)
- [ ] Task definitions file exists and is valid
- [ ] Dataset paths are accessible
- [ ] Model selection metric is implemented and available
- [ ] ignore_index values are consistent across sections
- [ ] Hyperparameter search space paths are valid (if using study)
- [ ] Output directories are writable

### Quick Validation Script

```python
import yaml
from coral_mtl.ExperimentFactory import ExperimentFactory

def validate_config(config_path):
    """Quick configuration validation."""
    try:
        factory = ExperimentFactory(config_path=config_path)
        print("✅ Configuration is valid")
        return True
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

    ```

    ---

    ## Reproducibility Pack: Multi-Task Optimization Strategy Templates

    Below are copy/paste fragments demonstrating the major multi-task optimization strategies. Insert them under the existing `loss` (and `optimizer` when needed) sections.

    ### 1. Baseline: Uncertainty Weighting (Single Backward Pass)
    ```yaml
    loss:
      type: "CompositeHierarchical"
      params:
        w_consistency: 0.1
        hybrid_alpha: 0.5
        focal_gamma: 2.0
        ignore_index: 0
      weighting_strategy:
        type: "Uncertainty"
    ```

    ### 2. Nash-MTL (Fairness / Scale Invariance)
    Higher computational cost; reuse weights with `update_frequency` to reduce overhead.
    ```yaml
    loss:
      type: "CompositeHierarchical"
      params:
        w_consistency: 0.1
        hybrid_alpha: 0.5
        focal_gamma: 2.0
        ignore_index: 0
      weighting_strategy:
        type: "NashMTL"
        params:
          solver: "ccp"          # or "iterative" / "auto"
          update_frequency: 10    # reuse computed weights for N steps
          max_norm: 0.0           # optional gradient clipping
    ```

    ### 3. IMGrad (Magnitude Imbalance Focused)
    ```yaml
    loss:
      type: "CompositeHierarchical"
      params:
        w_consistency: 0.1
        hybrid_alpha: 0.5
        focal_gamma: 2.0
        ignore_index: 0
      weighting_strategy:
        type: "IMGrad"
        params:
          solver: "qp"           # or "pgd" / "auto"
    ```

    ### 4. Uncertainty + PCGrad (Conflict Mitigation Layer)
    Retain simple weighting; add gradient surgery via optimizer wrapper.
    ```yaml
    loss:
      type: "CompositeHierarchical"
      params:
        w_consistency: 0.1
        hybrid_alpha: 0.5
        focal_gamma: 2.0
        ignore_index: 0
      weighting_strategy:
        type: "Uncertainty"

    optimizer:
      type: "AdamWPolyDecay"
      use_pcgrad_wrapper: true
      params:
        lr: 6.0e-5
        weight_decay: 0.01
        adam_betas: [0.9, 0.999]
        warmup_ratio: 0.1
        power: 1.0
    ```

    ### 5. Strategy Selection Heuristic (Comment Only)
    ```yaml
    # Heuristic:
    # 1. Start with Uncertainty
    # 2. If large gradient_norm disparity -> NashMTL or IMGrad
    # 3. If many negative cosine similarities -> enable PCGrad
    # 4. If both -> prefer NashMTL; compare with IMGrad & Uncertainty+PCGrad
    ```

    ### 6. Where to Look for Diagnostics
    - Gradient norms, cosine similarity, update norms: `validation/loss_diagnostics.jsonl`
    - Epoch summaries: `history.json`
    - Strategy-specific fields: `imgrad_cos_theta`, `task_weights`, `log_variances`, `nash_objective`.

    For full theoretical context and formulations see:
    - `project_specification/loss_and_optim_specification.md`
def validate_config(config_path):
    """Quick configuration validation using ExperimentFactory."""
    try:
        factory = ExperimentFactory(config_path=config_path)
        print(f"✅ Configuration '{config_path}' is valid")
        
        # Test component instantiation
        model = factory.get_model()
        print(f"   Model: {model.__class__.__name__}")
        
        dataloaders = factory.get_dataloaders()
        print(f"   Dataloaders: {list(dataloaders.keys())}")
        
        loss_fn = factory.get_loss_function() 
        print(f"   Loss: {loss_fn.__class__.__name__}")
        
        return True
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

# Usage
validate_config("path/to/your/config.yaml")
```

---

## Notes and Known Issues

1. **Enhanced Global Metrics**: The global BIoU and TIDE error metrics are now fully implemented and available for model selection.

2. **Task Definitions Dependency**: All configurations require a valid `task_definitions.yaml` file, even for baseline models that don't use multi-task learning.

3. **Memory Considerations**: 
   - Larger batch sizes and patch sizes require more GPU memory
   - Mixed precision training can reduce memory usage
   - Consider gradient accumulation for effective larger batch sizes

4. **Device Auto-Detection**: Using `device: "auto"` will automatically select CUDA if available, otherwise CPU.

5. **Path Handling**: Both absolute and relative paths are supported, with relative paths resolved from the config file location.

6. **Enhanced Validation Data Storage**: 
   - Per-image confusion matrices are now stored with epoch information for detailed analysis across training runs
   - Predicted masks are stored alongside confusion matrices for complete reproducibility
   - **Validation format**: `{"image_id": str, "epoch": int, "confusion_matrices": {...}, "predicted_masks": {...}}`
   - **Test format**: `{"image_id": str, "confusion_matrices": {...}, "predicted_masks": {...}}`
   - BIoU metrics can now be fully reconstructed from stored data without requiring model re-inference The format is:
   ```json
   {"image_id": "img_001", "epoch": 15, "confusion_matrices": {"task1": [[...]], "global": [[...]]}}
   ```

7. **Metrics Derivability**: Most metrics (mIoU, precision, recall, F1, TIDE errors) can be reconstructed from stored confusion matrices. However, BIoU metrics require the original predictions and cannot be derived from CMs alone.

This guide covers all configurable parameters in the coral segmentation experiment system. For additional examples and advanced configurations, refer to the `configs/` directory in the project repository.

---

## Stored Data Format

The experiment system automatically stores detailed per-image data during validation and testing for comprehensive analysis and reproducibility.

### Validation Data Storage (`validation_cms.jsonl`)

Each line in the JSONL file represents one image from one validation epoch:

```json
{
  "image_id": "image_001",
  "epoch": 5,
  "confusion_matrices": {
    "genus": [[10, 2], [1, 15]],
    "health": [[8, 3], [2, 12]],
    "global": [[20, 5], [3, 27]]
  },
  "predicted_masks": {
    "genus": [[0, 1, 1, 0], [1, 1, 0, 0]],
    "health": [[0, 1, 0, 1], [1, 0, 1, 0]],
    "global": [[0, 1, 1, 1], [1, 1, 1, 0]]
  }
}
```

### Test Data Storage (`test_cms.jsonl`)

Similar to validation but without epoch information:

```json
{
  "image_id": "test_image_001", 
  "confusion_matrices": {
    "genus": [[12, 1], [2, 18]],
    "health": [[9, 2], [1, 14]],
    "global": [[21, 3], [3, 32]]
  },
  "predicted_masks": {
    "genus": [[0, 1, 1, 0], [1, 1, 0, 0]],
    "health": [[0, 1, 0, 1], [1, 0, 1, 0]], 
    "global": [[0, 1, 1, 1], [1, 1, 1, 0]]
  }
}
```

### Data Usage and Analysis

The stored data enables:

1. **Complete Metrics Reconstruction**: All standard metrics (IoU, precision, recall, F1, TIDE errors) can be recalculated from confusion matrices
2. **Boundary Metrics Calculation**: BIoU can be computed from stored predicted masks without model re-inference
3. **Cross-Epoch Analysis**: Track per-image performance evolution during training
4. **Error Analysis**: Identify consistently misclassified images and regions
5. **Reproducibility**: Full experiment state can be restored for detailed analysis