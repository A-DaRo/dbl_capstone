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
11. [Visualization Configuration](#visualization-configuration)
12. [Available Model Selection Metrics](#available-model-selection-metrics)
13. [Configuration Examples](#configuration-examples)
14. [Configuration Validation](#configuration-validation)

---

## Configuration Structure

All configurations are defined in YAML format and contain the following top-level sections:

```yaml
model:          # Model architecture and parameters
data:           # Dataset and dataloader configuration  
augmentations:  # Data augmentation settings
loss:           # Loss function configuration
optimizer:      # Optimizer and scheduler settings
metrics:        # Metrics calculation parameters
trainer:        # Training loop configuration
evaluator:      # Evaluation settings
study:          # Optuna hyperparameter optimization (optional)
visualizer:     # Plotting and visualization (optional)
```

**Required Sections**: `model`, `data`, `loss`, `optimizer`, `trainer`  
**Optional Sections**: `augmentations`, `metrics`, `evaluator`, `study`, `visualizer`

---

## Model Configuration

### Required Parameters

```yaml
model:
  type: str  # Model architecture type
  params:    # Model-specific parameters
```

### Model Types

#### 1. CoralMTL (Multi-Task Learning)

```yaml
model:
  type: "CoralMTL"
  params:
    backbone: str           # REQUIRED: SegFormer backbone (e.g., "nvidia/mit-b2")
    decoder_channel: int    # REQUIRED: Decoder channel dimension (default: 256)
    attention_dim: int      # REQUIRED: Cross-attention dimension (default: 128)
  
  tasks:
    primary: list[str]      # REQUIRED: Primary task names for main decoders
    auxiliary: list[str]    # REQUIRED: Auxiliary task names for additional decoders
```

**Example**:
```yaml
model:
  type: "CoralMTL"
  params:
    backbone: "nvidia/mit-b2"
    decoder_channel: 256
    attention_dim: 128
  tasks:
    primary: ["genus", "health"]
    auxiliary: ["fish", "human_artifacts", "substrate", "background", "biota"]
```

#### 2. SegFormerBaseline (Single-Task)

```yaml
model:
  type: "SegFormerBaseline"
  params:
    backbone: str           # REQUIRED: SegFormer backbone (e.g., "nvidia/mit-b2")
    decoder_channel: int    # REQUIRED: Decoder channel dimension (default: 256)
    num_classes: int        # REQUIRED: Total number of output classes
```

**Example**:
```yaml
model:
  type: "SegFormerBaseline"
  params:
    backbone: "nvidia/mit-b2"
    decoder_channel: 256
    num_classes: 40
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

### Required Parameters

```yaml
data:
  dataset_name: str              # REQUIRED: Local path or HuggingFace dataset ID
  task_definitions_path: str     # REQUIRED: Path to task definitions YAML
  batch_size: int               # REQUIRED: Training batch size
  num_workers: int              # REQUIRED: DataLoader worker processes
  patch_size: int               # REQUIRED: Image patch size for training
```

### Optional Parameters

```yaml
data:
  pds_train_path: str           # OPTIONAL: Path to PDS-sampled training patches
  data_root_path: str           # OPTIONAL: Path to raw dataset (fallback)
  ignore_index: int             # OPTIONAL: Class index to ignore (default: 255)
```

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

### Dataset Sources

1. **Local Path**: Absolute or relative path to local dataset
2. **HuggingFace Hub**: Dataset ID (e.g., `"EPFL-ECEO/coralscapes"`)

---

## Augmentations Configuration

**Optional Section**: If omitted, no augmentations are applied.

```yaml
augmentations:
  crop_scale: list[float]       # OPTIONAL: RandomResizedCrop scale range (default: [0.5, 1.0])
  rotation_degrees: int         # OPTIONAL: Random rotation range in degrees (default: 15)
  jitter_params:               # OPTIONAL: ColorJitter parameters
    brightness: float          # OPTIONAL: Brightness variation (default: 0.2)
    contrast: float            # OPTIONAL: Contrast variation (default: 0.2)  
    saturation: float          # OPTIONAL: Saturation variation (default: 0.2)
    hue: float                # OPTIONAL: Hue variation (default: 0.1)
```

**Example**:
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

---

## Loss Configuration

### Baseline Loss (SegFormerBaseline)

```yaml
loss:
  type: "HybridLoss"
  params:
    primary_loss_type: str      # REQUIRED: "focal" or "cross_entropy"
    hybrid_alpha: float         # REQUIRED: Weight for primary loss component (0.0-1.0)
    focal_gamma: float          # OPTIONAL: Focal loss gamma (required if primary_loss_type="focal")
    dice_smooth: float          # OPTIONAL: Dice loss smoothing factor (default: 1.0)
    ignore_index: int           # REQUIRED: Class index to ignore in loss calculation
```

**Examples**:
```yaml
# MTL Loss
loss:
  type: "CompositeHierarchical"
  params:
    w_consistency: 0.1
    hybrid_alpha: 0.5
    focal_gamma: 2.0
    ignore_index: 0

# Baseline Loss  
loss:
  type: "HybridLoss"
  params:
    primary_loss_type: "focal"
    hybrid_alpha: 0.5
    focal_gamma: 2.0
    dice_smooth: 1.0
    ignore_index: 0
```


#### Loss Weighting & Gradient Combination Strategies

For `CoralMTL` models, you must specify a strategy to balance the contributions of different tasks. This is configured under `loss.weighting_strategy`.

```yaml
loss:
  type: "CompositeHierarchical"
  # ... other params
  weighting_strategy:
    type: str         # REQUIRED: The name of the strategy to use.
    params: dict      # OPTIONAL: Parameters for the chosen strategy.
```

**Available Strategies:**

1.  **`Uncertainty` (Default, Loss-based)**
    *   Balances tasks by learning their homoscedastic uncertainty. Computationally cheap.
    *   **Params:** None.

    ```yaml
    weighting_strategy:
      type: "Uncertainty"
    ```

2.  **`IMGrad` (Gradient-based)**
    *   An advanced strategy that adaptively balances between mitigating gradient imbalance (via MGDA) and ensuring Pareto improvement (via CAGrad). Requires multiple backward passes per step.
    *   **Params:**
        *   `solver`: `str` - `'auto'`, `'qp'`, or `'pgd'`. Defaults to `'auto'`, which uses the `cvxopt` QP solver if installed, otherwise falls back to a PGD approximation.
    
    ```yaml
    weighting_strategy:
      type: "IMGrad"
      params:
        solver: "auto"
    ```

3.  **`NashMTL` (Gradient-based)**
    *   A state-of-the-art strategy that frames task balancing as a bargaining game to find a proportionally fair update. It is invariant to loss scales but is computationally expensive.
    *   **Params:**
        *   `update_frequency`: `int` - How often to recompute the Nash weights (e.g., `10`). Higher values are much faster but may use slightly stale weights. Default: `1`.
        *   `solver`: `str` - `'auto'`, `'ccp'`, or `'iterative'`. Defaults to `'auto'`, which uses the `cvxpy` solver if installed.
        *   `max_norm`: `float` - Optional gradient clipping applied to the final update vector. Default: `0.0` (disabled).

    ```yaml
    weighting_strategy:
      type: "NashMTL"
      params:
        update_frequency: 10
        solver: "auto"
        max_norm: 1.0
    ```

---

## Optimizer Configuration

#### AdamWPolyDecay (Default)

```yaml
optimizer:
  type: "AdamWPolyDecay"
  use_pcgrad_wrapper: bool      # OPTIONAL: Set to true to enable PCGrad. Default: false.
  params:
    lr: float                   # REQUIRED: Learning rate (e.g., 6.0e-5)
    weight_decay: float         # REQUIRED: L2 regularization strength (e.g., 0.01)
    adam_betas: list[float]     # REQUIRED: Adam beta parameters (e.g., [0.9, 0.999])
    warmup_ratio: float         # REQUIRED: Fraction of training for warmup (e.g., 0.1)
    power: float                # REQUIRED: Polynomial decay power (e.g., 1.0)
```

**Example with PCGrad:**

PCGrad is an orthogonal technique that mitigates gradient conflict. It can be combined with any loss or weighting strategy.

```yaml
optimizer:
  type: "AdamWPolyDecay"
  use_pcgrad_wrapper: true  # Enable PCGrad
  params:
    lr: 6.0e-5
    # ... other params
```
---


## Metrics Configuration

**Optional Section**: If omitted, default metrics are calculated.

```yaml
metrics:
  boundary_thickness: int       # OPTIONAL: BIoU boundary thickness in pixels (default: 2)
  ignore_index: int            # OPTIONAL: Class index to ignore (should match loss)
  primary_tasks: list[str]     # OPTIONAL: Tasks for H-Mean calculation (MTL only)
```

**Example**:
```yaml
metrics:
  boundary_thickness: 4
  ignore_index: 0
  primary_tasks: ["genus", "health"]
```

---

## Trainer Configuration

### Required Parameters

```yaml
trainer:
  device: str                   # REQUIRED: "cuda", "cpu", or "auto"
  epochs: int                   # REQUIRED: Number of training epochs
  output_dir: str               # REQUIRED: Directory for saving outputs
  model_selection_metric: str   # REQUIRED: Metric for best model selection
```

### Optional Parameters

```yaml
trainer:
  gradient_accumulation_steps: int    # OPTIONAL: Gradient accumulation (default: 1)
  inference_stride: int              # OPTIONAL: Sliding window stride (default: patch_size/2)
  inference_batch_size: int          # OPTIONAL: Validation batch size (default: 16)
  val_frequency: int                 # OPTIONAL: Validation every N epochs (default: 1)
  checkpoint_frequency: int          # OPTIONAL: Save checkpoint every N epochs (default: 10)
  save_best_only: bool              # OPTIONAL: Save only best model (default: true)
  early_stopping_patience: int      # OPTIONAL: Early stopping patience (default: 15)
  min_delta: float                  # OPTIONAL: Minimum improvement threshold (default: 1e-4)
  use_mixed_precision: bool         # OPTIONAL: Use FP16 training (default: false)
  max_grad_norm: float              # OPTIONAL: Gradient clipping norm (default: 1.0)
  log_frequency: int                # OPTIONAL: Log every N steps (default: 100)
```

**Complete Example**:
```yaml
trainer:
  device: "cuda"
  epochs: 100
  output_dir: "experiments/baseline_comparisons/coral_baseline_b2_run"
  gradient_accumulation_steps: 2
  model_selection_metric: "global.mIoU"
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

---

## Evaluator Configuration

**Optional Section**: If omitted, automatic evaluation settings are used.

```yaml
evaluator:
  checkpoint_path: str          # OPTIONAL: Path to specific checkpoint (null for auto-detect)
  output_dir: str              # OPTIONAL: Evaluation output directory (null for auto)
  num_visualizations: int       # OPTIONAL: Number of qualitative samples (default: 8)
```

**Example**:
```yaml
evaluator:
  checkpoint_path: null
  output_dir: null
  num_visualizations: 8
```

---

## Hyperparameter Study Configuration

**Optional Section**: Only required for hyperparameter optimization.

### Required Study Parameters

```yaml
study:
  name: str                     # REQUIRED: Study name for identification
  storage: str                  # REQUIRED: Storage backend (e.g., "sqlite:///study.db")
  direction: str                # REQUIRED: "maximize" or "minimize"
  n_trials: int                # REQUIRED: Number of optimization trials
  config_path: str             # REQUIRED: Path to search space configuration
```

### Optional Study Parameters

```yaml
study:
  pruner:                      # OPTIONAL: Early trial termination
    type: str                  # Pruner type (e.g., "MedianPruner")
    params:                    # Pruner-specific parameters
      n_startup_trials: int    # Trials before pruning starts
      n_warmup_steps: int      # Steps before pruning evaluation
      interval_steps: int      # Pruning evaluation interval
```

### Search Space Configuration

The search space is defined in a separate YAML file:

```yaml
# search_space.yaml
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
```

### Complete Study Example

```yaml
study:
  name: "coral_hyperopt_study"
  storage: "sqlite:///experiments/hyperopt.db"
  direction: "maximize"
  n_trials: 50
  config_path: "configs/search_space.yaml"
  pruner:
    type: "MedianPruner"
    params:
      n_startup_trials: 5
      n_warmup_steps: 10
      interval_steps: 5
```

### Supported Hyperparameter Types

1. **Float**: `type: "float"` with `low`, `high`, `log` (optional)
2. **Integer**: `type: "int"` with `low`, `high`, `step` (optional)
3. **Categorical**: `type: "categorical"` with `choices` list

---

## Visualization Configuration

**Optional Section**: For customizing plots and visualizations.

```yaml
visualizer:
  style: str                    # OPTIONAL: Matplotlib style (default: "seaborn-v0_8-whitegrid")
  dpi: int                     # OPTIONAL: Figure DPI (default: 300)
  figure_size: list[int]       # OPTIONAL: Figure size [width, height] (default: [12, 8])
  color_palette: str           # OPTIONAL: Color palette (default: "viridis")
  show_plots: bool             # OPTIONAL: Show interactive plots (default: false)
  plot_format: str             # OPTIONAL: File format for saved plots (default: "png")
```

**Example**:
```yaml
visualizer:
  style: "seaborn-v0_8-whitegrid"
  dpi: 300
  figure_size: [12, 8]
  color_palette: "viridis"
  show_plots: false
  plot_format: "png"
```

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
model:
  type: "CoralMTL"
  params:
    backbone: "nvidia/mit-b2"
    decoder_channel: 256
    attention_dim: 128
  tasks:
    primary: ["genus", "health"]
    auxiliary: ["fish", "human_artifacts", "substrate", "background", "biota"]

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

### Full-Featured Configuration with Hyperparameter Study

```yaml
model:
  type: "CoralMTL"
  params:
    backbone: "nvidia/mit-b2"
    decoder_channel: 256
    attention_dim: 128
  tasks:
    primary: ["genus", "health"]
    auxiliary: ["fish", "human_artifacts", "substrate", "background", "biota"]

data:
  dataset_name: "EPFL-ECEO/coralscapes"
  pds_train_path: "./data/processed/pds_patches/"
  data_root_path: "./data/raw/coralscapes_raw/"
  task_definitions_path: "configs/task_definitions.yaml"
  batch_size: 4
  num_workers: 4
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

optimizer:
  type: "AdamWPolyDecay"
  params:
    lr: 6.0e-5
    weight_decay: 0.01
    adam_betas: [0.9, 0.999]
    warmup_ratio: 0.1
    power: 1.0

metrics:
  primary_tasks: ["genus", "health"]
  boundary_thickness: 2
  ignore_index: 0

trainer:
  epochs: 100
  device: "auto"
  output_dir: "experiments/mtl_optimized"
  model_selection_metric: "H-Mean"
  metric_mode: "max"
  val_frequency: 1
  checkpoint_frequency: 10
  save_best_only: true
  early_stopping_patience: 15
  min_delta: 1e-4
  inference_stride: 256
  inference_batch_size: 16
  use_mixed_precision: true
  max_grad_norm: 1.0
  log_frequency: 100

evaluator:
  checkpoint_path: null
  output_dir: null
  num_visualizations: 8

study:
  name: "coral_mtl_hyperopt"
  storage: "sqlite:///experiments/hyperopt.db"
  direction: "maximize"
  n_trials: 50
  config_path: "configs/search_space.yaml"
  pruner:
    type: "MedianPruner"
    params:
      n_startup_trials: 5
      n_warmup_steps: 10
      interval_steps: 5

visualizer:
  style: "seaborn-v0_8-whitegrid"
  dpi: 300
  figure_size: [12, 8]
  color_palette: "viridis"
  show_plots: false
  plot_format: "png"
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
    - Section "Choosing a Multi-Task Strategy" in `project_specification/theorethical_specification.md`


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