# Technical Specification

This document provides a detailed breakdown of the technical implementation of the Coral-MTL project, including class structures, method signatures, and key configuration details. It is intended as a guide for developers working on the codebase. For the theoretical background and design justifications, please refer to the [**Theoretical Specification**](./theorethical_specification.md).

---

### 1. Project Codebase Structure

The project follows a standardized structure to ensure modularity, reproducibility, and a clear separation of concerns between library code, scripts, configurations, and results.

```
coral-mtl-project/
├── configs/                  # YAML configuration files for experiments
├── data/                     # Raw and processed datasets
├── notebooks/                # Jupyter notebooks for exploration and analysis
├── scripts/                  # Standalone utility scripts (e.g., data preparation)
├── src/
│   └── coral_mtl/            # The core, installable Python package
│       ├── __init__.py
│       ├── ExperimentFactory.py  # Main factory for creating model, optimizer, etc.
│       ├── data/               # Data loading and processing
│       ├── model/              # Model architecture components
│       ├── engine/             # Training, evaluation, loss, and metric logic
│       └── utils/              # Helper utilities (e.g., visualization)
├── tests/                    # Unit and integration tests
├── experiments/              # Output directory for all training artifacts
├── evaluate.py               # Top-level script for evaluation
├── train.py                  # Top-level script for training
├── tune.py                   # Top-level script for hyperparameter tuning
└── requirements.txt
```

---

### 2. The Data Pipeline: `src/coral_mtl/data/`

#### 2.1. The Label Transformation Pipeline (`dataset.py`)

This logic is encapsulated within the `CoralscapesDataset` class, which inherits from `torch.utils.data.Dataset`.

```python
class CoralscapesDataset(Dataset):
    def __init__(self, image_paths: list, mask_paths: list, transform=None):
        # ... initialization ...

    def __getitem__(self, idx: int) -> dict:
        # 1. Load image and 39-class mask
        # 2. Apply label transformation to get a dict of masks
        # 3. Apply augmentations (transform)
        # 4. Return {'image': image_tensor, 'masks': mask_dict}

    def _transform_mask(self, mask: np.ndarray) -> dict:
        # ... implementation of the mapping logic ...
```

**Key Mappings:**
*   **Genus Mask:**
    *   `Acropora Alive`, `Acropora Bleached`, `Acropora Dead` -> `Acropora` class index.
    *   ... (and so on for all defined coral genera).
    *   All other pixels -> `Background` class index (usually 0).
*   **Health Mask:**
    *   Any pixel from a class name containing "Alive" -> `Healthy` class index.
    *   Any pixel from a class name containing "Bleached" -> `Bleached` class index.
    *   Any pixel from a class name containing "Dead" -> `Dead` class index.
    *   All non-coral pixels -> `Background` class index.
*   **Binary Masks (Fish, Human-Artifact):** A simple mapping of one or more source classes to a single foreground class index (1) and all others to background (0).
*   **Substrate Mask:** Maps `Sand`, `Seagrass`, etc., to their respective class indices.

---

### 3. The Model Architecture: `src/coral_mtl/model/`

The architecture is modular, with each component defined in its own file.

#### 3.1. `CoralMTLModel` (`core.py`)
This class assembles the full architecture from its constituent parts.

```python
class CoralMTLModel(nn.Module):
    def __init__(self, encoder_name: str = "mit_b2", num_classes: dict = None):
        super().__init__()
        # Initializes the shared encoder (e.g., from timm)
        self.encoder = create_encoder(encoder_name, pretrained=True)

        # Initializes decoders based on num_classes dict
        self.decoders = nn.ModuleDict()
        self.prediction_heads = nn.ModuleDict()
        for task, n_cls in num_classes.items():
            if task in ["genus", "health"]: # Primary tasks
                self.decoders[task] = AllMLPDecoder(...)
            else: # Auxiliary tasks
                self.decoders[task] = LightweightDecoder(...)
            self.prediction_heads[task] = nn.Conv2d(..., n_cls, kernel_size=1)

        # Initializes cross-attention modules
        self.cross_attention_genus = ExpandedCrossAttention(...)
        self.cross_attention_health = ExpandedCrossAttention(...)

    def forward(self, x: torch.Tensor) -> dict:
        # 1. Pass input through encoder to get multi-scale features
        # 2. Pass features through each decoder to get task-specific features
        # 3. Form context blocks and apply cross-attention for primary tasks
        # 4. Pass final features through prediction heads
        # 5. Return dictionary of output logits for each task
```

#### 3.2. `ExpandedCrossAttention` (`attention.py`)
Implements the core information exchange mechanism.

```python
class ExpandedCrossAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, query_features: torch.Tensor, context_features: list[torch.Tensor]) -> torch.Tensor:
        # 1. Generate Query (Q) from query_features
        # 2. Generate Keys (K) and Values (V) from each tensor in context_features
        # 3. Concatenate all K and V to form K_context and V_context
        # 4. Perform attention: self.mha(Q, K_context, V_context)
        # 5. Return enriched features
```

---

### 4. Data Sampling: `scripts/create_pds_dataset.py`

This is a standalone script run once as a pre-processing step.

*   **Core Algorithm:** A Python implementation of the Poisson Disk Sampling algorithm, heavily optimized with Numba.
    ```python
    @numba.jit(nopython=True)
    def pds_core(foreground_mask, r_min, k_attempts):
        # ... iterative dart-throwing logic ...
        return sample_points
    ```
*   **Parallelism:** The script uses Python's `multiprocessing.Pool` to process multiple large orthomosaics in parallel, drastically reducing the total time needed.
    ```python
    with Pool(processes=os.cpu_count()) as pool:
        pool.map(process_orthomosaic, list_of_orthos)
    ```

---

### 5. Sample Augmentation: `src/coral_mtl/data/augmentations.py`

We use the `albumentations` library for efficient, GPU-accelerated transformations. The augmentation pipeline is defined as a composition of transforms.

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_training_augmentations(height: int, width: int):
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomResizedCrop(height=height, width=width, scale=(0.5, 1.0), p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8),
        A.GaussianBlur(blur_limit=(3, 7), p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
```

---

### 6. Optimizer and Scheduler (`train.py`)

The optimizer and scheduler are instantiated in the main training script, typically based on a YAML configuration file.

```python
# In train.py
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config['optimizer']['lr'],
    weight_decay=config['optimizer']['weight_decay']
)

scheduler = torch.optim.lr_scheduler.PolynomialLR(
    optimizer,
    total_iters=config['scheduler']['total_iters'],
    power=config['scheduler']['power']
)
# Note: A separate warmup schedule is typically handled in the training loop.
```

---

### 7. Loss Function: `src/coral_mtl/engine/losses.py`

The composite loss is implemented as a single `nn.Module`.

```python
class CompositeHierarchicalLoss(nn.Module):
    def __init__(self, aux_weight: float = 0.4, focal_alpha: float = 0.25, focal_gamma: float = 2.0):
        super().__init__()
        self.aux_weight = aux_weight
        # Learnable uncertainty parameters for primary tasks
        self.log_var_genus = nn.Parameter(torch.zeros(1))
        self.log_var_health = nn.Parameter(torch.zeros(1))

        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice_loss = DiceLoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, predictions: dict, targets: dict) -> torch.Tensor:
        # Calculate hybrid loss for Genus
        loss_genus = self.focal_loss(predictions['genus'], targets['genus']) + \
                     self.dice_loss(predictions['genus'], targets['genus'])

        # Calculate hybrid loss for Health
        loss_health = self.focal_loss(predictions['health'], targets['health']) + \
                      self.dice_loss(predictions['health'], targets['health'])

        # Apply uncertainty weighting
        primary_loss = (0.5 * torch.exp(-self.log_var_genus) * loss_genus + 0.5 * self.log_var_genus) + \
                       (0.5 * torch.exp(-self.log_var_health) * loss_health + 0.5 * self.log_var_health)

        # Calculate auxiliary losses
        aux_loss = sum(self.ce_loss(predictions[t], targets[t]) for t in aux_tasks)

        # Return total loss
        return primary_loss + self.aux_weight * aux_loss
```

---

### 8. Evaluation Metrics: `src/coral_mtl/engine/metrics.py`

Metrics are calculated using a helper class that accumulates confusion matrices.

```python
class MetricsCalculator:
    def __init__(self, num_classes_per_task: dict):
        # ... initializes confusion matrices for each task ...

    def update(self, predictions: dict, targets: dict):
        # ... update confusion matrices for each task ...

    def compute(self) -> dict:
        # ... compute mIoU, BIoU, diagnostic errors from matrices ...
        # Returns a dictionary of all computed metrics
```

---

### 9. Testing: `tests/`

The `tests/` directory uses `pytest` to run tests that mirror the `src/` structure.

*   **`test_dataset.py`:**
    *   `test_label_transformation()`: Creates a dummy mask with known values and asserts that the output dictionary of masks has the correct class indices for each task.
*   **`test_losses.py`:**
    *   `test_composite_loss_forward()`: Creates dummy predictions and targets and asserts that the loss computes a valid scalar value without crashing.
    *   `test_uncertainty_weighting()`: Checks if the loss value changes appropriately when the `log_var` parameters are manually altered.
*   **`test_model_forward.py`:**
    *   `test_coral_mtl_forward_pass()`: Instantiates the `CoralMTLModel`, passes a dummy input tensor, and asserts that the output is a dictionary where each value is a tensor of the expected shape `(N, C, H, W)`.

---

### 10. Possible Improvements and Extendability

This section discusses technical enhancements and provides a guide for extending the codebase.

#### 10.1. Code and Pipeline Improvements
*   **Configuration Management with Hydra:** Refactor the current argument parsing or simple YAML loading to use [Hydra](https://hydra.cc/). This would provide more powerful and flexible configuration management, including command-line overrides, composition, and multi-run capabilities for sweeping hyperparameters.
*   **Distributed Training:** Implement support for `torch.nn.parallel.DistributedDataParallel` to enable multi-GPU and multi-node training. This involves setting up process groups, wrapping the model, and using `DistributedSampler` for the data loaders.
*   **Deployment Optimization (ONNX/TensorRT):** Add a script (`export.py`) to convert the trained PyTorch model to the ON_NX format. A further step could involve using NVIDIA's TensorRT to optimize the ONNX model for high-throughput inference on deployment hardware.
*   **CI/CD Pipeline:** Set up a GitHub Actions workflow (`.github/workflows/ci.yml`) to automatically install dependencies, run `pytest`, and perform linting (e.g., with `flake8` or `black`) on every push and pull request to the `main` branch.

#### 10.2. Guide to Adding a New Task
To add a new task (e.g., "Disease Segmentation") to the model:
1.  **Update Label Mappings (`dataset.py`):** Add the logic to `_transform_mask` to create the new `disease` mask from the source 39-class annotation.
2.  **Update Configuration (`configs/`):** In your experiment's YAML file, add `"disease": num_disease_classes` to the `model.num_classes` dictionary.
3.  **Update Model (`core.py`):** The `CoralMTLModel` is designed to be extensible. As long as the new task is in the `num_classes` dictionary, the `__init__` loop will automatically create a new decoder and prediction head for it. You may need to decide if it's a primary (full decoder) or auxiliary (lightweight decoder) task.
4.  **Update Loss Function (`losses.py`):** In `CompositeHierarchicalLoss`, add the calculation for `loss_disease`. If it's a new primary task, you must also add a new `nn.Parameter` for its uncertainty (`log_var_disease`) and include it in the `primary_loss` calculation. If it's an auxiliary task, simply add it to the `aux_loss` sum.
5.  **Update Metrics (`metrics.py`):** Add the new task to the `MetricsCalculator` so its performance is tracked.
6.  **Add Tests (`tests/`):** Add a test case to `test_dataset.py` to verify the new label transformation and update the model/loss tests to account for the new task.

---
For theoretical background, see the [**Theoretical Specification**](./theorethical_specification.md).
