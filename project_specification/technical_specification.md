# Technical Specification

This document provides a comprehensive technical implementation guide for the Coral-MTL project. It details all components, their interfaces, method signatures, and how they work together through the ExperimentFactory orchestration system. This specification serves as both implementation documentation and developer reference.

For theoretical background and design justifications, please refer to the [**Theoretical Specification**](./theorethical_specification.md).

---

## 1. Project Architecture & Pipeline Overview

The Coral-MTL project follows a factory-based architecture pattern centered around the `ExperimentFactory` class, which orchestrates all components through dependency injection. The system is designed for modularity, reproducibility, and clear separation of concerns.

### 1.1. Codebase Structure

```
coral-mtl-project/
├── configs/                         # YAML configuration files for experiments
├── data/                            # Raw and processed datasets (external)
├── notebooks/                       # Jupyter notebooks for exploration and analysis
├── pds_launcher/                    # Poisson Disk Sampling dataset generation
├── project_specification/           # Comprehensive documentation
├── src/coral_mtl/                   # Core installable Python package
│   ├── ExperimentFactory.py         # Central orchestrator and dependency injector
│   ├── data/                        # Data loading and processing
│   │   ├── dataset.py               # Dataset classes for MTL and baseline models
│   │   └── augmentations.py         # Segmentation-aware augmentation pipeline
│   ├── model/                       # Model architecture components
│   │   ├── core.py                  # Main model classes (CoralMTLModel, BaselineSegformer)
│   │   ├── encoder.py               # SegFormer encoder wrapper
│   │   ├── decoders.py              # Hierarchical and standard decoders
│   │   └── attention.py             # Cross-attention mechanisms
│   ├── engine/                      # Training, evaluation, and optimization logic
│   │   ├── trainer.py               # Training orchestration with mixed precision
│   │   ├── evaluator.py             # Comprehensive testing pipeline
│   │   ├── losses.py                # Multi-task and baseline loss functions
│   │   ├── loss_weighting.py        # Multi-task weighting strategies
│   │   ├── gradient_strategies.py   # Advanced gradient manipulation (NashMTL, IMGrad)
│   │   ├── pcgrad.py                # PCGrad optimizer wrapper
│   │   └── optimizer.py             # Optimizer and scheduler factory
│   ├── metrics/                     # Three-tier metrics system
│   │   ├── metrics.py               # Tier 1: Real-time GPU metrics calculation
│   │   └── metrics_storer.py        # Tier 2/3: Async CPU processing and storage
│   ├── scripts/                     # Data processing and analysis scripts
│   │   ├── analyze_patch_distribution.py    # Statistical analysis utilities
│   │   ├── compare_distributions.py         # Distribution comparison tools
│   │   ├── create_pds_dataset.py           # PDS dataset generation
│   │   ├── create_pds_patches_and_report.py # Patch creation and analysis
│   │   └── id2labels_labels2colors_coralscapes.py # Label mapping utilities
│   └── utils/                       # Supporting utilities
│       ├── task_splitter.py         # Task definition parsing and mapping
│       └── visualization.py         # Comprehensive plotting and analysis
├── tests/coral_mtl_tests/           # Comprehensive test suite mirroring src/ structure
└── experiments/                     # Output directory for all training artifacts
```

### 1.2. The ExperimentFactory: Central Pipeline Orchestrator

The `ExperimentFactory` class serves as the central orchestrator for the entire machine learning pipeline. It implements the Factory pattern with dependency injection to ensure consistent, configuration-driven component instantiation.

#### Key Design Principles:
- **Dependency Injection**: All components are created through getter methods
- **Lazy Loading**: Components are instantiated only when needed and cached
- **Configuration-Driven**: All behavior controlled through YAML configurations
- **Component Isolation**: Each component has a clear interface and responsibility

---

## 2. ExperimentFactory Interface & Pipeline Orchestration

### 2.1. Core ExperimentFactory Class

```python
class ExperimentFactory:
    """Master factory and orchestrator for all experimental workflows."""
    
    def __init__(self, config_path: str = None, config_dict: Dict = None):
        """Initialize factory with configuration and parse task definitions."""
        
    # Component Factory Methods
    def get_model(self) -> torch.nn.Module
    def get_dataloaders(self) -> Dict[str, DataLoader]
    def get_optimizer_and_scheduler(self) -> Tuple[Optimizer, Any]
    def get_loss_function(self) -> nn.Module
    def get_metrics_calculator(self) -> AbstractCoralMetrics
    def get_metrics_storer(self) -> MetricsStorer
    def get_advanced_metrics_processor(self) -> Optional[AdvancedMetricsProcessor]
    
    # Workflow Orchestration Methods
    def run_training(self, trial: Optional[optuna.Trial] = None) -> None
    def run_evaluation(self, checkpoint_path: Optional[str] = None) -> Dict[str, Any]
    def run_hyperparameter_study(self) -> None
```

### 2.2. Pipeline Execution Flow

#### Training Pipeline (`run_training`):
1. **Component Assembly**: Uses getter methods to instantiate all required components
2. **Configuration Preparation**: Creates trainer-specific configuration namespace
3. **Trainer Instantiation**: Passes all components to the Trainer class
4. **Training Execution**: Delegates actual training loop to the Trainer
5. **Result Handling**: Returns training and validation history

#### Evaluation Pipeline (`run_evaluation`):
1. **Component Assembly**: Assembles model, test data, and metrics calculator
2. **Checkpoint Loading**: Loads best model checkpoint
3. **Configuration Setup**: Prepares evaluator-specific configuration
4. **Evaluation Execution**: Delegates to Evaluator for sliding window inference
5. **Results Storage**: Saves comprehensive evaluation report

#### Hyperparameter Optimization (`run_hyperparameter_study`):
1. **Study Configuration**: Sets up Optuna study with pruning
2. **Objective Definition**: Creates trial-specific configurations
3. **Trial Execution**: Reuses `run_training` for each trial
4. **Optimization**: Manages study execution and result reporting

---

## 3. Data Pipeline Components: `src/coral_mtl/data/`

### 3.1. Abstract Dataset Architecture

```python
class AbstractCoralscapesDataset(Dataset, ABC):
    """Abstract base class handling unified data loading from HF Hub or local files."""
    
    def __init__(self, split: str, patch_size: int, 
                 augmentations: Optional[SegmentationAugmentation] = None,
                 hf_dataset_name: Optional[str] = None,
                 data_root_path: Optional[str] = None,
                 pds_train_path: Optional[str] = None):
        """Initialize with flexible data source configuration."""
        
    def _load_data(self, idx: int) -> Tuple[Image.Image, np.ndarray, str]:
        """Load raw image, mask, and unique identifier."""
        
    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return processed data batch with standardized keys."""
```

#### Key Features:
- **Unified Data Loading**: Handles both Hugging Face Hub and local file systems
- **Flexible Source Priority**: Supports PDS training data with fallback to standard splits
- **Rich Metadata**: Returns image IDs and original masks for comprehensive evaluation

### 3.2. Multi-Task Learning Dataset

```python
class CoralscapesMTLDataset(AbstractCoralscapesDataset):
    """Dataset for Multi-Task Learning models driven by MTLTaskSplitter."""
    
    def __init__(self, splitter: MTLTaskSplitter, split: str, patch_size: int,
                 augmentations: Optional[SegmentationAugmentation] = None,
                 hf_dataset_name: Optional[str] = None,
                 data_root_path: Optional[str] = None,
                 pds_train_path: Optional[str] = None):
        """Initialize with MTL task splitter and flexible data source configuration."""
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Returns:
        {
            'image': torch.Tensor,           # Augmented input image
            'image_id': str,                 # Unique identifier
            'original_mask': torch.Tensor,   # Unmodified ground truth
            'masks': Dict[str, torch.Tensor] # Task-specific transformed masks
        }
        """
```

### 3.3. Baseline Model Dataset

```python
class CoralscapesDataset(AbstractCoralscapesDataset):
    """Dataset for baseline single-head models driven by BaseTaskSplitter."""
    
    def __init__(self, splitter: BaseTaskSplitter, split: str, patch_size: int,
                 augmentations: Optional[SegmentationAugmentation] = None,
                 hf_dataset_name: Optional[str] = None,
                 data_root_path: Optional[str] = None,
                 pds_train_path: Optional[str] = None):
        """Initialize with baseline task splitter and flexible data source configuration."""
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Returns:
        {
            'image': torch.Tensor,         # Augmented input image
            'image_id': str,               # Unique identifier
            'original_mask': torch.Tensor, # Unmodified ground truth
            'mask': torch.Tensor           # Single flattened mask
        }
        """
```

### 3.4. Segmentation Augmentation Pipeline

```python
class SegmentationAugmentation:
    """Comprehensive augmentation pipeline using torchvision.transforms.v2."""
    
    def __init__(self, patch_size: int = 512,
                 crop_scale: Tuple[float, float] = (0.5, 1.0),
                 rotation_degrees: int = 15,
                 jitter_params: Dict[str, float] = None,
                 imagenet_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
                 imagenet_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)):
        """Initialize synchronized image-mask augmentation pipeline."""
        
    def __call__(self, image: Image.Image, masks: Dict[str, Image.Image]) 
                -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Apply geometric transforms (synchronized) + color transforms (image only)."""
```

#### Augmentation Strategy:
- **Geometric Transforms**: Applied synchronously to images and all masks
- **Color Transforms**: Applied only to images to preserve mask integrity
- **Final Processing**: Tensor conversion and ImageNet normalization

---

## 4. Model Architecture Components: `src/coral_mtl/model/`

### 4.1. Core Model Classes

```python
class CoralMTLModel(nn.Module):
    """Main multi-task learning model for coral segmentation."""
    
    def __init__(self, encoder_name: str, decoder_channel: int, 
                 num_classes: Dict[str, int], attention_dim: int,
                 primary_tasks: List[str] = ['genus', 'health'],
                 aux_tasks: List[str] = ['fish', 'human_artifacts', 'substrate']):
        """Initialize with hierarchical context-aware decoder architecture."""
        
    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Returns dictionary of upsampled logits for each task."""

class BaselineSegformer(nn.Module):
    """Standard SegFormer baseline for 39-class segmentation."""
    
    def __init__(self, encoder_name: str, decoder_channel: int, num_classes: int = 39):
        """Initialize with standard All-MLP decoder."""
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Returns single upsampled logit tensor."""
```

### 4.2. Encoder Component

```python
class SegFormerEncoder(nn.Module):
    """Wrapper for Hugging Face SegformerModel as encoder backbone."""
    
    def __init__(self, pretrained_weights_path: str = "nvidia/mit-b2"):
        """Load pre-trained Mix Transformer model."""
        
    @property
    def channels(self) -> List[int]:
        """Output channel dimensions for each encoder stage."""
        
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract multi-scale feature maps from 4 encoder stages."""
```

### 4.3. Decoder Architecture

```python
class SegFormerMLPDecoder(nn.Module):
    """Standard All-MLP decoder from SegFormer paper."""
    
    def __init__(self, encoder_channels: List[int], decoder_channel: int, 
                 dropout_prob: float = 0.1):
        """Initialize channel unification and feature fusion MLPs."""
        
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Unify channels, upsample, concatenate, and fuse features."""

class HierarchicalContextAwareDecoder(nn.Module):
    """Advanced decoder with asymmetric heads and cross-attention."""
    
    def __init__(self, encoder_channels: List[int], decoder_channel: int,
                 num_classes: Dict[str, int], attention_dim: int = 256,
                 primary_tasks: List[str] = ['genus', 'health'],
                 aux_tasks: List[str] = ['fish', 'human_artifacts', 'substrate']):
        """Initialize with task-specific decoders and attention modules."""
        
    def forward(self, features: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply asymmetric decoding with cross-attention enrichment."""
```

#### Decoder Architecture Details:
- **Asymmetric Design**: Full MLP decoders for primary tasks, lightweight heads for auxiliary tasks
- **Cross-Attention**: Primary tasks attend to context from all other tasks
- **Gated Integration**: Learnable gates balance original and enriched features

### 4.4. Attention Mechanisms

```python
class MultiTaskCrossAttentionModule(nn.Module):
    """Legacy symmetric cross-attention between genus and health tasks."""
    
    def __init__(self, in_channels: int):
        """Initialize learnable projection layers for Q, K, V."""
        
    def forward(self, F_genus: torch.Tensor, F_health: torch.Tensor) 
                -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform bidirectional cross-attention."""
```

---

## 5. Training & Evaluation Engine: `src/coral_mtl/engine/`

### 5.1. Training Orchestration

```python
class Trainer:
    """Generic training and validation engine with mixed precision and three-tier metrics support."""
    
    def __init__(self, model, train_loader, val_loader, loss_fn,
                 metrics_calculator: AbstractCoralMetrics,
                 metrics_storer: MetricsStorer, optimizer, scheduler,
                 config, trial: optuna.Trial = None,
                 metrics_processor: AdvancedMetricsProcessor = None):
        """Initialize with all training components and optional Tier 2/3 metrics processor."""
        
    def train(self):
        """Main training loop with epoch management, validation, and checkpointing."""
        
    def _train_one_epoch(self):
        """Execute single training epoch with gradient accumulation and strategy support."""
        
    def _validate_one_epoch(self, epoch: int = None) -> Dict[str, Any]:
        """Execute validation with sliding window inference and dual-tier metrics dispatch."""
```

#### Training Features:
- **Mixed Precision**: Automatic mixed precision with gradient scaling
- **Advanced Gradient Strategies**: Support for NashMTL, IMGrad, PCGrad, and uncertainty weighting
- **Gradient Accumulation**: Support for effective batch sizes larger than memory allows
- **Three-Tier Metrics Integration**: Concurrent real-time and advanced metrics computation
- **Sliding Window Validation**: Full-resolution validation inference with dual-tier dispatch
- **Optuna Integration**: Built-in support for hyperparameter optimization with pruning
- **Comprehensive Logging**: Detailed loss component tracking and metrics storage

### 5.2. Evaluation Pipeline

```python
class Evaluator:
    """Comprehensive final evaluation pipeline with three-tier metrics support."""
    
    def __init__(self, model: torch.nn.Module, test_loader: torch.utils.data.DataLoader,
                 metrics_calculator: AbstractCoralMetrics,
                 metrics_storer: MetricsStorer, config: object,
                 metrics_processor: Optional[AdvancedMetricsProcessor] = None,
                 loss_fn: Optional[torch.nn.Module] = None):
        """Initialize evaluation components with optional Tier 2/3 processor."""
        
    def evaluate(self) -> Dict[str, Any]:
        """Execute complete testing pipeline with dual-tier metrics computation."""
```

#### Evaluation Features:
- **Checkpoint Loading**: Automatic best model loading
- **Sliding Window Inference**: Memory-efficient full-resolution prediction
- **Three-Tier Metrics**: Real-time GPU metrics plus comprehensive CPU analysis
- **Comprehensive Metrics**: Hierarchical evaluation across all task levels
- **Advanced Analytics**: Surface distance, clustering, and panoptic metrics
- **Result Persistence**: Detailed storage of per-image results and final reports

### 5.3. Loss Functions

```python
class CoralMTLLoss(nn.Module):
    """Complete multi-task loss with configurable weighting strategies."""
    
    def __init__(self, num_classes: Dict[str, int],
                 primary_tasks: List[str],
                 aux_tasks: List[str],
                 weighting_strategy: WeightingStrategy,
                 ignore_index: int = 0, w_consistency: float = 0.1,
                 hybrid_alpha: float = 0.5, focal_gamma: float = 2.0):
        """Initialize with configurable weighting strategy and hybrid loss components."""
        
    def forward(self, predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Return comprehensive loss dictionary with all components."""

class CoralLoss(nn.Module):
    """Flexible hybrid loss for baseline models."""
    
    def __init__(self, primary_loss_type: str = 'focal',
                 hybrid_alpha: float = 0.5, focal_gamma: float = 2.0,
                 dice_smooth: float = 1.0,
                 class_weights: Optional[torch.Tensor] = None,
                 ignore_index: int = 0):
        """Initialize hybrid Focal/CE + Dice loss combination."""
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Return combined loss value."""
```

#### Loss Function Features:
- **Configurable Weighting Strategies**: Uncertainty, NashMTL, IMGrad, and other advanced strategies
- **Hybrid Loss Components**: Combines Focal/Cross-Entropy with Dice loss
- **Consistency Regularization**: Penalizes logically inconsistent predictions
- **Gradient Strategy Integration**: Support for PCGrad and advanced gradient manipulation
- **Comprehensive Logging**: Returns detailed loss component breakdown

### 5.4. Optimizer Factory

```python
def create_optimizer_and_scheduler(
    model: nn.Module, learning_rate: float = 6e-5,
    weight_decay: float = 0.01, adam_betas: Tuple[float, float] = (0.9, 0.999),
    num_training_steps: int = 10000, num_warmup_steps: int = 1500,
    power: float = 1.0) -> Tuple[optim.Optimizer, Any]:
    """Create AdamW optimizer with polynomial decay + warmup scheduler."""
```

#### Optimizer Features:
- **Parameter Grouping**: Separate weight decay for different parameter types
- **Polynomial Decay**: Stable learning rate schedule with warmup
- **Transformer-Optimized**: Best practices for SegFormer-based architectures

### 5.5. Multi-Task Weighting Strategies

The system provides sophisticated multi-task learning optimization through configurable weighting strategies:

```python
class WeightingStrategy(ABC):
    """Abstract base for multi-task weighting strategies."""
    
    @abstractmethod
    def compute_weights(self, losses: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Compute task weights based on current losses."""

class UncertaintyWeighting(WeightingStrategy):
    """Learnable uncertainty-based task weighting (Kendall & Gal 2017)."""

class NashMTLStrategy(GradientUpdateStrategy):
    """Nash equilibrium-based multi-task optimization."""

class IMGradStrategy(GradientUpdateStrategy):
    """Impartial multi-task gradient-based optimization."""
```

#### Weighting Strategy Features:
- **Uncertainty Weighting**: Learnable homoscedastic uncertainty parameters
- **NashMTL**: Scale-invariant Nash equilibrium optimization
- **IMGrad**: Adaptive gradient blending with cosine similarity
- **PCGrad Integration**: Gradient projection for conflict reduction
- **Configurable Selection**: YAML-driven strategy configuration

### 5.6. Advanced Gradient Manipulation

```python
class PCGrad:
    """Projecting Conflicting Gradients optimizer wrapper."""
    
    def __init__(self, optimizer, reduction: str = 'mean'):
        """Wrap any PyTorch optimizer with gradient projection."""

class GradientUpdateStrategy(ABC):
    """Abstract base for gradient-based multi-task optimization."""
    
    @abstractmethod
    def update_step(self, model: nn.Module, losses: Dict[str, torch.Tensor],
                   optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
        """Perform gradient-aware parameter update."""
```

---

## 6. Metrics Components: `src/coral_mtl/metrics/`

### 6.1. Three-Tier Metrics Architecture

The project implements a sophisticated three-tier metrics system that separates real-time GPU computation from comprehensive CPU analysis:

#### Tier 1: Real-Time GPU Metrics Engine

```python
class AbstractCoralMetrics(ABC):
    """Abstract base for hierarchical coral segmentation metrics with GPU acceleration."""
    
    def __init__(self, splitter: TaskSplitter, storer: MetricsStorer,
                 device: torch.device, boundary_thickness: int = 2, 
                 ignore_index: int = 255, use_async_storage: bool = True):
        """Initialize with task splitter, storage, and GPU acceleration."""
        
    def reset(self):
        """Reset accumulated statistics including GPU tensors for new epoch/run."""
        
    @abstractmethod
    def update(self, predictions: Any, original_targets: torch.Tensor, 
               image_ids: List[str], epoch: int = None, 
               predictions_logits: Any = None):
        """Update metrics with batch predictions and raw logits for calibration."""
        
    def compute(self) -> Dict[str, Any]:
        """Compute comprehensive metrics report including calibration metrics."""

class CoralMTLMetrics(AbstractCoralMetrics):
    """Multi-task metrics calculator for dictionary-based predictions."""
    
class CoralMetrics(AbstractCoralMetrics):
    """Baseline metrics calculator with flattened prediction unrolling."""
```

#### Tier 2/3: Advanced Asynchronous Processing

```python
class AdvancedMetricsProcessor:
    """Multi-process CPU worker pool for comprehensive per-image metrics."""
    
    def __init__(self, output_dir: str, num_cpu_workers: int = 30,
                 enabled_tasks: List[str] = ["ASSD", "HD95", "PanopticQuality", "ARI"]):
        """Initialize with configurable worker pool and task selection."""
        
    def start(self):
        """Start multiprocessing infrastructure with job queues and workers."""
        
    def dispatch_image_job(self, image_id: str, pred_mask: np.ndarray, 
                          target_mask: np.ndarray):
        """Non-blocking dispatch of per-image advanced metric computation."""
        
    def shutdown(self):
        """Graceful shutdown with proper resource cleanup."""
```

#### Tier 1 Metrics Features:
- **Real-Time GPU Computation**: Confusion matrices, Boundary IoU, calibration metrics
- **Probabilistic Metrics**: NLL, Brier Score, Expected Calibration Error (ECE)
- **Hierarchical Evaluation**: Computes metrics at both grouped and ungrouped task levels
- **Global Metrics**: Unified metrics across all classes for comprehensive comparison
- **TIDE-Inspired Error Analysis**: Classification, background, and missed error decomposition

#### Tier 2/3 Advanced Metrics Features:
- **Surface Distance Metrics**: ASSD, HD95 using scipy distance transforms
- **Clustering Metrics**: Adjusted Rand Index (ARI), Variation of Information
- **Panoptic Metrics**: Per-image Panoptic Quality (PQ) component computation
- **Asynchronous Processing**: Non-blocking CPU computation with 30+ worker pool
- **JSONL Streaming**: Per-image results streamed for post-processing analysis

### 6.2. Metrics Storage & Persistence

```python
class MetricsStorer:
    """Handles persistent storage of metrics and raw evaluation data."""
    
    def __init__(self, experiment_name: str, output_dir: Path):
        """Initialize with experiment identification and output directory."""
        
    def store_batch_results(self, image_ids: List[str], confusion_matrices: List[np.ndarray]):
        """Store per-image confusion matrices in JSONL format."""
        
    def finalize_and_save(self, metrics_report: Dict[str, Any]):
        """Save final aggregated metrics report."""

class AsyncMetricsStorer(MetricsStorer):
    """Asynchronous version for high-throughput scenarios."""

class AdvancedMetricsProcessor:
    """Advanced analysis and visualization of stored metrics."""
```

#### Storage Features:
- **JSONL Streaming**: Memory-efficient per-image data storage
- **Safe Writing**: Atomic file operations to prevent corruption
- **Comprehensive Logging**: Stores both aggregated metrics and raw confusion matrices
- **Flexible Output**: Supports both validation and testing workflows

---

## 7. Utility Components: `src/coral_mtl/utils/`

### 6.1. Task Definition Processing

```python
class TaskSplitter(ABC):
    """Abstract base for parsing and structuring task definitions."""
    
    def __init__(self, task_definitions: Dict[str, Dict[str, Any]]):
        """Parse raw task definitions into hierarchical structures."""
        
    # Key Properties
    hierarchical_definitions: Dict[str, Dict[str, Any]]  # Per-task parsed structures
    global_mapping_array: np.ndarray                     # Unified label space mapping
    global_id2label: Dict[int, str]                      # Global class names
    num_global_classes: int                              # Total unique classes

class MTLTaskSplitter(TaskSplitter):
    """TaskSplitter for Multi-Task Learning models."""

class BaseTaskSplitter(TaskSplitter):
    """TaskSplitter for baseline models with flattened training space."""
    
    # Additional Properties
    flat_mapping_array: np.ndarray                       # Training space mapping
    flat_to_original_mapping_array: np.ndarray           # Inverse mapping for evaluation
```

#### Task Splitter Features:
- **Hierarchical Parsing**: Handles grouped/ungrouped task structures
- **Global Label Space**: Creates unified, non-overlapping class space
- **Mapping Arrays**: Efficient NumPy-based label transformations
- **Validation Support**: Enables fair comparison between MTL and baseline models

### 6.2. Metrics Storage & Persistence

```python
class MetricsStorer:
    """Handles persistent storage of metrics and raw evaluation data."""
    
    def __init__(self, output_dir: str):
        """Initialize output directory and file paths."""
        
    def open_for_run(self, is_testing: bool = False):
        """Open file handles for validation or testing run."""
        
    def store_epoch_history(self, metrics_report: Dict, epoch: int):
        """Update and save training history with new epoch metrics."""
        
    def store_per_image_cms(self, image_id: str, 
                           confusion_matrices: Dict[str, np.ndarray],
                           predicted_masks: Dict[str, np.ndarray] = None,
                           is_testing: bool = False, epoch: int = None):
        """Store raw confusion matrices and predictions to JSONL."""
        
    def save_final_report(self, metrics_report: Dict[str, Any], filename: str):
        """Save comprehensive evaluation report as JSON."""
```

#### Storage Features:
- **JSONL Streaming**: Memory-efficient per-image data storage
- **Safe Writing**: Atomic file operations to prevent corruption
- **Comprehensive Logging**: Stores both aggregated metrics and raw confusion matrices
- **Flexible Output**: Supports both validation and testing workflows

### 6.3. Comprehensive Visualization

```python
class Visualizer:
    """Unified visualization system for all Coral-MTL project plots."""
    
    def __init__(self, output_dir: str, task_info: Dict = None, 
                 style: str = 'seaborn-v0_8-whitegrid'):
        """Initialize with output directory and styling."""
        
    # High-Level Performance Plots
    def plot_model_comparison(self, results: Dict[str, Dict[str, float]], 
                             filename: str = "model_comparison.png")
    def plot_validation_performance(self, val_history: Dict[str, List[float]], 
                                   filename: str = "validation_performance.png")
    
    # Training Monitoring Plots
    def plot_training_losses(self, log_history: Dict[str, List[float]], 
                            filename: str = "training_losses.png")
    def plot_learning_rate(self, log_history: Dict[str, List[float]], 
                          warmup_steps: int = 0, 
                          filename: str = "learning_rate.png")
    def plot_uncertainty_weights(self, log_history: Dict[str, List[float]], 
                                filename: str = "uncertainty_weights.png")
    
    # Detailed Analysis Plots  
    def plot_qualitative_results(self, images: torch.Tensor, 
                                gt_masks: torch.Tensor, 
                                pred_masks: torch.Tensor,
                                task_name: str, 
                                filename: str = "qualitative_results.png",
                                num_samples: int = 4)
    def plot_diagnostic_error_breakdown(self, 
                                       results: Dict[str, Tuple[np.ndarray, List[str]]],
                                       task_name: str, 
                                       filename: str = "error_breakdown.png")
    def plot_confusion_analysis(self, cm: np.ndarray, class_names: List[str],
                               task_name: str, 
                               filename: str = "confusion_matrix.png",
                               top_k: int = 3, threshold: int = 10)
```

#### Visualization Features:
- **Comprehensive Coverage**: All aspects of model performance and training
- **Data Persistence**: Saves plot data as JSON alongside images
- **Adaptive Plotting**: Chooses optimal visualization based on data characteristics
- **TIDE-Inspired Analysis**: Error decomposition following computer vision best practices
- **Professional Styling**: Publication-ready plots with consistent formatting

---

## 8. Scripts Components: `src/coral_mtl/scripts/`

The scripts directory contains utility scripts for data processing and analysis:

### 8.1. Data Processing Scripts

- **`analyze_patch_distribution.py`**: Statistical analysis utilities for patch distributions
- **`compare_distributions.py`**: Tools for comparing data distributions across splits
- **`create_pds_dataset.py`**: PDS dataset generation utilities
- **`create_pds_patches_and_report.py`**: Patch creation and comprehensive reporting
- **`id2labels_labels2colors_coralscapes.py`**: Label mapping and color palette utilities

---

## 9. Configuration System & Advanced Features

### 9.1. Comprehensive Configuration Schema

The system uses YAML configuration files to control all aspects of model training and evaluation. The ExperimentFactory reads these configurations and instantiates components accordingly.

#### Core Configuration Structure

```yaml
# Model Architecture Configuration
model:
  type: "CoralMTL"  # or "SegFormerBaseline"
  params:
    backbone: "nvidia/mit-b2"
    decoder_channel: 256
    attention_dim: 256
  tasks:
    primary: ["genus", "health"]
    auxiliary: ["fish", "human_artifacts", "substrate"]

# Data Pipeline Configuration
data:
  dataset_name: "EPFL-ECEO/coralscapes"
  task_definitions_path: "configs/task_definitions.yaml"
  patch_size: 512
  batch_size_per_gpu: 4
  num_workers: 4
  # Optional: PDS training data override
  pds_train_path: "./dataset/processed/pds_patches/"
  data_root_path: "./dataset/"

# Loss Function Configuration
loss:
  type: "CompositeHierarchical"
  params:
    w_consistency: 0.1
    hybrid_alpha: 0.5
    focal_gamma: 2.0
    ignore_index: 0
  # Advanced: Multi-task weighting strategy
  weighting_strategy:
    type: "NashMTL"  # "Uncertainty", "IMGrad", etc.
    params:
      update_frequency: 10

# Optimization Configuration
optimizer:
  type: "AdamWPolyDecay"
  params:
    lr: 6e-5
    weight_decay: 0.01
    warmup_ratio: 0.1
    power: 1.0
  # Optional: PCGrad integration
  use_pcgrad_wrapper: false

# Three-Tier Metrics Configuration
metrics_processor:
  enabled: true
  num_cpu_workers: 30
  tasks: ["ASSD", "HD95", "PanopticQuality", "ARI"]

# Training Configuration
trainer:
  epochs: 100
  device: "auto"
  output_dir: "experiments/experiment_name"
  model_selection_metric: "optimization_metrics.H-Mean"
  gradient_accumulation_steps: 1
  use_mixed_precision: true
  inference_stride: 256
  inference_batch_size: 16

# Evaluation Configuration
evaluator:
  checkpoint_path: null  # Auto-detect from trainer.output_dir
  output_dir: null       # Auto-set to trainer.output_dir/evaluation
  inference_stride: 256
  inference_batch_size: 16

# Hyperparameter Study Configuration
study:
  name: "coral_mtl_study"
  storage: "sqlite:///study.db"
  config_path: "configs/search_space.yaml"
  n_trials: 50
  direction: "maximize"
  pruner:
    type: "MedianPruner"
    params:
      n_warmup_steps: 5
```

### 9.2. Advanced Configuration Features

#### Multi-Task Weighting Strategy Configuration

```yaml
loss:
  weighting_strategy:
    type: "NashMTL"
    params:
      update_frequency: 10
      solver: "cvxopt"  # Optional: specify solver preference
```

#### PCGrad Integration

```yaml
optimizer:
  use_pcgrad_wrapper: true

trainer:
  pcgrad:
    enabled: true
    reduction: "mean"
```

#### Three-Tier Metrics System

```yaml
metrics_processor:
  enabled: true
  num_cpu_workers: 30
  tasks: ["ASSD", "HD95", "PanopticQuality", "ARI", "VariationOfInformation"]
```

---

## 10. Configuration-Driven Architecture

The system uses YAML configuration files to control all aspects of model training and evaluation. The ExperimentFactory reads these configurations and instantiates components accordingly.

### 7.1. Configuration Structure

```yaml
# Example configuration structure
data:
  dataset_name: "path/to/dataset"
  task_definitions_path: "configs/task_definitions.yaml"
  patch_size: 512
  batch_size_per_gpu: 4

model:
  type: "CoralMTL"  # or "SegFormerBaseline"
  params:
    backbone: "nvidia/mit-b2"
    decoder_channel: 256
    attention_dim: 256
  tasks:
    primary: ["genus", "health"]
    auxiliary: ["fish", "human_artifacts", "substrate"]

optimizer:
  type: "AdamWPolyDecay"
  params:
    lr: 6e-5
    weight_decay: 0.01
    warmup_ratio: 0.1

loss:
  type: "CompositeHierarchical"
  params:
    w_consistency: 0.1
    hybrid_alpha: 0.5
    focal_gamma: 2.0

trainer:
  epochs: 100
  device: "cuda"
  output_dir: "experiments/experiment_name"
  model_selection_metric: "optimization_metrics.H-Mean"
```

### 7.2. Component Factory Methods

Each component in the ExperimentFactory follows a consistent pattern:
1. **Cache Check**: Avoid re-instantiation of expensive components
2. **Configuration Reading**: Extract relevant configuration section
3. **Dependency Resolution**: Ensure required components are available
4. **Instantiation**: Create component with configuration parameters
5. **Caching**: Store component for reuse

This pattern ensures that components are created in the correct order with proper dependencies, while the configuration-driven approach allows for easy experimentation and reproducibility.

---

## 10. Advanced Features

### 8.1. Sliding Window Inference

```python
class SlidingWindowInferrer:
    """Memory-efficient inference for full-resolution images."""
    
    def predict(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Predict on single full-resolution image with overlapping patches."""
        
    def predict_batch(self, images: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Batch prediction with sliding window approach."""
```

### 8.2. Hyperparameter Optimization Integration

The system provides seamless integration with Optuna for hyperparameter optimization:
- **Trial Configuration**: Dynamic configuration modification per trial
- **Pruning Support**: Early stopping of unpromising trials
- **Metric Optimization**: Flexible metric selection for optimization
- **Study Management**: Persistent study storage and resumption

### 8.3. Mixed Precision Training

All training components support automatic mixed precision:
- **Memory Efficiency**: Reduced memory usage with FP16
- **Performance Optimization**: Faster training on modern GPUs
- **Numerical Stability**: Gradient scaling to prevent underflow
- **Backward Compatibility**: Automatic fallback for unsupported hardware

---

## 11. Extension Guide

### 9.1. Adding New Tasks

To add a new task (e.g., "coral_disease"):

1. **Update Task Definitions** (`configs/task_definitions.yaml`):
```yaml
coral_disease:
  id2label:
    0: "healthy"
    1: "bleached"
    2: "diseased"
```

2. **Update Model Configuration**:
```yaml
model:
  tasks:
    auxiliary: ["fish", "human_artifacts", "substrate", "coral_disease"]
```

3. **Update Loss Function**: Add new task to auxiliary loss calculation in `CoralMTLLoss`

4. **Update Metrics**: The system will automatically handle the new task through the TaskSplitter

### 9.2. Adding New Model Architectures

1. **Create Model Class**: Implement in `src/coral_mtl/model/`
2. **Update ExperimentFactory**: Add new model type to `get_model()` method
3. **Update Configuration**: Add new model type to configuration schema
4. **Add Tests**: Create corresponding test cases

### 9.3. Adding New Loss Functions

1. **Implement Loss Class**: Create in `src/coral_mtl/engine/losses.py`
2. **Update Factory**: Add to `get_loss_function()` method
3. **Update Configuration**: Add loss type to configuration options

---

## 11. Two‑Tier Metric Evaluation Architecture

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

### 8. Evaluation Metrics: `src/coral_mtl/metrics/metrics.py`

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

## 11. Three‑Tier Metric Evaluation Architecture

This project implements a concurrent, three‑tier metrics system that separates fast, GPU‑friendly aggregation from expensive CPU‑only analytics. The design minimizes GPU↔CPU transfers, avoids I/O contention, and keeps training/validation loops non‑blocking.

### 11.1. Overview

- Tier 1 (GPU Collector): Real‑time accumulation of confusion matrices and boundary/probabilistic statistics directly on GPU; final normalization on CPU at epoch end.
- Tier 2 (Advanced CPU Processor): Asynchronous dispatcher feeding a pool of CPU workers and a dedicated writer process to compute and stream per‑image advanced metrics to JSONL.

Key classes and files:
- Tier 1: `AbstractCoralMetrics`, `CoralMTLMetrics`, `CoralMetrics` in `src/coral_mtl/metrics/metrics.py`
- Tier 2: `AdvancedMetricsProcessor` in `src/coral_mtl/metrics/metrics_storer.py`
- Orchestration: `Trainer`, `Evaluator` in `src/coral_mtl/engine/`, and factory wiring in `src/coral_mtl/ExperimentFactory.py`

### 11.2. Tier 1 — Real-Time GPU Collector

Contract:
- Input per batch: predictions (dict or tensor), original_targets (global mask), image_ids, and optional predictions_logits.
- Operations: Update accumulators for per‑task/global confusion matrices, Boundary IoU stats, and probabilistic calibration stats (NLL, Brier, ECE bins) using logits.
- Output at compute(): Structured report with grouped/ungrouped task metrics, global metrics, diagnostic errors, boundary IoU, and calibration metrics added to `optimization_metrics`.

Implementation notes:
- Accumulators added in `AbstractCoralMetrics.reset()`:
    - Confusion matrices per task and global space
    - Boundary stats (TP/FP/FN) for BIoU, including global BIoU
    - Probabilistic accumulators: total NLL sum, Brier sum, ECE bin counts/accuracies/confidences, total_pixels
- GPU helpers aggregate per batch with minimal CPU transfers:
    - `_update_boundary_stats_gpu(...)`
    - `_update_global_biou_stats_gpu(...)`
    - `_update_probabilistic_stats_gpu(logits, target)` (uses representative logits for calibration)
- `compute()` constructs final metrics, including:
    - Global BIoU
    - NLL, Brier Score, and ECE from bin statistics (zero‑safe fallbacks)

### 11.3. Tier 2 — Advanced CPU Processor

Advanced, CPU‑intensive metrics are processed asynchronously to avoid blocking the main loop.

Core components in `AdvancedMetricsProcessor`:
- Dispatcher API:
    - `start()`: Spawn multiprocessing Manager, job/result queues, CPU worker pool, and a single writer process
    - `dispatch_image_job(image_id, pred_mask, target_mask)`: Non‑blocking transfer to CPU and enqueue minimal uint8 arrays per image
    - `shutdown()`: Graceful termination, drain/flush, join workers and writer
- Worker Gauntlet (per job, task‑gated):
    - Surface metrics: ASSD, HD95 (SimpleITK/scipy)
    - Clustering/segmentation metrics: ARI, Variation of Information (sklearn)
    - Panoptic metrics: PQ/AP (panopticapi/pycocotools)
- Writer Process:
    - Streams one JSON object per image to a JSONL file (validation_cms.jsonl or test_cms.jsonl)
    - Dedicated single writer avoids file contention across many workers

Outputs:
- Tier 1: Final JSON report via `MetricsStorer.save_final_report()`
- Tier 2: JSONL stream with per‑image advanced metrics for downstream analysis

### 11.4. Lifecycle Integration (Trainer/Evaluator)

- The `ExperimentFactory` builds and injects `AdvancedMetricsProcessor` when enabled in config.
- `Trainer`/`Evaluator` lifecycle:
    - At run start: `metrics_processor.start()`
    - For each batch:
        - Tier 1: `metrics_calculator.update(..., predictions_logits=...)`
        - Tier 2: derive `pred_mask` (argmax of logits) and call `dispatch_image_job(image_id, pred_mask, original_mask)` per image
    - On completion or exception: `metrics_processor.shutdown()` in a finally block

This ensures validation/testing remain responsive while advanced metrics are computed in parallel.

### 11.5. Configuration Schema

Add a `metrics_processor` block to your experiment config:

```yaml
metrics_processor:
    enabled: true           # enable asynchronous Tier 2 processing
    num_cpu_workers: 30     # size of the CPU worker pool
    tasks:                  # which advanced metrics to compute
        - "ASSD"
        - "HD95"
        - "PanopticQuality"
        - "ARI"
```

Factory wiring:
- `ExperimentFactory.get_advanced_metrics_processor()` parses this block and constructs the processor
- `run_training` and `run_evaluation` inject the processor into `Trainer`/`Evaluator`

### 11.6. Optional Dependencies and Task Gating

Some Tier 2 metrics depend on optional libraries (e.g., SimpleITK, scikit‑image, panopticapi, scikit‑learn). The processor uses lazy imports and task‑gated execution to avoid importing unused libraries. If a dependency is missing while a corresponding task is enabled, the worker will:
- Log a clear warning per missing package
- Skip the specific metric for that image
- Continue processing remaining enabled metrics

Recommendations:
- Keep Tier 2 tasks minimal during training (or disable entirely) to reduce CPU load
- Enable full task set for final evaluation where wall‑time is acceptable

### 11.7. Post‑Processing and Aggregation

Tier 2 per‑image outputs (JSONL) are designed for external aggregation (e.g., computing dataset‑wide PQ/AP or percentile summaries of distance metrics). Provide a simple analysis script to:
- Read JSONL, group by metric/class, and compute aggregates
- Join with Tier 1 final report for a complete evaluation dossier

File locations and names are managed by `MetricsStorer` based on split (validation/test) and experiment output directory.

---
For theoretical background, see the [**Theoretical Specification**](./theorethical_specification.md).
