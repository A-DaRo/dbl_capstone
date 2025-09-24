# Test Specification

This document defines the testing strategy and detailed test plans for the Coral‑MTL project. It consolidates guidance from the Technical Specification and the Theoretical Specification to ensure correctness, reliability, and reproducibility across unit, integration, and behavioral tests.

It covers:
- Unit tests for each module/class in `src/coral_mtl` (functional correctness, shapes/types, and error handling)
- Integration/behavioral tests for the full training/validation and evaluation flows
- Concurrency and I/O tests for the two‑tier metrics architecture
- Execution guidance, fixtures, markers, coverage targets, and CI notes

---

## 1. Scope and Goals

- Verify that all components behave according to their contracts in the Technical Specification
- Provide safety rails for refactors (API stability, backward compatibility where applicable)
- Validate scientific correctness of key metrics and losses using controlled synthetic data
- Ensure the concurrent, two‑tier metrics pipeline is robust (no deadlocks; graceful shutdown; correct outputs)
- Keep tests deterministic, parallelizable, and reasonably fast; isolate optional dependencies with skips

Out of scope: End‑to‑end perf benchmarks and large‑dataset metrics. Those are covered by notebooks and optional long‑running scripts.

---

## 2. Test Levels and Organization

- Unit tests: `tests/**` mirroring `src/coral_mtl/**`
  - Fast, deterministic, single‑process unless concurrency is the subject
  - Use synthetic data and light stubs; avoid GPU dependency unless explicitly marked
- Integration/behavioral tests: `tests/integration/**`
  - Exercise the training/validation loop, sliding‑window inference, and evaluation outputs
  - May run on CPU; keep images small (e.g., 32–128 px) to remain fast
- Concurrency & Tier‑2 tests: `tests/concurrency/**`
  - Focused on `AdvancedMetricsProcessor` lifecycle, dispatch, writer output, task gating

Naming conventions: `test_<module>_<feature>.py` with parametrized cases where appropriate.

Markers (suggested in `pytest.ini`):
- `@pytest.mark.gpu`: requires CUDA
- `@pytest.mark.slow`: >5s locally
- `@pytest.mark.integration`: integration/system scenarios
- `@pytest.mark.optdeps`: optional dependency required (SimpleITK, panopticapi, skimage, sklearn)

---

## 3. Test Environment and Fixtures

- Random seeds fixed (e.g., `torch.manual_seed(0)`, `np.random.seed(0)`)
- Default device: CPU; GPU tests guarded by `@pytest.mark.gpu` and `torch.cuda.is_available()`
- Temporary directories via `tmp_path`/`tmp_path_factory` (file I/O isolation)
- Synthetic tensors with tiny shapes (e.g., 1×3×32×32 for images; 32×32 masks)
- Minimal dummy Task Definitions for stable mappings

Shared fixtures:
- `splitter_mtl` and `splitter_base`: small, deterministic mappings
- `dummy_images`, `dummy_masks`: consistent synthetic data for unit tests
- `factory_config_dict`: minimal, valid config dict for ExperimentFactory

---

## 4. Module‑by‑Module Unit Tests

Below, “Acceptance” bullets describe pass/fail criteria; “Edge cases” list scenarios to cover in addition to the happy path.

### 4.1 Experiment Orchestration — `ExperimentFactory.py`

Tests:
- Build components from dict and from YAML (if path provided)
- Caching: repeated getter calls return same instances (where intended)
- Dependency injection: `Trainer`/`Evaluator` receive `metrics_calculator`, `metrics_storer`, and `AdvancedMetricsProcessor` (when enabled)
- Metrics processor config parsing (enabled/disabled, num workers, tasks)

Acceptance:
- Correct classes instantiated; attributes present; no unexpected side effects
- Metrics processor is None when disabled; constructed when enabled

Edge cases:
- Missing optional blocks; invalid values cause clear exceptions

### 4.2 Data — `data/dataset.py`

AbstractCoralscapesDataset:
- `_load_data` protocol honored by concrete subclasses (mocked or via fixtures)

CoralscapesMTLDataset:
- `__getitem__` returns expected keys: `image`, `image_id`, `original_mask`, `masks`
- Shapes/types correct; masks dict contains configured tasks only

CoralscapesDataset (baseline):
- `__getitem__` returns expected keys, including single `mask`

Acceptance:
- Label transformation aligns with `TaskSplitter` mappings

Edge cases:
- Ignore index handling; empty/single‑class tiles; missing PDS path fallback

### 4.3 Data — `data/augmentations.py`

SegmentationAugmentation:
- Geometric transforms applied consistently to image and all masks
- Color transforms apply only to image; masks unchanged apart from geometry
- Output normalization and dtype

Acceptance:
- Deterministic with fixed seeds; shapes preserved; masks remain integer labels

Edge cases:
- Extreme rotation/scale bounds; non‑square patch sizes

### 4.4 Model — `model/encoder.py`

SegFormerEncoder:
- `channels` property length and types
- Forward returns list of 4 feature maps with expected spatial scale ordering

Acceptance:
- Shapes match expected downsampling factors; dtype is float

Edge cases:
- Non‑standard input resolutions (ensure safe behavior)

### 4.5 Model — `model/decoders.py`

SegFormerMLPDecoder:
- Forward merges multi‑scale features; output spatial size expected by heads

HierarchicalContextAwareDecoder:
- Creates heads for all configured tasks; primary vs auxiliary capacity differences honored
- Forward returns per‑task feature maps before heads (or logits if integrated)

Acceptance:
- Per‑task output channels align with `num_classes`

Edge cases:
- Missing task configured; empty auxiliary list

### 4.6 Model — `model/attention.py`

MultiTaskCrossAttentionModule (and expanded attention variants if present):
- Forward shape integrity and attention causality (no NaNs/Inf)

Acceptance:
- Input/output tensors maintain batch/spatial dims; gradients backpropagate in autograd gradcheck on a tiny slice (optional)

### 4.7 Model — `model/core.py`

CoralMTLModel:
- Forward returns dict of logits for each configured task, aligned shapes/classes

BaselineSegformer:
- Forward returns single logits tensor (N,C,H,W)

Acceptance:
- Output shapes correct; no key/task drift; optional mixed precision path does not error

Edge cases:
- Only one primary task; zero auxiliary tasks

### 4.8 Engine — `engine/losses.py`

CoralMTLLoss:
- Forward returns dict with component losses and total; decreases under trivial overfit on tiny batch
- Uncertainty parameters exist and affect weighting when perturbed

CoralLoss:
- Hybrid combination produces finite scalar; respects `ignore_index`

Acceptance:
- Backward pass succeeds; grads finite; changing inputs changes loss meaningfully

Edge cases:
- Class weights provided; empty foreground; extreme imbalance

### 4.9 Engine — `engine/metrics.py`

AbstractCoralMetrics and implementations:
- `reset()` initializes accumulators (CMs, boundary stats, calibration bins)
- `update(...)` accepts predictions and `predictions_logits` (optional) + `original_targets`/`image_ids`
- Tier 1 accumulators update correctly on synthetic inputs:
  - mIoU for trivial perfect predictions is 1.0; 0.0 for fully wrong
  - Boundary stats reflect expected TP/FP/FN on simple shapes (e.g., squares)
  - Calibration metrics:
    - Perfect one‑hot logits on correct class ⇒ low NLL/Brier, ECE≈0
    - Uniform logits ⇒ high NLL/Brier, ECE>0
- `compute()` returns report with grouped/ungrouped metrics, global metrics, diagnostic errors, and `optimization_metrics` including BIoU and calibration metrics

Acceptance:
- All expected keys present; values within numeric tolerances

Edge cases:
- No pixels (all ignore_index); single bin ECE; logits absent ⇒ calibration metrics gracefully skipped or neutral

### 4.10 Engine — `engine/optimizer.py`

create_optimizer_and_scheduler:
- Returns optimizer and scheduler; scheduler decreases LR as steps advance

Acceptance:
- Step a few iterations and assert LR monotonically decreases after warmup

### 4.11 Engine — `engine/inference.py`

SlidingWindowInferrer:
- `predict` and/or `predict_batch` reconstruct full‑res prediction from overlapping tiles

Acceptance:
- Output spatial dims equal input; reproducible with fixed seeds

Edge cases:
- Image dims not multiples of window/stride; border handling

### 4.12 Engine — `engine/trainer.py`

Trainer:
- One mini epoch on CPU with a tiny model/dataloader:
  - Forward/backward steps run with mixed precision disabled on CPU
  - Validation path calls metrics `update()` with logits (Tier 1); if metrics processor enabled, dispatches Tier 2 jobs per image
  - History logged; best‑model metric computed

Acceptance:
- No exceptions; metrics_calculator `compute()` returns sane values; optional processor lifecycle `start()`/`shutdown()` invoked

Edge cases:
- Gradient accumulation >1; scheduler present/absent; early stopping hooks optional

### 4.13 Engine — `engine/evaluator.py`

Evaluator:
- With a tiny test set, runs sliding‑window inference; invokes Tier 1 update and Tier 2 dispatch; writes a final report and JSONL

Acceptance:
- Final JSON exists with expected keys; JSONL contains one line per image with configured advanced metrics fields (gated by tasks)

Edge cases:
- Missing/disabled advanced processor ⇒ no JSONL; report still saved

### 4.14 Utils — `utils/task_splitter.py`

TaskSplitter/MTLTaskSplitter/BaseTaskSplitter:
- Parse minimal task definitions; create mapping arrays (global and flat)
- Verify non‑overlapping global class space; inverse mapping correctness for baseline

Acceptance:
- Known input labels transform to expected outputs; round‑trip checks where applicable

Edge cases:
- Unseen labels; empty id2label entries rejected with clear errors

### 4.15 Metrics — `metrics/metrics_storer.py`

MetricsStorer:
- `open_for_run()` creates paths; `store_epoch_history()` appends; `store_per_image_cms()` writes JSONL; `save_final_report()` writes JSON

Acceptance:
- Files exist; JSON loads without errors; fields match schemas

Edge cases:
- Validation vs test paths; concurrent writes avoided (single writer in Tier 2 handles JSONL)

AdvancedMetricsProcessor (Tier 2):
- Lifecycle: `start()` once; idempotent guard; `shutdown()` graceful; multiple `shutdown()` calls safe
- Dispatch: enqueue small `uint8` masks; workers compute configured metrics; writer emits JSONL lines
- Backpressure: many enqueues do not deadlock; queues drained; writer keeps up for small tests
- Task gating: if tasks list excludes a metric, it’s not computed/written
- Optional deps: when missing, the corresponding metrics are skipped with warnings (test by monkeypatching imports or marking `@pytest.mark.optdeps`)

Acceptance:
- After dispatching N jobs, JSONL contains N lines; each has only enabled metric keys
- No leaked processes/threads after `shutdown()`; temporary dirs empty

Edge cases:
- Zero workers (force synchronous fallback or clear error); repeated `start()` calls; invalid tasks silently ignored vs error (decide and assert contract)

### 4.16 Utils — `utils/visualization.py`

Visualizer:
- Smoke tests that plotting functions create files without raising (use tiny random data, temp dir); do not assert image content

Acceptance:
- Files exist; no exceptions

Edge cases:
- Missing optional packages (mark `@pytest.mark.optdeps` if needed)

---

## 5. Integration and Behavioral Tests

### 5.1 Test Data and Environment Setup

Test datasets:
- **Coralscapes Subset**: Small train/val/test splits (10-20 images each) from user-provided coralscapes data
- **PDS Patches**: Corresponding PDS-generated patches for training efficiency testing
- **Encoder**: Use "nvidia/mit-b0" (lightweight) for faster test execution while maintaining architectural integrity

Test environment isolation:
- Each test class runs in isolated temporary directories
- Deterministic seeds across PyTorch, NumPy, and Python random
- CPU-only execution unless marked @pytest.mark.gpu
- Automatic cleanup of generated artifacts (models, logs, outputs)

### 5.2 Core Integration Test Framework

```python
@pytest.mark.integration
class TestCoralMTLIntegration:
    """Comprehensive integration tests mimicking experiments/ patterns."""
    
    def setup_method(self):
        # Setup isolated test environment with nvidia/mit-b0 encoder
        pass
    
    def teardown_method(self):
        # Clean up generated artifacts
        pass
```

### 5.3 Basic Training/Validation Pipeline Tests

#### 5.3.1 Minimal Training Loop Verification

**Test**: `test_minimal_mtl_training_loop`
- Config: MTL model with 2 primary + 2 auxiliary tasks, 2 epochs, batch size 2
- Data: 4 train samples, 2 val samples from coralscapes subset
- Encoder: nvidia/mit-b0, decoder_channel: 128
- Device: CPU, patch_size: 64, no advanced metrics processor

**Acceptance**:
- Training completes without exceptions
- Loss values finite and logged to history.json
- Validation metrics computed (mIoU > 0, BIoU present)
- Best model checkpoint saved
- Final metrics report contains all task hierarchies (grouped/ungrouped)

**Verification Steps**:
1. Build ExperimentFactory from config dict
2. Run factory.run_training()
3. Assert history.json exists and contains epoch progression
4. Assert best_model.pth saved with finite loss
5. Verify validation metrics structure matches Technical Spec Section 5.4

#### 5.3.2 Baseline Model Training Verification

**Test**: `test_minimal_baseline_training_loop`
- Config: BaselineSegformer with 39-class flattened output
- Same data/device constraints as MTL test
- Verify baseline vs MTL training patterns are consistent

### 5.4 Advanced Configuration Stress Tests

#### 5.4.1 Extreme Task Definitions Test

**Test**: `test_extreme_task_configurations`

**Configurations tested**:

A. **Minimal Configuration**:
```yaml
task_definitions:
  genus:
    id2label: {0: "Background", 1: "Single_Genus"}
  health:
    id2label: {0: "Background", 1: "Healthy"}
```

B. **Maximal Configuration**:
```yaml
task_definitions:
  genus:
    grouped:
      Acropora: [1, 2, 3, 4, 5]  # 5 Acropora species
      Porites: [6, 7, 8]          # 3 Porites species
      Other: [9, 10, 11, 12]      # 4 other genera
  health:
    id2label: {0: "Background", 1: "Healthy", 2: "Bleached", 3: "Dead"}
  fish:
    id2label: {0: "Background", 1: "Fish"}
  substrate:
    grouped:
      Substrate: [1, 2, 3, 4, 5]  # Multiple substrate types
  human_artifacts:
    id2label: {0: "Background", 1: "Human", 2: "Tools", 3: "Trash"}
```

C. **Imbalanced Configuration**:
```yaml
task_definitions:
  genus:
    id2label: {0: "Background", 1: "Common_Genus", 2: "Rare_Genus_1", 3: "Rare_Genus_2"}
    # Simulate extreme class imbalance in synthetic data
  health:
    id2label: {0: "Background", 1: "Healthy", 2: "Bleached"}  # No dead samples
```

**Acceptance**:
- All configurations train without crashing
- Task splitters correctly parse grouped vs ungrouped structures
- Metrics computed for all defined hierarchies
- Loss components properly weighted and finite
- Memory usage remains reasonable (< 2GB for test)

#### 5.4.2 Concurrent Metrics Processor Stress Test

**Test**: `test_advanced_metrics_processor_integration`

**Configuration**:
```yaml
metrics_processor:
  enabled: true
  num_cpu_workers: 3
  tasks: ["ASSD", "HD95", "PanopticQuality", "ARI"]
```

**Stress Scenarios**:
1. **High Volume**: Dispatch 50 jobs rapidly, ensure no deadlocks
2. **Heterogeneous Tasks**: Enable/disable different metric combinations
3. **Worker Pool Variations**: Test with 1, 3, and 8 workers
4. **Graceful Shutdown**: Interrupt processing mid-stream, verify clean shutdown

**Acceptance**:
- JSONL output contains exactly one record per processed image
- All enabled metrics present in JSONL records
- No process/thread leaks after shutdown
- Memory usage bounded during high-volume dispatch
- Writer process handles concurrent queue access safely

### 5.5 End-to-End Evaluation Pipeline Tests

#### 5.5.1 Complete Evaluation Workflow

**Test**: `test_complete_evaluation_pipeline`

**Setup**:
- Train minimal model (1 epoch to get valid checkpoint)
- Prepare test set (5 images from coralscapes subset)
- Configure evaluator with sliding window inference

**Pipeline Steps**:
1. Load trained checkpoint via ExperimentFactory.run_evaluation()
2. Execute sliding window inference on test images
3. Generate final metrics report (JSON) and per-image results (JSONL)
4. Verify output file integrity and metric completeness

**Acceptance**:
- test_results.json contains hierarchical metrics structure
- test_cms.jsonl has 5 records (one per test image)
- Sliding window reconstruction preserves spatial dimensions
- All configured task metrics present in final report
- Boundary IoU and calibration metrics included when logits available

#### 5.5.2 Cross-Model Evaluation Consistency

**Test**: `test_mtl_vs_baseline_evaluation_consistency`

**Comparison Points**:
- Train both MTL and Baseline models on same data subset
- Evaluate both on same test images
- Compare global metrics where applicable (e.g., overall accuracy)
- Verify evaluation pipeline produces comparable result structures

**Acceptance**:
- Both models produce valid evaluation outputs
- Metrics computation doesn't crash for either architecture
- File output formats consistent (JSON structure, JSONL schemas)
- Performance differences explainable by architectural choices

### 5.6 Data Pipeline Integration Tests

#### 5.6.1 PDS Training Data Integration

**Test**: `test_pds_patches_integration`

**Data Sources**:
- Standard coralscapes train split
- PDS-generated patches (user-provided)
- Mixed data loading (PDS primary, standard fallback)

**Scenarios**:
1. **PDS-only Training**: Use only PDS patches for training data
2. **Hybrid Loading**: PDS available for train, standard for val/test
3. **Missing PDS Fallback**: Simulate missing PDS path, verify fallback behavior

**Acceptance**:
- DataLoader correctly prioritizes PDS when available
- Fallback to standard splits works seamlessly
- Image-mask alignment preserved across data sources
- Task splitter mappings consistent regardless of data source

#### 5.6.2 Augmentation Pipeline Robustness

**Test**: `test_augmentation_consistency_integration`

**Scenarios**:
- Extreme augmentation parameters (high rotation, extreme scales)
- Non-square patch sizes and aspect ratios
- Edge cases (single-pixel objects, all-background patches)
- Multi-task mask synchronization under heavy augmentation

**Acceptance**:
- No augmentation produces invalid masks (wrong dtype, shape mismatch)
- Geometric transforms apply consistently to all task masks
- Color transforms affect only input images, not masks
- Extreme parameters don't cause numerical instabilities

### 5.7 Configuration Robustness and Error Handling

#### 5.7.1 Configuration Validation Tests

**Test**: `test_config_validation_and_error_handling`

**Invalid Configurations**:
```python
invalid_configs = [
    # Missing required sections
    {"model": {"type": "CoralMTL"}, "data": {}},  # No task definitions
    
    # Invalid model specifications
    {"model": {"type": "InvalidModel"}},
    
    # Metrics processor configuration errors
    {"metrics_processor": {"enabled": True, "tasks": ["InvalidTask"]}},
    
    # Incompatible task/model combinations
    {"model": {"type": "BaselineSegformer"}, "tasks": {"primary": ["genus", "health"]}},
    
    # Resource constraints
    {"trainer": {"batch_size_per_gpu": 1000}},  # Memory overflow test
]
```

**Acceptance**:
- ExperimentFactory raises clear, actionable error messages
- No silent failures or undefined behavior
- Error messages reference specific config sections and expected formats
- Partial configs don't corrupt factory state

#### 5.7.2 Resource Management Tests

**Test**: `test_resource_management_integration`

**Resource Scenarios**:
1. **Memory Constraints**: Large batch sizes on small test device
2. **Disk Space**: Verify cleanup of temporary files during training
3. **Process Management**: Ensure all spawned processes (metrics workers) are properly cleaned up
4. **Exception Handling**: Interrupt training/evaluation, verify graceful cleanup

**Acceptance**:
- Memory usage doesn't exceed reasonable bounds for test environment
- Temporary files cleaned up after test completion
- No zombie processes after test teardown
- Exception during training doesn't leave corrupted state

### 5.8 Performance and Scalability Baseline Tests

#### 5.8.1 Training Performance Benchmarks

**Test**: `test_training_performance_baselines`

**Metrics Tracked**:
- Time per epoch (should be < 30s for test configuration)
- Memory peak usage (should be < 2GB on CPU)
- Disk I/O patterns (log files, checkpoint saving)
- Convergence rate (loss decrease over epochs)

**Configurations**:
- MTL vs Baseline speed comparison
- Advanced metrics processor overhead
- Different batch sizes and patch sizes
- CPU vs GPU performance (when available)

**Acceptance**:
- Performance within expected bounds for configuration size
- No memory leaks across epochs
- Metrics processor doesn't dramatically slow training
- Results reproducible with fixed random seeds

#### 5.8.2 Sliding Window Inference Scalability

**Test**: `test_sliding_window_scalability`

**Image Sizes Tested**:
- 128x128 (single patch)
- 256x256 (4 patches with overlap)
- 512x512 (16 patches with overlap)
- Non-square: 384x256, 256x384

**Acceptance**:
- Output spatial dimensions exactly match input
- Reconstruction artifacts minimal at patch boundaries
- Memory usage scales linearly with image size
- Processing time reasonable for test sizes (< 10s per image)

### 5.9 Scientific Correctness Verification

#### 5.9.1 Metrics Mathematical Correctness

**Test**: `test_metrics_mathematical_correctness`

**Synthetic Test Cases**:
```python
# Perfect prediction case
perfect_pred = torch.eye(num_classes)  # Identity matrix
perfect_target = torch.arange(num_classes)
expected_miou = 1.0

# Worst case prediction
worst_pred = torch.zeros_like(perfect_pred)
worst_pred[:, 0] = 1  # All predict background
expected_miou = 0.0

# Calibration test cases
overconfident_logits = torch.tensor([10.0, -10.0, -10.0])  # Very confident
underconfident_logits = torch.tensor([0.1, 0.1, 0.1])     # Very uncertain
```

**Acceptance**:
- Perfect predictions yield mIoU = 1.0, boundary IoU = 1.0
- Worst predictions yield mIoU = 0.0
- Calibration metrics (NLL, Brier, ECE) respond correctly to confidence levels
- Boundary statistics correctly identify edge pixels

#### 5.9.2 Loss Function Behavior Verification

**Test**: `test_loss_functions_scientific_correctness`

**Loss Behavior Tests**:
- Focal loss down-weights easy examples, emphasizes hard cases
- Dice loss optimizes IoU directly
- Uncertainty weighting adapts to task difficulty
- Class imbalance handling via weighted losses

**Synthetic Scenarios**:
- Balanced vs imbalanced synthetic datasets
- Easy vs hard synthetic classification tasks
- Consistent vs inconsistent multi-task predictions

**Acceptance**:
- Loss components behave as theoretically expected
- Gradient flow preserved through all loss components
- Uncertainty parameters converge to meaningful values
- Loss decreases monotonically on synthetic overfit scenarios

### 5.10 Regression and Stability Tests

#### 5.10.1 Model Architecture Stability

**Test**: `test_model_architecture_stability`

**Stability Checks**:
- Forward pass determinism with fixed seeds
- Gradient computation consistency
- Parameter initialization reproducibility
- Cross-attention information flow verification

**Acceptance**:
- Identical inputs produce identical outputs (determinism)
- Gradients computed consistently across runs
- Model parameters initialize to expected distributions
- Cross-attention weights sum to 1.0 and are non-negative

#### 5.10.2 Training Reproducibility

**Test**: `test_training_reproducibility`

**Reproducibility Verification**:
- Run identical training twice with same seeds
- Compare final model weights, training history, validation metrics
- Verify checkpoint loading/saving consistency
- Test across different random initialization seeds

**Acceptance**:
- Identical configurations produce identical results
- Training history matches exactly (loss values, metric progression)
- Saved and loaded models produce identical predictions
- Different seeds produce different but valid results

---

## 6. Data & Synthetic Fixtures

- Small masks with simple shapes (filled squares/rectangles) to test BIoU and IoU
- Logits contrived for calibration extremes (perfect one‑hot; uniform random)
- Minimal task definition YAML/dict with 1–2 classes per task

Provide factories to generate batches consistently across tests.

---

## 7. Coverage Targets and CI

- Line coverage goal: 85%+ overall; critical modules 90%+
- Branch coverage preferred for metrics and losses
- Mark slow/GPU/optdeps to keep default CI under 5–7 minutes
- Run `pytest -q` by default; `pytest -m "integration or slow"` for extended runs

Optional local performance tip: enable `pytest-xdist` to parallelize CPU‑bound tests.

---

## 8. Traceability to Specifications

- Technical Spec Section 5 (Engine) ⇄ Losses, Metrics, Inference, Trainer/Evaluator tests
- Technical Spec Section 6 (Utils) ⇄ TaskSplitter, MetricsStorer, Visualization tests
- Technical Spec Section 11 (Two‑Tier Architecture) ⇄ AdvancedMetricsProcessor lifecycle/concurrency tests; Tier 1 calibration/boundary accumulators
- Theoretical Spec Sections 6–8 ⇄ Boundary‑aware metrics, calibration metrics rationale, hierarchical evaluation methodology

This mapping ensures that verification covers both implementation contracts and scientific intent.

---

## 9. Running Tests

Basic:

```bash
# Windows cmd
pytest -q
```

With markers:

```bash
# Windows cmd
pytest -m integration -q
pytest -m gpu -q
pytest -m optdeps -q
```

To see durations and verbose output:

```bash
# Windows cmd
pytest -vv --durations=15
```

---

## 10. Maintenance Guidelines

- Add tests with any public API change (new args/returns)
- Keep synthetic fixtures small and deterministic
- Prefer parametrized tests for class‑count and shape variations
- For concurrency tests, enforce timeouts and clean shutdowns in `teardown`
- Document any intentional deviations in this file and link to corresponding PRs

---

This specification should guide building and maintaining a robust test suite that validates both software engineering quality and scientific validity of Coral‑MTL.