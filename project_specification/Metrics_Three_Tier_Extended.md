# Three-Tier Metric System Implementation

## Overview
three-tier system for segregated, concurrent computation in the coral segmentation pipeline. The system maximizes GPU utilization for neural network operations while leveraging massive CPU parallelism for complex metric computations.

## Architecture Components

### Tier 1: Real-time GPU Aggregation Engine (The "Collector")
**Location:** Enhanced `AbstractCoralMetrics` class
**Purpose:** GPU-based accumulation of fast metrics for real-time model selection

**New Features Added:**
- GPU-based accumulators for boundary statistics (TP, FP, FN)
- Probabilistic statistics computation (NLL, Brier Score, ECE)
- Enhanced `reset()` method with new tensor accumulators
- Updated `update()` signature to accept raw logits
- New GPU updater methods:
  - `_update_boundary_stats_gpu()` - Boundary F1 computation
  - `_update_probabilistic_stats_gpu()` - ECE, NLL, Brier scores
- Enhanced `compute()` method with new Tier 1 metrics

**New Metrics Available:**
- `global.Boundary_F1`, `global.Boundary_Precision`, `global.Boundary_Recall`
- `global.NLL`, `global.Brier_Score`, `global.ECE`
- All existing metrics (mIoU, BIoU, TIDE errors) remain unchanged

### Tier 2: Asynchronous Dispatcher & I/O Engine (The "Quarterback") 
**Location:** New `AdvancedMetricsProcessor` class in `metrics_storer.py`
**Purpose:** Multi-process job dispatch and I/O management

**Key Features:**
- Multiprocessing Manager with shared job and results queues
- Configurable CPU worker pool (default: 30 workers)
- Dedicated I/O writer process to prevent bottlenecks
- Non-blocking `dispatch_image_job()` method
- Graceful shutdown with proper resource cleanup
- Automatic conversion of tensors to NumPy arrays

### Tier 3: Multi-Process CPU Worker Pool (The "Workhorses")
**Location:** `run_metric_gauntlet()` function and helper functions
**Purpose:** Comprehensive per-image metric computation

**Advanced Metrics Implemented:**
- **Surface Distance Metrics:** ASSD, HD95 using scipy distance transforms
- **Clustering Metrics:** Adjusted Rand Index (ARI), Variation of Information (VI)
- **Panoptic Statistics:** Per-image PQ component computation (TP, FP, FN, IoU sums)
- Extensible framework for additional metrics

## Integration Points

### Updated Classes

**1. AbstractCoralMetrics (Enhanced)**
- New constructor parameters for Tier 1 accumulators
- Modified `update()` signature with `predictions_logits` parameter
- Both `CoralMTLMetrics` and `CoralMetrics` updated accordingly

**2. Trainer (Enhanced)**
- New constructor parameter: `metrics_processor`
- Enhanced `_validate_one_epoch()` with dual-tier dispatch
- Updated `train()` method with processor lifecycle management
- Non-blocking validation loop with concurrent GPU and CPU processing

**3. Evaluator (Enhanced)** 
- New constructor parameter: `metrics_processor`
- Enhanced `evaluate()` method with dual-tier system
- Comprehensive reporting for both Tier 1 and Tier 2 outputs
- Proper resource management and shutdown

**4. ExperimentFactory (Enhanced)**
- New method: `get_advanced_metrics_processor()`
- Updated `run_training()` and `run_evaluation()` to inject processor
- Configuration parsing for `metrics_processor` section

## Configuration System

### New Configuration Section: `metrics_processor`
```yaml
metrics_processor:
  enabled: true                    # Master switch
  num_cpu_workers: 30             # CPU process pool size  
  tasks: ["ASSD", "HD95", "PQ", "ARI", "VI"]  # Enabled metrics
```

### Updated Configuration Files
- `configs/three_tier_baseline_config.yaml` - Full demonstration config
- `configs/baseline_segformer.yaml` - Updated with new section (disabled by default)

## Data Flow Architecture

```
┌─────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│   Tier 1: GPU   │    │  Tier 2: Dispatch   │    │ Tier 3: CPU Pool   │
│   Collector     │    │     Quarterback     │    │    Workhorses      │
└─────────────────┘    └─────────────────────┘    └─────────────────────┘
         │                        │                          │
    ┌────▼────┐              ┌────▼────┐              ┌─────▼─────┐
    │ Logits  │              │  Masks  │              │   Jobs    │
    │   ↓     │              │   ↓     │              │     ↓     │
    │GPU Ops  │              │Job Queue│              │  Worker   │
    │   ↓     │              │   ↓     │              │ Processes │
    │Fast     │              │CPU-GPU  │              │     ↓     │
    │Metrics  │              │Transfer │              │ Advanced  │
    └─────────┘              │   ↓     │              │ Metrics   │
                             │Results  │              │     ↓     │
                             │Queue    │              │I/O Writer │
                             └─────────┘              └───────────┘
```

## Benefits Achieved

### Performance Benefits
- **GPU Efficiency:** 100% GPU focus on neural network computation
- **Massive Parallelism:** 30+ CPU cores working concurrently
- **Non-blocking Operations:** Training never waits for advanced metrics
- **Scalable:** Linear scaling with available CPU cores

### Architecture Benefits  
- **Separation of Concerns:** Clear division between fast and comprehensive metrics
- **Resource Isolation:** GPU and CPU workloads completely separated
- **Fault Tolerance:** Worker failures don't affect training
- **Memory Efficiency:** Minimal GPU-CPU data transfer

### User Experience Benefits
- **Two-Tiered Output:** Both real-time feedback and comprehensive analysis
- **Configurable:** Can disable advanced processing for faster runs
- **Backward Compatible:** Existing configs work with `enabled: false`
- **Comprehensive Reporting:** Clear distinction between Tier 1 and Tier 2 results

## Output Files Generated

### During Training/Validation
- `validation_cms.jsonl` - Per-image confusion matrices (existing)
- `advanced_metrics.jsonl` - Per-image advanced metrics (new, Tier 2)
- `history.json` - Epoch-wise training history with new Tier 1 metrics

### During Evaluation  
- `test_metrics_full_report.json` - Comprehensive Tier 1 report
- `test_cms.jsonl` - Per-image confusion matrices
- `advanced_metrics.jsonl` - Per-image advanced metrics (Tier 2)

## Usage Examples

### Enable Three-Tier System
```yaml
metrics_processor:
  enabled: true
  num_cpu_workers: 30
  tasks: ["ASSD", "HD95", "PanopticQuality", "ARI"]
```

### Disable for Faster Training
```yaml
metrics_processor:
  enabled: false
```

### Select Tier 1 Metrics for Model Selection
```yaml
trainer:
  model_selection_metric: "global.Boundary_F1"  # or ECE, NLL, etc.
```

## Implementation Status: ✅ COMPLETE

All seven planned components have been successfully implemented:
1. ✅ Analyzed existing codebase structure
2. ✅ Upgraded AbstractCoralMetrics to Tier 1 Engine  
3. ✅ Transformed AsyncMetricsStorer to AdvancedMetricsProcessor
4. ✅ Updated Trainer orchestration logic
5. ✅ Updated Evaluator for dual-tier dispatch
6. ✅ Updated ExperimentFactory configuration
7. ✅ Created configuration schema updates

The three-tier metric evaluation architecture is now ready for deployment and provides a scalable, efficient solution for comprehensive coral segmentation evaluation.