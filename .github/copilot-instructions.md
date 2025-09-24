---
description: High-level architectural and coding guidance for AI assistants operating
  in this repo.
status: stable
tags:
- architecture
- instructions
- style
title: Repository Copilot Instructions
---

## Coral-MTL Project – AI Coding Agent Guide

Purpose: Enable immediate productive assistance on this hierarchical multi‑task coral segmentation project for coral reef health assessment. Focus on concrete, existing patterns—avoid inventing new abstractions unless extending a documented seam.

**Project Context**: Multi-task learning for coral reef segmentation with hierarchical label structure (genus/health as primary tasks, fish/human_artifacts/substrate as auxiliary). Uses SegFormer backbones with custom MTL decoders and sophisticated metrics including boundary IoU, calibration metrics, and advanced per-image measurements.

See also:
- `docs/AGENTS_SPEC.md` (agent contracts & multi-agent orchestration)
- `docs/AGENT_CHAINING.md` (practical multi-agent conversation & CLI chaining guide)
- `project_specification/*.md` (technical requirements & architecture documentation)

### 1. Mental Model / Architecture
- Central orchestrator: `ExperimentFactory` (dependency injection + lazy caching). Always prefer calling its getters (`get_model()`, `get_dataloaders()`, etc.) instead of re‑instantiating components.
- Two model families:
	- `CoralMTLModel` (multi‑task: primary tasks = genus, health; auxiliary tasks = fish, human_artifacts, substrate unless config overrides)
	- `BaselineSegformer` (single flattened label space via `BaseTaskSplitter`)
- Configuration is YAML driven (see `configs/*.yaml`). Relative paths are resolved to absolute in `ExperimentFactory._resolve_config_paths()`.
- Task semantics + class counts sourced exclusively from `configs/task_definitions.yaml` via `MTLTaskSplitter` or `BaseTaskSplitter` → never hard‑code class numbers.
- Metrics system is tiered:
	- Tier 1 (GPU, real‑time): confusion matrices, mIoU, Boundary IoU, calibration (NLL, Brier, ECE)
	- Tier 2/3 (async CPU): advanced per‑image metrics (ASSD, HD95, PanopticQuality, ARI, etc.) via `AdvancedMetricsProcessor` (optional, config gated)
- Output hierarchy lives under experiment `output_dir` (training) + `evaluation/` (test). Key files: `history.json`, `validation_cms.jsonl`, `advanced_metrics.jsonl`, final reports.

### 2. Key Conventions & Patterns
- Factory caching contract: each `get_*` builds once; reuse the instance. Do not bypass (prevents duplicated schedulers / multiple workers).
- Loss selection:
	- MTL: `CoralMTLLoss` (CompositeHierarchical) → returns dict of component losses + overall; includes uncertainty weighting + consistency.
	- Baseline: `CoralLoss` (HybridLoss) or default CE fallback.
- Task hierarchy: Primary tasks (genus, health) drive model selection metrics. Auxiliary tasks (fish, human_artifacts, substrate) computed but don't affect optimization. Never hardcode task lists—always source from `task_definitions.yaml`.
- Optimizer pattern: only `AdamWPolyDecay` via `create_optimizer_and_scheduler`; warmup steps derived as `int(total_steps * warmup_ratio)`.
- Augmentation: `SegmentationAugmentation` (train only). For non‑train splits pass `augmentations=None`.
- Device selection: `trainer.device == 'auto'` → resolved inside factory; don’t duplicate logic.
- Model class counts are derived dynamically from splitter (`hierarchical_definitions['<task>']['ungrouped']['id2label']`). Changes to task definitions propagate automatically if you follow this pattern.
- Advanced metrics must be explicitly enabled under `metrics_processor.enabled: true`; otherwise code should gracefully treat processor as `None`.
- Project setup: Install as editable package `pip install -e .` from project root for proper imports. Use `src/` layout with namespace packages under `coral_mtl/`.

### 3. Extension Seams (safe places to add code)
- New model type: add class under `src/coral_mtl/model/`, then extend `ExperimentFactory.get_model()` switch.
- New loss: implement in `engine/losses.py`, wire in `get_loss_function()` via `loss.type`.
- New advanced metric: extend worker gauntlet in `metrics_storer.py`; gate on `enabled_tasks` string; ensure missing dependency fails soft (warn + skip).
- Add config field: parse in `ExperimentFactory` (path resolution if it’s a path), thread through to Trainer/Evaluator via `SimpleNamespace`.

### 4. Metrics & Logging Nuances
- Calibration metrics require passing logits into metrics `update()`; preserve existing signature when modifying validation/eval loops.
- Boundary metrics use `boundary_thickness` from metrics config (default 2). If adding edge metrics, reuse existing accumulators pattern.
- Tier 2 jobs: dispatch one per image (not batch) using argmax’d masks; keep payload minimal (uint8 arrays).
- Model selection metric comes from `trainer.model_selection_metric` (e.g., `optimization_metrics.H-Mean` or `global.Boundary_F1`). Always write new metrics under a stable namespace if you want them selectable.

### 5. Testing Expectations (pytest)
- Fixtures derive real task definitions (`tests/conftest.py`); don’t hard‑code alternative dummy label sets unless adding a new fixture.
- Add markers (`integration`, `gpu`, `optdeps`) if introducing heavier tests; update `pytest.ini` only if adding new global markers.
- Shape contracts: MTL forward returns dict[str, Tensor (N,C,H,W)]; baseline returns single Tensor. Enforce in new tests.

### 6. Common Pitfalls (avoid these)
- Recomputing optimizer/scheduler after training starts (breaks warmup math). Use cached objects.
- Hard‑coding class counts or task names → always inspect splitter.
- Performing CPU heavy metrics inside training loop instead of dispatching to processor.
- Writing files directly into output dir without using `MetricsStorer` (risks inconsistent naming / collisions).
- Forgetting to move new modules to device in `run_training` / `run_evaluation` (follow existing pattern: get, then move just before Trainer/Evaluator instantiation).

### 7. Quick Recipes
- Minimal training run (programmatic):
	```python
	factory = ExperimentFactory(config_path='configs/baseline_segformer.yaml')
	factory.run_training()
	```
- Evaluation with explicit checkpoint:
	```python
	factory = ExperimentFactory(config_path='configs/baseline_segformer.yaml')
	factory.run_evaluation(checkpoint_path='experiments/your_run/best_model.pth')
	```
- Enable advanced metrics in a config snippet:
	```yaml
	metrics_processor:
		enabled: true
		num_cpu_workers: 8
		tasks: ["ASSD", "HD95"]
	```

### 8. When Adding Features
- Maintain lazy instantiation & caching.
- Prefer extending config + factory over ad‑hoc environment variables.
- Keep new per‑image outputs streamable (JSONL) instead of aggregating in RAM.
- Log warnings (not errors) for optional dependency absence in advanced metrics.

### 9. Style & Quality
- Follow existing naming (PascalCase classes, snake_case functions, descriptive but concise).
- Avoid large monolithic functions; mirror existing modular segmentation of responsibilities.
- Preserve backward compatibility of public factory method signatures.

### 10. Development Workflows & Scripts
- **PDS Dataset Creation**: Use `pds_launcher/pds_simple_script.py` for Poisson Disk Sampling dataset generation. Config in `pds_launcher/pds_config.py`.
- **Training**: `ExperimentFactory(config_path).run_training()` or direct script execution. All configs in `configs/` directory.
- **Evaluation**: Use factory's `run_evaluation(checkpoint_path)` method; outputs to `experiments/{run_name}/evaluation/`.
- **Multi-Agent Orchestration**: `scripts/agent_orchestrator.py` chains reformatter → refactorer → testing workflows. See `copilot-agents/agent-manifests/` for agent contracts.

### 11. Testing Strategy & Markers
- Fixtures use real task definitions from `tests/configs/tasks/task_definitions.yaml` (not hardcoded).
- Test markers: `@pytest.mark.gpu`, `@pytest.mark.integration`, `@pytest.mark.optdeps`, `@pytest.mark.slow`.
- Run subset: `pytest -m "not integration"` (unit only), `pytest -m "integration"` (slower tests).
- Coverage: Use `coverage run -m pytest` then `coverage report -m` for detailed line coverage.
- Random seeds fixed in `tests/conftest.py` for reproducible testing.

### 12. Multi-Agent Architecture
- **Agent Roles**: MD Reformatter, Code Refactorer (3-phase: plan→confirm→execute), Testing Agent, Orchestrator.
- **Validation Chain**: All agents produce JSON manifests; validate with `scripts/validate_*.py` scripts.
- **Refactoring Gate**: Phase 1 analysis → mandatory human confirmation → Phase 2 execution → Phase 3 validation.
- **Safety**: Dual import resolution prevents broken dependencies; all structural changes require explicit confirmation.

### 13. Ask If Unsure
If a change touches: task splitting semantics, metrics accumulator layouts, training loop control flow, or multi-agent orchestration contracts—surface for review (they have cascading effects on evaluation reproducibility and agent behavior).

---
Feedback welcome: highlight unclear sections or missing edge cases to refine these instructions.

Reference: Extended agent & pipeline contracts in `docs/AGENTS_SPEC.md`.
