from pathlib import Path
import sys
import yaml
import torch

# Make 'src' importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coral_mtl.utils.task_splitter import MTLTaskSplitter
from coral_mtl.metrics.metrics_storer import MetricsStorer
from coral_mtl.metrics.metrics import CoralMTLMetrics

# 1) Load task definitions
defs_path = ROOT / "tests" / "configs" / "tasks" / "task_definitions.yaml"
with defs_path.open("r") as f:
    task_defs = yaml.safe_load(f)

# 2) Build splitter + metrics
splitter = MTLTaskSplitter(task_defs)
print(f"Initialized TaskSplitter with {len(splitter.hierarchical_definitions)} tasks and "
      f"a unified global space of {splitter.num_global_classes} classes.")

storer = MetricsStorer(output_dir=str(ROOT / "tests" / "mock_dir" / "micro_metrics"))
metrics = CoralMTLMetrics(
    splitter=splitter, storer=storer,
    device=torch.device("cpu"),
    boundary_thickness=2, ignore_index=255,   # keep 255 so we don't ignore background inadvertently
    use_async_storage=False
)

# 3) Tiny batch + per-head logits (use grouped size if grouped)
B, H, W = 1, 16, 16
logits = {}
for task, det in splitter.hierarchical_definitions.items():
    C = len(det['grouped']['id2label']) if det.get('is_grouped') else len(det['ungrouped']['id2label'])
    logits[task] = torch.randn(B, C, H, W)

# 4) Choose one ORIGINAL foreground id and one ORIGINAL background id
gm = splitter.global_mapping_array  # numpy: original -> global
o_star = next(i for i, g in enumerate(gm) if g > 0)  # any foreground
o_bg = next(i for i, g in enumerate(gm) if g == 0)   # a background-mapped original

# 5) Create a two-region target to induce a boundary (left=foreground, right=background)
original_targets = torch.empty((B, H, W), dtype=torch.long)
original_targets[:, :, :W // 2] = o_star
original_targets[:, :, W // 2:] = o_bg

# 6) Make each head's logits match those targets pixelwise
for task, det in splitter.hierarchical_definitions.items():
    if det.get('is_grouped'):
        idx_map = det['grouped']['mapping_array']      # original -> grouped idx
    else:
        idx_map = det['ungrouped']['mapping_array']    # original -> ungrouped idx
    c_star = int(idx_map[o_star])
    c_bg = int(idx_map[o_bg])

    L = logits[task]
    L[:] = -6.0
    L[:, c_star, :, :W // 2] = 6.0
    L[:, c_bg,   :, W // 2:] = 6.0

# 7) Run update/compute
metrics.reset()
metrics.update(
    predictions={}, original_targets=original_targets, image_ids=["dummy-0"],
    epoch=0, predictions_logits=logits, store_per_image=False, is_testing=False
)
report = metrics.compute()

print("global mIoU:", report["optimization_metrics"]["global.mIoU"])
print("global BIoU:", report["optimization_metrics"]["global.BIoU"])
