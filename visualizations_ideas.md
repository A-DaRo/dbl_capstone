# Publication-focused visuals for Coral-MTL — complete, reproducible guide

Purpose: a one-stop, implementation-focused catalogue covering all required visualization ideas (A–F) with concrete inputs/outputs and code snippets that operate on artifacts under `experiments/`. The figures are aligned with the poster (`latex/Poster_Data_shallange/poster1.tex`) and methodology report (`latex/Methodology/final-report.tex`).

General notes
- Prefer vector export (PDF/SVG) for lines/text; rasterize only image layers (e.g., `imshow`) and save PNG for the poster when needed.
- Source metrics exclusively from `experiments/baseline_comparisons/**` and `configs/task_definitions.yaml`. Do not hard-code class counts or task lists.
- For per-pixel predictions or probability-based plots, use CPU inference and `best_model.pth` with `ExperimentFactory`; keep samples small (e.g., 100–200 images) for speed.

Artifacts (per run)
- history.json — epoch-wise losses and Tier-1 metrics (mIoU, BIoU, Boundary-F1, etc.).
- test_metrics_full_report.json — final global and per-task metrics; may include grouped/per-class summaries.
- test_cms.jsonl — per-image confusion matrices (global or per task depending on config).
- advanced_metrics.jsonl — per-image Tier-2/3 metrics (ASSD, HD95) if enabled.
- loss_diagnostics.jsonl — optional gradient stats for IMGrad/diagnostics.
- best_model.pth — checkpoint for optional CPU inference.

Export folders
- Poster: `latex/Poster_Data_shallange/Result-figures/`
- Report: `latex/Methodology/Result-figures/`

Helper: CPU inference pattern
```python
import torch
from coral_mtl.ExperimentFactory import ExperimentFactory

def load_model_for_inference(config_path: str, checkpoint_path: str, device: str='cpu'):
  factory = ExperimentFactory(config_path=config_path)
  model = factory.get_model()
  state = torch.load(checkpoint_path, map_location=device)
  model.load_state_dict(state)
  model.to(device); model.eval()
  dls = factory.get_dataloaders()
  return model, dls, factory
```

## A — Basic statistical & performance summaries

### 1) Train / Val curves (loss, metric vs epoch)
- What: Line plots of training loss, validation loss, and core metrics (mIoU, BIoU, Boundary-F1).
- Why: Convergence, stability, relative performance.
- Library: Matplotlib/Seaborn.
- Inputs: `experiments/.../history.json`
- Outputs: `latex/Poster_Data_shallange/Result-figures/training_progress_3models.png`
```python
import json, pathlib
import pandas as pd, matplotlib.pyplot as plt, seaborn as sns

def load_history(path, model):
  d = json.load(open(path))
  rows = []
  for ep, rec in enumerate(d.get('epochs', d), start=1):
    rows.append({'epoch': ep, 'model': model,
           'mIoU': rec.get('global.mIoU') or rec.get('mIoU'),
           'BIoU': rec.get('global.BIoU') or rec.get('BIoU'),
           'Boundary_F1': rec.get('global.Boundary_F1') or rec.get('Boundary_F1'),
           'loss': rec.get('loss.total') or rec.get('loss')})
  return pd.DataFrame(rows)

hist_paths = {
  'Baseline': 'experiments/baseline_comparisons/coral_baseline_b2_run/history.json',
  'MTL Focused': 'experiments/baseline_comparisons/coral_mtl_b2_focused_run/history.json',
  'MTL Holistic': 'experiments/baseline_comparisons/coral_mtl_b2_holistic_run/history.json',
}
df = pd.concat([load_history(p, m) for m,p in hist_paths.items()])
fig, axes = plt.subplots(1,3, figsize=(11,3), sharex=True)
for i,(y,lab) in enumerate([('mIoU','Global mIoU'),('BIoU','Global BIoU'),('Boundary_F1','Boundary F1')]):
  sns.lineplot(data=df, x='epoch', y=y, hue='model', ax=axes[i])
  axes[i].set_ylabel(lab); axes[i].grid(True, alpha=0.3)
for ax in axes: ax.set_xlabel('Epoch')
out = pathlib.Path('latex/Poster_Data_shallange/Result-figures/training_progress_3models.png')
out.parent.mkdir(parents=True, exist_ok=True); fig.tight_layout(); fig.savefig(out, dpi=300)
```

### 2) Per-class performance (bar + violin/box)
- What: Per-class IoU bars with optional per-image distribution (violin/box).
- Why: Identify failing genera/health classes and variability.
- Inputs: `test_metrics_full_report.json` (per-class/grouped IoU) OR derive per-image IoU from `test_cms.jsonl`.
- Outputs: `latex/Methodology/Result-figures/per_class_iou_bars.png`
```python
import json, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
paths = {
  'Baseline': 'experiments/baseline_comparisons/coral_baseline_b2_run/test_metrics_full_report.json',
  'MTL Focused': 'experiments/baseline_comparisons/coral_mtl_b2_focused_run/test_metrics_full_report.json',
  'MTL Holistic': 'experiments/baseline_comparisons/coral_mtl_b2_holistic_run/test_metrics_full_report.json',
}
rows = []
for model,p in paths.items():
  rep = json.load(open(p))
  # Prefer grouped genus; fallback to per_class
  pc = rep.get('tasks',{}).get('genus',{}).get('grouped',{}).get('label2iou') \
     or rep.get('tasks',{}).get('genus',{}).get('per_class',{})
  if pc:
    for lab,val in pc.items(): rows.append({'Model':model,'Class':lab,'IoU':val})
df = pd.DataFrame(rows)
fig, ax = plt.subplots(figsize=(9,4));
sns.barplot(data=df, x='Class', y='IoU', hue='Model', ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=7)
ax.grid(True, axis='y', alpha=0.3); fig.tight_layout();
fig.savefig('latex/Methodology/Result-figures/per_class_iou_bars.png', dpi=300)
```

## B — Classification / calibration diagnostics

### 3) Confusion matrix (normalized)
- What: Heatmap of predicted vs true (row-normalized recall).
- Why: Shows systematic confusions.
- Inputs: `test_cms.jsonl`, `configs/task_definitions.yaml`
- Outputs: `latex/Methodology/Result-figures/confusion_genus_*.png`
```python
import json, yaml, numpy as np, matplotlib.pyplot as plt, seaborn as sns
def load_id2label(task='genus'):
  conf = yaml.safe_load(open('configs/task_definitions.yaml'))
  return conf['hierarchical_definitions'][task]['ungrouped']['id2label']
def sum_cm(jsonl):
  M=None
  for line in open(jsonl):
    cm = np.array(json.loads(line)['confusion_matrix'])
    M = cm if M is None else M+cm
  return M
M = sum_cm('experiments/baseline_comparisons/coral_mtl_b2_holistic_run/test_cms.jsonl')
labels = [load_id2label('genus')[str(i)] for i in range(M.shape[0])]
R = M/(M.sum(1,keepdims=True)+1e-9)
fig,ax=plt.subplots(figsize=(7,6)); sns.heatmap(R, cmap='viridis', ax=ax, cbar_kws={'label':'Recall'})
ax.set_xticklabels(labels, rotation=90, fontsize=7); ax.set_yticklabels(labels, fontsize=7);
ax.set_xlabel('Pred'); ax.set_ylabel('True'); fig.tight_layout(); fig.savefig('latex/Methodology/Result-figures/confusion_genus_holistic.png', dpi=300)
```

### 4) Precision–Recall / ROC (per-class; micro/macro)
- What: PR curves per selected classes and micro/macro averages.
- Why: PR is informative for imbalance.
- Inputs: CPU inference to collect per-pixel probabilities for a sample (or stored logits if available).
- Outputs: `latex/Methodology/Result-figures/pr_curves_selected.png`
```python
import numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

# Assume we collected per-pixel class probabilities P (N,C) and true labels y (N,) for a sample
# For segmentation, flatten across pixels of sampled images. Limit N for speed.
def pr_for_class(P, y, c):
  y_bin = (y==c).astype(np.uint8)
  prec, rec, _ = precision_recall_curve(y_bin, P[:,c])
  ap = average_precision_score(y_bin, P[:,c])
  return prec, rec, ap

classes_to_plot = [0,1,2]  # example indices
fig, ax = plt.subplots(figsize=(5,4))
for c in classes_to_plot:
  prec, rec, ap = pr_for_class(P, y, c)
  ax.plot(rec, prec, label=f'class {c} (AP={ap:.2f})')
ax.set_xlabel('Recall'); ax.set_ylabel('Precision'); ax.legend(); fig.tight_layout()
fig.savefig('latex/Methodology/Result-figures/pr_curves_selected.png', dpi=300)
```

### 5) Calibration / reliability diagram (+ optional risk–coverage)
- What: Reliability diagram with ECE; optionally plot accuracy vs coverage by thresholding max-prob.
- Why: Probability honesty; matches report.
- Inputs: CPU inference to collect max-prob and correctness flags.
- Outputs: `latex/Methodology/Result-figures/reliability_diagram_3models.png`
```python
from sklearn.calibration import calibration_curve
import numpy as np, matplotlib.pyplot as plt

def reliability(ax, y_true, y_prob, n_bins=15, label=''):
  frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='quantile')
  ax.plot([0,1],[0,1],'k--',alpha=0.4); ax.plot(mean_pred, frac_pos, marker='o', label=label)

def risk_coverage(ax, correct, conf):
  # correct: 1 if correct per pixel; conf: max prob per pixel
  ths = np.linspace(0,1,21); cov=[]; acc=[]
  for t in ths:
    sel = conf>=t
    cov.append(sel.mean())
    acc.append((correct[sel].mean() if sel.any() else np.nan))
  ax.plot(cov, acc, marker='o'); ax.set_xlabel('Coverage'); ax.set_ylabel('Accuracy')
```

## C — Segmentation-specific (per-image and aggregated)

### 6) RGB + GT + Prediction overlay + boundary contour
- What: 3–4 panel qualitative: RGB, GT overlay, Pred overlay, Error map.
- Why: Staple qualitative segmentation figure.
- Inputs: CPU inference + dataset GT; consistent colormap.
- Outputs: `latex/Methodology/Result-figures/qual_panels/example.png`
```python
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries

fig, axes = plt.subplots(1,4, figsize=(12,3))
axes[0].imshow(rgb); axes[0].set_title('RGB'); axes[0].axis('off')
axes[1].imshow(rgb); axes[1].imshow(gt_mask, alpha=0.4, cmap='tab20'); axes[1].set_title('GT'); axes[1].axis('off')
axes[2].imshow(rgb); axes[2].imshow(pred_mask, alpha=0.4, cmap='tab20'); axes[2].set_title('Pred'); axes[2].axis('off')
error = (pred_mask!=gt_mask)
axes[3].imshow(error, cmap='Reds'); axes[3].set_title('Error map'); axes[3].axis('off')
bound = find_boundaries(pred_mask>0, mode='outer')
axes[2].contour(bound, colors='white', linewidths=0.5)
plt.tight_layout()
```

### 7) Boundary-focused diagnostic (BIoU / boundary distance)
- What: Signed distance difference maps and boundary band overlays.
- Why: Visualize where BIoU/Boundary-F1 gains come from.
- Inputs/Outputs: as in 6), save to `latex/Methodology/Result-figures/boundary_band_examples.png`
```python
from scipy.ndimage import distance_transform_edt as edt
import numpy as np
def signed_distance(mask):
  pos = edt(mask>0); neg = edt(mask==0); return pos - neg
```

### 8) Mean error maps across a set (tile-grid heatmap)
- What: Aggregate per-image pixel error rate onto a grid; proxy for spatial failure modes.
- Why: Identify acquisition/scene factors (shadow/turbidity zones) if grid/coords exist.
- Inputs: `test_cms.jsonl` (per-image), plus a mapping from image index→grid location if available.
- Outputs: `latex/Methodology/Result-figures/error_heatmap.png`
```python
import json, numpy as np, matplotlib.pyplot as plt
errs=[]
for line in open('experiments/baseline_comparisons/coral_mtl_b2_holistic_run/test_cms.jsonl'):
  rec = json.loads(line)
  cm = np.array(rec['confusion_matrix'])
  tp = np.diag(cm).sum(); tot = cm.sum(); errs.append(1.0 - (tp/(tot+1e-9)))
errs = np.array(errs)
# If you have (r,c) per tile, scatter into an image; else plot histogram
plt.figure(figsize=(4,3)); plt.hist(errs, bins=30); plt.xlabel('Per-image error rate'); plt.ylabel('#images'); plt.tight_layout()
plt.savefig('latex/Methodology/Result-figures/error_hist.png', dpi=300)
```

## D — Representation & embedding diagnostics

### 9) UMAP / t-SNE scatter
- What: 2D embedding of features colored by label or uncertainty.
- Why: Check separability and failure clusters.
- Inputs: CPU inference extracting pooled encoder features for a small image set (e.g., GAP of last encoder stage) and GT labels.
- Outputs: `latex/Methodology/Result-figures/umap_scatter.png`
```python
# Optional dependency: pip install umap-learn
import numpy as np, matplotlib.pyplot as plt
try:
  import umap
except ImportError:
  from sklearn.manifold import TSNE as UMAP
  umap = None

# feats: (N,D), labels: (N,)
if umap:
  emb = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine').fit_transform(feats)
else:
  emb = UMAP(n_components=2, perplexity=30).fit_transform(feats)
plt.figure(figsize=(5,4)); sc=plt.scatter(emb[:,0], emb[:,1], c=labels, s=5, cmap='tab20'); plt.colorbar(sc); plt.tight_layout()
plt.savefig('latex/Methodology/Result-figures/umap_scatter.png', dpi=300)
```

### 10) Cluster montage & representative exemplars
- What: Grid of representative thumbnails from selected clusters.
- Why: Human-readable evidence of clusters.
- Inputs: indices per cluster, underlying RGB tiles.
- Outputs: `latex/Methodology/Result-figures/cluster_montage.png`
```python
import matplotlib.pyplot as plt
def montage(images, nrows, ncols, out):
  fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*2, nrows*2))
  for ax, img in zip(axes.ravel(), images): ax.imshow(img); ax.axis('off')
  fig.tight_layout(); fig.savefig(out, dpi=300)
```

## E — Multi-Task Learning (MTL) specific

### 11) Per-task learning curves & small multiples
- What: Task metrics (genus/health mIoU) across epochs in small multiples.
- Why: Identify dominance/conflict across tasks.
- Inputs: `history.json` (if logs per-task) or extract from validation snapshots when available.
- Outputs: `latex/Methodology/Result-figures/per_task_learning_curves.png`
```python
# Extend the training-curves snippet: draw separate subplots per task using columns like 'genus.mIoU' if present.
```

### 12) Pareto/trade-off: task A vs task B
- What: Scatter genus vs health mIoU across models (and epochs optionally).
- Why: Shows trade-offs.
- Inputs: `test_metrics_full_report.json` (and optionally `history.json`).
- Outputs: `latex/Methodology/Result-figures/pareto_genus_health.png`
```python
# See snippet earlier in this file (Section A→8) for a concrete implementation.
```

### 13) Gradient conflict / cosine similarity heatmap
- What: Heatmap of task gradient cosine similarities or norm balance.
- Why: Evidence for IMGrad usefulness.
- Inputs: `loss_diagnostics.jsonl` (if enabled in training).
- Outputs: `latex/Methodology/Result-figures/gradient_conflict_heatmap.png`
```python
import json, numpy as np, seaborn as sns, matplotlib.pyplot as plt
pair2cos=[]
for line in open('experiments/baseline_comparisons/coral_mtl_b2_holistic_run/loss_diagnostics.jsonl'):
  rec = json.loads(line)
  for k,v in rec.get('cosine', {}).items(): pair2cos.append((k,v))
# Aggregate mean per pair
pairs = {}
for k,v in pair2cos: pairs.setdefault(k, []).append(v)
labels = sorted({p.split('::')[0] for p in pairs}|{p.split('::')[1] for p in pairs})
M = np.zeros((len(labels),len(labels)))
for k,vals in pairs.items():
  a,b = k.split('::'); i,j = labels.index(a), labels.index(b); M[i,j]=M[j,i]=np.mean(vals)
fig,ax=plt.subplots(figsize=(6,5)); sns.heatmap(M, xticklabels=labels, yticklabels=labels, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
fig.tight_layout(); fig.savefig('latex/Methodology/Result-figures/gradient_conflict_heatmap.png', dpi=300)
```

### 14) Task-conditioned attribution maps
- What: Grad-CAM/Integrated Gradients per task head for the same image.
- Why: Show whether tasks attend to different regions.
- Inputs: Model + one image.
- Outputs: `latex/Methodology/Result-figures/attributions_example.png`
```python
# Optional (requires captum or torchcam). Keep as opt-deps and guard imports.
```

## F — Advanced statistical summarizations

### 15) Distribution of per-patch IoU (violin + swarm)
- What: Violin plots of per-image/per-class IoU distributions.
- Why: Reveal skew and variance.
- Inputs: Derive per-image IoU from `test_cms.jsonl`.
- Outputs: `latex/Methodology/Result-figures/per_image_iou_violin.png`
```python
import json, numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt
rows=[]
for line in open('experiments/baseline_comparisons/coral_mtl_b2_holistic_run/test_cms.jsonl'):
  cm = np.array(json.loads(line)['confusion_matrix'])
  inter = np.diag(cm); union = cm.sum(1)+cm.sum(0)-np.diag(cm)
  ious = inter/(union+1e-9)
  rows.append({'IoU_mean': np.nanmean(ious)})
df = pd.DataFrame(rows)
fig,ax=plt.subplots(figsize=(4,3)); sns.violinplot(data=df, y='IoU_mean', ax=ax, cut=0); ax.set_ylabel('Per-image mean IoU')
fig.tight_layout(); fig.savefig('latex/Methodology/Result-figures/per_image_iou_violin.png', dpi=300)
```

### 16) Metric vs dataset property scatter (IoU vs size/brightness)
- What: Scatter/hexbin of IoU as a function of GT instance size or image brightness.
- Why: Diagnose dataset-dependent failure modes.
- Inputs: Per-image IoU from `test_cms.jsonl` + brightness/size from dataloader or image files.
- Outputs: `latex/Methodology/Result-figures/iou_vs_brightness.png`
```python
from PIL import Image
import numpy as np, json, matplotlib.pyplot as plt

ious=[]; br=[]
# Example assuming you can map JSONL entries to image paths in your dataloader
for rec_line in open('experiments/baseline_comparisons/coral_mtl_b2_holistic_run/test_cms.jsonl'):
  rec = json.loads(rec_line)
  cm = np.array(rec['confusion_matrix']); inter=np.diag(cm).sum(); union=cm.sum()- (cm.sum()-inter)
  iou = inter/(cm.sum()+1e-9)  # proxy: accuracy; replace by true IoU if per-class needed
  img_path = rec.get('image_path')  # if available; else derive from dataset index
  if img_path:
    arr = np.asarray(Image.open(img_path).convert('L'))
    br.append(arr.mean()/255.0)
    ious.append(iou)
plt.figure(figsize=(4,3)); plt.scatter(br, ious, s=10, alpha=0.5)
plt.xlabel('Mean brightness'); plt.ylabel('Per-image IoU (proxy)'); plt.tight_layout();
plt.savefig('latex/Methodology/Result-figures/iou_vs_brightness.png', dpi=300)
```

## Script layout (recommended)
- Put figure generators in `figures_scripts/` with one file per section above (e.g., `make_training_curves.py`, `make_confusions.py`, etc.).
- Provide a `figures_scripts/make_all_figures.py` orchestrator that calls each and writes to `latex/**/Result-figures/`.

## G — Model Architecture confront (Baseline vs MTL Focused vs MTL Holistic)

### 17) Side-by-side architecture diagram
- What: A simple, effective plot highlighting architectural differences: Baseline single-head vs MTL Focused (two primary heads with feature exchange) vs MTL Holistic (all tasks as primary, with auxiliaries). Include encoder, feature exchange block, and heads.
- Why: Quickly communicates design rationale emphasized in poster/report.
- Library: Graphviz (dot) rendered to PNG/PDF and embedded in LaTeX.
- Inputs: None (schematic), optionally derive task names from `configs/task_definitions.yaml`.
- Outputs:
  - `latex/Poster_Data_shallange/Result-figures/architecture_confront.png`
  - `latex/Methodology/Result-figures/architecture_confront.pdf`
Snippet (Graphviz via graphviz python package; fallback: write .dot and call dot):
```python
from graphviz import Digraph

def segformer_box(dot, name, label):
  dot.node(name, label, shape='box', style='rounded,filled', fillcolor='white')

def make_architecture_confront():
  dot = Digraph('G', format='png')
  dot.attr(rankdir='LR', splines='ortho', bgcolor='transparent')

  # Baseline
  with dot.subgraph(name='cluster_baseline') as c:
    c.attr(label='Baseline SegFormer', style='rounded', color='#4e79a7')
    segformer_box(c, 'b_enc', 'SegFormer B2\nEncoder')
    segformer_box(c, 'b_head', 'Single Head\n(Global classes)')
    c.edge('b_enc', 'b_head')

  # MTL Focused
  with dot.subgraph(name='cluster_focused') as c:
    c.attr(label='MTL Focused', style='rounded', color='#f28e2b')
    segformer_box(c, 'f_enc', 'SegFormer B2\nEncoder')
    segformer_box(c, 'f_xchg', 'Feature Exchange')
    segformer_box(c, 'f_genus', 'Head: Genus')
    segformer_box(c, 'f_health', 'Head: Health')
    c.edge('f_enc', 'f_xchg'); c.edge('f_xchg', 'f_genus'); c.edge('f_xchg', 'f_health')

  # MTL Holistic
  with dot.subgraph(name='cluster_holistic') as c:
    c.attr(label='MTL Holistic', style='rounded', color='#e15759')
    segformer_box(c, 'h_enc', 'SegFormer B2\nEncoder')
    segformer_box(c, 'h_xchg', 'Feature Exchange')
    for head in ['Genus','Health','Fish','Human\nArtifacts','Substrate','Background','Biota']:
      nid = f'h_{head.lower().replace(" ", "_").replace("\\n","_")}'
      segformer_box(c, nid, f'Head: {head}')
      c.edge('h_xchg', nid)
    c.edge('h_enc', 'h_xchg')

  # Arrange clusters left-to-right
  dot.edge('b_head', 'f_enc', style='invis')
  dot.edge('f_health', 'h_enc', style='invis')
  return dot

dot = make_architecture_confront()
dot.render('latex/Poster_Data_shallange/Result-figures/architecture_confront', cleanup=True)
# Also export PDF
dot.format='pdf'; dot.render('latex/Methodology/Result-figures/architecture_confront', cleanup=True)
```

Design tips
- Keep boxes minimal (title + 1–2 lines). Use distinct accent colors per variant consistent with your model colors in KPI plots.
- If Graphviz is unavailable, draw with Matplotlib + `matplotlib.patches.FancyBboxPatch` and `annotate` arrows.

## Design rules (quick)
- Consistent model colors across all plots; colorblind-safe palettes.
- Save PNG for poster and PDF for report when useful.
- Set `rcParams['pdf.fonttype']=42`; rasterize only image layers.
- Use identical y-limits across comparable panels for fair visual comparison.

This guide now fully covers all requested visualization ideas (A–F) with concrete inputs/outputs tied to `experiments/`, and includes CPU inference patterns where probabilities/predictions are required.
