# Loss & Optimization Strategy Specification

This document formalizes the theoretical foundations, design rationale, and practical usage guidelines for the multi-task loss weighting and gradient-based optimization strategies implemented in Coral-MTL. It complements the broader `theorethical_specification.md` by focusing specifically on **task interaction dynamics** during training: *weighting*, *conflict mitigation*, *imbalance correction*, and *fairness enforcement*.

---
## 1. Problem Framing: Why Task Interaction Matters
Multi-Task Learning (MTL) introduces two central optimization pathologies:

1. **Gradient Imbalance** – Some task gradients have much larger magnitudes than others, dominating the shared parameter updates (hurting weaker tasks).
2. **Gradient Conflict** – Task gradients point in opposing directions (negative cosine similarity), so improving one task harms another.

Advanced strategies attempt to either *reweight losses* (loss-space intervention) or *reconstruct the joint update direction* (gradient-space intervention). Coral-MTL supports both classes.

| Dimension | Loss-Based (e.g., Uncertainty) | Gradient-Based (e.g., IMGrad, Nash-MTL, PCGrad) |
|-----------|--------------------------------|-----------------------------------------------|
| Backward Passes per Step | 1 | >= number of tasks (cached partial grads) |
| Handles Imbalance | Indirectly | Directly |
| Handles Conflict | No | Yes (PCGrad), Partially (IMGrad), Implicitly (Nash-MTL) |
| Scale Invariance | No | Yes (Nash-MTL), Partial (IMGrad) |
| Overhead | Low | Moderate–High |

---
## 2. Strategy Catalogue

### 2.1 Uncertainty Weighting (Kendall et al., 2018)
**Type:** Loss-based (single backward pass)  
**Goal:** Let the model *learn* relative task weights via homoscedastic uncertainty.  
**Formulation:**
$$L_{total} = \sum_i \exp(-s_i) L_i + 0.5 s_i, \quad s_i = \log(\sigma_i^2)$$
`s_i` are learned scalar parameters (log-variances).  
**Intuition:** High-uncertainty tasks (noisier or harder) receive lower effective weight (precision = exp(-s_i)). The regularizer prevents trivial inflation of variance.
**When to Use:** Always as a *baseline*. Low overhead, strong robustness.
**Diagnostics Logged:** `task_weights`, `log_variances`.

### 2.2 PCGrad (Yu et al., 2020a)
**Type:** Optimizer wrapper (post-loss, gradient surgery)  
**Goal:** Remove mutually harmful directions between task gradients.  
**Rule:** For a conflicting pair (g_i, g_j) with \(<g_i, g_j> < 0\):
$$g_i \leftarrow g_i - \frac{\langle g_i, g_j \rangle}{\|g_j\|^2} g_j$$
**Aggregation:** Average of projected gradients.  
**When to Use:** You observe **frequent negative cosine similarity** spikes (< -0.5) in diagnostics. Can stack with Uncertainty weighting.  
**Strength:** Simple, orthogonal add-on.  
**Limit:** Does not address imbalance (different magnitudes) directly.
**Diagnostics Logged:** `gradient_cosine_similarity` (indirect evidence of reduced conflicts post-projection).

### 2.3 IMGrad (Zhou et al., 2025)
**Type:** Gradient-based blended direction (imbalance-aware)  
**Goal:** Adaptively blend *average-progress* and *minimum-norm fairness* directions based on imbalance severity.  
**Update:**
$$d = (1 - \alpha) g_0 + \alpha g_m$$
where `g_0` = mean gradient, `g_m` = MGDA solution (minimum-norm point in task gradient convex hull).  
Blending factor: \( \alpha = f(\cos \theta) \), with \( \theta = \angle(g_0, g_m) \).  
**Intuition:** If gradients are very imbalanced, `g_m` deviates strongly from `g_0`, so `\cos\theta` is small -> larger contribution from fairness-corrective component.  
**When to Use:** Strong **magnitude imbalance** while conflicts are moderate.  
**Requirements:** Optional QP solver (`cvxopt`) for exact MGDA; PGD fallback.  
**Diagnostics Logged:** `imgrad_cos_theta`, `gradient_norm`, `gradient_update_norm`.

### 2.4 Nash-MTL (Navon et al., 2022)
**Type:** Gradient-based game-theoretic fairness  
**Goal:** Proportionally fair bargaining solution invariant to scaling of losses.  
**Fixed Point System:** Solve for non-negative weights a:
$$G^T G a = 1 / a$$
Then update direction: \( d = G a \).  
**Practical Implementation:** Uses CCP (`cvxpy`) when available; iterative fallback approximation.  
**When to Use:** Severe cross-task scale disparity; you need *stable fairness* and invariance.  
**Trade-off:** Higher computational cost mitigated via `update_frequency` (reuse weights for several steps).  
**Diagnostics Logged:** `nash_objective`, `gradient_norm`, `gradient_update_norm`.

---
## 3. Diagnostic-Driven Strategy Selection
Diagnostics are continuously written to `validation/loss_diagnostics.jsonl` (or training logs). Key fields:

| Field | Meaning | Use For |
|-------|---------|---------|
| `gradient_norm` | Per-task raw gradient magnitudes | Detect imbalance |
| `gradient_cosine_similarity` | Pairwise cosine similarity map | Detect conflict |
| `gradient_update_norm` | Norm of final applied update | Compare stability across strategies |
| `imgrad_cos_theta` | Angle proxy between g0 & gm | Assess imbalance severity (IMGrad) |
| `task_weights` | Effective scalar multipliers | Inspect weighting evolution (Uncertainty) |

**Workflow:**
1. Start with `Uncertainty`.
2. If some tasks' `gradient_norm` >> others (persistent order(s) of magnitude) → try `NashMTL` or `IMGrad`.
3. If many negative `gradient_cosine_similarity` entries (< -0.3 to -0.5) → enable `PCGrad` (optionally keep `Uncertainty`).
4. If both phenomena: prefer `NashMTL` (robust) or experiment with `IMGrad` vs `NashMTL` + PCGrad.

---
## 4. Computational Considerations
| Strategy | Extra Backward Passes | Solver Dependency | Suggested `update_frequency` | Notes |
|----------|-----------------------|-------------------|------------------------------|-------|
| Uncertainty | 0 (single pass) | None | N/A | Baseline always run first |
| PCGrad | 0 (wrapping only) | None | N/A | Adds projection loop over tasks |
| IMGrad | ~T (per-task grads) | `cvxopt` optional | 1 | PGD fallback slower but dependency-free |
| Nash-MTL | ~T (cached) | `cvxpy` optional | 5–20 | Increase frequency for speed |

`T` = number of tasks providing gradients (usually primary + auxiliary depending on inclusion policy).

Mixed precision: ensure per-task backward uses scaled loss if `use_mixed_precision=true`; current implementation routes through trainer utilities that already handle gradient scaler.

---
## 5. Configuration Snippets
### 5.1 Baseline (Uncertainty)
```yaml
loss:
  type: "CompositeHierarchical"
  weighting_strategy:
    type: "Uncertainty"
```

### 5.2 Nash-MTL (Fairness / Scale Invariance)
```yaml
loss:
  type: "CompositeHierarchical"
  weighting_strategy:
    type: "NashMTL"
    params:
      solver: "ccp"          # or "iterative" / "auto"
      update_frequency: 10    # reuse weights every 10 steps
      max_norm: 0.0           # optional gradient clipping
```

### 5.3 IMGrad (Imbalance-Focused)
```yaml
loss:
  type: "CompositeHierarchical"
  weighting_strategy:
    type: "IMGrad"
    params:
      solver: "qp"           # or "pgd" / "auto"
```

### 5.4 Uncertainty + PCGrad (Conflict Mitigation Layer)
```yaml
loss:
  type: "CompositeHierarchical"
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

---
## 6. Practical Tips & Edge Cases
- If enabling `PCGrad` with a gradient-based strategy (e.g., Nash-MTL), benefits may diminish—avoid stacking unless diagnostics show residual conflict.
- For `NashMTL`, large `update_frequency` (e.g., 50) can drastically reduce cost with minor performance impact; monitor drift in `gradient_norm`.
- When using IMGrad without `cvxopt`, expect slower MGDA approximation (PGD); consider smaller task subset if exploratory.
- Always verify improvement using the model selection metric (`trainer.model_selection_metric`) — not just raw loss.
- If gradients explode (rare with current decays), enable global clipping via `trainer.max_grad_norm`.

---
## 7. References
- Kendall, A., Gal, Y. (2018). *Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics.* CVPR.
- Yu, T. et al. (2020a). *Gradient Surgery for Multi-Task Learning.* NeurIPS (PCGrad).
- Navon, A. et al. (2022). *Multi-Objective Learning as a Bargaining Game.* ICML (Nash-MTL).
- Zhou, X. et al. (2025). *IMGrad: Balancing Gradient Magnitude in Multi-Task Learning.* (Fictitious / Forthcoming reference placeholder if preprint).

---
This document should be kept in sync with implementation changes in:
- `src/coral_mtl/engine/loss_weighting.py`
- `src/coral_mtl/engine/gradient_strategies.py`
- `src/coral_mtl/engine/pcgrad.py`
- `src/coral_mtl/engine/trainer.py`

Please update both this specification and `configs/CONFIGS_README.md` when adding new strategies.
