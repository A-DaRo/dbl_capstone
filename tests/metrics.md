# Unified Metrics Pipeline for MTL vs Baseline

This document explains how metrics are computed consistently across the baseline (single flat head) and MTL (multi-head, possibly grouped) segmentation models. The goal is to ensure that all numbers — mIoU, BIoU, Boundary-F1, NLL, Brier, ECE — are computed in the same global label space, making results apples-to-apples comparable.

---

## Objective

Both models must be evaluated in a unified global label space (`GLOBAL`):

* **Baseline**: outputs logits in a flat space → mapped `flat → original → global`.
* **MTL**: multiple heads over (possibly grouped) class sets → fused into `original` space → marginalized to `global`.

From this point forward, all metrics are identical.

---

## Metrics Evaluated

### 1) Confusion-Matrix Family (computed in global space)

* **mIoU**
  IoU per class, then mean across global classes. Derived from a global confusion matrix `CM ∈ ℕ^{C×C}` (C = #global classes).
  For each class `c`:

  * `TP_c = CM[c,c]`
  * `FP_c = Σ_r CM[r,c] - CM[c,c]`
  * `FN_c = Σ_r CM[c,r] - CM[c,c]`
  * `IoU_c = TP_c / (TP_c + FP_c + FN_c)`
  * `mIoU = mean_c IoU_c` (background included/excluded consistently across models per config).

* **Pixel Accuracy / Precision / Recall / F1**
  Derived from the same `CM`, computed per-class, then macro-averaged.

* **TIDE-style error breakdown**

  * **Classification error**: off-diagonal within foreground block / total pixels
  * **Background error**: predicted foreground when gt is background / total pixels
  * **Missed error**: predicted background when gt is foreground / total pixels

### 2) Boundary-Aware Metrics

* **Boundary IoU (BIoU)**
  For each non-background class `c`:

  * Extract `(gt==c)` and `(pred==c)`
  * Compute boundary bands via 3×3 dilation (repeated `boundary_thickness` times) and subtract interior
  * Accumulate `intersection += |boundary_gt ∧ boundary_pred|` and `union += |boundary_gt ∨ boundary_pred|`
  * `BIoU = intersection / (union + ε)`
    Computed per-task and globally with GPU-native ops.

* **Boundary-F1 (global)**
  Across all non-background classes:

  * `TP = |boundary_gt ∧ boundary_pred|`
  * `FP = |boundary_pred ∧ ¬boundary_gt|`
  * `FN = |boundary_gt ∧ ¬boundary_pred|`
  * `Precision = TP / (TP+FP+ε)`, `Recall = TP / (TP+FN+ε)`, `F1 = 2PR/(P+R+ε)`
    Accumulate counts over images before computing P/R/F1.

### 3) Probabilistic / Calibration Metrics (on global probabilities)

* **NLL**: `mean_{valid pixels} (−log P(y_true|x))`
* **Brier Score**: `mean ||p − one_hot(y)||^2`
* **ECE**: Bin by max confidence; for each bin `k`, `ECE += (|B_k|/N)*|acc_k − conf_k|`

All three use global probabilities, ensuring direct comparability.

---

## Label Spaces & Mappings

* **Original (`O`)**: fine-grained dataset class IDs.
* **Global (`G`)**: unified label set (`|G| = C`) used for evaluation.
* **Task head spaces**: per-head class sets (may be grouped or ungrouped).

**Mappings provided by TaskSplitter (and related helpers):**

* `orig → global` (single source of truth; shared across models)
* Per-head index maps to align original classes to that head’s channels
* Background and `ignore_index` handled consistently via a `_safe_lookup`-style mapping

**Ignore/background handling**

* `ignore_index` pixels are excluded from CM and probability metrics
* Background is a defined global id (often `0`) and used consistently

---

## MTL Pipeline (log-prob fusion in ORIGINAL → marginalize to GLOBAL)

Let `logits_t ∈ ℝ^{B×C_t×H×W}` be the output of head `t`.

1. **Per-head log-probs**

   ```python
   logp_t = log_softmax(logits_t, dim=1)  # in head space
   ```

2. **Align to ORIGINAL**
   Each head has an index tensor `idx_t ∈ ℤ^{|O|}` mapping original classes to the head’s channels.
   Gather to obtain per-original log-probs:

   ```
   logp_t^orig[b, o, h, w] = logp_t[b, idx_t[o], h, w]  # shape: B×|O|×H×W
   ```

3. **Fuse across heads (product-of-experts in log-space)**

   ```
   fused_logp_orig = Σ_t logp_t^orig
   ```

   Optional calibration: weights/temperatures

   ```
   logp_t = log_softmax(logits_t / τ_t, dim=1)
   fused_logp_orig = Σ_t α_t * logp_t^orig
   ```

4. **Marginalize ORIGINAL → GLOBAL (prob-space)**

   ```
   fused_prob_orig   = exp(fused_logp_orig)               # B×|O|×H×W
   fused_prob_global = zeros(B×|G|×H×W)
   fused_prob_global[g] += Σ_{o: orig2global[o]=g} fused_prob_orig[o]
   normalize over g: fused_prob_global /= Σ_g fused_prob_global[g]
   ```

5. **Predictions & Metrics in GLOBAL**

   * `pred_global = argmax_g fused_prob_global`
   * Global CM → IoU/F1/TIDE
   * Boundaries(gt,pred) → BIoU & Boundary-F1
   * `fused_prob_global` → NLL / Brier / ECE

**Properties**

* No target leakage (only static maps used)
* Mixed grouped/ungrouped heads supported
* Identical downstream metrics as baseline

---

## Baseline Pipeline (flat → ORIGINAL → GLOBAL)

* Map flat head to original classes, then to global:

  * **Argmax route (for CM)**: `argmax(flat) → original → global`
  * **Probabilistic route**: gather logits/probs `flat → original`, then marginalize to global exactly as in MTL

From the global distribution onward, all metrics are identical to the MTL path.

---

## Why This Is Apples-to-Apples

* Both models yield probabilities over the same global classes.
* The same ground-truth mapping is used to build global targets.
* Confusion matrix, boundary extraction, and probabilistic computations are identical.
* Grouping differences are neutralized by `original → global`.

---

## Safeguards & Edge Cases

* **ignore_index pixels**: excluded everywhere.
* **No boundary present**: `union=0 ⇒ BIoU=0`; Boundary-F1 uses ε-safe divisions.
* **Classes absent in a batch**: IoU denominators can be zero → use NaN-safe means.
* **Heads disagree**: fusion softens confidences; calibration metrics (NLL/ECE) reflect this.
* **Missing head**: fusion over available heads; if none, default background or fail fast.
* **Calibration tunability**: tune `α_t` / `τ_t` on a validation split.

---

## TL;DR Pipeline Diagram

```
MTL heads logits_t
  └─ log_softmax
       └─ gather to ORIGINAL
            └─ sum log-probs (product-of-experts)
                 └─ exp → ORIGINAL probs
                      └─ scatter-add to GLOBAL
                           └─ normalize → global probs
                                ├─ argmax → CM → IoU/F1/TIDE
                                ├─ boundaries → BIoU & Boundary-F1
                                └─ probs → NLL / Brier / ECE
```

*Baseline joins at the GLOBAL stage after flat→original→global mapping.*

---

## Bottom Line

This ensures that MTL vs baseline results are directly comparable, with all metrics living in the same global coordinate system.
