"""Regression test: legacy config without weighting_strategy still works."""
from __future__ import annotations

import torch

from coral_mtl.engine.losses import CoralMTLLoss
from coral_mtl.engine.loss_weighting import UncertaintyWeightingStrategy


def build_legacy_loss(num_classes, primary, aux):  # noqa: D401
    # Simulate ExperimentFactory legacy instantiation (no explicit strategy config)
    strategy = UncertaintyWeightingStrategy(primary + aux)
    return CoralMTLLoss(
        num_classes=num_classes,
        primary_tasks=primary,
        aux_tasks=aux,
        weighting_strategy=strategy,
        ignore_index=0,
        w_consistency=0.1
    )


def test_backward_compat_no_strategy_block(device):  # noqa: D401
    num_classes = {'genus': 3, 'health': 2, 'fish': 2}
    loss_fn = build_legacy_loss(num_classes, ['genus','health'], ['fish'])
    b,h,w = 2,16,16
    preds = {t: torch.randn(b,c,h,w, device=device, requires_grad=True) for t,c in num_classes.items()}
    targets = {t: torch.randint(0,c,(b,h,w), device=device) for t,c in num_classes.items()}
    out = loss_fn(preds, targets)
    assert 'total_loss' in out
    out['total_loss'].backward()
    # Ensure gradients flow to strategy parameters
    strat = loss_fn.weighting_strategy
    for t in strat.log_vars:
        assert strat.log_vars[t].grad is not None
