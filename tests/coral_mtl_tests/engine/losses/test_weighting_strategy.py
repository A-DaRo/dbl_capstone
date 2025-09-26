"""Tests for the weighting strategy abstraction (Phase B)."""
from __future__ import annotations

import math

import torch

from coral_mtl.engine.loss_weighting import (
    UncertaintyWeightingStrategy,
    DWAWeightingStrategy,
    GradNormWeightingStrategy,
)


def test_uncertainty_strategy_shapes_and_keys(device):  # noqa: D401
    strategy = UncertaintyWeightingStrategy(tasks=['a','b'])
    losses = {
        'a': torch.tensor(2.0, device=device, requires_grad=True),
        'b': torch.tensor(3.0, device=device, requires_grad=True)
    }
    out = strategy(losses)
    assert 'total_loss' in out
    assert 'weighted_a_loss' in out and 'weighted_b_loss' in out
    assert out['total_loss'].requires_grad
    out['total_loss'].backward()
    # Ensure log_vars received gradients
    for t in ['a','b']:
        assert strategy.log_vars[t].grad is not None


def test_uncertainty_strategy_clamp(device):  # noqa: D401
    strategy = UncertaintyWeightingStrategy(tasks=['x'], clamp_range=1.0)
    with torch.no_grad():
        strategy.log_vars['x'].fill_(5.0)
    out = strategy({'x': torch.tensor(1.0, device=device)})
    # Effective log var should be clamped to 1 -> weight = exp(-1) * 1 + 0.5 * 1
    expected = torch.exp(torch.tensor(-1.0, device=device)) * 1.0 + 0.5 * 1.0
    assert torch.isclose(out['weighted_x_loss'], expected, atol=1e-5)


def test_dwa_updates_after_history(device):  # noqa: D401
    strategy = DWAWeightingStrategy(tasks=['t1', 't2'], temperature=2.0)
    base_losses = {
        't1': torch.tensor(1.0, device=device, requires_grad=True),
        't2': torch.tensor(1.0, device=device, requires_grad=True)
    }
    strategy(base_losses)
    strategy.update_epoch_losses({'unweighted_t1_loss': 1.0, 'unweighted_t2_loss': 1.0}, epoch=1)
    strategy.update_epoch_losses({'unweighted_t1_loss': 0.5, 'unweighted_t2_loss': 1.5}, epoch=2)
    updated = strategy({'t1': torch.tensor(1.0, device=device), 't2': torch.tensor(1.0, device=device)})
    w1 = updated['dwa_weight_t1']
    w2 = updated['dwa_weight_t2']
    assert not torch.isclose(w1, w2)


def test_gradnorm_adjusts_weights(device):  # noqa: D401
    model = torch.nn.Linear(4, 2).to(device)
    strategy = GradNormWeightingStrategy(['task_a', 'task_b'], lr=0.1)
    inp = torch.randn(2, 4, device=device)

    out = model(inp)
    losses = {
        'task_a': out[:, 0].pow(2).mean(),
        'task_b': (out[:, 1] - 2).pow(2).mean()
    }
    strategy(losses)
    strategy.register_shared_parameters(list(model.parameters()))
    strategy.manual_backward_update(model)
    # Initial call sets baseline, so weights unchanged
    initial_weights = torch.stack([strategy.weights['task_a'], strategy.weights['task_b']]).clone()

    out2 = model(inp)
    losses2 = {
        'task_a': out2[:, 0].pow(2).mean(),
        'task_b': (out2[:, 1] - 2).pow(2).mean() * 3
    }
    strategy(losses2)
    strategy.manual_backward_update(model)
    new_weights = torch.stack([strategy.weights['task_a'], strategy.weights['task_b']])
    assert not torch.allclose(initial_weights, new_weights)


def test_uncertainty_weighting_logic(device):
    """Increasing log_var should down-weight the task while increasing regulariser."""
    strategy = UncertaintyWeightingStrategy(tasks=['A', 'B'])
    losses = {
        'A': torch.tensor(2.0, device=device),
        'B': torch.tensor(2.0, device=device)
    }

    out1 = strategy(losses)
    weighted_a_first = out1['weighted_A_loss'].detach()
    reg_first = 0.5 * strategy.log_vars['A'].detach().clone()

    with torch.no_grad():
        strategy.log_vars['A'].add_(1.0)

    out2 = strategy(losses)
    weighted_a_second = out2['weighted_A_loss'].detach()
    reg_second = 0.5 * strategy.log_vars['A'].detach().clone()

    assert weighted_a_second < weighted_a_first
    assert reg_second > reg_first


def test_gradnorm_adaptation(monkeypatch, device):
    """GradNorm should increase weight for the slower improving task."""
    model = torch.nn.Linear(2, 2).to(device)
    strategy = GradNormWeightingStrategy(['A', 'B'], lr=0.1)
    strategy.register_shared_parameters(list(model.parameters()))

    losses_initial = {
        'A': torch.tensor(10.0, device=device),
        'B': torch.tensor(10.0, device=device)
    }
    strategy(losses_initial)
    strategy.manual_backward_update(model)  # establishes initial losses

    losses_next = {
        'A': torch.tensor(9.0, device=device),
        'B': torch.tensor(5.0, device=device)
    }
    strategy(losses_next)

    def fake_autograd_grad(output, inputs, retain_graph=True, allow_unused=True):
        value = float(output.detach())
        grad_val = 1.0  # identical gradient norms for both tasks
        grads = []
        for idx, _ in enumerate(inputs):
            if idx == 0:
                grads.append(torch.tensor(grad_val, device=output.device))
            else:
                grads.append(None)
        return tuple(grads)

    monkeypatch.setattr(torch.autograd, 'grad', fake_autograd_grad)
    strategy.manual_backward_update(model)

    weight_a = strategy.weights['A'].item()
    weight_b = strategy.weights['B'].item()
    assert math.isfinite(weight_a) and math.isfinite(weight_b)
    assert weight_a > weight_b
