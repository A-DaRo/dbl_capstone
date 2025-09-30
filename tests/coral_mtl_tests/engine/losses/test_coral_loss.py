"""Unit tests for :class:`HybridSegmentationLoss`."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from coral_mtl.engine.losses import HybridSegmentationLoss


@pytest.fixture
def synthetic_data(device):
    """Provides logits/targets triplet for perfect vs. adversarial predictions."""
    batch_size, num_classes, h, w = 2, 5, 8, 8
    logits = torch.full((batch_size, num_classes, h, w), -10.0, device=device)
    logits[:, 1] = 10.0
    perfect_target = torch.full((batch_size, h, w), 1, dtype=torch.long, device=device)
    wrong_target = torch.full((batch_size, h, w), 2, dtype=torch.long, device=device)
    return logits, perfect_target, wrong_target


def test_hybrid_loss_perfect_prediction_small(device, synthetic_data):
    logits, perfect_target, _ = synthetic_data
    loss_fn = HybridSegmentationLoss(hybrid_alpha=0.5)
    loss = loss_fn(logits, perfect_target)
    assert torch.isfinite(loss)
    assert loss.item() < 0.1


def test_hybrid_loss_wrong_prediction_large(device, synthetic_data):
    logits, _, wrong_target = synthetic_data
    loss_fn = HybridSegmentationLoss(hybrid_alpha=0.5)
    loss = loss_fn(logits, wrong_target)
    assert torch.isfinite(loss)
    assert loss.item() > 0.5


def test_ignore_index_exclusion(device):
    ignore_index = -100
    loss_fn = HybridSegmentationLoss(ignore_index=ignore_index)
    logits = torch.randn(1, 4, 6, 6, device=device)
    target = torch.ones(1, 6, 6, dtype=torch.long, device=device)
    target[:, :3] = ignore_index
    logits_variant = logits.clone()
    logits_variant[:, :, :3] = torch.randn_like(logits_variant[:, :, :3]) * 50
    loss1 = loss_fn(logits, target)
    loss2 = loss_fn(logits_variant, target)
    assert torch.isclose(loss1, loss2), "Ignored pixels should not influence loss"


def test_hybrid_alpha_controls_balance(device, synthetic_data):
    logits, _, wrong_target = synthetic_data
    loss_primary = HybridSegmentationLoss(hybrid_alpha=1.0)(logits, wrong_target)
    loss_dice = HybridSegmentationLoss(hybrid_alpha=0.0)(logits, wrong_target)
    hybrid_loss = HybridSegmentationLoss(hybrid_alpha=0.3)(logits, wrong_target)
    expected = 0.3 * loss_primary + 0.7 * loss_dice
    assert torch.isclose(hybrid_loss, expected, atol=1e-5)


def test_toy_overfit_reduces_loss(device):
    model = nn.Conv2d(3, 5, kernel_size=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    loss_fn = HybridSegmentationLoss()
    image = torch.randn(2, 3, 8, 8, device=device)
    target = torch.randint(0, 5, (2, 8, 8), device=device)
    losses = []
    for _ in range(5):
        optimizer.zero_grad()
        logits = model(image)
        loss = loss_fn(logits, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    for earlier, later in zip(losses, losses[1:]):
        assert later <= earlier


@pytest.mark.parametrize("target_fill", [-100, 0])
def test_nan_resilience(device, target_fill):
    logits = torch.zeros((1, 4, 4, 4), device=device)
    target = torch.full((1, 4, 4), target_fill, dtype=torch.long, device=device)
    loss = HybridSegmentationLoss(ignore_index=-100)(logits, target)
    assert loss.shape == torch.Size([])
    assert torch.isfinite(loss)