# Create file: tests/coral_mtl/engine/losses/test_coral_loss.py
import pytest
import torch
import torch.nn as nn

from coral_mtl.engine.losses import CoralLoss

# --- Fixtures for Synthetic Data ---

@pytest.fixture
def synthetic_data(device):
    """Provides synthetic logits and targets for testing."""
    batch_size, num_classes, h, w = 2, 5, 8, 8
    # Logits with high confidence for class 1
    logits = torch.full((batch_size, num_classes, h, w), -10.0, device=device)
    logits[:, 1, :, :] = 10.0
    
    # Target where all pixels are class 1
    perfect_target = torch.full((batch_size, h, w), 1, dtype=torch.long, device=device)
    
    # Target where all pixels are class 2 (completely wrong)
    wrong_target = torch.full((batch_size, h, w), 2, dtype=torch.long, device=device)
    
    return logits, perfect_target, wrong_target, num_classes


# --- Core Correctness Tests ---

@pytest.mark.parametrize("primary_loss_type", ["focal", "cross_entropy"])
def test_coral_loss_perfect_prediction(synthetic_data, primary_loss_type):
    """Test that a perfect prediction results in a near-zero loss."""
    logits, perfect_target, _, num_classes = synthetic_data
    loss_fn = CoralLoss(primary_loss_type=primary_loss_type, hybrid_alpha=0.5)
    
    loss = loss_fn(logits, perfect_target)
    
    assert torch.isfinite(loss)
    assert loss.item() < 1e-3, "Loss for a perfect prediction should be close to zero."


@pytest.mark.parametrize("primary_loss_type", ["focal", "cross_entropy"])
def test_coral_loss_wrong_prediction(synthetic_data, primary_loss_type):
    """Test that a completely incorrect prediction results in a high positive loss."""
    logits, _, wrong_target, num_classes = synthetic_data
    loss_fn = CoralLoss(primary_loss_type=primary_loss_type, hybrid_alpha=0.5)

    loss = loss_fn(logits, wrong_target)
    
    assert torch.isfinite(loss)
    assert loss.item() > 2.0, "Loss for a completely wrong prediction should be high."


def test_ignore_index_handling(device):
    """Verify that pixels with ignore_index do not contribute to the loss."""
    ignore_index = -100
    num_classes = 4
    h, w = 8, 8
    loss_fn = CoralLoss(ignore_index=ignore_index)
    
    # Target mask with a section to be ignored
    target = torch.ones(1, h, w, dtype=torch.long, device=device)
    target[:, :h//2, :] = ignore_index

    # Case 1: Logits are correct for the ignored pixels
    logits1 = torch.randn(1, num_classes, h, w, device=device)
    
    # Case 2: Logits are wildly incorrect for the ignored pixels
    logits2 = logits1.clone()
    logits2[:, 1, :h//2, :] = -1000.0 # Make prediction for class 1 very wrong in ignored area
    logits2[:, 2, :h//2, :] = 1000.0  # Make prediction for class 2 very high in ignored area

    loss1 = loss_fn(logits1, target)
    loss2 = loss_fn(logits2, target)

    assert torch.isclose(loss1, loss2), "Loss should be identical when only ignored pixels' logits change."
    assert loss1 > 0, "Loss should not be zero as some pixels are not ignored."


def test_hybrid_alpha_weighting(synthetic_data):
    """Verify the alpha parameter correctly weights the primary and dice losses."""
    logits, _, wrong_target, num_classes = synthetic_data
    
    loss_fn_primary_only = CoralLoss(primary_loss_type='focal', hybrid_alpha=1.0)
    loss_fn_dice_only = CoralLoss(primary_loss_type='focal', hybrid_alpha=0.0)
    loss_fn_hybrid = CoralLoss(primary_loss_type='focal', hybrid_alpha=0.5)

    loss_primary = loss_fn_primary_only(logits, wrong_target)
    loss_dice = loss_fn_dice_only(logits, wrong_target)
    loss_hybrid = loss_fn_hybrid(logits, wrong_target)

    expected_hybrid = 0.5 * loss_primary + 0.5 * loss_dice

    assert not torch.isclose(loss_primary, loss_dice), "Primary and dice losses should be different"
    assert torch.isclose(loss_hybrid, expected_hybrid), "Hybrid loss is not the correct weighted average."


def test_toy_overfit_decreases_loss(device):
    """
    A mini integration test to ensure the loss decreases over a few steps
    when training a tiny model on a single batch.
    """
    model = nn.Conv2d(3, 5, kernel_size=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    loss_fn = CoralLoss()
    
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
        
    for i in range(len(losses) - 1):
        assert losses[i+1] < losses[i], f"Loss did not decrease at step {i+1}"