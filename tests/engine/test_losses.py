import torch
import pytest
from coral_mtl.engine.losses import CoralLoss, CoralMTLLoss

def test_coral_loss_forward_pass():
    """Tests the forward pass of the baseline CoralLoss."""
    B, C, H, W = 4, 10, 64, 64
    # FIX: Set requires_grad=True to simulate model output
    logits = torch.randn(B, C, H, W, requires_grad=True)
    targets = torch.randint(0, C, (B, H, W))
    
    loss_fn = CoralLoss(ignore_index=0)
    loss = loss_fn(logits, targets)
    
    assert torch.is_tensor(loss)
    assert loss.requires_grad
    assert loss.item() >= 0

@pytest.fixture
def mtl_loss_data():
    """Provides dummy data for CoralMTLLoss tests."""
    B, H, W = 2, 32, 32
    num_classes = {
        'genus': 9, 'health': 4, 'fish': 2, 'human_artifacts': 2, 'substrate': 4
    }
    predictions = {
        task: torch.randn(B, n_cls, H, W, requires_grad=True)
        for task, n_cls in num_classes.items()
    }
    targets = {
        task: torch.randint(0, n_cls, (B, H, W))
        for task, n_cls in num_classes.items()
    }
    return predictions, targets, num_classes

def test_coral_mtl_loss_forward_and_backward(mtl_loss_data):
    """
    Tests the forward and backward pass of CoralMTLLoss, ensuring the output
    is a valid dictionary and gradients can be computed.
    """
    predictions, targets, num_classes = mtl_loss_data
    loss_fn = CoralMTLLoss(num_classes=num_classes, ignore_index=0)
    
    loss_dict = loss_fn(predictions, targets)
    
    assert isinstance(loss_dict, dict)
    expected_keys = [
        'total_loss', 'primary_balanced_loss', 'unweighted_genus_loss',
        'log_var_genus', 'unweighted_fish_loss'
    ]
    for key in expected_keys:
        assert key in loss_dict
        assert torch.is_tensor(loss_dict[key])
        
    # Test backward pass
    try:
        total_loss = loss_dict['total_loss']
        total_loss.backward()
    except Exception as e:
        pytest.fail(f"Backward pass failed with an error: {e}")

    # Check if a parameter has received a gradient
    assert predictions['genus'].grad is not None