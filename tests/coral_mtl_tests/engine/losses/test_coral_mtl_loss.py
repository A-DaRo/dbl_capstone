# Edit file: tests/coral_mtl/engine/losses/test_coral_mtl_loss.py
import pytest
import torch
import torch.nn as nn

from coral_mtl.engine.losses import CoralMTLLoss
from coral_mtl.utils.task_splitter import MTLTaskSplitter

# --- Fixtures for Synthetic Data ---

@pytest.fixture
def mtl_num_classes(splitter_mtl: MTLTaskSplitter):
    """Dynamically creates the num_classes dict from the splitter fixture."""
    return {
        task: len(info['ungrouped']['id2label'])
        for task, info in splitter_mtl.hierarchical_definitions.items()
    }


@pytest.fixture
def mtl_synthetic_data(device, mtl_num_classes):
    """Provides synthetic predictions and targets for all tasks."""
    batch_size, h, w = 2, 8, 8
    predictions = {}
    perfect_targets = {}
    wrong_targets = {}

    for task, n_cls in mtl_num_classes.items():
        if n_cls > 1:
            # Predictions are confident for class 1
            preds = torch.full((batch_size, n_cls, h, w), -10.0, device=device)
            preds[:, 1, :, :] = 10.0
            predictions[task] = preds
            
            # Perfect targets are all class 1
            perfect_targets[task] = torch.full((batch_size, h, w), 1, dtype=torch.long, device=device)
            
            # Wrong targets are class 0 (background)
            wrong_targets[task] = torch.full((batch_size, h, w), 0, dtype=torch.long, device=device)
            
    return predictions, perfect_targets, wrong_targets


# --- Core Correctness Tests ---

def test_mtl_loss_perfect_prediction(mtl_num_classes, mtl_synthetic_data):
    """Test that perfect predictions across all tasks yield a near-zero loss."""
    predictions, perfect_targets, _ = mtl_synthetic_data
    all_tasks = list(mtl_num_classes.keys())
    
    loss_fn = CoralMTLLoss(
        num_classes=mtl_num_classes,
        primary_tasks=[t for t in ['genus', 'health'] if t in all_tasks],
        aux_tasks=[t for t in all_tasks if t not in ['genus', 'health']]
    )
    
    loss_dict = loss_fn(predictions, perfect_targets)
    
    assert loss_dict['total_loss'].item() < 0.1, "Total loss for perfect predictions should be very low (only log_var terms remain)."
    for task in loss_fn.primary_tasks:
        assert loss_dict[f'unweighted_{task}_loss'].item() < 1e-3
    for task in loss_fn.aux_tasks:
        if f'unweighted_{task}_loss' in loss_dict:
             assert loss_dict[f'unweighted_{task}_loss'].item() < 1e-3


def test_mtl_loss_wrong_prediction(mtl_num_classes, mtl_synthetic_data):
    """Test that completely wrong predictions yield a high positive loss."""
    predictions, _, wrong_targets = mtl_synthetic_data
    all_tasks = list(mtl_num_classes.keys())

    loss_fn = CoralMTLLoss(
        num_classes=mtl_num_classes,
        primary_tasks=[t for t in ['genus', 'health'] if t in all_tasks],
        aux_tasks=[t for t in all_tasks if t not in ['genus', 'health']]
    )
    
    loss_dict = loss_fn(predictions, wrong_targets)
    
    assert loss_dict['total_loss'].item() > 2.0, "Total loss for wrong predictions should be high."


# --- Dynamic Behavior & Edge Case Tests ---

def test_dynamic_tasks_handle_missing_primary_task(mtl_num_classes, mtl_synthetic_data):
    """
    CRITICAL: Verifies the fix for hard-coded tasks. The loss function must successfully
    compute a loss even if a primary task like 'genus' is not configured.
    """
    predictions, targets, _ = mtl_synthetic_data
    
    # Simulate a configuration where 'genus' is not a primary task
    if 'genus' in mtl_num_classes and 'health' in mtl_num_classes:
        primary_tasks = ['health'] # 'genus' is omitted
        aux_tasks = [t for t in mtl_num_classes if t != 'health']
        
        # Instantiate loss with the dynamic configuration
        loss_fn = CoralMTLLoss(
            num_classes=mtl_num_classes,
            primary_tasks=primary_tasks,
            aux_tasks=aux_tasks
        )
        
        # This should now pass without error
        loss_dict = loss_fn(predictions, targets)

        assert torch.isfinite(loss_dict['total_loss'])
        # The unweighted genus loss should NOT be in the dict as a primary loss, but as an aux loss.
        assert 'unweighted_genus_loss' in loss_dict
        assert 'log_var_genus' not in loss_dict # No uncertainty param should be created for it


@pytest.mark.parametrize("empty_list", ["primary", "auxiliary"])
def test_handles_empty_task_lists(mtl_num_classes, mtl_synthetic_data, empty_list):
    """Verify loss calculation succeeds with empty primary or auxiliary task lists."""
    predictions, targets, _ = mtl_synthetic_data
    all_tasks = list(mtl_num_classes.keys())

    if empty_list == "primary":
        primary_tasks, aux_tasks = [], all_tasks
    else: # auxiliary
        primary_tasks, aux_tasks = all_tasks, []

    loss_fn = CoralMTLLoss(
        num_classes=mtl_num_classes, primary_tasks=primary_tasks, aux_tasks=aux_tasks
    )

    loss_dict = loss_fn(predictions, targets)
    assert torch.isfinite(loss_dict['total_loss'])
    if empty_list == "primary":
        assert torch.isclose(loss_dict['primary_balanced_loss'], torch.tensor(0.0))
    else: # auxiliary
        assert torch.isclose(loss_dict['aux_balanced_loss'], torch.tensor(0.0))


# --- Component-Specific Tests ---

def test_uncertainty_weighting_effect(mtl_num_classes, mtl_synthetic_data):
    """Verify that changing log_var parameters affects the total loss as expected."""
    predictions, targets, _ = mtl_synthetic_data
    
    if 'genus' not in mtl_num_classes:
        pytest.skip("Test requires 'genus' task.")
        
    loss_fn = CoralMTLLoss(num_classes=mtl_num_classes, primary_tasks=['genus'], aux_tasks=[])
    
    initial_loss = loss_fn(predictions, targets)['total_loss'].item()
    
    # Manually increase the uncertainty penalty for the 'genus' task
    with torch.no_grad():
        loss_fn.log_vars_primary['genus'].fill_(3.0) # A large value
        
    new_loss = loss_fn(predictions, targets)['total_loss'].item()
    
    assert new_loss != initial_loss, "Changing log_var should alter the total loss."
    assert new_loss > initial_loss, "Increasing log_var should increase the total loss term."


@pytest.mark.parametrize("config, should_be_active", [
    (["genus", "health"], True),  # Both present, should be active
    (["genus"], False),           # health missing, should be inactive
    (["health"], False),          # genus missing, should be inactive
    ([], False),                  # Both missing, should be inactive
])
def test_consistency_loss_is_conditional(mtl_num_classes, device, config, should_be_active):
    """Verify the consistency penalty is active only if both 'genus' and 'health' are primary."""
    if not all(t in mtl_num_classes for t in ["genus", "health"]):
        pytest.skip("Test requires both 'genus' and 'health' tasks.")

    loss_fn = CoralMTLLoss(num_classes=mtl_num_classes, primary_tasks=config, aux_tasks=[], w_consistency=1.0)
    
    # Create an illogical prediction: P(genus=bg) > 0.5 and P(health=alive) > 0.5
    b, h, w = 1, 4, 4
    genus_logits = torch.full((b, mtl_num_classes['genus'], h, w), -10.0, device=device)
    genus_logits[:, 0, :, :] = 10.0 
    health_logits = torch.full((b, mtl_num_classes['health'], h, w), -10.0, device=device)
    health_logits[:, 1, :, :] = 10.0
    
    predictions = {'genus': genus_logits, 'health': health_logits}
    targets = {'genus': torch.zeros_like(genus_logits[:,0,:,:]).long(), 'health': torch.zeros_like(health_logits[:,0,:,:]).long()}

    loss_dict = loss_fn(predictions, targets)

    if should_be_active:
        assert loss_dict['consistency_loss'].item() > 0
    else:
        assert torch.isclose(loss_dict['consistency_loss'], torch.tensor(0.0))


def test_toy_overfit_decreases_mtl_loss(device, mtl_num_classes):
    """A mini integration test to ensure the composite loss can be optimized."""
    if not mtl_num_classes:
        pytest.skip("No tasks available for this configuration.")
        
    all_tasks = list(mtl_num_classes.keys())
    primary_tasks = [t for t in ['genus', 'health'] if t in all_tasks]
    aux_tasks = [t for t in all_tasks if t not in primary_tasks]
    
    class TinyMTLModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.convs = nn.ModuleDict({
                task: nn.Conv2d(3, n_cls, 1) for task, n_cls in mtl_num_classes.items()
            })
        def forward(self, x):
            return {task: conv(x) for task, conv in self.convs.items()}
            
    model = TinyMTLModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    loss_fn = CoralMTLLoss(mtl_num_classes, primary_tasks, aux_tasks)

    image = torch.randn(2, 3, 8, 8, device=device)
    targets = {
        task: torch.randint(0, n_cls, (2, 8, 8), device=device)
        for task, n_cls in mtl_num_classes.items()
    }
    
    losses = []
    for _ in range(5):
        optimizer.zero_grad()
        preds = model(image)
        loss_dict = loss_fn(preds, targets)
        loss = loss_dict['total_loss']
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    for i in range(len(losses) - 1):
        assert losses[i+1] < losses[i], f"Loss did not decrease at step {i+1}"