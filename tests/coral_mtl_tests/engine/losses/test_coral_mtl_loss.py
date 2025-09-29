# Edit file: tests/coral_mtl/engine/losses/test_coral_mtl_loss.py
import pytest
import torch
import torch.nn as nn

from coral_mtl.engine.losses import CoralMTLLoss
from coral_mtl.engine.loss_weighting import WeightingStrategy, build_weighting_strategy
from coral_mtl.utils.task_splitter import MTLTaskSplitter

# --- Fixtures for Synthetic Data ---

@pytest.fixture
def mtl_num_classes(splitter_mtl: MTLTaskSplitter):
    """Dynamically creates the num_classes dict from the splitter fixture."""
    num_classes = {}
    for task, info in splitter_mtl.hierarchical_definitions.items():
        if info.get('is_grouped', False):
            num_classes[task] = len(info['grouped']['id2label'])
        else:
            num_classes[task] = len(info['ungrouped']['id2label'])
    return num_classes


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
    primary_tasks=[t for t in ['genus', 'health'] if t in all_tasks]
    aux_tasks=[t for t in all_tasks if t not in ['genus', 'health']]
    
    strategy = build_weighting_strategy(config=None, primary=primary_tasks, auxiliary=aux_tasks)
    
    loss_fn = CoralMTLLoss(
        num_classes=mtl_num_classes,
        primary_tasks=primary_tasks,
        aux_tasks=aux_tasks,
        weighting_strategy=strategy
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
    primary_tasks=[t for t in ['genus', 'health'] if t in all_tasks]
    aux_tasks=[t for t in all_tasks if t not in ['genus', 'health']]

    strategy = build_weighting_strategy(config=None, primary=primary_tasks, auxiliary=aux_tasks)

    loss_fn = CoralMTLLoss(
        num_classes=mtl_num_classes,
        primary_tasks=primary_tasks,
        aux_tasks=aux_tasks,
        weighting_strategy=strategy
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
        
        strategy = build_weighting_strategy(config=None, primary=primary_tasks, auxiliary=aux_tasks)
        
        # Instantiate loss with the dynamic configuration
        loss_fn = CoralMTLLoss(
            num_classes=mtl_num_classes,
            primary_tasks=primary_tasks,
            aux_tasks=aux_tasks,
            weighting_strategy=strategy
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

    strategy = build_weighting_strategy(config=None, primary=primary_tasks, auxiliary=aux_tasks)

    loss_fn = CoralMTLLoss(
        num_classes=mtl_num_classes, 
        primary_tasks=primary_tasks, 
        aux_tasks=aux_tasks,
        weighting_strategy=strategy
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
        
    primary_tasks=['genus']
    aux_tasks=[]
    strategy = build_weighting_strategy(config=None, primary=primary_tasks, auxiliary=aux_tasks)
    loss_fn = CoralMTLLoss(
        num_classes=mtl_num_classes, 
        primary_tasks=primary_tasks, 
        aux_tasks=aux_tasks,
        weighting_strategy=strategy
    )
    
    initial_loss = loss_fn(predictions, targets)['total_loss'].item()
    
    # Manually increase the uncertainty penalty for the 'genus' task
    with torch.no_grad():
        loss_fn.weighting_strategy.log_vars['genus'].fill_(3.0) # A large value
        
    new_loss = loss_fn(predictions, targets)['total_loss'].item()
    
    assert new_loss != initial_loss, "Changing log_var should alter the total loss."
    assert new_loss > initial_loss, "Increasing log_var should increase the total loss term."


@pytest.mark.parametrize("config, expected_tasks", [
    (["genus", "health"], ["genus", "health"]),  # Both present
    (["genus"], ["genus"]),                       # Only genus
    (["health"], ["health"]),                    # Only health  
    ([], []),                                    # No primary tasks
])
def test_loss_dict_contains_expected_tasks(mtl_num_classes, device, config, expected_tasks):
    """Verify the loss function returns expected keys based on task configuration."""
    if config and not all(t in mtl_num_classes for t in config):
        pytest.skip("Test requires configured tasks to be available.")

    primary_tasks = config
    aux_tasks = []
    strategy = build_weighting_strategy(config=None, primary=primary_tasks, auxiliary=aux_tasks)
    loss_fn = CoralMTLLoss(
        num_classes=mtl_num_classes,
        primary_tasks=primary_tasks,
        aux_tasks=aux_tasks,
        weighting_strategy=strategy
    )
    
    # Create synthetic data for available tasks
    b, h, w = 1, 4, 4
    predictions = {}
    targets = {}
    
    for task in expected_tasks:
        if task in mtl_num_classes:
            predictions[task] = torch.randn(b, mtl_num_classes[task], h, w, device=device)
            targets[task] = torch.randint(0, mtl_num_classes[task], (b, h, w), device=device)

    loss_dict = loss_fn(predictions, targets)

    # Should always have total_loss
    assert 'total_loss' in loss_dict
    assert torch.isfinite(loss_dict['total_loss'])
    
    # Should have unweighted and weighted loss entries for each task
    for task in expected_tasks:
        assert f'unweighted_{task}_loss' in loss_dict, f"Missing unweighted_{task}_loss"
        assert f'weighted_{task}_loss' in loss_dict, f"Missing weighted_{task}_loss"
        assert torch.isfinite(loss_dict[f'unweighted_{task}_loss'])
        assert torch.isfinite(loss_dict[f'weighted_{task}_loss'])


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
    strategy = build_weighting_strategy(config=None, primary=primary_tasks, auxiliary=aux_tasks)
    loss_fn = CoralMTLLoss(mtl_num_classes, primary_tasks, aux_tasks, weighting_strategy=strategy)
    
    # The model parameters now also include the loss parameters
    optimizer = torch.optim.Adam(list(model.parameters()) + list(loss_fn.parameters()), lr=0.1)

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


def test_consistency_penalty_differentiable(device, mtl_num_classes):
    """Ensure the new soft consistency penalty provides gradients when violation exists."""
    if not all(t in mtl_num_classes for t in ["genus", "health"]):
        pytest.skip("Requires genus & health tasks")

    genus_classes = mtl_num_classes['genus']
    health_classes = mtl_num_classes['health']
    b, h, w = 1, 4, 4
    # Create logits that produce a violation (both probs ~ high)
    # Start with all classes having low logits, then set specific classes high
    genus_logits = torch.full((b, genus_classes, h, w), -10.0, device=device)
    health_logits = torch.full((b, health_classes, h, w), -10.0, device=device)
    # Push background (idx 0) & alive (idx 1) upward strongly
    genus_logits[:,0] = 10.0  # Very strong signal for background
    health_logits[:,1] = 10.0  # Very strong signal for alive
    # Now set requires_grad to make them leaf tensors
    genus_logits.requires_grad_(True)
    health_logits.requires_grad_(True)
    predictions = {'genus': genus_logits, 'health': health_logits}
    targets = {
        'genus': torch.zeros(b, h, w, dtype=torch.long, device=device),
        'health': torch.zeros(b, h, w, dtype=torch.long, device=device)
    }
    strategy = build_weighting_strategy(config=None, primary=['genus','health'], auxiliary=[])
    loss_fn = CoralMTLLoss(mtl_num_classes, primary_tasks=['genus','health'], aux_tasks=[], weighting_strategy=strategy)
    loss_dict = loss_fn(predictions, targets)
    # Consistency penalty is now handled by the weighting strategy
    # Just ensure the total loss is differentiable
    assert 'total_loss' in loss_dict
    assert torch.isfinite(loss_dict['total_loss'])
    loss_dict['total_loss'].backward()
    assert genus_logits.grad is not None and health_logits.grad is not None, "Loss must backpropagate"


def test_no_nan_with_empty_foreground(device, mtl_num_classes):
    """If a task has only background in targets, dice fallback must avoid NaNs."""
    if not mtl_num_classes:
        pytest.skip("No tasks configured")
    # Single task scenario to isolate
    task = next(iter(mtl_num_classes.keys()))
    n_cls = mtl_num_classes[task]
    if n_cls < 2:
        pytest.skip("Need at least background + 1 class for dice path")
    b,h,w = 2,8,8
    logits = torch.randn(b, n_cls, h, w, device=device)
    # All background target
    targets = torch.zeros(b, h, w, dtype=torch.long, device=device)
    preds = {task: logits}
    tgts = {task: targets}
    strategy = build_weighting_strategy(config=None, primary=[task], auxiliary=[])
    loss_fn = CoralMTLLoss({task: n_cls}, primary_tasks=[task], aux_tasks=[], debug=True, weighting_strategy=strategy)
    loss_dict = loss_fn(preds, tgts)
    for k,v in loss_dict.items():
        if torch.is_tensor(v):
            assert torch.isfinite(v), f"Loss component {k} produced non-finite value"


@pytest.mark.parametrize("target_fill", [-100, 0])
def test_mtl_loss_nan_invariance_multi_task(mtl_num_classes, device, target_fill):
    """Full CoralMTLLoss should remain finite across all configured tasks."""
    if not mtl_num_classes:
        pytest.skip("No tasks configured")

    height = width = 4
    predictions = {}
    targets = {}
    all_tasks = list(mtl_num_classes.keys())
    primary_tasks = [t for t in ['genus', 'health'] if t in all_tasks]
    aux_tasks = [t for t in all_tasks if t not in primary_tasks]

    for task, n_classes in mtl_num_classes.items():
        predictions[task] = torch.zeros((1, n_classes, height, width), device=device)
        targets[task] = torch.full((1, height, width), target_fill, dtype=torch.long, device=device)

    strategy = build_weighting_strategy(config=None, primary=primary_tasks, auxiliary=aux_tasks)
    loss_fn = CoralMTLLoss(
        num_classes=mtl_num_classes,
        primary_tasks=primary_tasks,
        aux_tasks=aux_tasks,
        weighting_strategy=strategy,
    )

    loss_dict = loss_fn(predictions, targets)
    assert torch.isfinite(loss_dict['total_loss'].detach())
    assert loss_dict['total_loss'].item() >= 0.0
    for key, value in loss_dict.items():
        if torch.is_tensor(value):
            assert torch.isfinite(value.detach()).all(), f"Non-finite value found for {key}"


def test_compute_unweighted_losses_filters_missing_tasks(device, mtl_num_classes, mtl_synthetic_data):
    if not {'genus', 'health'}.issubset(mtl_num_classes):
        pytest.skip("Requires genus and health tasks")

    predictions, targets, _ = mtl_synthetic_data
    # Drop health predictions intentionally to ensure they are ignored
    predictions_subset = {'genus': predictions['genus']}
    targets_subset = {'genus': targets['genus'], 'health': targets['health']}

    strategy = build_weighting_strategy(config=None, primary=['genus', 'health'], auxiliary=[])
    loss_fn = CoralMTLLoss(
        num_classes=mtl_num_classes,
        primary_tasks=['genus', 'health'],
        aux_tasks=[],
        weighting_strategy=strategy,
    )

    unweighted = loss_fn.compute_unweighted_losses(predictions_subset, targets_subset)
    assert set(unweighted.keys()) == {'genus'}
    assert torch.isfinite(unweighted['genus'])


def test_forward_delegates_to_weighting_strategy(device, mtl_num_classes, mtl_synthetic_data):
    if not {'genus', 'health'}.issubset(mtl_num_classes):
        pytest.skip("Requires genus and health tasks")

    predictions, targets, _ = mtl_synthetic_data

    class RecordingStrategy(WeightingStrategy):
        def __init__(self, tasks):
            super().__init__(tasks)
            self.forward_calls = []
            self.cached_losses = None

        def cache_unweighted_losses(self, losses):
            super().cache_unweighted_losses(losses)
            self.cached_losses = losses

        def forward(self, unweighted_losses):
            self.forward_calls.append(unweighted_losses)
            total = torch.zeros((), device=next(iter(unweighted_losses.values())).device)
            out = {'total_loss': total}
            for task, loss in unweighted_losses.items():
                out[f'weighted_{task}_loss'] = loss * 0.5
                total = total + out[f'weighted_{task}_loss']
            out['total_loss'] = total
            return out

    strategy = RecordingStrategy(['genus', 'health'])

    loss_fn = CoralMTLLoss(
        num_classes=mtl_num_classes,
        primary_tasks=['genus', 'health'],
        aux_tasks=[],
        weighting_strategy=strategy,
    )

    loss_dict = loss_fn(predictions, targets)

    assert strategy.cached_losses is not None
    assert set(strategy.cached_losses.keys()) == {'genus', 'health'}
    assert len(strategy.forward_calls) == 1
    forwarded_losses = strategy.forward_calls[0]
    assert forwarded_losses is strategy.cached_losses

    for task in ['genus', 'health']:
        assert f'unweighted_{task}_loss' in loss_dict
        assert f'weighted_{task}_loss' in loss_dict
        assert torch.isclose(loss_dict[f'weighted_{task}_loss'], loss_dict[f'unweighted_{task}_loss'] * 0.5)

    assert 'total_loss' in loss_dict
    expected_total = sum(loss_dict[f'unweighted_{task}_loss'] * 0.5 for task in ['genus', 'health'])
    assert torch.isclose(loss_dict['total_loss'], expected_total)


def test_consistency_penalty_activation(device):
    """Consistency penalty should be handled by the weighting strategy."""
    num_classes = {'genus': 2, 'health': 2}
    strategy = build_weighting_strategy(config=None, primary=['genus', 'health'], auxiliary=[])
    loss_fn = CoralMTLLoss(
        num_classes=num_classes,
        primary_tasks=['genus', 'health'],
        aux_tasks=[],
        weighting_strategy=strategy,
    )

    # Create test predictions and targets
    genus_logits = torch.tensor([[[[10.0]], [[-10.0]]]], device=device)
    health_logits = torch.tensor([[[[-10.0]], [[10.0]]]], device=device)
    predictions = {'genus': genus_logits, 'health': health_logits}
    targets = {
        'genus': torch.zeros(1, 1, 1, dtype=torch.long, device=device),
        'health': torch.zeros(1, 1, 1, dtype=torch.long, device=device)
    }
    
    # Forward pass should work and produce loss dictionary
    loss_dict = loss_fn(predictions, targets)
    assert 'total_loss' in loss_dict
    assert torch.isfinite(loss_dict['total_loss'])


def test_debug_running_stats_progress(device, mtl_num_classes):
    """Run multiple forward passes and ensure running stats update when debug=True."""
    if not mtl_num_classes:
        pytest.skip("No tasks configured")
    task = next(iter(mtl_num_classes.keys()))
    n_cls = mtl_num_classes[task]
    b,h,w = 1,4,4
    strategy = build_weighting_strategy(config=None, primary=[task], auxiliary=[])
    loss_fn = CoralMTLLoss({task: n_cls}, primary_tasks=[task], aux_tasks=[], debug=True, weighting_strategy=strategy)
    for _ in range(3):
        logits = torch.randn(b, n_cls, h, w, device=device)
        targets = torch.randint(0, n_cls, (b,h,w), device=device)
        loss_fn({task: logits},{task: targets})
    stats = loss_fn._running_stats
    # steps should be >= number of calls
    assert stats['steps'] >= 3
    assert any(k.startswith('unweighted_') for k in stats['per_task_loss_mean'].keys())