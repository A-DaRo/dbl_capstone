import torch
import torch.nn as nn
import pytest
import os
from unittest.mock import MagicMock
from types import SimpleNamespace
import optuna

from coral_mtl.engine.trainer import Trainer
from coral_mtl.engine.inference import SlidingWindowInferrer

# --- 1. Mock Components for Predictable Testing ---

class MockSimpleModel(nn.Module):
    """
    A minimal model that produces segmentation-like output (B, C, H, W).
    It uses a simple convolution to change channel dimensions while preserving
    spatial dimensions, which is what the SlidingWindowInferrer expects.
    """
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        # This 1x1 convolution changes the number of channels from in_channels
        # to num_classes while keeping height and width the same.
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        # x shape: (B, C, H, W)
        # The model's forward pass needs to be compatible with the SlidingWindowInferrer,
        # which expects a dictionary of tensors with spatial dimensions.
        logits = self.conv(x)  # -> (B, num_classes, H, W)
        return {'task1': logits}


class MockMetricsCalculator:
    """
    A mock metrics calculator that returns predictable, increasing scores,
    and is robust to the trainer's reset() call between epochs.
    """
    def __init__(self):
        self.epoch_count = 0

    def update(self, preds, targets):
        pass  # No-op for testing

    def compute(self):
        # This counter is not reset, so the metric improves each time compute is called.
        self.epoch_count += 1
        return {'H-Mean': 0.5 + self.epoch_count * 0.1}

    def reset(self):
        # This method is called by the trainer, but we make it a no-op 
        # for our epoch counter to simulate an improving metric.
        pass

# --- 2. Pytest Fixtures for Reusable Setup ---

@pytest.fixture
def mock_config(tmp_path):
    """Provides a mock configuration object and a temporary output directory."""
    output_dir = tmp_path / "outputs"
    return SimpleNamespace(
        DEVICE=torch.device("cpu"),
        NUM_EPOCHS=2,
        GRADIENT_ACCUMULATION_STEPS=1,
        OUTPUT_DIR=str(output_dir),
        BEST_MODEL_NAME="best_model.pth",
        # Add inference-related parameters required by the Trainer's validation loop
        PATCH_SIZE=16,
        INFERENCE_STRIDE=8,
        INFERENCE_BATCH_SIZE=2,
        MODEL_SELECTION_METRIC='H-Mean'
    )

@pytest.fixture
def mock_data_loader():
    """
    Provides a mock data loader that yields batches suitable for the SlidingWindowInferrer.
    The validation loader should yield full-sized images, not patches.
    The training loader provides patches and corresponding mask patches.
    """
    # Training data: batches of patches (4 patches per batch)
    train_batch1 = {'image': torch.randn(4, 3, 16, 16), 'masks': {'task1': torch.randint(0, 10, (4, 16, 16))}}
    train_batch2 = {'image': torch.randn(4, 3, 16, 16), 'masks': {'task1': torch.randint(0, 10, (4, 16, 16))}}
    
    # Validation data: a single "full" image and its corresponding full mask
    val_image = torch.randn(1, 3, 32, 32)
    val_masks = {'task1': torch.randint(0, 10, (1, 32, 32))}
    val_batch = {'image': val_image, 'masks': val_masks}
    
    return [train_batch1, train_batch2], [val_batch]

# --- 3. Test Cases for the Trainer Class ---

def test_trainer_instantiation(mock_config, mock_data_loader):
    """Tests if the Trainer can be initialized without errors."""
    train_loader, val_loader = mock_data_loader
    model = MockSimpleModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    loss_fn = lambda pred, target: nn.CrossEntropyLoss()(pred['task1'], target['task1'].long())
    metrics_calc = MockMetricsCalculator()
    
    try:
        Trainer(model, train_loader, val_loader, loss_fn, metrics_calc,
                optimizer, scheduler, mock_config)
    except Exception as e:
        pytest.fail(f"Trainer instantiation failed: {e}")

def test_trainer_single_step_updates_weights(mock_config, mock_data_loader):
    """
    Verifies the most critical trainer function: that a training step
    actually results in a model weight update.
    """
    train_loader, val_loader = mock_data_loader
    model = MockSimpleModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    loss_fn = lambda pred, target: nn.CrossEntropyLoss()(pred['task1'], target['task1'].long())
    metrics_calc = MockMetricsCalculator()
    
    # Clone initial weights to compare against later
    initial_weights = model.conv.weight.clone().detach()

    trainer = Trainer(model, train_loader, val_loader, loss_fn, metrics_calc,
                      optimizer, scheduler, mock_config)
    
    # Run only one training epoch
    trainer.config.NUM_EPOCHS = 1
    trainer.train()

    final_weights = model.conv.weight.clone().detach()

    # The core assertion: weights must have changed after training.
    assert not torch.equal(initial_weights, final_weights)

def test_trainer_checkpointing_on_metric_improvement(mock_config, mock_data_loader):
    """
    Tests that a model checkpoint is saved when the validation metric improves.
    """
    train_loader, val_loader = mock_data_loader
    model = MockSimpleModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    loss_fn = lambda pred, target: torch.tensor(1.0, requires_grad=True) # Dummy loss
    metrics_calc = MockMetricsCalculator() # This mock returns improving scores

    trainer = Trainer(model, train_loader, val_loader, loss_fn, metrics_calc,
                      optimizer, scheduler, mock_config)
    
    trainer.train()

    expected_checkpoint_path = os.path.join(mock_config.OUTPUT_DIR, mock_config.BEST_MODEL_NAME)
    assert os.path.exists(expected_checkpoint_path), "Checkpoint was not saved on metric improvement."
    # The best metric should be from the last epoch (0.5 + 2*0.1)
    assert trainer.best_metric == pytest.approx(0.7)

def test_trainer_with_optuna_pruning(mock_config, mock_data_loader):
    """Tests if the trainer correctly raises a TrialPruned exception when told to."""
    train_loader, val_loader = mock_data_loader
    model = MockSimpleModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    loss_fn = lambda pred, target: torch.tensor(1.0, requires_grad=True)
    metrics_calc = MockMetricsCalculator()
    
    # Create a mock Optuna trial that should always prune
    mock_trial = MagicMock(spec=optuna.trial.Trial)
    mock_trial.should_prune.return_value = True

    trainer = Trainer(model, train_loader, val_loader, loss_fn, metrics_calc,
                      optimizer, scheduler, mock_config, trial=mock_trial)
    
    # Expect the trainer to raise this specific exception
    with pytest.raises(optuna.exceptions.TrialPruned):
        trainer.train()

    # Verify that the trial's report method was called before pruning
    mock_trial.report.assert_called_once()

def test_trainer_with_optuna_reporting(mock_config, mock_data_loader):
    """Tests that the trainer reports its metric to Optuna each epoch."""
    train_loader, val_loader = mock_data_loader
    model = MockSimpleModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    loss_fn = lambda pred, target: torch.tensor(1.0, requires_grad=True)
    metrics_calc = MockMetricsCalculator()
    
    # Create a mock Optuna trial that never prunes
    mock_trial = MagicMock(spec=optuna.trial.Trial)
    mock_trial.should_prune.return_value = False

    trainer = Trainer(model, train_loader, val_loader, loss_fn, metrics_calc,
                      optimizer, scheduler, mock_config, trial=mock_trial)
    
    trainer.train()

    # The report method should have been called once for each epoch
    assert mock_trial.report.call_count == mock_config.NUM_EPOCHS
    # Check that the last reported value was the best metric from the last epoch
    mock_trial.report.assert_called_with(pytest.approx(0.7), 1)