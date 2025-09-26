# Edit file: tests/coral_mtl/engine/test_trainer.py
"""
Robust unit tests for the Trainer class.

These tests focus on the Trainer's orchestration logic, using mocks to isolate
it from its dependencies (model, data, metrics, etc.). This ensures that we are
testing the Trainer's core responsibilities: managing the training/validation
loops, handling component lifecycles, and making correct decisions based on metrics.
"""

import pytest
import torch
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, call
import optuna

from coral_mtl.engine.trainer import Trainer
from coral_mtl.engine.gradient_strategies import GradientUpdateStrategy


# --- Fixtures for Mocking Dependencies ---

@pytest.fixture
def mock_config(tmp_path: Path, device):
    """Provides a mock configuration object for the Trainer."""
    return SimpleNamespace(
        device=device.type,
        output_dir=str(tmp_path),
        epochs=2,
        model_selection_metric='optimization_metrics.H-Mean',
        use_mixed_precision=False,
        gradient_accumulation_steps=1,
        patch_size=[32, 32],
        inference_stride=[16, 16],
        inference_batch_size=1
    )


@pytest.fixture
def mock_model(device):
    """Provides a MagicMock for the model."""
    model = MagicMock(spec=torch.nn.Module)
    model.parameters.return_value = [torch.nn.Parameter(torch.randn(1))]
    model.to.return_value = model
    # Fix for PicklingError: mock state_dict to return a serializable object
    model.state_dict.return_value = {'param': torch.randn(1)}
    # Fix for StopIteration: mock model's forward pass to return a dict
    model.return_value = {'genus': torch.randn(1, 10, 32, 32)}
    return model


@pytest.fixture
def mock_loaders():
    """Provides dummy data loaders."""
    train_batch = {'image': torch.randn(2, 3, 32, 32), 'masks': {}}
    val_batch = {'image': torch.randn(2, 3, 64, 64), 'original_mask': torch.zeros(2, 64, 64), 'image_id': ['id1', 'id2']}
    train_loader = [train_batch, train_batch]  # 2 batches per epoch
    val_loader = [val_batch]
    return train_loader, val_loader


@pytest.fixture
def mock_loss_fn():
    """Provides a mock loss function that returns a dummy tensor."""
    loss_fn = MagicMock()
    loss_fn.return_value = {'total_loss': torch.tensor(1.0, requires_grad=True)}
    return loss_fn


@pytest.fixture
def mock_optimizer():
    """Provides a MagicMock for the optimizer."""
    return MagicMock(spec=torch.optim.Optimizer)


@pytest.fixture
def mock_scheduler():
    """Provides a MagicMock for the LR scheduler."""
    scheduler = MagicMock()
    scheduler.get_last_lr.return_value = [1e-4]
    return scheduler

@pytest.fixture
def mock_metrics_calculator():
    """Provides a MagicMock for the metrics calculator."""
    metrics_calc = MagicMock()
    # Simulate improving metrics
    metrics_calc.compute.side_effect = [
        {'optimization_metrics': {'H-Mean': 0.8}},  # Epoch 1
        {'optimization_metrics': {'H-Mean': 0.9}}   # Epoch 2
    ]
    return metrics_calc


@pytest.fixture
def mock_metrics_storer():
    """Provides a MagicMock for the metrics storer."""
    return MagicMock()


@pytest.fixture
def mock_metrics_processor():
    """Provides a MagicMock for the advanced metrics processor."""
    return MagicMock()


@pytest.fixture
def mock_trial():
    """Provides a MagicMock for an Optuna trial."""
    trial = MagicMock(spec=optuna.trial.Trial)
    trial.should_prune.return_value = False
    return trial

# --- Test Cases ---

def test_trainer_smoke_run(mock_model, mock_loaders, mock_loss_fn, mock_optimizer, mock_scheduler,
                           mock_metrics_calculator, mock_metrics_storer, mock_config):
    """A smoke test to ensure a full training run completes without errors."""
    # Arrange
    train_loader, val_loader = mock_loaders
    trainer = Trainer(
        model=mock_model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=mock_loss_fn,
        metrics_calculator=mock_metrics_calculator,
        metrics_storer=mock_metrics_storer,
        optimizer=mock_optimizer,
        scheduler=mock_scheduler,
        config=mock_config
    )

    # Act
    best_metric = trainer.train()

    # Assert
    assert best_metric == 0.9  # The best metric from the mock_metrics_calculator side_effect
    # Check if the main components were called
    assert mock_optimizer.step.call_count == 4  # 2 epochs * 2 batches
    assert mock_scheduler.step.call_count == 4
    assert mock_metrics_calculator.compute.call_count == 2 # Called each validation epoch
    # Validate that train loss keys were merged into optimization_metrics when history stored
    # We inspect calls to store_epoch_history
    assert mock_metrics_storer.store_epoch_history.call_count == 2
    stored_args, _ = mock_metrics_storer.store_epoch_history.call_args
    metrics_payload = stored_args[0]
    opt_metrics = metrics_payload.get('optimization_metrics', {})
    assert any(k.startswith('train_') for k in opt_metrics.keys()), "Train namespaced losses missing in optimization_metrics"


def test_trainer_manages_model_modes(mock_model, mock_loaders, mock_loss_fn, mock_optimizer, mock_scheduler,
                                     mock_metrics_calculator, mock_metrics_storer, mock_config):
    """Verify the trainer correctly sets model.train() and model.eval()."""
    train_loader, val_loader = mock_loaders
    trainer = Trainer(
        model=mock_model, train_loader=train_loader, val_loader=val_loader, loss_fn=mock_loss_fn,
        metrics_calculator=mock_metrics_calculator, metrics_storer=mock_metrics_storer,
        optimizer=mock_optimizer, scheduler=mock_scheduler, config=mock_config
    )
    
    with patch('torch.save'):
        trainer.train()

    # Simply verify that train() and eval() were called the expected number of times
    # For 2 epochs: train() called 2 times (once per epoch)
    # eval() called at least 2 times (once per validation + SlidingWindowInferrer calls)
    assert mock_model.train.call_count == 2
    assert mock_model.eval.call_count >= 2


def test_checkpoint_saving_logic(mock_model, mock_loaders, mock_loss_fn, mock_optimizer, mock_scheduler,
                                 mock_metrics_calculator, mock_metrics_storer, mock_config):
    """Verify that a checkpoint is saved ONLY when the metric improves."""
    train_loader, val_loader = mock_loaders
    # Simulate metric improving then declining
    mock_metrics_calculator.compute.side_effect = [
        {'optimization_metrics': {'H-Mean': 0.8}},  # Epoch 1 (New best)
        {'optimization_metrics': {'H-Mean': 0.7}}   # Epoch 2 (Worse)
    ]
    
    trainer = Trainer(
        model=mock_model, train_loader=train_loader, val_loader=val_loader, loss_fn=mock_loss_fn,
        metrics_calculator=mock_metrics_calculator, metrics_storer=mock_metrics_storer,
        optimizer=mock_optimizer, scheduler=mock_scheduler, config=mock_config
    )

    with patch('torch.save') as mock_torch_save:
        trainer.train()
        
        # torch.save should have been called exactly ONCE, after the first epoch.
        mock_torch_save.assert_called_once()
        expected_path = str(Path(mock_config.output_dir) / "best_model.pth")
        assert mock_torch_save.call_args[0][1] == expected_path


def test_advanced_metrics_processor_lifecycle(
    mock_model, mock_loaders, mock_loss_fn, mock_optimizer, mock_scheduler,
    mock_metrics_calculator, mock_metrics_storer, mock_config, mock_metrics_processor
):
    """Verify the lifecycle of the AdvancedMetricsProcessor is correctly managed."""
    train_loader, val_loader = mock_loaders
    trainer = Trainer(
        model=mock_model, train_loader=train_loader, val_loader=val_loader, loss_fn=mock_loss_fn,
        metrics_calculator=mock_metrics_calculator, metrics_storer=mock_metrics_storer,
        optimizer=mock_optimizer, scheduler=mock_scheduler, config=mock_config,
        metrics_processor=mock_metrics_processor
    )
    
    trainer.train()
    # Ensure training loss integration occurred
    assert mock_metrics_storer.store_epoch_history.call_count == mock_config.epochs

    mock_metrics_processor.start.assert_called_once()
    mock_metrics_processor.shutdown.assert_called_once()
    # 2 validation epochs * 1 batch per epoch * 2 items in batch = 4 jobs
    assert mock_metrics_processor.dispatch_image_job.call_count == (mock_config.epochs * len(val_loader) * 2)


def test_lifecycle_on_error(
    mock_model, mock_loaders, mock_loss_fn, mock_optimizer, mock_scheduler,
    mock_metrics_calculator, mock_metrics_storer, mock_config, mock_metrics_processor
):
    """Ensures cleanup methods are called even if training loop fails."""
    train_loader, val_loader = mock_loaders
    mock_loss_fn.side_effect = ValueError("Training failed!")
    
    trainer = Trainer(
        model=mock_model, train_loader=train_loader, val_loader=val_loader, loss_fn=mock_loss_fn,
        metrics_calculator=mock_metrics_calculator, metrics_storer=mock_metrics_storer,
        optimizer=mock_optimizer, scheduler=mock_scheduler, config=mock_config,
        metrics_processor=mock_metrics_processor
    )

    with pytest.raises(ValueError, match="Training failed!"):
        trainer.train()

    # The key check: cleanup methods must be called via the 'finally' block
    mock_metrics_storer.close.assert_called_once()
    mock_metrics_processor.shutdown.assert_called_once()


def test_optuna_pruning(
    mock_model, mock_loaders, mock_loss_fn, mock_optimizer, mock_scheduler,
    mock_metrics_calculator, mock_metrics_storer, mock_config, mock_trial
):
    """Verify that an OptunaPruned exception is raised if the trial should be pruned."""
    train_loader, val_loader = mock_loaders
    mock_trial.should_prune.return_value = True  # Simulate a prune signal
    
    trainer = Trainer(
        model=mock_model, train_loader=train_loader, val_loader=val_loader, loss_fn=mock_loss_fn,
        metrics_calculator=mock_metrics_calculator, metrics_storer=mock_metrics_storer,
        optimizer=mock_optimizer, scheduler=mock_scheduler, config=mock_config,
        trial=mock_trial
    )
    
    with pytest.raises(optuna.exceptions.TrialPruned):
        trainer.train()
        
    # Verify that trial.report was still called before pruning
    mock_trial.report.assert_called_once_with(0.8, 0) # Metric from epoch 0


def test_trainer_handles_gradient_strategy_without_backward(monkeypatch, tmp_path):
    """Ensure manual gradient strategies integrate without autograd backward."""

    class ToyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(3 * 32 * 32, 16),
                torch.nn.ReLU(),
                torch.nn.Linear(16, 4),
            )

        def forward(self, x):
            logits = self.net(x)
            return {'task_a': logits}

    class DummyStrategy(GradientUpdateStrategy):
        def __init__(self, tasks):
            super().__init__(tasks)

        def compute_update_vector(self, per_task_gradients):
            task_names = list(per_task_gradients.keys())
            stacked = torch.stack([per_task_gradients[name] for name in task_names], dim=0)
            weights = torch.ones(len(task_names), device=stacked.device, dtype=stacked.dtype) / max(1, len(task_names))
            self._record_weights(task_names, weights)
            return stacked.mean(dim=0)

    class DummyLoss:
        def __init__(self, strategy):
            self.weighting_strategy = strategy
            self.primary_tasks = strategy.tasks

        def compute_unweighted_losses(self, predictions, masks):
            base = predictions['task_a'].mean()
            return {task: base + (0.1 * idx) for idx, task in enumerate(self.primary_tasks)}

        def __call__(self, predictions, masks):
            return {'total_loss': predictions['task_a'].mean()}

    train_batch = {'image': torch.randn(2, 3, 32, 32), 'masks': {'task_a': torch.zeros(2, 32, 32)}}
    train_loader = [train_batch, train_batch]
    val_loader = []

    strategy = DummyStrategy(tasks=['task_a', 'task_b'])
    model = ToyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    optimizer_step_spy = MagicMock(side_effect=optimizer.step)
    optimizer.step = optimizer_step_spy
    scheduler_step_spy = MagicMock(side_effect=scheduler.step)
    scheduler.step = scheduler_step_spy

    config = SimpleNamespace(
        device='cpu',
        output_dir=str(tmp_path),
        epochs=1,
        model_selection_metric='global.BIoU',
        use_mixed_precision=False,
        gradient_accumulation_steps=1,
        patch_size=[32, 32],
        inference_stride=[16, 16],
        inference_batch_size=1
    )

    metrics_calculator = MagicMock()
    metrics_storer = MagicMock()

    dummy_loss = DummyLoss(strategy=strategy)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=dummy_loss,
        metrics_calculator=metrics_calculator,
        metrics_storer=metrics_storer,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config
    )

    monkeypatch.setattr(Trainer, '_should_validate', lambda self, epoch: False)

    trainer.train()

    assert optimizer_step_spy.call_count == len(train_loader)
    assert scheduler_step_spy.call_count == len(train_loader)
    metrics_calculator.compute.assert_not_called()