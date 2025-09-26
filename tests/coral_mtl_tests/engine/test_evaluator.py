# Edit file: tests/coral_mtl/engine/test_evaluator.py
"""
Robust tests for the Evaluator class.

These tests verify the orchestration logic of the Evaluator, ensuring it correctly
manages the evaluation lifecycle, interacts with its components (model, inferrer,
metrics calculators, storers), and handles different model architectures.
"""
import pytest
import torch
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, call

from coral_mtl.engine.evaluator import Evaluator


@pytest.fixture
def mock_config(tmp_path: Path, device):
    """Provides a mock configuration object for the Evaluator."""
    return SimpleNamespace(
        device=device.type,
        output_dir=str(tmp_path),
        checkpoint_path=None,
        patch_size_h=32,
        patch_size_w=32,
        inference_stride_h=16,
        inference_stride_w=16,
        inference_batch_size=1
    )


@pytest.fixture
def mock_metrics_storer():
    """Provides a MagicMock for MetricsStorer."""
    return MagicMock()


@pytest.fixture
def mock_metrics_calculator():
    """Provides a MagicMock for AbstractCoralMetrics."""
    mock = MagicMock()
    mock.compute.return_value = {"optimization_metrics": {"H-Mean": 0.95}}
    return mock


@pytest.fixture
def mock_metrics_processor():
    """Provides a MagicMock for AdvancedMetricsProcessor."""
    return MagicMock()


@pytest.fixture
def dummy_mtl_test_loader(device):
    """Provides a dummy dataloader yielding MTL-style batches."""
    batch = {
        'image': torch.randn(2, 3, 64, 64, device=device),
        'original_mask': torch.randint(0, 10, (2, 64, 64), device=device, dtype=torch.long),
        'image_id': ['img_001', 'img_002']
    }
    return [batch]  # Dataloader is an iterable of batches


@pytest.fixture
def dummy_baseline_test_loader(device):
    """Provides a dummy dataloader yielding baseline-style batches."""
    # Note: The batch structure is identical, as per AbstractCoralscapesDataset
    batch = {
        'image': torch.randn(2, 3, 64, 64, device=device),
        'original_mask': torch.randint(0, 10, (2, 64, 64), device=device, dtype=torch.long),
        'image_id': ['img_003', 'img_004']
    }
    return [batch]


def test_evaluator_full_run_mtl_model(
    minimal_coral_mtl_model,
    dummy_mtl_test_loader,
    mock_metrics_calculator,
    mock_metrics_storer,
    mock_config,
    device
):
    """
    Test a full evaluation run with an MTL model and no advanced processor.
    Verifies the entire lifecycle: setup, loop, metrics calculation, and storage.
    """
    # Arrange
    evaluator = Evaluator(
        model=minimal_coral_mtl_model,
        test_loader=dummy_mtl_test_loader,
        metrics_calculator=mock_metrics_calculator,
        metrics_storer=mock_metrics_storer,
        config=mock_config
    )

    # Act
    final_report = evaluator.evaluate()

    # Assert
    # 1. Model state
    assert not minimal_coral_mtl_model.training, "Model should be in eval mode"
    assert next(minimal_coral_mtl_model.parameters()).device.type == device.type

    # 2. Metrics Calculator lifecycle
    mock_metrics_calculator.reset.assert_called_once()
    mock_metrics_calculator.update.assert_called_once()
    mock_metrics_calculator.compute.assert_called_once()

    # 3. Metrics Storer lifecycle
    mock_metrics_storer.open_for_run.assert_called_once_with(is_testing=True)
    mock_metrics_storer.save_final_report.assert_called_once_with(
        final_report, "test_metrics_full_report.json"
    )
    mock_metrics_storer.close.assert_called_once()

    # 4. Correct data passed to metrics_calculator.update
    update_call_args = mock_metrics_calculator.update.call_args[1]
    assert 'predictions' in update_call_args
    assert 'original_targets' in update_call_args
    assert 'predictions_logits' in update_call_args
    assert isinstance(update_call_args['predictions'], dict) # MTL model passes a dict
    assert torch.is_tensor(update_call_args['original_targets'])
    assert update_call_args['image_ids'] == ['img_001', 'img_002']
    assert update_call_args['is_testing'] is True


def test_evaluator_with_advanced_metrics_processor(
    minimal_coral_mtl_model,
    dummy_mtl_test_loader,
    mock_metrics_calculator,
    mock_metrics_storer,
    mock_metrics_processor,
    mock_config
):
    """Verify the lifecycle of the AdvancedMetricsProcessor is correctly managed."""
    # Arrange
    evaluator = Evaluator(
        model=minimal_coral_mtl_model,
        test_loader=dummy_mtl_test_loader,
        metrics_calculator=mock_metrics_calculator,
        metrics_storer=mock_metrics_storer,
        config=mock_config,
        metrics_processor=mock_metrics_processor
    )
    
    # Act
    evaluator.evaluate()

    # Assert
    # 1. Processor lifecycle
    mock_metrics_processor.start.assert_called_once()
    mock_metrics_processor.shutdown.assert_called_once()
    
    # 2. Job dispatching
    # The dummy loader has one batch of size 2, so dispatch should be called twice.
    assert mock_metrics_processor.dispatch_image_job.call_count == 2
    
    # Inspect the first call to ensure correct data is dispatched
    first_call_args = mock_metrics_processor.dispatch_image_job.call_args_list[0].kwargs
    assert first_call_args['image_id'] == 'img_001'
    assert torch.is_tensor(first_call_args['pred_mask_tensor'])
    assert torch.is_tensor(first_call_args['target_mask_tensor'])
    assert first_call_args['pred_mask_tensor'].shape == (64, 64)
    assert first_call_args['target_mask_tensor'].shape == (64, 64)


@patch('coral_mtl.engine.evaluator.torch.load')
def test_evaluator_loads_checkpoint(
    mock_torch_load,
    minimal_coral_mtl_model,
    dummy_mtl_test_loader,
    mock_metrics_calculator,
    mock_metrics_storer,
    mock_config
):
    """Verify that a provided checkpoint path is used to load model weights."""
    # Arrange
    mock_config.checkpoint_path = "/fake/path/to/best_model.pth"
    minimal_coral_mtl_model.load_state_dict = MagicMock()
    mock_state_dict = {"param1": torch.randn(1)}
    mock_torch_load.return_value = mock_state_dict
    
    evaluator = Evaluator(
        model=minimal_coral_mtl_model,
        test_loader=dummy_mtl_test_loader,
        metrics_calculator=mock_metrics_calculator,
        metrics_storer=mock_metrics_storer,
        config=mock_config
    )
    
    # Act
    evaluator.evaluate()
    
    # Assert
    mock_torch_load.assert_called_once_with(mock_config.checkpoint_path, map_location=torch.device(mock_config.device))
    minimal_coral_mtl_model.load_state_dict.assert_called_once_with(mock_state_dict)


def test_evaluator_handles_baseline_model(
    minimal_baseline_model,
    dummy_baseline_test_loader,
    mock_metrics_calculator,
    mock_metrics_storer,
    mock_config
):
    """
    Test that the evaluator correctly processes the single-tensor output
    from a baseline model.
    """
    # Arrange
    evaluator = Evaluator(
        model=minimal_baseline_model,
        test_loader=dummy_baseline_test_loader,
        metrics_calculator=mock_metrics_calculator,
        metrics_storer=mock_metrics_storer,
        config=mock_config
    )

    # Act
    evaluator.evaluate()

    # Assert
    # The key check: ensure the single tensor is passed to metrics.update, not a dict
    update_call_args = mock_metrics_calculator.update.call_args[1]
    assert 'predictions' in update_call_args
    # It should be a tensor, not a dictionary. The logic inside evaluator should have
    # unpacked the {'segmentation': tensor} from the inferrer.
    assert torch.is_tensor(update_call_args['predictions'])
    assert update_call_args['image_ids'] == ['img_003', 'img_004']


def test_evaluator_lifecycle_on_error(
    minimal_coral_mtl_model,
    dummy_mtl_test_loader,
    mock_metrics_calculator,
    mock_metrics_storer,
    mock_metrics_processor,
    mock_config
):
    """
    Ensures that shutdown() and close() are called in a finally block
    even if the evaluation loop fails.
    """
    # Arrange
    # Simulate an error during the metric update phase
    mock_metrics_calculator.update.side_effect = ValueError("Test Exception")
    
    evaluator = Evaluator(
        model=minimal_coral_mtl_model,
        test_loader=dummy_mtl_test_loader,
        metrics_calculator=mock_metrics_calculator,
        metrics_storer=mock_metrics_storer,
        config=mock_config,
        metrics_processor=mock_metrics_processor
    )
    
    # Act & Assert
    with pytest.raises(ValueError, match="Test Exception"):
        evaluator.evaluate()
        
    # Crucial assertions: check that cleanup methods were still called
    mock_metrics_storer.close.assert_called_once()
    mock_metrics_processor.shutdown.assert_called_once()


def test_evaluator_persists_test_loss_metrics(tmp_path: Path, device):
    """Evaluator should create test_loss_metrics.json including namespaced test_ keys."""
    class TinyBaseline(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 3, 1)
        def forward(self, x):
            return {'segmentation': self.conv(x)}

    model = TinyBaseline().to(device)
    batch = {
        'image': torch.randn(1,3,32,32, device=device),
        'original_mask': torch.zeros(1,32,32, dtype=torch.long, device=device),
        'image_id': ['sample'],
        'mask': torch.zeros(1,32,32, dtype=torch.long, device=device)
    }
    test_loader = [batch]

    # Minimal metrics calculator mock
    metrics_calc = MagicMock()
    metrics_calc.reset.return_value = None
    metrics_calc.update.return_value = None
    metrics_calc.compute.return_value = {'optimization_metrics': {'H-Mean': 0.42}}

    from coral_mtl.metrics.metrics_storer import MetricsStorer
    storer = MetricsStorer(str(tmp_path))

    config = SimpleNamespace(
        device=device.type,
        output_dir=str(tmp_path),
        checkpoint_path=None,
        patch_size=[32,32],
        inference_stride=[16,16],
        inference_batch_size=1
    )

    evaluator = Evaluator(
        model=model,
        test_loader=test_loader,
        metrics_calculator=metrics_calc,
        metrics_storer=storer,
        config=config
    )

    # Inject a simple loss_fn onto evaluator
    def dummy_loss(preds, target):
        if isinstance(preds, dict):
            tensor = preds.get('segmentation')
        else:
            tensor = preds
        return {'total_loss': tensor.abs().mean() + 1.0}
    evaluator.loss_fn = dummy_loss

    evaluator.evaluate()

    loss_file = Path(tmp_path) / 'test_loss_metrics.json'
    assert loss_file.exists(), "Expected test_loss_metrics.json to be written"
    text = loss_file.read_text()
    assert 'test_total_loss' in text, "Namespaced test_total_loss missing in persisted file"


def test_evaluator_skips_loss_when_not_provided(tmp_path: Path, device):
    class TinyBaseline(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 3, 1)
        def forward(self, x):
            return {'segmentation': self.conv(x)}

    model = TinyBaseline().to(device)
    batch = {
        'image': torch.randn(1,3,32,32, device=device),
        'original_mask': torch.zeros(1,32,32, dtype=torch.long, device=device),
        'image_id': ['sample'],
        'mask': torch.zeros(1,32,32, dtype=torch.long, device=device)
    }
    test_loader = [batch]

    metrics_calc = MagicMock()
    metrics_calc.reset.return_value = None
    metrics_calc.update.return_value = None
    metrics_calc.compute.return_value = {'optimization_metrics': {}}

    storer = MagicMock()

    config = SimpleNamespace(
        device=device.type,
        output_dir=str(tmp_path),
        checkpoint_path=None,
        patch_size=[32,32],
        inference_stride=[16,16],
        inference_batch_size=1
    )

    evaluator = Evaluator(
        model=model,
        test_loader=test_loader,
        metrics_calculator=metrics_calc,
        metrics_storer=storer,
        config=config,
        loss_fn=None
    )

    evaluator.evaluate()

    # Ensure no loss entries were accumulated/persisted when loss_fn is missing
    for call_kwargs in metrics_calc.update.call_args_list:
        assert 'predictions' in call_kwargs.kwargs
    storer.save_test_loss_report.assert_not_called()