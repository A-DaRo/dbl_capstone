import pytest
import numpy as np
import torch

# Use a non-interactive backend for testing to prevent plots from showing up
import matplotlib
matplotlib.use('Agg')

from coral_mtl.utils import visualization as viz

@pytest.fixture
def mock_training_log():
    """Provides a mock log history simulating a training run."""
    num_steps = 50
    return {
        'total_loss': np.log(np.linspace(10, 1.5, num_steps)),
        'primary_loss': np.log(np.linspace(8, 1, num_steps)),
        'auxiliary_loss': np.log(np.linspace(5, 1, num_steps)),
        'genus_loss': np.log(np.linspace(4.5, 0.6, num_steps)),
        'health_loss': np.log(np.linspace(3.5, 0.4, num_steps)),
        'log_var_genus': np.linspace(0, 0.8, num_steps),
        'log_var_health': np.linspace(0, -0.5, num_steps),
        'lr': np.linspace(0, 6e-5, num_steps)
    }

@pytest.fixture
def mock_validation_history():
    """Provides a mock validation history over several epochs."""
    return {
        'H-Mean': [0.45, 0.55, 0.62, 0.68, 0.71],
        'mIoU_genus': [0.40, 0.50, 0.58, 0.65, 0.69],
        'mIoU_health': [0.50, 0.60, 0.66, 0.71, 0.73]
    }

def test_plot_functions_smoke_test(mock_training_log, mock_validation_history):
    """
    Runs a smoke test on all plotting functions to ensure they execute
    without raising exceptions given valid mock data.
    """
    try:
        viz.plot_composite_losses(mock_training_log)
        viz.plot_individual_task_losses(mock_training_log)
        viz.plot_uncertainty_weights(mock_training_log)
        viz.plot_learning_rate(mock_training_log, warmup_steps=10)
        viz.plot_primary_performance(mock_validation_history)
        
        mock_class_iou = {"class_a": 0.8, "class_b": 0.65}
        viz.plot_per_class_iou_bar_chart(mock_class_iou, "Test Task")
        
        mock_cm = np.random.randint(0, 100, size=(5, 5))
        viz.plot_confusion_matrix(mock_cm, ["a", "b", "c", "d", "e"], "Test Task")
        
        B, C, H, W = 2, 3, 64, 64
        mock_images = torch.rand(B, C, H, W)
        mock_gt = {'Genus': torch.randint(0, 5, (B, H, W))}
        mock_pred = {'Genus': torch.randint(0, 5, (B, H, W))}
        viz.plot_qualitative_grid(mock_images, mock_gt, mock_pred)
        
    except Exception as e:
        pytest.fail(f"A visualization function failed its smoke test: {e}")

def test_print_results_table(capsys):
    """
    Tests the results table printing by capturing stdout.
    """
    mock_results = {
        "Model A": {"mIoU_genus": 0.74, "BIoU_genus": 0.65, "mIoU_health": 0.76, "H-Mean": 0.75},
        "Model B": {"mIoU_genus": 0.68, "BIoU_genus": 0.55, "mIoU_health": 0.72, "H-Mean": 0.70}
    }
    
    viz.print_results_table(mock_results)
    
    captured = capsys.readouterr()
    stdout = captured.out
    
    assert "Model A" in stdout
    assert "75.00%" in stdout  # Check for H-Mean of Model A
    assert "Model B" in stdout
    assert "70.00%" in stdout  # Check for H-Mean of Model B
    assert "H-Mean" in stdout  # Check for header