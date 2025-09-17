import pytest
import numpy as np
import torch
from pathlib import Path

# Use a non-interactive backend for testing to prevent plots from showing up
import matplotlib
matplotlib.use('Agg')

from coral_mtl.utils import visualization as viz

@pytest.fixture
def visualizer(tmp_path):
    """Provides a Visualizer instance with a temporary output directory."""
    return viz.Visualizer(output_dir=str(tmp_path))

@pytest.fixture
def mock_training_log():
    """Provides a mock log history simulating a training run."""
    num_steps = 50
    return {
        'total_loss': np.log(np.linspace(10, 1.5, num_steps)),
        'primary_loss': np.log(np.linspace(8, 1, num_steps)),
        'auxiliary_loss': np.log(np.linspace(5, 1, num_steps)),
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

def test_plot_functions_smoke_test(visualizer, mock_training_log, mock_validation_history):
    """
    Runs a smoke test on all plotting functions to ensure they execute
    without raising exceptions given valid mock data.
    """
    try:
        # Renamed from plot_composite_losses
        visualizer.plot_training_losses(mock_training_log)
        # plot_individual_task_losses is removed, covered by the above
        visualizer.plot_uncertainty_weights(mock_training_log)
        visualizer.plot_learning_rate(mock_training_log, warmup_steps=10)
        # Renamed from plot_primary_performance
        visualizer.plot_validation_performance(mock_validation_history)
        
        # plot_per_class_iou_bar_chart is removed, functionality is part of confusion_analysis
        
        mock_cm = np.random.randint(0, 100, size=(5, 5))
        # Renamed from plot_confusion_matrix
        visualizer.plot_confusion_analysis(mock_cm, ["a", "b", "c", "d", "e"], "Test Task")
        
        B, C, H, W = 2, 3, 64, 64
        mock_images = torch.rand(B, C, H, W)
        # plot_qualitative_grid is now plot_qualitative_results and takes tensors directly
        mock_gt_tensor = torch.randint(0, 5, (B, H, W))
        mock_pred_tensor = torch.randint(0, 5, (B, H, W))
        visualizer.plot_qualitative_results(mock_images, mock_gt_tensor, mock_pred_tensor, "Genus")
        
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
    
    # print_results_table is now a static method on the Visualizer class
    viz.Visualizer.print_results_table(mock_results)
    
    captured = capsys.readouterr()
    stdout = captured.out
    
    assert "Model A" in stdout
    assert "75.00%" in stdout  # Check for H-Mean of Model A
    assert "Model B" in stdout
    assert "70.00%" in stdout  # Check for H-Mean of Model B
    assert "H-Mean" in stdout  # Check for header