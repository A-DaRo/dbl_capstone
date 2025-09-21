import pytest
import numpy as np
import torch
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

# Use a non-interactive backend for testing to prevent plots from showing up
import matplotlib
matplotlib.use('Agg')

from coral_mtl.utils.visualization import Visualizer


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary output directory for testing."""
    return str(tmp_path / "test_plots")


@pytest.fixture
def sample_task_info():
    """Create sample task information for testing."""
    return {
        'id2label': {
            'genus': {0: 'background', 1: 'acropora', 2: 'pocillopora'},
            'health': {0: 'background', 1: 'healthy', 2: 'bleached'}
        }
    }


@pytest.fixture
def visualizer(tmp_path):
    """Provides a Visualizer instance with a temporary output directory."""
    return Visualizer(output_dir=str(tmp_path))


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

class TestVisualizerInit:
    """Test Visualizer initialization."""

    def test_init_basic(self, temp_output_dir):
        """Test basic initialization."""
        viz = Visualizer(temp_output_dir)
        assert Path(temp_output_dir).exists()
        assert viz.output_dir == Path(temp_output_dir)
        assert viz.task_info == {}

    def test_init_with_task_info(self, temp_output_dir, sample_task_info):
        """Test initialization with task info."""
        viz = Visualizer(temp_output_dir, task_info=sample_task_info)
        assert viz.task_info == sample_task_info

    def test_init_with_custom_style(self, temp_output_dir):
        """Test initialization with custom matplotlib style."""
        with patch('matplotlib.pyplot.style.use') as mock_style:
            viz = Visualizer(temp_output_dir, style='custom-style')
            mock_style.assert_called_once_with('custom-style')


class TestVisualizerHelperMethods:
    """Test Visualizer helper methods."""

    def test_get_class_names_with_task_info(self, temp_output_dir, sample_task_info):
        """Test getting class names when task info is available."""
        viz = Visualizer(temp_output_dir, task_info=sample_task_info)
        genus_classes = viz._get_class_names('genus')
        expected = ['background', 'acropora', 'pocillopora']  # Sorted by ID
        assert genus_classes == expected

    def test_get_class_names_without_task_info(self, temp_output_dir):
        """Test getting class names when task info is not available."""
        viz = Visualizer(temp_output_dir)
        classes = viz._get_class_names('genus')
        assert classes == []

    def test_get_class_names_unknown_task(self, temp_output_dir, sample_task_info):
        """Test getting class names for unknown task."""
        viz = Visualizer(temp_output_dir, task_info=sample_task_info)
        classes = viz._get_class_names('unknown_task')
        assert classes == []

    def test_save_plot_data_basic(self, temp_output_dir):
        """Test saving basic plot data."""
        viz = Visualizer(temp_output_dir)
        data = {'x': [1, 2, 3], 'y': [4, 5, 6]}
        viz._save_plot_data(data, 'test_plot.png')
        
        json_path = viz.output_dir / 'test_plot.json'
        assert json_path.exists()
        
        with open(json_path, 'r') as f:
            saved_data = json.load(f)
        assert saved_data == data

    def test_save_plot_data_with_numpy(self, temp_output_dir):
        """Test saving plot data with numpy arrays."""
        viz = Visualizer(temp_output_dir)
        data = {'array': np.array([1, 2, 3]), 'scalar': 42}
        viz._save_plot_data(data, 'numpy_test.png')
        
        json_path = viz.output_dir / 'numpy_test.json'
        with open(json_path, 'r') as f:
            saved_data = json.load(f)
        
        assert saved_data['array'] == [1, 2, 3]
        assert saved_data['scalar'] == 42

    def test_save_plot_data_with_torch(self, temp_output_dir):
        """Test saving plot data with torch tensors."""
        viz = Visualizer(temp_output_dir)
        tensor_data = torch.tensor([1.0, 2.0, 3.0])
        tensor_list = [torch.tensor(1.5), torch.tensor(2.5)]
        data = {'tensor': tensor_data, 'tensor_list': tensor_list}
        viz._save_plot_data(data, 'torch_test.png')
        
        json_path = viz.output_dir / 'torch_test.json'
        with open(json_path, 'r') as f:
            saved_data = json.load(f)
        
        assert saved_data['tensor'] == [1.0, 2.0, 3.0]
        assert saved_data['tensor_list'] == [1.5, 2.5]

    def test_denormalize_image(self):
        """Test image denormalization."""
        # Create a normalized image tensor (3, H, W)
        image = torch.randn(3, 32, 32)
        denorm_image = Visualizer.denormalize(image)
        
        # Check output shape and range
        assert denorm_image.shape == (32, 32, 3)
        assert denorm_image.min() >= 0.0
        assert denorm_image.max() <= 1.0
        assert isinstance(denorm_image, np.ndarray)


def test_plot_functions_smoke_test(visualizer, mock_training_log, mock_validation_history):
    """
    Runs a smoke test on all plotting functions to ensure they execute
    without raising exceptions given valid mock data.
    """
    try:
        # Test all plotting functions
        visualizer.plot_training_losses(mock_training_log)
        visualizer.plot_uncertainty_weights(mock_training_log)
        visualizer.plot_learning_rate(mock_training_log, warmup_steps=10)
        visualizer.plot_validation_performance(mock_validation_history)
        
        mock_cm = np.random.randint(0, 100, size=(5, 5))
        visualizer.plot_confusion_analysis(mock_cm, ["a", "b", "c", "d", "e"], "Test Task")
        
        B, C, H, W = 2, 3, 64, 64
        mock_images = torch.rand(B, C, H, W)
        mock_gt_tensor = torch.randint(0, 5, (B, H, W))
        mock_pred_tensor = torch.randint(0, 5, (B, H, W))
        visualizer.plot_qualitative_results(mock_images, mock_gt_tensor, mock_pred_tensor, "Genus")
        
    except Exception as e:
        pytest.fail(f"A visualization function failed its smoke test: {e}")


class TestVisualizerPlottingMethods:
    """Test plotting methods with mocked matplotlib."""

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.tight_layout')
    @patch('matplotlib.pyplot.subplots')
    def test_plot_model_comparison(self, mock_subplots, mock_tight, mock_close, mock_savefig, visualizer):
        """Test model comparison plotting."""
        # Setup mocks
        mock_fig, mock_ax = MagicMock(), MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        mock_ax.patches = []  # Empty patches for annotation loop
        
        results = {
            'CoralMTL': {'mIoU_genus': 0.8, 'H-Mean': 0.75},
            'Baseline': {'mIoU_genus': 0.7, 'H-Mean': 0.65}
        }
        
        with patch('seaborn.barplot') as mock_barplot, \
             patch('pandas.DataFrame') as mock_df:
            visualizer.plot_model_comparison(results)
        
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.subplots')
    def test_plot_validation_performance(self, mock_subplots, mock_close, mock_savefig, visualizer, mock_validation_history):
        """Test validation performance plotting."""
        mock_fig, mock_ax = MagicMock(), MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        visualizer.plot_validation_performance(mock_validation_history)
        
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()
        
        # Check that plot method was called on the axis
        assert mock_ax.plot.call_count >= 1  # At least H-Mean should be plotted


class TestVisualizerStaticMethods:
    """Test static utility methods."""

    @patch('builtins.print')
    def test_print_results_table(self, mock_print):
        """Test results table printing."""
        results = {
            'CoralMTL': {
                'mIoU_genus': 0.80, 'BIoU_genus': 0.75, 
                'mIoU_health': 0.85, 'H-Mean': 0.82
            },
            'Baseline': {
                'mIoU_genus': 0.70, 'BIoU_genus': 0.65,
                'mIoU_health': 0.75, 'H-Mean': 0.72
            }
        }
        
        Visualizer.print_results_table(results)
        
        # Should have called print multiple times (header + separator + data rows)
        assert mock_print.call_count >= 4


def test_print_results_table_output(capsys):
    """
    Tests the results table printing by capturing stdout.
    """
    mock_results = {
        "Model A": {"mIoU_genus": 0.74, "BIoU_genus": 0.65, "mIoU_health": 0.76, "H-Mean": 0.75},
        "Model B": {"mIoU_genus": 0.68, "BIoU_genus": 0.55, "mIoU_health": 0.72, "H-Mean": 0.70}
    }
    
    Visualizer.print_results_table(mock_results)
    
    captured = capsys.readouterr()
    stdout = captured.out
    
    assert "Model A" in stdout
    assert "75.00%" in stdout  # Check for H-Mean of Model A
    assert "Model B" in stdout
    assert "70.00%" in stdout  # Check for H-Mean of Model B
    assert "H-Mean" in stdout  # Check for header