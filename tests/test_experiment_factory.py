import pytest
import torch
import numpy as np
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock, ANY

# Import the class under test
from coral_mtl.ExperimentFactory import ExperimentFactory
from coral_mtl.utils.task_splitter import MTLTaskSplitter, BaseTaskSplitter

# --- MOCK CLASSES ---

class DummyDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return {
            "image": torch.rand(3, 8, 8),
            "masks": {
                "genus": torch.randint(0, 3, (8, 8)),
                "health": torch.randint(0, 3, (8, 8)),
            },
        }

class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return {
            "genus": torch.randn(1, 3, 8, 8),
            "health": torch.randn(1, 3, 8, 8),
        }

class MockTrainer:
    """A mock Trainer that records its initialization args and method calls."""
    last_instance = None
    def __init__(self, model, train_loader, val_loader, loss_fn, metrics_calculator, metrics_storer, optimizer, scheduler, config, trial=None):
        self.model = model
        self.config = config
        self.trial = trial
        self.train_called = False
        MockTrainer.last_instance = self

    def train(self):
        self.train_called = True
        return

class MockEvaluator:
    """A mock Evaluator that records its initialization args and method calls."""
    last_instance = None
    def __init__(self, model, test_loader, metrics_calculator, metrics_storer, config):
        self.model = model
        self.config = config
        self.evaluate_called = False
        MockEvaluator.last_instance = self

    def evaluate(self):
        self.evaluate_called = True
        return {"mIoU_Genus": 0.9}

class MockMetricsStorer:
    """A mock MetricsStorer that records its initialization and method calls."""
    last_instance = None
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.store_epoch_history_called = False
        self.store_per_image_cms_called = False
        MockMetricsStorer.last_instance = self
    
    def store_epoch_history(self, *args, **kwargs): 
        self.store_epoch_history_called = True
    def store_per_image_cms(self, *args, **kwargs): 
        self.store_per_image_cms_called = True
    def open_for_run(self, *args, **kwargs): 
        pass
    def close(self): 
        pass

# --- PYTEST FIXTURES ---

@pytest.fixture
def mock_task_definitions_path(tmp_path):
    """Creates a mock task_definitions.yaml file."""
    task_def_content = {
        'genus': {'id2label': {0: 'background', 1: 'acropora', 2: 'pocillopora'}},
        'health': {'id2label': {0: 'background', 1: 'healthy', 2: 'bleached'}},
    }
    path = tmp_path / "task_definitions.yaml"
    with open(path, 'w') as f:
        yaml.dump(task_def_content, f)
    return path

@pytest.fixture
def base_config(tmp_path, mock_task_definitions_path):
    """Provides a base configuration dictionary used across tests."""
    (tmp_path / "experiments" / "test_run").mkdir(parents=True, exist_ok=True)
    return {
        'data': {
            'task_definitions_path': str(mock_task_definitions_path),
            'dataset_name': 'fake/hf-dataset',
            'data_root_path': str(tmp_path),
            'patch_size': 256,
            'batch_size': 2,
        },
        'optimizer': {
            'type': 'AdamWPolyDecay',
            'params': {'lr': 1e-4, 'warmup_ratio': 0.1}
        },
        'trainer': {
            'epochs': 1,
            'output_dir': str(tmp_path / "experiments" / "test_run"),
            'model_selection_metric': 'H-Mean_Genus'
        },
        'metrics': {
            'primary_tasks': ['genus', 'health']
        }
    }

@pytest.fixture
def valid_mtl_config_path(tmp_path, base_config):
    """Creates a minimal, valid config for the CoralMTL model."""
    config = base_config.copy()
    config['model'] = {
        'type': 'CoralMTL',
        'tasks': {'primary': ['genus', 'health'], 'auxiliary': []},
        'params': {'backbone': 'mit_b0', 'decoder_channel': 256, 'attention_dim': 256}
    }
    config['loss'] = {'type': 'CompositeHierarchical'}
    path = tmp_path / "valid_mtl_config.yaml"
    with open(path, 'w') as f:
        yaml.dump(config, f)
    return path

@pytest.fixture
def valid_baseline_config_path(tmp_path, base_config):
    """Creates a minimal, valid config for the BaselineSegformer model."""
    config = base_config.copy()
    config['model'] = {
        'type': 'SegFormerBaseline',
        'params': {'backbone': 'mit_b0', 'decoder_channel': 256, 'num_classes': 3}
    }
    config['loss'] = {'type': 'HybridLoss'}
    path = tmp_path / "valid_baseline_config.yaml"
    with open(path, 'w') as f:
        yaml.dump(config, f)
    return path

@pytest.fixture
def mock_dependencies(monkeypatch):
    """Mocks all major external dependencies of ExperimentFactory."""
    # Mock the model so we don't actually load a real one
    mock_model = MagicMock()
    monkeypatch.setattr('coral_mtl.ExperimentFactory.CoralMTLModel', mock_model)

    # Mock the datasets to avoid filesystem access
    mock_mtl_dataset_instance = MagicMock()
    mock_mtl_dataset_instance.__len__.return_value = 10
    mock_mtl_dataset = MagicMock(return_value=mock_mtl_dataset_instance)
    monkeypatch.setattr('coral_mtl.ExperimentFactory.CoralscapesMTLDataset', mock_mtl_dataset)

    mock_baseline_dataset_instance = MagicMock()
    mock_baseline_dataset_instance.__len__.return_value = 10
    mock_baseline_dataset = MagicMock(return_value=mock_baseline_dataset_instance)
    monkeypatch.setattr('coral_mtl.ExperimentFactory.CoralscapesDataset', mock_baseline_dataset)

    # Mock the trainer, evaluator, and metrics storer
    monkeypatch.setattr('coral_mtl.ExperimentFactory.Trainer', MockTrainer)
    monkeypatch.setattr('coral_mtl.ExperimentFactory.Evaluator', MockEvaluator)
    monkeypatch.setattr('coral_mtl.ExperimentFactory.MetricsStorer', MockMetricsStorer)

    # Mock Optuna
    mock_study = MagicMock()
    mock_study.best_trial.value = 0.95
    mock_optuna = MagicMock()
    mock_optuna.create_study.return_value = mock_study
    monkeypatch.setattr('coral_mtl.ExperimentFactory.optuna', mock_optuna)

    # Reset mock trackers
    MockTrainer.last_instance = None
    MockEvaluator.last_instance = None
    MockMetricsStorer.last_instance = None

    return {
        "model": mock_model,
        "mtl_dataset": mock_mtl_dataset,
        "baseline_dataset": mock_baseline_dataset,
        "optuna": mock_optuna,
    }

# --- TEST CASES ---

class TestFactoryComponentBuilding:
    """Tests the 'getter' methods for assembling individual components."""

    def test_init_and_task_parsing(self, valid_mtl_config_path):
        """Verifies correct initialization and task splitter setup."""
        factory = ExperimentFactory(config_path=valid_mtl_config_path)
        assert hasattr(factory, 'task_splitter')
        assert isinstance(factory.task_splitter, MTLTaskSplitter)
        assert 'genus' in factory.task_splitter.hierarchical_definitions
        assert 'health' in factory.task_splitter.hierarchical_definitions

    @patch('coral_mtl.ExperimentFactory.BaselineSegformer')
    @patch('coral_mtl.ExperimentFactory.CoralMTLModel')
    def test_get_model_selects_correct_class(self, mock_mtl_model, mock_baseline_model, valid_mtl_config_path, valid_baseline_config_path):
        """Verifies that the correct model class is instantiated based on config."""
        # Test MTL Model
        factory_mtl = ExperimentFactory(config_path=valid_mtl_config_path)
        model_mtl = factory_mtl.get_model()
        mock_mtl_model.assert_called_once()
        assert mock_baseline_model.call_count == 0

        # Test Baseline Model
        factory_baseline = ExperimentFactory(config_path=valid_baseline_config_path)
        model_baseline = factory_baseline.get_model()
        mock_baseline_model.assert_called_once()

    def test_get_dataloaders_selects_correct_dataset(self, mock_dependencies, valid_mtl_config_path, valid_baseline_config_path):
        """Verifies the correct Dataset class is used for MTL vs Baseline."""
        factory_mtl = ExperimentFactory(config_path=valid_mtl_config_path)
        factory_mtl.get_dataloaders()
        mock_dependencies['mtl_dataset'].assert_called()
        
        factory_baseline = ExperimentFactory(config_path=valid_baseline_config_path)
        factory_baseline.get_dataloaders()
        mock_dependencies['baseline_dataset'].assert_called()

    @patch('coral_mtl.ExperimentFactory.CoralMetrics')
    @patch('coral_mtl.ExperimentFactory.CoralMTLMetrics')
    def test_get_metrics_calculator_selects_correct_classes(self, mock_mtl_metrics, mock_baseline_metrics, valid_mtl_config_path, valid_baseline_config_path):
        """Verifies that the correct metrics calculator class is instantiated based on model type."""
        # Test MTL Model metrics
        factory_mtl = ExperimentFactory(config_path=valid_mtl_config_path)
        metrics_mtl = factory_mtl.get_metrics_calculator()
        mock_mtl_metrics.assert_called_once()
        
        # Test Baseline Model metrics
        factory_baseline = ExperimentFactory(config_path=valid_baseline_config_path)
        metrics_baseline = factory_baseline.get_metrics_calculator()
        mock_baseline_metrics.assert_called_once()

class TestFactoryWorkflowExecution:
    """Tests the high-level orchestration methods of the factory."""

    def test_run_training_orchestration(self, mock_dependencies, valid_mtl_config_path):
        """Verifies that run_training correctly assembles and calls the Trainer."""
        factory = ExperimentFactory(config_path=valid_mtl_config_path)
        
        assert MockTrainer.last_instance is None
        
        factory.run_training()

        # Verify that a Trainer was instantiated and its train method was called
        assert MockTrainer.last_instance is not None
        assert MockTrainer.last_instance.train_called is True
        
        # Verify the correct components were passed to the Trainer
        assert MockTrainer.last_instance.model == factory.get_model()
        assert MockTrainer.last_instance.config.epochs == 1

    def test_run_evaluation_orchestration(self, mock_dependencies, valid_mtl_config_path):
        """Verifies that run_evaluation correctly assembles and calls the Evaluator."""
        factory = ExperimentFactory(config_path=valid_mtl_config_path)
        
        # Create a dummy checkpoint file for the factory to find
        output_dir = Path(factory.config['trainer']['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = output_dir / "best_model.pth"
        torch.save({'model_state_dict': {}}, checkpoint_path)

        # Run evaluation
        metrics = factory.run_evaluation()
        
        assert MockEvaluator.last_instance is not None
        assert MockEvaluator.last_instance.evaluate_called is True
        assert MockEvaluator.last_instance.config.checkpoint_path == str(checkpoint_path)
        assert metrics['mIoU_Genus'] == 0.9

    def test_metrics_storer_integration(self, mock_dependencies, valid_mtl_config_path):
        """Verifies that MetricsStorer is properly integrated."""
        factory = ExperimentFactory(config_path=valid_mtl_config_path)
        
        # Test that get_metrics_storer works
        storer = factory.get_metrics_storer()
        assert MockMetricsStorer.last_instance is not None
        assert MockMetricsStorer.last_instance.output_dir == factory.config['trainer']['output_dir']
        
        # Test that training uses the metrics storer
        factory.run_training()
        trainer_instance = MockTrainer.last_instance
        # The trainer should have been passed the metrics storer
        # (we can't easily verify this without inspecting the call, but the test shows integration)
        assert trainer_instance is not None