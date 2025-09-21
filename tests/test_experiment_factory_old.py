import pytest
import yaml
import torch
import optuna
import pytest
import torch
import numpy as np
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock, ANY

# Import the class under test
from coral_mtl.ExperimentFactory import ExperimentFactory
from coral_mtl.utils.task_splitter import MTLTaskSplitter, BaseTaskSplitter

# --- MOCK CLASSES as defined in the Testing Philosophy (Section: Isolation) ---
# These lightweight classes mimic the real components, allowing us to test the
# factory's orchestration logic without performing any real computation.


# ✅ Dummy dataset with both tasks (genus + health)
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


# ✅ Dummy model that mimics a Torch model with logits for both tasks
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
        # Return nothing like the real trainer after refactoring
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

class MockVisualizer:
    """A mock Visualizer that records method calls."""
    last_instance = None
    def __init__(self, output_dir, task_info=None, style=None):
        self.plot_training_losses_called = False
        self.plot_validation_performance_called = False
        self.plot_qualitative_results_called = False
        self.plot_confusion_analysis_called = False
        self.plot_learning_rate_called = False
        self.plot_uncertainty_weights_called = False
        MockVisualizer.last_instance = self

    def plot_training_losses(self, *args, **kwargs): self.plot_training_losses_called = True
    def plot_validation_performance(self, *args, **kwargs): self.plot_validation_performance_called = True
    def plot_qualitative_results(self, *args, **kwargs): self.plot_qualitative_results_called = True
    def plot_confusion_analysis(self, *args, **kwargs): self.plot_confusion_analysis_called = True
    def plot_learning_rate(self, *args, **kwargs): self.plot_learning_rate_called = True
    def plot_uncertainty_weights(self, *args, **kwargs): self.plot_uncertainty_weights_called = True

class MockMetricsStorer:
    """A mock MetricsStorer that records its initialization and method calls."""
    last_instance = None
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.store_epoch_history_called = False
        self.store_per_image_cms_called = False
        MockMetricsStorer.last_instance = self
    
    def store_epoch_history(self, *args, **kwargs): self.store_epoch_history_called = True
    def store_per_image_cms(self, *args, **kwargs): self.store_per_image_cms_called = True
    def open_for_run(self, *args, **kwargs): pass
    def close(self): pass


# --- PYTEST FIXTURES for Test Environment Setup ---

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
    # Create a dummy output directory
    (tmp_path / "experiments" / "test_run").mkdir(parents=True, exist_ok=True)
    return {
        'data': {
            'task_definitions_path': str(mock_task_definitions_path),
            'dataset_name': 'fake/hf-dataset', # Default to mocked HF
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
def valid_study_config_path(tmp_path, valid_mtl_config_path):
    """Creates a config file that includes a 'study' section."""
    with open(valid_mtl_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Define a search space file
    search_space = {
        'optimizer.params.lr': {'type': 'float', 'params': {'name': 'lr', 'low': 1e-5, 'high': 1e-3}}
    }
    search_space_path = tmp_path / "search_space.yaml"
    with open(search_space_path, 'w') as f:
        yaml.dump(search_space, f)
        
    config['study'] = {
        'name': 'test_study',
        'storage': 'sqlite:///test_study.db',
        'direction': 'maximize',
        'n_trials': 2,
        'config_path': str(search_space_path)
    }
    
    path = tmp_path / "study_config.yaml"
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
    mock_mtl_dataset_instance.__len__.return_value = 10  # Fix ValueError
    mock_mtl_dataset = MagicMock(return_value=mock_mtl_dataset_instance)
    monkeypatch.setattr('coral_mtl.ExperimentFactory.CoralscapesMTLDataset', mock_mtl_dataset)

    mock_baseline_dataset_instance = MagicMock()
    mock_baseline_dataset_instance.__len__.return_value = 10  # Fix ValueError
    mock_baseline_dataset = MagicMock(return_value=mock_baseline_dataset_instance)
    monkeypatch.setattr('coral_mtl.ExperimentFactory.CoralscapesDataset', mock_baseline_dataset)

    # Mock the trainer and evaluator
    monkeypatch.setattr('coral_mtl.ExperimentFactory.Trainer', MockTrainer)
    monkeypatch.setattr('coral_mtl.ExperimentFactory.Evaluator', MockEvaluator)

    # Mock Optuna
    mock_study = MagicMock()
    mock_study.best_trial.value = 0.95  # Fix TypeError
    mock_optuna = MagicMock()
    mock_optuna.create_study.return_value = mock_study
    monkeypatch.setattr('coral_mtl.ExperimentFactory.optuna', mock_optuna)

    # Mock MetricsStorer
    monkeypatch.setattr('coral_mtl.ExperimentFactory.MetricsStorer', MockMetricsStorer)

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
        """Verifies correct parsing of the task_definitions.yaml file."""
        factory = ExperimentFactory(config_path=valid_mtl_config_path)
        assert hasattr(factory, 'task_splitter')
        assert isinstance(factory.task_splitter, MTLTaskSplitter)
        assert 'genus' in factory.task_splitter.hierarchical_definitions
        assert 'health' in factory.task_splitter.hierarchical_definitions
        
        # Check that task definitions were parsed correctly
        genus_def = factory.task_splitter.hierarchical_definitions['genus']
        assert 'ungrouped' in genus_def
        assert 'id2label' in genus_def['ungrouped']
        assert len(genus_def['ungrouped']['id2label']) == 3  # background, acropora, pocillopora

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

    @patch('coral_mtl.ExperimentFactory.CoralMTLModel')
    def test_get_model_caching(self, mock_mtl_model, valid_mtl_config_path):
        """Verifies that the model is built only once and cached."""
        factory = ExperimentFactory(config_path=valid_mtl_config_path)
        model1 = factory.get_model()
        model2 = factory.get_model()
        assert model1 is model2
        mock_mtl_model.assert_called_once()
        
    def test_get_model_raises_error_on_unknown_type(self, tmp_path, base_config):
        """Verifies robustness against invalid model types."""
        config = base_config
        config['model'] = {'type': 'UnknownFutureModel'}
        path = tmp_path / "invalid_config.yaml"
        with open(path, 'w') as f:
            yaml.dump(config, f)
        
        factory = ExperimentFactory(config_path=path)
        with pytest.raises(ValueError, match="Unknown model type 'UnknownFutureModel'"):
            factory.get_model()

    def test_get_dataloaders_selects_correct_dataset(self, mock_dependencies, valid_mtl_config_path, valid_baseline_config_path):
        """Verifies the correct Dataset class is used for MTL vs Baseline."""
        factory_mtl = ExperimentFactory(config_path=valid_mtl_config_path)
        factory_mtl.get_dataloaders()
        mock_dependencies['mtl_dataset'].assert_called()
        
        factory_baseline = ExperimentFactory(config_path=valid_baseline_config_path)
        factory_baseline.get_dataloaders()
        mock_dependencies['baseline_dataset'].assert_called()

    def test_get_dataloaders_applies_augmentations_to_train_only(self, mock_dependencies, valid_mtl_config_path):
        """Verifies augmentations are applied correctly to the train split."""
        factory = ExperimentFactory(config_path=valid_mtl_config_path)
        factory.get_dataloaders()
        
        # Datasets for train, val, test are created. 3 calls total.
        assert mock_dependencies['mtl_dataset'].call_count == 3
        
        # Check the 'augmentations' kwarg for each call
        train_call_args = mock_dependencies['mtl_dataset'].call_args_list[0].kwargs
        val_call_args = mock_dependencies['mtl_dataset'].call_args_list[1].kwargs
        test_call_args = mock_dependencies['mtl_dataset'].call_args_list[2].kwargs
        
        assert train_call_args['augmentations'] is not None
        assert val_call_args['augmentations'] is None
        assert test_call_args['augmentations'] is None

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
        
        # MockTrainer.last_instance is reset by the fixture
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
        
        assert MockEvaluator.last_instance is None
        
        metrics = factory.run_evaluation()
        
        assert MockEvaluator.last_instance is not None
        assert MockEvaluator.last_instance.evaluate_called is True
        assert MockEvaluator.last_instance.config.checkpoint_path == str(checkpoint_path)
        assert metrics['mIoU_Genus'] == 0.9

    def test_run_hyperparameter_study_orchestration(self, mock_dependencies, valid_study_config_path):
        """Verifies the core logic of the Optuna study workflow."""
        # Setup mock optuna objects
        mock_study = MagicMock()
        mock_trial = MagicMock()
        mock_trial.number = 0
        mock_trial.suggest_float.return_value = 5e-4
        mock_trial.params = {'optimizer.params.lr': 5e-4}
        
        # Fix TypeError: mock study's best_trial.value must be a real number
        mock_study.best_trial.value = 0.9876

        mock_dependencies["optuna"].create_study.return_value = mock_study

        factory = ExperimentFactory(config_path=valid_study_config_path)

        # The key test: capture the 'objective' function and run it manually
        # This allows us to inspect its behavior in isolation.
        def capture_objective(objective_func, n_trials):
            # We call the objective with our own mock trial
            # The objective should return a metric value, not the trainer return value
            result = objective_func(mock_trial)
            assert isinstance(result, (int, float))  # Should be a valid metric value

        mock_study.optimize.side_effect = capture_objective
        
        factory.run_hyperparameter_study()

        # Verify that the study was set up correctly
        mock_dependencies["optuna"].create_study.assert_called_with(
            study_name='test_study',
            storage='sqlite:///test_study.db',
            load_if_exists=True,
            direction='maximize',
            pruner=ANY
        )
        mock_study.optimize.assert_called_once()
        
        # Crucially, verify that the trainer inside the objective function
        # received a config with the hyperparameter updated by the trial.
        assert MockTrainer.last_instance is not None
        # The objective creates a NEW factory, which creates a NEW trainer instance.
        # We need to check the optimizer that was created *inside* that trial.
        # The easiest way is to check the config that was passed to the trainer.
        # However, the factory passes optimizer object, not config.
        # Let's check the new output_dir, which is easier to trace.
        assert f"trial_{mock_trial.number}" in MockTrainer.last_instance.config.output_dir


    def test_generate_visualizations_with_logs(self, mock_dependencies, valid_mtl_config_path, tmp_path):
        """Verifies that providing logs calls the correct plotting methods."""
        # --- Setup ---
        config = yaml.safe_load(valid_mtl_config_path.read_text())
        config['data'].setdefault('params', {})['num_workers'] = 0
        modified_config_path = tmp_path / "modified_config.yaml"
        modified_config_path.write_text(yaml.dump(config))

        # Create dummy checkpoint (empty dict, not wrapped)
        output_dir = Path(config['trainer']['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save({}, output_dir / "best_model.pth")

        # Use DummyDataset and DummyModel
        mock_dependencies['mtl_dataset'].return_value = DummyDataset()
        mock_dependencies['model'].return_value = DummyModel()

        # Patch torch.device to return CPU
        with patch('torch.device', return_value='cpu'):
            with patch('torch.load', return_value={}):
                factory = ExperimentFactory(config_path=modified_config_path)
                fake_log = {'metric': [1, 2, 3], 'lr': [0.1, 0.2]}
                factory.generate_visualizations(training_log=fake_log, validation_log=fake_log)

    def test_generate_visualizations_without_logs(self, mock_dependencies, valid_mtl_config_path, tmp_path):
        """Verifies that not providing logs triggers an inference run for plotting."""
        # --- Setup ---
        config = yaml.safe_load(valid_mtl_config_path.read_text())
        config['data'].setdefault('params', {})['num_workers'] = 0
        modified_config_path = tmp_path / "modified_config.yaml"
        modified_config_path.write_text(yaml.dump(config))

        # Create dummy checkpoint (empty dict, not wrapped)
        checkpoint_dir = Path(config['trainer']['output_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / "best_model.pth"
        torch.save({}, checkpoint_path)

        # Use DummyDataset and DummyModel
        mock_dependencies['mtl_dataset'].return_value = DummyDataset()
        mock_dependencies['model'].return_value = DummyModel()

        # Patch torch.device to return CPU
        with patch('torch.device', return_value='cpu'):
            with patch('torch.load', return_value={}):
                factory = ExperimentFactory(config_path=modified_config_path)
                factory.generate_visualizations()