# Create file: tests/coral_mtl/ExperimentFactory/test_factory_initialization.py
import pytest
import yaml
from pathlib import Path

from coral_mtl.ExperimentFactory import ExperimentFactory
from coral_mtl.utils.task_splitter import MTLTaskSplitter, BaseTaskSplitter

class TestFactoryInitialization:
    """Tests for the ExperimentFactory constructor and initial setup."""

    def test_init_success(self, mtl_config_yaml: Path):
        """Verify factory initializes correctly with a valid config path."""
        factory = ExperimentFactory(config_path=str(mtl_config_yaml))
        assert factory is not None
        assert isinstance(factory.config, dict)
        assert factory.task_splitter is not None

    def test_init_with_dict(self, factory_config_dict_mtl: dict):
        """Verify factory initializes correctly with a config dictionary."""
        factory = ExperimentFactory(config_dict=factory_config_dict_mtl)
        assert factory is not None
        assert factory.config == factory_config_dict_mtl

    def test_init_raises_error_with_no_config(self):
        """Verify ValueError is raised if neither a path nor a dict is provided."""
        with pytest.raises(ValueError, match="Either 'config_path' or 'config_dict' must be provided."):
            ExperimentFactory()

    def test_init_raises_error_with_invalid_path(self):
        """Verify FileNotFoundError is raised for a non-existent config file."""
        with pytest.raises(FileNotFoundError):
            ExperimentFactory(config_path="invalid/path/to/config.yaml")

    def test_init_raises_error_with_malformed_yaml(self, tmp_path: Path):
        """Verify YAMLError is raised for a syntactically incorrect config file."""
        malformed_yaml = tmp_path / "malformed.yaml"
        malformed_yaml.write_text("model: { type: 'CoralMTL'") # Unclosed brace
        with pytest.raises(yaml.YAMLError):
            ExperimentFactory(config_path=str(malformed_yaml))

    def test_correct_splitter_type_is_created(self, mtl_config_yaml: Path, baseline_config_yaml: Path):
        """
        Verify that the correct TaskSplitter subclass is instantiated based on the
        model type specified in the configuration. This is a critical test for
        dynamic behavior.
        """
        factory_mtl = ExperimentFactory(config_path=str(mtl_config_yaml))
        assert isinstance(factory_mtl.task_splitter, MTLTaskSplitter)

        factory_baseline = ExperimentFactory(config_path=str(baseline_config_yaml))
        assert isinstance(factory_baseline.task_splitter, BaseTaskSplitter)

    def test_path_resolution(self, tmp_path: Path):
        """
        Verify that relative paths in the configuration are correctly resolved
        to absolute paths.
        """
        relative_config_path = tmp_path / "relative_config.yaml"
        config_content = {
            'data': {
                'task_definitions_path': 'tests/configs/tasks/task_definitions.yaml'
            },
            'trainer': {
                'output_dir': './experiments/my_test_run'
            },
            'model': {'type': 'CoralMTL'}
        }
        with open(relative_config_path, 'w') as f:
            yaml.dump(config_content, f)

        factory = ExperimentFactory(config_path=str(relative_config_path))
        
        resolved_task_path = Path(factory.config['data']['task_definitions_path'])
        assert resolved_task_path.is_absolute()
        assert resolved_task_path.exists()
        
        resolved_output_dir = Path(factory.config['trainer']['output_dir'])
        assert resolved_output_dir.is_absolute()