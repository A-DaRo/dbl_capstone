"""Unit tests for ExperimentFactory."""
import pytest
import yaml
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from coral_mtl.ExperimentFactory import ExperimentFactory


class TestExperimentFactory:
    """Test cases for ExperimentFactory."""
    
    def test_init_from_dict(self, factory_config_dict_mtl, temp_output_dir):
        """Test ExperimentFactory initialization from config dict."""
        # Update output dir to use temp dir
        config = factory_config_dict_mtl.copy()
        config['output_dir'] = str(temp_output_dir)
        
        factory = ExperimentFactory(config_dict=config)
        
        assert factory.config is not None
        assert factory.config.run_name == 'test_mtl'
        assert factory.config.device == 'cpu'
        assert factory.config.seed == 42
    
    def test_init_from_yaml(self, temp_output_dir):
        """Test ExperimentFactory initialization from YAML file."""
        config_data = {
            'run_name': 'test_yaml',
            'output_dir': str(temp_output_dir),
            'seed': 42,
            'device': 'cpu',
            'data': {
                'dataset_dir': 'tests/dataset/coralscapes',
                'batch_size': 2,
                'img_size': [32, 32]
            },
            'model': {
                'name': 'CoralMTLModel',
                'encoder': {'name': 'nvidia/mit-b0'}
            },
            'trainer': {
                'epochs': 1,
                'optimizer': {'name': 'AdamW', 'params': {'lr': 0.001}}
            }
        }
        
        # Create temporary YAML file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            yaml_path = f.name
        
        try:
            factory = ExperimentFactory(config_path=yaml_path)
            assert factory.config.run_name == 'test_yaml'
        finally:
            Path(yaml_path).unlink()
    
    @patch('coral_mtl.ExperimentFactory.MTLTaskSplitter')
    def test_get_task_splitter_caching(self, mock_splitter_class, factory_config_dict_mtl, temp_output_dir):
        """Test that task splitter is cached on repeated calls."""
        config = factory_config_dict_mtl.copy()
        config['output_dir'] = str(temp_output_dir)
        
        mock_instance = MagicMock()
        mock_splitter_class.return_value = mock_instance
        
        factory = ExperimentFactory(config_dict=config)
        
        # First call
        splitter1 = factory.get_task_splitter()
        # Second call
        splitter2 = factory.get_task_splitter()
        
        # Should return same instance
        assert splitter1 is splitter2
        # Constructor should only be called once
        mock_splitter_class.assert_called_once()
    
    @patch('coral_mtl.ExperimentFactory.get_dataloaders')
    def test_get_dataloaders_caching(self, mock_get_dataloaders, factory_config_dict_mtl, temp_output_dir):
        """Test that dataloaders are cached."""
        config = factory_config_dict_mtl.copy()
        config['output_dir'] = str(temp_output_dir)
        
        mock_loaders = {'train': MagicMock(), 'val': MagicMock(), 'test': MagicMock()}
        mock_get_dataloaders.return_value = mock_loaders
        
        factory = ExperimentFactory(config_dict=config)
        
        loaders1 = factory.get_dataloaders()
        loaders2 = factory.get_dataloaders()
        
        assert loaders1 is loaders2
        mock_get_dataloaders.assert_called_once()
    
    def test_metrics_processor_disabled(self, factory_config_dict_mtl, temp_output_dir):
        """Test that metrics processor is None when disabled."""
        config = factory_config_dict_mtl.copy()
        config['output_dir'] = str(temp_output_dir)
        config['metrics']['advanced_metrics']['enabled'] = False
        
        factory = ExperimentFactory(config_dict=config)
        processor = factory.get_advanced_metrics_processor()
        
        assert processor is None
    
    def test_metrics_processor_enabled(self, factory_config_dict_mtl, temp_output_dir):
        """Test metrics processor creation when enabled."""
        config = factory_config_dict_mtl.copy()
        config['output_dir'] = str(temp_output_dir)
        config['metrics']['advanced_metrics'] = {
            'enabled': True,
            'num_workers': 2,
            'tasks': ['ASSD', 'HD95']
        }
        
        with patch('coral_mtl.ExperimentFactory.AdvancedMetricsProcessor') as mock_processor:
            mock_instance = MagicMock()
            mock_processor.return_value = mock_instance
            
            factory = ExperimentFactory(config_dict=config)
            processor = factory.get_advanced_metrics_processor()
            
            assert processor is mock_instance
            mock_processor.assert_called_once()
    
    def test_missing_config_raises_error(self):
        """Test that missing config raises appropriate error."""
        with pytest.raises((ValueError, TypeError)):
            ExperimentFactory()
    
    def test_invalid_config_dict_raises_error(self):
        """Test that invalid config dict raises error."""
        invalid_config = {'invalid': 'config'}
        
        with pytest.raises((KeyError, AttributeError)):
            factory = ExperimentFactory(config_dict=invalid_config)
            # Try to access required attributes to trigger error
            _ = factory.config.run_name
    
    def test_config_path_resolution(self, factory_config_dict_mtl, temp_output_dir):
        """Test that relative paths in config are resolved to absolute."""
        config = factory_config_dict_mtl.copy()
        config['output_dir'] = str(temp_output_dir)
        # Use relative path
        config['data']['dataset_dir'] = 'tests/dataset/coralscapes'
        
        factory = ExperimentFactory(config_dict=config)
        
        # The path should be resolved to absolute
        assert Path(factory.config.data.dataset_dir).is_absolute()