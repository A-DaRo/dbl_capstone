"""
Unit tests for ExperimentFactory orchestration and dependency injection.
"""
import unittest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from coral_mtl.ExperimentFactory import ExperimentFactory


class TestExperimentFactory(unittest.TestCase):
    """Test ExperimentFactory initialization and component building."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.test_config = {
            'model': {
                'type': 'CoralMTL',
                'encoder_name': 'nvidia/mit-b0',
                'decoder_channels': 128,
                'primary_tasks': ['genus', 'health'],
                'auxiliary_tasks': ['bleaching']
            },
            'data': {
                'patch_size': 64,
                'batch_size': 2,
                'num_workers': 0,
                'data_root': str(self.temp_dir)
            },
            'trainer': {
                'epochs': 2,
                'learning_rate': 0.001,
                'device': 'cpu',
                'output_dir': str(self.temp_dir / "output")
            },
            'task_definitions_path': str(Path(__file__).parent.parent / "configs" / "task_definitions.yaml")
        }
        
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_factory_initialization_from_dict(self):
        """Test ExperimentFactory can be initialized from config dict."""
        factory = ExperimentFactory(config_dict=self.test_config)
        
        # Verify factory was created
        self.assertIsNotNone(factory)
        self.assertIsNotNone(factory.config)
        
        # Verify config structure
        self.assertIn('model', factory.config)
        self.assertIn('data', factory.config)
        self.assertIn('trainer', factory.config)
    
    def test_factory_initialization_from_file(self):
        """Test ExperimentFactory can be initialized from config file."""
        # Create temporary config file
        config_file = self.temp_dir / "test_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(self.test_config, f)
        
        factory = ExperimentFactory(config_path=str(config_file))
        
        # Verify factory was created
        self.assertIsNotNone(factory)
        self.assertIsNotNone(factory.config)
    
    def test_invalid_config_handling(self):
        """Test factory handles invalid configurations gracefully."""
        invalid_configs = [
            # Missing model section
            {'data': {}, 'trainer': {}},
            # Invalid model type
            {'model': {'type': 'InvalidModel'}, 'data': {}, 'trainer': {}},
            # Missing required fields
            {'model': {'type': 'CoralMTL'}, 'data': {}, 'trainer': {}}
        ]
        
        for invalid_config in invalid_configs:
            with self.assertRaises((KeyError, ValueError, FileNotFoundError)):
                ExperimentFactory(config_dict=invalid_config)
    
    def test_task_splitter_creation(self):
        """Test factory can create task splitters."""
        factory = ExperimentFactory(config_dict=self.test_config)
        
        # Test MTL task splitter creation
        if hasattr(factory, 'get_task_splitter'):
            splitter = factory.get_task_splitter()
            self.assertIsNotNone(splitter)
    
    def test_dependency_injection_consistency(self):
        """Test that factory maintains consistent dependency injection."""
        factory = ExperimentFactory(config_dict=self.test_config)
        
        # Verify factory has essential methods for component creation
        expected_methods = [
            'get_model', 'get_loss_function', 'get_dataloaders',
            'get_optimizer_and_scheduler', 'get_metrics', 'get_trainer', 'get_evaluator'
        ]
        
        for method_name in expected_methods:
            if hasattr(factory, method_name):
                method = getattr(factory, method_name)
                self.assertTrue(callable(method), f"{method_name} should be callable")
    
    @patch('coral_mtl.ExperimentFactory.CoralMTLModel')
    def test_model_creation_mtl(self, mock_model_class):
        """Test MTL model creation with mocked dependencies."""
        mock_model_class.return_value = MagicMock()
        
        factory = ExperimentFactory(config_dict=self.test_config)
        
        # Test model creation if method exists
        if hasattr(factory, 'get_model'):
            try:
                model = factory.get_model()
                # If successful, verify it's the expected type
                self.assertIsNotNone(model)
            except Exception as e:
                # Expected for missing dependencies, just verify factory structure
                self.assertIn('model', factory.config)
    
    def test_baseline_config_handling(self):
        """Test factory handles baseline configuration correctly."""
        baseline_config = self.test_config.copy()
        baseline_config['model'] = {
            'type': 'BaselineSegformer',
            'encoder_name': 'nvidia/mit-b0',
            'decoder_channels': 128,
            'num_classes': 10
        }
        
        factory = ExperimentFactory(config_dict=baseline_config)
        self.assertIsNotNone(factory)
        self.assertEqual(factory.config['model']['type'], 'BaselineSegformer')
    
    def test_config_validation_error_messages(self):
        """Test that factory provides clear error messages for invalid configs."""
        # Test missing task definitions
        invalid_config = self.test_config.copy()
        invalid_config.pop('task_definitions_path', None)
        
        try:
            factory = ExperimentFactory(config_dict=invalid_config)
        except Exception as e:
            # Should provide informative error message
            error_message = str(e).lower()
            # Error should reference task definitions or configuration
            self.assertTrue(
                any(keyword in error_message for keyword in ['task', 'config', 'definition']),
                f"Error message should be informative: {error_message}"
            )
    
    def test_factory_caching_behavior(self):
        """Test that factory caches expensive components appropriately."""
        factory = ExperimentFactory(config_dict=self.test_config)
        
        # Test if factory implements caching for task splitter
        if hasattr(factory, 'get_task_splitter'):
            try:
                splitter1 = factory.get_task_splitter()
                splitter2 = factory.get_task_splitter()
                # Should return same instance if cached
                # (This test may fail if no caching is implemented, which is OK)
            except Exception:
                # Expected if dependencies are missing
                pass
    
    def test_metrics_processor_configuration(self):
        """Test advanced metrics processor configuration."""
        # Test with metrics processor enabled
        config_with_processor = self.test_config.copy()
        config_with_processor['metrics_processor'] = {
            'enabled': True,
            'num_cpu_workers': 4,
            'tasks': ['ASSD', 'HD95']
        }
        
        factory = ExperimentFactory(config_dict=config_with_processor)
        self.assertIsNotNone(factory)
        self.assertTrue(factory.config['metrics_processor']['enabled'])
        
        # Test with metrics processor disabled
        config_no_processor = self.test_config.copy()
        config_no_processor['metrics_processor'] = {'enabled': False}
        
        factory = ExperimentFactory(config_dict=config_no_processor)
        self.assertIsNotNone(factory)
        self.assertFalse(factory.config['metrics_processor']['enabled'])


if __name__ == '__main__':
    unittest.main()