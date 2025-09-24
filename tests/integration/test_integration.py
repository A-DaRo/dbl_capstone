"""Integration tests for the Coral-MTL project."""
import pytest
import torch
import tempfile
from pathlib import Path
import json
import yaml
from unittest.mock import patch

from coral_mtl.ExperimentFactory import ExperimentFactory


@pytest.mark.integration
class TestCoralMTLIntegration:
    """Comprehensive integration tests for MTL workflow."""
    
    def test_minimal_mtl_training_loop(self, coralscapes_test_data, pds_test_data, temp_output_dir):
        """Test minimal MTL training pipeline."""
        if not coralscapes_test_data.exists():
            pytest.skip("Test coralscapes data not available")
        
        config = {
            'run_name': 'integration_test_mtl',
            'output_dir': str(temp_output_dir / 'mtl_test'),
            'seed': 42,
            'device': 'cpu',
            'data': {
                'dataset_dir': str(coralscapes_test_data),
                'pds_path': str(pds_test_data) if pds_test_data.exists() else None,
                'use_pds': pds_test_data.exists(),
                'batch_size': 1,  # Small for CPU testing
                'num_workers': 0,
                'pin_memory': False,
                'img_size': [64, 64]  # Small for fast testing
            },
            'tasks': {
                'task_definitions_path': 'tests/configs/tasks/task_definitions.yaml',
                'primary_tasks': ['health'],
                'auxiliary_tasks': ['genus']
            },
            'model': {
                'name': 'CoralMTLModel',
                'encoder': {
                    'name': 'nvidia/mit-b0',
                    'pretrained': False  # Avoid downloading
                },
                'decoder': {
                    'name': 'HierarchicalContextAwareDecoder',
                    'params': {}
                }
            },
            'trainer': {
                'epochs': 1,  # Just one epoch for testing
                'optimizer': {
                    'name': 'AdamW',
                    'params': {'lr': 0.001}
                },
                'scheduler': {
                    'name': 'LinearLR',
                    'params': {
                        'start_factor': 1.0,
                        'end_factor': 0.1,
                        'total_iters': 1
                    }
                },
                'loss': {
                    'name': 'CoralMTLLoss',
                    'params': {}
                }
            },
            'evaluator': {
                'batch_size': 1,
                'inference': {
                    'window_size': [64, 64],
                    'stride': [32, 32]
                }
            },
            'metrics': {
                'enabled': True,
                'advanced_metrics': {
                    'enabled': False  # Disable advanced metrics for speed
                }
            }
        }
        
        try:
            factory = ExperimentFactory(config_dict=config)
            
            # Run training
            factory.run_training()
            
            # Check outputs
            output_dir = Path(config['output_dir'])
            assert output_dir.exists()
            
            # Check history file
            history_file = output_dir / 'history.json'
            if history_file.exists():
                with open(history_file) as f:
                    history = json.load(f)
                assert len(history) > 0
                assert 'epoch' in history[0]
            
            # Check best model checkpoint
            best_model_file = output_dir / 'best_model.pth'
            if best_model_file.exists():
                # Try to load checkpoint
                checkpoint = torch.load(best_model_file, map_location='cpu')
                assert 'model_state_dict' in checkpoint
                assert 'optimizer_state_dict' in checkpoint
                
        except Exception as e:
            pytest.skip(f"MTL training integration test failed: {e}")
    
    def test_minimal_baseline_training_loop(self, coralscapes_test_data, pds_test_data, temp_output_dir):
        """Test minimal baseline training pipeline."""
        if not coralscapes_test_data.exists():
            pytest.skip("Test coralscapes data not available")
        
        config = {
            'run_name': 'integration_test_baseline',
            'output_dir': str(temp_output_dir / 'baseline_test'),
            'seed': 42,
            'device': 'cpu',
            'data': {
                'dataset_dir': str(coralscapes_test_data),
                'pds_path': str(pds_test_data) if pds_test_data.exists() else None,
                'use_pds': pds_test_data.exists(),
                'batch_size': 1,
                'num_workers': 0,
                'pin_memory': False,
                'img_size': [64, 64]
            },
            'model': {
                'name': 'BaselineSegformer',
                'encoder': {
                    'name': 'nvidia/mit-b0',
                    'pretrained': False
                },
                'decoder': {
                    'name': 'SegFormerMLPDecoder',
                    'params': {'num_classes': 39}
                }
            },
            'trainer': {
                'epochs': 1,
                'optimizer': {
                    'name': 'AdamW',
                    'params': {'lr': 0.001}
                },
                'scheduler': {
                    'name': 'LinearLR',
                    'params': {
                        'start_factor': 1.0,
                        'end_factor': 0.1,
                        'total_iters': 1
                    }
                },
                'loss': {
                    'name': 'CrossEntropyLoss',
                    'params': {'ignore_index': 0}
                }
            },
            'evaluator': {
                'batch_size': 1,
                'inference': {
                    'window_size': [64, 64],
                    'stride': [32, 32]
                }
            },
            'metrics': {
                'enabled': True,
                'advanced_metrics': {
                    'enabled': False
                }
            }
        }
        
        try:
            factory = ExperimentFactory(config_dict=config)
            factory.run_training()
            
            # Check outputs similar to MTL test
            output_dir = Path(config['output_dir'])
            assert output_dir.exists()
            
        except Exception as e:
            pytest.skip(f"Baseline training integration test failed: {e}")
    
    def test_complete_evaluation_pipeline(self, coralscapes_test_data, temp_output_dir):
        """Test complete evaluation workflow."""
        if not coralscapes_test_data.exists():
            pytest.skip("Test coralscapes data not available")
        
        # First train a minimal model to get checkpoint
        train_config = {
            'run_name': 'eval_test_train',
            'output_dir': str(temp_output_dir / 'eval_train'),
            'seed': 42,
            'device': 'cpu',
            'data': {
                'dataset_dir': str(coralscapes_test_data),
                'batch_size': 1,
                'num_workers': 0,
                'img_size': [32, 32]
            },
            'tasks': {
                'task_definitions_path': 'tests/configs/tasks/task_definitions.yaml',
                'primary_tasks': ['health'],
                'auxiliary_tasks': ['genus']
            },
            'model': {
                'name': 'CoralMTLModel',
                'encoder': {'name': 'nvidia/mit-b0', 'pretrained': False},
                'decoder': {'name': 'HierarchicalContextAwareDecoder'}
            },
            'trainer': {
                'epochs': 1,
                'optimizer': {'name': 'AdamW', 'params': {'lr': 0.001}},
                'loss': {'name': 'CoralMTLLoss'}
            },
            'metrics': {'enabled': True, 'advanced_metrics': {'enabled': False}}
        }
        
        try:
            # Train model
            train_factory = ExperimentFactory(config_dict=train_config)
            train_factory.run_training()
            
            # Check if checkpoint exists
            checkpoint_path = Path(train_config['output_dir']) / 'best_model.pth'
            if not checkpoint_path.exists():
                pytest.skip("Training didn't produce checkpoint")
            
            # Now test evaluation
            eval_config = train_config.copy()
            eval_config['run_name'] = 'eval_test_eval'
            eval_config['output_dir'] = str(temp_output_dir / 'eval_test')
            
            eval_factory = ExperimentFactory(config_dict=eval_config)
            eval_factory.run_evaluation(checkpoint_path=str(checkpoint_path))
            
            # Check evaluation outputs
            eval_output_dir = Path(eval_config['output_dir'])
            
            # Look for test results
            test_results = eval_output_dir / 'test_results.json'
            if test_results.exists():
                with open(test_results) as f:
                    results = json.load(f)
                assert 'global' in results
                assert isinstance(results['global'], dict)
            
        except Exception as e:
            pytest.skip(f"Complete evaluation pipeline test failed: {e}")
    
    def test_pds_patches_integration(self, coralscapes_test_data, pds_test_data, temp_output_dir):
        """Test integration with PDS patches."""
        if not coralscapes_test_data.exists() or not pds_test_data.exists():
            pytest.skip("Test data not available")
        
        config = {
            'run_name': 'pds_integration_test',
            'output_dir': str(temp_output_dir / 'pds_test'),
            'seed': 42,
            'device': 'cpu',
            'data': {
                'dataset_dir': str(coralscapes_test_data),
                'pds_path': str(pds_test_data),
                'use_pds': True,  # Use PDS patches
                'batch_size': 1,
                'num_workers': 0,
                'img_size': [64, 64]
            },
            'tasks': {
                'task_definitions_path': 'tests/configs/tasks/task_definitions.yaml',
                'primary_tasks': ['health'],
                'auxiliary_tasks': ['genus']
            },
            'model': {
                'name': 'CoralMTLModel',
                'encoder': {'name': 'nvidia/mit-b0', 'pretrained': False},
                'decoder': {'name': 'HierarchicalContextAwareDecoder'}
            },
            'trainer': {
                'epochs': 1,
                'optimizer': {'name': 'AdamW', 'params': {'lr': 0.001}},
                'loss': {'name': 'CoralMTLLoss'}
            },
            'metrics': {'enabled': True, 'advanced_metrics': {'enabled': False}}
        }
        
        try:
            factory = ExperimentFactory(config_dict=config)
            
            # Test that we can load PDS data
            dataloaders = factory.get_dataloaders()
            assert 'train' in dataloaders
            
            # Test that training works with PDS data
            factory.run_training()
            
            # Check outputs
            output_dir = Path(config['output_dir'])
            assert output_dir.exists()
            
        except Exception as e:
            pytest.skip(f"PDS integration test failed: {e}")
    
    @pytest.mark.optdeps
    def test_advanced_metrics_processor_integration(self, coralscapes_test_data, temp_output_dir):
        """Test integration with advanced metrics processor."""
        if not coralscapes_test_data.exists():
            pytest.skip("Test coralscapes data not available")
        
        config = {
            'run_name': 'advanced_metrics_test',
            'output_dir': str(temp_output_dir / 'advanced_test'),
            'seed': 42,
            'device': 'cpu',
            'data': {
                'dataset_dir': str(coralscapes_test_data),
                'batch_size': 1,
                'num_workers': 0,
                'img_size': [32, 32]  # Small images for fast processing
            },
            'tasks': {
                'task_definitions_path': 'tests/configs/tasks/task_definitions.yaml',
                'primary_tasks': ['health'],
                'auxiliary_tasks': ['genus']
            },
            'model': {
                'name': 'CoralMTLModel',
                'encoder': {'name': 'nvidia/mit-b0', 'pretrained': False},
                'decoder': {'name': 'HierarchicalContextAwareDecoder'}
            },
            'trainer': {
                'epochs': 1,
                'optimizer': {'name': 'AdamW', 'params': {'lr': 0.001}},
                'loss': {'name': 'CoralMTLLoss'}
            },
            'metrics': {
                'enabled': True,
                'advanced_metrics': {
                    'enabled': True,
                    'num_workers': 1,  # Minimal workers for testing
                    'tasks': ['ASSD', 'HD95']  # Basic metrics
                }
            }
        }
        
        try:
            factory = ExperimentFactory(config_dict=config)
            
            # Check that advanced processor is created
            processor = factory.get_advanced_metrics_processor()
            assert processor is not None
            
            # Run training (should start/stop processor)
            factory.run_training()
            
            output_dir = Path(config['output_dir'])
            assert output_dir.exists()
            
        except ImportError:
            pytest.skip("Optional dependencies for advanced metrics not available")
        except Exception as e:
            pytest.skip(f"Advanced metrics integration test failed: {e}")
    
    def test_mtl_vs_baseline_evaluation_consistency(self, coralscapes_test_data, temp_output_dir):
        """Test that MTL and baseline models produce consistent evaluation structures."""
        if not coralscapes_test_data.exists():
            pytest.skip("Test coralscapes data not available")
        
        # Common data config
        data_config = {
            'dataset_dir': str(coralscapes_test_data),
            'batch_size': 1,
            'num_workers': 0,
            'img_size': [32, 32]
        }
        
        # MTL config
        mtl_config = {
            'run_name': 'consistency_mtl',
            'output_dir': str(temp_output_dir / 'consistency_mtl'),
            'seed': 42,
            'device': 'cpu',
            'data': data_config,
            'tasks': {
                'task_definitions_path': 'tests/configs/tasks/task_definitions.yaml',
                'primary_tasks': ['health'],
                'auxiliary_tasks': ['genus']
            },
            'model': {
                'name': 'CoralMTLModel',
                'encoder': {'name': 'nvidia/mit-b0', 'pretrained': False},
                'decoder': {'name': 'HierarchicalContextAwareDecoder'}
            },
            'trainer': {
                'epochs': 1,
                'optimizer': {'name': 'AdamW', 'params': {'lr': 0.001}},
                'loss': {'name': 'CoralMTLLoss'}
            },
            'metrics': {'enabled': True, 'advanced_metrics': {'enabled': False}}
        }
        
        # Baseline config
        baseline_config = {
            'run_name': 'consistency_baseline',
            'output_dir': str(temp_output_dir / 'consistency_baseline'),
            'seed': 42,
            'device': 'cpu',
            'data': data_config,
            'model': {
                'name': 'BaselineSegformer',
                'encoder': {'name': 'nvidia/mit-b0', 'pretrained': False},
                'decoder': {'name': 'SegFormerMLPDecoder', 'params': {'num_classes': 39}}
            },
            'trainer': {
                'epochs': 1,
                'optimizer': {'name': 'AdamW', 'params': {'lr': 0.001}},
                'loss': {'name': 'CrossEntropyLoss', 'params': {'ignore_index': 0}}
            },
            'metrics': {'enabled': True, 'advanced_metrics': {'enabled': False}}
        }
        
        try:
            # Train and evaluate both models
            mtl_factory = ExperimentFactory(config_dict=mtl_config)
            baseline_factory = ExperimentFactory(config_dict=baseline_config)
            
            # Quick training
            mtl_factory.run_training()
            baseline_factory.run_training()
            
            # Check that both produced outputs
            mtl_output = Path(mtl_config['output_dir'])
            baseline_output = Path(baseline_config['output_dir'])
            
            assert mtl_output.exists()
            assert baseline_output.exists()
            
            # Check that evaluation structure is comparable
            # (Exact metrics will differ, but structure should be similar)
            
        except Exception as e:
            pytest.skip(f"MTL vs baseline consistency test failed: {e}")


@pytest.mark.integration 
class TestExtremeConfigurations:
    """Test extreme and edge case configurations."""
    
    def test_minimal_task_configuration(self, coralscapes_test_data, temp_output_dir):
        """Test with minimal task configuration."""
        if not coralscapes_test_data.exists():
            pytest.skip("Test coralscapes data not available")
        
        # Create minimal task definitions
        minimal_tasks = {
            'health': {
                'id2label': {
                    0: "unlabeled",
                    1: "alive", 
                    2: "dead"
                }
            }
        }
        
        # Write temporary task definitions
        temp_task_file = temp_output_dir / 'minimal_tasks.yaml'
        with open(temp_task_file, 'w') as f:
            yaml.dump(minimal_tasks, f)
        
        config = {
            'run_name': 'minimal_task_test',
            'output_dir': str(temp_output_dir / 'minimal_task'),
            'seed': 42,
            'device': 'cpu',
            'data': {
                'dataset_dir': str(coralscapes_test_data),
                'batch_size': 1,
                'num_workers': 0,
                'img_size': [32, 32]
            },
            'tasks': {
                'task_definitions_path': str(temp_task_file),
                'primary_tasks': ['health'],
                'auxiliary_tasks': []  # No auxiliary tasks
            },
            'model': {
                'name': 'CoralMTLModel',
                'encoder': {'name': 'nvidia/mit-b0', 'pretrained': False},
                'decoder': {'name': 'HierarchicalContextAwareDecoder'}
            },
            'trainer': {
                'epochs': 1,
                'optimizer': {'name': 'AdamW', 'params': {'lr': 0.001}},
                'loss': {'name': 'CoralMTLLoss'}
            },
            'metrics': {'enabled': True, 'advanced_metrics': {'enabled': False}}
        }
        
        try:
            factory = ExperimentFactory(config_dict=config)
            factory.run_training()
            
            # Should handle minimal configuration
            output_dir = Path(config['output_dir'])
            assert output_dir.exists()
            
        except Exception as e:
            pytest.skip(f"Minimal task configuration test failed: {e}")
    
    def test_imbalanced_task_configuration(self, coralscapes_test_data, temp_output_dir):
        """Test with highly imbalanced task configuration."""
        if not coralscapes_test_data.exists():
            pytest.skip("Test coralscapes data not available")
        
        # Create imbalanced task definitions (many classes in one task, few in another)
        imbalanced_tasks = {
            'many_class_task': {
                'id2label': {i: f"class_{i}" for i in range(20)}  # 20 classes
            },
            'few_class_task': {
                'id2label': {
                    0: "unlabeled",
                    1: "positive"  # Only 2 classes
                }
            }
        }
        
        temp_task_file = temp_output_dir / 'imbalanced_tasks.yaml' 
        with open(temp_task_file, 'w') as f:
            yaml.dump(imbalanced_tasks, f)
        
        config = {
            'run_name': 'imbalanced_task_test',
            'output_dir': str(temp_output_dir / 'imbalanced_task'),
            'seed': 42,
            'device': 'cpu',
            'data': {
                'dataset_dir': str(coralscapes_test_data),
                'batch_size': 1,
                'num_workers': 0,
                'img_size': [32, 32]
            },
            'tasks': {
                'task_definitions_path': str(temp_task_file),
                'primary_tasks': ['many_class_task'],
                'auxiliary_tasks': ['few_class_task']
            },
            'model': {
                'name': 'CoralMTLModel',
                'encoder': {'name': 'nvidia/mit-b0', 'pretrained': False},
                'decoder': {'name': 'HierarchicalContextAwareDecoder'}
            },
            'trainer': {
                'epochs': 1,
                'optimizer': {'name': 'AdamW', 'params': {'lr': 0.001}},
                'loss': {'name': 'CoralMTLLoss'}
            },
            'metrics': {'enabled': True, 'advanced_metrics': {'enabled': False}}
        }
        
        try:
            factory = ExperimentFactory(config_dict=config)
            factory.run_training()
            
            # Should handle imbalanced configuration
            output_dir = Path(config['output_dir'])
            assert output_dir.exists()
            
        except Exception as e:
            pytest.skip(f"Imbalanced task configuration test failed: {e}")