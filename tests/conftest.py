"""Shared fixtures for the test suite."""
import pytest
import torch
import numpy as np
import yaml
from pathlib import Path
from unittest.mock import MagicMock, patch
import tempfile
from types import SimpleNamespace

from coral_mtl.utils.task_splitter import MTLTaskSplitter, BaseTaskSplitter


# Fix random seeds for reproducible tests
@pytest.fixture(autouse=True)
def fix_random_seeds():
    """Fix random seeds for all random number generators."""
    torch.manual_seed(42)
    np.random.seed(42)
    if hasattr(torch, 'cuda'):
        torch.cuda.manual_seed_all(42)


@pytest.fixture
def device():
    """Return appropriate device for testing (GPU if available, CPU otherwise)."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


@pytest.fixture
def test_task_definitions():
    """Load the test task definitions from tests/configs/tasks/task_definitions.yaml."""
    config_path = Path(__file__).parent / "configs" / "tasks" / "task_definitions.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


@pytest.fixture
def splitter_mtl(test_task_definitions):
    """Create MTLTaskSplitter with test task definitions."""
    return MTLTaskSplitter(test_task_definitions)


@pytest.fixture
def splitter_base(test_task_definitions):
    """Create BaseTaskSplitter with test task definitions."""
    return BaseTaskSplitter(test_task_definitions)


@pytest.fixture
def dummy_images(device):
    """Create small synthetic images for testing."""
    batch_size = 2
    channels = 3
    height, width = 32, 32
    return torch.rand(batch_size, channels, height, width, device=device, dtype=torch.float32)


@pytest.fixture
def dummy_masks(device, splitter_mtl):
    """Create small synthetic masks for testing."""
    batch_size = 2
    height, width = 32, 32
    # Create masks with valid class IDs from the task splitter
    masks = {}
    for task_name, task_info in splitter_mtl.hierarchical_definitions.items():
        max_id = max(task_info['ungrouped']['id2label'].keys())
        masks[task_name] = torch.randint(0, max_id + 1, (batch_size, height, width), device=device, dtype=torch.long)
    return masks


@pytest.fixture
def dummy_single_mask(device):
    """Create single mask for baseline testing."""
    batch_size = 2
    height, width = 32, 32
    return torch.randint(0, 39, (batch_size, height, width), device=device, dtype=torch.long)


@pytest.fixture
def factory_config_dict_mtl():
    """Minimal valid config dict for MTL ExperimentFactory."""
    return {
        'run_name': 'test_mtl',
        'output_dir': 'tests/outputs/test_mtl',
        'seed': 42,
        'device': 'auto',
        'data': {
            'dataset_dir': 'tests/dataset/coralscapes',
            'pds_path': 'tests/dataset/processed/pds_patches',
            'use_pds': True,
            'batch_size': 2,
            'num_workers': 0,
            'pin_memory': False,
            'img_size': [32, 32]
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
                'pretrained': False
            },
            'decoder': {
                'name': 'HierarchicalContextAwareDecoder',
                'params': {}
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
                'name': 'CoralMTLLoss',
                'params': {}
            }
        },
        'evaluator': {
            'batch_size': 1,
            'inference': {
                'window_size': [32, 32],
                'stride': [16, 16]
            }
        },
        'metrics': {
            'enabled': True,
            'advanced_metrics': {
                'enabled': False
            }
        }
    }


@pytest.fixture
def factory_config_dict_baseline():
    """Minimal valid config dict for baseline ExperimentFactory."""
    return {
        'run_name': 'test_baseline',
        'output_dir': 'tests/outputs/test_baseline',
        'seed': 42,
        'device': 'auto',
        'data': {
            'dataset_dir': 'tests/dataset/coralscapes',
            'pds_path': 'tests/dataset/processed/pds_patches',
            'use_pds': True,
            'batch_size': 2,
            'num_workers': 0,
            'pin_memory': False,
            'img_size': [32, 32]
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
                'window_size': [32, 32],
                'stride': [16, 16]
            }
        },
        'metrics': {
            'enabled': True,
            'advanced_metrics': {
                'enabled': False
            }
        }
    }


@pytest.fixture
def temp_output_dir():
    """Create temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture(scope="session")
def coralscapes_test_data():
    """Path to test coralscapes dataset."""
    return Path(__file__).parent / "dataset" / "coralscapes"


@pytest.fixture(scope="session")
def pds_test_data():
    """Path to test PDS patches dataset."""
    return Path(__file__).parent / "dataset" / "processed" / "pds_patches"


# Model fixtures
@pytest.fixture
def minimal_coral_mtl_model(splitter_mtl):
    """Create minimal CoralMTLModel for testing."""
    from coral_mtl.model.core import CoralMTLModel
    
    return CoralMTLModel(
        encoder_name="nvidia/mit-b0",
        decoder_channel=64,
        num_classes={
            task_name: len(task_info['ungrouped']['id2label'])
            for task_name, task_info in splitter_mtl.hierarchical_definitions.items()
        },
        attention_dim=64
    )


@pytest.fixture
def minimal_baseline_model(splitter_base):
    """Create minimal BaselineSegformer for testing."""
    from coral_mtl.model.core import BaselineSegformer
    
    return BaselineSegformer(
        encoder_name="nvidia/mit-b0",
        decoder_channel=64,
        num_classes=len(splitter_base.global_id2label)
    )


@pytest.fixture
def sample_config_path(tmp_path, factory_config_dict_mtl):
    """Create temporary config file for testing."""
    config_file = tmp_path / "test_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(factory_config_dict_mtl, f)
    return str(config_file)


# Mock fixtures for optional dependencies
@pytest.fixture
def mock_optional_deps():
    """Mock optional dependencies that might not be installed."""
    with patch.dict('sys.modules', {
        'SimpleITK': MagicMock(),
        'panopticapi': MagicMock(),
        'skimage': MagicMock(),
        'sklearn': MagicMock(),
    }):
        yield