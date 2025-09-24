"""Shared fixtures for the test suite."""
import pytest
import yaml
import numpy as np
import torch
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock
from coral_mtl.utils.task_splitter import MTLTaskSplitter, BaseTaskSplitter


@pytest.fixture(autouse=True)
def deterministic_seeds():
    """Set deterministic seeds for all random number generators."""
    torch.manual_seed(42)
    np.random.seed(42)
    import random
    random.seed(42)
    # Set deterministic behavior for CUDA if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@pytest.fixture
def device():
    """Get appropriate device for testing (CPU unless marked as GPU test)."""
    return torch.device('cpu')


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary directory for test outputs."""
    output_dir = tmp_path / "test_outputs"
    output_dir.mkdir(exist_ok=True)
    return str(output_dir)


@pytest.fixture
def real_task_definitions():
    """Load the real task definitions from configs/task_definitions.yaml."""
    config_path = Path(__file__).parent.parent / "configs" / "task_definitions.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


@pytest.fixture
def dummy_task_definitions():
    """Create minimal task definitions for testing."""
    return {
        'genus': {
            'id2label': {0: 'unlabeled', 1: 'coral_a', 2: 'coral_b'},
            'groupby': {
                'mapping': {0: 0, 1: 1, 2: 1},  # Group corals together
                'id2label': {0: 'unlabeled', 1: 'coral'}
            }
        },
        'health': {
            'id2label': {0: 'unlabeled', 1: 'healthy', 2: 'bleached'},
            'groupby': None
        }
    }


@pytest.fixture  
def dummy_masks(dummy_task_definitions):
    """Create dummy mask data for testing."""
    # Create simple 32x32 masks with some basic patterns
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[10:20, 10:20] = 1  # coral_a region
    mask[5:15, 20:30] = 2   # coral_b region
    
    return {
        'genus': mask,
        'health': mask.copy()  # Same pattern for health
    }


@pytest.fixture
def mtl_task_splitter(real_task_definitions):
    """Create a real MTLTaskSplitter instance."""
    return MTLTaskSplitter(real_task_definitions)


@pytest.fixture
def base_task_splitter(real_task_definitions):
    """Create a real BaseTaskSplitter instance."""
    return BaseTaskSplitter(real_task_definitions)


@pytest.fixture
def mock_metrics_storer(temp_output_dir):
    """Create a mocked MetricsStorer instance."""
    mock_storer = Mock()
    mock_storer.output_dir = temp_output_dir
    mock_storer.store_confusion_matrices = Mock()
    mock_storer.store_per_image_data = Mock()
    mock_storer.finalize_epoch = Mock()
    return mock_storer


@pytest.fixture
def synthetic_batch():
    """Create synthetic batch data for testing."""
    batch_size, height, width = 2, 64, 64
    num_classes = 4
    
    # Create synthetic images
    images = torch.randn(batch_size, 3, height, width)
    
    # Create synthetic masks
    masks = torch.randint(0, num_classes, (batch_size, height, width))
    
    # Create synthetic predictions (argmax of logits)
    predictions = torch.randint(0, num_classes, (batch_size, height, width))
    
    # Create synthetic logits
    logits = torch.randn(batch_size, num_classes, height, width)
    
    image_ids = [f"test_image_{i}" for i in range(batch_size)]
    
    return {
        'images': images,
        'masks': masks,
        'predictions': predictions,
        'logits': logits,
        'image_ids': image_ids
    }


@pytest.fixture
def config_mtl_minimal():
    """Minimal MTL configuration for testing."""
    return {
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
            'num_workers': 0
        },
        'trainer': {
            'epochs': 2,
            'learning_rate': 0.001,
            'device': 'cpu'
        },
        'metrics_processor': {
            'enabled': False  # Disable for unit tests
        }
    }


@pytest.fixture
def config_baseline_minimal():
    """Minimal baseline configuration for testing.""" 
    return {
        'model': {
            'type': 'BaselineSegformer',
            'encoder_name': 'nvidia/mit-b0',
            'decoder_channels': 128,
            'num_classes': 10
        },
        'data': {
            'patch_size': 64,
            'batch_size': 2,
            'num_workers': 0
        },
        'trainer': {
            'epochs': 2,
            'learning_rate': 0.001,
            'device': 'cpu'
        }
    }


# Legacy mock fixtures for backward compatibility
@pytest.fixture
def mock_mtl_task_splitter(real_task_definitions):
    """Create a properly mocked MTLTaskSplitter with real task definitions."""
    mock_splitter = MagicMock(spec=MTLTaskSplitter)
    
    # Initialize the real splitter to get proper attributes
    real_splitter = MTLTaskSplitter(real_task_definitions)
    
    # Copy important attributes from real splitter to mock
    mock_splitter.raw_definitions = real_splitter.raw_definitions
    mock_splitter.hierarchical_definitions = real_splitter.hierarchical_definitions
    mock_splitter.max_original_id = real_splitter.max_original_id
    mock_splitter.global_mapping = real_splitter.global_mapping
    mock_splitter.flat_mapping_array = real_splitter.flat_mapping_array
    mock_splitter.flat_id2label = real_splitter.flat_id2label
    
    return mock_splitter


@pytest.fixture
def mock_base_task_splitter(real_task_definitions):
    """Create a properly mocked BaseTaskSplitter with real task definitions."""
    mock_splitter = MagicMock(spec=BaseTaskSplitter)
    
    # Initialize the real splitter to get proper attributes
    real_splitter = BaseTaskSplitter(real_task_definitions)
    
    # Copy important attributes from real splitter to mock
    mock_splitter.raw_definitions = real_splitter.raw_definitions
    mock_splitter.hierarchical_definitions = real_splitter.hierarchical_definitions
    mock_splitter.max_original_id = real_splitter.max_original_id
    mock_splitter.flat_mapping_array = real_splitter.flat_mapping_array
    mock_splitter.flat_id2label = real_splitter.flat_id2label
    
    return mock_splitter