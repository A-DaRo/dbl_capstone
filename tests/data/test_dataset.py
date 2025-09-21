import torch
import numpy as np
from PIL import Image
import pytest
from datasets import Dataset as HFDataset

from coral_mtl.data.dataset import CoralscapesMTLDataset, CoralscapesDataset
from coral_mtl.data.augmentations import SegmentationAugmentation
from coral_mtl.utils.task_splitter import MTLTaskSplitter, BaseTaskSplitter

@pytest.fixture
def task_definitions():
    """DEPRECATED: Use real_task_definitions from conftest.py instead."""
    # This fixture is kept for backward compatibility but should be replaced
    return {
        'health': {
            'id2label': {0: 'background', 1: 'alive', 2: 'bleached', 3: 'dead'},
            'groupby': {
                'id2label': {0: 'background', 1: 'healthy', 2: 'bleached', 3: 'dead'},
                'mapping': {
                    0: [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
                    1: [17],
                    2: [16], 
                    3: [20]
                }
            }
        },
        'genus': {
            'id2label': {0: 'background', 1: 'acropora_1', 2: 'acropora_2', 3: 'acropora_3', 4: 'acropora_4', 5: 'pocillopora', 6: 'porites', 7: 'montipora', 8: 'favites', 9: 'other'},
            'groupby': {
                'id2label': {0: 'background', 1: 'acropora', 2: 'pocillopora', 3: 'porites', 4: 'montipora', 5: 'favites', 6: 'other'},
                'mapping': {
                    0: [0, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
                    1: [1, 2, 3, 4],
                    2: [5],
                    3: [6],
                    4: [7],
                    5: [8],
                    6: [9, 10, 11, 12, 13, 14, 15]
                }
            }
        }
    }

def test_task_splitter_integration(real_task_definitions):
    """Unit test for the task splitter integration with datasets using real task definitions."""
    # Test MTL Task Splitter
    mtl_splitter = MTLTaskSplitter(real_task_definitions)
    
    # Check that hierarchical definitions were created properly
    assert 'genus' in mtl_splitter.hierarchical_definitions
    assert 'health' in mtl_splitter.hierarchical_definitions
    
    # Check ungrouped definitions exist - real task definitions have different structure
    genus_def = mtl_splitter.hierarchical_definitions['genus']
    assert 'ungrouped' in genus_def
    assert 'id2label' in genus_def['ungrouped']
    # Real genus task has different number of classes
    
    # Verify all expected tasks are present
    expected_tasks = {'genus', 'health', 'fish', 'human_artifacts', 'substrate', 'background', 'biota'}
    assert expected_tasks.issubset(set(real_task_definitions.keys()))
    
    # Test Base Task Splitter for baseline models
    base_splitter = BaseTaskSplitter(real_task_definitions)
    
    # Should have global/flat mapping
    assert hasattr(base_splitter, 'flat_id2label')
    assert hasattr(base_splitter, 'flat_mapping_array')

@pytest.fixture
def mock_hf_dataset_dict():
    """Creates a mock Hugging Face dataset dictionary for testing."""
    data = {
        'image': [Image.new('RGB', (256, 256), color='red')],
        'label': [Image.fromarray(np.zeros((256, 256), dtype=np.uint8))]
    }
    hf_dataset = HFDataset.from_dict(data)
    return {'train': hf_dataset, 'validation': hf_dataset}

def test_mtl_dataset_getitem_shapes_types(mock_hf_dataset_dict, real_task_definitions, monkeypatch):
    """
    Tests the __getitem__ method for MTL dataset in both training and validation modes.
    """
    patch_size = 128
    
    # Mock load_dataset to return our mock dataset
    def mock_load_dataset(name, split=None):
        return mock_hf_dataset_dict[split] if split else mock_hf_dataset_dict
    monkeypatch.setattr('coral_mtl.data.dataset.load_dataset', mock_load_dataset)

    # Create task splitter
    mtl_splitter = MTLTaskSplitter(real_task_definitions)

    # 1. Test validation mode (no augmentations)
    val_dataset = CoralscapesMTLDataset(
        splitter=mtl_splitter,
        hf_dataset_name="mock/dataset",
        split='validation',
        augmentations=None,
        patch_size=patch_size
    )
    val_sample = val_dataset[0]
    
    assert val_sample['image'].shape == (3, patch_size, patch_size)
    assert val_sample['image'].dtype == torch.float32
    assert set(val_sample['masks'].keys()) == set(real_task_definitions.keys())
    for task_name, mask in val_sample['masks'].items():
        assert mask.shape == (patch_size, patch_size)
        assert mask.dtype == torch.long

    # 2. Test training mode (with augmentations)
    train_augs = SegmentationAugmentation(patch_size=patch_size)
    train_dataset = CoralscapesMTLDataset(
        splitter=mtl_splitter,
        hf_dataset_name="mock/dataset",
        split='train',
        augmentations=train_augs,
        patch_size=patch_size
    )
    train_sample = train_dataset[0]

    assert train_sample['image'].shape == (3, patch_size, patch_size)
    assert train_sample['image'].dtype == torch.float32
    for task_name, mask in train_sample['masks'].items():
        assert mask.shape == (patch_size, patch_size)
        assert mask.dtype == torch.long

def test_baseline_dataset_getitem_shapes_types(mock_hf_dataset_dict, real_task_definitions, monkeypatch):
    """
    Tests the __getitem__ method for baseline dataset.
    """
    patch_size = 128
    
    # Mock load_dataset to return our mock dataset
    def mock_load_dataset(name, split=None):
        return mock_hf_dataset_dict[split] if split else mock_hf_dataset_dict
    monkeypatch.setattr('coral_mtl.data.dataset.load_dataset', mock_load_dataset)

    # Create task splitter
    base_splitter = BaseTaskSplitter(real_task_definitions)

    # Test baseline dataset
    baseline_dataset = CoralscapesDataset(
        splitter=base_splitter,
        hf_dataset_name="mock/dataset",
        split='validation',
        augmentations=None,
        patch_size=patch_size
    )
    sample = baseline_dataset[0]
    
    assert sample['image'].shape == (3, patch_size, patch_size)
    assert sample['image'].dtype == torch.float32
    assert sample['mask'].shape == (patch_size, patch_size)
    assert sample['mask'].dtype == torch.long
    assert 'original_mask' in sample  # For evaluation purposes