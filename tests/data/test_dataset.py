import torch
import numpy as np
from PIL import Image
import pytest
from datasets import Dataset as HFDataset

from coral_mtl.data.dataset import create_lookup_table, CoralscapesMTLDataset
from coral_mtl.data.augmentations import SegmentationAugmentation

@pytest.fixture
def task_definitions():
    """Provides a mock of the task definitions dictionary."""
    return {
        'health': {
            'mapping': {
                0: [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
                1: [17],
                2: [16],
                3: [20]
            },
            'num_classes': 4,
            'weight': 1.0
        },
        'genus': {
            'mapping': {
                0: [0, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
                1: [1, 2, 3, 4],
                2: [5],
                3: [6],
                4: [7],
                5: [8],
                6: [9],
                7: [10, 11, 12, 13, 14],
                8: [15]
            },
            'num_classes': 9,
            'weight': 1.0
        }
    }

def test_create_lookup_table(task_definitions):
    """Unit test for the label transformation logic."""
    # Using the 'health' task mapping as a test case
    health_mapping = task_definitions['health']['mapping']
    lookup_table = create_lookup_table(health_mapping)

    # Create a synthetic raw mask with known original class IDs
    # Original IDs: 16 (bleached), 17 (alive), 20 (dead), 5 (sand -> unlabeled)
    raw_mask_np = np.array([
        [16, 17],
        [20, 5]
    ], dtype=np.int64)

    # Apply the lookup table
    health_mask_np = lookup_table[raw_mask_np]

    # Define the expected output mask based on the health mapping
    # New IDs: 2 (bleached), 1 (alive), 3 (dead), 0 (unlabeled)
    expected_mask_np = np.array([
        [2, 1],
        [3, 0]
    ], dtype=np.int64)

    assert np.array_equal(health_mask_np, expected_mask_np)

@pytest.fixture
def mock_hf_dataset_dict():
    """Creates a mock Hugging Face dataset dictionary for testing."""
    data = {
        'image': [Image.new('RGB', (256, 256), color='red')],
        'label': [Image.fromarray(np.zeros((256, 256), dtype=np.uint8))]
    }
    hf_dataset = HFDataset.from_dict(data)
    return {'train': hf_dataset, 'validation': hf_dataset}

def test_dataset_getitem_shapes_types(mock_hf_dataset_dict, task_definitions, monkeypatch):
    """
    Tests the __getitem__ method for both training (with augs) and validation modes.
    """
    patch_size = 128
    
    # Mock load_dataset to return our mock dataset
    monkeypatch.setattr('coral_mtl.data.dataset.load_dataset', lambda name: mock_hf_dataset_dict)

    # 1. Test validation mode (no augmentations)
    val_dataset = CoralscapesMTLDataset(
        hf_dataset_name="mock/dataset",
        split='validation',
        augmentations=None,
        patch_size=patch_size,
        task_definitions=task_definitions
    )
    val_sample = val_dataset[0]
    
    assert val_sample['image'].shape == (3, patch_size, patch_size)
    assert val_sample['image'].dtype == torch.float32
    assert set(val_sample['masks'].keys()) == set(task_definitions.keys())
    for task_name, mask in val_sample['masks'].items():
        assert mask.shape == (patch_size, patch_size)
        assert mask.dtype == torch.long

    # 2. Test training mode (with augmentations)
    train_augs = SegmentationAugmentation(patch_size=patch_size)
    train_dataset = CoralscapesMTLDataset(
        hf_dataset_name="mock/dataset",
        split='train',
        augmentations=train_augs,
        patch_size=patch_size,
        task_definitions=task_definitions
    )
    train_sample = train_dataset[0]

    assert train_sample['image'].shape == (3, patch_size, patch_size)
    assert train_sample['image'].dtype == torch.float32
    for task_name, mask in train_sample['masks'].items():
        assert mask.shape == (patch_size, patch_size)
        assert mask.dtype == torch.long