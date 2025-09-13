import torch
import numpy as np
from PIL import Image
import pytest

from coral_mtl.data.dataset import create_lookup_table, CoralscapesMTLDataset, TASK_DEFINITIONS

def test_create_lookup_table():
    """Unit test for the label transformation logic."""
    # Using the 'health' task mapping as a test case
    health_mapping = TASK_DEFINITIONS['health']['mapping']
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
def mock_hf_dataset():
    """Creates a mock Hugging Face dataset for testing without network calls."""
    # Mocking the structure of datasets.Dataset
    from datasets import Dataset as HFDataset
    
    data = {
        'image': [Image.new('RGB', (256, 256), color='red')],
        'label': [Image.fromarray(np.zeros((256, 256), dtype=np.uint8))]
    }
    hf_dataset = HFDataset.from_dict(data)
    
    # The class expects a dictionary-like object with splits
    class MockDatasetDict:
        def __init__(self, dataset):
            self._dataset = dataset
        def __getitem__(self, key):
            return self._dataset
            
    return MockDatasetDict(hf_dataset)

def test_dataset_getitem_shapes_types(mock_hf_dataset):
    """
    Tests the __getitem__ method for both training (with augs) and validation modes.
    """
    patch_size = 128
    
    # 1. Test validation mode (no augmentations)
    val_dataset = CoralscapesMTLDataset(
        hf_dataset=mock_hf_dataset,
        split='validation',
        augmentations=None,
        patch_size=patch_size
    )
    val_sample = val_dataset[0]
    
    assert val_sample['image'].shape == (3, patch_size, patch_size)
    assert val_sample['image'].dtype == torch.float32
    assert set(val_sample['masks'].keys()) == set(TASK_DEFINITIONS.keys())
    for task_name, mask in val_sample['masks'].items():
        assert mask.shape == (patch_size, patch_size)
        assert mask.dtype == torch.long

    # 2. Test training mode (with augmentations)
    from coral_mtl.data.augmentations import SegmentationAugmentation
    train_augs = SegmentationAugmentation(patch_size=patch_size)
    train_dataset = CoralscapesMTLDataset(
        hf_dataset=mock_hf_dataset,
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