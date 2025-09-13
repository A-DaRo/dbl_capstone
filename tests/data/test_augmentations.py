import torch
import numpy as np
from PIL import Image
import pytest
from torchvision.transforms import v2

from coral_mtl.data.augmentations import SegmentationAugmentation

@pytest.fixture
def dummy_data():
    """Provides a consistent dummy image and mask for augmentation tests."""
    patch_size = 128
    # Create a distinct image with 4 colored quadrants
    dummy_np_image = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
    dummy_np_image[:patch_size//2, :patch_size//2, 0] = 255  # Top-left Red
    dummy_np_image[:patch_size//2, patch_size//2:, 1] = 255  # Top-right Green
    dummy_np_image[patch_size//2:, :patch_size//2, 2] = 255  # Bottom-left Blue
    
    # Create a mask with 4 distinct class quadrants
    mask_np = np.zeros((patch_size, patch_size), dtype=np.uint8)
    mask_np[:patch_size//2, :patch_size//2] = 1
    mask_np[:patch_size//2, patch_size//2:] = 2
    mask_np[patch_size//2:, :patch_size//2] = 3
    mask_np[patch_size//2:, patch_size//2:] = 4

    image = Image.fromarray(dummy_np_image)
    masks = {'task1': Image.fromarray(mask_np)}
    return image, masks, patch_size

def test_augmentations_output_shape_and_type(dummy_data):
    """Tests that the output tensors have the correct shape and dtype."""
    image, masks, patch_size = dummy_data
    aug_pipeline = SegmentationAugmentation(patch_size=patch_size)
    
    aug_image, aug_masks = aug_pipeline(image, masks)
    
    assert aug_image.shape == (3, patch_size, patch_size)
    assert aug_image.dtype == torch.float32
    
    assert isinstance(aug_masks, dict)
    assert 'task1' in aug_masks
    assert aug_masks['task1'].shape == (patch_size, patch_size)
    assert aug_masks['task1'].dtype == torch.long

def test_geometric_transforms_are_synced(dummy_data):
    """Ensures geometric transforms are applied identically to image and mask."""
    image, masks, patch_size = dummy_data
    
    # Create a pipeline with a deterministic horizontal flip
    aug_pipeline = SegmentationAugmentation(patch_size=patch_size)
    aug_pipeline.geometric_transforms = v2.Compose([
        v2.RandomHorizontalFlip(p=1.0)
    ])
    
    aug_image, aug_masks = aug_pipeline(image, masks)
    
    # Check if the mask was flipped correctly.
    # Class 1 (top-left) should now be in the top-right quadrant.
    original_mask_np = np.array(masks['task1'])
    flipped_mask_tensor = aug_masks['task1']
    
    top_left_original_val = original_mask_np[0, 0]
    top_right_flipped_val = flipped_mask_tensor[0, -1].item()
    
    assert top_left_original_val == 1
    assert top_right_flipped_val == 1

def test_color_transforms_only_affect_image(dummy_data):
    """Ensures color jitter does not change the segmentation mask."""
    image, masks, patch_size = dummy_data
    original_mask_tensor = v2.functional.to_image(masks['task1']).to(torch.long)
    
    # Create a pipeline with ONLY a strong color jitter
    aug_pipeline = SegmentationAugmentation(patch_size=patch_size)
    aug_pipeline.geometric_transforms = v2.Compose([v2.Resize((patch_size, patch_size))]) # No geometry change
    aug_pipeline.color_transforms = v2.Compose([
        v2.ColorJitter(brightness=0.9, contrast=0.9)
    ])
    
    aug_image, aug_masks = aug_pipeline(image, masks)
    
    # The mask should be completely unchanged
    assert torch.equal(original_mask_tensor.squeeze(0), aug_masks['task1'])