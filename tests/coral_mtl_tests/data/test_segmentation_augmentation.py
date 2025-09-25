"""Robust tests for ``SegmentationAugmentation`` enforcing spec compliance."""
from __future__ import annotations

import random
from typing import Dict

import numpy as np
import pytest
import torch
from PIL import Image, ImageOps
from torchvision.transforms import v2

from coral_mtl.data.augmentations import SegmentationAugmentation


def _tensor_to_pil_image(tensor: torch.Tensor) -> Image.Image:
    arr = tensor.detach().cpu().clamp(0, 1).mul(255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(arr)


def _tensor_to_pil_mask(tensor: torch.Tensor) -> Image.Image:
    arr = tensor.detach().cpu().to(torch.uint8).numpy()
    return Image.fromarray(arr, mode="L")


@pytest.mark.gpu
def test_geometric_transforms_affect_image_and_masks_identically(dummy_images, dummy_masks):
    """Test that geometric transforms are applied synchronously to both image and masks."""
    patch_size = 16
    augmentation = SegmentationAugmentation(patch_size=patch_size)
    # Use identity transform for color to isolate geometric effects
    augmentation.color_transforms = v2.Compose([v2.Lambda(lambda x: x)])
    # Set deterministic geometric transforms
    augmentation.geometric_transforms = v2.Compose([
        v2.Resize((patch_size, patch_size), antialias=True),
        v2.RandomHorizontalFlip(p=1.0),  # Always flip
    ])

    image_pil = _tensor_to_pil_image(dummy_images[0].to("cpu"))
    mask_pils: Dict[str, Image.Image] = {
        task: _tensor_to_pil_mask(mask[0].to("cpu")) for task, mask in dummy_masks.items()
    }

    # Apply augmentation twice with same seed to ensure deterministic behavior
    torch.manual_seed(42)
    np.random.seed(42)
    transformed_image1, transformed_masks1 = augmentation(image_pil, mask_pils)
    
    torch.manual_seed(42)
    np.random.seed(42)
    transformed_image2, transformed_masks2 = augmentation(image_pil, mask_pils)

    # The same augmentation should produce identical results with same seed
    torch.testing.assert_close(transformed_image1, transformed_image2)
    for task_name in transformed_masks1:
        torch.testing.assert_close(transformed_masks1[task_name], transformed_masks2[task_name])

    # Test that output has expected properties
    assert transformed_image1.shape == (3, patch_size, patch_size)
    for task_name, mask_tensor in transformed_masks1.items():
        assert mask_tensor.shape == (patch_size, patch_size)
        assert mask_tensor.dtype == torch.long


@pytest.mark.gpu
def test_color_transforms_only_modify_image(dummy_images, dummy_masks):
    """Test that color transforms only affect the image, leaving masks unchanged."""
    patch_size = 24
    augmentation = SegmentationAugmentation(patch_size=patch_size)
    # Use deterministic geometric transform
    augmentation.geometric_transforms = v2.Compose([v2.Resize((patch_size, patch_size), antialias=True)])
    # Start with identity color transform
    augmentation.color_transforms = v2.Compose([v2.Lambda(lambda x: x)])

    image_pil = _tensor_to_pil_image(dummy_images[0].to("cpu"))
    mask_pils = {task: _tensor_to_pil_mask(mask[0].to("cpu")) for task, mask in dummy_masks.items()}

    # Get baseline results with no color transformation
    torch.manual_seed(42)
    np.random.seed(42)
    baseline_image, baseline_masks = augmentation(image_pil, mask_pils)

    # Apply color transformation (invert the image)
    augmentation.color_transforms = v2.Compose([v2.Lambda(lambda x: ImageOps.invert(x))])
    torch.manual_seed(42)
    np.random.seed(42)
    inverted_image, inverted_masks = augmentation(image_pil, mask_pils)

    # Masks should remain identical (color transforms don't affect masks)
    for task in baseline_masks:
        torch.testing.assert_close(baseline_masks[task], inverted_masks[task])

    # Images should be different (color transforms affect images)
    with pytest.raises(AssertionError):
        torch.testing.assert_close(baseline_image, inverted_image)


@pytest.mark.gpu
def test_outputs_have_expected_shapes_and_dtypes(dummy_images, dummy_masks):
    patch_size = 20
    augmentation = SegmentationAugmentation(patch_size=patch_size)

    image_pil = _tensor_to_pil_image(dummy_images[0].to("cpu"))
    mask_pils = {task: _tensor_to_pil_mask(mask[0].to("cpu")) for task, mask in dummy_masks.items()}

    augmented_image, augmented_masks = augmentation(image_pil, mask_pils)

    assert augmented_image.dtype == torch.float32
    assert augmented_image.shape == (3, patch_size, patch_size)

    assert set(augmented_masks.keys()) == set(mask_pils.keys())
    for task_name, mask_tensor in augmented_masks.items():
        assert mask_tensor.dtype == torch.long
        assert mask_tensor.shape == (patch_size, patch_size)
        source_values = torch.unique(dummy_masks[task_name][0].to("cpu")).tolist()
        assert set(torch.unique(mask_tensor).tolist()).issubset(set(source_values))


@pytest.mark.gpu
def test_augmentation_is_deterministic_with_fixed_seeds(dummy_images, dummy_masks):
    def run_once() -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        torch.manual_seed(123)
        np.random.seed(123)
        random.seed(123)
        augmentation = SegmentationAugmentation(patch_size=18)
        image = _tensor_to_pil_image(dummy_images[0].to("cpu"))
        masks = {task: _tensor_to_pil_mask(mask[0].to("cpu")) for task, mask in dummy_masks.items()}
        return augmentation(image, masks)

    image_a, masks_a = run_once()
    image_b, masks_b = run_once()

    torch.testing.assert_close(image_a, image_b)
    for task in masks_a:
        torch.testing.assert_close(masks_a[task], masks_b[task])

