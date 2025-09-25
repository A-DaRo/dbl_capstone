# Edit file: tests/coral_mtl/engine/test_sliding_window_inferrer.py
"""
Tests for the SlidingWindowInferrer class.

This module verifies the core logic of the sliding window inferrer, ensuring
it correctly handles image padding, patch extraction, batched inference,
and stitching for various image sizes and model types.
"""
import pytest
import torch
import math
from unittest.mock import MagicMock

from coral_mtl.engine.inference import SlidingWindowInferrer


@pytest.fixture
def mock_mtl_model(minimal_coral_mtl_model, device):
    """A mock CoralMTLModel that returns a dictionary of tensors."""
    model = minimal_coral_mtl_model.to(device)
    # Spy on the forward method to count calls
    model.forward = MagicMock(wraps=model.forward)
    return model


@pytest.fixture
def mock_baseline_model(minimal_baseline_model, device):
    """A mock BaselineSegformer that returns a single tensor."""
    model = minimal_baseline_model.to(device)
    # Spy on the forward method to count calls
    model.forward = MagicMock(wraps=model.forward)
    return model


@pytest.mark.parametrize("image_shape, patch_size, stride", [
    ((1, 3, 256, 256), 128, 64),  # Perfect fit
    ((1, 3, 200, 310), 128, 64),  # Requires padding
    ((2, 3, 256, 256), 128, 128), # Batch of images
    ((1, 3, 512, 256), 256, 128), # Rectangular image
])
def test_output_shape_is_preserved(mock_mtl_model, device, image_shape, patch_size, stride):
    """
    CRITICAL: Verifies the output logits tensor has the exact same spatial
    dimensions as the input image, regardless of padding.
    """
    # Arrange
    inferrer = SlidingWindowInferrer(
        model=mock_mtl_model,
        patch_size_h=patch_size,
        patch_size_w=patch_size,
        stride_h=stride,
        stride_w=stride,
        device=device,
        batch_size=4
    )
    input_image = torch.randn(image_shape, device=device)

    # Act
    output_logits = inferrer.predict(input_image)

    # Assert
    assert isinstance(output_logits, dict)
    for task_name, logits_tensor in output_logits.items():
        assert logits_tensor.shape[0] == image_shape[0], "Batch size should be preserved"
        # The core assertion: H and W must match the original input
        assert logits_tensor.shape[2] == image_shape[2], "Height must match original input"
        assert logits_tensor.shape[3] == image_shape[3], "Width must match original input"
        assert logits_tensor.device.type == device.type
        assert logits_tensor.dtype == torch.float32


def test_mtl_model_output_structure(mock_mtl_model, device):
    """Ensures the inferrer preserves the dictionary structure from an MTL model."""
    # Arrange
    inferrer = SlidingWindowInferrer(
        model=mock_mtl_model, patch_size_h=64, patch_size_w=64, stride_h=32, stride_w=32, device=device
    )
    input_image = torch.randn(1, 3, 128, 128, device=device)

    # Act
    output_logits = inferrer.predict(input_image)

    # Assert
    # The output keys should match the tasks defined in the model's internal config
    expected_tasks = set(mock_mtl_model.decoder.predictors.keys())
    assert isinstance(output_logits, dict)
    assert set(output_logits.keys()) == expected_tasks


def test_baseline_model_output_standardization(mock_baseline_model, device):
    """
    Ensures the inferrer wraps a single tensor output from a baseline model
    into the standard {'segmentation': tensor} dictionary format.
    """
    # Arrange
    inferrer = SlidingWindowInferrer(
        model=mock_baseline_model, patch_size_h=64, patch_size_w=64, stride_h=32, stride_w=32, device=device
    )
    input_image = torch.randn(1, 3, 128, 128, device=device)

    # Act
    output_logits = inferrer.predict(input_image)

    # Assert
    assert isinstance(output_logits, dict)
    assert list(output_logits.keys()) == ['segmentation']
    assert torch.is_tensor(output_logits['segmentation'])
    assert output_logits['segmentation'].shape[-2:] == (128, 128)


@pytest.mark.parametrize("image_wh, patch_size, stride, patch_batch_size, expected_calls", [
    # Case 1: 4 patches total, batch size 4 -> 1 model call
    (256, 128, 128, 4, 1),
    # Case 2: 4 patches total, batch size 2 -> 2 model calls
    (256, 128, 128, 2, 2),
    # Case 3: 4 patches total, batch size 1 -> 4 model calls
    (256, 128, 128, 1, 4),
    # Case 4: 9 patches total (3x3 grid), batch size 4 -> 3 calls (4, 4, 1)
    (256, 128, 64, 4, 3),
])
def test_patch_batching_logic(mock_mtl_model, device, image_wh, patch_size, stride, patch_batch_size, expected_calls):
    """
    Verifies that the internal batching of patches works correctly by counting
    the number of forward passes made by the model.
    """
    # Arrange
    inferrer = SlidingWindowInferrer(
        model=mock_mtl_model,
        patch_size_h=patch_size,
        patch_size_w=patch_size,
        stride_h=stride,
        stride_w=stride,
        device=device,
        batch_size=patch_batch_size
    )
    input_image = torch.randn(1, 3, image_wh, image_wh, device=device)
    mock_mtl_model.forward.reset_mock() # Reset call count for this test

    # Act
    inferrer.predict(input_image)

    # Assert
    assert mock_mtl_model.forward.call_count == expected_calls


def test_no_grad_context(mock_mtl_model, device):
    """
    Ensures that the entire predict method is wrapped in `torch.no_grad()`
    by checking if the model's output tensor requires grad.
    """
    inferrer = SlidingWindowInferrer(model=mock_mtl_model, patch_size_h=32, patch_size_w=32, stride_h=16, stride_w=16, device=device)
    input_image = torch.randn(1, 3, 64, 64, device=device, requires_grad=False)

    # The mock model's forward pass uses the real model, which will track gradients
    # if not under a no_grad() context.
    output = inferrer.predict(input_image)

    # If predict() correctly uses no_grad(), the output should not require grad.
    for task_logits in output.values():
        assert not task_logits.requires_grad, "Output tensor should not require grad"