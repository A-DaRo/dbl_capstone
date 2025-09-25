# tests/coral_mtl/model/test_baseline_segformer.py

import pytest
import torch

from coral_mtl.model.core import BaselineSegformer


@pytest.mark.gpu
class TestBaselineSegformer:
    """
    Test suite for the BaselineSegformer model.

    These tests verify the model's core contract: producing a single tensor output
    where the number of classes is dynamically determined by the task splitter.
    """

    @pytest.mark.parametrize(
        "h, w",
        [
            (128, 128),  # Standard square input
            (96, 160),   # Non-square, rectangular input
        ],
    )
    def test_forward_pass_shape_and_device(self, minimal_baseline_model, splitter_base, device, h, w):
        """
        Verifies the model's primary contract:
        1. It returns a single tensor.
        2. The output class dimension dynamically matches the splitter's global class count.
        3. The output spatial dimensions match the input image dimensions.
        4. It works with non-square inputs.
        """
        # Setup
        model = minimal_baseline_model.to(device)
        model.eval()
        dummy_images = torch.randn(2, 3, h, w, device=device)

        # Action
        with torch.no_grad():
            output = model(dummy_images)

        # Assertion
        assert isinstance(output, torch.Tensor), "Baseline model must return a single tensor."

        expected_classes = splitter_base.num_global_classes
        expected_shape = (dummy_images.size(0), expected_classes, h, w)

        assert output.shape == expected_shape, "Output tensor shape is incorrect."
        assert output.device.type == device.type, "Output tensor is on the wrong device."

    def test_gradient_flow(self, minimal_baseline_model, dummy_images, device):
        """
        Ensures that gradients can be backpropagated through all model parameters,
        confirming the model is fully trainable.
        """
        # Setup
        model = minimal_baseline_model.to(device)
        model.train()

        # Action
        output = model(dummy_images)
        loss = output.mean()
        loss.backward()

        # Assertion
        component_checks = {
            "encoder": False,
            "decoder": False,
            "head": False,
        }

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            assert param.grad is not None, f"Parameter '{name}' has no gradient."
            has_grad = bool(torch.any(param.grad != 0))
            if name.startswith("encoder.") and has_grad:
                component_checks["encoder"] = True
            elif name.startswith("decoder.") and has_grad:
                component_checks["decoder"] = True
            elif name.startswith("prediction_head") and has_grad:
                component_checks["head"] = True

        missing = [component for component, ok in component_checks.items() if not ok]
        assert not missing, f"No non-zero gradients detected for components: {missing}"

    def test_mode_switching(self, minimal_baseline_model):
        """
        Verifies that calling train() and eval() correctly toggles the training mode
        of the model and its submodules (encoder, decoder).
        """
        # Test eval mode
        minimal_baseline_model.eval()
        assert not minimal_baseline_model.training
        assert not minimal_baseline_model.encoder.training
        assert not minimal_baseline_model.decoder.training

        # Test train mode
        minimal_baseline_model.train()
        assert minimal_baseline_model.training
        assert minimal_baseline_model.encoder.training
        assert minimal_baseline_model.decoder.training