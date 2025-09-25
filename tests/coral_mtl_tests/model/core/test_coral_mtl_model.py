# tests/coral_mtl/model/test_coral_mtl_model.py

import pytest
import torch

from coral_mtl.model.core import CoralMTLModel


@pytest.mark.gpu
class TestCoralMTLModel:
    """
    Test suite for the CoralMTLModel.

    These tests rigorously verify the model's primary contract: producing a dictionary
    of tensors where the tasks and class counts are dynamically determined by the
    task splitter, preventing any hard-coded behavior.
    """

    @pytest.mark.parametrize(
        "h, w",
        [
            (128, 128),  # Standard square input
            (96, 160),   # Non-square, rectangular input
        ],
    )
    def test_forward_pass_dynamic_tasks(self, splitter_mtl, device, h, w):
        """
        Verifies the model's most critical contract:
        1. It returns a dictionary.
        2. The keys of the dictionary EXACTLY match the tasks from the splitter.
        3. The class dimension for each task's output tensor EXACTLY matches the
           class count defined in the splitter for that task.
        4. The output spatial dimensions match the input image dimensions.
        5. It works with non-square inputs.

        This test is parametrized over multiple splitter configurations to detect
        any hard-coded task names or class counts.
        """
        # Setup: Dynamically create the model based on the current splitter variant
        defined_tasks = splitter_mtl.hierarchical_definitions
        num_classes = {
            task: len(info["ungrouped"]["id2label"]) for task, info in defined_tasks.items()
        }
        # For this test, assume all defined tasks are primary to maximize complexity
        model = CoralMTLModel(
            encoder_name="mit_b0",
            decoder_channel=64,
            attention_dim=32,
            num_classes=num_classes,
            primary_tasks=list(defined_tasks.keys()),
            aux_tasks=[]
        ).to(device)
        model.eval()
        dummy_images = torch.randn(2, 3, h, w, device=device)

        # Action
        with torch.no_grad():
            outputs = model(dummy_images)

        # Assertion
        assert isinstance(outputs, dict), "CoralMTLModel must return a dictionary."
        assert set(outputs.keys()) == set(defined_tasks.keys()), "Output tasks do not match splitter tasks."

        for task_name, output_tensor in outputs.items():
            expected_classes = len(defined_tasks[task_name]["ungrouped"]["id2label"])
            expected_shape = (dummy_images.size(0), expected_classes, h, w)
            assert output_tensor.shape == expected_shape, f"Shape mismatch for task '{task_name}'."
            assert output_tensor.device.type == device.type, f"Tensor for task '{task_name}' is on wrong device."

    def test_gradient_flow(self, minimal_coral_mtl_model, dummy_images, device):
        """
        Ensures that gradients can be backpropagated through all model parameters,
        confirming the model is fully trainable.
        """
        # Setup
        model = minimal_coral_mtl_model.to(device)
        model.train()

        # Action
        outputs = model(dummy_images)
        # Combine all task outputs into a single scalar loss
        loss = sum(output.mean() for output in outputs.values())
        loss.backward()

        # Assertion
        component_checks = {
            "encoder": False,
            "decoder": False,
            "predictors": False,
            "attention": False,
        }

        primary_tasks = getattr(model.decoder, "primary_tasks", [])

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("decoder.to_qkv."):
                task_name = name.split(".")[2]
                if task_name not in primary_tasks:
                    continue
            assert param.grad is not None, f"Parameter '{name}' has no gradient."
            has_grad = bool(torch.any(param.grad != 0))
            if has_grad:
                if name.startswith("encoder."):
                    component_checks["encoder"] = True
                elif name.startswith("decoder.decoders"):
                    component_checks["decoder"] = True
                elif name.startswith("decoder.predictors"):
                    component_checks["predictors"] = True
                elif ".attn" in name or ".gating_layers" in name:
                    component_checks["attention"] = True

        required_components = ["encoder", "decoder", "predictors"]
        if primary_tasks:
            required_components.append("attention")

        missing = [comp for comp in required_components if not component_checks[comp]]
        assert not missing, f"No non-zero gradients detected for components: {missing}"

    def test_mode_switching(self, minimal_coral_mtl_model):
        """
        Verifies that calling train() and eval() correctly toggles the training mode
        of the model and its submodules (encoder, decoder).
        """
        # Test eval mode
        minimal_coral_mtl_model.eval()
        assert not minimal_coral_mtl_model.training
        assert not minimal_coral_mtl_model.encoder.training
        assert not minimal_coral_mtl_model.decoder.training

        # Test train mode
        minimal_coral_mtl_model.train()
        assert minimal_coral_mtl_model.training
        assert minimal_coral_mtl_model.encoder.training
        assert minimal_coral_mtl_model.decoder.training