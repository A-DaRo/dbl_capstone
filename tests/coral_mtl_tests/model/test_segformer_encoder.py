# tests/coral_mtl/model/test_segformer_encoder.py

import pytest
import torch

from coral_mtl.model.encoder import SegFormerEncoder


@pytest.mark.gpu
class TestSegFormerEncoder:
    """
    Test suite for the SegFormerEncoder wrapper.

    These tests verify the encoder's initialization, shape contracts,
    gradient flow, and robustness to different configurations.
    """

    @pytest.mark.parametrize(
        "model_id, expected_channels",
        [
            ("mit_b0", [32, 64, 160, 256]),
            ("mit_b1", [64, 128, 320, 512]),
        ],
    )
    def test_initialization_and_channels(self, model_id, expected_channels):
        """
        Verifies that the encoder initializes correctly with different model sizes
        and that the `channels` property reports the exact, correct channel dimensions
        for each encoder stage.
        """
        # Action
        try:
            encoder = SegFormerEncoder(name=model_id)
        except Exception as e:
            pytest.fail(f"Initialization of SegFormerEncoder with '{model_id}' failed: {e}")

        # Assertion
        assert encoder is not None, "Encoder object should be created."
        channels = encoder.channels
        assert isinstance(channels, list), "Channels property should return a list."
        assert len(channels) == 6, "SMP encoders for MiT models must have 6 stages (including input)."
        # We are interested in the 4 feature maps that the decoder will use.
        # smp returns [in, stem, stage1, stage2, stage3, stage4]
        # We skip in_channels and stem_channels (which is 0 for MiT)
        assert channels[2:] == expected_channels, f"Channel dimensions for {model_id} are incorrect."

    @pytest.mark.parametrize(
        "h, w",
        [
            (128, 128),  # Standard square input
            (96, 160),   # Non-square, rectangular input
        ],
    )
    def test_forward_pass_shape_and_device(self, device, h, w):
        """
        Verifies the output shapes and device placement from the forward pass using
        both square and non-square inputs. This ensures the spatial downsampling
        logic is robust to varying aspect ratios.
        """
        # Setup
        encoder = SegFormerEncoder(name="mit_b0").to(device)
        encoder.eval()
        dummy_images = torch.randn(2, 3, h, w, device=device)

        # Action
        with torch.no_grad():
            features = encoder(dummy_images)

        # Assertion
        assert isinstance(features, list), "Forward pass should return a list of tensors."
        assert len(features) == 6, "Should return 6 feature maps for the 6 encoder stages."

        batch_size = dummy_images.shape[0]
        # SMP includes the input image as the first "feature"
        downsample_factors = [1, 2, 4, 8, 16, 32]
        expected_channels = encoder.channels

        for i, feature in enumerate(features):
            # The stem layer for MiT outputs a tensor with 0 channels, which is a valid case.
            if expected_channels[i] == 0:
                continue

            factor = downsample_factors[i]
            expected_shape = (batch_size, expected_channels[i], h // factor, w // factor)
            assert feature.shape == expected_shape, f"Shape mismatch at stage {i} for input {h}x{w}."
            assert feature.device.type == device.type, f"Tensor at stage {i} is on the wrong device."

    def test_forward_pass_is_deterministic_in_eval_mode(self, dummy_images, device):
        """
        Ensures that the forward pass is deterministic when the model is in evaluation
        mode. This is critical for reproducible inference.
        """
        # Setup
        encoder = SegFormerEncoder(name="mit_b0").to(device)
        encoder.eval()

        # Action
        with torch.no_grad():
            features1 = encoder(dummy_images)
            features2 = encoder(dummy_images)

        # Assertion
        for f1, f2 in zip(features1, features2):
            assert torch.equal(f1, f2), "Encoder output is not deterministic in eval mode."

    def test_gradient_flow(self, device):
        """
        Verifies that gradients flow back through all trainable parameters of the encoder,
        confirming that the model is properly connected for training.
        """
        # Setup
        encoder = SegFormerEncoder(name="mit_b0").to(device)
        encoder.train()
        # Bypassing the fixture and creating a local tensor to isolate the issue.
        dummy_images = torch.randn(2, 3, 128, 128, device=device)

        # Action
        features = encoder(dummy_images)
        # To isolate the gradient flow, we compute loss only on the last, deepest feature map.
        loss = features[-1].mean()
        loss.backward()

        # Assertion
        for name, param in encoder.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter '{name}' has no gradient."
                assert not torch.all(param.grad == 0), f"Gradient for '{name}' is all zeros."

    def test_invalid_model_path(self):
        """
        Tests that the encoder raises an OSError when provided with an invalid
        or non-existent model identifier, ensuring graceful failure.
        """
        invalid_name = "this_is_not_a_real_model_name"
        with pytest.raises(KeyError):
            _ = SegFormerEncoder(name=invalid_name)