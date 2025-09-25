# tests/coral_mtl/model/decoders/test_segformer_mlp_decoder.py

import pytest
import torch
from typing import List

from coral_mtl.model.decoders import SegFormerMLPDecoder


def create_dummy_encoder_features(
    batch_size: int = 2,
    h: int = 64,
    w: int = 64,
    channels: List[int] = None,
    device: torch.device = torch.device("cpu"),
) -> List[torch.Tensor]:
    """Helper to generate a list of multi-scale feature maps."""
    if channels is None:
        channels = [32, 64, 160, 256]  # Corresponds to mit-b0
    downsample_factors = [4, 8, 16, 32]
    features = [
        torch.randn(batch_size, c, h // f, w // f, device=device)
        for c, f in zip(channels, downsample_factors)
    ]
    return features


@pytest.mark.gpu
class TestSegFormerMLPDecoder:
    """Test suite for the SegFormerMLPDecoder."""

    @pytest.mark.parametrize(
        "encoder_channels, decoder_channel",
        [
            ([32, 64, 160, 256], 256),  # mit-b0
            ([64, 128, 320, 512], 768),  # mit-b2
        ],
    )
    def test_initialization(self, encoder_channels, decoder_channel):
        """Tests successful initialization with valid parameters."""
        try:
            decoder = SegFormerMLPDecoder(
                encoder_channels=encoder_channels, decoder_channel=decoder_channel
            )
            assert decoder is not None
            assert len(decoder.linear_c) == 4
            assert decoder.linear_fuse.proj.in_channels == decoder_channel * 4
        except Exception as e:
            pytest.fail(f"Initialization failed with valid parameters: {e}")

    def test_initialization_fails_with_invalid_channels(self):
        """Tests that initialization raises an AssertionError with incorrect number of channels."""
        with pytest.raises(AssertionError):
            SegFormerMLPDecoder(encoder_channels=[64, 128, 320], decoder_channel=256)

    def test_forward_pass_shape(self, device):
        """
        Verifies the output shape of the forward pass. The output tensor should have
        the specified decoder channel dimension and the spatial size of the first
        (largest) input feature map.
        """
        # Setup
        encoder_channels = [32, 64, 160, 256]
        decoder_channel = 128
        batch_size, h, w = 4, 128, 128
        
        features = create_dummy_encoder_features(
            batch_size, h, w, encoder_channels, device
        )
        decoder = SegFormerMLPDecoder(
            encoder_channels=encoder_channels, decoder_channel=decoder_channel
        ).to(device)
        decoder.eval()

        # Action
        with torch.no_grad():
            output = decoder(features)

        # Assertion
        assert isinstance(output, torch.Tensor)
        expected_shape = (batch_size, decoder_channel, h // 4, w // 4)
        assert output.shape == expected_shape, "Output tensor shape is incorrect."
        assert output.device.type == device.type, "Output tensor is on the wrong device."

    def test_gradient_flow(self, device):
        """Ensures that gradients can be backpropagated through all decoder parameters."""
        # Setup
        encoder_channels = [16, 32, 64, 128]
        decoder_channel = 64
        features = create_dummy_encoder_features(channels=encoder_channels, device=device)
        decoder = SegFormerMLPDecoder(
            encoder_channels=encoder_channels, decoder_channel=decoder_channel
        ).to(device)
        decoder.train()

        # Action
        output = decoder(features)
        loss = output.mean()
        loss.backward()

        # Assertion
        for name, param in decoder.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter '{name}' has no gradient."
                assert not torch.all(
                    param.grad == 0
                ), f"Gradient for '{name}' is all zeros."

    def test_dropout_behavior(self, device):
        """
        Verifies that dropout is applied during training and disabled during evaluation.
        """
        # Setup
        encoder_channels = [16, 32, 64, 128]
        decoder_channel = 64
        features = create_dummy_encoder_features(channels=encoder_channels, device=device)
        decoder = SegFormerMLPDecoder(
            encoder_channels=encoder_channels,
            decoder_channel=decoder_channel,
            dropout_prob=0.5,
        ).to(device)

        # In eval mode, output should be deterministic
        decoder.eval()
        with torch.no_grad():
            output1_eval = decoder(features)
            output2_eval = decoder(features)
        assert torch.equal(output1_eval, output2_eval), "Dropout should be off in eval mode."

        # In train mode, output should be stochastic
        decoder.train()
        with torch.no_grad():
            output1_train = decoder(features)
            output2_train = decoder(features)
        assert not torch.equal(
            output1_train, output2_train
        ), "Dropout should be active in train mode."