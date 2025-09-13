import torch
import pytest
from coral_mtl.model.decoders import HierarchicalContextAwareDecoder

@pytest.fixture
def decoder_config():
    """Provides a standard configuration for the decoder tests."""
    return {
        "encoder_channels": [64, 128, 320, 512],
        "decoder_channel": 256,
        "attention_dim": 128,
        "num_classes": {
            'panoptic_shape': 9,
            'panoptic_health': 4,
            'fish': 2,
            'human_artifacts': 2,
            'substrate': 4
        }
    }

def test_hierarchical_decoder_instantiation(decoder_config):
    """Tests if the decoder can be instantiated without errors."""
    try:
        HierarchicalContextAwareDecoder(**decoder_config)
    except Exception as e:
        pytest.fail(f"Decoder instantiation failed with error: {e}")

def test_hierarchical_decoder_forward_pass(decoder_config):
    """
    Tests the forward pass of the decoder with dummy encoder features,
    ensuring the output logits have the correct shapes.
    """
    B = 2
    H, W = 512, 512
    
    # Create dummy input features matching encoder output scales
    dummy_features = [
        torch.randn(B, decoder_config["encoder_channels"][0], H // 4, W // 4),
        torch.randn(B, decoder_config["encoder_channels"][1], H // 8, W // 8),
        torch.randn(B, decoder_config["encoder_channels"][2], H // 16, W // 16),
        torch.randn(B, decoder_config["encoder_channels"][3], H // 32, W // 32),
    ]

    decoder = HierarchicalContextAwareDecoder(**decoder_config)
    output_logits = decoder(dummy_features)

    assert isinstance(output_logits, dict)
    assert set(output_logits.keys()) == set(decoder_config["num_classes"].keys())

    for task_name, logits in output_logits.items():
        assert logits.shape[0] == B
        assert logits.shape[1] == decoder_config["num_classes"][task_name]
        assert logits.shape[2] == H // 4  # Decoder outputs at 1/4 resolution
        assert logits.shape[3] == W // 4