import torch
import pytest
from coral_mtl.model.encoder import SegFormerEncoder

@pytest.mark.integration  # Requires network access to download model
def test_segformer_encoder_loading_and_forward():
    """
    Tests if the SegFormerEncoder can be loaded from Hugging Face
    and if its forward pass produces the correct number and shapes of feature maps.
    """
    model_path = "nvidia/mit-b0"  # Use a smaller model for faster testing
    
    try:
        encoder = SegFormerEncoder(pretrained_weights_path=model_path)
    except OSError as e:
        pytest.fail(f"Could not download or load model '{model_path}': {e}")
        
    batch_size = 2
    image_size = 256
    dummy_input = torch.randn(batch_size, 3, image_size, image_size)

    with torch.no_grad():
        feature_maps = encoder(dummy_input)

    expected_channels = encoder.channels
    assert len(feature_maps) == len(expected_channels), "Encoder did not return 4 feature maps."
    
    for i, features in enumerate(feature_maps):
        expected_size = image_size // (2**(i+2))
        assert features.shape[0] == batch_size
        assert features.shape[1] == expected_channels[i]
        assert features.shape[2] == expected_size
        assert features.shape[3] == expected_size