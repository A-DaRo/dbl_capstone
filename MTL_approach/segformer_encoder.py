import torch
import torch.nn as nn
from typing import List
from transformers import SegformerModel, PretrainedConfig

class SegFormerEncoder(nn.Module):
    """
    A wrapper for the Hugging Face SegformerModel to act as an encoder backbone.

    This class loads a pre-trained Mix Transformer (MiT) model and provides a simple
    interface to extract multi-scale feature maps from an input image. It is designed
    to be used with a custom decoder, such as the HierarchicalContextAwareDecoder.
    """
    def __init__(self, pretrained_weights_path: str = "nvidia/mit-b2"):
        """
        Initializes the SegFormerEncoder.

        Args:
            pretrained_weights_path (str): The path to the pre-trained model.
                - Can be a model ID from the Hugging Face Hub (e.g., "nvidia/mit-b2").
                - Can be a path to a local directory containing the model's `config.json`
                  and `pytorch_model.bin` files (e.g., "./models/mit-b2-finetuned").
        """
        super().__init__()
        
        print(f"--- Loading SegFormer encoder from: {pretrained_weights_path} ---")
        
        # Load the pre-trained SegFormer model.
        # The `from_pretrained` method can handle both Hub IDs and local paths.
        self.encoder = SegformerModel.from_pretrained(pretrained_weights_path)
        
        # Extract the hidden sizes (channel dimensions) from each stage of the encoder.
        # This is crucial for connecting the encoder to the decoder.
        self.hidden_sizes: List[int] = self.encoder.config.hidden_sizes
        
    @property
    def channels(self) -> List[int]:
        """
        Provides the number of output channels for each feature map from the encoder.

        Returns:
            List[int]: A list of integers representing the channel dimensions.
                       For MiT-B2: [64, 128, 320, 512]
        """
        return self.hidden_sizes

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Performs the forward pass to extract feature maps.

        Args:
            x (torch.Tensor): The input image tensor of shape (B, 3, H, W).

        Returns:
            List[torch.Tensor]: A list of 4 feature maps from the encoder's stages.
                                The shapes will be (B, C_i, H/(2^(i+2)), W/(2^(i+2))).
        """
        # The encoder returns a BaseModelOutput object. We need to extract the hidden_states.
        # `output_hidden_states=True` ensures that the model returns the feature maps
        # from all intermediate layers.
        encoder_outputs = self.encoder(x, output_hidden_states=True)
        
        # The `hidden_states` is a tuple containing the feature maps from each stage.
        hidden_states = encoder_outputs.hidden_states
        
        return list(hidden_states)


if __name__ == '__main__':
    # --- Example Usage and Sanity Check ---
    print("--- Running Sanity Check for SegFormerEncoder ---")

    # 1. Specify the model to use.
    # To use a local path, change this to your directory path, e.g., "./my_local_mit_b2"
    # For this example, we use a Hugging Face Hub ID. `transformers` will download and cache it.
    MODEL_PATH = "nvidia/mit-b2"
    
    # 2. Instantiate the encoder
    try:
        encoder = SegFormerEncoder(pretrained_weights_path=MODEL_PATH)
    except OSError as e:
        print(f"Error loading model: {e}")
        print(f"Please ensure '{MODEL_PATH}' is a valid Hugging Face model ID or a local path to a model.")
        exit()
        
    # 3. Create a dummy input tensor
    batch_size = 2
    image_height, image_width = 512, 512
    dummy_input = torch.randn(batch_size, 3, image_height, image_width)
    print(f"\nDummy input tensor shape: {dummy_input.shape}")

    # 4. Perform a forward pass
    # The encoder should not require gradients during inference/feature extraction
    with torch.no_grad():
        feature_maps = encoder(dummy_input)

    # 5. Check the output
    print(f"\nEncoder output channels (from encoder.channels): {encoder.channels}")
    print("\n--- Output Feature Maps Shapes ---")
    
    expected_channels = encoder.channels
    assert len(feature_maps) == len(expected_channels), "Encoder did not return 4 feature maps."
    
    for i, features in enumerate(feature_maps):
        expected_size = image_height // (2**(i+2))
        print(f"Stage {i+1}: {list(features.shape)}")
        
        # Verify shape consistency
        assert features.shape[0] == batch_size
        assert features.shape[1] == expected_channels[i]
        assert features.shape[2] == expected_size
        assert features.shape[3] == expected_size

    print("\nSanity check passed! The SegFormerEncoder is working correctly.")