import torch
import torch.nn as nn
from typing import List
from transformers import SegformerModel

class SegFormerEncoder(nn.Module):
    """
    A wrapper for the Hugging Face SegformerModel to act as an encoder backbone.

    This class loads a pre-trained Mix Transformer (MiT) model and provides a simple
    interface to extract multi-scale feature maps from an input image.
    """
    def __init__(self, pretrained_weights_path: str = "nvidia/mit-b2"):
        """
        Initializes the SegFormerEncoder.

        Args:
            pretrained_weights_path (str): The path to the pre-trained model.
                - Can be a model ID from the Hugging Face Hub (e.g., "nvidia/mit-b2").
                - Can be a path to a local directory containing the model's `config.json`
                  and `pytorch_model.bin` files.
        """
        super().__init__()
        
        # Load the pre-trained SegFormer model.
        self.encoder = SegformerModel.from_pretrained(pretrained_weights_path)
        
        # Extract the hidden sizes (channel dimensions) from each stage of the encoder.
        self.hidden_sizes: List[int] = self.encoder.config.hidden_sizes
        
    @property
    def channels(self) -> List[int]:
        """
        Provides the number of output channels for each feature map from the encoder.
        """
        return self.hidden_sizes

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Performs the forward pass to extract feature maps.

        Args:
            x (torch.Tensor): The input image tensor of shape (B, 3, H, W).

        Returns:
            List[torch.Tensor]: A list of 4 feature maps from the encoder's stages.
        """
        encoder_outputs = self.encoder(x, output_hidden_states=True)
        hidden_states = encoder_outputs.hidden_states
        
        return list(hidden_states)