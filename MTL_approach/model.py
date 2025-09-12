# model.py

import torch.nn as nn
from typing import Dict

# Import the custom encoder and decoder modules
from segformer_encoder import SegFormerEncoder
from hierarchical_decoder import HierarchicalContextAwareDecoder

class CoralMTLModel(nn.Module):
    """
    The main multi-task learning model for coral segmentation.

    This class encapsulates the SegFormer encoder and the custom hierarchical
    context-aware decoder into a single, cohesive model.
    """
    def __init__(self, encoder_name: str, decoder_channel: int, num_classes: Dict[str, int], attention_dim: int):
        """
        Args:
            encoder_name (str): The Hugging Face ID or local path for the SegFormer backbone.
            decoder_channel (int): The unified channel dimension for all decoder streams.
            num_classes (Dict[str, int]): A dictionary mapping task names to their number of classes.
            attention_dim (int): The dimension for the query, key, and value in the attention module.
        """
        super().__init__()
        
        # 1. Instantiate the pre-trained encoder backbone
        self.encoder = SegFormerEncoder(pretrained_weights_path=encoder_name)
        
        # 2. Instantiate the custom multi-task decoder
        self.decoder = HierarchicalContextAwareDecoder(
            encoder_channels=self.encoder.channels,
            decoder_channel=decoder_channel,
            num_classes=num_classes,
            attention_dim=attention_dim
        )

    def forward(self, x):
        """
        Defines the forward pass of the model.
        
        Args:
            x (torch.Tensor): The input image tensor of shape (B, 3, H, W).

        Returns:
            Dict[str, torch.Tensor]: A dictionary of output logits for each task.
        """
        # Get multi-scale feature maps from the encoder
        features = self.encoder(x)
        
        # Process features through the decoder to get multi-task logits
        logits = self.decoder(features)
        
        return logits