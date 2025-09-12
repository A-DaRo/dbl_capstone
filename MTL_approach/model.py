# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List

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

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Defines the forward pass of the model.
        
        Args:
            images (torch.Tensor): The input image tensor of shape (B, 3, H, W).

        Returns:
            Dict[str, torch.Tensor]: A dictionary of output logits for each task,
                                     upsampled to the original image size (B, C, H, W).
        """
        # 1. Get multi-scale feature maps from the encoder
        features: List[torch.Tensor] = self.encoder(images)
        
        # 2. Process features through the decoder to get multi-task logits
        # Note: The decoder outputs logits at 1/4 resolution (e.g., 128x128 for a 512x512 input)
        logits_at_quarter_res = self.decoder(features)
        
        # 3. Upsample all output logits to match the original image size.
        # This is the crucial step that resolves the size mismatch with the ground truth masks
        # that caused the previous RuntimeError.
        original_size = images.shape[-2:] # Get (H, W) from the input tensor
        
        upsampled_logits = {}
        for task_name, logit_tensor in logits_at_quarter_res.items():
            upsampled_logits[task_name] = F.interpolate(
                logit_tensor,
                size=original_size,
                mode='bilinear',
                align_corners=False
            )
        
        return upsampled_logits