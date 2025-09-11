# File: model.py
# Description: Assembles the full CoralMTLSegFormer model.

import torch
import torch.nn as nn
from typing import Dict

from transformers import SegformerModel

# Assumes decoder.py is in the same directory
from MTL_approach.decoder import CoralMTLDecoder

class CoralMTLSegFormerHuggingFace(nn.Module):
    """
    The complete MTL model. Uses OOP to compose the encoder, decoder, and heads.
    """
    def __init__(
        self,
        n_genus_classes: int,
        n_health_classes: int,
        encoder_name: str = "nvidia/mit-b5",
        decoder_segmentation_channels: int = 768,
    ):
        super().__init__()
        
        self.encoder = SegformerModel.from_pretrained(encoder_name)
        
        self.decoder = CoralMTLDecoder(
            encoder_channels=self.encoder.config.hidden_sizes,
            segmentation_channels=decoder_segmentation_channels,
        )
        
        head_in_channels = decoder_segmentation_channels * len(self.encoder.config.hidden_sizes)

        # Prediction heads for each task
        self.genus_head = nn.Sequential(
            nn.Conv2d(head_in_channels, n_genus_classes, kernel_size=1),
            nn.UpsamplingBilinear2d(scale_factor=4)
        )
        self.health_head = nn.Sequential(
            nn.Conv2d(head_in_channels, n_health_classes, kernel_size=1),
            nn.UpsamplingBilinear2d(scale_factor=4)
        )

    def forward(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        The forward pass is a high-level composition of the model's components.
        """
        # Encoder -> Decoder -> Heads
        features = self.encoder(pixel_values, output_hidden_states=True).hidden_states
        final_f_genus, final_f_health = self.decoder(features)
        
        mask_genus = self.genus_head(final_f_genus)
        mask_health = self.health_head(final_f_health)

        return {"genus": mask_genus, "health": mask_health}