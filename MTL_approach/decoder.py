# File: decoder.py
# Description: Defines the novel Coral-MTL decoder for explicit feature exchange.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

# Assumes attention_module.py is in the same directory
from MTL_approach.attention_module import MultiTaskCrossAttentionModule

class CoralMTLDecoder(nn.Module):
    """
    The Coral-MTL Decoder, which orchestrates the feature processing pipeline.
    This component uses OOP to manage its layers.
    """
    def __init__(self, encoder_channels: List[int], segmentation_channels: int):
        super().__init__()
        self.mlps = nn.ModuleList([
            nn.Linear(in_features, segmentation_channels) for in_features in encoder_channels
        ])
        fused_channels = segmentation_channels * len(encoder_channels)
        self.cross_attention = MultiTaskCrossAttentionModule(in_channels=fused_channels)

    def forward(self, features: List[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Defines the forward pass as a clear, functional sequence of operations.
        
        Args:
            features: List of hierarchical feature maps from the encoder [F1, F2, F3, F4].

        Returns:
            A tuple of final refined feature tensors for (genus, health) tasks.
        """
        # Operation 1: Shared Initial Feature Processing
        unified_features = []
        for i, feature_map in enumerate(features):
            B, C, H, W = feature_map.shape
            # Reshape, project, reshape back
            unified_map = self.mlps[i](feature_map.flatten(2).transpose(1, 2)) \
                              .transpose(1, 2).reshape(B, -1, H, W)
            unified_features.append(unified_map)

        # Operation 2: Task-Specific Branching and Upsampling
        target_size = unified_features[0].shape[2:]
        upsampled_features = [
            F.interpolate(f, size=target_size, mode='bilinear', align_corners=False)
            for f in unified_features
        ]
        F_fused = torch.cat(upsampled_features, dim=1)
        F_genus, F_health = F_fused, F_fused # Branching

        # Operation 3: Multi-Task Cross-Attention
        Enriched_F_genus, Enriched_F_health = self.cross_attention(F_genus, F_health)

        # Operation 4: Feature Integration (Residual Connection)
        Final_F_genus = F_genus + Enriched_F_genus
        Final_F_health = F_health + Enriched_F_health

        return Final_F_genus, Final_F_health