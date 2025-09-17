import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List

from .encoder import SegFormerEncoder
from .decoders import HierarchicalContextAwareDecoder, SegFormerMLPDecoder


class BaselineSegformer(nn.Module):
    """
    A standard SegFormer model for semantic segmentation.

    This model serves as a baseline, directly predicting the 39 monolithic classes
    from the Coralscapes dataset. It uses the same MiT encoder as the CoralMTLModel
    but replaces the complex hierarchical decoder with a standard All-MLP decoder head.
    """
    def __init__(self,
                 encoder_name: str,
                 decoder_channel: int,
                 num_classes: int = 39):
        """
        Args:
            encoder_name (str): The Hugging Face ID for the SegFormer backbone (e.g., 'nvidia/mit-b2').
            decoder_channel (int): The unified channel dimension for the MLP decoder.
            num_classes (int): The total number of output classes. For this project,
                               this is fixed at 39 to match the original dataset.
        """
        super().__init__()
        
        # --- Component 1: Shared Encoder (Spec Section 3.1) ---
        # For a fair comparison, the feature extractor is identical to the MTL model.
        self.encoder = SegFormerEncoder(pretrained_weights_path=encoder_name)

        # --- Component 2: Standard All-MLP Decoder (SegFormer's original design) ---
        # This replaces the custom HierarchicalContextAwareDecoder.
        self.decoder = SegFormerMLPDecoder(
            encoder_channels=self.encoder.channels,
            decoder_channel=decoder_channel
        )

        # --- Component 3: Prediction Head ---
        # A single 1x1 convolution to map the decoder features to the 39 class logits.
        self.prediction_head = nn.Conv2d(decoder_channel, num_classes, kernel_size=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the baseline model.
        
        Args:
            images (torch.Tensor): The input image tensor of shape (B, 3, H, W).

        Returns:
            torch.Tensor: A single output logit tensor of shape (B, num_classes, H, W),
                          upsampled to the original image size. Note that this is NOT
                          a dictionary, unlike the MTL model's output.
        """
        # 1. Get multi-scale features from the encoder
        features: List[torch.Tensor] = self.encoder(images)
        
        # 2. Fuse features using the standard MLP decoder
        fused_features = self.decoder(features) # Shape: (B, C_decoder, H/4, W/4)
        
        # 3. Get logits from the prediction head
        logits_at_quarter_res = self.prediction_head(fused_features) # Shape: (B, 39, H/4, W/4)
        
        # 4. Upsample to original resolution
        original_size = images.shape[-2:]
        upsampled_logits = F.interpolate(
            logits_at_quarter_res,
            size=original_size,
            mode='bilinear',
            align_corners=False
        )
        
        return upsampled_logits

class CoralMTLModel(nn.Module):
    """
    The main multi-task learning model for coral segmentation.

    This class encapsulates the SegFormer encoder and the custom hierarchical
    context-aware decoder into a single, cohesive model.
    """
    def __init__(self, encoder_name: str, decoder_channel: int, num_classes: Dict[str, int], attention_dim: int, primary_tasks: List[str] = ['genus', 'health'], aux_tasks: List[str] = ['fish', 'human_artifacts', 'substrate']):
        """
        Args:
            encoder_name (str): The Hugging Face ID for the SegFormer backbone.
            decoder_channel (int): The unified channel dimension for all decoder streams.
            num_classes (Dict[str, int]): A dictionary mapping task names to their number of classes.
            attention_dim (int): The dimension for the query, key, and value in the attention module.
            primary_tasks (List[str]): List of primary task names.
            aux_tasks (List[str]): List of auxiliary task names.
        """
        super().__init__()
        
        self.encoder = SegFormerEncoder(pretrained_weights_path=encoder_name)
        
        self.decoder = HierarchicalContextAwareDecoder(
            encoder_channels=self.encoder.channels,
            decoder_channel=decoder_channel,
            num_classes=num_classes,
            attention_dim=attention_dim,
            primary_tasks=primary_tasks,
            aux_tasks=aux_tasks
        )

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Defines the forward pass of the model.
        
        Args:
            images (torch.Tensor): The input image tensor of shape (B, 3, H, W).

        Returns:
            Dict[str, torch.Tensor]: A dictionary of output logits for each task,
                                     upsampled to the original image size.
        """
        features: List[torch.Tensor] = self.encoder(images)
        logits_at_quarter_res = self.decoder(features)
        
        original_size = images.shape[-2:]
        
        upsampled_logits = {
            task_name: F.interpolate(
                logit_tensor,
                size=original_size,
                mode='bilinear',
                align_corners=False
            )
            for task_name, logit_tensor in logits_at_quarter_res.items()
        }
        
        return upsampled_logits