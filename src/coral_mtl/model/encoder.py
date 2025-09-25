import warnings
from typing import List, Optional

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


def _normalize_encoder_name(name: str) -> str:
    """Convert backbone identifiers to segmentation-models-pytorch compatible names."""
    if name.startswith("nvidia/"):
        name = name.split("/", 1)[1]
    return name.replace("-", "_")

class SegFormerEncoder(nn.Module):
    """
    An encoder wrapper using the segmentation-models-pytorch library.

    This class loads a pre-trained encoder from smp and provides a simple
    interface to extract multi-scale feature maps from an input image.
    """
    def __init__(
        self,
        name: str = "mit_b2",
        weights: Optional[str] = "imagenet",
        depth: int = 5,
    ):
        """
        Initializes the smp.Encoder.

        Args:
            name (str): The name of the encoder architecture (e.g., "mit_b2", "resnet34").
            pretrained (str): The dataset for pre-trained weights (e.g., "imagenet").
            depth (int): The number of stages to use in the encoder.
        """
        super().__init__()

        normalized_name = _normalize_encoder_name(name)
        self.original_name = name
        self.normalized_name = normalized_name

        try:
            self.encoder = smp.encoders.get_encoder(
                name=normalized_name,
                in_channels=3,
                depth=depth,
                weights=weights,
            )
        except (RuntimeError, ValueError) as exc:
            if weights is None:
                raise
            warnings.warn(
                f"Failed to load pretrained weights '{weights}' for encoder '{name}': {exc}. "
                "Falling back to randomly initialised weights.",
                RuntimeWarning,
            )
            self.encoder = smp.encoders.get_encoder(
                name=normalized_name,
                in_channels=3,
                depth=depth,
                weights=None,
            )
        # The first channel is the input, and the last is for the final classification layer, which we don't use.
        # The decoder expects features from 4 stages.
        self._out_channels = self.encoder.out_channels
        self._decoder_out_channels = self._out_channels[-4:]

    @property
    def channels(self) -> List[int]:
        """
        Provides the number of output channels for each feature map from the encoder.
        """
        return self._out_channels

    @property
    def decoder_channels(self) -> List[int]:
        """Channels corresponding to the four feature maps consumed by decoders."""
        return self._decoder_out_channels

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Performs the forward pass to extract feature maps.

        Args:
            x (torch.Tensor): The input image tensor of shape (B, 3, H, W).

        Returns:
            List[torch.Tensor]: A list of feature maps from the encoder's stages.
        """
        features = self.encoder(x)
        return features