import torch
import numpy as np
from PIL import Image
from typing import Dict, Tuple

# Use the new v2 transforms for synchronized image and mask transformations
from torchvision.transforms import v2
import torchvision.transforms.functional as F

class SegmentationAugmentation:
    """
    A comprehensive augmentation pipeline for semantic segmentation tasks using torchvision.transforms.v2.

    This class applies a series of geometric and colorimetric augmentations.
    - Geometric transforms are applied synchronously to both the image and the segmentation masks.
    - Colorimetric transforms are applied only to the image.
    """
    def __init__(self,
                 patch_size: int = 512,
                 crop_scale: Tuple[float, float] = (0.5, 1.0),
                 rotation_degrees: int = 15,
                 jitter_params: Dict[str, float] = None,
                 imagenet_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
                 imagenet_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)):
        """
        Args:
            patch_size (int): The final output size of the image and masks.
            crop_scale (Tuple[float, float]): The range for RandomResizedCrop scale.
            rotation_degrees (int): The range for random rotation (-deg, +deg).
            jitter_params (Dict[str, float], optional): Parameters for ColorJitter.
            imagenet_mean (Tuple[float, float, float]): Mean for normalization.
            imagenet_std (Tuple[float, float, float]): Standard deviation for normalization.
        """
        if jitter_params is None:
            jitter_params = {
                'brightness': 0.2, 'contrast': 0.2, 'saturation': 0.2, 'hue': 0.1
            }

        # Geometric transforms (sync image + masks)
        self.geometric_transforms = v2.Compose([
            v2.RandomResizedCrop(size=(patch_size, patch_size), scale=crop_scale, antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomRotation(degrees=(-rotation_degrees, rotation_degrees)),
        ])

        # Color transforms (image only)
        self.color_transforms = v2.Compose([
            v2.ColorJitter(**jitter_params),
            v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.0)),
        ])

        # Final conversion + normalization
        self.to_tensor_and_normalize = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=imagenet_mean, std=imagenet_std)
        ])

    def __call__(self, image: Image.Image, masks: Dict[str, Image.Image]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Applies the full augmentation pipeline.

        Args:
            image (PIL.Image.Image): The input image.
            masks (Dict[str, PIL.Image.Image]): A dictionary of segmentation masks.

        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
                - The augmented and normalized image tensor (C, H, W).
                - A dictionary of augmented mask tensors (H, W) with dtype=torch.long.
        """
        inputs = {"image": image, **masks}
        transformed = self.geometric_transforms(inputs)

        aug_image = transformed.pop("image")
        aug_masks = transformed

        aug_image = self.color_transforms(aug_image)
        final_image = self.to_tensor_and_normalize(aug_image)

        final_masks = {
            key: v2.functional.to_image(mask).to(torch.long).squeeze(0)
            for key, mask in aug_masks.items()
        }

        return final_image, final_masks