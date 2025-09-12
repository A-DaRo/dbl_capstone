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

    The pipeline is designed to be used within a PyTorch Dataset's __getitem__ method.
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
                Defaults to a standard set if None.
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
            v2.ToImage(),              # PIL -> Tensor [0,255] -> uint8
            v2.ToDtype(torch.float32, scale=True),  # -> float in [0,1]
            v2.Normalize(mean=imagenet_mean, std=imagenet_std)
        ])

    def __call__(self, image: Image.Image, masks: Dict[str, Image.Image]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Applies the full augmentation pipeline.

        Args:
            image (PIL.Image.Image): The input image.
            masks (Dict[str, PIL.Image.Image]): A dictionary of segmentation masks as PIL Images.

        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
                - The augmented and normalized image tensor (C, H, W).
                - A dictionary of augmented mask tensors (H, W) with dtype=torch.long.
        """
        # --- Step 1: Apply geometric transforms to image and all masks together ---
        inputs = {"image": image, **masks}
        transformed = self.geometric_transforms(inputs)

        aug_image = transformed.pop("image")
        aug_masks = transformed

        # Step 2: color transforms only on image
        aug_image = self.color_transforms(aug_image)

        # Step 3: convert + normalize image
        final_image = self.to_tensor_and_normalize(aug_image)

        # Step 4: convert masks to tensors (long, no scaling)
        final_masks = {
            key: v2.functional.to_image(mask).to(torch.long).squeeze(0)
            for key, mask in aug_masks.items()
        }

        return final_image, final_masks

# --- Sanity Check and Visualization ---
def denormalize(tensor, mean, std):
    """Reverses normalization for visualization."""
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def visualize_augmentations(original_image, original_masks, aug_image, aug_masks):
    """Helper function to plot and compare original vs. augmented data."""
    import matplotlib.pyplot as plt

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    vis_image = denormalize(aug_image, mean, std)
    vis_image = vis_image.permute(1, 2, 0).numpy().clip(0, 1)

    num_masks = len(original_masks)
    fig, axes = plt.subplots(num_masks + 1, 2, figsize=(8, 4 * (num_masks + 1)))
    fig.suptitle("Augmentation Sanity Check", fontsize=16)

    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(vis_image)
    axes[0, 1].set_title("Augmented Image")
    axes[0, 1].axis('off')
    
    for i, task_name in enumerate(original_masks.keys()):
        row = i + 1
        axes[row, 0].imshow(original_masks[task_name], cmap='viridis', interpolation='nearest')
        axes[row, 0].set_title(f"Original Mask: {task_name}")
        axes[row, 0].axis('off')
        
        axes[row, 1].imshow(aug_masks[task_name].numpy(), cmap='viridis', interpolation='nearest')
        axes[row, 1].set_title(f"Augmented Mask: {task_name}")
        axes[row, 1].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == '__main__':
    print("--- Running Sanity Check for SegmentationAugmentation ---")
    
    patch_size = 512
    dummy_np_image = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
    dummy_np_image[:patch_size//2, :patch_size//2, 0] = 255
    dummy_np_image[:patch_size//2, patch_size//2:, 1] = 255
    dummy_np_image[patch_size//2:, :patch_size//2, 2] = 255
    dummy_np_image[patch_size//2:, patch_size//2:, :] = 128
    original_image = Image.fromarray(dummy_np_image)

    genus_mask = np.zeros((patch_size, patch_size), dtype=np.uint8)
    genus_mask[:patch_size//2, :patch_size//2] = 1
    genus_mask[:patch_size//2, patch_size//2:] = 2
    
    health_mask = np.zeros((patch_size, patch_size), dtype=np.uint8)
    health_mask[:patch_size//2, :patch_size//2] = 1
    health_mask[patch_size//2:, :patch_size//2] = 2
    
    original_masks = {
        'genus': Image.fromarray(genus_mask),
        'health': Image.fromarray(health_mask)
    }

    augmentation_pipeline = SegmentationAugmentation(patch_size=patch_size)
    augmented_image, augmented_masks = augmentation_pipeline(original_image, original_masks)
    
    print("\n--- Output Tensor Shapes ---")
    print(f"Augmented Image: {augmented_image.shape}, dtype: {augmented_image.dtype}")
    for task, mask_tensor in augmented_masks.items():
        print(f"Augmented Mask '{task}': {mask_tensor.shape}, dtype: {mask_tensor.dtype}")
        assert mask_tensor.dtype == torch.long
        assert mask_tensor.dim() == 2

    print("\nVisualizing results... Close the plot to continue.")
    visualize_augmentations(original_image, original_masks, augmented_image, augmented_masks)
    
    print("\nSanity check complete. Augmentation pipeline is working correctly.")