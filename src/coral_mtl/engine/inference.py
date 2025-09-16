import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm
from typing import Dict, List, Tuple, Union

class SlidingWindowInferrer:
    """
    Implements a sliding window inference engine for semantic segmentation.

    This class handles the logic of tiling a large image into overlapping patches,
    running a model on these patches in batches, and stitching the results
    back together into a seamless, full-resolution prediction map.

    This implementation directly follows the specification in Section 8.3, Component A.
    """

    def __init__(self,
                 model: nn.Module,
                 patch_size: int,
                 stride: int,
                 device: Union[str, torch.device],
                 batch_size: int = 1):
        """
        Initializes the SlidingWindowInferrer.

        Args:
            model (nn.Module): The PyTorch model, already loaded with weights and in eval() mode.
            patch_size (int): The height and width of the square patches (e.g., 512).
            stride (int): The step size for the sliding window (e.g., 256 for 50% overlap).
            device (Union[str, torch.device]): The device to perform inference on (e.g., 'cuda:0').
            batch_size (int): The number of patches to process in a single forward pass.
        """
        self.model = model
        self.patch_size = patch_size
        self.stride = stride
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.model.to(self.device)
        self.model.eval()

    def predict(self, full_image_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Performs inference on a single large image using the sliding window strategy.

        Args:
            full_image_tensor (torch.Tensor): A single large image tensor of shape (C, H, W).

        Returns:
            Dict[str, torch.Tensor]: A dictionary where keys are task names (e.g., 'genus')
                                     and values are the final, full-resolution logit tensors
                                     of shape (Num_Classes, H, W).
        """
        # Ensure model is in eval mode and no gradients are computed
        # Spec Section 8.2.1: Switch to Eval Mode
        with torch.no_grad():
            return self._execute_inference(full_image_tensor)

    def _execute_inference(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        # --- Step 1: Padding (Spec Section 8.3, predict method, step 1) ---
        original_h, original_w = image.shape[-2:]
        padded_image, (pad_h, pad_w) = self._pad_image(image)
        padded_h, padded_w = padded_image.shape[-2:]
        
        # --- Step 2: Patch Unfolding (Spec Section 8.3, predict method, step 2) ---
        patches, patch_coords = self._extract_patches(padded_image)
        num_patches = patches.shape[0]

        # --- Step 3: Batched Inference Loop (Spec Section 8.3, predict method, step 3) ---
        all_patch_logits = {}
        for i in tqdm(range(0, num_patches, self.batch_size), desc="Sliding Window Inference"):
            batch_patches = patches[i:i + self.batch_size].to(self.device)
            
            model_output = self.model(batch_patches)
            
            # Standardize model output to handle both MTL and non-MTL cases
            if isinstance(model_output, torch.Tensor):
                model_output = {'segmentation': model_output}
            
            for task_name, logits in model_output.items():
                if task_name not in all_patch_logits:
                    all_patch_logits[task_name] = []
                all_patch_logits[task_name].append(logits.cpu())

        for task_name, logits_list in all_patch_logits.items():
            all_patch_logits[task_name] = torch.cat(logits_list, dim=0)

        # --- Step 4: Stitching (Spec Section 8.3, predict method, step 4) ---
        final_logits = self._stitch_patches(all_patch_logits, patch_coords,
                                            (padded_h, padded_w))

        # --- Step 5: Cropping (Spec Section 8.3, predict method, step 5) ---
        for task_name, logits_tensor in final_logits.items():
            final_logits[task_name] = logits_tensor[..., :original_h, :original_w]
            
        return final_logits

    def _pad_image(self, image: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Pads the image to be divisible by the stride."""
        c, h, w = image.shape
        
        pad_h = (self.stride - (h - self.patch_size) % self.stride) % self.stride
        pad_w = (self.stride - (w - self.patch_size) % self.stride) % self.stride

        # Use 'reflect' mode to minimize edge artifacts, as specified
        padded_image = F.pad(image, (0, pad_w, 0, pad_h), mode='reflect')
        return padded_image, (pad_h, pad_w)

    def _extract_patches(self, padded_image: torch.Tensor) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        """Efficiently extracts patches and their coordinates using unfold."""
        c, h_pad, w_pad = padded_image.shape
        
        # This is a highly efficient way to create views of all patches
        patches = padded_image.unfold(1, self.patch_size, self.stride).unfold(2, self.patch_size, self.stride)
        # patches shape: (C, num_patches_h, num_patches_w, patch_size, patch_size)
        
        # Reshape into a batch of patches: (N, C, patch_size, patch_size)
        patches = patches.permute(1, 2, 0, 3, 4).contiguous()
        patches = patches.view(-1, c, self.patch_size, self.patch_size)
        
        # Generate corresponding coordinates for stitching
        patch_coords = []
        for y in range(0, h_pad - self.patch_size + 1, self.stride):
            for x in range(0, w_pad - self.patch_size + 1, self.stride):
                patch_coords.append((y, x))
                
        return patches, patch_coords

    def _stitch_patches(self, all_patch_logits: Dict[str, torch.Tensor],
                        patch_coords: list, padded_dims: tuple) -> Dict[str, torch.Tensor]:
        """Stitches patch logits back into full-sized logit maps by averaging overlaps."""
        padded_h, padded_w = padded_dims
        final_logits = {}

        # Initialize accumulators on the CPU
        # Spec: Initialize two dictionaries: logit_accumulators and count_accumulators
        for task_name, example_logits in all_patch_logits.items():
            num_classes = example_logits.shape[1]
            logit_accumulator = torch.zeros((num_classes, padded_h, padded_w), dtype=torch.float32)
            count_accumulator = torch.zeros((padded_h, padded_w), dtype=torch.float32)

            for i, (y, x) in enumerate(patch_coords):
                patch_logit = example_logits[i]
                
                # Add logits to the accumulator
                logit_accumulator[:, y:y+self.patch_size, x:x+self.patch_size] += patch_logit
                
                # Increment the count for the overlapping region
                count_accumulator[y:y+self.patch_size, x:x+self.patch_size] += 1
            
            # Average the logits where patches overlapped
            # Spec: element-wise division with a small epsilon
            epsilon = 1e-8
            final_logits[task_name] = logit_accumulator / (count_accumulator + epsilon)
            
        return final_logits