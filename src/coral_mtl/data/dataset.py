import torch
from torch.utils.data import Dataset
import numpy as np
from datasets import Dataset as HFDataset
from PIL import Image
from typing import Dict, Optional

import yaml

# Correctly import from within the same package
from .augmentations import SegmentationAugmentation
from torchvision.transforms import v2


def create_lookup_table(task_mapping, num_raw_classes=40):
    """Creates a NumPy array to efficiently map raw class IDs to new task-specific IDs."""
    lookup_table = np.zeros(num_raw_classes, dtype=np.int64)
    for new_id, old_ids_list in task_mapping.items():
        lookup_table[old_ids_list] = new_id
    return lookup_table

class CoralscapesMTLDataset(Dataset):
    """
    A standardized PyTorch Dataset for the CoralScapes dataset with multi-task learning labels.
    """
    def __init__(self,
                 hf_dataset: 'HFDataset',
                 task_definitions_path: str,
                 split: str = 'train',
                 augmentations: Optional[SegmentationAugmentation] = None,
                 patch_size: int = 512):
        self.dataset_split = hf_dataset[split]
        self.augmentations = augmentations
        self.patch_size = patch_size
        
        with open(task_definitions_path, 'r') as f:
            self.task_definitions = yaml.safe_load(f)

        self.lookup_tables = {
            task_name: create_lookup_table(task_info["mapping"])
            for task_name, task_info in self.task_definitions.items()
        }

        if self.augmentations is None:
            self.default_transform = v2.Compose([
                v2.Resize((self.patch_size, self.patch_size), antialias=True),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self) -> int:
        return len(self.dataset_split)

    def __getitem__(self, idx: int) -> Dict[str, any]:
        example = self.dataset_split[idx]
        original_image = example['image']
        raw_label_mask = np.array(example['label'])

        target_masks_pil = {
            task_name: Image.fromarray(table[raw_label_mask].astype(np.uint8))
            for task_name, table in self.lookup_tables.items()
        }

        if self.augmentations:
            final_image, final_masks = self.augmentations(original_image, target_masks_pil)
        else:
            final_image = self.default_transform(original_image)
            resized_masks = {
                key: torch.from_numpy(np.array(v2.functional.resize(
                    mask_pil, (self.patch_size, self.patch_size), interpolation=v2.InterpolationMode.NEAREST
                ))).long()
                for key, mask_pil in target_masks_pil.items()
            }
            final_masks = resized_masks
            
        return {'image': final_image, 'masks': final_masks}
    


class CoralscapesDataset(Dataset):
    """
    A general-purpose PyTorch Dataset for CoralScapes, producing a single output mask.
    - If a task_definitions_path is provided, it flattens all specified classes
      into a single label space.
    - If no config is provided, it uses the original 39 class labels (identity mapping).
    """
    def __init__(self,
                 hf_dataset: 'HFDataset',
                 task_definitions_path: Optional[str] = None,
                 split: str = 'train',
                 augmentations: Optional[SegmentationAugmentation] = None,
                 patch_size: int = 512,
                 num_original_classes: int = 40):
        self.dataset_split = hf_dataset[split]
        self.augmentations = augmentations
        self.patch_size = patch_size
        self.lookup_table = self._create_flattened_lookup_table(task_definitions_path, num_original_classes)

        if self.augmentations is None:
            self.default_transform = v2.Compose([
                v2.Resize((self.patch_size, self.patch_size), antialias=True),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def _create_flattened_lookup_table(self, task_definitions_path: Optional[str], num_classes: int) -> np.ndarray:
        """Helper to create a single lookup table from an optional YAML file."""
        if task_definitions_path is None:
            # Default case: 1-to-1 mapping of original labels
            return np.arange(num_classes, dtype=np.int64)

        with open(task_definitions_path, 'r') as f:
            task_definitions = yaml.safe_load(f)

        # Flatten all mappings from all tasks into a single LUT
        lookup_table = np.zeros(num_classes, dtype=np.int64)
        all_new_ids = set()
        for task_name, details in task_definitions.items():
            for new_id_str, old_ids_list in details.get('mapping', {}).items():
                new_id = int(new_id_str)
                # Simple collision check, as requested
                if new_id in all_new_ids and lookup_table[old_ids_list[0]] != new_id:
                     print(f"Warning: new_id {new_id} is being reused for task '{task_name}'. Ensure this is intentional.")
                all_new_ids.add(new_id)
                lookup_table[old_ids_list] = new_id
        return lookup_table

    def __len__(self) -> int:
        return len(self.dataset_split)

    def __getitem__(self, idx: int) -> Dict[str, any]:
        example = self.dataset_split[idx]
        original_image = example['image']
        raw_label_mask = np.array(example['label'])

        # Create a single remapped mask using the flattened lookup table
        remapped_mask_array = self.lookup_table[raw_label_mask].astype(np.uint8)
        target_mask_pil = Image.fromarray(remapped_mask_array)

        if self.augmentations:
            # The augmentation pipeline should handle a single mask input
            # by wrapping it in a dict, as specified in the request.
            final_image, final_masks_dict = self.augmentations(original_image, {'mask': target_mask_pil})
            final_mask = final_masks_dict['mask']
        else:
            final_image = self.default_transform(original_image)
            final_mask = torch.from_numpy(np.array(v2.functional.resize(
                target_mask_pil, (self.patch_size, self.patch_size), interpolation=v2.InterpolationMode.NEAREST
            ))).long()

        # Return a dictionary with the singular 'mask' key for non-MTL models
        return {'image': final_image, 'mask': final_mask}