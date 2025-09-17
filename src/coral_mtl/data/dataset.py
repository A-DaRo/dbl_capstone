import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from typing import Dict, Optional, List, Tuple
from abc import ABC, abstractmethod
from pathlib import Path
import yaml
from datasets import load_dataset

# Correctly import from within the same package
from .augmentations import SegmentationAugmentation
from torchvision.transforms import v2

def create_lookup_table(task_mapping: Dict[int, List[int]], num_raw_classes: int = 40) -> np.ndarray:
    """Creates a NumPy array to efficiently map raw class IDs to new task-specific IDs."""
    lookup_table = np.zeros(num_raw_classes, dtype=np.int64)
    for new_id, old_ids_list in task_mapping.items():
        lookup_table[old_ids_list] = new_id
    return lookup_table


class AbstractCoralscapesDataset(Dataset, ABC):
    """
    Abstract Base Class for Coralscapes datasets.
    Handles data loading from either Hugging Face Hub or local file systems.
    Subclasses are responsible for implementing the label transformation logic.
    """
    def __init__(self,
                 split: str = 'train',
                 augmentations: Optional[SegmentationAugmentation] = None,
                 patch_size: int = 512,
                 hf_dataset_name: Optional[str] = None,
                 data_root_path: Optional[str] = None,
                 pds_train_path: Optional[str] = None):
        
        self.split = split
        self.augmentations = augmentations
        self.patch_size = patch_size
        self.hf_split_dataset = None
        self.file_paths: List[Tuple[Path, Path]] = []

        if hf_dataset_name:
            print(f"Loading dataset '{hf_dataset_name}' from Hugging Face Hub.")
            self.hf_split_dataset = load_dataset(hf_dataset_name)[self.split]
        else:
            print("Loading dataset from local file paths.")
            self._load_local_paths(data_root_path, pds_train_path)
            
        if self.augmentations is None:
            self.default_transform = v2.Compose([
                v2.Resize((self.patch_size, self.patch_size), antialias=True),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def _load_local_paths(self, data_root_path: Optional[str], pds_train_path: Optional[str]):
        """
        Populates self.file_paths with (image_path, mask_path) tuples from local directories.
        Prioritizes the PDS path for training, but falls back to the raw data path if needed.
        """
        # As per Spec Section 8.1, the PDS dataset is the primary source for training.
        if self.split == 'train' and pds_train_path:
            print(f"Loading 'train' split from prioritized PDS path: {pds_train_path}")
            base_path = Path(pds_train_path)
            image_dir = base_path / "images"
            mask_dir = base_path / "masks"
            image_files = sorted(list(image_dir.glob("*.png")))
            self.file_paths = [(p, mask_dir / p.name) for p in image_files]
        
        # Fallback for train, or standard path for val/test.
        elif self.split in ['train', 'val', 'test']:
            if not data_root_path:
                raise ValueError(f"data_root_path must be provided for '{self.split}' split when not using PDS path.")
            
            print(f"Loading '{self.split}' split from data_root_path: {data_root_path}")
            base_path = Path(data_root_path)
            image_base_dir = base_path / "leftImg8bit" / self.split
            mask_base_dir = base_path / "gtFine" / self.split
            
            image_files = sorted(list(image_base_dir.glob("**/*_leftImg8bit.png")))
            for img_path in image_files:
                identifier = img_path.name.replace('_leftImg8bit.png', '')
                mask_path = mask_base_dir / img_path.relative_to(image_base_dir).parent / f"{identifier}_gtFine.png"
                if mask_path.exists():
                    self.file_paths.append((img_path, mask_path))
        else:
            raise ValueError(f"Invalid split '{self.split}'. Must be one of ['train', 'val', 'test'].")

        if not self.file_paths:
            raise FileNotFoundError(f"No image/mask pairs found for split '{self.split}' in the provided paths.")
        print(f"Found {len(self.file_paths)} pairs for split '{self.split}'.")

    def __len__(self) -> int:
        return len(self.hf_split_dataset) if self.hf_split_dataset else len(self.file_paths)

    def _load_data(self, idx: int) -> Tuple[Image.Image, np.ndarray]:
        """Loads a single raw image and mask from the configured data source."""
        if self.hf_split_dataset:
            example = self.hf_split_dataset[idx]
            original_image = example['image']
            raw_label_mask = np.array(example['label'])
        else:
            img_path, mask_path = self.file_paths[idx]
            original_image = Image.open(img_path).convert("RGB")
            raw_label_mask = np.array(Image.open(mask_path))
        return original_image, raw_label_mask

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, any]:
        raise NotImplementedError


class CoralscapesMTLDataset(AbstractCoralscapesDataset):
    """
    A PyTorch Dataset for CoralScapes that produces multiple masks for Multi-Task Learning,
    as specified in Section 2.1 of project_specification.md.
    """
    def __init__(self,
                 task_definitions: Dict[str, dict],
                 **kwargs):
        super().__init__(**kwargs)

        self.task_definitions = task_definitions

        self.lookup_tables = {
            task_name: create_lookup_table(task_info["mapping"])
            for task_name, task_info in self.task_definitions.items()
        }

    def __getitem__(self, idx: int) -> Dict[str, any]:
        original_image, raw_label_mask = self._load_data(idx)

        target_masks_pil = {
            task_name: Image.fromarray(table[raw_label_mask].astype(np.uint8))
            for task_name, table in self.lookup_tables.items()
        }

        if self.augmentations:
            final_image, final_masks = self.augmentations(original_image, target_masks_pil)
        else:
            final_image = self.default_transform(original_image)
            final_masks = {
                key: torch.from_numpy(np.array(v2.functional.resize(
                    mask_pil, (self.patch_size, self.patch_size), interpolation=v2.InterpolationMode.NEAREST
                ))).long()
                for key, mask_pil in target_masks_pil.items()
            }
            
        return {'image': final_image, 'masks': final_masks}


class CoralscapesDataset(AbstractCoralscapesDataset):
    """
    A general-purpose PyTorch Dataset for CoralScapes that produces a single,
    flattened output mask based on an optional remapping configuration file.
    """
    def __init__(self,
                 task_definitions: Optional[Dict[str, dict]] = None,
                 num_original_classes: int = 40,
                 **kwargs):
        super().__init__(**kwargs)

        self.lookup_table = self._create_flattened_lookup_table(task_definitions, num_original_classes)

    def _create_flattened_lookup_table(self, task_definitions: Optional[Dict[str, dict]], num_classes: int) -> np.ndarray:
        if task_definitions is None:
            return np.arange(num_classes, dtype=np.int64)

        lookup_table = np.zeros(num_classes, dtype=np.int64)
        all_new_ids = set()
        for task_name, details in task_definitions.items():
            for new_id_str, old_ids_list in details.get('mapping', {}).items():
                new_id = int(new_id_str)
                if new_id in all_new_ids and lookup_table[old_ids_list[0]] != new_id:
                     print(f"Warning: new_id {new_id} is being reused for task '{task_name}'. Ensure this is intentional.")
                all_new_ids.add(new_id)
                lookup_table[old_ids_list] = new_id
        return lookup_table

    def __getitem__(self, idx: int) -> Dict[str, any]:
        original_image, raw_label_mask = self._load_data(idx)

        remapped_mask_array = self.lookup_table[raw_label_mask].astype(np.uint8)
        target_mask_pil = Image.fromarray(remapped_mask_array)

        if self.augmentations:
            final_image, final_masks_dict = self.augmentations(original_image, {'mask': target_mask_pil})
            final_mask = final_masks_dict['mask']
        else:
            final_image = self.default_transform(original_image)
            final_mask = torch.from_numpy(np.array(v2.functional.resize(
                target_mask_pil, (self.patch_size, self.patch_size), interpolation=v2.InterpolationMode.NEAREST
            ))).long()

        return {'image': final_image, 'mask': final_mask}