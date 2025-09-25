import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import os
from typing import Dict, Optional, List, Tuple, Any
from abc import ABC, abstractmethod
from pathlib import Path
from datasets import load_dataset

# Correctly import from within the same package
from .augmentations import SegmentationAugmentation
from torchvision.transforms import v2
from coral_mtl.utils.task_splitter import TaskSplitter, MTLTaskSplitter, BaseTaskSplitter


class AbstractCoralscapesDataset(Dataset, ABC):
    """
    Abstract Base Class for Coralscapes datasets.
    
    Handles data loading from either Hugging Face Hub or local file systems.
    It now returns a richer dictionary per sample, including a unique image_id
    and the original, un-transformed ground truth mask, which are essential for
    the advanced metrics and storage pipeline.
    """
    def __init__(self,
                 split: str,
                 patch_size: int,
                 augmentations: Optional[SegmentationAugmentation] = None,
                 hf_dataset_name: Optional[str] = None,
                 data_root_path: Optional[str] = None,
                 pds_train_path: Optional[str] = None):
        
        # Validate patch_size
        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")
        
        self.split = split
        self.augmentations = augmentations
        self.patch_size = patch_size
        self.hf_split_dataset = None
        self.file_paths: List[Tuple[Path, Path]] = []

        # Unified data source loading
        if hf_dataset_name and not os.path.isdir(hf_dataset_name):
            print(f"Loading dataset '{hf_dataset_name}' from Hugging Face Hub for split '{split}'.")
            self.hf_split_dataset = load_dataset(hf_dataset_name, split=self.split)
        else:
            print(f"Loading dataset from local file paths for split '{split}'.")
            # Use hf_dataset_name as a potential path if it's a directory
            self._load_local_paths(data_root_path or (hf_dataset_name if os.path.isdir(hf_dataset_name) else None), pds_train_path)
            
        # Default transform for validation/testing if no augmentations are provided
        if self.augmentations is None:
            if self.split == 'train':
                # For training, resize to patch_size (in case no augmentations)
                self.default_transform = v2.Compose([
                    v2.Resize((self.patch_size, self.patch_size), antialias=True),
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
            else:
                # For validation/test, keep original resolution for sliding window inference
                self.default_transform = v2.Compose([
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])

    def _load_local_paths(self, data_root_path: Optional[str], pds_train_path: Optional[str]):
        """
        Populates self.file_paths with (image_path, mask_path) tuples from local directories.
        """
        # (Implementation from the prompt is correct and can be reused here)
        # As per Spec Section 8.1, the PDS dataset is the primary source for training.
        if self.split == 'train' and pds_train_path:
            print(f"Loading 'train' split from prioritized PDS path: {pds_train_path}")
            base_path = Path(pds_train_path)
            if not base_path.exists(): raise FileNotFoundError(f"PDS train path does not exist: {pds_train_path}")
            image_dir, mask_dir = base_path / "images", base_path / "masks"
            if not image_dir.exists(): raise FileNotFoundError(f"Images directory not found: {image_dir}")
            if not mask_dir.exists(): raise FileNotFoundError(f"Masks directory not found: {mask_dir}")
            
            image_files = sorted(list(image_dir.glob("*.png")))
            for img_path in image_files:
                mask_path = mask_dir / img_path.name
                if mask_path.exists():
                    self.file_paths.append((img_path, mask_path))

        # Fallback for train, or standard path for val/test.
        elif self.split in ['train', 'validation', 'test']:
            if not data_root_path: raise ValueError(f"data_root_path must be provided for '{self.split}' split.")
            
            print(f"Loading '{self.split}' split from data_root_path: {data_root_path}")
            split_name = 'val' if self.split == 'validation' else self.split
            base_path = Path(data_root_path)
            if not base_path.exists(): raise FileNotFoundError(f"Data root path does not exist: {data_root_path}")
            
            image_base_dir = base_path / "leftImg8bit" / split_name
            mask_base_dir = base_path / "gtFine" / split_name
            if not image_base_dir.exists(): raise FileNotFoundError(f"Image directory not found: {image_base_dir}")
            if not mask_base_dir.exists(): raise FileNotFoundError(f"Mask directory not found: {mask_base_dir}")

            image_files = sorted(list(image_base_dir.glob("**/*_leftImg8bit.png")))
            for img_path in image_files:
                identifier = img_path.name.replace('_leftImg8bit.png', '')
                mask_path = mask_base_dir / img_path.relative_to(image_base_dir).parent / f"{identifier}_gtFine.png"
                if mask_path.exists():
                    self.file_paths.append((img_path, mask_path))
        else:
            raise ValueError(f"Invalid split '{self.split}'. Must be one of ['train', 'validation', 'test'].")

        if not self.file_paths and not self.hf_split_dataset:
            raise FileNotFoundError(f"No image/mask pairs found for split '{self.split}' in the provided paths.")
        print(f"Successfully loaded {len(self)} samples for split '{self.split}'.")

    def __len__(self) -> int:
        return len(self.hf_split_dataset) if self.hf_split_dataset else len(self.file_paths)

    def _load_data(self, idx: int) -> Tuple[Image.Image, np.ndarray, str]:
        """
        Loads a single raw image, mask, and a unique identifier for that sample.
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds for dataset of size {len(self)}")
        
        try:
            image_id: str = ""
            if self.hf_split_dataset:
                example = self.hf_split_dataset[idx]
                original_image = example['image'].convert("RGB")
                # Handle mask field - try 'label' first, then 'mask'
                label_data = example.get('label')
                if label_data is not None:
                    raw_label_mask = np.array(label_data)
                else:
                    mask_data = example.get('mask')
                    if mask_data is not None:
                        raw_label_mask = np.array(mask_data)
                    else:
                        raise ValueError(f"No 'label' or 'mask' field found in HuggingFace dataset example at index {idx}")
                image_id = example.get('id', f"hf_{self.split}_{idx}")
            else:
                img_path, mask_path = self.file_paths[idx]
                original_image = Image.open(img_path).convert("RGB")
                raw_label_mask = np.array(Image.open(mask_path))
                image_id = img_path.name
            
            if raw_label_mask.ndim > 2:
                raw_label_mask = raw_label_mask[..., 0]
                    
            return original_image, raw_label_mask, image_id
            
        except Exception as e:
            raise RuntimeError(f"Failed to load data at index {idx} (ID: {image_id}): {str(e)}")

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        raise NotImplementedError


class CoralscapesMTLDataset(AbstractCoralscapesDataset):
    """
    Dataset for Multi-Task Learning models. It is driven by an `MTLTaskSplitter`
    to produce a dictionary of transformed masks for the loss function, alongside
    the original mask and image_id for evaluation.
    """
    def __init__(self, splitter: MTLTaskSplitter, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(splitter, MTLTaskSplitter):
            raise TypeError("CoralscapesMTLDataset requires an MTLTaskSplitter instance.")
        self.splitter = splitter

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        original_image, raw_label_mask, image_id = self._load_data(idx)
        
        task_masks_np = {}
        for task, details in self.splitter.hierarchical_definitions.items():
            mapping_array = details['ungrouped']['mapping_array']
            clipped_mask = np.clip(raw_label_mask, 0, len(mapping_array) - 1)
            task_masks_np[task] = mapping_array[clipped_mask].astype(np.uint8)
        
        target_masks_pil = {name: Image.fromarray(mask) for name, mask in task_masks_np.items()}

        if self.augmentations:
            final_image, final_masks = self.augmentations(original_image, target_masks_pil)
        else:
            final_image = self.default_transform(original_image)
            final_masks = {}
            for task_name, mask_pil in target_masks_pil.items():
                if self.split == 'train':
                    # For training, resize mask to patch_size
                    final_masks[task_name] = torch.from_numpy(np.array(v2.functional.resize(
                        mask_pil, (self.patch_size, self.patch_size), 
                        interpolation=v2.InterpolationMode.NEAREST
                    ))).long()
                else:
                    # For validation/test, keep mask at original resolution
                    final_masks[task_name] = torch.from_numpy(np.array(mask_pil)).long()

        return {
            'image': final_image,
            'image_id': image_id,
            'original_mask': torch.from_numpy(raw_label_mask.copy()).long(),
            'masks': final_masks  # This is a dict of masks for the MTL loss
        }


class CoralscapesDataset(AbstractCoralscapesDataset):
    """
    Dataset for Baseline (single-head) models. It is driven by a `BaseTaskSplitter`
    to produce a single flattened mask for the loss function, alongside the
    original mask and image_id for fair evaluation.
    """
    def __init__(self, splitter: BaseTaskSplitter, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(splitter, BaseTaskSplitter):
            raise TypeError("CoralscapesDataset requires a BaseTaskSplitter instance.")
        self.splitter = splitter

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        original_image, raw_label_mask, image_id = self._load_data(idx)
        
        mapping_array = self.splitter.flat_mapping_array
        clipped_mask = np.clip(raw_label_mask, 0, len(mapping_array) - 1)
        flattened_mask_np = mapping_array[clipped_mask].astype(np.uint8)
        
        target_mask_pil = Image.fromarray(flattened_mask_np)

        if self.augmentations:
            final_image, final_masks_dict = self.augmentations(original_image, {'mask': target_mask_pil})
            final_mask = final_masks_dict['mask']
        else:
            final_image = self.default_transform(original_image)
            if self.split == 'train':
                # For training, resize mask to patch_size
                final_mask = torch.from_numpy(np.array(v2.functional.resize(
                    target_mask_pil, (self.patch_size, self.patch_size), 
                    interpolation=v2.InterpolationMode.NEAREST
                ))).long()
            else:
                # For validation/test, keep mask at original resolution
                final_mask = torch.from_numpy(flattened_mask_np).long()

        return {
            'image': final_image,
            'image_id': image_id,
            'original_mask': torch.from_numpy(raw_label_mask.copy()).long(),
            'mask': final_mask # This is a single mask for the baseline loss
        }