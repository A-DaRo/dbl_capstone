import torch
from torch.utils.data import Dataset
import numpy as np
from datasets import load_dataset
from PIL import Image
from typing import Dict, Optional

# Import the augmentation pipeline and v2 transforms
from augmentations import SegmentationAugmentation
from torchvision.transforms import v2
# --- 1. Definition of Multi-Task Learning (MTL) Mappings ---
# Based on the analysis of functional relevance, semantic cohesion, and data-driven insights,
# we define the mappings from the 39 original classes to our 2 primary and 3 auxiliary tasks.

# The background class is 0 for all tasks. Other classes start from 1.

# A dictionary to hold all task definitions for clarity and scalability.
# Each task has a 'mapping' (new_id -> [old_ids]) and an 'id2label' map.
TASK_DEFINITIONS = {
    "genus": {
        "id2label": {
            0: "background", 1: "other_coral", 2: "massive_meandering", 3: "branching",
            4: "acropora", 5: "table_acropora", 6: "pocillopora", 7: "meandering", 8: "stylophora",
        },
        "mapping": {
            1: [6],             # other_coral
            2: [16, 17, 23],    # massive_meandering
            3: [19, 20, 22],    # branching
            4: [25],            # acropora
            5: [28, 32],        # table_acropora
            6: [31],            # pocillopora
            7: [33, 36, 37],    # meandering
            8: [34],            # stylophora
        },
    },
    "health": {
        "id2label": {0: "background", 1: "alive", 2: "bleached", 3: "dead"},
        "mapping": {
            1: [6, 17, 22, 25, 28, 31, 34, 36],  # alive
            2: [16, 19, 33],                   # bleached
            3: [20, 23, 32, 37],                   # dead
        },
    },
    "fish": {
        "id2label": {0: "background", 1: "fish"},
        "mapping": {
            1: [9], # fish
        },
    },
    "human_artifacts": {
        "id2label": {0: "background", 1: "artifact"},
        "mapping": {
            1: [7, 8, 15], # human, transect tools, transect line
        },
    },
    "substrate": {
        "id2label": {0: "background", 1: "sand", 2: "rock_rubble", 3: "algae_covered"},
        "mapping": {
            1: [5],         # sand
            2: [12, 18],    # unknown hard substrate, rubble
            3: [10],        # algae covered substrate
        },
    },
}

# --- 2. Helper Function to Create Efficient Lookup Tables ---

def create_lookup_table(task_mapping, num_raw_classes=40):
    """
    Creates a NumPy array to efficiently map raw class IDs to new task-specific IDs.

    Args:
        task_mapping (dict): A dictionary mapping new class IDs to a list of raw class IDs.
        num_raw_classes (int): The total number of raw classes (default is 40 for 0-39).

    Returns:
        np.ndarray: A 1D array where the index is the raw class ID and the value
                    is the new task-specific class ID.
    """
    # Initialize a table with zeros (background class)
    lookup_table = np.zeros(num_raw_classes, dtype=np.int64)
    # Populate the table based on the mapping
    for new_id, old_ids_list in task_mapping.items():
        lookup_table[old_ids_list] = new_id
    return lookup_table
# --- 2. The Standardized Dataset Class ---

class CoralscapesMTLDataset(Dataset):
    """
    A standardized PyTorch Dataset for the CoralScapes dataset with multi-task learning labels.

    This class handles:
    - Loading images and raw segmentation masks from the Hugging Face dataset.
    - Applying the predefined mappings to convert raw masks into five separate MTL task masks.
    - Applying a sophisticated augmentation pipeline for training, or a simple
      ToTensor/Normalize transform for validation/testing.
    """
    def __init__(self,
                 hf_dataset: 'datasets.Dataset',
                 split: str = 'train',
                 augmentations: Optional[SegmentationAugmentation] = None,
                 patch_size: int = 512):
        """
        Args:
            hf_dataset (datasets.Dataset): The loaded Hugging Face dataset object.
            split (str): The dataset split to use (e.g., 'train', 'validation', 'test').
            augmentations (Optional[SegmentationAugmentation]): An augmentation object.
                If provided (for training), it will be applied. If None (for validation/testing),
                a default tensor conversion and normalization will be applied.
        """
        self.dataset_split = hf_dataset[split]
        self.augmentations = augmentations
        self.patch_size = patch_size

        # Create efficient lookup tables for all tasks once during initialization
        self.lookup_tables = {
            task_name: create_lookup_table(task_info["mapping"])
            for task_name, task_info in TASK_DEFINITIONS.items()
        }

        # Define a default transformation for validation/testing if no augmentations are provided
        if self.augmentations is None:
            self.default_transform = v2.Compose([
                v2.Resize((self.patch_size, self.patch_size), antialias=True),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])


    def __len__(self) -> int:
        """Returns the number of samples in the dataset split."""
        return len(self.dataset_split)

    def __getitem__(self, idx: int) -> Dict[str, any]:
        """
        Fetches a single sample, applies mappings and transforms, and returns it.

        Returns:
            Dict[str, any]: A dictionary containing:
                - 'image': The transformed and normalized image tensor.
                - 'masks': A dictionary of transformed mask tensors for each task.
        """
        # 1. Load the original image and raw label mask
        example = self.dataset_split[idx]
        original_image = example['image']
        raw_label_mask = np.array(example['label'])

        # 2. Apply MTL mappings to create a dictionary of task-specific masks
        target_masks_pil = {}
        for task_name, table in self.lookup_tables.items():
            # Apply vectorized mapping
            new_mask_np = table[raw_label_mask]
            # Convert to PIL Image for the augmentation pipeline
            target_masks_pil[task_name] = Image.fromarray(new_mask_np.astype(np.uint8))

        # 3. Apply augmentations (for training) or default transforms (for validation/testing)
        if self.augmentations:
            final_image, final_masks = self.augmentations(original_image, target_masks_pil)
        else:
            final_image = self.default_transform(original_image)
            resized_masks = {}
            for key, mask_pil in target_masks_pil.items():
                # F.resize expects a PIL image, interpolation NEAREST for masks
                resized_mask = v2.functional.resize(mask_pil, (self.patch_size, self.patch_size), interpolation=v2.InterpolationMode.NEAREST)
                resized_masks[key] = torch.from_numpy(np.array(resized_mask)).long()

            final_masks = resized_masks
            
        return {'image': final_image, 'masks': final_masks}


# --- 3. Main Sanity Check ---

if __name__ == "__main__":
    print("--- Running Sanity Check for CoralscapesMTLDataset ---")

    # 1. Load the real Hugging Face dataset
    # NOTE: This will download the full dataset (~5.85 GB) if not already cached.
    print("--- Loading CoralScapes Dataset (this may take a while) ---")
    hf_dataset = load_dataset("EPFL-ECEO/coralscapes")

    # 2. Instantiate the augmentation pipeline for the training set
    train_augs = SegmentationAugmentation(patch_size=512)

    # 3. Create a training dataset instance (with augmentations)
    print("\nInstantiating training dataset (with augmentations)...")
    train_dataset = CoralscapesMTLDataset(hf_dataset=hf_dataset, split='train', augmentations=train_augs)
    
    # 4. Create a validation dataset instance (without augmentations)
    print("Instantiating validation dataset (without augmentations)...")
    val_dataset = CoralscapesMTLDataset(hf_dataset=hf_dataset, split='validation', augmentations=None)

    # 5. Fetch and inspect a sample from each
    print(f"\nTotal training samples: {len(train_dataset)}")
    print(f"Total validation samples: {len(val_dataset)}")
    
    print("\n--- Checking Training Sample (Index 0) ---")
    train_sample = train_dataset[0]
    train_image = train_sample['image']
    train_masks = train_sample['masks']
    
    print(f"Image Tensor Shape: {train_image.shape}, Dtype: {train_image.dtype}")
    assert train_image.dtype == torch.float32
    for task, mask in train_masks.items():
        print(f"  - Mask '{task}': Shape: {mask.shape}, Dtype: {mask.dtype}")
        assert mask.dtype == torch.long

    print("\n--- Checking Validation Sample (Index 0) ---")
    val_sample = val_dataset[0]
    val_image = val_sample['image']
    val_masks = val_sample['masks']

    print(f"Image Tensor Shape: {val_image.shape}, Dtype: {val_image.dtype}")
    assert val_image.dtype == torch.float32
    for task, mask in val_masks.items():
        print(f"  - Mask '{task}': Shape: {mask.shape}, Dtype: {mask.dtype}")
        assert mask.dtype == torch.long
        
    # 6. Verify that augmentations were applied
    # The augmented image should be different from the simple normalized one.
    # Note: We can't directly compare train_sample[0] and val_sample[0] as they
    # come from different splits. We test the augmentation by processing the same
    # sample with and without the augmentation pipeline.
    
    # Get a raw sample from the validation set
    raw_sample = val_dataset.dataset_split[0]
    raw_image = raw_sample['image']
    raw_mask_np = np.array(raw_sample['label'])
    raw_masks_pil = {
        task: Image.fromarray(table[raw_mask_np].astype(np.uint8))
        for task, table in val_dataset.lookup_tables.items()
    }

    # Process it with augmentations
    augmented_image_test, _ = train_augs(raw_image, raw_masks_pil)
    # Process it without augmentations (like in the val_dataset)
    non_augmented_image_test = val_dataset.default_transform(raw_image)
    # Resize val image to train size for fair comparison
    resized_val_image = v2.Resize((512, 512))(non_augmented_image_test)

    assert not torch.allclose(augmented_image_test, resized_val_image), "Training image was not augmented!"
    print("\nSuccessfully verified that training sample is augmented and validation sample is not.")

    print("\nSanity check passed! The Dataset class is ready for training.")