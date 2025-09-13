import torch
from torch.utils.data import Dataset
import numpy as np
from datasets import Dataset as HFDataset
from PIL import Image
from typing import Dict, Optional

# Correctly import from within the same package
from .augmentations import SegmentationAugmentation
from torchvision.transforms import v2

TASK_DEFINITIONS = {
    "genus": {
        "id2label": {
            0: "unlabeled", 1: "other_coral", 2: "massive_meandering", 3: "branching",
            4: "acropora", 5: "table_acropora", 6: "pocillopora", 7: "meandering", 8: "stylophora",
        },
        "mapping": {
            1: [6], 2: [16, 17, 23], 3: [19, 20, 22], 4: [25], 5: [28, 32], 6: [31], 7: [33, 36, 37], 8: [34],
        },
    },
    "health": {
        "id2label": {0: "unlabeled", 1: "alive", 2: "bleached", 3: "dead"},
        "mapping": {1: [6, 17, 22, 25, 28, 31, 34, 36], 2: [16, 19, 33], 3: [20, 23, 32, 37]},
    },
    "fish": {"id2label": {0: "unlabeled", 1: "fish"}, "mapping": {1: [9]}},
    "human_artifacts": {"id2label": {0: "unlabeled", 1: "artifact"}, "mapping": {1: [7, 8, 15]}},
    "substrate": {
        "id2label": {0: "unlabeled", 1: "sand", 2: "rock_rubble", 3: "algae_covered"},
        "mapping": {1: [5], 2: [12, 18], 3: [10]},
    },
}

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
                 split: str = 'train',
                 augmentations: Optional[SegmentationAugmentation] = None,
                 patch_size: int = 512):
        self.dataset_split = hf_dataset[split]
        self.augmentations = augmentations
        self.patch_size = patch_size

        self.lookup_tables = {
            task_name: create_lookup_table(task_info["mapping"])
            for task_name, task_info in TASK_DEFINITIONS.items()
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