"""Focused tests for the baseline ``CoralscapesDataset`` label mapping and augmentation paths."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision.transforms import v2

import coral_mtl.data.dataset as dataset_module
from coral_mtl.data.dataset import CoralscapesDataset


@dataclass
class _DummyHFDataset:
    sample: Dict[str, object]

    def __len__(self) -> int:  # pragma: no cover - trivial wrapper
        return 1

    def __getitem__(self, idx: int) -> Dict[str, object]:
        if idx != 0:
            raise IndexError("Dummy dataset only contains one element")
        return self.sample


def _patch_hf_dataset(monkeypatch: pytest.MonkeyPatch, sample: Dict[str, object]) -> None:
    monkeypatch.setattr(dataset_module, "load_dataset", lambda name, split: _DummyHFDataset(sample))


def _make_sample(raw_mask: np.ndarray, image: np.ndarray) -> Dict[str, object]:
    return {
        "image": Image.fromarray(image),
        "label": raw_mask,
        "id": "stub-000",
    }


def test_baseline_train_split_returns_flattened_mask(monkeypatch, splitter_base):
    max_len = len(splitter_base.flat_mapping_array)
    raw_mask = np.mod(np.arange(121, dtype=np.uint16).reshape(11, 11), max_len + 2).astype(np.uint8)
    raw_mask[2, 3] = max_len + 5  # exercise clipping
    image = np.random.default_rng(2).integers(0, 256, size=(11, 11, 3), dtype=np.uint8)

    sample = _make_sample(raw_mask, image)
    _patch_hf_dataset(monkeypatch, sample)

    dataset = CoralscapesDataset(
        splitter=splitter_base,
        hf_dataset_name="coral-stub",
        split="train",
        patch_size=10,
        augmentations=None,
    )

    item = dataset[0]

    assert set(item.keys()) == {"image", "image_id", "original_mask", "mask"}
    assert item["image"].shape == (3, 10, 10)
    assert item["mask"].shape == (10, 10)
    assert item["mask"].dtype == torch.long
    torch.testing.assert_close(item["original_mask"], torch.from_numpy(raw_mask).long())

    clipped = np.clip(raw_mask, 0, max_len - 1)
    mapped = splitter_base.flat_mapping_array[clipped].astype(np.uint8)
    resized = np.array(
        v2.functional.resize(Image.fromarray(mapped), (10, 10), interpolation=v2.InterpolationMode.NEAREST)
    )
    torch.testing.assert_close(item["mask"], torch.from_numpy(resized).long())


def test_baseline_validation_split_keeps_original_resolution(monkeypatch, splitter_base):
    raw_mask = np.arange(63, dtype=np.uint8).reshape(7, 9)
    image = np.random.default_rng(3).integers(0, 256, size=(7, 9, 3), dtype=np.uint8)

    sample = _make_sample(raw_mask, image)
    _patch_hf_dataset(monkeypatch, sample)

    dataset = CoralscapesDataset(
        splitter=splitter_base,
        hf_dataset_name="coral-stub",
        split="validation",
        patch_size=12,
        augmentations=None,
    )

    item = dataset[0]

    assert item["image"].shape == (3, 7, 9)
    assert item["mask"].shape == (7, 9)

    clipped = np.clip(raw_mask, 0, len(splitter_base.flat_mapping_array) - 1)
    mapped = splitter_base.flat_mapping_array[clipped].astype(np.uint8)
    torch.testing.assert_close(item["mask"], torch.from_numpy(mapped).long())


def test_baseline_dataset_uses_augmentations(monkeypatch, splitter_base):
    raw_mask = np.zeros((5, 5), dtype=np.uint8)
    image = np.zeros((5, 5, 3), dtype=np.uint8)

    sample = _make_sample(raw_mask, image)
    _patch_hf_dataset(monkeypatch, sample)

    class RecordingAugmentation:
        def __init__(self):
            self.calls = []
            self.return_value = None

        def __call__(self, image_pil, mask_dict):
            self.calls.append((image_pil, dict(mask_dict)))
            image_tensor = torch.full((3, 6, 6), 0.25, dtype=torch.float32)
            mask_tensor = torch.ones((6, 6), dtype=torch.long)
            self.return_value = (image_tensor, {"mask": mask_tensor})
            return image_tensor, {"mask": mask_tensor}

    augmentation = RecordingAugmentation()

    dataset = CoralscapesDataset(
        splitter=splitter_base,
        hf_dataset_name="coral-stub",
        split="train",
        patch_size=8,
        augmentations=augmentation,
    )

    item = dataset[0]

    assert len(augmentation.calls) == 1
    recorded_image, recorded_masks = augmentation.calls[0]
    assert isinstance(recorded_image, Image.Image)
    assert set(recorded_masks.keys()) == {"mask"}

    image_tensor, mask_dict = augmentation.return_value
    assert item["image"] is image_tensor
    assert item["mask"] is mask_dict["mask"]
    torch.testing.assert_close(image_tensor, torch.full((3, 6, 6), 0.25, dtype=torch.float32))
    torch.testing.assert_close(mask_dict["mask"], torch.ones((6, 6), dtype=torch.long))
    torch.testing.assert_close(item["original_mask"], torch.zeros((5, 5), dtype=torch.long))


def test_baseline_dataset_rejects_mtl_splitter(monkeypatch, splitter_mtl):
    raw_mask = np.zeros((4, 4), dtype=np.uint8)
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    sample = _make_sample(raw_mask, image)
    _patch_hf_dataset(monkeypatch, sample)

    with pytest.raises(TypeError):
        CoralscapesDataset(
            splitter=splitter_mtl,
            hf_dataset_name="coral-stub",
            split="train",
            patch_size=4,
        )