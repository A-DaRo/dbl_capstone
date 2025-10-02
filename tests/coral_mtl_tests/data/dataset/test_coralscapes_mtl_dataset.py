"""Unit tests for ``CoralscapesMTLDataset`` covering mapping and integration paths."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision.transforms import v2

import coral_mtl.data.dataset as dataset_module
from coral_mtl.data.dataset import CoralscapesMTLDataset


@dataclass
class _DummyHFDataset:
    sample: Dict[str, object]

    def __len__(self) -> int:  # pragma: no cover - trivial
        return 1

    def __getitem__(self, idx: int) -> Dict[str, object]:
        if idx != 0:
            raise IndexError("Dummy dataset only contains one element")
        return self.sample


def _patch_hf_dataset(monkeypatch: pytest.MonkeyPatch, sample: Dict[str, object]) -> None:
    monkeypatch.setattr(dataset_module, "load_dataset", lambda name, split: _DummyHFDataset(sample))


def _expected_task_mask(
    raw_mask: np.ndarray, task_info: Dict, size: Iterable[int] | None
) -> torch.Tensor:
    if task_info['is_grouped']:
        mapping_array = task_info['grouped']['mapping_array']
    else:
        mapping_array = task_info['ungrouped']['mapping_array']
    clipped = np.clip(raw_mask, 0, len(mapping_array) - 1)
    mapped = mapping_array[clipped].astype(np.uint8)
    mask_image = Image.fromarray(mapped)
    if size is not None:
        mask_image = v2.functional.resize(mask_image, size, interpolation=v2.InterpolationMode.NEAREST)
    return torch.from_numpy(np.array(mask_image)).long()


def _make_sample(raw_mask: np.ndarray, image: np.ndarray) -> Dict[str, object]:
    return {
        "image": Image.fromarray(image),
        "label": raw_mask,
        "id": "stub-000",
    }


def test_mtl_train_split_returns_expected_structure_and_mapped_masks(monkeypatch, splitter_mtl):
    max_len = max(len(info["ungrouped"]["mapping_array"]) for info in splitter_mtl.hierarchical_definitions.values())
    raw_mask = np.mod(np.arange(99, dtype=np.uint16).reshape(9, 11), max_len + 3).astype(np.uint8)
    raw_mask[0, 0] = max_len + 5  # exercise clipping path
    image = np.random.default_rng(0).integers(0, 256, size=(9, 11, 3), dtype=np.uint8)

    sample = _make_sample(raw_mask, image)
    _patch_hf_dataset(monkeypatch, sample)

    dataset = CoralscapesMTLDataset(
        splitter=splitter_mtl,
        hf_dataset_name="coral-stub",
        split="train",
        patch_size=8,
        augmentations=None,
    )

    assert len(dataset) == 1
    item = dataset[0]

    assert set(item.keys()) == {"image", "image_id", "original_mask", "masks"}
    assert item["image"].shape == (3, 8, 8)
    assert item["image"].dtype == torch.float32
    torch.testing.assert_close(item["original_mask"], torch.from_numpy(raw_mask).long())
    assert item["image_id"] == "stub-000"

    expected_tasks = set(splitter_mtl.hierarchical_definitions.keys())
    assert set(item["masks"].keys()) == expected_tasks

    for task_name, task_info in splitter_mtl.hierarchical_definitions.items():
        expected_mask = _expected_task_mask(
            raw_mask,
            task_info,
            size=(8, 8),
        )
        torch.testing.assert_close(item["masks"][task_name], expected_mask)


def test_mtl_validation_split_preserves_original_resolution(monkeypatch, splitter_mtl):
    raw_mask = np.arange(56, dtype=np.uint8).reshape(7, 8)
    image = np.random.default_rng(1).integers(0, 256, size=(7, 8, 3), dtype=np.uint8)

    sample = _make_sample(raw_mask, image)
    _patch_hf_dataset(monkeypatch, sample)

    dataset = CoralscapesMTLDataset(
        splitter=splitter_mtl,
        hf_dataset_name="coral-stub",
        split="validation",
        patch_size=12,
        augmentations=None,
    )

    item = dataset[0]

    assert item["image"].shape == (3, 7, 8)
    for task_name, task_info in splitter_mtl.hierarchical_definitions.items():
        expected_mask = _expected_task_mask(raw_mask, task_info, size=None)
        torch.testing.assert_close(item["masks"][task_name], expected_mask)

    torch.testing.assert_close(item["original_mask"], torch.from_numpy(raw_mask).long())


def test_mtl_dataset_uses_provided_augmentations(monkeypatch, splitter_mtl):
    raw_mask = np.zeros((6, 6), dtype=np.uint8)
    image = np.zeros((6, 6, 3), dtype=np.uint8)

    sample = _make_sample(raw_mask, image)
    _patch_hf_dataset(monkeypatch, sample)

    class RecordingAugmentation:
        def __init__(self):
            self.calls = []
            self.return_value = None

        def __call__(self, image_pil, mask_dict):
            self.calls.append((image_pil, dict(mask_dict)))
            image_tensor = torch.full((3, 5, 5), 0.5, dtype=torch.float32)
            mask_tensors = {
                name: torch.full((5, 5), idx, dtype=torch.long)
                for idx, name in enumerate(mask_dict.keys(), start=1)
            }
            self.return_value = (image_tensor, mask_tensors)
            return image_tensor, mask_tensors

    augmentation = RecordingAugmentation()

    dataset = CoralscapesMTLDataset(
        splitter=splitter_mtl,
        hf_dataset_name="coral-stub",
        split="train",
        patch_size=16,
        augmentations=augmentation,
    )

    item = dataset[0]

    assert len(augmentation.calls) == 1
    recorded_image, recorded_masks = augmentation.calls[0]
    assert isinstance(recorded_image, Image.Image)
    assert set(recorded_masks.keys()) == set(splitter_mtl.hierarchical_definitions.keys())

    image_tensor, mask_tensors = augmentation.return_value
    assert item["image"] is image_tensor
    assert image_tensor.shape == (3, 5, 5)
    for task_name, mask_tensor in mask_tensors.items():
        assert item["masks"][task_name] is mask_tensor
        assert mask_tensor.shape == (5, 5)
        assert torch.all(mask_tensor == mask_tensor[0, 0])

    torch.testing.assert_close(item["original_mask"], torch.zeros((6, 6), dtype=torch.long))


def test_mtl_dataset_rejects_non_mtl_splitter(monkeypatch, splitter_base):
    raw_mask = np.zeros((4, 4), dtype=np.uint8)
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    sample = _make_sample(raw_mask, image)
    _patch_hf_dataset(monkeypatch, sample)

    with pytest.raises(TypeError):
        CoralscapesMTLDataset(
            splitter=splitter_base,
            hf_dataset_name="coral-stub",
            split="train",
            patch_size=4,
        )