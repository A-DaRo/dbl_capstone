"""Inference helpers for notebook-driven visual analytics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from coral_mtl.ExperimentFactory import ExperimentFactory

from .config_utils import build_inference_ready_config
from .experiment_registry import ExperimentMetadata
from .label_utils import get_grouped_label_names
from .visualization_settings import VisualizationSettings


@dataclass
class TaskInferenceResult:
    """Container for task-specific inference payloads."""

    probabilities: np.ndarray
    targets: np.ndarray
    label_names: List[str]
    image_panels: List[Dict[str, np.ndarray]]
    image_ids: List[str]


_DEF_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_DEF_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _load_factory(metadata: ExperimentMetadata, target_image_ids: Optional[Iterable[str]] = None) -> ExperimentFactory:
    if not metadata.config_path:
        raise ValueError(f"No configuration associated with experiment {metadata.name}.")
    subset_ids = list(target_image_ids) if target_image_ids is not None else None
    config, fallback_used = build_inference_ready_config(metadata.config_path, subset_image_ids=subset_ids)
    return ExperimentFactory(config_dict=config)


def _load_model(factory: ExperimentFactory, checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    model = factory.get_model()
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def _unnormalize_image(tensor: torch.Tensor) -> np.ndarray:
    mean = _DEF_MEAN.to(tensor.device, dtype=tensor.dtype)
    std = _DEF_STD.to(tensor.device, dtype=tensor.dtype)
    unnorm = tensor * std + mean
    clipped = torch.clamp(unnorm, 0.0, 1.0)
    return clipped.permute(1, 2, 0).cpu().numpy()


def _prepare_baseline_group_mapping(splitter, task_name: str) -> Tuple[np.ndarray, np.ndarray]:
    task_def = splitter.hierarchical_definitions[task_name]
    if task_def["is_grouped"]:
        mapping_array = task_def["grouped"]["mapping_array"]
    else:
        mapping_array = task_def["ungrouped"]["mapping_array"]
    flat_to_original = splitter.flat_to_original_mapping_array
    flat_to_group = mapping_array[flat_to_original]
    num_groups = int(flat_to_group.max()) + 1
    aggregation = np.zeros((len(flat_to_group), num_groups), dtype=np.float32)
    for flat_idx, group_idx in enumerate(flat_to_group):
        aggregation[flat_idx, group_idx] += 1.0
    return flat_to_group, aggregation


def _baseline_targets_to_group(target: torch.Tensor, flat_to_group: np.ndarray) -> np.ndarray:
    flat = target.cpu().numpy().astype(np.int64)
    return flat_to_group[flat]


def collect_task_predictions(
    metadata: ExperimentMetadata,
    task_name: str,
    settings: VisualizationSettings,
    target_image_ids: Optional[Iterable[str]] = None,
) -> TaskInferenceResult:
    """
    Collect task-level probabilities and targets for downstream analytics.
    
    Simplified inference path:
    - Uses test dataset if full dataset path exists
    - Falls back to tests dataset path if full dataset unavailable
    - Direct full-image inference without sliding window
    - Model loaded with best_model.pth weights in eval mode
    - When target_image_ids provided, dataset is filtered to only those images
    """

    factory = _load_factory(metadata, target_image_ids=target_image_ids)
    splitter = factory.task_splitter
    checkpoint_path = metadata.artifacts.get("checkpoint")
    if not checkpoint_path:
        raise FileNotFoundError(f"Checkpoint not found for experiment {metadata.name}.")

    device = torch.device("cpu")
    model = _load_model(factory, checkpoint_path, device)
    
    dataloaders = factory.get_dataloaders()
    loader = dataloaders.get("test")
    
    if loader is None:
        print(f"[WARNING] Full test dataset not found for {metadata.name}. Using tests fallback dataset.")
        loader = dataloaders.get("validation")
        if loader is None:
            raise RuntimeError(f"No evaluation dataloader for experiment {metadata.name}.")

    model_type = factory.config.get("model", {}).get("type")

    if model_type == "SegFormerBaseline" and task_name not in ("global", *splitter.hierarchical_definitions.keys()):
        raise ValueError(f"Task '{task_name}' incompatible with baseline predictions.")

    flat_to_group: Optional[np.ndarray] = None
    aggregation_matrix: Optional[np.ndarray] = None
    if model_type == "SegFormerBaseline" and task_name != "global":
        flat_to_group, aggregation_matrix = _prepare_baseline_group_mapping(splitter, task_name)
        label_names = get_grouped_label_names(splitter, task_name)
    elif model_type == "SegFormerBaseline":
        label_names = [splitter.flat_id2label[i] for i in range(len(splitter.flat_id2label))]
    else:
        label_names = get_grouped_label_names(splitter, task_name)

    collected_probs: List[np.ndarray] = []
    collected_targets: List[np.ndarray] = []
    image_panels: List[Dict[str, np.ndarray]] = []
    collected_image_ids: List[str] = []
    total_pixels = 0
    max_pixels: Optional[int] = settings.max_pr_pixels if target_image_ids is None else None
    batch_limit: Optional[int] = (
        settings.max_samples_per_experiment if target_image_ids is None else None
    )
    qualitative_budget = settings.qualitative_samples
    target_set: Optional[Set[str]] = None
    remaining_targets: Optional[Set[str]] = None
    if target_image_ids is not None:
        target_set = {str(identifier) for identifier in target_image_ids}
        remaining_targets = set(target_set)
        figure_limit = settings.qualitative_samples if settings.qualitative_samples > 0 else len(target_set)
        qualitative_budget = min(len(target_set), figure_limit if figure_limit > 0 else len(target_set))

    for batch_index, batch in enumerate(loader):
        if batch_limit is not None and batch_index >= batch_limit:
            break
        if max_pixels is not None and total_pixels >= max_pixels:
            break

        images = batch["image"].to(device)
        
        with torch.no_grad():
            outputs = model(images)

        if model_type == "CoralMTL":
            logits = outputs.get(task_name)
            if logits is None:
                continue
            probabilities_tensor = F.softmax(logits, dim=1)
            targets_tensor = batch["masks"][task_name]
        else:
            logits = outputs.get("segmentation") if isinstance(outputs, dict) else outputs
            if logits is None:
                raise RuntimeError("Segmentation logits missing from inference output.")
            probabilities_flat = F.softmax(logits, dim=1)
            if task_name == "global":
                probabilities_tensor = probabilities_flat
                targets_tensor = batch["mask"]
            else:
                if aggregation_matrix is None or flat_to_group is None:
                    raise RuntimeError("Aggregation mapping unavailable for baseline task conversion.")
                aggregated_samples = []
                grouped_targets = []
                mask_tensor = batch["mask"]
                height, width = mask_tensor.shape[-2:]
                for sample_idx in range(probabilities_flat.shape[0]):
                    sample_probs = probabilities_flat[sample_idx].permute(1, 2, 0).reshape(-1, probabilities_flat.shape[1]).cpu().numpy()
                    aggregated_np = sample_probs @ aggregation_matrix
                    aggregated_tensor = torch.from_numpy(
                        aggregated_np.reshape(height, width, aggregation_matrix.shape[1])
                    ).permute(2, 0, 1).contiguous().float()
                    aggregated_samples.append(aggregated_tensor)
                    grouped_np = _baseline_targets_to_group(mask_tensor[sample_idx], flat_to_group)
                    grouped_targets.append(torch.from_numpy(grouped_np).long())
                probabilities_tensor = torch.stack(aggregated_samples, dim=0)
                targets_tensor = torch.stack(grouped_targets, dim=0)

        probabilities_cpu = probabilities_tensor.cpu()
        targets_cpu = targets_tensor.cpu()
        batch_size = probabilities_cpu.shape[0]

        image_ids = batch.get("image_id")
        resolved_ids: List[str] = []
        for sample_idx in range(batch_size):
            identifier: Optional[str] = None
            if image_ids is not None:
                if isinstance(image_ids, (list, tuple)):
                    identifier = str(image_ids[sample_idx])
                else:
                    try:
                        identifier = str(image_ids[sample_idx])
                    except Exception:
                        identifier = str(image_ids)
            if identifier is None:
                identifier = f"{metadata.name}_{batch_index}_{sample_idx}"
            resolved_ids.append(identifier)

        for sample_idx, identifier in enumerate(resolved_ids):
            if target_set is not None and identifier not in target_set:
                continue

            sample_probs_tensor = probabilities_cpu[sample_idx]
            sample_targets_tensor = targets_cpu[sample_idx]

            probs_np = (
                sample_probs_tensor.permute(1, 2, 0).contiguous().view(-1, sample_probs_tensor.shape[0]).numpy()
            )
            targets_np = sample_targets_tensor.view(-1).numpy().astype(np.int64)

            pixel_count = probs_np.shape[0]
            if max_pixels is not None and total_pixels + pixel_count > max_pixels:
                keep = max_pixels - total_pixels
                if keep <= 0:
                    break
                probs_np = probs_np[:keep]
                targets_np = targets_np[:keep]
                pixel_count = keep

            collected_probs.append(probs_np)
            collected_targets.append(targets_np)
            collected_image_ids.append(identifier)
            total_pixels += pixel_count

            if qualitative_budget > 0:
                rgb = _unnormalize_image(images[sample_idx])
                pred_mask = sample_probs_tensor.argmax(dim=0).detach().numpy()
                target_mask = sample_targets_tensor.detach().numpy()
                image_panels.append(
                    {
                        "image_id": identifier,
                        "rgb": rgb,
                        "prediction": pred_mask,
                        "target": target_mask,
                    }
                )
                qualitative_budget -= 1

            if remaining_targets is not None:
                remaining_targets.discard(identifier)

        if max_pixels is not None and total_pixels >= max_pixels:
            break

        if target_set is not None and remaining_targets is not None and not remaining_targets:
            break

    if not collected_probs:
        targets_str = f" for target ids {sorted(target_set)}" if target_set else ""
        raise RuntimeError(
            f"No predictions collected for experiment {metadata.name} and task {task_name}{targets_str}."
        )

    probabilities_arr = np.concatenate(collected_probs, axis=0)
    targets_arr = np.concatenate(collected_targets, axis=0)

    return TaskInferenceResult(
        probabilities=probabilities_arr,
        targets=targets_arr,
        label_names=label_names,
        image_panels=image_panels,
        image_ids=collected_image_ids,
    )
