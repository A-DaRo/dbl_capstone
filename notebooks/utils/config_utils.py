"""Configuration helpers for experiment discovery and manipulation.

The visualization notebook relies on these utilities to locate experiment
configuration files, resolve relative path references, and construct derived
configurations for lightweight inference workflows.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import yaml

from .project_paths import get_configs_dir, get_project_root


POSSIBLE_DATASET_PATHS = [
    Path("../coralscapes"),
    Path("./dataset/coralscapes"),
]


def load_yaml_config(config_path: Path) -> Dict:
    """Load a YAML configuration file into a dictionary."""
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def iter_config_paths() -> Iterator[Path]:
    """Yield all YAML configuration files under the configs/ hierarchy."""
    configs_dir = get_configs_dir()
    patterns = ["**/*.yaml", "**/*.yml"]
    for pattern in patterns:
        for path in configs_dir.glob(pattern):
            if path.is_file():
                yield path


def build_output_dir_to_config_map() -> Dict[Path, Path]:
    """Build a mapping from resolved trainer output directories to configs."""
    mapping: Dict[Path, Path] = {}
    project_root = get_project_root()
    for config_path in iter_config_paths():
        try:
            config = load_yaml_config(config_path)
        except yaml.YAMLError:
            continue

        trainer_cfg = config.get("trainer", {}) if isinstance(config, dict) else {}
        output_value = trainer_cfg.get("output_dir")
        if not output_value:
            continue

        resolved_output = (project_root / output_value).resolve() if not os.path.isabs(str(output_value)) else Path(output_value).resolve()
        mapping[resolved_output] = config_path
    return mapping


def get_config_for_experiment(experiment_path: Path) -> Optional[Path]:
    """Return the config path associated with the given experiment directory."""
    mapping = build_output_dir_to_config_map()
    resolved_experiment = experiment_path.resolve()
    return mapping.get(resolved_experiment)


def build_inference_ready_config(config_path: Path, subset_image_ids: Optional[List[str]] = None) -> Dict:
    """Produce a config dictionary tailored for lightweight CPU inference.

    The function clones the original configuration and adjusts dataset paths to
    fall back to the test fixtures when the referenced locations are missing.
    It also reduces data loader worker usage to avoid resource pressure during
    interactive notebook sessions.
    
    Args:
        config_path: Path to the base configuration file.
        subset_image_ids: Optional list of image IDs to filter the dataset to only those images.
    """

    base_config = load_yaml_config(config_path)
    config = copy.deepcopy(base_config)
    project_root = get_project_root()
    tests_dataset_root = project_root / "tests" / "dataset" / "coralscapes"
    tests_pds_root = project_root / "tests" / "dataset" / "processed" / "pds_patches"

    data_cfg = config.setdefault("data", {})

    def _resolve_existing_path(value: Optional[str]) -> Optional[Path]:
        if not value:
            return None
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = (project_root / value).resolve()
        return candidate if candidate.exists() else None

    resolved_dataset_root = _resolve_existing_path(data_cfg.get("data_root_path"))
    dataset_name_path = _resolve_existing_path(data_cfg.get("dataset_name"))

    if resolved_dataset_root is None:
        resolved_dataset_root = dataset_name_path

    if resolved_dataset_root is None:
        for rel_path in POSSIBLE_DATASET_PATHS:
            candidate = (project_root / rel_path).resolve()
            if candidate.exists():
                resolved_dataset_root = candidate
                break

    fallback_used = False
    if resolved_dataset_root is None and tests_dataset_root.exists():
        resolved_dataset_root = tests_dataset_root.resolve()
        fallback_used = True

    if resolved_dataset_root is not None:
        data_cfg["data_root_path"] = str(resolved_dataset_root)
        original_dataset_name = data_cfg.get("dataset_name")
        if fallback_used or (
            original_dataset_name
            and (
                Path(original_dataset_name).exists()
                or (project_root / original_dataset_name).exists()
                or original_dataset_name.startswith("./")
                or original_dataset_name.startswith("../")
            )
        ):
            data_cfg["dataset_name"] = str(resolved_dataset_root)

    if fallback_used:
        candidate_list = ", ".join(str((project_root / rel).resolve()) for rel in POSSIBLE_DATASET_PATHS)
        print(f"[warn] Falling back to tests dataset at {resolved_dataset_root} because none of {candidate_list} were found.")

    pds_path = _resolve_existing_path(data_cfg.get("pds_train_path"))
    if pds_path is not None:
        data_cfg["pds_train_path"] = str(pds_path)
    elif tests_pds_root.exists():
        data_cfg["pds_train_path"] = str(tests_pds_root.resolve())

    # Drop down to a conservative loader configuration for notebooks
    data_cfg["batch_size_per_gpu"] = 1
    data_cfg["num_workers"] = 0
    data_cfg["persistent_workers"] = False
    data_cfg["prefetch_factor"] = None

    trainer_cfg = config.setdefault("trainer", {})
    trainer_cfg["device"] = "cpu"
    trainer_cfg["use_mixed_precision"] = False
    trainer_cfg["mixed_precision_dtype"] = "fp32"

    cache_root = project_root / "notebooks" / ".visualization_cache"
    cache_output_dir = cache_root / config_path.stem
    trainer_cfg["output_dir"] = str(cache_output_dir)

    evaluator_cfg = config.setdefault("evaluator", {})
    evaluator_cfg["inference_batch_size"] = 1
    evaluator_cfg["inference_stride"] = evaluator_cfg.get("inference_stride", 256)
    evaluator_cfg["output_dir"] = str(cache_output_dir / "evaluation")

    metrics_processor_cfg = config.setdefault("metrics_processor", {})
    metrics_processor_cfg["enabled"] = False

    # Add subset filter if target image IDs are provided
    if subset_image_ids is not None:
        config["_inference_subset_image_ids"] = list(subset_image_ids)

    return config, fallback_used
