"""Shared fixtures for the test suite (revised).

Key changes:
 - Enforce CUDA (skip tests if not available) to catch device-specific issues early.
 - Load all experiment configurations strictly from YAML files; no inline config dicts.
 - Provide utilities to inject extreme parameter variants via parametrization helpers.
 - Add alternate extreme task definitions file to validate dynamic class handling.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Tuple
import copy

import numpy as np
import pytest
import torch
import yaml
from unittest.mock import MagicMock, patch


def pytest_addoption(parser):
    parser.addoption(
        "--run-parameter-coverage",
        action="store_true",
        default=False,
        help="Run the exhaustive ExperimentFactory parameter coverage suite.",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-parameter-coverage"):
        return

    skip_marker = pytest.mark.skip(
        reason="Use --run-parameter-coverage to enable ExperimentFactory parameter coverage tests.",
    )
    for item in items:
        if "parameter_coverage" in item.keywords:
            item.add_marker(skip_marker)


# Default to CPU-only execution for tests unless the user explicitly exposes GPUs.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
FORCED_CPU = os.environ.get("CUDA_VISIBLE_DEVICES", "") == ""

from coral_mtl.utils.task_splitter import MTLTaskSplitter, BaseTaskSplitter


################################################################################
# Global Test Preconditions
################################################################################

# Fix random seeds for reproducible tests
@pytest.fixture(autouse=True)
def fix_random_seeds():
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


@pytest.fixture(scope="session")
def device():  # noqa: D401
    """Return the default device for tests (CPU unless GPU explicitly enabled)."""
    if not FORCED_CPU and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


################################################################################
# Task Definitions Fixtures
################################################################################

@pytest.fixture(scope="session")
def task_definitions_path_default() -> Path:
    return Path(__file__).parent / "configs" / "tasks" / "task_definitions.yaml"





################################################################################
# Dynamic Extreme Variants Builders
################################################################################

def _load_default_tasks(task_definitions_path_default: Path) -> Dict[str, Any]:
    with open(task_definitions_path_default, 'r') as f:
        return yaml.safe_load(f)


def _union_label_space(base: Dict[str, Any]) -> Dict[int, str]:
    union: Dict[int, str] = {}
    for task_content in base.values():
        id2label = task_content.get('id2label', {})
        for rid, label in id2label.items():
            union.setdefault(rid, label)
    return dict(sorted(union.items(), key=lambda kv: kv[0]))


def build_all_in_one_variant(base: Dict[str, Any]) -> Dict[str, Any]:
    union = _union_label_space(base)
    # Replace genus & health with full union, drop groupby
    variant = _load_shallow_copy(base)
    for t in ['genus', 'health']:
        variant[t] = {'id2label': union}
    return variant


def build_each_label_tasks_variant(base: Dict[str, Any]) -> Dict[str, Any]:
    variant = _load_shallow_copy(base)
    # Add per-label tasks: label_<id>
    union = _union_label_space(base)
    for rid, label in union.items():
        if rid == 0:
            continue  # unlabeled already
        variant[f'label_{rid}'] = {
            'id2label': {
                0: 'unlabeled',
                rid: label
            }
        }
    return variant


def build_groupby_single_variant(base: Dict[str, Any]) -> Dict[str, Any]:
    variant = _load_shallow_copy(base)
    for t in ['genus', 'health']:
        task = variant.get(t)
        if not task:
            continue
        raw_ids = [rid for rid in task['id2label'].keys() if rid != 0]
        task['groupby'] = {
            'mapping': {
                0: 0,
                1: raw_ids
            },
            'id2label': {
                0: 'unlabeled',
                1: 'all'
            }
        }
    return variant


def build_groupby_all_split_variant(base: Dict[str, Any]) -> Dict[str, Any]:
    variant = _load_shallow_copy(base)
    for t in ['genus', 'health']:
        task = variant.get(t)
        if not task:
            continue
        mapping: Dict[int, List[int]] = {0: 0}
        gid = 1
        for rid in sorted(task['id2label'].keys()):
            if rid == 0:
                continue
            mapping[gid] = [rid]
            gid += 1
        group_id2label = {0: 'unlabeled'}
        for g, ids in mapping.items():
            if g == 0:
                continue
            group_id2label[g] = task['id2label'][ids[0]]
        task['groupby'] = {
            'mapping': mapping,
            'id2label': group_id2label
        }
    return variant


def build_extreme_variant(base: Dict[str, Any]) -> Dict[str, Any]:
    """Build an extreme variant with real labels but extreme edge case configurations.
    
    Creates extreme test scenarios by:
    - Swapping labels between different original IDs  
    - Repeating the same label multiple times
    - Omitting many classes to create sparse ID spaces
    - Creating extreme groupings (single item groups, huge groups)
    - Using labels from different tasks in unexpected places
    """
    variant = {}
    
    # EXTREME genus task: Mix coral/non-coral labels with huge gaps in ID space
    # Use original labels but in weird combinations and sparse IDs
    variant['genus'] = {
        'id2label': {
            0: "unlabeled",
            2: "trash",              # Swap: use biota label for genus
            7: "other coral alive",  # Swap: use health label  
            15: "fish",              # Repeat: same label appears twice
            29: "fish",              # Repeat: same fish label again
            38: "sponge"             # Swap: use different biota label
        },
        'groupby': {
            'mapping': {
                0: 0,
                1: [2],                    # Single item group (extreme)
                2: [7, 15, 29, 38]        # Huge group with mixed labels (extreme)
            },
            'id2label': {
                0: "unlabeled", 
                1: "trash_group",          # Group name doesn't match content
                2: "mixed_marine_life"     # Extremely broad grouping
            }
        }
    }
    
    # EXTREME health task: Only dead things, but use wrong original IDs
    variant['health'] = {
        'id2label': {
            0: "unlabeled",
            1: "other coral dead",     # Use original dead label but wrong ID
            11: "dead clam",           # Use biota dead label
            39: "massive/meandering dead" # Swap dead labels around
        },
        'groupby': {
            'mapping': {
                0: 0,
                1: [1, 11, 39]  # Group all dead things together (extreme simplification)
            },
            'id2label': {
                0: "unlabeled",
                1: "everything_dead"     # Extreme over-grouping
            }
        }
    }
    
    # EXTREME auxiliary task: Mix multiple original auxiliary tasks
    # Use substrate + human_artifacts labels but in fish task
    variant['fish'] = {
        'id2label': {
            0: "unlabeled",
            5: "sand",           # substrate label in fish task (swap)
            8: "transect tools", # human_artifacts label (swap) 
            18: "rubble"         # another substrate label (omit actual fish!)
        }
    }
    
    # EXTREME single-class task: Only background, but multiple IDs point to same thing
    variant['background'] = {
        'id2label': {
            0: "unlabeled", 
            13: "background",  # original
            14: "background"   # repeat same label (extreme duplication)
        }
    }
    
    return variant


def _load_shallow_copy(base: Dict[str, Any]) -> Dict[str, Any]:
    # Shallow copy per task
    return {k: dict(v) for k, v in base.items()}


VARIANT_NAMES: List[str] = [
    "default", "extreme", "all_in_one", "each_label_tasks", "groupby_single", "groupby_all_split"
]


@pytest.fixture(scope="session")
def cached_task_definitions_variants(task_definitions_path_default):
    """Session-level cache mapping variant name -> task definitions dict.

    This avoids rebuilding dynamic variants repeatedly.
    """
    cache: Dict[str, Dict[str, Any]] = {}
    base = _load_default_tasks(task_definitions_path_default)
    # Static variants
    cache['default'] = base
    cache['extreme'] = build_extreme_variant(base)
    # Dynamic variants derived from base
    cache['all_in_one'] = build_all_in_one_variant(base)
    cache['each_label_tasks'] = build_each_label_tasks_variant(base)
    cache['groupby_single'] = build_groupby_single_variant(base)
    cache['groupby_all_split'] = build_groupby_all_split_variant(base)
    return cache


def _apply_variant_markers(request, variant: str):
    """Attach pytest markers based on variant semantics."""
    if variant != 'default':
        request.node.add_marker(pytest.mark.extreme)
    if variant in {'groupby_single', 'groupby_all_split', 'all_in_one'}:
        request.node.add_marker(pytest.mark.grouping)
    if variant in {'each_label_tasks', 'groupby_all_split'}:
        request.node.add_marker(pytest.mark.expansion)


@pytest.fixture(params=VARIANT_NAMES, scope="session")
def task_definitions(request, cached_task_definitions_variants):
    variant = request.param
    _apply_variant_markers(request, variant)
    return cached_task_definitions_variants[variant]


@pytest.fixture
def splitter_mtl(task_definitions):
    return MTLTaskSplitter(task_definitions)


@pytest.fixture
def splitter_base(task_definitions):
    return BaseTaskSplitter(task_definitions)


################################################################################
# Dataset Paths
################################################################################

@pytest.fixture(scope="session")
def coralscapes_test_data():
    return Path(__file__).parent / "dataset" / "coralscapes"


@pytest.fixture(scope="session")
def pds_test_data():
    return Path(__file__).parent / "dataset" / "processed" / "pds_patches"


################################################################################
# YAML Config Loading & Mutation Utilities
################################################################################

def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def _dump_yaml(config: Dict[str, Any], path: Path) -> Path:
    with open(path, 'w') as f:
        yaml.safe_dump(config, f)
    return path


def _inject_overrides(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Shallow/deep merge helper (dict only)."""
    out = dict(base)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _inject_overrides(out[k], v)
        else:
            out[k] = v
    return out


@pytest.fixture(scope="session")
def base_config_paths():
    root = Path(__file__).parent / "configs"
    return {
        'baseline': root / 'test_config_baseline.yaml',
        'mtl': root / 'test_config_mtl.yaml'
    }


def _prepare_config_variant(tmp_path: Path, base_cfg: Dict[str, Any], *, run_name: str, task_defs_path: Path | None = None, overrides: Dict[str, Any] | None = None) -> Path:
    cfg = dict(base_cfg)
    cfg['run_name'] = run_name
    # Align config device with the test environment selection
    cfg['device'] = 'cuda' if not FORCED_CPU and torch.cuda.is_available() else 'cpu'
    # Normalize dataset paths to test fixtures
    cfg['data']['dataset_dir'] = 'tests/dataset/coralscapes'
    cfg['data']['pds_path'] = 'tests/dataset/processed/pds_patches'
    cfg['data']['use_pds'] = True
    # Replace task definitions if provided
    if task_defs_path is not None:
        if 'tasks' in cfg:
            # Both baseline & mtl may have tasks; baseline might not use them
            for key in ('task_definitions_path', 'task_definition_path'):
                if key in cfg['tasks']:
                    cfg['tasks'][key] = str(task_defs_path)
            # Align naming differences
            cfg['tasks']['task_definitions_path'] = str(task_defs_path)
    if overrides:
        cfg = _inject_overrides(cfg, overrides)
    out_path = tmp_path / f"{run_name}.yaml"
    _dump_yaml(cfg, out_path)
    return out_path


@pytest.fixture
def baseline_config_yaml(tmp_path, base_config_paths, task_definitions_path_default):
    base_cfg = _load_yaml(base_config_paths['baseline'])
    return _prepare_config_variant(tmp_path, base_cfg, run_name='baseline_default', task_defs_path=task_definitions_path_default)


@pytest.fixture
def mtl_config_yaml(tmp_path, base_config_paths, task_definitions_path_default):
    base_cfg = _load_yaml(base_config_paths['mtl'])
    return _prepare_config_variant(tmp_path, base_cfg, run_name='mtl_default', task_defs_path=task_definitions_path_default)


@pytest.fixture(scope="session")
def cached_mtl_config_paths(tmp_path_factory, base_config_paths, task_definitions_path_default, cached_task_definitions_variants):
    """Session cache mapping variant -> config YAML path (deduplicated IO)."""
    base_cfg = _load_yaml(base_config_paths['mtl'])
    base_dir = tmp_path_factory.mktemp('mtl_config_variants')
    config_paths: Dict[str, Path] = {}
    for variant in VARIANT_NAMES:
        if variant == 'default':
            task_path = task_definitions_path_default
        else:
            # Write dynamic task variant once (including extreme)
            tasks_dict = cached_task_definitions_variants[variant]
            task_path = base_dir / f"tasks_{variant}.yaml"
            if not task_path.exists():
                with open(task_path, 'w') as f:
                    yaml.safe_dump(tasks_dict, f)
        run_name = f"mtl_{variant}"
        config_paths[variant] = _prepare_config_variant(base_dir, base_cfg, run_name=run_name, task_defs_path=task_path)
    return config_paths


@pytest.fixture(params=VARIANT_NAMES)
def mtl_config_yaml_param(request, cached_mtl_config_paths):
    variant = request.param
    _apply_variant_markers(request, variant)
    return cached_mtl_config_paths[variant]


@pytest.fixture(params=VARIANT_NAMES)
def mtl_variant_config(request, cached_mtl_config_paths) -> Tuple[str, Path]:
    """Return (variant_name, config_path) with markers for introspection tests."""
    variant = request.param
    _apply_variant_markers(request, variant)
    return variant, cached_mtl_config_paths[variant]


################################################################################
# Synthetic Tensor Fixtures (small shapes for speed)
################################################################################

@pytest.fixture
def dummy_images(device):
    return torch.rand(2, 3, 32, 32, device=device)


@pytest.fixture
def dummy_masks(device, splitter_mtl):
    masks = {}
    for task_name, task_info in splitter_mtl.hierarchical_definitions.items():
        max_id = max(task_info['ungrouped']['id2label'].keys())
        masks[task_name] = torch.randint(0, max_id + 1, (2, 32, 32), device=device)
    return masks


@pytest.fixture
def dummy_single_mask(device, splitter_base):
    return torch.randint(0, len(splitter_base.global_id2label), (2, 32, 32), device=device)


################################################################################
# Model Fixtures (Minimal) Using Current Task Definitions
################################################################################

@pytest.fixture
def minimal_coral_mtl_model(splitter_mtl):
    from coral_mtl.model.core import CoralMTLModel
    
    # Dynamically determine tasks from the splitter, which is aware of the current variant
    all_tasks = list(splitter_mtl.hierarchical_definitions.keys())
    
    # A simple heuristic to divide tasks for testing purposes.
    # This is not perfect but more robust than a hardcoded list.
    # We assume at least two tasks can be primary, otherwise just one.
    num_primary = min(len(all_tasks), 2)
    primary_tasks = all_tasks[:num_primary]
    aux_tasks = all_tasks[num_primary:]

    # Ensure all tasks defined in the variant are accounted for in num_classes
    # Use is_grouped logic to determine which label space to use
    num_classes = {}
    for task, info in splitter_mtl.hierarchical_definitions.items():
        if info.get('is_grouped', False):
            num_classes[task] = len(info['grouped']['id2label'])
        else:
            num_classes[task] = len(info['ungrouped']['id2label'])

    return CoralMTLModel(
        encoder_name='mit_b0',
        decoder_channel=32,
        num_classes=num_classes,
        primary_tasks=primary_tasks,
        aux_tasks=aux_tasks,
        attention_dim=32
    )


@pytest.fixture
def minimal_baseline_model(splitter_base):
    from coral_mtl.model.core import BaselineSegformer
    return BaselineSegformer(
        encoder_name='mit_b0',
        decoder_channel=32,
        num_classes=len(splitter_base.global_id2label)
    )


################################################################################
# Optional Dependencies Mocking
################################################################################

@pytest.fixture
def mock_optional_deps():
    with patch.dict('sys.modules', {
        'SimpleITK': MagicMock(),
        'panopticapi': MagicMock(),
        'skimage': MagicMock(),
        'sklearn': MagicMock(),
    }):
        yield


################################################################################
# Temporary Output Directory
################################################################################

@pytest.fixture
def temp_output_dir():
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


################################################################################
# Param Injection Helpers (for @pytest.mark.parametrize use)
################################################################################

def extreme_lr_variants():
    return [1e-6, 1e-5, 5e-4]


def tiny_batch_variants():
    return [1, 2]


def window_stride_pairs():
    return [([32, 32], [16, 16]), ([64, 64], [32, 32])]


CONFIG_SECTION_NAMES: Tuple[str, ...] = (
    'model',
    'data',
    'augmentations',
    'loss',
    'optimizer',
    'metrics',
    'trainer',
    'evaluator',
    'metrics_processor',
)


@pytest.fixture(scope="session")
def factory_config_section_catalog(base_config_paths) -> Dict[str, Dict[str, Any]]:
    """Load baseline and MTL configs into a session cache keyed by config type."""
    catalog: Dict[str, Dict[str, Any]] = {}
    for name, path in base_config_paths.items():
        catalog[name] = _load_yaml(path)
    return catalog


@pytest.fixture(params=('baseline', 'mtl'))
def factory_config_kind(request):
    """Parametrize over baseline and MTL factory configurations."""
    return request.param


@pytest.fixture(params=CONFIG_SECTION_NAMES)
def factory_section_name(request):
    """Iterate across all documented configuration sections."""
    return request.param


@pytest.fixture
def factory_section_config(factory_config_kind, factory_section_name, factory_config_section_catalog):
    """Provide a config section dict for the requested config kind and section."""
    config_dict = copy.deepcopy(factory_config_section_catalog[factory_config_kind])
    section = config_dict.get(factory_section_name)
    if section is None:
        pytest.skip(f"Section '{factory_section_name}' unavailable for config '{factory_config_kind}'")
    return factory_config_kind, factory_section_name, section


@pytest.fixture
def experiment_config_bundle(factory_config_kind, tmp_path, base_config_paths, task_definitions_path_default):
    """Return (config_kind, config_dict, config_path) ready for ExperimentFactory use."""
    base_cfg = _load_yaml(base_config_paths[factory_config_kind])
    config_path = _prepare_config_variant(
        tmp_path,
        base_cfg,
        run_name=f"{factory_config_kind}_bundle",
        task_defs_path=task_definitions_path_default
    )
    config_dict = _load_yaml(config_path)
    return factory_config_kind, config_dict, config_path


__all__ = [
    'device', 'splitter_mtl', 'splitter_base', 'dummy_images', 'dummy_masks',
    'dummy_single_mask', 'minimal_coral_mtl_model', 'minimal_baseline_model',
    'baseline_config_yaml', 'mtl_config_yaml', 'mtl_config_yaml_param', 'mtl_variant_config',
    'extreme_lr_variants', 'tiny_batch_variants', 'window_stride_pairs',
    'factory_section_config', 'factory_section_name', 'factory_config_kind',
    'experiment_config_bundle'
]

################################################################################
# Backward Compatibility Fixtures (legacy names)
################################################################################

@pytest.fixture
def factory_config_dict_mtl(mtl_config_yaml):  # legacy consumer support
    return _load_yaml(Path(mtl_config_yaml))


@pytest.fixture
def factory_config_dict_baseline(baseline_config_yaml):  # legacy consumer support
    return _load_yaml(Path(baseline_config_yaml))