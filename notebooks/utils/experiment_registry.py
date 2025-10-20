"""Discovery utilities for experiments stored under experiments/.

The registry exposes structured metadata for each run so downstream analysis can
uniformly access histories, confusion matrices, and checkpoints without
hard-coded filenames embedded in the notebook itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .config_utils import get_config_for_experiment
from .project_paths import get_experiments_root


@dataclass(frozen=True)
class ExperimentMetadata:
    """Container describing a single experiment run on disk."""

    name: str
    path: Path
    config_path: Optional[Path]
    artifacts: Dict[str, Path]


def _collect_artifact_paths(experiment_dir: Path) -> Dict[str, Path]:
    """Enumerate known artifact files within an experiment directory."""
    expected_files = {
        "history": "history.json",
        "test_metrics": "test_metrics_full_report.json",
        "test_cms": "test_cms.jsonl",
        "advanced_metrics": "advanced_metrics.jsonl",
        "validation_cms": "validation_cms.jsonl",
        "test_losses": "test_loss_metrics.json",
        "loss_diagnostics": "loss_diagnostics.jsonl",
        "checkpoint": "best_model.pth",
    }
    artifacts: Dict[str, Path] = {}
    for key, relative_name in expected_files.items():
        candidate = experiment_dir / relative_name
        if candidate.exists():
            artifacts[key] = candidate
    return artifacts


def discover_experiments() -> List[ExperimentMetadata]:
    """Discover all experiment runs available on disk."""
    experiments_root = get_experiments_root()
    metadata: List[ExperimentMetadata] = []
    if not experiments_root.exists():
        return metadata

    for experiment_dir in sorted(experiments_root.rglob("*")):
        if not experiment_dir.is_dir():
            continue
        artifacts = _collect_artifact_paths(experiment_dir)
        if not artifacts:
            continue
        config_path = get_config_for_experiment(experiment_dir)
        metadata.append(
            ExperimentMetadata(
                name=experiment_dir.relative_to(experiments_root).as_posix(),
                path=experiment_dir,
                config_path=config_path,
                artifacts=artifacts,
            )
        )
    return metadata
