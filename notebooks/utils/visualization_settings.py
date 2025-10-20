"""Centralized parameters governing visualization routines."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class VisualizationSettings:
    max_pr_pixels: int
    max_samples_per_experiment: int
    reliability_bins: int
    qualitative_samples: int


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def get_visualization_settings() -> VisualizationSettings:
    """Return visualization parameters with optional environment overrides."""
    return VisualizationSettings(
        max_pr_pixels=_env_int("CORAL_MTL_VIS_MAX_PR_PIXELS", 500_000),
        max_samples_per_experiment=_env_int("CORAL_MTL_VIS_MAX_SAMPLES", 16),
        reliability_bins=_env_int("CORAL_MTL_VIS_RELIABILITY_BINS", 15),
        qualitative_samples=_env_int("CORAL_MTL_VIS_QUAL_SAMPLES", 2),
    )
