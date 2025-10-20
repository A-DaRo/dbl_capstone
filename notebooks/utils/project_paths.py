"""Utility helpers for resolving project-relative paths used in notebooks.

These helpers centralize resolution logic so that visualization notebooks can
refer to directories without embedding hard-coded relative strings. All path
values returned here are ensured to be absolute ``pathlib.Path`` instances.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict


def get_project_root() -> Path:
    """Return the absolute project root for the Coral-MTL workspace."""
    return Path(__file__).resolve().parents[2]


def get_configs_dir() -> Path:
    """Return the root directory that stores YAML configuration files."""
    return get_project_root() / "configs"


def get_experiments_root() -> Path:
    """Return the directory containing experiment outputs."""
    return get_project_root() / "experiments"


def get_notebooks_root() -> Path:
    """Return the absolute path to the notebooks directory."""
    return get_project_root() / "notebooks"


def get_visual_output_dirs() -> Dict[str, Path]:
    """Return the canonical export directories for generated figures.

    The returned dictionary keys describe the intended consumer of the figures,
    aligning with the project conventions defined in the visualization guide.
    """

    root = get_project_root()
    return {
        "poster": root / "latex" / "Poster_Data_shallange" / "Result-figures",
        "report": root / "latex" / "Methodology" / "Result-figures",
    }


def ensure_output_dirs_exist() -> Dict[str, Path]:
    """Ensure that all visualization export directories exist on disk."""
    directories = get_visual_output_dirs()
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)
    return directories
