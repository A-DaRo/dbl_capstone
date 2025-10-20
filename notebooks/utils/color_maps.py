"""Color palette helpers for qualitative visualization panels."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _sample_colormap(name: str, count: int) -> np.ndarray:
    cmap = plt.get_cmap(name)
    if count <= cmap.N:
        indices = np.linspace(0, 1, count)
        return cmap(indices)
    linspace = np.linspace(0, 1, count)
    return cmap(linspace % 1.0)


def build_index_color_map(label_names: Iterable[str], cmap_name: str = "tab20") -> Dict[int, Tuple[float, float, float]]:
    """Assign an RGB color to each label index."""
    labels = list(label_names)
    colors = _sample_colormap(cmap_name, len(labels))
    return {idx: tuple(color[:3]) for idx, color in enumerate(colors)}
