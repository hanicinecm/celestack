"""Morphological operations for boolean mask cleanup."""

from __future__ import annotations

from typing import cast

import numpy as np
from scipy import ndimage


def _small_component_metadata(
    mask: np.ndarray,
    max_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Label connected components and mark those up to the size threshold.

    Args:
        mask: 2D boolean array where ``True`` denotes the phase to label.
        max_size: Maximum component area (in pixels) to flag as small.

    Returns:
        Tuple ``(labels, sizes, small)`` where:
        - ``labels`` is the integer label image,
        - ``sizes`` contains the pixel area for each label id,
        - ``small`` is a boolean lookup array with ``small[label_id]`` set
          when the component area is <= ``max_size`` (label 0 is always False).
    """
    labeled = cast(tuple[np.ndarray, int], ndimage.label(mask))
    labels = np.asarray(labeled[0], dtype=np.intp)
    count = int(labeled[1])
    if count == 0:
        sizes = np.zeros(1, dtype=np.intp)
        small = np.zeros(1, dtype=np.bool_)
        return labels, sizes, small

    sizes = np.bincount(labels.ravel()).astype(np.intp, copy=False)
    small = sizes <= max_size
    small[0] = False
    return labels, sizes, small


def remove_small_components(mask: np.ndarray, max_size: int) -> np.ndarray:
    """Remove small connected components from both phases of a boolean mask.

    Foreground regions with area <= *max_size* are cleared to background.
    Background regions with area <= *max_size* are filled to foreground.
    Uses 4-connectivity (cross structuring element).

    Args:
        mask: 2D boolean array.
        max_size: Maximum component area (in pixels) to remove.

    Returns:
        Cleaned boolean array (copy of input).
    """
    if max_size < 1:
        msg = "max_size must be >= 1"
        raise ValueError(msg)

    result = mask.copy()

    # Remove small foreground components
    fg_labels, _, small = _small_component_metadata(result, max_size)
    if small.any():
        result[small[fg_labels]] = False

    # Remove small background components (holes)
    bg_labels, _, small = _small_component_metadata(~result, max_size)
    if small.any():
        result[small[bg_labels]] = True

    return result
