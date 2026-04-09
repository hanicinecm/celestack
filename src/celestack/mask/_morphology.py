"""Morphological operations for boolean mask cleanup."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

DEFAULT_NOISE_MAX_SIZE = 8
"""Default maximum component area (in pixels) for noise removal."""


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
    fg_labels, fg_count = ndimage.label(result)
    if fg_count > 0:
        sizes = np.bincount(fg_labels.ravel())
        # sizes[0] is the background label — leave it alone
        small = sizes <= max_size
        small[0] = False
        result[small[fg_labels]] = False

    # Remove small background components (holes)
    bg_labels, bg_count = ndimage.label(~result)
    if bg_count > 0:
        sizes = np.bincount(bg_labels.ravel())
        small = sizes <= max_size
        small[0] = False
        result[small[bg_labels]] = True

    return result
