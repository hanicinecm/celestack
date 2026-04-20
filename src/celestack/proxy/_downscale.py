"""Integer-factor downscaling helpers for proxy construction."""

from __future__ import annotations

import numpy as np


def block_average(array: np.ndarray, factor: int) -> np.ndarray:
    """Downscale an image by averaging non-overlapping pixel blocks.

    Trailing rows/columns that do not fit a whole block are trimmed.

    Args:
        array: 2D or 3D array with shape (H, W) or (H, W, C).
        factor: Integer downscale factor; must be >= 1.

    Returns:
        Downscaled array. Shape is (H // factor, W // factor[, C]).
        The dtype is preserved when possible (NumPy's ``mean`` may
        upcast integer inputs to float).

    Raises:
        ValueError: If *factor* is less than 1.
    """
    if factor < 1:
        msg = "downscale_factor must be >= 1"
        raise ValueError(msg)
    if factor == 1:
        return array.copy()

    height, width = array.shape[:2]
    new_height = max(1, height // factor)
    new_width = max(1, width // factor)
    trim_height = new_height * factor
    trim_width = new_width * factor
    trimmed = array[:trim_height, :trim_width]

    if array.ndim == 2:
        reshaped = trimmed.reshape(new_height, factor, new_width, factor)
        return reshaped.mean(axis=(1, 3))

    channels = array.shape[2]
    reshaped = trimmed.reshape(new_height, factor, new_width, factor, channels)
    return reshaped.mean(axis=(1, 3))


def majority_vote(mask: np.ndarray, factor: int) -> np.ndarray:
    """Downscale a boolean mask by majority vote (>50% foreground).

    Blocks with a strictly greater than 50% foreground fraction become
    ``True`` in the output; all other blocks become ``False``. Trailing
    rows/columns that do not fit a whole block are trimmed.

    Args:
        mask: 2D boolean array with shape (H, W).
        factor: Integer downscale factor; must be >= 1.

    Returns:
        Downscaled boolean array with shape (H // factor, W // factor).

    Raises:
        ValueError: If *factor* is less than 1 or *mask* is not 2D boolean.
    """
    if factor < 1:
        msg = "downscale_factor must be >= 1"
        raise ValueError(msg)
    if mask.ndim != 2 or mask.dtype != np.bool_:
        msg = "majority_vote requires a 2D boolean array"
        raise ValueError(msg)
    if factor == 1:
        return mask.copy()

    height, width = mask.shape
    new_height = max(1, height // factor)
    new_width = max(1, width // factor)
    trim_height = new_height * factor
    trim_width = new_width * factor
    trimmed = mask[:trim_height, :trim_width]

    reshaped = trimmed.reshape(new_height, factor, new_width, factor)
    fraction = reshaped.mean(axis=(1, 3))
    return fraction > 0.5
