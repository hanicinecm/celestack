"""Photutils-based background subtraction for proxy frames."""

from __future__ import annotations

import numpy as np
from photutils.background import Background2D


def subtract_background(
    array: np.ndarray,
    mask: np.ndarray,
    *,
    box_size: int,
    filter_size: int,
) -> np.ndarray:
    """Subtract a 2D sky background model from an array and zero masked pixels.

    Builds a :class:`photutils.background.Background2D` model over the
    unmasked (background) pixels of *array* and subtracts it. The
    foreground pixels (``mask == True``) are then zeroed in the result so
    the mask region does not leak signal into downstream statistics.

    Args:
        array: 2D float array of shape (H, W). Values should already be
            rescaled into a reasonable float range (typically ~[0, 1]).
        mask: 2D boolean foreground mask of the same shape. ``True``
            marks foreground (excluded from the background fit and zeroed
            in the output).
        box_size: ``photutils.Background2D`` box size in array pixels.
        filter_size: Median filter window for the ``Background2D`` mesh.

    Returns:
        A new float16 array with the 2D background subtracted and
        foreground pixels set to ``0.0``.
    """
    if array.shape != mask.shape:
        msg = (
            f"Shape mismatch: array has shape {array.shape}, "
            f"mask has shape {mask.shape}"
        )
        raise ValueError(msg)

    bkg = Background2D(
        array.astype(np.float32, copy=False),
        box_size=box_size,
        filter_size=filter_size,
        mask=mask,
    )
    result = array.astype(np.float16) - bkg.background.astype(np.float16)
    result[mask] = np.float16(0.0)
    return result
