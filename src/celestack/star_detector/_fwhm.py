"""FWHM auto-estimation via multi-region sweep."""

from __future__ import annotations

import warnings

import numpy as np
from photutils.detection import DAOStarFinder
from photutils.utils.exceptions import NoDetectionsWarning

_N_SUBREGIONS = 5
"""Number of sky sub-regions sampled during FWHM estimation."""

_SUBREGION_DIVISOR = 4
"""Sub-region patch size as a fraction of the bounding-box extent
(patch_size = max(height, width) // _SUBREGION_DIVISOR)."""


def _pick_subregions(
    sky_mask: np.ndarray,
    n: int,
) -> list[tuple[int, int, int, int]]:
    """Return (y0, y1, x0, x1) bounding boxes for *n* sub-regions of the sky bbox.

    Picks 4 corners and a center region from the bounding box of all sky
    pixels. Each sub-region is a square patch roughly 1/_SUBREGION_DIVISOR of
    the bounding-box extent on a side.

    Args:
        sky_mask: 2D boolean array where ``True`` marks foreground.
        n: Number of sub-regions to return (up to 5).

    Returns:
        List of ``(y0, y1, x0, x1)`` tuples.
    """
    sky_rows, sky_cols = np.where(~sky_mask)
    min_r, max_r = int(sky_rows.min()), int(sky_rows.max())
    min_c, max_c = int(sky_cols.min()), int(sky_cols.max())

    extent_r = max_r - min_r + 1
    extent_c = max_c - min_c + 1
    patch_size = max(extent_r, extent_c) // _SUBREGION_DIVISOR

    mid_r = (min_r + max_r) // 2
    mid_c = (min_c + max_c) // 2

    def _clip(y: int, x: int) -> tuple[int, int, int, int]:
        y0 = max(min_r, y)
        x0 = max(min_c, x)
        y1 = min(max_r + 1, y + patch_size)
        x1 = min(max_c + 1, x + patch_size)
        return (y0, y1, x0, x1)

    candidates = [
        _clip(min_r, min_c),  # top-left
        _clip(min_r, max_c - patch_size + 1),  # top-right
        _clip(max_r - patch_size + 1, min_c),  # bottom-left
        _clip(max_r - patch_size + 1, max_c - patch_size + 1),  # bottom-right
        _clip(mid_r - patch_size // 2, mid_c - patch_size // 2),  # center
    ]
    return candidates[:n]


def estimate_fwhm(
    image: np.ndarray,
    sky_mask: np.ndarray,
    fwhm_range: tuple[float, float],
    n_fwhm_steps: int,
    threshold_sigma: float,
) -> float:
    """Estimate the optimal FWHM for DAOStarFinder on the given image.

    Picks several sub-regions from the sky area, sweeps a range of FWHM
    values at a fixed high threshold, and returns the mean FWHM that
    maximises detections across all sub-regions.

    Args:
        image: 2D grayscale array.
        sky_mask: 2D boolean array (``True`` = foreground).
        fwhm_range: (min, max) FWHM in image pixels.
        n_fwhm_steps: Number of FWHM values to sweep.
        threshold_sigma: Detection threshold as a multiple of the
            background standard deviation.

    Returns:
        Estimated FWHM in image pixels.
    """
    subregions = _pick_subregions(sky_mask, _N_SUBREGIONS)
    fwhm_values = np.linspace(fwhm_range[0], fwhm_range[1], n_fwhm_steps)

    best_fwhms: list[float] = []

    for y0, y1, x0, x1 in subregions:
        patch = image[y0:y1, x0:x1].astype(np.float64)
        patch_mask = sky_mask[y0:y1, x0:x1]

        sky_pixels = patch[~patch_mask]
        if sky_pixels.size == 0:
            continue

        med = float(np.median(sky_pixels))
        mad = float(np.median(np.abs(sky_pixels - med)))
        bg_std = 1.4826 * mad
        if bg_std == 0:
            continue

        threshold = threshold_sigma * bg_std
        best_count = -1
        best_fwhm = float(fwhm_values[len(fwhm_values) // 2])

        for fwhm in fwhm_values:
            finder = DAOStarFinder(threshold=threshold, fwhm=float(fwhm))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", NoDetectionsWarning)
                result = finder(patch, mask=patch_mask)
            count = len(result) if result is not None else 0
            if count > best_count:
                best_count = count
                best_fwhm = float(fwhm)

        best_fwhms.append(best_fwhm)

    if not best_fwhms:
        return float(np.mean(fwhm_values))

    return float(np.mean(best_fwhms))
