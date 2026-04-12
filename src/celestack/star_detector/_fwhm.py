"""FWHM auto-estimation via a spaced multi-segment sample."""

from __future__ import annotations

import warnings

import numpy as np
from photutils.detection import DAOStarFinder
from photutils.utils.exceptions import NoDetectionsWarning

from celestack.progress import progress_factory
from celestack.star_detector._segmentation import spaced_segments

_N_SAMPLE_SEGMENTS = 5


def _sample_mask(segment_labels: np.ndarray, n: int) -> np.ndarray:
    """Build a DAOStarFinder mask exposing *n* well-spread sky segments.

    The returned mask is ``True`` everywhere except on the chosen segments,
    i.e. it hides foreground and all non-selected segments from DAOStarFinder.
    Selection uses :func:`spaced_segments` to maximise spatial dispersion
    across the sky.

    Args:
        segment_labels: 2D int32 label array, as produced by
            :func:`segment_sky`.
        n: Number of segments to expose (capped at the number of available
            segments).

    Returns:
        Boolean mask with the same shape as *segment_labels*; ``True`` marks
        hidden pixels.
    """
    available = int((np.unique(segment_labels) >= 0).sum())
    k = min(n, available)
    chosen = spaced_segments(segment_labels, k=k)
    return ~np.isin(segment_labels, chosen)


def estimate_fwhm(
    image: np.ndarray,
    segment_labels: np.ndarray,
    fwhm_range: tuple[float, float],
    n_fwhm_steps: int,
    threshold_sigma: float,
) -> float:
    """Estimate the optimal FWHM for DAOStarFinder on the given image.

    Builds a mask that exposes a handful of spatially-spread sky segments,
    computes background statistics on the exposed pixels, then sweeps a
    range of FWHM values at a fixed high threshold and returns the FWHM
    that maximises detection count.

    Args:
        image: 2D grayscale array.
        segment_labels: 2D int32 label array, as produced by
            :func:`segment_sky`. Foreground must be labeled ``-1``.
        fwhm_range: ``(min, max)`` FWHM in image pixels.
        n_fwhm_steps: Number of FWHM values to sweep.
        threshold_sigma: Detection threshold as a multiple of the
            background standard deviation.

    Returns:
        Estimated FWHM in image pixels.  Falls back to the midpoint of
        *fwhm_range* if background statistics are degenerate.
    """
    fwhm_values = np.linspace(fwhm_range[0], fwhm_range[1], n_fwhm_steps)

    mask = _sample_mask(segment_labels, _N_SAMPLE_SEGMENTS)
    float_image = image.astype(np.float64)
    sample_pixels = float_image[~mask]

    if sample_pixels.size == 0:
        msg = "FWHM estimation failed: no sky pixels available for sampling"
        raise ValueError(msg)

    med = float(np.median(sample_pixels))
    mad = float(np.median(np.abs(sample_pixels - med)))
    bg_std = 1.4826 * mad
    if bg_std == 0:
        fallback = float(np.mean(fwhm_values))
        return fallback

    threshold = threshold_sigma * bg_std
    best_count = -1
    best_fwhm = fwhm_values[0]

    with progress_factory(n_fwhm_steps, "Estimating FWHM") as bar:
        for fwhm in fwhm_values:
            finder = DAOStarFinder(threshold=threshold, fwhm=float(fwhm))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", NoDetectionsWarning)
                result = finder(float_image, mask=mask)
            count = len(result) if result is not None else 0
            if count > best_count:
                best_count = count
                best_fwhm = float(fwhm)
            bar.update()

    return best_fwhm
