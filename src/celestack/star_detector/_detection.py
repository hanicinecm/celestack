"""DAOStarFinder detection with binary-search threshold tuning."""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
from photutils.detection import DAOStarFinder
from photutils.utils.exceptions import NoDetectionsWarning

from celestack.exceptions import StarDetectorError
from celestack.star_detector._photometry import compute_local_flux

_MAX_ITERATIONS = 15  # Maximum binary-search depth for detection threshold tuning


def _binary_search_threshold(
    image: np.ndarray,
    mask: np.ndarray,
    fwhm: float,
    target_count: int,
    roundness_range: tuple[float, float],
    threshold_bounds: tuple[float, float],
    initial_threshold: float | None = None,
) -> pl.DataFrame:
    """Binary-search the detection threshold to yield closest to *target_count* stars.

    If the target is unreachable (fewer real stars than requested), returns
    the lowest-threshold result within bounds.

    Args:
        image: 2D grayscale array.
        mask: 2D boolean array (``True`` = ignored by DAOStarFinder).
        fwhm: FWHM in image pixels.
        target_count: Desired number of stars.
        roundness_range: (min, max) roundness for DAOStarFinder.
        threshold_bounds: (low, high) threshold range.
        initial_threshold: Optional seed for the first binary-search guess.
            Clamped into *threshold_bounds* if provided.  Use this to warm
            up the search with a prior close to the expected answer (e.g.
            a neighboring segment's result).  When ``None``, the search
            starts at the midpoint of *threshold_bounds*.

    Returns:
        DataFrame with columns ``x, y, flux, fwhm, roundness, threshold``,
        where ``threshold`` is the absolute DAOStarFinder threshold that
        produced this row (constant across all rows of a single call).
    """
    lo, hi = threshold_bounds
    best_df = pl.DataFrame(
        schema={
            "x": pl.Float32,
            "y": pl.Float32,
            "flux": pl.Float32,
            "fwhm": pl.Float32,
            "roundness": pl.Float32,
            "threshold": pl.Float32,
        }
    )
    best_diff = float("inf")
    next_mid: float | None = (
        min(max(initial_threshold, lo), hi) if initial_threshold is not None else None
    )

    for _ in range(_MAX_ITERATIONS):
        mid = next_mid if next_mid is not None else (lo + hi) / 2.0
        next_mid = None
        finder = DAOStarFinder(
            threshold=mid,
            fwhm=fwhm,
            roundlo=roundness_range[0],
            roundhi=roundness_range[1],
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NoDetectionsWarning)
            result = finder(image, mask=mask)

        if result is None or len(result) == 0:
            hi = mid
            continue

        count = len(result)
        df = pl.DataFrame(
            {
                "x": np.array(result["xcentroid"], dtype=np.float32),
                "y": np.array(result["ycentroid"], dtype=np.float32),
                "flux": np.array(result["flux"], dtype=np.float32),
                "fwhm": np.full(count, fwhm, dtype=np.float32),
                "roundness": np.array(result["roundness1"], dtype=np.float32),
                "threshold": np.full(count, mid, dtype=np.float32),
            }
        )

        diff = abs(count - target_count)
        if diff < best_diff:
            best_diff = diff
            best_df = df

        if count < target_count:
            hi = mid
        elif count > target_count:
            lo = mid
        else:
            break

    return best_df


def detect_in_segment(
    image: np.ndarray,
    segment_labels: np.ndarray,
    seg_id: int,
    target_count: int,
    fwhm: float,
    roundness_range: tuple[float, float],
    *,
    threshold_min_sigma: float,
    threshold_max_sigma: float,
    initial_threshold_sigma: float | None = None,
) -> pl.DataFrame:
    """Run adaptive detection inside a single sky segment.

    Computes the segment's background σ from its own pixels, builds the
    DAOStarFinder mask that exposes only this segment, and binary-searches
    the threshold to match *target_count*.  The returned DataFrame is
    already annotated with ``threshold_sigma`` and ``segment_id`` so the
    caller can concatenate results across segments verbatim.

    Args:
        image: 2D grayscale array.
        segment_labels: 2D int32 array (sky pixels 0..N-1, foreground -1).
        seg_id: Label of the segment to detect in.
        target_count: Desired number of stars in this segment.
        fwhm: FWHM in image pixels.
        roundness_range: (min, max) roundness bounds.
        threshold_min_sigma: Lower threshold bound as a multiple of background σ.
        threshold_max_sigma: Upper threshold bound as a multiple of background σ.
        initial_threshold_sigma: Optional seed for the binary search's first
            guess, expressed as a multiple of background σ.  Useful when a
            prior (e.g. a neighboring segment's result) suggests a likely
            threshold.  Clamped into the ``[min, max]`` bounds internally.

    Returns:
        DataFrame with columns
        ``x, y, flux, fwhm, roundness, threshold, threshold_sigma, segment_id``.

    Raises:
        StarDetectorError: If the segment contains no pixels or has
            degenerate (zero) background spread.
    """
    seg_pixels = image[segment_labels == seg_id]
    if seg_pixels.size == 0:
        msg = f"Segment {seg_id} contains no pixels"
        raise StarDetectorError(msg)

    med = float(np.median(seg_pixels))
    mad = float(np.median(np.abs(seg_pixels - med)))
    bg_std = 1.4826 * mad
    if bg_std == 0:
        msg = f"Segment {seg_id} has zero background spread (MAD=0)"
        raise StarDetectorError(msg)

    detection_mask = segment_labels != seg_id
    initial_threshold = (
        initial_threshold_sigma * bg_std
        if initial_threshold_sigma is not None
        else None
    )
    stars_df = _binary_search_threshold(
        image,
        detection_mask,
        fwhm,
        target_count,
        roundness_range,
        (threshold_min_sigma * bg_std, threshold_max_sigma * bg_std),
        initial_threshold=initial_threshold,
    )

    flux_local = compute_local_flux(
        image,
        stars_df["x"].to_numpy(),
        stars_df["y"].to_numpy(),
        fwhm,
    )

    return stars_df.with_columns(
        (pl.col("threshold") / pl.lit(float(bg_std)))
        .cast(pl.Float32)
        .alias("threshold_sigma"),
        pl.lit(np.uint16(seg_id)).alias("segment_id"),
        pl.Series("flux_local", flux_local, dtype=pl.Float32),
    )
