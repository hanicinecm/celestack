"""DAOStarFinder detection with binary-search threshold tuning."""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
from photutils.detection import DAOStarFinder
from photutils.utils.exceptions import NoDetectionsWarning

from celestack.progress import progress_factory

_MAX_ITERATIONS = 15  # Maximum binary-search depth for detection threshold tuning


def binary_search_threshold(
    image: np.ndarray,
    mask: np.ndarray,
    fwhm: float,
    target_count: int,
    roundness_range: tuple[float, float],
    threshold_bounds: tuple[float, float],
) -> pl.DataFrame:
    """Binary-search the detection threshold to yield closest to *target_count* stars.

    If the target is unreachable (fewer real stars than requested), returns
    the lowest-threshold result within bounds.

    Args:
        image: 2D grayscale array (float64).
        mask: 2D boolean array (``True`` = ignored by DAOStarFinder).
        fwhm: FWHM in image pixels.
        target_count: Desired number of stars.
        roundness_range: (min, max) roundness for DAOStarFinder.
        threshold_bounds: (low, high) threshold range.

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

    for _ in range(_MAX_ITERATIONS):
        mid = (lo + hi) / 2.0
        finder = DAOStarFinder(
            threshold=mid,
            fwhm=fwhm,
            roundlo=roundness_range[0],
            roundhi=roundness_range[1],
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NoDetectionsWarning)
            result = finder(image.astype(np.float64), mask=mask)

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


def detect_in_segments(
    image: np.ndarray,
    segment_labels: np.ndarray,
    target_per_segment: dict[int, int],
    fwhm: float,
    roundness_range: tuple[float, float],
    *,
    threshold_min_sigma: float,
    threshold_max_sigma: float,
) -> pl.DataFrame:
    """Run adaptive detection independently in each sky segment.

    For each segment, builds a mask that exposes only that segment's
    pixels and binary-searches the threshold to match the per-segment
    target count.

    The detection is tracked with a progress bar that updates on completion
    of each segment's detection.

    Args:
        image: 2D grayscale array.
        segment_labels: 2D int32 array (sky pixels 0..N-1, foreground -1).
        target_per_segment: Mapping from segment label to target star count.
        fwhm: FWHM in image pixels.
        roundness_range: (min, max) roundness bounds.
        threshold_min_sigma: Lower threshold bound as a multiple of background σ.
        threshold_max_sigma: Upper threshold bound as a multiple of background σ.

    Returns:
        Combined DataFrame with columns
        ``x, y, flux, fwhm, roundness, threshold, threshold_sigma, segment_id``.
        ``threshold`` is the absolute DAOStarFinder threshold used, and
        ``threshold_sigma`` is the same value expressed as a multiple of
        the per-segment background σ.  Coordinates are in the same pixel
        space as *image*.
    """
    all_frames: list[pl.DataFrame] = []
    float_image = image.astype(np.float64)

    with progress_factory(len(target_per_segment), "Detecting stars") as bar:
        for seg_id, target in sorted(target_per_segment.items()):
            seg_pixels = float_image[segment_labels == seg_id]
            if seg_pixels.size == 0:
                bar.update()
                continue

            med = float(np.median(seg_pixels))
            mad = float(np.median(np.abs(seg_pixels - med)))
            bg_std = 1.4826 * mad
            if bg_std == 0:
                bar.update()
                continue

            threshold_min = threshold_min_sigma * bg_std
            threshold_max = threshold_max_sigma * bg_std

            detection_mask = segment_labels != seg_id

            stars_df = binary_search_threshold(
                float_image,
                detection_mask,
                fwhm,
                target,
                roundness_range,
                (threshold_min, threshold_max),
            )

            if len(stars_df) > 0:
                stars_df = stars_df.with_columns(
                    (pl.col("threshold") / np.float32(bg_std)).alias("threshold_sigma"),
                    pl.lit(np.uint16(seg_id)).alias("segment_id"),
                )
                all_frames.append(stars_df)

            bar.update()

    if not all_frames:
        return pl.DataFrame(
            schema={
                "x": pl.Float32,
                "y": pl.Float32,
                "flux": pl.Float32,
                "fwhm": pl.Float32,
                "roundness": pl.Float32,
                "threshold": pl.Float32,
                "threshold_sigma": pl.Float32,
                "segment_id": pl.UInt16,
            }
        )

    return pl.concat(all_frames)
