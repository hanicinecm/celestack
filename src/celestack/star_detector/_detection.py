"""DAOStarFinder detection, filtering, and the adaptive refinement loop."""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
from photutils.detection import DAOStarFinder
from photutils.utils.exceptions import NoDetectionsWarning
from scipy import ndimage

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


def filter_stars(
    stars: pl.DataFrame,
    sky_mask: np.ndarray,
    target_stars: int,
    min_separation: float,
    edge_margin: int,
) -> pl.DataFrame:
    """Filter detected stars by edge proximity, mutual separation, and count cap.

    All coordinates in *stars* (``x``, ``y``) must be in the same pixel
    space as *sky_mask*.  The filtering pipeline is:

    1. Edge exclusion (frame edges and mask boundary).
    2. Proximity filter (greedy, brightest-first).
    3. Cap to *target_stars* brightest.

    Args:
        stars: DataFrame with ``x``, ``y``, ``flux_local`` columns.
        sky_mask: 2D boolean mask (``True`` = foreground).
        target_stars: Maximum number of stars to keep.
        min_separation: Minimum distance in pixels between stars.
        edge_margin: Exclusion zone in pixels around frame and mask edges.

    Returns:
        Filtered DataFrame, sorted by descending local flux.
    """
    if len(stars) == 0:
        return stars

    h, w = sky_mask.shape

    # --- 1. Edge exclusion ---
    stars = stars.filter(
        (pl.col("x") >= edge_margin)
        & (pl.col("x") < w - edge_margin)
        & (pl.col("y") >= edge_margin)
        & (pl.col("y") < h - edge_margin)
    )

    if len(stars) == 0:
        return stars

    # Mask boundary exclusion: dilate foreground by 1px to find boundary.
    dilated = ndimage.binary_dilation(sky_mask, iterations=1)
    boundary = dilated & ~sky_mask
    boundary_yx = np.argwhere(boundary)

    if boundary_yx.size > 0:
        boundary_y = boundary_yx[:, 0].astype(np.float64)
        boundary_x = boundary_yx[:, 1].astype(np.float64)

        star_x = stars["x"].to_numpy()
        star_y = stars["y"].to_numpy()

        keep = np.ones(len(stars), dtype=bool)
        for i in range(len(stars)):
            dists = np.sqrt(
                (boundary_x - star_x[i]) ** 2 + (boundary_y - star_y[i]) ** 2
            )
            if dists.min() < edge_margin:
                keep[i] = False

        stars = stars.filter(pl.Series(keep))

    if len(stars) == 0:
        return stars

    # --- 2. Proximity filter (greedy, brightest-first) ---
    stars = stars.sort("flux_local", descending=True)
    xs = stars["x"].to_numpy()
    ys = stars["y"].to_numpy()

    accepted = np.zeros(len(stars), dtype=bool)
    for i in range(len(stars)):
        if i == 0:
            accepted[i] = True
            continue
        prev_accepted = np.where(accepted[:i])[0]
        if prev_accepted.size == 0:
            accepted[i] = True
            continue
        dists = np.sqrt(
            (xs[prev_accepted] - xs[i]) ** 2 + (ys[prev_accepted] - ys[i]) ** 2
        )
        if dists.min() >= min_separation:
            accepted[i] = True

    stars = stars.filter(pl.Series(accepted))

    # --- 3. Cap ---
    stars = stars.sort("flux_local", descending=True).head(target_stars)

    return stars


def detect_in_segment_adaptive(
    image: np.ndarray,
    segment_labels: np.ndarray,
    seg_id: int,
    sky_mask: np.ndarray,
    seg_target: int,
    fwhm: float,
    roundness_range: tuple[float, float],
    *,
    min_separation: float,
    edge_margin: int,
    threshold_min_sigma: float,
    threshold_max_sigma: float,
    overdetect_factor: float,
    initial_threshold_sigma: float | None = None,
    max_refinements: int = 3,
) -> pl.DataFrame | None:
    """Detect and filter stars in one segment, refining until the target is met.

    Wraps :func:`detect_in_segment` and :func:`filter_stars` in an adaptive
    loop: each pass over-detects by *overdetect_factor*, filters, and — if
    the post-filter count falls short — grows the raw request and narrows
    the threshold ceiling for the next attempt.  The best (largest)
    filtered result across attempts is returned.

    Args:
        image: 2D grayscale array.
        segment_labels: 2D int32 array (sky pixels 0..N-1, foreground -1).
        seg_id: Label of the segment to detect in.
        sky_mask: 2D boolean foreground mask used by :func:`filter_stars`.
        seg_target: Desired post-filter star count for this segment.
        fwhm: FWHM in image pixels.
        roundness_range: (min, max) roundness bounds.
        min_separation: Minimum pixel distance between kept stars.
        edge_margin: Exclusion zone around frame and mask edges.
        threshold_min_sigma: Lower threshold bound as a multiple of background σ.
        threshold_max_sigma: Upper threshold bound as a multiple of background σ.
        overdetect_factor: Initial raw-target multiplier (>= 1.0).
        initial_threshold_sigma: Optional seed for the binary search on the
            first attempt (e.g. a neighboring segment's result).
        max_refinements: Maximum number of detect→filter attempts.

    Returns:
        The best filtered DataFrame, or ``None`` if no stars were detected
        at all.
    """
    raw_target = max(1, int(round(seg_target * overdetect_factor)))
    best: pl.DataFrame | None = None
    current_max_sigma = threshold_max_sigma

    shortfall_refinement_factor = 3

    for i in range(max_refinements):
        raw = detect_in_segment(
            image,
            segment_labels,
            seg_id,
            raw_target,
            fwhm,
            roundness_range,
            threshold_min_sigma=threshold_min_sigma,
            threshold_max_sigma=current_max_sigma,
            initial_threshold_sigma=initial_threshold_sigma if i == 0 else None,
        )
        if len(raw) == 0:
            break

        filtered_seg = filter_stars(
            raw, sky_mask, seg_target, min_separation, edge_margin
        )

        if best is None or len(filtered_seg) > len(best):
            best = filtered_seg
        if len(filtered_seg) >= seg_target or len(raw) < raw_target:
            # Either we hit the target, or the segment is detection-limited
            # (can't produce more raw stars even if we ask) — no point retrying.
            break

        # Grow the raw request by the observed shortfall multiplied by the
        # refinement factor — hopefully enough slack to absorb another round
        # of filtering.
        shortfall = seg_target - len(filtered_seg)
        raw_target += max(1, shortfall_refinement_factor * shortfall)
        # Narrow the threshold search: we need more stars next time, so
        # the new upper bound is the threshold that just produced too few.
        current_max_sigma = float(raw["threshold_sigma"][0])

    if best is None or len(best) == 0:
        return None
    return best
