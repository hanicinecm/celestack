"""Post-detection filtering: edge exclusion, proximity, and capping."""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy import ndimage


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
