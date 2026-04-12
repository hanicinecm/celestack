"""K-Means sky segmentation for adaptive star detection."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from celestack.progress import progress_factory

_RANDOM_STATE = 42


def segment_sky(
    sky_mask: np.ndarray,
    n_segments: int,
) -> np.ndarray:
    """Partition sky pixels into spatial segments via K-Means on coordinates.

    Foreground pixels (where ``sky_mask`` is ``True``) are labeled ``-1``.
    Sky pixels are assigned cluster labels ``0`` through ``n_segments - 1``.
    Progress bar is displayed for the duration of the K-Means fit.

    Args:
        sky_mask: 2D boolean array where ``True`` marks foreground.
        n_segments: Number of spatial segments to create.

    Returns:
        2D int32 label array with the same shape as *sky_mask*.
    """
    with progress_factory(total=None, description="Segmenting sky"):
        sky_rows, sky_cols = np.where(~sky_mask)
        coords = np.column_stack([sky_rows, sky_cols]).astype(np.float32)

        kmeans = KMeans(
            n_clusters=n_segments, random_state=_RANDOM_STATE, n_init="auto"
        )
        cluster_ids = kmeans.fit_predict(coords).astype(np.int32)

        labels = np.full(sky_mask.shape, -1, dtype=np.int32)
        labels[sky_rows, sky_cols] = cluster_ids

    return labels
