"""K-Means sky segmentation and segment adjacency for adaptive detection."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.cluster import KMeans

from celestack.progress import progress_factory


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

        kmeans = KMeans(n_clusters=n_segments, random_state=42, n_init="auto")
        cluster_ids = kmeans.fit_predict(coords).astype(np.int32)

        labels = np.full(sky_mask.shape, -1, dtype=np.int32)
        labels[sky_rows, sky_cols] = cluster_ids

    return labels


def segment_adjacency(labels: np.ndarray) -> dict[int, set[int]]:
    """Build the 4-connected adjacency graph of segments in a label array.

    Two segments are considered neighbors if any of their pixels share a
    horizontal or vertical edge.  Foreground pixels (label ``-1``) are
    ignored and never appear in the graph.  Every segment present in
    *labels* appears as a key, even if it has no neighbors.

    Args:
        labels: 2D int32 label array, as produced by :func:`segment_sky`.

    Returns:
        Mapping from segment id to the set of its neighboring segment ids.
    """
    adjacency: dict[int, set[int]] = defaultdict(set)

    # Ensure every segment is a key, even isolated ones.
    for seg in np.unique(labels):
        if seg >= 0:
            adjacency[int(seg)] = set()

    # Horizontal neighbor pairs.
    left, right = labels[:, :-1], labels[:, 1:]
    _collect_pairs(left, right, adjacency)

    # Vertical neighbor pairs.
    top, bottom = labels[:-1, :], labels[1:, :]
    _collect_pairs(top, bottom, adjacency)

    return dict(adjacency)


def _collect_pairs(
    a: np.ndarray,
    b: np.ndarray,
    adjacency: dict[int, set[int]],
) -> None:
    """Record neighbor pairs between two aligned label slices.

    Pairs where either side is foreground (``-1``) or where both labels are
    equal are skipped.  Updates *adjacency* in place.

    Args:
        a: First label slice.
        b: Second label slice, aligned with *a*.
        adjacency: Adjacency map to update in place.
    """
    differ = (a != b) & (a >= 0) & (b >= 0)
    pairs = np.unique(np.stack([a[differ], b[differ]], axis=1), axis=0)
    for seg_a, seg_b in pairs:
        adjacency[int(seg_a)].add(int(seg_b))
        adjacency[int(seg_b)].add(int(seg_a))
