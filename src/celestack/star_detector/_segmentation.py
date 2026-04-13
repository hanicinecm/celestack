"""K-Means sky segmentation and segment adjacency for adaptive detection."""

from __future__ import annotations

from collections import defaultdict, deque

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


def segment_bfs_tree(
    labels: np.ndarray,
    start_point: tuple[float, float],
) -> list[tuple[int, int | None]]:
    """Return segments in BFS order with each segment's parent id.

    The BFS starts at the segment whose centroid is closest to *start_point*
    (via :func:`closest_segment`) and traverses the 4-connected segment
    adjacency graph.  Each entry is ``(segment_id, parent_id)`` — the seed
    segment has ``parent_id=None``, and any segments unreachable from the
    seed are appended at the end with ``parent_id=None`` (same treatment).

    Args:
        labels: 2D int32 label array, as produced by :func:`segment_sky`.
        start_point: ``(row, col)`` seed for the BFS.

    Returns:
        List of ``(segment_id, parent_id)`` pairs in BFS visitation order.

    Raises:
        ValueError: If *labels* contains no sky pixels.
    """
    adjacency: dict[int, set[int]] = defaultdict(set)
    all_segments: set[int] = set()
    for seg in np.unique(labels):
        if seg >= 0:
            adjacency[int(seg)] = set()
            all_segments.add(int(seg))

    left, right = labels[:, :-1], labels[:, 1:]
    _collect_pairs(left, right, adjacency)
    top, bottom = labels[:-1, :], labels[1:, :]
    _collect_pairs(top, bottom, adjacency)

    seed = closest_segment(labels, start_point)
    parent: dict[int, int | None] = {seed: None}
    order: list[tuple[int, int | None]] = []
    queue: deque[int] = deque([seed])
    while queue:
        seg = queue.popleft()
        order.append((seg, parent[seg]))
        for nb in adjacency[seg]:
            if nb not in parent:
                parent[nb] = seg
                queue.append(nb)

    # Orphans (not reachable via adjacency) fall back to no parent.
    for seg in all_segments:
        if seg not in parent:
            parent[seg] = None
            order.append((seg, None))

    return order


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


def closest_segment(labels: np.ndarray, point: tuple[float, float]) -> int:
    """Return the segment id whose centroid is closest to *point*.

    Distance is Euclidean in pixel ``(row, col)`` units.  *point* need not
    lie on a sky pixel — only centroid proximity matters.  Centroid-based
    proximity aligns naturally with K-Means Voronoi-like segments; other
    segmentation schemes with strongly non-convex regions may mispick.

    Args:
        labels: 2D int32 label array, as produced by :func:`segment_sky`.
        point: ``(row, col)`` coordinates of the query point.

    Returns:
        Segment id of the closest segment.

    Raises:
        ValueError: If *labels* contains no sky pixels.
    """
    sky_rows, sky_cols = np.where(labels >= 0)
    if sky_rows.size == 0:
        msg = "labels contains no sky pixels"
        raise ValueError(msg)

    seg_ids = labels[sky_rows, sky_cols]
    unique_ids, inverse = np.unique(seg_ids, return_inverse=True)

    row_sums = np.bincount(inverse, weights=sky_rows)
    col_sums = np.bincount(inverse, weights=sky_cols)
    counts = np.bincount(inverse)

    centroid_rows = row_sums / counts
    centroid_cols = col_sums / counts

    dr = centroid_rows - point[0]
    dc = centroid_cols - point[1]
    sq_dist = dr * dr + dc * dc

    return int(unique_ids[int(np.argmin(sq_dist))])


def spaced_segments(labels: np.ndarray, k: int) -> list[int]:
    """Pick *k* segments with maximally-spread centroids.

    Uses Gonzalez's greedy farthest-point sampling: seeds with the segment
    whose centroid is closest to the sky's overall centroid, then repeatedly
    adds the segment that maximizes the minimum Euclidean distance to the
    already-chosen set.  The result is a 2-approximation to the optimal
    max-min dispersion, which in practice is near-optimal for K-Means-style
    Voronoi segments.

    Args:
        labels: 2D int32 label array, as produced by :func:`segment_sky`.
        k: Number of segments to select.

    Returns:
        List of *k* segment ids in the order they were selected.

    Raises:
        ValueError: If *k* < 1, *labels* has no sky pixels, or there are
            fewer than *k* distinct segments available.
    """
    if k < 1:
        msg = "k must be at least 1"
        raise ValueError(msg)

    sky_rows, sky_cols = np.where(labels >= 0)
    if sky_rows.size == 0:
        msg = "labels contains no sky pixels"
        raise ValueError(msg)

    seg_ids = labels[sky_rows, sky_cols]
    unique_ids, inverse = np.unique(seg_ids, return_inverse=True)
    n_segments = unique_ids.size

    if k > n_segments:
        msg = f"k={k} exceeds the number of available segments ({n_segments})"
        raise ValueError(msg)

    counts = np.bincount(inverse)
    centroid_rows = np.bincount(inverse, weights=sky_rows) / counts
    centroid_cols = np.bincount(inverse, weights=sky_cols) / counts

    # Seed: segment whose centroid is closest to the sky's overall centroid.
    sky_cr = float(sky_rows.mean())
    sky_cc = float(sky_cols.mean())
    seed_dr = centroid_rows - sky_cr
    seed_dc = centroid_cols - sky_cc
    seed_idx = int(np.argmin(seed_dr * seed_dr + seed_dc * seed_dc))

    chosen: list[int] = [seed_idx]
    min_sq_dist = (centroid_rows - centroid_rows[seed_idx]) ** 2 + (
        centroid_cols - centroid_cols[seed_idx]
    ) ** 2

    while len(chosen) < k:
        next_idx = int(np.argmax(min_sq_dist))
        chosen.append(next_idx)
        new_sq_dist = (centroid_rows - centroid_rows[next_idx]) ** 2 + (
            centroid_cols - centroid_cols[next_idx]
        ) ** 2
        min_sq_dist = np.minimum(min_sq_dist, new_sq_dist)

    return [int(unique_ids[i]) for i in chosen]
