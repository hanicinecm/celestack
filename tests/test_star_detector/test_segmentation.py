"""Tests for sky segmentation."""

from __future__ import annotations

import numpy as np
import pytest

from celestack.progress import set_progress_factory
from celestack.star_detector._segmentation import (
    closest_segment,
    segment_bfs_tree,
    segment_sky,
    spaced_segments,
)

set_progress_factory(None)


def test_foreground_pixels_labeled_minus_one():
    """Foreground pixels are always labeled -1."""
    mask = np.zeros((64, 64), dtype=bool)
    mask[32:, :] = True  # bottom half foreground
    labels = segment_sky(mask, n_segments=3)
    assert np.all(labels[32:, :] == -1)


def test_sky_pixels_labeled_zero_to_n():
    """Sky pixels receive labels in [0, n_segments)."""
    mask = np.zeros((64, 64), dtype=bool)
    mask[32:, :] = True
    labels = segment_sky(mask, n_segments=4)
    sky_labels = labels[:32, :]
    assert sky_labels.min() >= 0
    assert sky_labels.max() < 4


def test_single_segment():
    """n_segments=1 assigns the entire sky to a single segment."""
    mask = np.zeros((64, 64), dtype=bool)
    mask[48:, :] = True
    labels = segment_sky(mask, n_segments=1)
    sky_labels = labels[:48, :]
    assert np.all(sky_labels == 0)


def test_roughly_equal_segment_sizes():
    """Each segment has roughly equal pixel count (within 30%)."""
    mask = np.zeros((100, 100), dtype=bool)
    mask[80:, :] = True
    n_seg = 5
    labels = segment_sky(mask, n_segments=n_seg)

    sky_labels = labels[labels >= 0]
    unique, counts = np.unique(sky_labels, return_counts=True)
    assert len(unique) == n_seg
    expected = len(sky_labels) / n_seg
    for c in counts:
        assert abs(c - expected) / expected < 0.30


def test_output_shape_matches_input():
    """Label array has the same shape as the input mask."""
    mask = np.zeros((48, 72), dtype=bool)
    labels = segment_sky(mask, n_segments=3)
    assert labels.shape == mask.shape


def test_output_dtype_is_int32():
    """Label array has int32 dtype."""
    mask = np.zeros((32, 32), dtype=bool)
    labels = segment_sky(mask, n_segments=2)
    assert labels.dtype == np.int32


def test_bfs_tree_vertical_split_starts_at_nearest():
    """BFS tree is seeded by the segment closest to start_point."""
    labels = np.array([[0, 0], [0, 0], [1, 1], [1, 1]], dtype=np.int32)
    order = segment_bfs_tree(labels, (0.0, 0.0))
    assert order == [(0, None), (1, 0)]

    order = segment_bfs_tree(labels, (3.0, 0.0))
    assert order == [(1, None), (0, 1)]


def test_bfs_tree_ignores_foreground():
    """Foreground (-1) never appears in the BFS output."""
    labels = np.array([[0, -1, 1], [0, -1, 1]], dtype=np.int32)
    order = segment_bfs_tree(labels, (0.0, 0.0))
    seg_ids = [seg for seg, _ in order]
    assert -1 not in seg_ids
    assert set(seg_ids) == {0, 1}


def test_bfs_tree_four_connectivity_only():
    """Diagonal-only segments are not adjacent and must be reached via a chain."""
    labels = np.array(
        [
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [2, 2, 3, 3],
            [2, 2, 3, 3],
        ],
        dtype=np.int32,
    )
    order = segment_bfs_tree(labels, (0.0, 0.0))
    parent_of = dict(order)
    # 0 is the seed (top-left corner).
    assert parent_of[0] is None
    # 3 is diagonally opposite to 0, so its parent must be 1 or 2 (not 0).
    assert parent_of[3] in {1, 2}


def test_bfs_tree_isolated_segments_still_appear():
    """Segments unreachable via adjacency are appended at the end with parent=None."""
    labels = np.array(
        [
            [0, -1, -1],
            [-1, -1, -1],
            [-1, -1, 1],
        ],
        dtype=np.int32,
    )
    order = segment_bfs_tree(labels, (0.0, 0.0))
    parent_of = dict(order)
    assert parent_of == {0: None, 1: None}
    assert {seg for seg, _ in order} == {0, 1}


def test_bfs_tree_visits_every_sky_segment():
    """BFS output covers every segment exactly once."""
    labels = segment_sky(np.zeros((40, 40), dtype=bool), n_segments=5)
    order = segment_bfs_tree(labels, (0.0, 0.0))
    seg_ids = [seg for seg, _ in order]
    assert len(seg_ids) == len(set(seg_ids)) == 5
    assert set(seg_ids) == set(range(5))


def test_bfs_tree_parent_precedes_child():
    """Every non-seed segment's parent appears earlier in the BFS order."""
    labels = segment_sky(np.zeros((40, 40), dtype=bool), n_segments=5)
    order = segment_bfs_tree(labels, (0.0, 0.0))
    position = {seg: i for i, (seg, _) in enumerate(order)}
    for seg, parent in order:
        if parent is not None:
            assert position[parent] < position[seg]


def test_closest_segment_picks_obvious_winner():
    """Query point inside a segment picks that segment."""
    labels = np.array(
        [
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [2, 2, 3, 3],
            [2, 2, 3, 3],
        ],
        dtype=np.int32,
    )
    assert closest_segment(labels, (0.0, 0.0)) == 0
    assert closest_segment(labels, (0.0, 3.0)) == 1
    assert closest_segment(labels, (3.0, 0.0)) == 2
    assert closest_segment(labels, (3.0, 3.0)) == 3


def test_closest_segment_ignores_foreground():
    """Foreground pixels do not define centroids."""
    labels = np.array(
        [
            [0, -1, -1],
            [-1, -1, -1],
            [-1, -1, 1],
        ],
        dtype=np.int32,
    )
    # Point near the foreground center still resolves to one of 0 or 1.
    result = closest_segment(labels, (1.0, 1.0))
    assert result in (0, 1)


def test_closest_segment_empty_raises():
    """All-foreground labels raise ValueError."""
    labels = np.full((5, 5), -1, dtype=np.int32)
    with pytest.raises(ValueError, match=r"no sky pixels"):
        closest_segment(labels, (0.0, 0.0))


def test_spaced_segments_returns_k_distinct_ids():
    """Result has exactly k distinct, valid segment ids."""
    labels = segment_sky(np.zeros((64, 64), dtype=bool), n_segments=8)
    chosen = spaced_segments(labels, k=4)
    assert len(chosen) == 4
    assert len(set(chosen)) == 4
    for seg in chosen:
        assert 0 <= seg < 8


def test_spaced_segments_picks_corners_on_grid():
    """On a 2x2 segment grid with k=4, all four corners are chosen."""
    labels = np.array(
        [
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [2, 2, 3, 3],
            [2, 2, 3, 3],
        ],
        dtype=np.int32,
    )
    chosen = spaced_segments(labels, k=4)
    assert set(chosen) == {0, 1, 2, 3}


def test_spaced_segments_k_equals_one():
    """k=1 returns the seed segment only."""
    labels = segment_sky(np.zeros((32, 32), dtype=bool), n_segments=5)
    chosen = spaced_segments(labels, k=1)
    assert len(chosen) == 1


def test_spaced_segments_k_too_large_raises():
    """k greater than the number of segments raises ValueError."""
    labels = np.array([[0, 0], [1, 1]], dtype=np.int32)
    with pytest.raises(ValueError, match=r"exceeds the number"):
        spaced_segments(labels, k=3)


def test_spaced_segments_invalid_k_raises():
    """k < 1 raises ValueError."""
    labels = np.array([[0, 0], [1, 1]], dtype=np.int32)
    with pytest.raises(ValueError, match=r"k must be at least 1"):
        spaced_segments(labels, k=0)


def test_spaced_segments_empty_raises():
    """All-foreground labels raise ValueError."""
    labels = np.full((5, 5), -1, dtype=np.int32)
    with pytest.raises(ValueError, match=r"no sky pixels"):
        spaced_segments(labels, k=2)


def test_spaced_segments_deterministic():
    """Same input yields the same selection."""
    labels = segment_sky(np.zeros((48, 48), dtype=bool), n_segments=10)
    assert spaced_segments(labels, k=5) == spaced_segments(labels, k=5)


def test_spaced_segments_spread_beats_arbitrary():
    """Greedy selection has better min pairwise distance than naive order."""
    labels = segment_sky(np.zeros((80, 80), dtype=bool), n_segments=12)
    k = 5
    chosen = spaced_segments(labels, k=k)

    def _min_pairwise(seg_ids: list[int]) -> float:
        centroids = []
        for seg in seg_ids:
            rows, cols = np.where(labels == seg)
            centroids.append((rows.mean(), cols.mean()))
        dists = [
            (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
            for i, a in enumerate(centroids)
            for b in centroids[i + 1 :]
        ]
        return float(min(dists))

    greedy_spread = _min_pairwise(chosen)
    naive_spread = _min_pairwise(list(range(k)))
    assert greedy_spread >= naive_spread
