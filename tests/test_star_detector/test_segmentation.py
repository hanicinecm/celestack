"""Tests for sky segmentation."""

from __future__ import annotations

import numpy as np
import pytest

from celestack.progress import set_progress_factory
from celestack.star_detector._segmentation import (
    closest_segment,
    segment_adjacency,
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


def test_adjacency_simple_vertical_split():
    """Two vertically-stacked segments are neighbors."""
    labels = np.array([[0, 0], [0, 0], [1, 1], [1, 1]], dtype=np.int32)
    adj = segment_adjacency(labels)
    assert adj == {0: {1}, 1: {0}}


def test_adjacency_ignores_foreground():
    """Foreground (-1) never appears in the adjacency graph."""
    labels = np.array([[0, -1, 1], [0, -1, 1]], dtype=np.int32)
    adj = segment_adjacency(labels)
    assert -1 not in adj
    assert all(-1 not in neighbors for neighbors in adj.values())
    assert adj == {0: set(), 1: set()}


def test_adjacency_four_connectivity_only():
    """Diagonal-only touches are not considered adjacent."""
    labels = np.array(
        [
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [2, 2, 3, 3],
            [2, 2, 3, 3],
        ],
        dtype=np.int32,
    )
    adj = segment_adjacency(labels)
    # 0 touches 1 (right) and 2 (down); 0 does NOT touch 3 (diagonal only).
    assert adj[0] == {1, 2}
    assert adj[3] == {1, 2}
    assert 3 not in adj[0]
    assert 0 not in adj[3]


def test_adjacency_isolated_segment_has_empty_set():
    """A segment with only foreground neighbors still appears with an empty set."""
    labels = np.array(
        [
            [0, -1, -1],
            [-1, -1, -1],
            [-1, -1, 1],
        ],
        dtype=np.int32,
    )
    adj = segment_adjacency(labels)
    assert adj == {0: set(), 1: set()}


def test_adjacency_symmetric():
    """Neighbor relationships are symmetric."""
    labels = segment_sky(np.zeros((40, 40), dtype=bool), n_segments=5)
    adj = segment_adjacency(labels)
    for seg, neighbors in adj.items():
        for nbr in neighbors:
            assert seg in adj[nbr]


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
