"""Tests for sky segmentation."""

from __future__ import annotations

import numpy as np

from celestack.progress import set_progress_factory
from celestack.star_detector._segmentation import segment_adjacency, segment_sky

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
