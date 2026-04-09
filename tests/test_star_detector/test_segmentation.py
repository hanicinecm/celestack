"""Tests for sky segmentation."""

from __future__ import annotations

import numpy as np

from celestack.star_detector._segmentation import segment_sky


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
