"""Tests for DAOStarFinder binary search and segment-based detection."""

from __future__ import annotations

import numpy as np

from celestack.progress import set_progress_factory
from celestack.star_detector._detection import (
    binary_search_threshold,
    detect_in_segments,
)

set_progress_factory(None)


def _synthetic_image_with_stars(n_stars: int = 15, seed: int = 42) -> np.ndarray:
    """Create a 128x128 image with known star count."""
    rng = np.random.default_rng(seed)
    image = rng.normal(100, 10, (128, 128)).astype(np.float64)
    sigma = 3.0 / 2.3548
    yy, xx = np.mgrid[0:128, 0:128]
    for _ in range(n_stars):
        r = rng.integers(15, 113)
        c = rng.integers(15, 113)
        gauss = 600 * np.exp(-((yy - r) ** 2 + (xx - c) ** 2) / (2 * sigma**2))
        image += gauss
    return image


def test_binary_search_finds_stars():
    """Binary search on a synthetic image detects a reasonable star count."""
    image = _synthetic_image_with_stars(15)
    mask = np.zeros(image.shape, dtype=bool)
    threshold, stars = binary_search_threshold(
        image,
        mask,
        fwhm=3.0,
        target_count=10,
        roundness_range=(-1.0, 1.0),
        threshold_bounds=(10, 200),
    )
    assert len(stars) > 0
    assert threshold > 0


def test_binary_search_target_unreachable():
    """Requesting more stars than exist returns all found without crashing."""
    image = _synthetic_image_with_stars(3)
    mask = np.zeros(image.shape, dtype=bool)
    _, stars = binary_search_threshold(
        image,
        mask,
        fwhm=3.0,
        target_count=100,
        roundness_range=(-1.0, 1.0),
        threshold_bounds=(10, 200),
    )
    assert len(stars) >= 0  # doesn't crash


def test_binary_search_returns_expected_columns():
    """Result DataFrame has the expected columns."""
    image = _synthetic_image_with_stars(10)
    mask = np.zeros(image.shape, dtype=bool)
    _, stars = binary_search_threshold(
        image,
        mask,
        fwhm=3.0,
        target_count=5,
        roundness_range=(-1.0, 1.0),
        threshold_bounds=(10, 200),
    )
    assert set(stars.columns) == {"x", "y", "flux", "fwhm", "roundness"}


def test_detect_in_segments_combines_results():
    """detect_in_segments returns stars from multiple segments."""
    image = _synthetic_image_with_stars(20)
    labels = np.full((128, 128), -1, dtype=np.int32)
    labels[:64, :] = 0
    labels[64:, :] = 1
    target = {0: 5, 1: 5}

    result = detect_in_segments(
        image,
        labels,
        target,
        fwhm=3.0,
        roundness_range=(-1.0, 1.0),
        threshold_min_sigma=2.0,
        threshold_max_sigma=15.0,
    )
    assert len(result) > 0
    assert "segment_id" in result.columns
    assert "threshold" in result.columns


def test_detect_in_segments_empty_on_flat_image():
    """No stars detected on a perfectly flat image."""
    image = np.full((64, 64), 100.0)
    labels = np.zeros((64, 64), dtype=np.int32)
    target = {0: 10}
    result = detect_in_segments(
        image,
        labels,
        target,
        fwhm=3.0,
        roundness_range=(-1.0, 1.0),
        threshold_min_sigma=2.0,
        threshold_max_sigma=15.0,
    )
    assert len(result) == 0
