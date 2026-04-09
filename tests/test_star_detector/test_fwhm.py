"""Tests for FWHM auto-estimation."""

from __future__ import annotations

import numpy as np

from celestack.star_detector._fwhm import _pick_subregions, estimate_fwhm


def _gaussian_star_image(
    height: int = 128,
    width: int = 128,
    fwhm: float = 5.0,
    n_stars: int = 20,
    seed: int = 0,
) -> np.ndarray:
    """Create a synthetic image with Gaussian blobs of known FWHM."""
    rng = np.random.default_rng(seed)
    image = rng.normal(100, 10, (height, width)).astype(np.float64)
    sigma = fwhm / 2.3548
    yy, xx = np.mgrid[0:height, 0:width]
    for _ in range(n_stars):
        r = rng.integers(10, height - 10)
        c = rng.integers(10, width - 10)
        gauss = 500 * np.exp(-((yy - r) ** 2 + (xx - c) ** 2) / (2 * sigma**2))
        image += gauss
    return image.astype(np.uint16)


def test_estimated_fwhm_within_range():
    """Estimated FWHM for known Gaussian blobs is within ±2px of the true value."""
    true_fwhm = 5.0
    image = _gaussian_star_image(fwhm=true_fwhm)
    sky_mask = np.zeros(image.shape, dtype=bool)  # all sky
    estimated = estimate_fwhm(image, sky_mask)
    assert abs(estimated - true_fwhm) <= 2.0


def test_pick_subregions_returns_correct_count():
    """_pick_subregions returns the requested number of sub-regions."""
    mask = np.zeros((100, 100), dtype=bool)
    mask[80:, :] = True
    regions = _pick_subregions(mask, 5)
    assert len(regions) == 5


def test_pick_subregions_within_bounds():
    """All sub-regions fall within the sky bounding box."""
    mask = np.zeros((100, 100), dtype=bool)
    mask[80:, :] = True
    regions = _pick_subregions(mask, 5)
    for y0, y1, x0, x1 in regions:
        assert 0 <= y0 < y1 <= 100
        assert 0 <= x0 < x1 <= 100


def test_estimate_fwhm_fallback_on_no_sky():
    """Returns a reasonable fallback when sub-regions have no detections."""
    image = np.full((32, 32), 100, dtype=np.uint16)  # flat image, no stars
    sky_mask = np.zeros((32, 32), dtype=bool)
    result = estimate_fwhm(image, sky_mask)
    assert 2.0 <= result <= 15.0
