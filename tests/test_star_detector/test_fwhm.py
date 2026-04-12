"""Tests for FWHM auto-estimation."""

from __future__ import annotations

import numpy as np

from celestack.progress import set_progress_factory
from celestack.star_detector._fwhm import _sample_mask, estimate_fwhm
from celestack.star_detector._segmentation import segment_sky

set_progress_factory(None)


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
    """Estimated FWHM for known Gaussian blobs is within +/-2 px of the true value."""
    true_fwhm = 5.0
    image = _gaussian_star_image(fwhm=true_fwhm)
    sky_mask = np.zeros(image.shape, dtype=bool)
    labels = segment_sky(sky_mask, n_segments=8)
    estimated = estimate_fwhm(image, labels, (2.0, 15.0), 14, 8.0)
    assert abs(estimated - true_fwhm) <= 2.0


def test_sample_mask_exposes_only_chosen_segments():
    """The sample mask exposes exactly the pixels of the chosen segments."""
    labels = segment_sky(np.zeros((64, 64), dtype=bool), n_segments=8)
    mask = _sample_mask(labels, n=3)
    exposed = labels[~mask]
    assert exposed.size > 0
    assert np.all(exposed >= 0)
    assert len(np.unique(exposed)) == 3


def test_sample_mask_hides_foreground():
    """Foreground pixels are always hidden by the sample mask."""
    sky_mask = np.zeros((64, 64), dtype=bool)
    sky_mask[40:, :] = True
    labels = segment_sky(sky_mask, n_segments=5)
    mask = _sample_mask(labels, n=3)
    assert np.all(mask[labels == -1])


def test_sample_mask_caps_at_available_segments():
    """Requesting more segments than available caps to the number available."""
    labels = segment_sky(np.zeros((32, 32), dtype=bool), n_segments=3)
    mask = _sample_mask(labels, n=10)
    exposed = labels[~mask]
    assert len(np.unique(exposed)) == 3


def test_estimate_fwhm_fallback_on_flat_image():
    """Returns a reasonable fallback when the image has no detectable signal."""
    image = np.full((64, 64), 100, dtype=np.uint16)
    labels = segment_sky(np.zeros((64, 64), dtype=bool), n_segments=5)
    result = estimate_fwhm(image, labels, (2.0, 15.0), 14, 8.0)
    assert 2.0 <= result <= 15.0
