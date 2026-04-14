"""Tests for the StarDetector core class."""

from __future__ import annotations

import pytest

from celestack.frame.core import Frame
from celestack.mask.core import Mask
from celestack.star_detector.core import StarDetector


def test_detect_uses_fwhm_override_without_estimate(
    proxy_frame: Frame, sky_mask_proxy: Mask
):
    """Passing fwhm lets detect run without a prior estimate_fwhm call."""
    detector = StarDetector(proxy_frame, sky_mask_proxy)
    detector.segment(n_segments=2)
    # fwhm in full-res pixels; proxy has downscale_factor=2.
    stars = detector.detect(target_stars=5, fwhm=6.0)
    assert len(stars) > 0


def test_detect_without_fwhm_or_estimate_raises(
    proxy_frame: Frame, sky_mask_proxy: Mask
):
    """Omitting fwhm without having estimated it first raises."""
    detector = StarDetector(proxy_frame, sky_mask_proxy)
    detector.segment(n_segments=2)
    with pytest.raises(AttributeError, match=r"FWHM has not been estimated"):
        detector.detect(target_stars=5)


def test_detect_fwhm_argument_overrides_estimated(
    proxy_frame: Frame, sky_mask_proxy: Mask
):
    """An explicit fwhm argument is used in place of the estimated value."""
    detector = StarDetector(proxy_frame, sky_mask_proxy)
    detector.segment(n_segments=2)
    detector.estimate_fwhm()
    estimated = detector.fwhm
    # Pass a very different full-res fwhm; the property should remain unchanged.
    detector.detect(target_stars=5, fwhm=estimated * 2 * proxy_frame.downscale_factor)
    assert detector.fwhm == estimated
