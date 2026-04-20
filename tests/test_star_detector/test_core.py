"""Tests for the StarDetector core class."""

from __future__ import annotations

import numpy as np
import pytest

from celestack.exceptions import StarDetectorError
from celestack.proxy.frame import ProxyFrame
from celestack.proxy.mask import ProxyMask
from celestack.star_detector.core import StarDetector


def test_detect_uses_fwhm_override_without_estimate(
    proxy_frame: ProxyFrame, sky_mask_top_half: ProxyMask
) -> None:
    """Passing fwhm lets detect run without a prior estimate_fwhm call."""
    detector = StarDetector(proxy_frame, sky_mask_top_half)
    detector.segment(n_segments=2)
    stars = detector.detect(target_stars=5, fwhm=3.0)
    assert len(stars) > 0


def test_detect_without_fwhm_or_estimate_raises(
    proxy_frame: ProxyFrame, sky_mask_top_half: ProxyMask
) -> None:
    """Omitting fwhm without having estimated it first raises."""
    detector = StarDetector(proxy_frame, sky_mask_top_half)
    detector.segment(n_segments=2)
    with pytest.raises(AttributeError, match=r"FWHM has not been estimated"):
        detector.detect(target_stars=5)


def test_detect_fwhm_argument_overrides_estimated(
    proxy_frame: ProxyFrame, sky_mask_top_half: ProxyMask
) -> None:
    """An explicit fwhm argument is used in place of the estimated value."""
    detector = StarDetector(proxy_frame, sky_mask_top_half)
    detector.segment(n_segments=2)
    detector.estimate_fwhm()
    estimated = detector.fwhm
    detector.detect(target_stars=5, fwhm=estimated * 2)
    assert detector.fwhm == estimated


def test_stars_are_in_proxy_coordinates(
    proxy_frame: ProxyFrame, sky_mask_top_half: ProxyMask
) -> None:
    """Detected stars live in proxy pixel space; no x0/y0 columns remain."""
    detector = StarDetector(proxy_frame, sky_mask_top_half)
    detector.segment(n_segments=2)
    stars = detector.detect(target_stars=5, fwhm=3.0)
    assert "x0" not in stars.columns
    assert "y0" not in stars.columns
    assert "flux_local" not in stars.columns
    h, w = proxy_frame.shape
    xs = stars["x"].to_numpy()
    ys = stars["y"].to_numpy()
    assert np.all(xs < w)
    assert np.all(ys < h)


def test_shape_mismatch_raises(
    proxy_frame: ProxyFrame, shape_mismatch_mask: ProxyMask
) -> None:
    """Mismatched shapes raise StarDetectorError."""
    with pytest.raises(StarDetectorError, match="shape"):
        StarDetector(proxy_frame, shape_mismatch_mask)


def test_downscale_factor_mismatch_raises(
    proxy_frame: ProxyFrame, wrong_downscale_mask: ProxyMask
) -> None:
    """Mismatched downscale factors raise StarDetectorError."""
    with pytest.raises(StarDetectorError, match="downscale_factor"):
        StarDetector(proxy_frame, wrong_downscale_mask)


def test_all_foreground_mask_raises(
    proxy_frame: ProxyFrame, all_foreground_mask: ProxyMask
) -> None:
    """A mask covering every pixel raises StarDetectorError."""
    with pytest.raises(StarDetectorError, match="entire frame"):
        StarDetector(proxy_frame, all_foreground_mask)


def test_segment_requires_positive_n(
    proxy_frame: ProxyFrame, sky_mask_top_half: ProxyMask
) -> None:
    """segment(n_segments=0) raises ValueError."""
    detector = StarDetector(proxy_frame, sky_mask_top_half)
    with pytest.raises(ValueError, match="at least 1"):
        detector.segment(n_segments=0)


def test_detect_before_segment_raises(
    proxy_frame: ProxyFrame, sky_mask_top_half: ProxyMask
) -> None:
    """Calling detect() before segment() raises StarDetectorError."""
    detector = StarDetector(proxy_frame, sky_mask_top_half)
    with pytest.raises(StarDetectorError, match="segment"):
        detector.detect(target_stars=5, fwhm=3.0)
