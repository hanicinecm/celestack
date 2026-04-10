"""Integration tests for the StarDetector class."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from celestack.exceptions import StarDetectorError
from celestack.frame.core import Frame
from celestack.mask.core import Mask
from celestack.star_detector.core import StarDetector
from tests.utils import write_rgb_tiff


def test_full_pipeline_returns_stars(
    full_res_frame: Frame,
    sky_mask_top_half: Mask,
):
    """Full detect pipeline returns a non-empty star table with expected columns."""
    sd = StarDetector(full_res_frame, sky_mask_top_half)
    stars = sd.detect(target_stars=5, n_segments=2)
    assert len(stars) > 0
    assert "star_id" in stars.columns
    assert "x" in stars.columns
    assert "y" in stars.columns
    assert "x0" in stars.columns
    assert "y0" in stars.columns


def test_coordinates_in_frame_space(
    proxy_frame: Frame,
    sky_mask_proxy: Mask,
):
    """x/y stay in proxy space; x0/y0 hold full-res equivalents."""
    sd = StarDetector(proxy_frame, sky_mask_proxy)
    stars = sd.detect(target_stars=5, n_segments=2)
    assert len(stars) > 0
    # Proxy is 128px wide/tall; x/y must stay within proxy bounds.
    assert stars["x"].max() <= 128
    assert stars["y"].max() <= 128
    # x0/y0 are proxy coords * downscale_factor=2, so within 0-256.
    assert "x0" in stars.columns
    assert "y0" in stars.columns
    assert stars["x0"].max() <= 256
    assert stars["y0"].max() <= 256
    # Verify the relationship holds element-wise.
    assert (stars["x0"] == stars["x"] * 2).all()
    assert (stars["y0"] == stars["y"] * 2).all()


def test_fwhm_is_in_frame_space(
    proxy_frame: Frame,
    sky_mask_proxy: Mask,
):
    """fwhm property returns value in the frame's own pixel coordinates."""
    sd = StarDetector(proxy_frame, sky_mask_proxy)
    # fwhm is in proxy pixels, so it equals the internal _fwhm_proxy directly.
    assert sd.fwhm == sd._fwhm


def test_detect_replaces_previous(
    full_res_frame: Frame,
    sky_mask_top_half: Mask,
):
    """Calling detect() twice replaces previous results."""
    sd = StarDetector(full_res_frame, sky_mask_top_half)
    stars1 = sd.detect(target_stars=5, n_segments=2)
    stars2 = sd.detect(target_stars=3, n_segments=2)
    assert stars2 is sd.stars
    assert stars1 is not stars2


def test_stars_before_detect_raises(
    full_res_frame: Frame,
    sky_mask_top_half: Mask,
):
    """Accessing stars before detect() raises StarDetectorError."""
    sd = StarDetector(full_res_frame, sky_mask_top_half)
    with pytest.raises(StarDetectorError, match="detect"):
        _ = sd.stars


def test_plot_before_detect_raises(
    full_res_frame: Frame,
    sky_mask_top_half: Mask,
):
    """Calling plot() before detect() raises StarDetectorError."""
    sd = StarDetector(full_res_frame, sky_mask_top_half)
    with pytest.raises(StarDetectorError, match="detect"):
        sd.plot()


def test_non_grayscale_frame_raises(tmp_path: Path):
    """Passing a 3-channel RGB frame raises StarDetectorError."""
    array = np.zeros((128, 128, 3), dtype=np.uint8)
    path = tmp_path / "rgb.tif"
    write_rgb_tiff(path, array)
    frame = Frame(path)
    mask = Mask._from_array(np.zeros((128, 128), dtype=bool))
    with pytest.raises(StarDetectorError, match="grayscale"):
        StarDetector(frame, mask)


def test_shape_mismatch_raises(tmp_path: Path, sky_mask_top_half: Mask):
    """Mismatched frame and mask dimensions raise StarDetectorError."""
    from tests.utils import write_gray_tiff

    array = np.full((64, 64), 100, dtype=np.uint16)
    path = tmp_path / "small.tif"
    write_gray_tiff(path, array)
    frame = Frame(path)
    with pytest.raises(StarDetectorError, match="shape"):
        StarDetector(frame, sky_mask_top_half)


def test_all_foreground_raises(
    full_res_frame: Frame,
    all_foreground_mask: Mask,
):
    """All-foreground mask raises StarDetectorError."""
    with pytest.raises(StarDetectorError, match="no sky"):
        StarDetector(full_res_frame, all_foreground_mask)


def test_invalid_target_stars(
    full_res_frame: Frame,
    sky_mask_top_half: Mask,
):
    """target_stars < 1 raises ValueError."""
    sd = StarDetector(full_res_frame, sky_mask_top_half)
    with pytest.raises(ValueError, match="target_stars"):
        sd.detect(target_stars=0)


def test_invalid_n_segments(
    full_res_frame: Frame,
    sky_mask_top_half: Mask,
):
    """n_segments < 1 raises ValueError."""
    sd = StarDetector(full_res_frame, sky_mask_top_half)
    with pytest.raises(ValueError, match="n_segments"):
        sd.detect(n_segments=0)


def test_invalid_roundness_range(
    full_res_frame: Frame,
    sky_mask_top_half: Mask,
):
    """Invalid roundness_range raises ValueError."""
    sd = StarDetector(full_res_frame, sky_mask_top_half)
    with pytest.raises(ValueError, match="roundness"):
        sd.detect(roundness_range=(1.0, -1.0))
