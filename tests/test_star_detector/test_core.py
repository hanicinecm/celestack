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
    """Full segment+detect pipeline returns a non-empty star table with expected columns."""
    sd = StarDetector(full_res_frame, sky_mask_top_half)
    sd.segment(n_segments=2)
    stars = sd.detect(target_stars=5)
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
    sd.segment(n_segments=2)
    stars = sd.detect(target_stars=5)
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
    # fwhm is in proxy pixels, so it equals the internal _fwhm directly.
    assert sd.fwhm == sd._fwhm


def test_segment_stores_labels(
    full_res_frame: Frame,
    sky_mask_top_half: Mask,
):
    """segment() stores a label array accessible via segment_labels property."""
    sd = StarDetector(full_res_frame, sky_mask_top_half)
    labels = sd.segment(n_segments=3)
    assert labels is sd.segment_labels
    assert labels.shape == full_res_frame.shape[:2]
    assert set(np.unique(labels)) <= set(range(3)) | {-1}


def test_resegment_clears_stars(
    full_res_frame: Frame,
    sky_mask_top_half: Mask,
):
    """Calling segment() again after detect() clears the star table."""
    sd = StarDetector(full_res_frame, sky_mask_top_half)
    sd.segment(n_segments=2)
    sd.detect(target_stars=5)
    assert sd._stars is not None
    sd.segment(n_segments=3)
    assert sd._stars is None


def test_detect_without_segment_raises(
    full_res_frame: Frame,
    sky_mask_top_half: Mask,
):
    """Calling detect() without prior segment() raises StarDetectorError."""
    sd = StarDetector(full_res_frame, sky_mask_top_half)
    with pytest.raises(StarDetectorError, match="segment"):
        sd.detect(target_stars=5)


def test_detect_replaces_previous(
    full_res_frame: Frame,
    sky_mask_top_half: Mask,
):
    """Calling detect() twice replaces previous results."""
    sd = StarDetector(full_res_frame, sky_mask_top_half)
    sd.segment(n_segments=2)
    stars1 = sd.detect(target_stars=5)
    stars2 = sd.detect(target_stars=3)
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


def test_segment_labels_before_segment_raises(
    full_res_frame: Frame,
    sky_mask_top_half: Mask,
):
    """Accessing segment_labels before segment() raises StarDetectorError."""
    sd = StarDetector(full_res_frame, sky_mask_top_half)
    with pytest.raises(StarDetectorError, match="segment"):
        _ = sd.segment_labels


def test_plot_always_returns_figure(
    full_res_frame: Frame,
    sky_mask_top_half: Mask,
):
    """plot() returns a figure regardless of whether segment/detect were called."""
    import plotly.graph_objects as go

    sd = StarDetector(full_res_frame, sky_mask_top_half)
    assert isinstance(sd.plot(), go.Figure)
    assert isinstance(sd.plot(show_segments=True), go.Figure)

    sd.segment(n_segments=2)
    assert isinstance(sd.plot(), go.Figure)
    assert isinstance(sd.plot(show_segments=True), go.Figure)

    sd.detect(target_stars=5)
    assert isinstance(sd.plot(), go.Figure)
    assert isinstance(sd.plot(show_segments=True), go.Figure)


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
    sd.segment(n_segments=2)
    with pytest.raises(ValueError, match="target_stars"):
        sd.detect(target_stars=0)


def test_invalid_n_segments(
    full_res_frame: Frame,
    sky_mask_top_half: Mask,
):
    """n_segments < 1 raises ValueError."""
    sd = StarDetector(full_res_frame, sky_mask_top_half)
    with pytest.raises(ValueError, match="n_segments"):
        sd.segment(n_segments=0)
