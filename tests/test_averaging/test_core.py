"""Tests for the average_frames function."""

from pathlib import Path

import numpy as np
import pytest
import tifffile

from celestack.averaging import average_frames
from celestack.frame import Frame
from tests.utils import (
    CAMERA_MAKE,
    CAMERA_MODEL,
    LENS_MODEL,
    write_gray_tiff,
    write_tiff,
)


def test_mean_grayscale(gray_frames: list[Frame]) -> None:
    """Mean of [100, 200, 300] produces 200."""
    result = average_frames(gray_frames, method="mean")
    assert isinstance(result, Frame)
    np.testing.assert_array_equal(result.array, 200)


def test_median_grayscale(gray_frames: list[Frame]) -> None:
    """Median of [100, 200, 300] produces 200."""
    result = average_frames(gray_frames, method="median")
    np.testing.assert_array_equal(result.array, 200)


def test_mean_rgb(rgb_frames: list[Frame]) -> None:
    """Mean averaging works on RGB frames."""
    result = average_frames(rgb_frames, method="mean")
    np.testing.assert_array_equal(result.array, 100)


def test_sigma_clip_rejects_outlier(tmp_path: Path) -> None:
    """Sigma-clipped mean excludes a clear outlier."""
    arrays = [np.full((24, 32), fill_value=100, dtype=np.uint16) for _ in range(19)]
    arrays.append(np.full((24, 32), fill_value=60000, dtype=np.uint16))
    frames = []
    for i, arr in enumerate(arrays):
        p = write_gray_tiff(tmp_path / f"sc_{i}.tif", arr)
        frames.append(Frame(p))

    result = average_frames(frames, method="sigma_clip")
    # The outlier should be clipped, leaving ~100
    assert np.all(result.array < 200)


def test_returns_correct_dtype(gray_frames: list[Frame]) -> None:
    """Output dtype matches input dtype."""
    result = average_frames(gray_frames, method="mean")
    assert result.array.dtype == np.uint16


def test_band_processing_matches_full(tmp_path: Path) -> None:
    """Row-band processing produces the same result regardless of band size."""
    rng = np.random.default_rng(99)
    frames = []
    for i in range(5):
        arr = rng.integers(0, 255, (30, 40), dtype=np.uint8)
        p = write_gray_tiff(tmp_path / f"rand_{i}.tif", arr)
        frames.append(Frame(p))

    full = average_frames(frames, method="median", band_height=9999)
    banded = average_frames(frames, method="median", band_height=7)
    np.testing.assert_array_equal(full.array, banded.array)


def test_empty_frames_raises() -> None:
    """Empty frame list raises ValueError."""
    with pytest.raises(ValueError, match="non-empty"):
        average_frames([])


def test_unknown_method_raises(gray_frames: list[Frame]) -> None:
    """Unknown method name raises ValueError."""
    with pytest.raises(ValueError, match="Unknown averaging method"):
        average_frames(gray_frames, method="bogus")


def test_shape_mismatch_raises(tmp_path: Path) -> None:
    """Frames with different shapes raise ValueError."""
    a = write_gray_tiff(tmp_path / "a.tif", np.zeros((24, 32), dtype=np.uint16))
    b = write_gray_tiff(tmp_path / "b.tif", np.zeros((24, 48), dtype=np.uint16))
    with pytest.raises(ValueError, match=r"Shape mismatch: Frame\("):
        average_frames([Frame(a), Frame(b)])


def test_single_frame_returns_copy(tmp_path: Path) -> None:
    """Averaging a single frame returns the same data."""
    arr = np.arange(24 * 32, dtype=np.uint16).reshape(24, 32)
    p = write_gray_tiff(tmp_path / "single.tif", arr)
    result = average_frames([Frame(p)], method="mean")
    np.testing.assert_array_equal(result.array, arr)


def test_non_tiled_input_works(tmp_path: Path) -> None:
    """Strip-layout (non-tiled) TIFFs are handled without error."""
    frames = []
    for i in range(3):
        arr = np.full((24, 32), fill_value=i * 100, dtype=np.uint16)
        p = tmp_path / f"strip_{i}.tif"
        tifffile.imwrite(p, data=arr, photometric="minisblack")
        frames.append(Frame(p))
    result = average_frames(frames, method="mean")
    assert result.array.shape == (24, 32)


def test_bit_depth_mismatch_raises(tmp_path: Path) -> None:
    """Frames with different bit depths raise ValueError."""
    a = write_gray_tiff(tmp_path / "a.tif", np.zeros((24, 32), dtype=np.uint16))
    b = write_gray_tiff(tmp_path / "b.tif", np.zeros((24, 32), dtype=np.uint8))
    with pytest.raises(ValueError, match=r"Bit depth mismatch: Frame\("):
        average_frames([Frame(a), Frame(b)])


def test_reference_frame_metadata_inherited(tmp_path: Path) -> None:
    """Output inherits EXIF metadata from the reference frame."""
    frames = []
    for i in range(3):
        arr = np.full((24, 32), fill_value=100, dtype=np.uint16)
        p = write_gray_tiff(tmp_path / f"f_{i}.tif", arr)
        frames.append(Frame(p))

    # Create a reference frame with EXIF metadata
    ref_arr = np.full((24, 32, 3), fill_value=100, dtype=np.uint16)
    ref_path = tmp_path / "ref.tif"
    extratags = [
        (271, "s", 0, CAMERA_MAKE, True),
        (272, "s", 0, CAMERA_MODEL, True),
        (42036, "s", 0, LENS_MODEL, True),
    ]
    write_tiff(ref_path, ref_arr, extratags=extratags)
    ref_frame = Frame(ref_path)

    result = average_frames(frames, method="mean", reference_frame=ref_frame)
    assert result.metadata.camera_make == CAMERA_MAKE
    assert result.metadata.camera_model == CAMERA_MODEL
    assert result.metadata.lens_model == LENS_MODEL


def test_band_height_none_full_image(tmp_path: Path) -> None:
    """band_height=None processes the entire image at once."""
    rng = np.random.default_rng(42)
    frames = []
    for i in range(5):
        arr = rng.integers(0, 255, (30, 40), dtype=np.uint8)
        p = write_gray_tiff(tmp_path / f"f_{i}.tif", arr)
        frames.append(Frame(p))

    full = average_frames(frames, method="median", band_height=None)
    banded = average_frames(frames, method="median", band_height=7)
    np.testing.assert_array_equal(full.array, banded.array)


def test_sigma_clip_identical_frames(tmp_path: Path) -> None:
    """Sigma-clipped mean of identical frames returns the input value."""
    frames = []
    for i in range(5):
        arr = np.full((24, 32), fill_value=150, dtype=np.uint16)
        p = write_gray_tiff(tmp_path / f"same_{i}.tif", arr)
        frames.append(Frame(p))

    result = average_frames(frames, method="sigma_clip")
    np.testing.assert_array_equal(result.array, 150)


def test_result_is_detached(gray_frames: list[Frame]) -> None:
    """average_frames returns a detached (in-memory) frame with no backing path."""
    result = average_frames(gray_frames, method="mean")
    with pytest.raises(AttributeError):
        _ = result.path
