"""Tests for the average_frames function."""

from pathlib import Path

import numpy as np
import pytest
import tifffile

from celestack.averaging import average_frames
from celestack.frame import Frame


def _write_gray_tiff(path: Path, array: np.ndarray) -> Path:
    """Write a simple grayscale TIFF for testing."""
    tifffile.imwrite(path, data=array, photometric="minisblack")
    return path


def _write_rgb_tiff(path: Path, array: np.ndarray) -> Path:
    """Write a simple RGB TIFF for testing."""
    tifffile.imwrite(path, data=array, photometric="rgb")
    return path


@pytest.fixture()
def gray_frames(tmp_path: Path) -> list[Frame]:
    """Three 32x24 grayscale uint16 TIFFs with known values."""
    arrays = [np.full((24, 32), fill_value=v, dtype=np.uint16) for v in [100, 200, 300]]
    frames = []
    for i, arr in enumerate(arrays):
        p = _write_gray_tiff(tmp_path / f"frame_{i}.tif", arr)
        frames.append(Frame(p))
    return frames


@pytest.fixture()
def rgb_frames(tmp_path: Path) -> list[Frame]:
    """Three 32x24 RGB uint8 TIFFs with known values."""
    arrays = [
        np.full((24, 32, 3), fill_value=v, dtype=np.uint8) for v in [50, 100, 150]
    ]
    frames = []
    for i, arr in enumerate(arrays):
        p = _write_rgb_tiff(tmp_path / f"rgb_{i}.tif", arr)
        frames.append(Frame(p))
    return frames


def test_mean_grayscale(gray_frames: list[Frame], tmp_path: Path) -> None:
    """Mean of [100, 200, 300] produces 200."""
    result = average_frames(gray_frames, tmp_path / "mean.tif", method="mean")
    assert isinstance(result, Frame)
    np.testing.assert_array_equal(result.array, 200)


def test_median_grayscale(gray_frames: list[Frame], tmp_path: Path) -> None:
    """Median of [100, 200, 300] produces 200."""
    result = average_frames(gray_frames, tmp_path / "median.tif", method="median")
    np.testing.assert_array_equal(result.array, 200)


def test_mean_rgb(rgb_frames: list[Frame], tmp_path: Path) -> None:
    """Mean averaging works on RGB frames."""
    result = average_frames(rgb_frames, tmp_path / "mean.tif", method="mean")
    np.testing.assert_array_equal(result.array, 100)


def test_sigma_clip_rejects_outlier(tmp_path: Path) -> None:
    """Sigma-clipped mean excludes a clear outlier."""
    arrays = [np.full((24, 32), fill_value=100, dtype=np.uint16) for _ in range(19)]
    arrays.append(np.full((24, 32), fill_value=60000, dtype=np.uint16))
    frames = []
    for i, arr in enumerate(arrays):
        p = _write_gray_tiff(tmp_path / f"sc_{i}.tif", arr)
        frames.append(Frame(p))

    result = average_frames(frames, tmp_path / "sc.tif", method="sigma_clip")
    # The outlier should be clipped, leaving ~100
    assert np.all(result.array < 200)


def test_returns_correct_dtype(gray_frames: list[Frame], tmp_path: Path) -> None:
    """Output dtype matches input dtype."""
    result = average_frames(gray_frames, tmp_path / "out.tif", method="mean")
    assert result.array.dtype == np.uint16


def test_band_processing_matches_full(tmp_path: Path) -> None:
    """Row-band processing produces the same result regardless of band size."""
    rng = np.random.default_rng(99)
    frames = []
    for i in range(5):
        arr = rng.integers(0, 255, (30, 40), dtype=np.uint8)
        p = _write_gray_tiff(tmp_path / f"rand_{i}.tif", arr)
        frames.append(Frame(p))

    full = average_frames(
        frames, tmp_path / "full.tif", method="median", band_height=9999
    )
    banded = average_frames(
        frames, tmp_path / "banded.tif", method="median", band_height=7
    )
    np.testing.assert_array_equal(full.array, banded.array)


def test_empty_frames_raises(tmp_path: Path) -> None:
    """Empty frame list raises ValueError."""
    with pytest.raises(ValueError, match="non-empty"):
        average_frames([], tmp_path / "out.tif")


def test_unknown_method_raises(gray_frames: list[Frame], tmp_path: Path) -> None:
    """Unknown method name raises ValueError."""
    with pytest.raises(ValueError, match="Unknown averaging method"):
        average_frames(gray_frames, tmp_path / "out.tif", method="bogus")


def test_shape_mismatch_raises(tmp_path: Path) -> None:
    """Frames with different shapes raise ValueError."""
    a = _write_gray_tiff(tmp_path / "a.tif", np.zeros((24, 32), dtype=np.uint16))
    b = _write_gray_tiff(tmp_path / "b.tif", np.zeros((24, 48), dtype=np.uint16))
    with pytest.raises(ValueError, match="Shape mismatch"):
        average_frames([Frame(a), Frame(b)], tmp_path / "out.tif")


def test_single_frame_returns_copy(tmp_path: Path) -> None:
    """Averaging a single frame returns the same data."""
    arr = np.arange(24 * 32, dtype=np.uint16).reshape(24, 32)
    p = _write_gray_tiff(tmp_path / "single.tif", arr)
    result = average_frames([Frame(p)], tmp_path / "out.tif", method="mean")
    np.testing.assert_array_equal(result.array, arr)


def test_non_tiled_input_works(tmp_path: Path) -> None:
    """Strip-layout (non-tiled) TIFFs are handled without error."""
    frames = []
    for i in range(3):
        arr = np.full((24, 32), fill_value=i * 100, dtype=np.uint16)
        p = tmp_path / f"strip_{i}.tif"
        tifffile.imwrite(p, data=arr, photometric="minisblack")
        frames.append(Frame(p))
    result = average_frames(frames, tmp_path / "out.tif", method="mean")
    assert result.array.shape == (24, 32)
