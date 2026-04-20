"""Tests for ProxyFrame."""

from pathlib import Path

import numpy as np
import pytest

from celestack.exceptions import ProxyMetadataError
from celestack.frame.core import Frame
from celestack.mask.core import Mask
from celestack.proxy._core import METADATA_FILENAME
from celestack.proxy.frame import ProxyFrame


def test_from_frame_produces_float16_array(rgb_frame: Frame, sky_mask: Mask) -> None:
    """from_frame returns a 2D float16 array with the expected downscaled shape."""
    pf = ProxyFrame.from_frame(rgb_frame, sky_mask, 4, box_size=8, filter_size=1)
    assert pf.dtype == "float16"
    assert pf.array.dtype == np.float16
    assert pf.shape == (rgb_frame.shape[0] // 4, rgb_frame.shape[1] // 4)
    assert pf.downscale_factor == 4


def test_from_frame_zeros_foreground(rgb_frame: Frame, sky_mask: Mask) -> None:
    """Masked pixels in the proxy are exactly 0.0."""
    pf = ProxyFrame.from_frame(rgb_frame, sky_mask, 4, box_size=8, filter_size=1)
    # Bottom 8 rows of the full-res mask → bottom 2 rows of a /4 proxy
    assert np.all(pf.array[-2:, :] == np.float16(0.0))


def test_from_frame_sky_near_zero(rgb_frame: Frame, sky_mask: Mask) -> None:
    """Sky pixel median is close to zero after background subtraction."""
    pf = ProxyFrame.from_frame(rgb_frame, sky_mask, 4, box_size=8, filter_size=1)
    sky = pf.array[pf.array != 0]
    assert abs(float(np.median(sky))) < 0.1


def test_from_frame_rejects_non_integer_dtype(rgb_frame: Frame, sky_mask: Mask) -> None:
    """Only uint8/uint16 source frames are accepted."""

    class FakeFrame:
        dtype = "float32"
        shape = rgb_frame.shape
        array = rgb_frame.array.astype(np.float32)

    with pytest.raises(ValueError, match="uint8 or uint16"):
        ProxyFrame.from_frame(FakeFrame(), sky_mask, 2)


def test_from_frame_mask_shape_mismatch(rgb_frame: Frame) -> None:
    """Mask shape must match the frame."""
    bad_mask = Mask._from_array(np.zeros((10, 10), dtype=np.bool_))
    with pytest.raises(ValueError, match="Mask shape"):
        ProxyFrame.from_frame(rgb_frame, bad_mask, 2)


def test_save_as_round_trip(tmp_path: Path, rgb_frame: Frame, sky_mask: Mask) -> None:
    """Saving and reloading round-trips the array and downscale factor."""
    pf = ProxyFrame.from_frame(rgb_frame, sky_mask, 4, box_size=8, filter_size=1)
    target = tmp_path / "proxy.npy"
    pf.save_as(target)
    assert target.exists()
    assert (tmp_path / METADATA_FILENAME).exists()

    loaded = ProxyFrame(target)
    np.testing.assert_array_equal(loaded.array, pf.array)
    assert loaded.downscale_factor == 4
    assert loaded.path == target


def test_save_as_requires_npy(tmp_path: Path, rgb_frame: Frame, sky_mask: Mask) -> None:
    """Non-.npy suffix is rejected."""
    pf = ProxyFrame.from_frame(rgb_frame, sky_mask, 4, box_size=8, filter_size=1)
    with pytest.raises(ValueError, match=".npy"):
        pf.save_as(tmp_path / "proxy.tif")


def test_save_as_no_overwrite(tmp_path: Path, rgb_frame: Frame, sky_mask: Mask) -> None:
    """Existing file without overwrite=True raises."""
    pf = ProxyFrame.from_frame(rgb_frame, sky_mask, 4, box_size=8, filter_size=1)
    target = tmp_path / "proxy.npy"
    pf.save_as(target)
    with pytest.raises(FileExistsError):
        pf.save_as(target)


def test_detached_path_raises(rgb_frame: Frame, sky_mask: Mask) -> None:
    """Detached proxy raises when .path is accessed."""
    pf = ProxyFrame.from_frame(rgb_frame, sky_mask, 4, box_size=8, filter_size=1)
    with pytest.raises(AttributeError, match="not backed"):
        _ = pf.path


def test_load_wrong_dtype_raises(tmp_path: Path) -> None:
    """Loading a non-float16 array raises ValueError."""
    np.save(tmp_path / "proxy.npy", np.zeros((4, 4), dtype=np.float32))
    (tmp_path / METADATA_FILENAME).write_text("downscale_factor = 2\n")
    with pytest.raises(ValueError, match="float16"):
        ProxyFrame(tmp_path / "proxy.npy")


def test_load_missing_sidecar_raises(tmp_path: Path) -> None:
    """Missing sidecar raises ProxyMetadataError on load."""
    np.save(tmp_path / "proxy.npy", np.zeros((4, 4), dtype=np.float16))
    with pytest.raises(ProxyMetadataError):
        ProxyFrame(tmp_path / "proxy.npy")


def test_load_missing_file_raises(tmp_path: Path) -> None:
    """Loading a nonexistent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        ProxyFrame(tmp_path / "nope.npy")
