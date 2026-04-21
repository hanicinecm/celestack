"""Tests for ProxyMask."""

from pathlib import Path

import numpy as np
import pytest

from celestack.exceptions import ProxyMetadataError
from celestack.mask.core import Mask
from celestack.proxy._core import METADATA_FILENAME
from celestack.proxy.mask import ProxyMask


def _make_mask(shape: tuple[int, int] = (16, 16)) -> Mask:
    array = np.zeros(shape, dtype=np.bool_)
    array[: shape[0] // 2, :] = True
    return Mask._from_array(array)


def test_from_mask_downscales_by_majority_vote() -> None:
    """from_mask yields a detached proxy with the expected shape."""
    mask = _make_mask((16, 16))
    pm = ProxyMask.from_mask(mask, 4)
    assert pm.shape == (4, 4)
    assert pm.dtype == "bool"
    assert pm.downscale_factor == 4
    assert pm.array.dtype == np.bool_


def test_from_mask_detached_has_no_path() -> None:
    """Detached proxy mask raises when .path is accessed."""
    pm = ProxyMask.from_mask(_make_mask(), 2)
    with pytest.raises(AttributeError, match="not backed"):
        _ = pm.path


def test_save_as_round_trip(tmp_path: Path) -> None:
    """Saving then reloading preserves array and downscale_factor."""
    pm = ProxyMask.from_mask(_make_mask((16, 16)), 4)
    target = tmp_path / "mask.npy"
    pm.save_as(target)
    assert target.exists()
    assert (tmp_path / METADATA_FILENAME).exists()

    loaded = ProxyMask(target)
    np.testing.assert_array_equal(loaded.array, pm.array)
    assert loaded.downscale_factor == 4
    assert loaded.path == target


def test_save_as_requires_npy(tmp_path: Path) -> None:
    """Non-.npy suffix is rejected."""
    pm = ProxyMask.from_mask(_make_mask(), 2)
    with pytest.raises(ValueError, match=".npy"):
        pm.save_as(tmp_path / "mask.tif")


def test_save_as_no_overwrite(tmp_path: Path) -> None:
    """Existing file without overwrite=True raises."""
    pm = ProxyMask.from_mask(_make_mask(), 2)
    target = tmp_path / "mask.npy"
    pm.save_as(target)
    with pytest.raises(FileExistsError):
        pm.save_as(target)


def test_save_as_overwrite_true(tmp_path: Path) -> None:
    """overwrite=True replaces the file."""
    pm = ProxyMask.from_mask(_make_mask(), 2)
    target = tmp_path / "mask.npy"
    pm.save_as(target)
    pm.save_as(target, overwrite=True)


def test_save_as_conflicting_sidecar_raises(tmp_path: Path) -> None:
    """A sidecar from a different factor blocks save."""
    (tmp_path / METADATA_FILENAME).write_text("downscale_factor = 8\n")
    pm = ProxyMask.from_mask(_make_mask(), 2)
    with pytest.raises(ProxyMetadataError, match="mismatch"):
        pm.save_as(tmp_path / "mask.npy")


def test_load_missing_file_raises(tmp_path: Path) -> None:
    """Loading a nonexistent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        ProxyMask(tmp_path / "nope.npy")


def test_load_wrong_dtype_raises(tmp_path: Path) -> None:
    """Loading a non-bool array raises ValueError."""
    arr = np.zeros((4, 4), dtype=np.uint8)
    target = tmp_path / "mask.npy"
    np.save(target, arr)
    (tmp_path / METADATA_FILENAME).write_text("downscale_factor = 2\n")
    with pytest.raises(ValueError, match="2D boolean"):
        ProxyMask(target)


def test_load_missing_sidecar_raises(tmp_path: Path) -> None:
    """Missing sidecar raises ProxyMetadataError on load."""
    arr = np.zeros((4, 4), dtype=np.bool_)
    target = tmp_path / "mask.npy"
    np.save(target, arr)
    with pytest.raises(ProxyMetadataError):
        ProxyMask(target)


def test_repr_detached() -> None:
    """Repr indicates detached state."""
    pm = ProxyMask.from_mask(_make_mask(), 3)
    assert "<detached>" in repr(pm)
    assert "downscale_factor=3" in repr(pm)
