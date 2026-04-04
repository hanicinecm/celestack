"""Tests for the Mask class (src/celestack/mask/_mask.py)."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from celestack.mask._mask import Mask

# ---------------------------------------------------------------------------
# Construction from PNG and JPG sources
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("suffix", [".png", ".jpg", ".jpeg"])
def test_mask_init_from_pillow_format(tmp_path: Path, suffix: str) -> None:
    """Mask loads from PNG and JPEG files via the Pillow backend."""
    array = np.zeros((24, 32), dtype=np.uint8)
    array[4:20, 8:24] = 255
    img = Image.fromarray(array, mode="L")
    path = tmp_path / f"mask{suffix}"
    img.save(path)

    mask = Mask(path)
    assert mask.shape == (24, 32)
    assert mask.array.dtype == np.bool_
    assert mask.array[10, 16]  # inside the nonzero region
    assert not mask.array[0, 0]  # outside


@pytest.mark.parametrize("suffix", [".png", ".jpg", ".jpeg"])
def test_mask_rgb_pillow_format_booleanizes(tmp_path: Path, suffix: str) -> None:
    """RGB PNG/JPEG masks are collapsed to a boolean array via grayscale."""
    array = np.zeros((24, 32, 3), dtype=np.uint8)
    array[0:12, :, :] = 200  # top half nonzero
    img = Image.fromarray(array, mode="RGB")
    path = tmp_path / f"mask{suffix}"
    img.save(path)

    mask = Mask(path)
    assert mask.array.dtype == np.bool_
    assert mask.array[0, 0]
    assert not mask.array[23, 0]


# ---------------------------------------------------------------------------
# _from_array — downscale_factor override
# ---------------------------------------------------------------------------


def test_from_array_custom_downscale_factor() -> None:
    """_from_array stores a custom downscale_factor when supplied."""
    mask = Mask._from_array(np.zeros((8, 8), dtype=np.bool_), downscale_factor=4)
    assert mask.downscale_factor == 4


def test_from_array_default_downscale_factor_is_1() -> None:
    """_from_array uses downscale_factor=1 when the argument is omitted."""
    mask = Mask._from_array(np.zeros((8, 8), dtype=np.bool_))
    assert mask.downscale_factor == 1


# ---------------------------------------------------------------------------
# Lazy reload after unload
# ---------------------------------------------------------------------------


def test_array_reloads_after_unload(gray_mask_path: Path) -> None:
    """array can be accessed again after unload() without error."""
    mask = Mask(gray_mask_path)
    first = mask.array.copy()
    mask.unload()
    assert mask._array_cache is None
    np.testing.assert_array_equal(mask.array, first)


# ---------------------------------------------------------------------------
# save — in-memory mask (no backing path)
# ---------------------------------------------------------------------------


def test_save_in_memory_mask(tmp_path: Path) -> None:
    """save() works on an in-memory mask produced by _from_array."""
    array = np.zeros((10, 20), dtype=np.bool_)
    array[2:8, 5:15] = True
    mask = Mask._from_array(array)
    saved = mask.save(tmp_path / "out.tif")
    np.testing.assert_array_equal(saved.array, array)


# ---------------------------------------------------------------------------
# save_downscaled — factor=1 edge case
# ---------------------------------------------------------------------------


def test_save_downscaled_factor_1_preserves_content(
    gray_mask: Mask, tmp_path: Path
) -> None:
    """save_downscaled with factor=1 produces the same boolean content."""
    proxy = gray_mask.save_downscaled(tmp_path / "proxy.tif", 1)
    assert proxy.downscale_factor == 1
    np.testing.assert_array_equal(proxy.array, gray_mask.array)


# ---------------------------------------------------------------------------
# _conserve_cache
# ---------------------------------------------------------------------------


def test_conserve_cache_unloads_if_not_preloaded(gray_mask_path: Path) -> None:
    """_conserve_cache unloads the array on exit if it wasn't loaded before."""
    mask = Mask(gray_mask_path)
    assert mask._array_cache is None
    with mask._conserve_cache():
        _ = mask.array
        assert mask._array_cache is not None
    assert mask._array_cache is None


def test_conserve_cache_keeps_array_if_preloaded(gray_mask_path: Path) -> None:
    """_conserve_cache leaves the array loaded if it was already cached."""
    mask = Mask(gray_mask_path)
    _ = mask.array  # pre-load
    assert mask._array_cache is not None
    with mask._conserve_cache():
        pass
    assert mask._array_cache is not None


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------


def test_repr_with_path(gray_mask_path: Path) -> None:
    """repr includes the path string and downscale_factor."""
    mask = Mask(gray_mask_path)
    r = repr(mask)
    assert str(gray_mask_path) in r
    assert "downscale_factor=1" in r


def test_repr_without_path() -> None:
    """repr renders None for in-memory masks."""
    mask = Mask._from_array(np.zeros((4, 4), dtype=np.bool_))
    r = repr(mask)
    assert "None" in r
    assert "downscale_factor=1" in r
