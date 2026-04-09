"""Tests for the Mask class (src/celestack/mask/core.py)."""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import pytest
import tifffile
from PIL import Image

from celestack.exceptions import DownscaleError
from celestack.mask.core import Mask


# ===========================================================================
# Mask — construction from path
# ===========================================================================


def test_mask_init_grayscale(gray_mask_path: Path) -> None:
    """Mask loads from a grayscale TIFF without error."""
    mask = Mask(gray_mask_path)
    assert mask.path == gray_mask_path


def test_mask_init_not_found_raises(tmp_path: Path) -> None:
    """Mask raises FileNotFoundError for a missing path."""
    with pytest.raises(FileNotFoundError, match="Mask not found"):
        Mask(tmp_path / "nonexistent.tif")


def test_mask_init_unsupported_format_raises(tmp_path: Path) -> None:
    """Mask raises ValueError for an unsupported file extension."""
    bad = tmp_path / "mask.xyz"
    bad.write_bytes(b"data")
    with pytest.raises(ValueError, match="No backend found"):
        Mask(bad)


def test_mask_shape_is_2d(gray_mask_path: Path) -> None:
    """Shape is always (H, W) regardless of on-disk format."""
    mask = Mask(gray_mask_path)
    assert len(mask.shape) == 2
    assert mask.shape == (24, 32)


def test_mask_shape_from_rgb_file(rgb_mask_path: Path) -> None:
    """Shape is 2D even when loaded from an RGB file."""
    mask = Mask(rgb_mask_path)
    assert mask.shape == (24, 32)


def test_mask_downscale_factor_default(gray_mask_path: Path) -> None:
    """downscale_factor defaults to 1 for a plain TIFF."""
    assert Mask(gray_mask_path).downscale_factor == 1


# ===========================================================================
# Mask — array property
# ===========================================================================


def test_mask_array_is_bool(gray_mask_path: Path) -> None:
    """array dtype is always np.bool_."""
    assert Mask(gray_mask_path).array.dtype == np.bool_


def test_mask_array_thresholds_nonzero_grayscale(gray_mask_path: Path) -> None:
    """Nonzero pixels become True; zero pixels become False."""
    mask = Mask(gray_mask_path)
    assert mask.array[5, 8]
    assert not mask.array[0, 0]


def test_mask_array_thresholds_rgb(rgb_mask_path: Path) -> None:
    """RGB mask: top half (value 200) -> True, bottom half (0) -> False."""
    mask = Mask(rgb_mask_path)
    assert mask.array[0, 0]
    assert not mask.array[23, 0]


def test_mask_array_nonzero_but_not_255(bool_values_mask_path: Path) -> None:
    """Values of 1 (not just 255) are treated as foreground."""
    mask = Mask(bool_values_mask_path)
    assert mask.array[10, 10]
    assert not mask.array[0, 0]


def test_mask_array_shape_is_2d(gray_mask_path: Path) -> None:
    """array always has exactly 2 dimensions."""
    assert Mask(gray_mask_path).array.ndim == 2


def test_mask_array_lazy_loaded(gray_mask_path: Path) -> None:
    """Cache is None before the first array access."""
    mask = Mask(gray_mask_path)
    assert mask._array_cache is None
    _ = mask.array
    assert mask._array_cache is not None


# ===========================================================================
# Mask — unload
# ===========================================================================


def test_mask_unload_clears_cache(gray_mask: Mask) -> None:
    """unload() drops the array cache."""
    _ = gray_mask.array
    gray_mask.unload()
    assert gray_mask._array_cache is None


# ===========================================================================
# Mask — _from_array
# ===========================================================================


def test_mask_from_array_is_detached() -> None:
    """_from_array produces an in-memory mask; path raises AttributeError."""
    array = np.zeros((10, 20), dtype=np.bool_)
    mask = Mask._from_array(array)
    with pytest.raises(AttributeError, match="not backed by a file"):
        _ = mask.path


def test_mask_from_array_stores_bool() -> None:
    """_from_array casts to bool and caches the result immediately."""
    array = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    mask = Mask._from_array(array)
    assert mask.array.dtype == np.bool_
    assert mask.array[0, 0]
    assert not mask.array[0, 1]


def test_mask_from_array_downscale_factor_is_1() -> None:
    """_from_array always sets downscale_factor to 1."""
    mask = Mask._from_array(np.zeros((8, 8), dtype=np.bool_))
    assert mask.downscale_factor == 1


def test_mask_from_array_shape() -> None:
    """_from_array records the correct shape."""
    array = np.zeros((15, 30), dtype=np.bool_)
    mask = Mask._from_array(array)
    assert mask.shape == (15, 30)


# ===========================================================================
# Mask — save_as
# ===========================================================================


def test_mask_save_as_returns_none(gray_mask: Mask, tmp_path: Path) -> None:
    """save_as() returns None."""
    result = gray_mask.save_as(tmp_path / "out.tif")
    assert result is None


def test_mask_save_as_binds_path(gray_mask: Mask, tmp_path: Path) -> None:
    """save_as() binds the instance to the new path."""
    out = tmp_path / "out.tif"
    gray_mask.save_as(out)
    assert gray_mask.path == out


def test_mask_save_as_writes_uint8_grayscale(gray_mask: Mask, tmp_path: Path) -> None:
    """Saved file is a 2D uint8 array."""
    out = tmp_path / "out.tif"
    gray_mask.save_as(out)
    data = tifffile.imread(out)
    assert data.ndim == 2
    assert data.dtype == np.uint8


def test_mask_save_as_true_pixels_are_255(gray_mask: Mask, tmp_path: Path) -> None:
    """True pixels are saved as 255, False pixels as 0."""
    out = tmp_path / "out.tif"
    gray_mask.save_as(out)
    data = tifffile.imread(out)
    assert data[5, 8] == 255
    assert data[0, 0] == 0


def test_mask_save_as_round_trips(gray_mask: Mask, tmp_path: Path) -> None:
    """Loading a saved mask produces the same boolean array."""
    original = gray_mask.array.copy()
    out = tmp_path / "out.tif"
    gray_mask.save_as(out)
    loaded = Mask(out)
    np.testing.assert_array_equal(loaded.array, original)


def test_mask_save_as_wrong_extension_raises(gray_mask: Mask, tmp_path: Path) -> None:
    """save_as() raises ValueError for non-TIFF paths."""
    with pytest.raises(ValueError, match="TIFF file"):
        gray_mask.save_as(tmp_path / "out.png")


def test_mask_save_as_creates_parent_dirs(gray_mask: Mask, tmp_path: Path) -> None:
    """save_as() creates intermediate directories as needed."""
    out = tmp_path / "a" / "b" / "out.tif"
    gray_mask.save_as(out)
    assert out.exists()


def test_mask_save_as_conserves_cache(gray_mask_path: Path, tmp_path: Path) -> None:
    """save_as() does not leave the array loaded if it wasn't before."""
    mask = Mask(gray_mask_path)
    assert mask._array_cache is None
    mask.save_as(tmp_path / "out.tif")
    assert mask._array_cache is None


def test_mask_save_as_overwrite_false_raises(gray_mask: Mask, tmp_path: Path) -> None:
    """save_as() raises FileExistsError when destination exists and overwrite is False."""
    out = tmp_path / "out.tif"
    gray_mask.save_as(out)
    with pytest.raises(FileExistsError):
        gray_mask.save_as(out)


def test_mask_save_as_overwrite_true_succeeds(gray_mask: Mask, tmp_path: Path) -> None:
    """save_as() succeeds when overwrite=True even if the file exists."""
    out = tmp_path / "out.tif"
    gray_mask.save_as(out)
    gray_mask.save_as(out, overwrite=True)
    assert out.exists()


def test_mask_save_as_in_memory_mask(tmp_path: Path) -> None:
    """save_as() works on an in-memory mask produced by _from_array."""
    array = np.zeros((10, 20), dtype=np.bool_)
    array[2:8, 5:15] = True
    mask = Mask._from_array(array)
    out = tmp_path / "out.tif"
    mask.save_as(out)
    assert mask.path == out
    loaded = Mask(out)
    np.testing.assert_array_equal(loaded.array, array)


# ===========================================================================
# Mask — downscaled_copy
# ===========================================================================


def test_mask_downscaled_copy_returns_mask(gray_mask: Mask) -> None:
    """downscaled_copy() returns a Mask instance."""
    result = gray_mask.downscaled_copy(2)
    assert isinstance(result, Mask)


def test_mask_downscaled_copy_is_detached(gray_mask: Mask) -> None:
    """downscaled_copy() returns an in-memory mask with no backing path."""
    proxy = gray_mask.downscaled_copy(2)
    with pytest.raises(AttributeError, match="not backed by a file"):
        _ = proxy.path


def test_mask_downscaled_copy_factor(gray_mask: Mask) -> None:
    """The returned proxy mask has the correct downscale_factor."""
    proxy = gray_mask.downscaled_copy(4)
    assert proxy.downscale_factor == 4


def test_mask_downscaled_copy_reduces_shape(gray_mask: Mask) -> None:
    """Proxy shape is approximately (H/factor, W/factor)."""
    proxy = gray_mask.downscaled_copy(2)
    assert proxy.shape == (12, 16)


def test_mask_downscaled_copy_factor_persisted_via_save_as(
    gray_mask: Mask, tmp_path: Path
) -> None:
    """Calling save_as on a downscaled_copy embeds the downscale_factor in metadata."""
    proxy = gray_mask.downscaled_copy(4)
    proxy.save_as(tmp_path / "proxy.tif")
    loaded = Mask(tmp_path / "proxy.tif")
    assert loaded.downscale_factor == 4


def test_mask_downscaled_copy_majority_voting(tmp_path: Path) -> None:
    """Blocks with >50% foreground become True; <=50% become False."""
    array = np.array(
        [
            [True, True, False, False],
            [True, False, False, False],
            [True, False, True, True],
            [False, False, False, True],
        ],
        dtype=np.bool_,
    )
    mask = Mask._from_array(array)
    proxy = mask.downscaled_copy(2)
    result = proxy.array
    assert result[0, 0]
    assert not result[0, 1]
    assert not result[1, 0]
    assert result[1, 1]


def test_mask_downscaled_copy_exactly_50_percent_is_background() -> None:
    """Blocks with exactly 50% foreground become False (threshold is strictly >0.5)."""
    array = np.array(
        [
            [True, False, True, True],
            [True, False, True, False],
        ],
        dtype=np.bool_,
    )
    mask = Mask._from_array(array)
    proxy = mask.downscaled_copy(2)
    result = proxy.array
    assert not result[0, 0]
    assert result[0, 1]


def test_mask_downscaled_copy_proxy_raises(gray_mask: Mask, tmp_path: Path) -> None:
    """downscaled_copy() raises DownscaleError when called on a proxy mask."""
    proxy = gray_mask.downscaled_copy(2)
    with pytest.raises(DownscaleError, match="proxy mask"):
        proxy.downscaled_copy(2)


def test_mask_downscaled_copy_invalid_factor_raises(gray_mask: Mask) -> None:
    """downscaled_copy() raises ValueError for factor < 1."""
    with pytest.raises(ValueError, match="downscale_factor"):
        gray_mask.downscaled_copy(0)


def test_mask_downscaled_copy_conserves_cache(gray_mask_path: Path) -> None:
    """downscaled_copy() does not leave the array loaded if it wasn't before."""
    mask = Mask(gray_mask_path)
    assert mask._array_cache is None
    mask.downscaled_copy(2)
    assert mask._array_cache is None


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
# downscaled_copy — factor=1 edge case
# ---------------------------------------------------------------------------


def test_downscaled_copy_factor_1_preserves_content(gray_mask: Mask) -> None:
    """downscaled_copy with factor=1 produces the same boolean content."""
    proxy = gray_mask.downscaled_copy(1)
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
    """repr renders '<detached>' for in-memory masks."""
    mask = Mask._from_array(np.zeros((4, 4), dtype=np.bool_))
    r = repr(mask)
    assert "<detached>" in r
    assert "downscale_factor=1" in r


# ---------------------------------------------------------------------------
# mask_rectangle
# ---------------------------------------------------------------------------


def test_mask_rectangle_sets_region() -> None:
    """mask_rectangle sets a rectangular region to the given value."""
    mask = Mask._from_array(np.zeros((12, 16), dtype=np.bool_))
    mask.mask_rectangle(0, 0, 8, 6, foreground=True)
    assert mask.array[0, 0]
    assert mask.array[5, 7]
    assert not mask.array[6, 0]


def test_mask_rectangle_clears_region() -> None:
    """mask_rectangle can clear a region to background."""
    mask = Mask._from_array(np.ones((12, 16), dtype=np.bool_))
    mask.mask_rectangle(0, 0, 4, 4, foreground=False)
    assert not mask.array[0, 0]
    assert not mask.array[3, 3]
    assert mask.array[4, 4]


def test_mask_rectangle_detaches(gray_mask_path: Path) -> None:
    """mask_rectangle detaches the mask from its backing path."""
    mask = Mask(gray_mask_path)
    assert mask._path is not None
    mask.mask_rectangle(0, 0, 4, 4, foreground=True)
    with pytest.raises(AttributeError, match="not backed by a file"):
        _ = mask.path


def test_mask_rectangle_inverted_bounds_raises() -> None:
    """mask_rectangle raises ValueError for x1 <= x0."""
    mask = Mask._from_array(np.zeros((12, 16), dtype=np.bool_))
    with pytest.raises(ValueError, match="x1 must be greater"):
        mask.mask_rectangle(8, 0, 4, 4, foreground=True)


def test_mask_rectangle_out_of_bounds_raises() -> None:
    """mask_rectangle raises ValueError when rectangle exceeds image bounds."""
    mask = Mask._from_array(np.zeros((12, 16), dtype=np.bool_))
    with pytest.raises(ValueError, match="outside image bounds"):
        mask.mask_rectangle(0, 0, 100, 100, foreground=True)


def test_mask_rectangle_negative_origin_raises() -> None:
    """mask_rectangle raises ValueError for negative coordinates."""
    mask = Mask._from_array(np.zeros((12, 16), dtype=np.bool_))
    with pytest.raises(ValueError, match="non-negative"):
        mask.mask_rectangle(-1, 0, 4, 4, foreground=True)


# ---------------------------------------------------------------------------
# mask_pixels
# ---------------------------------------------------------------------------


def test_mask_pixels_sets_values() -> None:
    """mask_pixels sets specific pixels to the given value."""
    mask = Mask._from_array(np.zeros((12, 16), dtype=np.bool_))
    pixels = np.array([[2, 3], [10, 5]])
    mask.mask_pixels(pixels, foreground=True)
    assert mask.array[3, 2]
    assert mask.array[5, 10]
    assert not mask.array[0, 0]


def test_mask_pixels_detaches(gray_mask_path: Path) -> None:
    """mask_pixels detaches the mask from its backing path."""
    mask = Mask(gray_mask_path)
    assert mask._path is not None
    mask.mask_pixels(np.array([[0, 0]]), foreground=True)
    with pytest.raises(AttributeError, match="not backed by a file"):
        _ = mask.path


def test_mask_pixels_wrong_shape_raises() -> None:
    """mask_pixels raises ValueError for non-(N,2) input."""
    mask = Mask._from_array(np.zeros((12, 16), dtype=np.bool_))
    with pytest.raises(ValueError, match=r"\(N, 2\)"):
        mask.mask_pixels(np.array([0, 1, 2]), foreground=True)


def test_mask_pixels_out_of_bounds_raises() -> None:
    """mask_pixels raises ValueError for out-of-bounds coordinates."""
    mask = Mask._from_array(np.zeros((12, 16), dtype=np.bool_))
    with pytest.raises(ValueError, match="out of image bounds"):
        mask.mask_pixels(np.array([[9999, 0]]), foreground=True)


# ---------------------------------------------------------------------------
# plot
# ---------------------------------------------------------------------------


def test_plot_returns_figure() -> None:
    """plot() returns a Plotly Figure."""
    mask = Mask._from_array(np.zeros((12, 16), dtype=np.bool_))
    fig = mask.plot()
    assert isinstance(fig, go.Figure)


def test_plot_has_layout_image() -> None:
    """plot() adds the mask image as a layout image."""
    mask = Mask._from_array(np.zeros((8, 10), dtype=np.bool_))
    fig = mask.plot()
    assert len(fig.layout.images) > 0


def test_plot_has_no_traces_when_highlight_disabled() -> None:
    """plot(highlight_noise=0) adds no data traces."""
    mask = Mask._from_array(np.zeros((8, 10), dtype=np.bool_))
    fig = mask.plot(highlight_noise=0)
    assert len(fig.data) == 0


def test_plot_axes_account_for_downscale_factor() -> None:
    """plot() scales axes by downscale_factor."""
    mask = Mask._from_array(np.zeros((10, 20), dtype=np.bool_), downscale_factor=4)
    fig = mask.plot()
    assert fig.layout.xaxis.range[1] == 80  # 20 * 4
    assert fig.layout.yaxis.range[0] == 40  # 10 * 4


def test_plot_conserves_cache(gray_mask_path: Path) -> None:
    """plot() does not leave the array loaded if it wasn't before."""
    mask = Mask(gray_mask_path)
    assert mask._array_cache is None
    mask.plot()
    assert mask._array_cache is None


def test_plot_title_shows_filename(gray_mask_path: Path) -> None:
    """plot() uses the filename as the figure title."""
    mask = Mask(gray_mask_path)
    fig = mask.plot()
    assert gray_mask_path.name in str(fig.layout.title.text)


def test_plot_title_detached() -> None:
    """plot() uses '<detached>' as the title for in-memory masks."""
    mask = Mask._from_array(np.zeros((4, 4), dtype=np.bool_))
    fig = mask.plot()
    assert "<detached>" in str(fig.layout.title.text)


# ---------------------------------------------------------------------------
# remove_noise
# ---------------------------------------------------------------------------


def test_remove_noise_cleans_single_pixel_noise() -> None:
    """remove_noise removes single-pixel foreground and background noise."""
    arr = np.zeros((20, 20), dtype=bool)
    arr[0:10, :] = True  # large foreground block
    arr[15, 15] = True  # single FG particle
    arr[5, 5] = False  # single BG hole
    mask = Mask._from_array(arr)
    mask.remove_noise(max_size=1)
    assert not mask.array[15, 15]
    assert mask.array[5, 5]
    assert mask.array[0:10, :].sum() == 200  # block preserved


def test_remove_noise_preserves_large_regions() -> None:
    """remove_noise does not touch regions larger than max_size."""
    arr = np.zeros((20, 20), dtype=bool)
    arr[0:5, 0:5] = True  # 25-pixel block
    mask = Mask._from_array(arr)
    mask.remove_noise(max_size=10)
    assert mask.array[0:5, 0:5].all()


def test_remove_noise_detaches(gray_mask_path: Path) -> None:
    """remove_noise detaches the mask from its backing path."""
    mask = Mask(gray_mask_path)
    assert mask._path is not None
    mask.remove_noise(max_size=1)
    with pytest.raises(AttributeError, match="not backed by a file"):
        _ = mask.path


def test_remove_noise_invalid_max_size_raises() -> None:
    """remove_noise raises ValueError for max_size < 1."""
    mask = Mask._from_array(np.zeros((4, 4), dtype=np.bool_))
    with pytest.raises(ValueError, match="max_size must be >= 1"):
        mask.remove_noise(max_size=0)


def test_remove_noise_default_max_size() -> None:
    """remove_noise defaults to DEFAULT_NOISE_MAX_SIZE (8)."""
    arr = np.zeros((20, 20), dtype=bool)
    arr[0, 0:8] = True  # 8 pixels — exactly at default threshold
    mask = Mask._from_array(arr)
    mask.remove_noise()
    assert not mask.array.any()


# ---------------------------------------------------------------------------
# plot with highlight_noise
# ---------------------------------------------------------------------------


def test_plot_highlight_noise_adds_traces() -> None:
    """plot(highlight_noise=N) adds Scatter traces for noise particles."""
    arr = np.zeros((20, 20), dtype=bool)
    arr[5, 5] = True  # single-pixel FG particle
    mask = Mask._from_array(arr)
    fig = mask.plot(highlight_noise=1)
    assert len(fig.data) > 0
    assert any(t.name == "FG noise" for t in fig.data)


def test_plot_highlight_noise_default_enables() -> None:
    """Default plot() highlights noise using the default max size."""
    arr = np.zeros((20, 20), dtype=bool)
    arr[5, 5] = True  # single-pixel particle, within default threshold
    mask = Mask._from_array(arr)
    fig = mask.plot()
    assert any(t.name == "FG noise" for t in fig.data)


def test_plot_highlight_noise_zero_disables() -> None:
    """plot(highlight_noise=0) disables highlighting."""
    arr = np.zeros((20, 20), dtype=bool)
    arr[5, 5] = True
    mask = Mask._from_array(arr)
    fig = mask.plot(highlight_noise=0)
    assert len(fig.data) == 0


def test_plot_highlight_noise_no_particles_found() -> None:
    """When no noise exists, highlight_noise still works with no traces."""
    arr = np.ones((20, 20), dtype=bool)
    mask = Mask._from_array(arr)
    fig = mask.plot(highlight_noise=1)
    assert len(fig.data) == 0


def test_plot_highlight_noise_conserves_cache(gray_mask_path: Path) -> None:
    """plot(highlight_noise=...) does not leave the array loaded."""
    mask = Mask(gray_mask_path)
    assert mask._array_cache is None
    mask.plot(highlight_noise=5)
    assert mask._array_cache is None
