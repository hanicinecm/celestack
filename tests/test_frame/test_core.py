"""Tests for the Frame class."""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import pytest

from celestack.exceptions import DownscaleError
from celestack.frame import Frame
from tests.utils import (
    CAMERA_MAKE,
    CAMERA_MODEL,
    EXIF_DATETIME_EPOCH,
    IMG_H,
    IMG_W,
    LENS_MODEL,
)

# --- Constructor + metadata ---


def test_load_rgb_tiff(tmp_rgb_tiff: Path) -> None:
    """Load an RGB TIFF and verify basic properties."""
    frame = Frame(tmp_rgb_tiff)
    assert frame.path == tmp_rgb_tiff
    assert frame.downscale_factor == 1
    assert frame.bit_depth == 16
    assert frame.metadata.camera_make == CAMERA_MAKE
    assert frame.metadata.camera_model == CAMERA_MODEL


def test_load_jpeg(tmp_rgb_jpeg: Path) -> None:
    """Load a JPEG and verify basic properties."""
    frame = Frame(tmp_rgb_jpeg)
    assert frame.path == tmp_rgb_jpeg
    assert frame.downscale_factor == 1
    assert frame.bit_depth == 8
    assert frame.metadata.camera_model == CAMERA_MODEL


def test_load_celestack_tiff(tmp_celestack_tiff: Path) -> None:
    """Load a Celestack proxy TIFF and verify embedded metadata."""
    frame = Frame(tmp_celestack_tiff)
    assert frame.downscale_factor == 4
    assert frame.timestamp == pytest.approx(1752620400.0)


def test_timestamp_from_exif(tmp_rgb_tiff: Path) -> None:
    """Verify float timestamp derived from EXIF datetime."""
    frame = Frame(tmp_rgb_tiff)
    assert frame.timestamp == pytest.approx(EXIF_DATETIME_EPOCH)


def test_no_metadata_timestamp_is_none(tmp_no_metadata_tiff: Path) -> None:
    """TIFF without EXIF and no Celestack metadata has None timestamp."""
    frame = Frame(tmp_no_metadata_tiff)
    assert frame.timestamp is None


def test_optional_metadata_missing(tmp_no_metadata_tiff: Path) -> None:
    """Missing EXIF fields default to None."""
    frame = Frame(tmp_no_metadata_tiff)
    assert frame.metadata.camera_make is None
    assert frame.metadata.camera_model is None
    assert frame.metadata.exposure is None
    assert frame.metadata.f_number is None
    assert frame.metadata.iso is None
    assert frame.metadata.focal_length is None
    assert frame.metadata.lens_model is None


def test_unsupported_suffix_raises(tmp_path: Path) -> None:
    """Reject files with unsupported extensions."""
    bmp = tmp_path / "image.bmp"
    bmp.write_bytes(b"\x00")
    with pytest.raises(ValueError, match="No backend found"):
        Frame(bmp)


def test_file_not_found_raises(tmp_path: Path) -> None:
    """Reject paths that do not exist."""
    with pytest.raises(FileNotFoundError, match="Frame not found"):
        Frame(tmp_path / "nonexistent.tif")


def test_external_file_downscale_factor_is_1(
    tmp_rgb_tiff: Path, tmp_rgb_jpeg: Path
) -> None:
    """External TIFF and JPEG always have downscale_factor=1."""
    assert Frame(tmp_rgb_tiff).downscale_factor == 1
    assert Frame(tmp_rgb_jpeg).downscale_factor == 1


# --- Timestamp property ---


def test_timestamp_setter_overrides_exif(tmp_rgb_tiff: Path) -> None:
    """Setter override takes precedence over EXIF datetime."""
    frame = Frame(tmp_rgb_tiff)
    frame.timestamp = 999.0
    assert frame.timestamp == pytest.approx(999.0)


def test_timestamp_setter_on_no_exif(tmp_no_metadata_tiff: Path) -> None:
    """Setter works on frames with no EXIF datetime."""
    frame = Frame(tmp_no_metadata_tiff)
    assert frame.timestamp is None
    frame.timestamp = 42.0
    assert frame.timestamp == pytest.approx(42.0)


def test_timestamp_getter_converts_exif_to_float(tmp_rgb_tiff: Path) -> None:
    """EXIF datetime is correctly converted to epoch seconds."""
    frame = Frame(tmp_rgb_tiff)
    ts = frame.timestamp
    assert isinstance(ts, float)
    assert ts == pytest.approx(EXIF_DATETIME_EPOCH)


def test_celestack_timestamp_overrides_exif(tmp_celestack_tiff: Path) -> None:
    """Celestack metadata timestamp takes precedence over EXIF."""
    frame = Frame(tmp_celestack_tiff)
    # Even if EXIF were present, the Celestack override wins
    assert frame.timestamp == pytest.approx(1752620400.0)


def test_timestamp_reset_to_none(tmp_rgb_tiff: Path) -> None:
    """Resetting override to None falls through to EXIF timestamp."""
    frame = Frame(tmp_rgb_tiff)
    original_ts = frame.timestamp
    frame.timestamp = 999.0
    assert frame.timestamp == pytest.approx(999.0)
    frame.timestamp = None
    assert frame.timestamp == pytest.approx(original_ts)


# --- Lazy loading ---


def test_array_not_loaded_on_init(tmp_rgb_tiff: Path) -> None:
    """Array is not loaded into memory during construction."""
    frame = Frame(tmp_rgb_tiff)
    assert frame._array_cache is None


def test_array_loads_on_access(tmp_rgb_tiff: Path) -> None:
    """Accessing .array returns a numpy array with expected shape and dtype."""
    frame = Frame(tmp_rgb_tiff)
    arr = frame.array
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (IMG_H, IMG_W, 3)
    assert arr.dtype == np.uint16


def test_unload_frees_array(tmp_rgb_tiff: Path) -> None:
    """After unload, the cached array is released."""
    frame = Frame(tmp_rgb_tiff)
    _ = frame.array
    assert frame._array_cache is not None
    frame.unload()
    assert frame._array_cache is None


def test_array_reloads_after_unload(tmp_rgb_tiff: Path) -> None:
    """Array can be re-loaded after unload with identical data."""
    frame = Frame(tmp_rgb_tiff)
    first = frame.array.copy()
    frame.unload()
    second = frame.array
    np.testing.assert_array_equal(first, second)


# --- Save ---


def test_save_returns_frame(tmp_rgb_tiff: Path, tmp_path: Path) -> None:
    """save() returns a Frame pointing to the new path."""
    frame = Frame(tmp_rgb_tiff)
    saved = frame.save(tmp_path / "saved.tif")
    assert isinstance(saved, Frame)
    assert saved.path == tmp_path / "saved.tif"


def test_save_copies_tiff_verbatim(tmp_gray_tiff: Path, tmp_path: Path) -> None:
    """TIFF is copied as-is (byte-identical)."""
    frame = Frame(tmp_gray_tiff)
    dest = tmp_path / "copied.tif"
    frame.save(dest)
    assert tmp_gray_tiff.read_bytes() == dest.read_bytes()


def test_save_preserves_array_data(tmp_rgb_tiff: Path, tmp_path: Path) -> None:
    """Pixel data survives a save-reload cycle."""
    frame = Frame(tmp_rgb_tiff)
    original = frame.array.copy()
    saved = frame.save(tmp_path / "copy.tif")
    np.testing.assert_array_equal(saved.array, original)


def test_save_preserves_exif_metadata(tmp_rgb_tiff: Path, tmp_path: Path) -> None:
    """Camera model and datetime survive a save-reload cycle."""
    frame = Frame(tmp_rgb_tiff)
    saved = frame.save(tmp_path / "with_meta.tif")
    assert saved.metadata.camera_model == CAMERA_MODEL
    assert saved.timestamp is not None


def test_save_preserves_all_exif_fields(tmp_rgb_jpeg: Path, tmp_path: Path) -> None:
    """All EXIF fields round-trip through save()."""
    frame = Frame(tmp_rgb_jpeg)
    saved = frame.save(tmp_path / "saved.tif")
    assert saved.metadata.camera_make == CAMERA_MAKE
    assert saved.metadata.camera_model == CAMERA_MODEL
    assert saved.metadata.datetime is not None
    assert saved.metadata.exposure is not None
    assert saved.metadata.f_number is not None
    assert saved.metadata.iso is not None
    assert saved.metadata.focal_length is not None
    assert saved.metadata.lens_model == LENS_MODEL


def test_save_does_not_embed_celestack_metadata(
    tmp_rgb_tiff: Path, tmp_path: Path
) -> None:
    """Saved full-res frames have no Celestack metadata."""
    frame = Frame(tmp_rgb_tiff)
    saved = frame.save(tmp_path / "fullres.tif")
    assert saved.downscale_factor == 1


def test_save_copies_tiled_tiff(tmp_tiled_tiff: Path, tmp_path: Path) -> None:
    """Tiled TIFF is copied verbatim, not re-encoded."""
    frame = Frame(tmp_tiled_tiff)
    dest = tmp_path / "copied.tif"
    frame.save(dest)
    assert tmp_tiled_tiff.read_bytes() == dest.read_bytes()


def test_save_rejects_non_tiff_extension(tmp_rgb_tiff: Path, tmp_path: Path) -> None:
    """save() raises ValueError for non-TIFF output path."""
    frame = Frame(tmp_rgb_tiff)
    with pytest.raises(ValueError, match="TIFF"):
        frame.save(tmp_path / "out.png")


def test_save_creates_parent_directories(tmp_rgb_tiff: Path, tmp_path: Path) -> None:
    """save() creates missing parent directories."""
    frame = Frame(tmp_rgb_tiff)
    dest = tmp_path / "nested" / "deep" / "out.tif"
    saved = frame.save(dest)
    assert saved.path == dest
    assert dest.exists()


def test_save_jpeg_as_tiff_preserves_pixels(tmp_rgb_jpeg: Path, tmp_path: Path) -> None:
    """Pixel data from a JPEG source survives re-encoding as TIFF."""
    frame = Frame(tmp_rgb_jpeg)
    original = frame.array.copy()
    saved = frame.save(tmp_path / "from_jpeg.tif")
    np.testing.assert_array_equal(saved.array, original)


# --- Downscale ---


def test_save_downscaled_dimensions(tmp_rgb_tiff: Path, tmp_path: Path) -> None:
    """Downscale factor=2 halves each spatial dimension."""
    frame = Frame(tmp_rgb_tiff)
    proxy = frame.save_downscaled(tmp_path / "proxy.tif", downscale_factor=2)
    arr = proxy.array
    assert arr.shape[0] == IMG_H // 2
    assert arr.shape[1] == IMG_W // 2


def test_save_downscaled_grayscale(tmp_rgb_tiff: Path, tmp_path: Path) -> None:
    """RGB input with grayscale=True produces a 2D array."""
    frame = Frame(tmp_rgb_tiff)
    proxy = frame.save_downscaled(tmp_path / "gray.tif", downscale_factor=2)
    assert proxy.array.ndim == 2


def test_save_downscaled_no_grayscale(tmp_rgb_tiff: Path, tmp_path: Path) -> None:
    """grayscale=False preserves RGB channels."""
    frame = Frame(tmp_rgb_tiff)
    proxy = frame.save_downscaled(
        tmp_path / "rgb.tif", downscale_factor=2, grayscale=False
    )
    assert proxy.array.ndim == 3
    assert proxy.array.shape[2] == 3


def test_save_downscaled_bit_depth(tmp_rgb_tiff: Path, tmp_path: Path) -> None:
    """16-bit input downscaled to 8-bit produces uint8 output."""
    frame = Frame(tmp_rgb_tiff)
    assert frame.bit_depth == 16
    proxy = frame.save_downscaled(
        tmp_path / "8bit.tif", downscale_factor=2, bit_depth=8
    )
    assert proxy.bit_depth == 8
    assert proxy.array.dtype == np.uint8


def test_save_downscaled_returns_frame(tmp_rgb_tiff: Path, tmp_path: Path) -> None:
    """Returned Frame has correct downscale_factor read from file."""
    frame = Frame(tmp_rgb_tiff)
    proxy = frame.save_downscaled(tmp_path / "proxy.tif", downscale_factor=4)
    assert proxy.downscale_factor == 4


def test_save_downscaled_from_proxy_raises(tmp_rgb_tiff: Path, tmp_path: Path) -> None:
    """Downscaling an already-downscaled frame raises DownscaleError."""
    frame = Frame(tmp_rgb_tiff)
    proxy = frame.save_downscaled(tmp_path / "proxy.tif", downscale_factor=2)
    with pytest.raises(DownscaleError, match="Cannot downscale a proxy"):
        proxy.save_downscaled(tmp_path / "double.tif", downscale_factor=2)


def test_save_downscaled_carries_timestamp(tmp_rgb_tiff: Path, tmp_path: Path) -> None:
    """Source timestamp is preserved in the downscaled output."""
    frame = Frame(tmp_rgb_tiff)
    source_ts = frame.timestamp
    proxy = frame.save_downscaled(tmp_path / "proxy.tif", downscale_factor=2)
    assert proxy.timestamp == pytest.approx(source_ts)


def test_save_downscaled_preserves_exif_metadata(
    tmp_rgb_jpeg: Path, tmp_path: Path
) -> None:
    """EXIF metadata survives a downscale round-trip."""
    frame = Frame(tmp_rgb_jpeg)
    proxy = frame.save_downscaled(tmp_path / "proxy.tif", downscale_factor=2)
    assert proxy.metadata.camera_make == CAMERA_MAKE
    assert proxy.metadata.camera_model == CAMERA_MODEL
    assert proxy.metadata.datetime is not None
    assert proxy.metadata.exposure is not None
    assert proxy.metadata.f_number is not None
    assert proxy.metadata.iso is not None
    assert proxy.metadata.focal_length is not None
    assert proxy.metadata.lens_model == LENS_MODEL


def test_save_downscaled_no_timestamp_raises(
    tmp_no_metadata_tiff: Path, tmp_path: Path
) -> None:
    """Frame with no timestamp raises ValueError on downscale."""
    frame = Frame(tmp_no_metadata_tiff)
    assert frame.timestamp is None
    with pytest.raises(ValueError, match="non-None timestamp"):
        frame.save_downscaled(tmp_path / "proxy.tif", downscale_factor=2)


def test_save_downscaled_rejects_non_tiff_extension(
    tmp_rgb_tiff: Path, tmp_path: Path
) -> None:
    """save_downscaled() raises ValueError for non-TIFF output path."""
    frame = Frame(tmp_rgb_tiff)
    with pytest.raises(ValueError, match="TIFF"):
        frame.save_downscaled(tmp_path / "proxy.png", downscale_factor=2)


def test_save_downscaled_rejects_factor_below_one(
    tmp_rgb_tiff: Path, tmp_path: Path
) -> None:
    """downscale_factor < 1 raises ValueError."""
    frame = Frame(tmp_rgb_tiff)
    with pytest.raises(ValueError, match="downscale_factor must be >= 1"):
        frame.save_downscaled(tmp_path / "proxy.tif", downscale_factor=0)


def test_save_downscaled_factor_one(tmp_rgb_tiff: Path, tmp_path: Path) -> None:
    """factor=1 embeds Celestack metadata without spatial change."""
    frame = Frame(tmp_rgb_tiff)
    proxy = frame.save_downscaled(
        tmp_path / "proxy.tif", downscale_factor=1, bit_depth=8
    )
    assert proxy.downscale_factor == 1
    assert proxy.array.shape[0] == IMG_H
    assert proxy.array.shape[1] == IMG_W
    assert proxy.bit_depth == 8
    assert proxy.timestamp is not None


def test_save_downscaled_to_16bit(tmp_rgb_tiff: Path, tmp_path: Path) -> None:
    """16-bit input downscaled to 16-bit preserves uint16 dtype."""
    frame = Frame(tmp_rgb_tiff)
    proxy = frame.save_downscaled(
        tmp_path / "16bit.tif", downscale_factor=2, bit_depth=16
    )
    assert proxy.bit_depth == 16
    assert proxy.array.dtype == np.uint16


def test_save_downscaled_to_32bit(tmp_rgb_tiff: Path, tmp_path: Path) -> None:
    """16-bit input downscaled to 32-bit produces float32 output."""
    frame = Frame(tmp_rgb_tiff)
    proxy = frame.save_downscaled(
        tmp_path / "32bit.tif", downscale_factor=2, bit_depth=32
    )
    assert proxy.bit_depth == 32
    assert proxy.array.dtype == np.float32


def test_save_downscaled_grayscale_input(tmp_gray_tiff: Path, tmp_path: Path) -> None:
    """Already-grayscale frame with grayscale=True stays 2D."""
    frame = Frame(tmp_gray_tiff)
    frame.timestamp = 1.0
    proxy = frame.save_downscaled(
        tmp_path / "gray_proxy.tif", downscale_factor=2, grayscale=True
    )
    assert proxy.array.ndim == 2


# --- Tile reading ---


def test_read_tile_correct_region(tmp_rgb_tiff: Path) -> None:
    """Tile read returns the same data as slicing the full array."""
    frame = Frame(tmp_rgb_tiff)
    full = frame.array
    tile = frame.read_tile(10, 5, 30, 20)
    np.testing.assert_array_equal(tile, full[5:20, 10:30])


def test_read_tile_grayscale_tiff(tmp_gray_tiff: Path) -> None:
    """Grayscale TIFF supports tile reading."""
    frame = Frame(tmp_gray_tiff)
    full = frame.array
    tile = frame.read_tile(0, 0, 16, 16)
    np.testing.assert_array_equal(tile, full[0:16, 0:16])


def test_read_tile_out_of_bounds_raises(tmp_rgb_tiff: Path) -> None:
    """Out-of-bounds tile coordinates raise ValueError."""
    frame = Frame(tmp_rgb_tiff)
    with pytest.raises(ValueError, match="outside frame bounds"):
        frame.read_tile(0, 0, IMG_W + 1, IMG_H)


def test_read_tile_zero_width_raises(tmp_rgb_tiff: Path) -> None:
    """Tile with zero width (x1 == x0) raises ValueError."""
    frame = Frame(tmp_rgb_tiff)
    with pytest.raises(ValueError, match="x1 must be greater than x0"):
        frame.read_tile(10, 0, 10, 20)


def test_read_tile_negative_coords_raises(tmp_rgb_tiff: Path) -> None:
    """Negative tile coordinates raise ValueError."""
    frame = Frame(tmp_rgb_tiff)
    with pytest.raises(ValueError, match="non-negative"):
        frame.read_tile(-1, 0, 10, 10)


def test_read_tile_unload_drops_cache(tmp_rgb_tiff: Path) -> None:
    """unload_array=True drops the cached array after reading tile."""
    frame = Frame(tmp_rgb_tiff)
    _ = frame.array
    assert frame._array_cache is not None
    frame.read_tile(0, 0, 10, 10, unload_array=True)
    assert frame._array_cache is None


def test_read_tile_full_extent(tmp_rgb_tiff: Path) -> None:
    """Full-extent tile equals the full array."""
    frame = Frame(tmp_rgb_tiff)
    full = frame.array.copy()
    tile = frame.read_tile(0, 0, IMG_W, IMG_H)
    np.testing.assert_array_equal(tile, full)


# --- Plot ---


def test_plot_rgb_returns_figure(tmp_rgb_tiff: Path) -> None:
    """Plotting an RGB frame returns a Plotly Figure."""
    fig = Frame(tmp_rgb_tiff).plot()
    assert isinstance(fig, go.Figure)


def test_plot_grayscale_returns_figure(tmp_gray_tiff: Path) -> None:
    """Plotting a grayscale frame returns a Plotly Figure."""
    fig = Frame(tmp_gray_tiff).plot()
    assert isinstance(fig, go.Figure)


def test_plot_boolean_mask_returns_figure(tmp_bool_mask_tiff: Path) -> None:
    """Plotting a boolean mask returns a Plotly Figure."""
    fig = Frame(tmp_bool_mask_tiff).plot()
    assert isinstance(fig, go.Figure)


def test_plot_axes_in_fullres_coords(tmp_rgb_tiff: Path) -> None:
    """Axes ranges match full-res pixel dimensions."""
    fig = Frame(tmp_rgb_tiff).plot()
    layout = fig.to_plotly_json()["layout"]
    x_range = layout["xaxis"]["range"]
    y_range = layout["yaxis"]["range"]
    assert x_range[0] == 0
    assert x_range[1] == IMG_W
    assert y_range[0] == IMG_H
    assert y_range[1] == 0


def test_plot_proxy_axes_in_fullres_coords(
    tmp_celestack_tiff: Path,
) -> None:
    """Proxy frame axes cover full-res extent, not proxy dimensions."""
    frame = Frame(tmp_celestack_tiff)
    proxy_h, proxy_w = frame.array.shape[:2]
    fig = frame.plot()
    layout = fig.to_plotly_json()["layout"]
    x_range = layout["xaxis"]["range"]
    y_range = layout["yaxis"]["range"]
    assert x_range[1] == proxy_w * frame.downscale_factor
    assert y_range[0] == proxy_h * frame.downscale_factor


def test_plot_show_pixels_grayscale(tmp_gray_tiff: Path) -> None:
    """show_pixels=True on grayscale produces a Heatmap trace."""
    fig = Frame(tmp_gray_tiff).plot(show_pixels=True)
    assert isinstance(fig, go.Figure)
    assert len(fig.to_plotly_json()["data"]) == 1
    assert isinstance(fig.data[0], go.Heatmap)


def test_plot_show_pixels_rgb(tmp_rgb_tiff: Path) -> None:
    """show_pixels=True on RGB produces an Image trace."""
    fig = Frame(tmp_rgb_tiff).plot(show_pixels=True)
    assert isinstance(fig, go.Figure)
    assert len(fig.to_plotly_json()["data"]) == 1
    assert isinstance(fig.data[0], go.Image)


# --- Repr ---


def test_repr_format(tmp_rgb_tiff: Path) -> None:
    """__repr__ includes the path and downscale_factor."""
    frame = Frame(tmp_rgb_tiff)
    r = repr(frame)
    assert "Frame(" in r
    assert str(tmp_rgb_tiff) in r
    assert "downscale_factor=1" in r
