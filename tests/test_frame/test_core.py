"""Tests for the Frame class."""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import pytest

from celestack.frame import Frame
from tests.utils import (
    CAMERA_MAKE,
    CAMERA_MODEL,
    EXIF_DATETIME_EPOCH,
    IMG_H,
    IMG_W,
    LENS_MODEL,
)


def test_load_rgb_tiff(tmp_rgb_tiff: Path) -> None:
    """Load an RGB TIFF and verify basic properties."""
    frame = Frame(tmp_rgb_tiff)
    assert frame.path == tmp_rgb_tiff
    assert frame.dtype == "uint16"
    assert frame.bit_depth == 16
    assert frame.metadata.camera_make == CAMERA_MAKE
    assert frame.metadata.camera_model == CAMERA_MODEL


def test_load_jpeg(tmp_rgb_jpeg: Path) -> None:
    """Load a JPEG and verify basic properties."""
    frame = Frame(tmp_rgb_jpeg)
    assert frame.path == tmp_rgb_jpeg
    assert frame.dtype == "uint8"
    assert frame.bit_depth == 8
    assert frame.metadata.camera_model == CAMERA_MODEL


def test_timestamp_from_exif(tmp_rgb_tiff: Path) -> None:
    """Verify float timestamp derived from EXIF datetime."""
    frame = Frame(tmp_rgb_tiff)
    assert frame.timestamp == pytest.approx(EXIF_DATETIME_EPOCH)


def test_no_metadata_timestamp_is_none(tmp_no_metadata_tiff: Path) -> None:
    """TIFF without EXIF has None timestamp."""
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


def test_unsupported_dtype_raises(tmp_path: Path) -> None:
    """Frame raises ValueError for images with unsupported dtype."""
    import tifffile

    path = tmp_path / "deep.tif"
    tifffile.imwrite(path, np.zeros((4, 4), dtype=np.uint32))
    with pytest.raises(ValueError, match="Unsupported dtype"):
        Frame(path)


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


def test_timestamp_reset_to_none(tmp_rgb_tiff: Path) -> None:
    """Resetting override to None falls through to EXIF timestamp."""
    frame = Frame(tmp_rgb_tiff)
    original_ts = frame.timestamp
    frame.timestamp = 999.0
    assert frame.timestamp == pytest.approx(999.0)
    frame.timestamp = None
    assert frame.timestamp == pytest.approx(original_ts)


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


def test_load_populates_cache(tmp_rgb_tiff: Path) -> None:
    """load() populates the array cache without returning anything."""
    frame = Frame(tmp_rgb_tiff)
    assert frame._array_cache is None
    result = frame.load()
    assert result is None
    assert frame._array_cache is not None


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


def test_save_as_binds_frame_to_path(tmp_rgb_tiff: Path, tmp_path: Path) -> None:
    """save_as() writes the file and binds the frame to the new path."""
    frame = Frame(tmp_rgb_tiff)
    dest = tmp_path / "saved.tif"
    result = frame.save_as(dest)
    assert result is None
    assert frame._path == dest
    assert dest.exists()


def test_save_as_copies_tiff_verbatim(tmp_gray_tiff: Path, tmp_path: Path) -> None:
    """TIFF source is copied as-is (byte-identical)."""
    frame = Frame(tmp_gray_tiff)
    dest = tmp_path / "copied.tif"
    frame.save_as(dest)
    assert tmp_gray_tiff.read_bytes() == dest.read_bytes()


def test_save_as_preserves_array_data(tmp_rgb_tiff: Path, tmp_path: Path) -> None:
    """Pixel data written by save_as survives a reload cycle."""
    frame = Frame(tmp_rgb_tiff)
    original = frame.array.copy()
    dest = tmp_path / "copy.tif"
    frame.save_as(dest)
    reloaded = Frame(dest)
    np.testing.assert_array_equal(reloaded.array, original)


def test_save_as_preserves_exif_metadata(tmp_rgb_tiff: Path, tmp_path: Path) -> None:
    """Camera model and datetime survive a save_as/reload cycle."""
    frame = Frame(tmp_rgb_tiff)
    dest = tmp_path / "with_meta.tif"
    frame.save_as(dest)
    reloaded = Frame(dest)
    assert reloaded.metadata.camera_model == CAMERA_MODEL
    assert reloaded.timestamp is not None


def test_save_as_preserves_all_exif_fields(tmp_rgb_jpeg: Path, tmp_path: Path) -> None:
    """All EXIF fields round-trip through save_as."""
    frame = Frame(tmp_rgb_jpeg)
    dest = tmp_path / "saved.tif"
    frame.save_as(dest)
    reloaded = Frame(dest)
    assert reloaded.metadata.camera_make == CAMERA_MAKE
    assert reloaded.metadata.camera_model == CAMERA_MODEL
    assert reloaded.metadata.datetime is not None
    assert reloaded.metadata.exposure is not None
    assert reloaded.metadata.f_number is not None
    assert reloaded.metadata.iso is not None
    assert reloaded.metadata.focal_length is not None
    assert reloaded.metadata.lens_model == LENS_MODEL


def test_save_as_copies_tiled_tiff(tmp_tiled_tiff: Path, tmp_path: Path) -> None:
    """Tiled TIFF is copied verbatim, not re-encoded."""
    frame = Frame(tmp_tiled_tiff)
    dest = tmp_path / "copied.tif"
    frame.save_as(dest)
    assert tmp_tiled_tiff.read_bytes() == dest.read_bytes()


def test_save_as_rejects_non_tiff_extension(tmp_rgb_tiff: Path, tmp_path: Path) -> None:
    """save_as() raises ValueError for non-TIFF output path."""
    frame = Frame(tmp_rgb_tiff)
    with pytest.raises(ValueError, match="TIFF"):
        frame.save_as(tmp_path / "out.png")


def test_save_as_raises_if_exists_without_overwrite(
    tmp_rgb_tiff: Path, tmp_path: Path
) -> None:
    """save_as() raises FileExistsError if destination exists and overwrite=False."""
    frame = Frame(tmp_rgb_tiff)
    dest = tmp_path / "out.tif"
    frame.save_as(dest)
    frame2 = Frame(tmp_rgb_tiff)
    with pytest.raises(FileExistsError):
        frame2.save_as(dest)


def test_save_as_overwrites_when_requested(tmp_rgb_tiff: Path, tmp_path: Path) -> None:
    """save_as(overwrite=True) replaces an existing file without error."""
    frame = Frame(tmp_rgb_tiff)
    dest = tmp_path / "out.tif"
    frame.save_as(dest)
    frame2 = Frame(tmp_rgb_tiff)
    frame2.save_as(dest, overwrite=True)
    assert dest.exists()


def test_save_as_creates_parent_directories(tmp_rgb_tiff: Path, tmp_path: Path) -> None:
    """save_as() creates missing parent directories."""
    frame = Frame(tmp_rgb_tiff)
    dest = tmp_path / "nested" / "deep" / "out.tif"
    frame.save_as(dest)
    assert dest.exists()


def test_save_as_jpeg_source_writes_tiff(tmp_rgb_jpeg: Path, tmp_path: Path) -> None:
    """Pixel data from a JPEG source survives re-encoding via save_as."""
    frame = Frame(tmp_rgb_jpeg)
    original = frame.array.copy()
    dest = tmp_path / "from_jpeg.tif"
    frame.save_as(dest)
    reloaded = Frame(dest)
    np.testing.assert_array_equal(reloaded.array, original)


def test_save_as_updates_backend_for_jpeg_source(
    tmp_rgb_jpeg: Path, tmp_path: Path
) -> None:
    """After save_as, a JPEG-sourced frame uses the TIFF backend."""
    from celestack.frame._backends import get_backends

    frame = Frame(tmp_rgb_jpeg)
    dest = tmp_path / "out.tif"
    frame.save_as(dest)
    tiff_backend_type = type(get_backends()[".tif"])
    assert isinstance(frame._backend, tiff_backend_type)


def test_subtract_dark_modifies_self(tmp_path: Path) -> None:
    """subtract_dark() subtracts into self in place and returns None."""
    from tests.utils import write_gray_tiff

    light_path = write_gray_tiff(
        tmp_path / "light.tif",
        np.array([[20, 5], [100, 0]], dtype=np.uint16),
    )
    dark_path = write_gray_tiff(
        tmp_path / "dark.tif",
        np.array([[3, 9], [20, 1]], dtype=np.uint16),
    )
    light = Frame(light_path)
    result = light.subtract_dark(Frame(dark_path))

    assert result is None
    np.testing.assert_array_equal(
        light.array,
        np.array([[17, 0], [80, 0]], dtype=np.uint16),
    )


def test_subtract_dark_hot_pixel_clamps_to_zero(tmp_path: Path) -> None:
    """subtract_dark does not interpolate hot pixels; they clamp to zero."""
    from tests.utils import write_gray_tiff

    light_arr = np.full((3, 3), 1000, dtype=np.uint16)
    dark_arr = np.full((3, 3), 100, dtype=np.uint16)
    dark_arr[1, 1] = 20000

    light = Frame(write_gray_tiff(tmp_path / "light.tif", light_arr))
    dark = Frame(write_gray_tiff(tmp_path / "dark.tif", dark_arr))

    light.subtract_dark(dark)

    assert light.array[0, 0] == 900
    assert light.array[1, 1] == 0


def test_subtract_dark_detaches_from_path(tmp_path: Path) -> None:
    """subtract_dark() detaches the frame from its backing path."""
    from tests.utils import write_gray_tiff

    light_path = write_gray_tiff(
        tmp_path / "light.tif",
        np.full((24, 32), 100, dtype=np.uint16),
    )
    dark_path = write_gray_tiff(
        tmp_path / "dark.tif",
        np.full((24, 32), 10, dtype=np.uint16),
    )
    light = Frame(light_path)
    light.subtract_dark(Frame(dark_path))
    assert light._path is None


def test_subtract_dark_inherits_metadata_and_timestamp(
    tmp_rgb_tiff: Path, tmp_path: Path
) -> None:
    """Metadata and timestamp are preserved after in-place dark subtraction."""
    from tests.utils import write_rgb_tiff

    dark_path = write_rgb_tiff(
        tmp_path / "dark.tif",
        np.zeros((IMG_H, IMG_W, 3), dtype=np.uint16),
    )
    light = Frame(tmp_rgb_tiff)
    light.timestamp = 123.0
    original_meta = light.metadata

    light.subtract_dark(Frame(dark_path))

    assert light.metadata.camera_make == original_meta.camera_make
    assert light.metadata.camera_model == original_meta.camera_model
    assert light.metadata.datetime == original_meta.datetime
    assert light.metadata.exposure == original_meta.exposure
    assert light.metadata.f_number == original_meta.f_number
    assert light.metadata.iso == original_meta.iso
    assert light.metadata.focal_length == original_meta.focal_length
    assert light.metadata.lens_model == original_meta.lens_model
    assert light.timestamp == pytest.approx(123.0)


def test_subtract_dark_result_can_be_saved(tmp_gray_tiff: Path, tmp_path: Path) -> None:
    """A dark-subtracted frame can be persisted via save_as."""
    from tests.utils import write_gray_tiff

    dark_path = write_gray_tiff(
        tmp_path / "dark.tif",
        np.full((IMG_H, IMG_W), fill_value=1, dtype=np.uint16),
    )
    light = Frame(tmp_gray_tiff)
    light.subtract_dark(Frame(dark_path))
    expected = light.array.copy()

    dest = tmp_path / "subtracted.tif"
    light.save_as(dest)

    assert light._path == dest
    reloaded = Frame(dest)
    np.testing.assert_array_equal(reloaded.array, expected)


def test_subtract_dark_shape_mismatch_raises(tmp_path: Path) -> None:
    """Frames with different shapes cannot be subtracted."""
    from tests.utils import write_gray_tiff

    light_path = write_gray_tiff(
        tmp_path / "light.tif",
        np.zeros((10, 10), dtype=np.uint16),
    )
    dark_path = write_gray_tiff(
        tmp_path / "dark.tif",
        np.zeros((10, 11), dtype=np.uint16),
    )

    with pytest.raises(ValueError, match="Shape mismatch"):
        Frame(light_path).subtract_dark(Frame(dark_path))


def test_subtract_dark_dtype_mismatch_raises(tmp_path: Path) -> None:
    """Frames with different dtypes cannot be subtracted."""
    from tests.utils import write_gray_tiff

    light_path = write_gray_tiff(
        tmp_path / "light.tif",
        np.zeros((10, 10), dtype=np.uint16),
    )
    dark_path = write_gray_tiff(
        tmp_path / "dark.tif",
        np.zeros((10, 10), dtype=np.uint8),
    )
    with pytest.raises(ValueError, match="Dtype mismatch"):
        Frame(light_path).subtract_dark(Frame(dark_path))


def test_subtract_dark_conserves_master_dark_cache(tmp_path: Path) -> None:
    """subtract_dark() conserves the cache state of the master dark operand."""
    from tests.utils import write_gray_tiff

    light_path = write_gray_tiff(
        tmp_path / "light.tif", np.full((24, 32), 100, dtype=np.uint16)
    )
    dark_path = write_gray_tiff(
        tmp_path / "dark.tif", np.full((24, 32), 10, dtype=np.uint16)
    )
    light = Frame(light_path)
    dark = Frame(dark_path)
    assert dark._array_cache is None
    light.subtract_dark(dark)
    assert dark._array_cache is None

    light2 = Frame(light_path)
    _ = dark.array
    assert dark._array_cache is not None
    light2.subtract_dark(dark)
    assert dark._array_cache is not None


def test_interpolate_bad_pixels_hot_pixel_replaced(tmp_path: Path) -> None:
    """Hot dark pixels are replaced by 8-neighbor mean in the light frame."""
    from tests.utils import write_gray_tiff

    light_arr = np.full((3, 3), 1000, dtype=np.uint16)
    dark_arr = np.full((3, 3), 100, dtype=np.uint16)
    dark_arr[1, 1] = 20000

    light = Frame(write_gray_tiff(tmp_path / "light.tif", light_arr))
    dark = Frame(write_gray_tiff(tmp_path / "dark.tif", dark_arr))

    light.interpolate_bad_pixels(dark)

    assert light.array[0, 0] == 1000
    assert light.array[1, 1] == 1000


def test_interpolate_bad_pixels_detaches_from_path(tmp_path: Path) -> None:
    """interpolate_bad_pixels() detaches the frame from its backing path."""
    from tests.utils import write_gray_tiff

    light_path = write_gray_tiff(
        tmp_path / "light.tif", np.full((24, 32), 500, dtype=np.uint16)
    )
    dark_path = write_gray_tiff(
        tmp_path / "dark.tif", np.full((24, 32), 10, dtype=np.uint16)
    )
    light = Frame(light_path)
    light.interpolate_bad_pixels(Frame(dark_path))
    assert light._path is None


def test_interpolate_bad_pixels_returns_none(tmp_path: Path) -> None:
    """interpolate_bad_pixels() returns None."""
    from tests.utils import write_gray_tiff

    light_path = write_gray_tiff(
        tmp_path / "light.tif", np.full((24, 32), 500, dtype=np.uint16)
    )
    dark_path = write_gray_tiff(
        tmp_path / "dark.tif", np.full((24, 32), 10, dtype=np.uint16)
    )
    result = Frame(light_path).interpolate_bad_pixels(Frame(dark_path))
    assert result is None


def test_interpolate_bad_pixels_shape_mismatch_raises(tmp_path: Path) -> None:
    """Frames with different shapes cannot be used together."""
    from tests.utils import write_gray_tiff

    light_path = write_gray_tiff(
        tmp_path / "light.tif", np.zeros((10, 10), dtype=np.uint16)
    )
    dark_path = write_gray_tiff(
        tmp_path / "dark.tif", np.zeros((10, 11), dtype=np.uint16)
    )
    with pytest.raises(ValueError, match="Shape mismatch"):
        Frame(light_path).interpolate_bad_pixels(Frame(dark_path))


def test_interpolate_bad_pixels_conserves_master_dark_cache(tmp_path: Path) -> None:
    """interpolate_bad_pixels() conserves the cache state of the master dark."""
    from tests.utils import write_gray_tiff

    light_path = write_gray_tiff(
        tmp_path / "light.tif", np.full((24, 32), 500, dtype=np.uint16)
    )
    dark_path = write_gray_tiff(
        tmp_path / "dark.tif", np.full((24, 32), 10, dtype=np.uint16)
    )
    light = Frame(light_path)
    dark = Frame(dark_path)
    assert dark._array_cache is None
    light.interpolate_bad_pixels(dark)
    assert dark._array_cache is None

    light2 = Frame(light_path)
    _ = dark.array
    assert dark._array_cache is not None
    light2.interpolate_bad_pixels(dark)
    assert dark._array_cache is not None


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


def test_plot_axes_in_native_coords(tmp_rgb_tiff: Path) -> None:
    """Axes ranges match the frame's native pixel dimensions."""
    fig = Frame(tmp_rgb_tiff).plot()
    layout = fig.to_plotly_json()["layout"]
    x_range = layout["xaxis"]["range"]
    y_range = layout["yaxis"]["range"]
    assert x_range[0] == 0
    assert x_range[1] == IMG_W
    assert y_range[0] == IMG_H
    assert y_range[1] == 0


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


def test_save_as_conserves_unloaded_cache(tmp_rgb_jpeg: Path, tmp_path: Path) -> None:
    """save_as() does not leave the array cached when it was not loaded before."""
    frame = Frame(tmp_rgb_jpeg)
    assert frame._array_cache is None
    frame.save_as(tmp_path / "out.tif")
    assert frame._array_cache is None


def test_save_as_conserves_loaded_cache(tmp_rgb_jpeg: Path, tmp_path: Path) -> None:
    """save_as() keeps the array cached when it was already loaded."""
    frame = Frame(tmp_rgb_jpeg)
    _ = frame.array
    assert frame._array_cache is not None
    frame.save_as(tmp_path / "out.tif")
    assert frame._array_cache is not None


def test_subtract_dark_conserves_unloaded_master_dark_cache(tmp_path: Path) -> None:
    """subtract_dark() does not leave the master dark array cached if unloaded."""
    from tests.utils import write_gray_tiff

    light = Frame(
        write_gray_tiff(tmp_path / "l.tif", np.full((24, 32), 100, dtype=np.uint16))
    )
    dark = Frame(
        write_gray_tiff(tmp_path / "r.tif", np.full((24, 32), 10, dtype=np.uint16))
    )
    assert dark._array_cache is None
    light.subtract_dark(dark)
    assert dark._array_cache is None


def test_subtract_dark_conserves_loaded_master_dark_cache(tmp_path: Path) -> None:
    """subtract_dark() keeps the master dark cached if it was already loaded."""
    from tests.utils import write_gray_tiff

    light = Frame(
        write_gray_tiff(tmp_path / "l.tif", np.full((24, 32), 100, dtype=np.uint16))
    )
    dark = Frame(
        write_gray_tiff(tmp_path / "r.tif", np.full((24, 32), 10, dtype=np.uint16))
    )
    _ = dark.array
    assert dark._array_cache is not None
    light.subtract_dark(dark)
    assert dark._array_cache is not None


def test_plot_conserves_unloaded_cache(tmp_rgb_tiff: Path) -> None:
    """plot() does not leave the array cached when it was not loaded before."""
    frame = Frame(tmp_rgb_tiff)
    assert frame._array_cache is None
    frame.plot()
    assert frame._array_cache is None


def test_plot_conserves_loaded_cache(tmp_rgb_tiff: Path) -> None:
    """plot() keeps the array cached when it was already loaded."""
    frame = Frame(tmp_rgb_tiff)
    _ = frame.array
    assert frame._array_cache is not None
    frame.plot()
    assert frame._array_cache is not None


def test_repr_format(tmp_rgb_tiff: Path) -> None:
    """__repr__ includes the path and dtype."""
    frame = Frame(tmp_rgb_tiff)
    r = repr(frame)
    assert "Frame(" in r
    assert str(tmp_rgb_tiff) in r
    assert "dtype='uint16'" in r
