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


def test_unsupported_bit_depth_raises(tmp_path: Path) -> None:
    """Frame raises ValueError for images with unsupported bit depth."""
    import tifffile

    path = tmp_path / "deep.tif"
    tifffile.imwrite(path, np.zeros((4, 4), dtype=np.uint32))
    with pytest.raises(ValueError, match="Unsupported bit depth"):
        Frame(path)


def test_path_raises_for_detached_frame(tmp_rgb_tiff: Path, tmp_path: Path) -> None:
    """path property raises AttributeError for a detached (in-memory) frame."""
    frame = Frame(tmp_rgb_tiff)
    detached = frame.subtract(Frame(tmp_rgb_tiff))
    with pytest.raises(AttributeError, match="not backed by a file"):
        _ = detached.path


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


# --- save_as ---


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


def test_save_as_does_not_embed_celestack_metadata(
    tmp_rgb_tiff: Path, tmp_path: Path
) -> None:
    """Saving a full-res frame produces no Celestack metadata."""
    frame = Frame(tmp_rgb_tiff)
    dest = tmp_path / "fullres.tif"
    frame.save_as(dest)
    reloaded = Frame(dest)
    assert reloaded.downscale_factor == 1


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


def test_save_as_embeds_celestack_metadata_for_proxy(
    tmp_rgb_tiff: Path, tmp_path: Path
) -> None:
    """Saving a proxy (downscale_factor > 1) embeds Celestack metadata."""
    proxy = Frame(tmp_rgb_tiff).downscaled_copy(4)
    dest = tmp_path / "proxy.tif"
    proxy.save_as(dest)
    reloaded = Frame(dest)
    assert reloaded.downscale_factor == 4


def test_save_as_proxy_without_timestamp_omits_timestamp(
    tmp_no_metadata_tiff: Path, tmp_path: Path
) -> None:
    """Proxy saved without a timestamp has None timestamp after reload."""
    proxy = Frame(tmp_no_metadata_tiff).downscaled_copy(2)
    assert proxy.timestamp is None
    dest = tmp_path / "proxy.tif"
    proxy.save_as(dest)
    reloaded = Frame(dest)
    assert reloaded.timestamp is None
    assert reloaded.downscale_factor == 2


# --- Subtraction ---


def test_subtract_returns_detached_frame(tmp_gray_tiff: Path, tmp_path: Path) -> None:
    """subtract() returns a pathless in-memory frame."""
    dark_path = tmp_path / "dark.tif"
    dark_array = np.full((IMG_H, IMG_W), fill_value=10, dtype=np.uint16)
    from tests.utils import write_gray_tiff

    dark = Frame(write_gray_tiff(dark_path, dark_array))
    light = Frame(tmp_gray_tiff)

    result = light.subtract(dark)

    assert isinstance(result, Frame)
    assert result._path is None
    assert result.shape == light.shape
    assert result.bit_depth == light.bit_depth


def test_subtract_subtracts_pixels_and_clamps_unsigned(tmp_path: Path) -> None:
    """Unsigned subtraction is saturating at zero."""
    from tests.utils import write_gray_tiff

    left_path = write_gray_tiff(
        tmp_path / "left.tif",
        np.array([[20, 5], [100, 0]], dtype=np.uint16),
    )
    right_path = write_gray_tiff(
        tmp_path / "right.tif",
        np.array([[3, 9], [20, 1]], dtype=np.uint16),
    )

    result = Frame(left_path).subtract(Frame(right_path))

    np.testing.assert_array_equal(
        result.array,
        np.array([[17, 0], [80, 0]], dtype=np.uint16),
    )


def test_subtract_inherits_left_metadata_and_timestamp(
    tmp_rgb_tiff: Path, tmp_path: Path
) -> None:
    """Result inherits metadata and timestamp from the left frame."""
    from tests.utils import write_rgb_tiff

    right_path = write_rgb_tiff(
        tmp_path / "right.tif",
        np.zeros((IMG_H, IMG_W, 3), dtype=np.uint16),
    )
    left = Frame(tmp_rgb_tiff)
    left.timestamp = 123.0

    result = left.subtract(Frame(right_path))

    assert result.metadata.camera_make == left.metadata.camera_make
    assert result.metadata.camera_model == left.metadata.camera_model
    assert result.metadata.datetime == left.metadata.datetime
    assert result.metadata.exposure == left.metadata.exposure
    assert result.metadata.f_number == left.metadata.f_number
    assert result.metadata.iso == left.metadata.iso
    assert result.metadata.focal_length == left.metadata.focal_length
    assert result.metadata.lens_model == left.metadata.lens_model
    assert result.timestamp == pytest.approx(123.0)


def test_subtract_result_can_be_saved(tmp_gray_tiff: Path, tmp_path: Path) -> None:
    """A subtraction result can be persisted via save_as."""
    from tests.utils import write_gray_tiff

    dark_path = write_gray_tiff(
        tmp_path / "dark.tif",
        np.full((IMG_H, IMG_W), fill_value=1, dtype=np.uint16),
    )
    result = Frame(tmp_gray_tiff).subtract(Frame(dark_path))
    expected = result.array.copy()

    dest = tmp_path / "subtracted.tif"
    result.save_as(dest)

    assert result._path == dest
    reloaded = Frame(dest)
    np.testing.assert_array_equal(reloaded.array, expected)


def test_subtract_inplace_modifies_self(tmp_path: Path) -> None:
    """inplace=True subtracts into self and returns None."""
    from tests.utils import write_gray_tiff

    left_path = write_gray_tiff(
        tmp_path / "left.tif",
        np.array([[20, 5], [100, 0]], dtype=np.uint16),
    )
    right_path = write_gray_tiff(
        tmp_path / "right.tif",
        np.array([[3, 9], [20, 1]], dtype=np.uint16),
    )
    left = Frame(left_path)
    result = left.subtract(Frame(right_path), inplace=True)

    assert result is None
    np.testing.assert_array_equal(
        left.array,
        np.array([[17, 0], [80, 0]], dtype=np.uint16),
    )


def test_subtract_inplace_detaches_from_path(tmp_path: Path) -> None:
    """inplace=True detaches the frame from its backing path."""
    from tests.utils import write_gray_tiff

    left_path = write_gray_tiff(
        tmp_path / "left.tif",
        np.full((24, 32), 100, dtype=np.uint16),
    )
    right_path = write_gray_tiff(
        tmp_path / "right.tif",
        np.full((24, 32), 10, dtype=np.uint16),
    )
    left = Frame(left_path)
    left.subtract(Frame(right_path), inplace=True)
    assert left._path is None


def test_subtract_shape_mismatch_raises(tmp_path: Path) -> None:
    """Frames with different shapes cannot be subtracted."""
    from tests.utils import write_gray_tiff

    left_path = write_gray_tiff(
        tmp_path / "left.tif",
        np.zeros((10, 10), dtype=np.uint16),
    )
    right_path = write_gray_tiff(
        tmp_path / "right.tif",
        np.zeros((10, 11), dtype=np.uint16),
    )

    with pytest.raises(ValueError, match="Shape mismatch"):
        Frame(left_path).subtract(Frame(right_path))


def test_subtract_downscale_factor_mismatch_raises(
    tmp_rgb_tiff: Path, tmp_path: Path
) -> None:
    """Frames with different downscale factors cannot be subtracted."""
    from tests.utils import write_rgb_tiff

    other_path = write_rgb_tiff(
        tmp_path / "other.tif",
        np.zeros((IMG_H, IMG_W, 3), dtype=np.uint16),
    )
    left = Frame(tmp_rgb_tiff).downscaled_copy(2)
    left_dest = tmp_path / "left_proxy.tif"
    left.save_as(left_dest)

    right = Frame(other_path)
    right.timestamp = 1.0
    right_proxy = right.downscaled_copy(4)
    right_dest = tmp_path / "right_proxy.tif"
    right_proxy.save_as(right_dest)

    with pytest.raises(ValueError, match="Downscale factor mismatch"):
        left.subtract(right_proxy)


def test_subtract_bit_depth_mismatch_raises(tmp_path: Path) -> None:
    """Frames with different bit depths cannot be subtracted."""
    from tests.utils import write_gray_tiff

    left_path = write_gray_tiff(
        tmp_path / "left.tif",
        np.zeros((10, 10), dtype=np.uint16),
    )
    right_path = write_gray_tiff(
        tmp_path / "right.tif",
        np.zeros((10, 10), dtype=np.uint8),
    )
    with pytest.raises(ValueError, match="Bit depth mismatch"):
        Frame(left_path).subtract(Frame(right_path))


def test_subtract_inplace_conserves_other_cache(tmp_path: Path) -> None:
    """inplace=True conserves the cache state of the other operand."""
    from tests.utils import write_gray_tiff

    left_path = write_gray_tiff(
        tmp_path / "left.tif", np.full((24, 32), 100, dtype=np.uint16)
    )
    right_path = write_gray_tiff(
        tmp_path / "right.tif", np.full((24, 32), 10, dtype=np.uint16)
    )
    left = Frame(left_path)
    right = Frame(right_path)
    assert right._array_cache is None
    left.subtract(right, inplace=True)
    assert right._array_cache is None

    # Also verify: if other was already loaded, it stays loaded.
    left2 = Frame(left_path)
    _ = right.array
    assert right._array_cache is not None
    left2.subtract(right, inplace=True)
    assert right._array_cache is not None


# --- downscaled_copy ---


def test_downscaled_copy_dimensions(tmp_rgb_tiff: Path) -> None:
    """downscale_factor=2 halves each spatial dimension."""
    proxy = Frame(tmp_rgb_tiff).downscaled_copy(downscale_factor=2)
    assert proxy.shape[0] == IMG_H // 2
    assert proxy.shape[1] == IMG_W // 2


def test_downscaled_copy_grayscale(tmp_rgb_tiff: Path) -> None:
    """RGB input with grayscale=True produces a 2D array."""
    proxy = Frame(tmp_rgb_tiff).downscaled_copy(downscale_factor=2)
    assert proxy.array.ndim == 2


def test_downscaled_copy_no_grayscale(tmp_rgb_tiff: Path) -> None:
    """grayscale=False preserves RGB channels."""
    proxy = Frame(tmp_rgb_tiff).downscaled_copy(downscale_factor=2, grayscale=False)
    assert proxy.array.ndim == 3
    assert proxy.array.shape[2] == 3


def test_downscaled_copy_bit_depth(tmp_rgb_tiff: Path) -> None:
    """16-bit input downscaled to 8-bit produces uint8 output."""
    frame = Frame(tmp_rgb_tiff)
    assert frame.bit_depth == 16
    proxy = frame.downscaled_copy(downscale_factor=2, bit_depth=8)
    assert proxy.bit_depth == 8
    assert proxy.array.dtype == np.uint8


def test_downscaled_copy_sets_downscale_factor(tmp_rgb_tiff: Path) -> None:
    """Returned frame has downscale_factor matching the requested value."""
    proxy = Frame(tmp_rgb_tiff).downscaled_copy(downscale_factor=4)
    assert proxy.downscale_factor == 4


def test_downscaled_copy_is_detached(tmp_rgb_tiff: Path) -> None:
    """downscaled_copy returns an in-memory detached frame."""
    proxy = Frame(tmp_rgb_tiff).downscaled_copy(downscale_factor=2)
    assert proxy._path is None


def test_downscaled_copy_from_proxy_raises(tmp_rgb_tiff: Path, tmp_path: Path) -> None:
    """Downscaling an already-downscaled frame raises DownscaleError."""
    proxy = Frame(tmp_rgb_tiff).downscaled_copy(downscale_factor=2)
    with pytest.raises(DownscaleError, match="Cannot downscale a proxy"):
        proxy.downscaled_copy(downscale_factor=2)


def test_downscaled_copy_inherits_timestamp(tmp_rgb_tiff: Path) -> None:
    """Downscaled copy inherits the source timestamp."""
    frame = Frame(tmp_rgb_tiff)
    source_ts = frame.timestamp
    proxy = frame.downscaled_copy(downscale_factor=2)
    assert proxy.timestamp == pytest.approx(source_ts)


def test_downscaled_copy_inherits_none_timestamp(
    tmp_no_metadata_tiff: Path,
) -> None:
    """Downscaled copy inherits None timestamp without raising."""
    frame = Frame(tmp_no_metadata_tiff)
    assert frame.timestamp is None
    proxy = frame.downscaled_copy(downscale_factor=2)
    assert proxy.timestamp is None


def test_downscaled_copy_preserves_exif_metadata(tmp_rgb_jpeg: Path) -> None:
    """EXIF metadata is inherited by the downscaled copy."""
    frame = Frame(tmp_rgb_jpeg)
    proxy = frame.downscaled_copy(downscale_factor=2)
    assert proxy.metadata.camera_make == CAMERA_MAKE
    assert proxy.metadata.camera_model == CAMERA_MODEL
    assert proxy.metadata.datetime is not None
    assert proxy.metadata.exposure is not None
    assert proxy.metadata.f_number is not None
    assert proxy.metadata.iso is not None
    assert proxy.metadata.focal_length is not None
    assert proxy.metadata.lens_model == LENS_MODEL


def test_downscaled_copy_rejects_factor_below_one(tmp_rgb_tiff: Path) -> None:
    """downscale_factor < 1 raises ValueError."""
    with pytest.raises(ValueError, match="downscale_factor must be >= 1"):
        Frame(tmp_rgb_tiff).downscaled_copy(downscale_factor=0)


def test_downscaled_copy_factor_one(tmp_rgb_tiff: Path) -> None:
    """factor=1 returns a detached copy with no spatial change."""
    frame = Frame(tmp_rgb_tiff)
    proxy = frame.downscaled_copy(downscale_factor=1, bit_depth=8)
    assert proxy.downscale_factor == 1
    assert proxy.shape[0] == IMG_H
    assert proxy.shape[1] == IMG_W
    assert proxy.bit_depth == 8


def test_downscaled_copy_to_16bit(tmp_rgb_tiff: Path) -> None:
    """16-bit input downscaled to 16-bit preserves uint16 dtype."""
    proxy = Frame(tmp_rgb_tiff).downscaled_copy(downscale_factor=2, bit_depth=16)
    assert proxy.bit_depth == 16
    assert proxy.array.dtype == np.uint16


def test_downscaled_copy_grayscale_input(tmp_gray_tiff: Path) -> None:
    """Already-grayscale frame with grayscale=True stays 2D."""
    frame = Frame(tmp_gray_tiff)
    frame.timestamp = 1.0
    proxy = frame.downscaled_copy(downscale_factor=2, grayscale=True)
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


# --- Cache conservation ---


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


def test_downscaled_copy_conserves_unloaded_cache(tmp_rgb_tiff: Path) -> None:
    """downscaled_copy() does not leave the source array cached when unloaded."""
    frame = Frame(tmp_rgb_tiff)
    assert frame._array_cache is None
    frame.downscaled_copy(downscale_factor=2)
    assert frame._array_cache is None


def test_downscaled_copy_conserves_loaded_cache(tmp_rgb_tiff: Path) -> None:
    """downscaled_copy() keeps the source array cached when already loaded."""
    frame = Frame(tmp_rgb_tiff)
    _ = frame.array
    assert frame._array_cache is not None
    frame.downscaled_copy(downscale_factor=2)
    assert frame._array_cache is not None


def test_subtract_conserves_unloaded_cache(tmp_path: Path) -> None:
    """subtract() does not leave arrays cached on either operand."""
    from tests.utils import write_gray_tiff

    left = Frame(
        write_gray_tiff(tmp_path / "l.tif", np.full((24, 32), 100, dtype=np.uint16))
    )
    right = Frame(
        write_gray_tiff(tmp_path / "r.tif", np.full((24, 32), 10, dtype=np.uint16))
    )
    assert left._array_cache is None
    assert right._array_cache is None
    left.subtract(right)
    assert left._array_cache is None
    assert right._array_cache is None


def test_subtract_conserves_loaded_cache(tmp_path: Path) -> None:
    """subtract() keeps arrays cached on operands that were already loaded."""
    from tests.utils import write_gray_tiff

    left = Frame(
        write_gray_tiff(tmp_path / "l.tif", np.full((24, 32), 100, dtype=np.uint16))
    )
    right = Frame(
        write_gray_tiff(tmp_path / "r.tif", np.full((24, 32), 10, dtype=np.uint16))
    )
    _ = left.array
    assert left._array_cache is not None
    assert right._array_cache is None
    left.subtract(right)
    assert left._array_cache is not None
    assert right._array_cache is None


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


# --- Repr ---


def test_repr_format(tmp_rgb_tiff: Path) -> None:
    """__repr__ includes the path and downscale_factor."""
    frame = Frame(tmp_rgb_tiff)
    r = repr(frame)
    assert "Frame(" in r
    assert str(tmp_rgb_tiff) in r
    assert "downscale_factor=1" in r


def test_repr_detached(tmp_rgb_tiff: Path) -> None:
    """__repr__ shows <detached> for in-memory frames."""
    proxy = Frame(tmp_rgb_tiff).downscaled_copy(2)
    r = repr(proxy)
    assert "<detached>" in r
