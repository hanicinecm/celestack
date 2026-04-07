"""Tests for the Mask class and MaskBuilder class."""

from pathlib import Path

import numpy as np
import pytest
import tifffile

from celestack.exceptions import DownscaleError, MaskError
from celestack.frame.core import Frame
from celestack.mask._mask import Mask
from celestack.mask.core import MaskBuilder

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
    """RGB mask: top half (value 200) → True, bottom half (0) → False."""
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
    # row 5, col 8 is in the nonzero region → True → 255
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
    # 4x4 boolean mask; downscale by 2 → 2x2 output
    # Top-left 2x2: [T,T,T,F] → 75% → True
    # Top-right 2x2: [F,F,F,F] → 0% → False
    # Bottom-left 2x2: [T,F,F,F] → 25% → False
    # Bottom-right 2x2: [T,T,F,T] → 75% → True
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
    # 2x4 mask; downscale by 2 → 1x2 output
    # Left 2x2: [T,F,T,F] → 50% → False
    # Right 2x2: [T,T,T,F] → 75% → True
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


# ===========================================================================
# MaskBuilder — constructor
# ===========================================================================


def test_init_accepts_rgb_frame(rgb_frame: Frame) -> None:
    """MaskBuilder accepts a valid 3-channel RGB Frame."""
    mb = MaskBuilder(rgb_frame)
    assert mb._height == rgb_frame.shape[0]
    assert mb._width == rgb_frame.shape[1]


def test_init_rejects_grayscale(gray_mask_path: Path) -> None:
    """MaskBuilder rejects a grayscale Frame with MaskError."""
    frame = Frame(gray_mask_path)
    with pytest.raises(MaskError, match="3-channel RGB"):
        MaskBuilder(frame)


# ===========================================================================
# MaskBuilder — compute_clusters
# ===========================================================================


def test_compute_clusters_stores_labels(rgb_frame: Frame) -> None:
    """Labels array has correct shape after clustering."""
    mb = MaskBuilder(rgb_frame)
    mb.compute_clusters(3)
    assert mb._labels is not None
    assert mb._labels.shape == rgb_frame.shape[:2]


def test_compute_clusters_rejects_k_less_than_2(rgb_frame: Frame) -> None:
    """compute_clusters raises ValueError for n_clusters < 2."""
    mb = MaskBuilder(rgb_frame)
    with pytest.raises(ValueError, match="at least 2"):
        mb.compute_clusters(1)


def test_compute_clusters_different_k(rgb_frame: Frame) -> None:
    """Re-clustering with different K updates n_clusters."""
    mb = MaskBuilder(rgb_frame)
    mb.compute_clusters(2)
    assert mb._n_clusters == 2
    mb.compute_clusters(4)
    assert mb._n_clusters == 4
    assert mb._labels is not None
    assert len(set(mb._labels.flatten())) == 4


def test_compute_clusters_reuses_features(rgb_frame: Frame) -> None:
    """Feature matrix is only computed once across multiple compute_clusters calls."""
    mb = MaskBuilder(rgb_frame)
    mb.compute_clusters(2)
    features_id = id(mb._features)
    mb.compute_clusters(3)
    assert id(mb._features) == features_id


def test_compute_clusters_resets_mask(rgb_frame: Frame) -> None:
    """Re-clustering resets mask and foreground_clusters to None."""
    mb = MaskBuilder(rgb_frame)
    mb.compute_clusters(2)
    mb.apply_labels({0})
    assert mb._mask is not None
    mb.compute_clusters(3)
    assert mb._mask is None
    assert mb._foreground_clusters is None


# ===========================================================================
# MaskBuilder — apply_labels
# ===========================================================================


def test_apply_labels_creates_mask(rgb_frame: Frame) -> None:
    """apply_labels creates a boolean mask with correct shape."""
    mb = MaskBuilder(rgb_frame)
    mb.compute_clusters(2)
    mb.apply_labels({0})
    assert mb.mask_array.dtype == np.bool_
    assert mb.mask_array.shape == rgb_frame.shape[:2]


def test_apply_labels_before_clustering_raises(rgb_frame: Frame) -> None:
    """apply_labels raises MaskError when compute_clusters not called."""
    mb = MaskBuilder(rgb_frame)
    with pytest.raises(MaskError, match="compute_clusters"):
        mb.apply_labels({0})


def test_apply_labels_invalid_cluster_raises(rgb_frame: Frame) -> None:
    """apply_labels raises ValueError for out-of-range cluster index."""
    mb = MaskBuilder(rgb_frame)
    mb.compute_clusters(2)
    with pytest.raises(ValueError, match="out of range"):
        mb.apply_labels({5})


def test_apply_labels_empty_set(rgb_frame: Frame) -> None:
    """apply_labels with empty set produces all-False mask."""
    mb = MaskBuilder(rgb_frame)
    mb.compute_clusters(2)
    mb.apply_labels(set())
    assert not mb.mask_array.any()


def test_apply_labels_all_clusters(rgb_frame: Frame) -> None:
    """apply_labels with all clusters produces all-True mask."""
    mb = MaskBuilder(rgb_frame)
    mb.compute_clusters(2)
    mb.apply_labels({0, 1})
    assert mb.mask_array.all()


# ===========================================================================
# MaskBuilder — mask property
# ===========================================================================


def test_mask_property_before_labels_raises(rgb_frame: Frame) -> None:
    """mask property raises MaskError before any mask is created."""
    mb = MaskBuilder(rgb_frame)
    with pytest.raises(MaskError, match="No mask available"):
        _ = mb.mask_array


def test_mask_property_after_compute_no_apply_raises(rgb_frame: Frame) -> None:
    """mask property raises after compute_clusters but before apply_labels."""
    mb = MaskBuilder(rgb_frame)
    mb.compute_clusters(2)
    with pytest.raises(MaskError):
        _ = mb.mask_array


# ===========================================================================
# MaskBuilder — mask_rectangle
# ===========================================================================


def test_mask_rectangle_sets_region(builder: MaskBuilder) -> None:
    """mask_rectangle correctly sets a rectangular region."""
    builder.apply_labels(set())
    builder.mask_rectangle(0, 0, 8, 6, foreground=True)
    assert builder.mask_array[0, 0]
    assert builder.mask_array[5, 7]
    assert not builder.mask_array[6, 0]


def test_mask_rectangle_foreground_false(builder: MaskBuilder) -> None:
    """mask_rectangle can clear a region to background."""
    builder.apply_labels({0, 1})
    builder.mask_rectangle(0, 0, 4, 4, foreground=False)
    assert not builder.mask_array[0, 0]
    assert not builder.mask_array[3, 3]


def test_mask_rectangle_before_mask_raises(rgb_frame: Frame) -> None:
    """mask_rectangle raises MaskError when no mask exists."""
    mb = MaskBuilder(rgb_frame)
    with pytest.raises(MaskError, match="mask must exist"):
        mb.mask_rectangle(0, 0, 4, 4, foreground=True)


def test_mask_rectangle_inverted_bounds_raises(builder: MaskBuilder) -> None:
    """mask_rectangle raises ValueError for x1 <= x0."""
    builder.apply_labels(set())
    with pytest.raises(ValueError, match="x1 must be greater"):
        builder.mask_rectangle(8, 0, 4, 4, foreground=True)


def test_mask_rectangle_out_of_bounds_raises(builder: MaskBuilder) -> None:
    """mask_rectangle raises ValueError when rectangle exceeds image."""
    builder.apply_labels(set())
    with pytest.raises(ValueError, match="outside image bounds"):
        builder.mask_rectangle(0, 0, 100, 100, foreground=True)


def test_mask_rectangle_negative_origin_raises(builder: MaskBuilder) -> None:
    """mask_rectangle raises ValueError for negative coordinates."""
    builder.apply_labels(set())
    with pytest.raises(ValueError, match="non-negative"):
        builder.mask_rectangle(-1, 0, 4, 4, foreground=True)


# ===========================================================================
# MaskBuilder — mask_pixels
# ===========================================================================


def test_mask_pixels_sets_values(builder: MaskBuilder) -> None:
    """mask_pixels correctly sets specific pixels."""
    builder.apply_labels(set())
    pixels = np.array([[2, 3], [10, 5]])  # (x, y)
    builder.mask_pixels(pixels, foreground=True)
    assert builder.mask_array[3, 2]
    assert builder.mask_array[5, 10]
    assert not builder.mask_array[0, 0]


def test_mask_pixels_before_mask_raises(rgb_frame: Frame) -> None:
    """mask_pixels raises MaskError when no mask exists."""
    mb = MaskBuilder(rgb_frame)
    with pytest.raises(MaskError, match="mask must exist"):
        mb.mask_pixels(np.array([[0, 0]]), foreground=True)


def test_mask_pixels_wrong_shape_raises(builder: MaskBuilder) -> None:
    """mask_pixels raises ValueError for non-(N,2) input."""
    builder.apply_labels(set())
    with pytest.raises(ValueError, match=r"\(N, 2\)"):
        builder.mask_pixels(np.array([0, 1, 2]), foreground=True)


def test_mask_pixels_out_of_bounds_raises(builder: MaskBuilder) -> None:
    """mask_pixels raises ValueError for out-of-bounds coordinates."""
    builder.apply_labels(set())
    with pytest.raises(ValueError, match="out of image bounds"):
        builder.mask_pixels(np.array([[9999, 0]]), foreground=True)


# ===========================================================================
# MaskBuilder — edit preservation across re-clustering
# ===========================================================================


def test_edits_preserved_across_reclustering(rgb_frame: Frame) -> None:
    """Manual edits are replayed after re-clustering and re-labeling."""
    mb = MaskBuilder(rgb_frame)
    mb.compute_clusters(2)
    mb.apply_labels(set())
    mb.mask_rectangle(0, 0, 4, 4, foreground=True)

    mb.compute_clusters(2)
    mb.apply_labels(set())

    assert mb.mask_array[0, 0]
    assert mb.mask_array[3, 3]
    assert not mb.mask_array[5, 0]


def test_edits_not_reset_by_compute_clusters(rgb_frame: Frame) -> None:
    """compute_clusters does not clear the edit log."""
    mb = MaskBuilder(rgb_frame)
    mb.compute_clusters(2)
    mb.apply_labels(set())
    mb.mask_rectangle(0, 0, 4, 4, foreground=True)

    mb.compute_clusters(3)
    assert len(mb._edits) == 1


def test_multiple_edits_replayed_in_order(rgb_frame: Frame) -> None:
    """Multiple edits are replayed in recorded order."""
    mb = MaskBuilder(rgb_frame)
    mb.compute_clusters(2)
    mb.apply_labels(set())
    mb.mask_rectangle(0, 0, 16, 12, foreground=True)
    mb.mask_rectangle(0, 0, 4, 4, foreground=False)

    mb.compute_clusters(2)
    mb.apply_labels(set())

    assert not mb.mask_array[0, 0]
    assert mb.mask_array[0, 5]


# ===========================================================================
# MaskBuilder — build
# ===========================================================================


def test_build_returns_mask(builder: MaskBuilder) -> None:
    """build() returns a Mask instance."""
    builder.apply_labels({0})
    result = builder.build()
    assert isinstance(result, Mask)


def test_build_before_labels_raises(builder: MaskBuilder) -> None:
    """build() raises MaskError when no mask has been created yet."""
    with pytest.raises(MaskError):
        builder.build()


def test_build_array_matches(builder: MaskBuilder) -> None:
    """build() array matches the internal mask numpy array."""
    builder.apply_labels({0, 1})
    mask = builder.build()
    np.testing.assert_array_equal(mask.array, builder.mask_array)


def test_build_is_independent_copy(builder: MaskBuilder) -> None:
    """Modifying the builder after build() does not affect the returned Mask."""
    builder.apply_labels({0, 1})
    mask = builder.build()
    original = mask.array.copy()
    builder.mask_rectangle(0, 0, 4, 4, foreground=False)
    np.testing.assert_array_equal(mask.array, original)


def test_build_is_detached(builder: MaskBuilder) -> None:
    """The Mask returned by build() has no backing path."""
    builder.apply_labels({0})
    with pytest.raises(AttributeError, match="not backed by a file"):
        _ = builder.build().path
