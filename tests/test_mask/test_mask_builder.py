"""Tests for the MaskBuilder class (src/celestack/mask/_mask_builder.py)."""

from pathlib import Path

import numpy as np
import pytest

from celestack.exceptions import MaskError
from celestack.frame.core import Frame
from celestack.mask._clustering import ClusterWeights
from celestack.mask._mask_builder import MaskBuilder
from celestack.mask.core import Mask


# ===========================================================================
# MaskBuilder — constructor
# ===========================================================================


def test_init_accepts_rgb_frame(rgb_frame: Frame) -> None:
    """MaskBuilder accepts a valid 3-channel RGB Frame."""
    mb = MaskBuilder(rgb_frame)
    assert mb._height == rgb_frame.shape[0]
    assert mb._width == rgb_frame.shape[1]


def test_init_creates_proxy(rgb_frame: Frame) -> None:
    """MaskBuilder creates a downscaled proxy on construction."""
    mb = MaskBuilder(rgb_frame, proxy_downscale_factor=2)
    assert mb._proxy is not None
    assert mb._proxy_height == rgb_frame.shape[0] // 2
    assert mb._proxy_width == rgb_frame.shape[1] // 2


def test_init_proxy_is_rgb(rgb_frame: Frame) -> None:
    """The proxy retains 3 channels (not grayscale)."""
    mb = MaskBuilder(rgb_frame, proxy_downscale_factor=2)
    assert len(mb._proxy.shape) == 3
    assert mb._proxy.shape[2] == 3


def test_init_rejects_grayscale(gray_mask_path: Path) -> None:
    """MaskBuilder rejects a grayscale Frame with MaskError."""
    frame = Frame(gray_mask_path)
    with pytest.raises(MaskError, match="3-channel RGB"):
        MaskBuilder(frame)


# ===========================================================================
# MaskBuilder — compute_clusters
# ===========================================================================


def test_compute_clusters_stores_labels(rgb_frame: Frame) -> None:
    """Labels array has proxy shape after clustering."""
    mb = MaskBuilder(rgb_frame, proxy_downscale_factor=2)
    mb.compute_clusters(3)
    assert mb._labels is not None
    expected_shape = (rgb_frame.shape[0] // 2, rgb_frame.shape[1] // 2)
    assert mb._labels.shape == expected_shape


def test_compute_clusters_rejects_k_less_than_2(rgb_frame: Frame) -> None:
    """compute_clusters raises ValueError for n_clusters < 2."""
    mb = MaskBuilder(rgb_frame)
    with pytest.raises(ValueError, match="at least 2"):
        mb.compute_clusters(1)


def test_compute_clusters_different_k(rgb_frame: Frame) -> None:
    """Re-clustering with different K updates n_clusters and produces new labels."""
    mb = MaskBuilder(rgb_frame, proxy_downscale_factor=2)
    mb.compute_clusters(2)
    assert mb._n_clusters == 2
    mb.compute_clusters(4)
    assert mb._n_clusters == 4
    assert mb._labels is not None
    assert mb._labels.shape == (rgb_frame.shape[0] // 2, rgb_frame.shape[1] // 2)


def test_compute_clusters_reuses_features_same_weights(rgb_frame: Frame) -> None:
    """Feature matrix is only computed once when weights are unchanged."""
    mb = MaskBuilder(rgb_frame)
    mb.compute_clusters(2)
    features_id = id(mb._features)
    mb.compute_clusters(3)
    assert id(mb._features) == features_id


def test_compute_clusters_recomputes_features_on_weight_change(
    rgb_frame: Frame,
) -> None:
    """Feature matrix is recomputed when weights change."""
    mb = MaskBuilder(rgb_frame)
    mb.compute_clusters(2, weights=ClusterWeights(r=1, g=1, b=1, x=0, y=0))
    features_id = id(mb._features)
    mb.compute_clusters(2, weights=ClusterWeights(r=1, g=1, b=1, x=1, y=1))
    assert id(mb._features) != features_id


def test_compute_clusters_accepts_custom_weights(rgb_frame: Frame) -> None:
    """compute_clusters succeeds with custom ClusterWeights."""
    mb = MaskBuilder(rgb_frame, proxy_downscale_factor=2)
    weights = ClusterWeights(r=1.0, g=0.5, b=2.0, x=0.0, y=0.0)
    mb.compute_clusters(2, weights=weights)
    assert mb._n_clusters == 2
    assert mb._current_weights == weights


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
    """apply_labels creates a boolean mask at proxy resolution."""
    mb = MaskBuilder(rgb_frame, proxy_downscale_factor=2)
    mb.compute_clusters(2)
    mb.apply_labels({0})
    assert mb.mask_array.dtype == np.bool_
    expected_shape = (rgb_frame.shape[0] // 2, rgb_frame.shape[1] // 2)
    assert mb.mask_array.shape == expected_shape


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
# MaskBuilder — mask_array property
# ===========================================================================


def test_mask_property_before_labels_raises(rgb_frame: Frame) -> None:
    """mask_array raises MaskError before any mask is created."""
    mb = MaskBuilder(rgb_frame)
    with pytest.raises(MaskError, match="No mask available"):
        _ = mb.mask_array


def test_mask_property_after_compute_no_apply_raises(rgb_frame: Frame) -> None:
    """mask_array raises after compute_clusters but before apply_labels."""
    mb = MaskBuilder(rgb_frame)
    mb.compute_clusters(2)
    with pytest.raises(MaskError):
        _ = mb.mask_array


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


def test_build_returns_full_res_shape(builder: MaskBuilder) -> None:
    """build() returns a Mask at full-resolution dimensions."""
    builder.apply_labels({0, 1})
    mask = builder.build()
    assert mask.shape == (builder._height, builder._width)


def test_build_is_detached(builder: MaskBuilder) -> None:
    """The Mask returned by build() has no backing path."""
    builder.apply_labels({0})
    with pytest.raises(AttributeError, match="not backed by a file"):
        _ = builder.build().path


def test_build_all_foreground_is_all_true(rgb_frame: Frame) -> None:
    """build() with all clusters as foreground produces an all-True mask."""
    mb = MaskBuilder(rgb_frame, proxy_downscale_factor=2)
    mb.compute_clusters(2)
    mb.apply_labels({0, 1})
    mask = mb.build()
    assert mask.array.all()


def test_build_no_foreground_is_all_false(rgb_frame: Frame) -> None:
    """build() with no foreground clusters produces an all-False mask."""
    mb = MaskBuilder(rgb_frame, proxy_downscale_factor=2)
    mb.compute_clusters(2)
    mb.apply_labels(set())
    mask = mb.build()
    assert not mask.array.any()
