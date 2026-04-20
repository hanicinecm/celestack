"""Tests for the MaskBuilder class (src/celestack/mask_builder/core.py)."""

from pathlib import Path

import numpy as np
import pytest

from celestack.exceptions import MaskError
from celestack.frame.core import Frame
from celestack.mask.core import Mask
from celestack.mask_builder._clustering import ClusterWeights
from celestack.mask_builder.core import MaskBuilder


# ===========================================================================
# MaskBuilder — constructor
# ===========================================================================


def test_init_accepts_rgb_frame(rgb_frame: Frame) -> None:
    """MaskBuilder accepts a valid 3-channel RGB Frame."""
    mb = MaskBuilder(rgb_frame)
    assert mb._height == rgb_frame.shape[0]
    assert mb._width == rgb_frame.shape[1]


def test_init_extracts_features(rgb_frame: Frame) -> None:
    """Standardized features are extracted at construction time."""
    mb = MaskBuilder(rgb_frame)
    h, w = rgb_frame.shape[0], rgb_frame.shape[1]
    assert mb._std_features.shape == (h * w, 5)


def test_init_rejects_grayscale(gray_mask_path: Path) -> None:
    """MaskBuilder rejects a grayscale Frame with MaskError."""
    frame = Frame(gray_mask_path)
    with pytest.raises(MaskError, match="3-channel RGB"):
        MaskBuilder(frame)


# ===========================================================================
# MaskBuilder — compute_clusters
# ===========================================================================


def test_compute_clusters_stores_labels(rgb_frame: Frame) -> None:
    """Labels array has image shape after clustering."""
    mb = MaskBuilder(rgb_frame)
    mb.compute_clusters(2)
    assert mb._labels is not None
    assert mb._labels.shape == (rgb_frame.shape[0], rgb_frame.shape[1])


def test_compute_clusters_rejects_k_less_than_2(rgb_frame: Frame) -> None:
    """compute_clusters raises ValueError for n_clusters < 2."""
    mb = MaskBuilder(rgb_frame)
    with pytest.raises(ValueError, match="at least 2"):
        mb.compute_clusters(1)


def test_compute_clusters_different_k(rgb_frame: Frame) -> None:
    """Re-clustering with different K updates n_clusters and produces new labels."""
    mb = MaskBuilder(rgb_frame)
    mb.compute_clusters(2)
    assert mb._n_clusters == 2
    mb.compute_clusters(3)
    assert mb._n_clusters == 3
    assert mb._labels is not None
    assert mb._labels.shape == (rgb_frame.shape[0], rgb_frame.shape[1])


def test_compute_clusters_accepts_custom_weights(rgb_frame: Frame) -> None:
    """compute_clusters succeeds with custom ClusterWeights."""
    mb = MaskBuilder(rgb_frame)
    weights = ClusterWeights(r=1.0, g=0.5, b=2.0, x=0.0, y=0.0)
    mb.compute_clusters(2, weights=weights)
    assert mb._n_clusters == 2


def test_compute_clusters_all_zero_weights_raises(rgb_frame: Frame) -> None:
    """compute_clusters raises ValueError when all weights are zero."""
    mb = MaskBuilder(rgb_frame)
    with pytest.raises(ValueError, match="non-zero"):
        mb.compute_clusters(2, weights=ClusterWeights(r=0, g=0, b=0, x=0, y=0))


def test_compute_clusters_labels_sorted_by_descending_y(rgb_frame: Frame) -> None:
    """Cluster 0 has the highest mean row index (bottom of image)."""
    mb = MaskBuilder(rgb_frame)
    mb.compute_clusters(2)
    labels = mb._labels
    assert labels is not None
    row_grid = np.broadcast_to(
        np.arange(labels.shape[0], dtype=np.float32)[:, np.newaxis],
        labels.shape,
    )
    mean_ys = [row_grid[labels == k].mean() for k in range(2)]
    # Cluster 0 should be at or below cluster 1.
    assert mean_ys[0] >= mean_ys[1]


# ===========================================================================
# MaskBuilder — build
# ===========================================================================


def test_build_returns_mask(rgb_frame: Frame) -> None:
    """build() returns a Mask instance."""
    mb = MaskBuilder(rgb_frame)
    mb.compute_clusters(2)
    result = mb.build({0})
    assert isinstance(result, Mask)


def test_build_before_clustering_raises(rgb_frame: Frame) -> None:
    """build() raises MaskError when compute_clusters not called."""
    mb = MaskBuilder(rgb_frame)
    with pytest.raises(MaskError, match="compute_clusters"):
        mb.build({0})


def test_build_returns_full_res_shape(rgb_frame: Frame) -> None:
    """build() returns a Mask at full-resolution dimensions."""
    mb = MaskBuilder(rgb_frame)
    mb.compute_clusters(2)
    mask = mb.build({0, 1})
    assert mask.shape == (rgb_frame.shape[0], rgb_frame.shape[1])


def test_build_is_detached(rgb_frame: Frame) -> None:
    """The Mask returned by build() has no backing path."""
    mb = MaskBuilder(rgb_frame)
    mb.compute_clusters(2)
    with pytest.raises(AttributeError, match="not backed by a file"):
        _ = mb.build({0}).path


def test_build_all_foreground_is_all_true(rgb_frame: Frame) -> None:
    """build() with all clusters as foreground produces an all-True mask."""
    mb = MaskBuilder(rgb_frame)
    mb.compute_clusters(2)
    mask = mb.build({0, 1})
    assert mask.array.all()


def test_build_no_foreground_is_all_false(rgb_frame: Frame) -> None:
    """build() with no foreground clusters produces an all-False mask."""
    mb = MaskBuilder(rgb_frame)
    mb.compute_clusters(2)
    mask = mb.build(set())
    assert not mask.array.any()


def test_build_invalid_cluster_raises(rgb_frame: Frame) -> None:
    """build() raises ValueError for out-of-range cluster index."""
    mb = MaskBuilder(rgb_frame)
    mb.compute_clusters(2)
    with pytest.raises(ValueError, match="out of range"):
        mb.build({5})
