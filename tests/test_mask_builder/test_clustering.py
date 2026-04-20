"""Tests for mask clustering internals."""

import numpy as np
import pytest

from celestack.mask_builder._clustering import (
    ClusterWeights,
    extract_features,
    relabel_by_descending_y,
    run_kmeans,
)

# ===========================================================================
# ClusterWeights — defaults
# ===========================================================================


def test_cluster_weights_defaults() -> None:
    """Default weights are RGB=1, spatial=0."""
    w = ClusterWeights()
    assert w.r == 1.0
    assert w.g == 1.0
    assert w.b == 1.0
    assert w.x == 0.0
    assert w.y == 0.0


def test_cluster_weights_equality() -> None:
    """Two ClusterWeights with the same values compare equal."""
    assert ClusterWeights(r=1.0, g=0.5, b=0.0, x=2.0, y=0.0) == ClusterWeights(
        r=1.0, g=0.5, b=0.0, x=2.0, y=0.0
    )


def test_cluster_weights_inequality() -> None:
    """ClusterWeights with different values compare unequal."""
    assert ClusterWeights(r=1.0) != ClusterWeights(r=0.5)


# ===========================================================================
# extract_features — shape and standardization
# ===========================================================================


def test_extract_features_shape() -> None:
    """Output shape is (H*W, 5) for any RGB input."""
    h, w = 12, 16
    array = np.random.default_rng(0).integers(0, 256, (h, w, 3), dtype=np.uint8)
    features = extract_features(array)
    assert features.shape == (h * w, 5)


def test_extract_features_dtype() -> None:
    """Output dtype is float32."""
    array = np.zeros((4, 4, 3), dtype=np.uint8)
    features = extract_features(array)
    assert features.dtype == np.float32


def test_extract_features_rgb_zero_mean() -> None:
    """RGB columns have approximately zero mean after standardization."""
    array = np.random.default_rng(1).integers(0, 65536, (16, 20, 3), dtype=np.uint16)
    features = extract_features(array)
    rgb_means = features[:, :3].mean(axis=0)
    np.testing.assert_allclose(rgb_means, 0.0, atol=1e-5)


def test_extract_features_rgb_unit_variance() -> None:
    """RGB columns have unit std after standardization."""
    array = np.random.default_rng(2).integers(0, 256, (16, 20, 3), dtype=np.uint8)
    features = extract_features(array)
    rgb_stds = features[:, :3].std(axis=0)
    np.testing.assert_allclose(rgb_stds, 1.0, atol=1e-5)


def test_extract_features_uniform_channel_stays_zero() -> None:
    """A column with zero variance (uniform channel) is left at zero."""
    array = np.full((8, 10, 3), 128, dtype=np.uint8)
    features = extract_features(array)
    np.testing.assert_array_equal(features[:, :3], 0.0)


def test_extract_features_spatial_range() -> None:
    """Spatial columns span [-1, 1]."""
    array = np.zeros((8, 10, 3), dtype=np.uint8)
    features = extract_features(array)
    for col in (3, 4):
        assert features[:, col].min() == pytest.approx(-1.0, abs=1e-6)
        assert features[:, col].max() == pytest.approx(1.0, abs=1e-6)


def test_extract_features_column_count_is_five() -> None:
    """Always produces exactly 5 columns (R, G, B, X, Y)."""
    array = np.zeros((6, 8, 3), dtype=np.uint16)
    features = extract_features(array)
    assert features.shape[1] == 5


# ===========================================================================
# run_kmeans
# ===========================================================================


@pytest.mark.parametrize("n_clusters", [2, 3, 5])
def test_run_kmeans_label_count(n_clusters: int) -> None:
    """Number of unique labels equals n_clusters."""
    n = 16 * 20
    rng = np.random.default_rng(42)
    features = rng.random((n, 3)).astype(np.float32)
    labels = run_kmeans(features, n_clusters)
    assert len(np.unique(labels)) == n_clusters


def test_run_kmeans_shape() -> None:
    """Output is a flat array with shape (N,)."""
    n = 12 * 15
    features = np.random.default_rng(0).random((n, 3)).astype(np.float32)
    labels = run_kmeans(features, 3)
    assert labels.shape == (n,)


def test_run_kmeans_dtype() -> None:
    """Output dtype is int32."""
    features = np.random.default_rng(0).random((20, 3)).astype(np.float32)
    labels = run_kmeans(features, 2)
    assert labels.dtype == np.int32


def test_run_kmeans_reproducible() -> None:
    """Same random_state produces identical labels."""
    features = np.random.default_rng(7).random((50, 3)).astype(np.float32)
    labels_a = run_kmeans(features, 3, random_state=99)
    labels_b = run_kmeans(features, 3, random_state=99)
    np.testing.assert_array_equal(labels_a, labels_b)


def test_run_kmeans_different_seeds_may_differ() -> None:
    """Different random states can produce different label assignments."""
    features = np.random.default_rng(5).random((200, 3)).astype(np.float32)
    labels_a = run_kmeans(features, 4, random_state=0)
    labels_b = run_kmeans(features, 4, random_state=999)
    assert labels_a.shape == labels_b.shape
    assert labels_a.dtype == labels_b.dtype


# ===========================================================================
# relabel_by_descending_y
# ===========================================================================


def test_relabel_by_descending_y_bottom_cluster_is_zero() -> None:
    """Cluster occupying the bottom rows gets label 0."""
    # Top half = old label 0, bottom half = old label 1.
    labels = np.zeros((8, 10), dtype=np.int32)
    labels[4:, :] = 1
    result = relabel_by_descending_y(labels, 2)
    # Bottom cluster (old 1) should become 0.
    assert result[7, 0] == 0
    assert result[0, 0] == 1


def test_relabel_by_descending_y_preserves_shape_and_dtype() -> None:
    """Output has the same shape and dtype as input."""
    labels = np.zeros((6, 8), dtype=np.int32)
    labels[3:, :] = 1
    result = relabel_by_descending_y(labels, 2)
    assert result.shape == labels.shape
    assert result.dtype == labels.dtype


def test_relabel_by_descending_y_three_clusters() -> None:
    """Three horizontal bands are relabeled bottom-to-top as 0, 1, 2."""
    labels = np.zeros((9, 6), dtype=np.int32)
    labels[3:6, :] = 1
    labels[6:, :] = 2
    result = relabel_by_descending_y(labels, 3)
    # Bottom band (old 2) → 0, middle (old 1) → 1, top (old 0) → 2.
    assert result[8, 0] == 0
    assert result[4, 0] == 1
    assert result[0, 0] == 2
