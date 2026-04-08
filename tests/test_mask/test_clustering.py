"""Tests for mask clustering internals."""

import numpy as np
import pytest

from celestack.mask._clustering import (
    ClusterWeights,
    _normalize_rgb,
    extract_features,
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
# _normalize_rgb
# ===========================================================================


@pytest.mark.parametrize("bit_depth", [8, 16])
def test_normalize_rgb_shape(bit_depth: int) -> None:
    """Output shape is (H*W, 3)."""
    h, w = 8, 10
    array = np.zeros((h, w, 3), dtype=np.uint8)
    result = _normalize_rgb(array, bit_depth)
    assert result.shape == (h * w, 3)


def test_normalize_rgb_range_uint8() -> None:
    """Normalized values are in [0, 1] for uint8 input."""
    array = np.random.default_rng(0).integers(0, 256, (8, 10, 3), dtype=np.uint8)
    result = _normalize_rgb(array, 8)
    assert float(result.min()) >= 0.0
    assert float(result.max()) <= 1.0 + 1e-6


def test_normalize_rgb_float_input() -> None:
    """Float arrays treat 1.0 as the max."""
    array = np.ones((4, 4, 3), dtype=np.float32) * 0.5
    result = _normalize_rgb(array, 32)
    assert np.allclose(result, 0.5, atol=1e-6)


# ===========================================================================
# extract_features — shape and content
# ===========================================================================


@pytest.mark.parametrize("bit_depth", [8, 16])
def test_extract_features_rgb_only_shape(bit_depth: int) -> None:
    """Default weights (RGB only) produce shape (H*W, 3)."""
    h, w = 12, 16
    max_val = (1 << bit_depth) - 1
    array = np.random.default_rng(0).integers(0, max_val, (h, w, 3), dtype=np.uint16)
    features = extract_features(array, bit_depth)
    assert features.shape == (h * w, 3)


def test_extract_features_all_five_shape() -> None:
    """All-nonzero weights produce shape (H*W, 5)."""
    h, w = 8, 10
    array = np.zeros((h, w, 3), dtype=np.uint8)
    weights = ClusterWeights(r=1.0, g=1.0, b=1.0, x=1.0, y=1.0)
    features = extract_features(array, 8, weights)
    assert features.shape == (h * w, 5)


@pytest.mark.parametrize(
    "weights, expected_cols",
    [
        (ClusterWeights(r=1, g=0, b=0, x=0, y=0), 1),
        (ClusterWeights(r=1, g=1, b=0, x=0, y=0), 2),
        (ClusterWeights(r=1, g=1, b=1, x=1, y=0), 4),
        (ClusterWeights(r=0, g=0, b=0, x=1, y=1), 2),
    ],
)
def test_extract_features_column_count(
    weights: ClusterWeights, expected_cols: int
) -> None:
    """Number of feature columns equals the number of non-zero weights."""
    array = np.zeros((6, 8, 3), dtype=np.uint8)
    features = extract_features(array, 8, weights)
    assert features.shape[1] == expected_cols


def test_extract_features_all_zero_weights_raises() -> None:
    """All-zero weights raise ValueError."""
    array = np.zeros((4, 4, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="non-zero"):
        extract_features(array, 8, ClusterWeights(r=0, g=0, b=0, x=0, y=0))


@pytest.mark.parametrize("bit_depth", [8, 16])
def test_extract_features_normalized_range(bit_depth: int) -> None:
    """All feature values are in [0, max_weight] for default weights."""
    h, w = 8, 10
    max_val = (1 << bit_depth) - 1
    array = np.random.default_rng(1).integers(
        0, max_val + 1, (h, w, 3), dtype=np.uint16
    )
    features = extract_features(array, bit_depth)
    assert float(features.min()) >= 0.0
    assert float(features.max()) <= 1.0 + 1e-5


def test_extract_features_float_input() -> None:
    """Float arrays are normalized assuming values already in [0, 1]."""
    h, w = 4, 4
    array = np.ones((h, w, 3), dtype=np.float32) * 0.5
    features = extract_features(array, bit_depth=32)
    assert np.allclose(features, 0.5, atol=1e-6)


def test_extract_features_spatial_corners() -> None:
    """Corner pixels have expected spatial coordinate values when x/y enabled."""
    h, w = 8, 10
    array = np.zeros((h, w, 3), dtype=np.uint8)
    weights = ClusterWeights(r=0, g=0, b=0, x=1.0, y=1.0)
    features = extract_features(array, bit_depth=8, weights=weights)
    # features columns: X, Y
    # pixel index (row=0, col=0) -> flat index 0 -> X=0.0, Y=0.0
    assert features[0, 0] == pytest.approx(0.0, abs=1e-6)
    assert features[0, 1] == pytest.approx(0.0, abs=1e-6)
    # pixel index (row=0, col=w-1) -> flat index w-1 -> X=1.0, Y=0.0
    assert features[w - 1, 0] == pytest.approx(1.0, abs=1e-6)
    assert features[w - 1, 1] == pytest.approx(0.0, abs=1e-6)
    # pixel index (row=h-1, col=0) -> flat index (h-1)*w -> X=0.0, Y=1.0
    assert features[(h - 1) * w, 0] == pytest.approx(0.0, abs=1e-6)
    assert features[(h - 1) * w, 1] == pytest.approx(1.0, abs=1e-6)
    # pixel index (row=h-1, col=w-1) -> last index -> X=1.0, Y=1.0
    assert features[-1, 0] == pytest.approx(1.0, abs=1e-6)
    assert features[-1, 1] == pytest.approx(1.0, abs=1e-6)


def test_extract_features_dtype() -> None:
    """Output dtype is float32."""
    array = np.zeros((4, 4, 3), dtype=np.uint8)
    features = extract_features(array, bit_depth=8)
    assert features.dtype == np.float32


def test_extract_features_weight_scaling() -> None:
    """Feature values are scaled by their weight."""
    array = np.full((4, 4, 3), 128, dtype=np.uint8)
    features_unit = extract_features(
        array, 8, ClusterWeights(r=1.0, g=0, b=0, x=0, y=0)
    )
    features_scaled = extract_features(
        array, 8, ClusterWeights(r=2.0, g=0, b=0, x=0, y=0)
    )
    assert np.allclose(features_scaled, features_unit * 2.0, atol=1e-6)


# ===========================================================================
# run_kmeans
# ===========================================================================


@pytest.mark.parametrize("n_clusters", [2, 3, 5])
def test_run_kmeans_label_count(n_clusters: int) -> None:
    """Number of unique labels equals n_clusters."""
    h, w = 16, 20
    rng = np.random.default_rng(42)
    features = rng.random((h * w, 3)).astype(np.float32)
    labels = run_kmeans(features, n_clusters, h, w)
    assert len(np.unique(labels)) == n_clusters


def test_run_kmeans_shape() -> None:
    """Output shape matches (H, W)."""
    h, w = 12, 15
    features = np.random.default_rng(0).random((h * w, 3)).astype(np.float32)
    labels = run_kmeans(features, 3, h, w)
    assert labels.shape == (h, w)


def test_run_kmeans_dtype() -> None:
    """Output dtype is int32."""
    features = np.random.default_rng(0).random((20, 3)).astype(np.float32)
    labels = run_kmeans(features, 2, 4, 5)
    assert labels.dtype == np.int32


def test_run_kmeans_reproducible() -> None:
    """Same random_state produces identical labels."""
    features = np.random.default_rng(7).random((50, 3)).astype(np.float32)
    labels_a = run_kmeans(features, 3, 5, 10, random_state=99)
    labels_b = run_kmeans(features, 3, 5, 10, random_state=99)
    np.testing.assert_array_equal(labels_a, labels_b)


def test_run_kmeans_different_seeds_may_differ() -> None:
    """Different random states can produce different label assignments."""
    features = np.random.default_rng(5).random((200, 3)).astype(np.float32)
    labels_a = run_kmeans(features, 4, 10, 20, random_state=0)
    labels_b = run_kmeans(features, 4, 10, 20, random_state=999)
    assert labels_a.shape == labels_b.shape
    assert labels_a.dtype == labels_b.dtype


def test_run_kmeans_centroid_sort_ordering() -> None:
    """With rgb_norm, labels are assigned in ascending centroid (R, G, B) order."""
    # Two clearly separated clusters: left half black (R=0), right half white (R=1)
    h, w = 4, 8
    array = np.zeros((h, w, 3), dtype=np.uint8)
    array[:, w // 2 :] = 255  # right half is white
    rgb_norm = _normalize_rgb(array, 8)
    features = extract_features(array, 8, ClusterWeights(r=1, g=1, b=1, x=0, y=0))
    labels = run_kmeans(features, 2, h, w, rgb_norm=rgb_norm)
    # Cluster 0 must be the darker cluster (left half)
    left_label = labels[0, 0]
    right_label = labels[0, w - 1]
    assert left_label == 0
    assert right_label == 1


def test_run_kmeans_without_rgb_norm_no_sort() -> None:
    """Without rgb_norm, labels are not guaranteed to be centroid-sorted."""
    features = np.random.default_rng(3).random((30, 3)).astype(np.float32)
    labels = run_kmeans(features, 3, 5, 6)
    # Just verify shape and dtype — no ordering guarantee
    assert labels.shape == (5, 6)
    assert labels.dtype == np.int32
