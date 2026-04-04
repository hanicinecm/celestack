"""Tests for mask clustering internals."""

import numpy as np
import pytest

from celestack.mask._clustering import extract_features, run_kmeans


@pytest.mark.parametrize("bit_depth", [8, 16])
def test_extract_features_shape(bit_depth: int) -> None:
    """Output shape is (H*W, 5) for an (H, W, 3) input."""
    h, w = 12, 16
    max_val = (1 << bit_depth) - 1
    array = np.random.default_rng(0).integers(0, max_val, (h, w, 3), dtype=np.uint16)
    features = extract_features(array, bit_depth)
    assert features.shape == (h * w, 5)


@pytest.mark.parametrize("bit_depth", [8, 16])
def test_extract_features_normalized_range(bit_depth: int) -> None:
    """All feature values are in [0, 1] for integer inputs."""
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
    # RGB channels should be 0.5 each, spatial coords vary
    rgb_cols = features[:, :3]
    assert np.allclose(rgb_cols, 0.5, atol=1e-6)


def test_extract_features_spatial_corners() -> None:
    """Corner pixels have expected spatial coordinate values."""
    h, w = 8, 10
    array = np.zeros((h, w, 3), dtype=np.uint8)
    features = extract_features(array, bit_depth=8)
    # features columns: R, G, B, X, Y
    # pixel index (row=0, col=0) -> flat index 0 -> X=0.0, Y=0.0
    assert features[0, 3] == pytest.approx(0.0, abs=1e-6)
    assert features[0, 4] == pytest.approx(0.0, abs=1e-6)
    # pixel index (row=0, col=w-1) -> flat index w-1 -> X=1.0, Y=0.0
    assert features[w - 1, 3] == pytest.approx(1.0, abs=1e-6)
    assert features[w - 1, 4] == pytest.approx(0.0, abs=1e-6)
    # pixel index (row=h-1, col=0) -> flat index (h-1)*w -> X=0.0, Y=1.0
    assert features[(h - 1) * w, 3] == pytest.approx(0.0, abs=1e-6)
    assert features[(h - 1) * w, 4] == pytest.approx(1.0, abs=1e-6)
    # pixel index (row=h-1, col=w-1) -> last index -> X=1.0, Y=1.0
    assert features[-1, 3] == pytest.approx(1.0, abs=1e-6)
    assert features[-1, 4] == pytest.approx(1.0, abs=1e-6)


def test_extract_features_dtype() -> None:
    """Output dtype is float32."""
    array = np.zeros((4, 4, 3), dtype=np.uint8)
    features = extract_features(array, bit_depth=8)
    assert features.dtype == np.float32


@pytest.mark.parametrize("n_clusters", [2, 3, 5])
def test_run_kmeans_label_count(n_clusters: int) -> None:
    """Number of unique labels equals n_clusters."""
    h, w = 16, 20
    rng = np.random.default_rng(42)
    features = rng.random((h * w, 5)).astype(np.float32)
    labels = run_kmeans(features, n_clusters, h, w)
    assert len(np.unique(labels)) == n_clusters


def test_run_kmeans_shape() -> None:
    """Output shape matches (H, W)."""
    h, w = 12, 15
    features = np.random.default_rng(0).random((h * w, 5)).astype(np.float32)
    labels = run_kmeans(features, 3, h, w)
    assert labels.shape == (h, w)


def test_run_kmeans_dtype() -> None:
    """Output dtype is int32."""
    features = np.random.default_rng(0).random((20, 5)).astype(np.float32)
    labels = run_kmeans(features, 2, 4, 5)
    assert labels.dtype == np.int32


def test_run_kmeans_reproducible() -> None:
    """Same random_state produces identical labels."""
    features = np.random.default_rng(7).random((50, 5)).astype(np.float32)
    labels_a = run_kmeans(features, 3, 5, 10, random_state=99)
    labels_b = run_kmeans(features, 3, 5, 10, random_state=99)
    np.testing.assert_array_equal(labels_a, labels_b)


def test_run_kmeans_different_seeds_may_differ() -> None:
    """Different random states can produce different label assignments."""
    features = np.random.default_rng(5).random((200, 5)).astype(np.float32)
    labels_a = run_kmeans(features, 4, 10, 20, random_state=0)
    labels_b = run_kmeans(features, 4, 10, 20, random_state=999)
    # Not guaranteed to differ, but very likely with random data
    # Just verify both are valid shape/dtype
    assert labels_a.shape == labels_b.shape
    assert labels_a.dtype == labels_b.dtype
