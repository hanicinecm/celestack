"""Feature extraction and K-Means clustering for mask building."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans


@dataclass
class ClusterWeights:
    """Per-channel weights applied to normalized features before clustering.

    Set a weight to ``0`` to exclude that feature dimension entirely from the
    feature matrix passed to K-Means.  The default configuration uses only RGB
    channels (spatial coordinates excluded).

    Attributes:
        r: Weight for the normalized red channel.
        g: Weight for the normalized green channel.
        b: Weight for the normalized blue channel.
        x: Weight for the normalized horizontal spatial coordinate.
        y: Weight for the normalized vertical spatial coordinate.
    """

    r: float = 1.0
    g: float = 1.0
    b: float = 1.0
    x: float = 0.0
    y: float = 0.0


def _normalize_rgb(array: np.ndarray, bit_depth: int) -> np.ndarray:
    """Normalize the first 3 channels of an image to [0, 1].

    Args:
        array: Image array with shape (H, W, 3) or (H, W, C>=3).
        bit_depth: Source bit depth for integer normalization.

    Returns:
        Float32 array with shape (H*W, 3).
    """
    if np.issubdtype(array.dtype, np.floating):
        max_val = 1.0
    else:
        max_val = float((1 << bit_depth) - 1)
    return (array[..., :3].astype(np.float32) / max_val).reshape(-1, 3)


def extract_features(
    array: np.ndarray,
    bit_depth: int,
    weights: ClusterWeights | None = None,
) -> np.ndarray:
    """Build the feature matrix from an RGB image.

    Only features whose corresponding weight is non-zero are included.
    Normalizes RGB channels to [0, 1] based on bit depth and generates
    normalized spatial coordinates in [0, 1].  Each included feature column
    is scaled by its weight.

    Args:
        array: RGB image array with shape (H, W, 3).
        bit_depth: Source bit depth used for integer normalization.
        weights: Per-channel weights.  Defaults to ``ClusterWeights()``
            (RGB only, no spatial features).

    Returns:
        Feature matrix with shape (H*W, D) and dtype float32, where D is
        the number of non-zero weights (1 ≤ D ≤ 5).

    Raises:
        ValueError: If all weights are zero.
    """
    if weights is None:
        weights = ClusterWeights()

    height, width = array.shape[:2]
    rgb_norm = _normalize_rgb(array, bit_depth)  # (H*W, 3)

    columns: list[np.ndarray] = []

    if weights.r != 0.0:
        columns.append(rgb_norm[:, 0] * weights.r)
    if weights.g != 0.0:
        columns.append(rgb_norm[:, 1] * weights.g)
    if weights.b != 0.0:
        columns.append(rgb_norm[:, 2] * weights.b)

    if weights.x != 0.0 or weights.y != 0.0:
        x_coords = np.linspace(0.0, 1.0, width, dtype=np.float32)
        y_coords = np.linspace(0.0, 1.0, height, dtype=np.float32)
        xx, yy = np.meshgrid(x_coords, y_coords)
        if weights.x != 0.0:
            columns.append(xx.ravel() * weights.x)
        if weights.y != 0.0:
            columns.append(yy.ravel() * weights.y)

    if not columns:
        msg = "At least one weight must be non-zero"
        raise ValueError(msg)

    return np.column_stack(columns).astype(np.float32)


def run_kmeans(
    features: np.ndarray,
    n_clusters: int,
    height: int,
    width: int,
    *,
    rgb_norm: np.ndarray | None = None,
    random_state: int = 42,
) -> np.ndarray:
    """Run K-Means clustering and return labels reshaped to image dimensions.

    When ``rgb_norm`` is supplied, cluster labels are remapped so that cluster
    IDs are assigned in ascending order of mean (R, G, B) centroid — making
    the ordering deterministic and independent of K-Means initialization.

    Args:
        features: Feature matrix with shape (H*W, D).
        n_clusters: Number of clusters.
        height: Image height for reshaping labels.
        width: Image width for reshaping labels.
        rgb_norm: Normalized RGB values with shape (H*W, 3) used for
            centroid-based label sorting.  Pass ``None`` to skip sorting.
        random_state: Seed for reproducibility.

    Returns:
        Label array with shape (H, W) and dtype int32.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    raw_labels = kmeans.fit_predict(features)

    if rgb_norm is not None:
        sort_order = _sort_by_rgb_centroid(raw_labels, rgb_norm, n_clusters)
        remap = np.empty(n_clusters, dtype=np.int32)
        for new_id, old_id in enumerate(sort_order):
            remap[old_id] = new_id
        raw_labels = remap[raw_labels]

    return raw_labels.reshape(height, width).astype(np.int32)


def _sort_by_rgb_centroid(
    labels: np.ndarray,
    rgb_norm: np.ndarray,
    n_clusters: int,
) -> np.ndarray:
    """Return cluster indices sorted by mean (R, G, B) centroid.

    Primary sort key is R, secondary G, tertiary B.

    Args:
        labels: Flat label array with shape (H*W,).
        rgb_norm: Normalized RGB with shape (H*W, 3).
        n_clusters: Total number of clusters.

    Returns:
        Array of length ``n_clusters`` where ``result[i]`` is the original
        cluster index that should become cluster ``i`` after sorting.
    """
    centroids = np.zeros((n_clusters, 3), dtype=np.float32)
    for i in range(n_clusters):
        mask = labels == i
        if mask.any():
            centroids[i] = rgb_norm[mask].mean(axis=0)
    # np.lexsort: last key is primary — so (B, G, R) → primary=R
    return np.lexsort((centroids[:, 2], centroids[:, 1], centroids[:, 0]))
