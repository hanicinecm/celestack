"""Feature extraction and K-Means clustering for mask building."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans

from celestack.progress import progress_factory


@dataclass
class ClusterWeights:
    """Per-channel weights applied to standardized features before clustering.

    Set a weight to ``0`` to exclude that feature dimension entirely from the
    feature matrix passed to K-Means.  The default configuration uses only RGB
    channels (spatial coordinates excluded).

    Attributes:
        r: Weight for the red channel.
        g: Weight for the green channel.
        b: Weight for the blue channel.
        x: Weight for the horizontal spatial coordinate.
        y: Weight for the vertical spatial coordinate.
    """

    r: float = 1.0
    g: float = 1.0
    b: float = 1.0
    x: float = 0.0
    y: float = 0.0


def extract_features(array: np.ndarray) -> np.ndarray:
    """Build a standardized feature matrix from an RGB image.

    RGB channels are standardized to zero mean and unit variance (columns
    with zero variance are left at zero).  Spatial X, Y coordinates are
    normalized to [-1, 1].

    Args:
        array: RGB image array with shape (H, W, 3).

    Returns:
        Standardized feature matrix with shape (H*W, 5) and dtype float32.
        Column order matches ClusterWeights fields: (R, G, B, X, Y).
    """
    height, width = array.shape[:2]
    n_pixels = height * width

    # Raw RGB channels flattened.
    rgb = array[..., :3].reshape(n_pixels, 3).astype(np.float32)

    # Standardize RGB columns to zero mean and unit variance.
    rgb_mean = rgb.mean(axis=0)
    rgb_std = rgb.std(axis=0)
    rgb_std[rgb_std == 0] = 1.0  # avoid division by zero for uniform channels
    rgb = (rgb - rgb_mean) / rgb_std

    # Spatial coordinates normalized to [-1, 1].
    x_coords = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    y_coords = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    xx, yy = np.meshgrid(x_coords, y_coords)

    return np.column_stack([rgb, xx.ravel(), yy.ravel()]).astype(np.float32)


def run_kmeans(
    features: np.ndarray,
    n_clusters: int,
    *,
    random_state: int = 42,
) -> np.ndarray:
    """Run K-Means clustering and return flat label assignments.

    Tracked with a progress bar that is displayed for the duration of the fit.

    Args:
        features: Feature matrix with shape (N, D).
        n_clusters: Number of clusters.
        random_state: Seed for reproducibility.

    Returns:
        Flat label array with shape (N,) and dtype int32.
    """
    with progress_factory(total=None, description="Clustering pixels"):
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
        labels = kmeans.fit_predict(features).astype(np.int32)

    return labels


def relabel_by_descending_y(labels: np.ndarray, n_clusters: int) -> np.ndarray:
    """Relabel clusters so that cluster 0 is closest to the bottom of the image.

    Computes the mean row index for each cluster and remaps labels so that
    the cluster with the highest mean row (bottom of the image) becomes
    cluster 0, the next becomes 1, and so on.  Empty clusters are assigned
    the highest labels.

    Args:
        labels: 2D label array with shape (H, W) and integer dtype.
        n_clusters: Total number of clusters.

    Returns:
        Relabeled array with the same shape and dtype as the input.
    """
    height = labels.shape[0]
    row_grid = np.broadcast_to(
        np.arange(height, dtype=np.float32)[:, np.newaxis],
        labels.shape,
    )
    mean_y = np.empty(n_clusters, dtype=np.float32)
    for k in range(n_clusters):
        members = row_grid[labels == k]
        mean_y[k] = members.mean() if members.size > 0 else -np.inf

    sort_order = np.argsort(-mean_y)
    remap = np.empty(n_clusters, dtype=np.int32)
    for new_id, old_id in enumerate(sort_order):
        remap[old_id] = new_id

    return remap[labels]
