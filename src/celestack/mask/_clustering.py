"""Feature extraction and K-Means clustering for mask building."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

# Per-channel weights applied to normalized features before clustering.
# All default to 1.0 (equal weighting). Adjust here for future calibration.
_RED_WEIGHT: float = 1.0
_GREEN_WEIGHT: float = 1.0
_BLUE_WEIGHT: float = 1.0
_X_WEIGHT: float = 1.0
_Y_WEIGHT: float = 1.0


def extract_features(array: np.ndarray, bit_depth: int) -> np.ndarray:
    """Build the (H*W, 5) feature matrix from an RGB image.

    Normalizes RGB channels to [0, 1] based on bit depth and generates
    normalized spatial coordinates. Each of the 5 features is scaled by
    its corresponding module-level weight constant.

    Args:
        array: RGB image array with shape (H, W, 3).
        bit_depth: Source bit depth used for integer normalization.

    Returns:
        Feature matrix with shape (H*W, 5) and dtype float32.
    """
    height, width = array.shape[:2]

    if np.issubdtype(array.dtype, np.floating):
        max_val = 1.0
    else:
        max_val = float((1 << bit_depth) - 1)

    rgb = array[..., :3].astype(np.float32) / max_val
    r = rgb[:, :, 0].ravel() * _RED_WEIGHT
    g = rgb[:, :, 1].ravel() * _GREEN_WEIGHT
    b = rgb[:, :, 2].ravel() * _BLUE_WEIGHT

    x_coords = np.linspace(0.0, 1.0, width, dtype=np.float32)
    y_coords = np.linspace(0.0, 1.0, height, dtype=np.float32)
    xx, yy = np.meshgrid(x_coords, y_coords)
    x_flat = xx.ravel() * _X_WEIGHT
    y_flat = yy.ravel() * _Y_WEIGHT

    return np.column_stack([r, g, b, x_flat, y_flat])


def run_kmeans(
    features: np.ndarray,
    n_clusters: int,
    height: int,
    width: int,
    *,
    random_state: int = 42,
) -> np.ndarray:
    """Run K-Means clustering and return labels reshaped to image dimensions.

    Args:
        features: Feature matrix with shape (H*W, D).
        n_clusters: Number of clusters.
        height: Image height for reshaping labels.
        width: Image width for reshaping labels.
        random_state: Seed for reproducibility.

    Returns:
        Label array with shape (H, W) and dtype int32.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = kmeans.fit_predict(features)
    return labels.reshape(height, width).astype(np.int32)
