"""Plotting helpers for mask and cluster visualization."""

from __future__ import annotations

import colorsys

import numpy as np
import plotly.graph_objects as go

from celestack.frame._plotting import as_png_data_uri

_PLOT_TARGET_LONG_EDGE = 2560
"""Target long-edge resolution for cluster plot PNG encoding."""


def _downscale_nearest(array: np.ndarray, target_long_edge: int) -> np.ndarray:
    """Downscale a 2D array by nearest-neighbor to fit a target long edge.

    Returns the array unchanged if it already fits.  The factor is an
    integer so the result dimensions are approximate, not exact.

    Args:
        array: 2D array with shape (H, W).
        target_long_edge: Maximum long-edge size in pixels.

    Returns:
        Downscaled array (view via slicing) with the same dtype.
    """
    long_edge = max(array.shape[0], array.shape[1])
    if long_edge <= target_long_edge:
        return array
    factor = max(1, long_edge // target_long_edge)
    return array[::factor, ::factor]


def _cluster_palette(n: int) -> list[tuple[int, ...]]:
    """Generate n visually distinct RGB colors via HSV hue rotation."""
    return [
        tuple(int(c * 255) for c in colorsys.hsv_to_rgb(i / n, 0.75, 0.90))
        for i in range(n)
    ]


def _base_layout(height: int, width: int, title: str) -> dict:
    """Return shared Plotly layout settings."""
    return dict(
        title=title,
        xaxis=dict(
            range=[0, width],
            visible=False,
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            range=[height, 0],
            visible=False,
            showgrid=False,
            zeroline=False,
            scaleanchor="x",
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0)",
    )


def plot_clusters(labels: np.ndarray) -> go.Figure:
    """Visualize K-Means cluster labels as a color-coded image.

    Renders each cluster as a solid color block.  Large label arrays are
    downscaled via nearest-neighbor before PNG encoding to keep rendering
    fast; the axes always reflect the original (full-resolution) dimensions.
    A non-interactive legend maps each color to its cluster index.

    Args:
        labels: Cluster label array with shape (H, W).

    Returns:
        Plotly figure with a static PNG cluster image and a legend.
    """
    full_h, full_w = labels.shape
    n_clusters = int(labels.max()) + 1
    palette = _cluster_palette(n_clusters)

    plot_labels = _downscale_nearest(labels, _PLOT_TARGET_LONG_EDGE)
    ph, pw = plot_labels.shape

    cluster_image = np.zeros((ph, pw, 3), dtype=np.uint8)
    for cluster_id, color in enumerate(palette):
        cluster_image[plot_labels == cluster_id] = color

    fig = go.Figure()
    fig.add_layout_image(
        dict(
            source=as_png_data_uri(cluster_image),
            xref="x",
            yref="y",
            x=0,
            y=0,
            sizex=full_w,
            sizey=full_h,
            sizing="stretch",
            layer="below",
        )
    )

    for cluster_id, (r, g, b) in reversed(list(enumerate(palette))):
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(color=f"rgb({r},{g},{b})", size=12, symbol="square"),
                name=f"Cluster {cluster_id}",
                showlegend=True,
            )
        )

    layout = _base_layout(full_h, full_w, f"Cluster Map (K={n_clusters})")
    layout["legend"] = dict(itemclick=False, itemdoubleclick=False)
    fig.update_layout(layout)
    return fig


def plot_mask(
    array: np.ndarray,
    *,
    title: str = "<Mask>",
    downscale_factor: int = 1,
) -> go.Figure:
    """Visualize a boolean mask as a binary image with Plotly.

    The mask is displayed as a black-and-white image, where foreground pixels
    are white and background pixels are black.  The axes are scaled to match the
    full-resolution dimensions based on the provided downscale factor, but pixel
    hover data is disabled for performance.

    Args:
        array: 2D boolean array with shape (H, W) representing the mask.
        title: Title for the plot. Optional.
        downscale_factor: Ratio between full-resolution and proxy dimensions. Optional.
    """
    full_h = array.shape[0] * downscale_factor
    full_w = array.shape[1] * downscale_factor

    fig = go.Figure()
    fig.add_layout_image(
        dict(
            source=as_png_data_uri(array),
            xref="x",
            yref="y",
            x=0,
            y=0,
            sizex=full_w,
            sizey=full_h,
            sizing="stretch",
            layer="below",
        )
    )
    layout = _base_layout(full_h, full_w, title)
    fig.update_layout(layout)

    return fig
