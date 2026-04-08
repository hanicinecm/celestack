"""Plotting helpers for mask and cluster visualization."""

from __future__ import annotations

import colorsys

import numpy as np
import plotly.graph_objects as go

from celestack.frame._plotting import as_png_data_uri


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

    Renders each cluster as a solid color block.  A non-interactive legend
    maps each color to its cluster index.

    Args:
        labels: Cluster label array with shape (H, W).

    Returns:
        Plotly figure with a static PNG cluster image and a legend.
    """
    height, width = labels.shape
    n_clusters = int(labels.max()) + 1
    palette = _cluster_palette(n_clusters)

    cluster_image = np.zeros((height, width, 3), dtype=np.uint8)
    for cluster_id, color in enumerate(palette):
        cluster_image[labels == cluster_id] = color

    fig = go.Figure()
    fig.add_layout_image(
        dict(
            source=as_png_data_uri(cluster_image),
            xref="x",
            yref="y",
            x=0,
            y=0,
            sizex=width,
            sizey=height,
            sizing="stretch",
            layer="below",
        )
    )

    for cluster_id, (r, g, b) in enumerate(palette):
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

    layout = _base_layout(height, width, f"Cluster Map (K={n_clusters})")
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
