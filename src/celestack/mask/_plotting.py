"""Plotting helpers for mask and cluster visualization."""

from __future__ import annotations

import colorsys

import numpy as np
import plotly.graph_objects as go

from celestack.frame._plotting import as_png_data_uri

# Alpha (0-255) for overlay traces.
_OVERLAY_ALPHA: int = 110


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


def plot_clusters(
    image_array: np.ndarray,
    labels: np.ndarray,
    bit_depth: int,
) -> go.Figure:
    """Visualize K-Means cluster labels overlaid on the source image.

    Args:
        image_array: RGB image array with shape (H, W, 3).
        labels: Cluster label array with shape (H, W).
        bit_depth: Source bit depth for preview scaling.

    Returns:
        Plotly figure with the image as background and color-coded
        cluster regions overlaid.
    """
    height, width = image_array.shape[:2]
    n_clusters = int(labels.max()) + 1
    palette = _cluster_palette(n_clusters)

    overlay = np.zeros((height, width, 4), dtype=np.uint8)
    for cluster_id, (r, g, b) in enumerate(palette):
        mask = labels == cluster_id
        overlay[mask, 0] = r
        overlay[mask, 1] = g
        overlay[mask, 2] = b
        overlay[mask, 3] = _OVERLAY_ALPHA

    fig = go.Figure()
    fig.add_layout_image(
        dict(
            source=as_png_data_uri(image_array),
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
    fig.add_trace(
        go.Image(
            z=overlay,
            x0=0,
            y0=0,
            dx=1,
            dy=1,
        )
    )
    fig.update_layout(_base_layout(height, width, f"Cluster Map (K={n_clusters})"))
    return fig


def plot_mask(
    image_array: np.ndarray,
    mask: np.ndarray,
    bit_depth: int,
) -> go.Figure:
    """Visualize a boolean mask overlaid on the source image.

    Args:
        image_array: Image array with shape (H, W, ...).
        mask: Boolean mask array with shape (H, W).
        bit_depth: Source bit depth for preview scaling.

    Returns:
        Plotly figure with the image as background and a semi-transparent
        red overlay highlighting foreground regions.
    """
    height, width = mask.shape[:2]

    overlay = np.zeros((height, width, 4), dtype=np.uint8)
    overlay[mask, 0] = 220
    overlay[mask, 3] = _OVERLAY_ALPHA

    fig = go.Figure()
    fig.add_layout_image(
        dict(
            source=as_png_data_uri(image_array),
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
    fig.add_trace(
        go.Image(
            z=overlay,
            x0=0,
            y0=0,
            dx=1,
            dy=1,
        )
    )
    fig.update_layout(_base_layout(height, width, "Mask Preview"))
    return fig
