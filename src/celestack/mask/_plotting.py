"""Plotting helpers for mask and cluster visualization."""

from __future__ import annotations

import colorsys
from typing import cast

import numpy as np
import plotly.graph_objects as go
from scipy import ndimage

from celestack.frame._plotting import as_png_data_uri
from celestack.mask._morphology import _small_component_metadata

DEFAULT_HIGHLIGHT_NOISE_MAX_SIZE = 128
"""Default maximum component area (in pixels) for noise highlighting."""

_FG_NOISE_COLOR = "rgb(0, 0, 255)"
"""Red for foreground noise particles."""

_BG_NOISE_COLOR = "rgb(255, 0, 0)"
"""Blue for background noise particles (holes)."""

_MIN_NOISE_MARKER_SIZE = 4
"""Minimum Scatter marker size for the smallest noise particles."""

_MAX_NOISE_MARKER_SIZE = 18
"""Maximum Scatter marker size for the largest noise particles."""

_PLOT_TARGET_LONG_EDGE = 2560
"""Target long-edge size in pixels for downscaling image before plotting."""


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


def _find_noise_particles(
    array: np.ndarray,
    max_size: int,
) -> tuple[list[float], list[float], list[int], list[float], list[float], list[int]]:
    """Locate small connected components and compute their centroids and sizes.

    Args:
        array: 2D boolean mask array.
        max_size: Maximum component area (in pixels) to consider noise.

    Returns:
        Tuple of ``(fg_x, fg_y, fg_sizes, bg_x, bg_y, bg_sizes)`` where
        each list contains the centroid coordinates and pixel areas of
        the small foreground / background components respectively.
    """
    fg_x: list[float] = []
    fg_y: list[float] = []
    fg_sizes: list[int] = []
    bg_x: list[float] = []
    bg_y: list[float] = []
    bg_sizes: list[int] = []

    # Foreground noise particles
    fg_labels, fg_all_sizes, fg_small = _small_component_metadata(array, max_size)
    fg_ids = np.flatnonzero(fg_small).astype(np.intp, copy=False)
    if fg_ids.size > 0:
        fg_centroids_raw = ndimage.center_of_mass(
            np.ones_like(array, dtype=np.uint8),
            labels=fg_labels,
            index=fg_ids.tolist(),
        )
        fg_centroids = cast(list[tuple[float, float]], fg_centroids_raw)
        for (y, x), size in zip(fg_centroids, fg_all_sizes[fg_ids], strict=True):
            fg_x.append(float(x))
            fg_y.append(float(y))
            fg_sizes.append(int(size))

    # Background noise particles (holes)
    bg_labels, bg_all_sizes, bg_small = _small_component_metadata(~array, max_size)
    bg_ids = np.flatnonzero(bg_small).astype(np.intp, copy=False)
    if bg_ids.size > 0:
        bg_centroids_raw = ndimage.center_of_mass(
            np.ones_like(array, dtype=np.uint8),
            labels=bg_labels,
            index=bg_ids.tolist(),
        )
        bg_centroids = cast(list[tuple[float, float]], bg_centroids_raw)
        for (y, x), size in zip(bg_centroids, bg_all_sizes[bg_ids], strict=True):
            bg_x.append(float(x))
            bg_y.append(float(y))
            bg_sizes.append(int(size))

    return fg_x, fg_y, fg_sizes, bg_x, bg_y, bg_sizes


def _particle_marker_sizes(sizes: list[int], max_size: int) -> list[float]:
    """Map particle pixel areas to Scatter marker sizes.

    Linearly interpolates between ``_MIN_MARKER_SIZE`` and
    ``_MAX_MARKER_SIZE`` based on particle area relative to *max_size*.

    Args:
        sizes: Pixel areas of each particle.
        max_size: Maximum particle area (used as the upper bound for scaling).

    Returns:
        List of marker sizes, one per particle.
    """
    span = _MAX_NOISE_MARKER_SIZE - _MIN_NOISE_MARKER_SIZE
    return [_MIN_NOISE_MARKER_SIZE + span * (s / max_size) for s in sizes]


def plot_mask(
    array: np.ndarray,
    *,
    title: str,
    downscale_factor: int,
    highlight_noise: int,
) -> go.Figure:
    """Visualize a boolean mask as a binary image with Plotly.

    The mask is displayed as a black-and-white image, where foreground pixels
    are white and background pixels are black.  The axes are scaled to match the
    full-resolution dimensions based on the provided downscale factor, but pixel
    hover data is disabled for performance.

    When *highlight_noise* is set, Scatter traces are added to mark
    small connected components: foreground noise in red and background
    noise (holes) in blue.  Marker size is proportional to particle area.
    The traces can be toggled via the legend.

    Args:
        array: 2D boolean array with shape (H, W) representing the mask.
        title: Title for the plot.
        downscale_factor: Ratio between full-resolution and proxy dimensions.
        highlight_noise: Maximum component area (in pixels) to highlight.
            Use ``0`` to disable highlighting.
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

    if highlight_noise > 0:
        fg_x, fg_y, fg_sizes, bg_x, bg_y, bg_sizes = _find_noise_particles(
            array, highlight_noise
        )
        scale = downscale_factor

        if fg_x:
            fig.add_trace(
                go.Scatter(
                    x=[v * scale for v in fg_x],
                    y=[v * scale for v in fg_y],
                    mode="markers",
                    marker=dict(
                        color=_FG_NOISE_COLOR,
                        size=_particle_marker_sizes(fg_sizes, highlight_noise),
                        symbol="circle",
                    ),
                    name="FG noise",
                    showlegend=True,
                    hovertext=[f"area={s}" for s in fg_sizes],
                    hoverinfo="text",
                )
            )

        if bg_x:
            fig.add_trace(
                go.Scatter(
                    x=[v * scale for v in bg_x],
                    y=[v * scale for v in bg_y],
                    mode="markers",
                    marker=dict(
                        color=_BG_NOISE_COLOR,
                        size=_particle_marker_sizes(bg_sizes, highlight_noise),
                        symbol="circle",
                    ),
                    name="BG noise",
                    showlegend=True,
                    hovertext=[f"area={s}" for s in bg_sizes],
                    hoverinfo="text",
                )
            )

    layout = _base_layout(full_h, full_w, title)
    fig.update_layout(layout)

    return fig
