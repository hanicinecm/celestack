"""Plotting helpers for mask visualization."""

from __future__ import annotations

from typing import cast

import numpy as np
import plotly.graph_objects as go
from scipy import ndimage

from celestack.frame._plotting import as_png_data_uri
from celestack.mask._morphology import _small_component_metadata

_FG_NOISE_COLOR = "rgb(0, 0, 255)"
"""Red for foreground noise particles."""

_BG_NOISE_COLOR = "rgb(255, 0, 0)"
"""Blue for background noise particles (holes)."""

_MIN_NOISE_MARKER_SIZE = 4
"""Minimum Scatter marker size for the smallest noise particles."""

_MAX_NOISE_MARKER_SIZE = 18
"""Maximum Scatter marker size for the largest noise particles."""


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
    highlight_noise: int,
) -> go.Figure:
    """Visualize a boolean mask as a binary image with Plotly.

    The mask is displayed as a black-and-white image, where foreground pixels
    are white and background pixels are black.  Axes are in the mask's native
    pixel coordinates; pixel hover data is disabled for performance.

    When *highlight_noise* is set, Scatter traces are added to mark
    small connected components: foreground noise in red and background
    noise (holes) in blue.  Marker size is proportional to particle area.
    The traces can be toggled via the legend.

    Args:
        array: 2D boolean array with shape (H, W) representing the mask.
        title: Title for the plot.
        highlight_noise: Maximum component area (in pixels) to highlight.
            Use ``0`` to disable highlighting.
    """
    full_h = array.shape[0]
    full_w = array.shape[1]

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

        if fg_x:
            fig.add_trace(
                go.Scatter(
                    x=fg_x,
                    y=fg_y,
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
                    x=bg_x,
                    y=bg_y,
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

    fig.update_layout(
        title=title,
        xaxis=dict(
            range=[0, full_w],
            visible=False,
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            range=[full_h, 0],
            visible=False,
            showgrid=False,
            zeroline=False,
            scaleanchor="x",
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0)",
    )

    return fig
