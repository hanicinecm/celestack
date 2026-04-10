"""Star overlay and segment boundary visualization."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import polars as pl

from celestack.frame.core import Frame

_STAR_COLOR = "rgba(0, 255, 100, 0.7)"
"""Colour for star markers."""

_SEGMENT_BOUNDARY_COLOR = "rgba(255, 255, 0, 0.35)"
"""Colour for segment boundary dots."""

_MIN_STAR_MARKER = 3
"""Minimum scatter marker size for the faintest stars."""

_MAX_STAR_MARKER = 14
"""Maximum scatter marker size for the brightest stars."""


def _segment_boundary_coords(
    segment_labels: np.ndarray,
    downscale_factor: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Find boundary pixels between segments and scale to full-res.

    A pixel is a boundary pixel if any of its 4-connected neighbours
    belongs to a different segment or is foreground (``-1``).

    Args:
        segment_labels: 2D int32 label array (proxy space).
        downscale_factor: Scale factor to full-res coordinates.

    Returns:
        ``(x_coords, y_coords)`` in full-resolution pixel space.
    """
    h, w = segment_labels.shape
    is_boundary = np.zeros((h, w), dtype=bool)

    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        shifted = np.roll(segment_labels, (dy, dx), axis=(0, 1))
        diff = shifted != segment_labels
        is_boundary |= diff

    # Exclude foreground pixels themselves.
    is_boundary &= segment_labels >= 0

    # Zero out rolled-in edges.
    if h > 0:
        is_boundary[0, :] = False
        is_boundary[-1, :] = False
    if w > 0:
        is_boundary[:, 0] = False
        is_boundary[:, -1] = False

    rows, cols = np.where(is_boundary)
    return (
        cols.astype(np.float64) * downscale_factor,
        rows.astype(np.float64) * downscale_factor,
    )


def plot_stars(
    frame: Frame,
    stars: pl.DataFrame | None,
    segment_labels: np.ndarray | None,
    show_segments: bool,
) -> go.Figure:
    """Create a Plotly figure with available detection results overlaid on the frame.

    Both *stars* and *segment_labels* are optional — the figure is always
    returned regardless of which have been computed.

    Args:
        frame: Source frame (provides the base image via ``plot()``).
        stars: Optional DataFrame with ``x0``, ``y0``, ``flux`` columns.
            ``x0`` and ``y0`` are in full-resolution pixel coordinates,
            matching the axes of ``frame.plot()``.  Pass ``None`` to omit.
        segment_labels: Optional 2D segment label array (label-image space).
        show_segments: Whether to overlay segment boundary lines.

    Returns:
        Plotly figure with the frame image and any available overlays.
    """
    fig = frame.plot()

    if stars is not None:
        flux = stars["flux"].to_numpy()
        if flux.size > 0:
            flux_min, flux_max = float(flux.min()), float(flux.max())
            flux_range = flux_max - flux_min if flux_max > flux_min else 1.0
            sizes = (
                _MIN_STAR_MARKER
                + (_MAX_STAR_MARKER - _MIN_STAR_MARKER) * (flux - flux_min) / flux_range
            )
        else:
            sizes = np.array([], dtype=np.float64)

        fig.add_trace(
            go.Scatter(
                x=stars["x0"].to_list(),
                y=stars["y0"].to_list(),
                mode="markers",
                marker=dict(
                    color=_STAR_COLOR,
                    size=sizes.tolist() if sizes.size > 0 else [],
                ),
                name="Stars",
                showlegend=True,
                hovertext=[f"flux={f:.0f}" for f in flux] if flux.size > 0 else [],
                hoverinfo="text",
            )
        )

    if show_segments and segment_labels is not None:
        bx, by = _segment_boundary_coords(segment_labels, frame.downscale_factor)
        fig.add_trace(
            go.Scatter(
                x=bx.tolist(),
                y=by.tolist(),
                mode="markers",
                marker=dict(
                    color=_SEGMENT_BOUNDARY_COLOR,
                    size=1,
                ),
                name="Segments",
                showlegend=True,
                hoverinfo="skip",
            )
        )

    return fig
