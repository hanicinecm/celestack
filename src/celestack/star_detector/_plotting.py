"""Star overlay and segment boundary visualization."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import polars as pl

from celestack.frame.core import Frame

_SEGMENT_BOUNDARY_COLOR = "rgba(255, 255, 0, 0.35)"  # segment boundary dots
_MIN_STAR_MARKER = 3  # scatter marker size for the faintest stars
_MAX_STAR_MARKER = 14  # scatter marker size for the brightest stars
_STAR_COLORSCALE = "Viridis"  # flux → colour scale for star markers

# Columns displayed in the star hover tooltip, in order.  Missing columns
# are silently skipped so the plot never breaks on partial data.
_STAR_HOVER_COLS: tuple[tuple[str, str], ...] = (
    ("star_id", "id=%d"),
    ("x0", "x0=%.1f"),
    ("y0", "y0=%.1f"),
    ("flux", "flux=%.0f"),
    ("fwhm", "fwhm=%.2f"),
    ("roundness", "roundness=%.2f"),
    ("threshold", "threshold=%.1f"),
    ("threshold_sigma", "threshold_sigma=%.2fσ"),
    ("segment_id", "segment=%d"),
)


def _segment_boundary_coords(
    segment_labels: np.ndarray,
    downscale_factor: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find boundary pixels between segments and scale to full-res.

    A pixel is a boundary pixel if any of its 4-connected neighbours
    belongs to a different segment or is foreground (``-1``).

    Args:
        segment_labels: 2D int32 label array (proxy space).
        downscale_factor: Scale factor to full-res coordinates.

    Returns:
        ``(x_coords, y_coords, seg_ids)`` — boundary pixel coordinates in
        full-resolution pixel space, plus the segment label owning each pixel.
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
        segment_labels[rows, cols],
    )


def plot_stars(
    frame: Frame,
    segment_labels: np.ndarray | None,
    stars: pl.DataFrame | None,
    show_segments: bool,
) -> go.Figure:
    """Create a Plotly figure with available detection results overlaid on the frame.

    Both *stars* and *segment_labels* are optional — the figure is always
    returned regardless of which have been computed.

    Args:
        frame: Source frame (provides the base image via ``plot()``).
        segment_labels: Optional 2D segment label array (label-image space).
        stars: Optional DataFrame with ``x0``, ``y0``, ``flux`` columns.
            ``x0`` and ``y0`` are in full-resolution pixel coordinates,
            matching the axes of ``frame.plot()``.  Pass ``None`` to omit.
        show_segments: Whether to overlay segment boundary lines.

    Returns:
        Plotly figure with the frame image and any available overlays.
    """
    fig = frame.plot()

    if show_segments and segment_labels is not None:
        bx, by, seg_ids = _segment_boundary_coords(
            segment_labels, frame.downscale_factor
        )
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
                hovertext=[f"segment={int(s)}" for s in seg_ids],
                hoverinfo="text",
            )
        )

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
            sizes = np.array([], dtype=np.float32)

        hovertext = _build_star_hover_lines(stars)

        fig.add_trace(
            go.Scatter(
                x=stars["x0"].to_list(),
                y=stars["y0"].to_list(),
                mode="markers",
                marker=dict(
                    color=flux.tolist() if flux.size > 0 else [],
                    colorscale=_STAR_COLORSCALE,
                    size=sizes.tolist() if sizes.size > 0 else [],
                    showscale=False,
                ),
                name="Stars",
                showlegend=True,
                hovertext=hovertext,
                hoverinfo="text",
            )
        )

    return fig


def _build_star_hover_lines(stars: pl.DataFrame) -> list[str]:
    """Assemble per-row hover strings from whichever columns are present."""
    available = [(col, fmt) for col, fmt in _STAR_HOVER_COLS if col in stars.columns]
    if not available:
        return []

    arrays = {col: stars[col].to_list() for col, _ in available}
    lines: list[str] = []
    for i in range(len(stars)):
        parts = [fmt % arrays[col][i] for col, fmt in available]
        lines.append("<br>".join(parts))
    return lines
