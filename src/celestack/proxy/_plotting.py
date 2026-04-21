"""Plotting helpers for proxy frame and proxy mask visualization."""

from __future__ import annotations

import base64
import io

import numpy as np
import plotly.graph_objects as go
from PIL import Image


def _linear_scale_uint8(array: np.ndarray) -> np.ndarray:
    """Linearly scale a float array to 8-bit using its min/max range.

    No percentile clipping — the full dynamic range is preserved so
    sparse bright stars stay bright and sky noise stays dim relative
    to them.

    Only used for static-background rendering of proxy frames and masks.
    """
    data = array.astype(np.float32, copy=False)
    vmin, vmax = float(data.min()), float(data.max())
    if vmax <= vmin:
        return np.zeros(data.shape, dtype=np.uint8)
    normalized = (data - vmin) / (vmax - vmin)
    return (normalized * 255.0).astype(np.uint8)


def _array_as_png_data_uri(array: np.ndarray) -> str:
    """Encode a uint8 grayscale array as a PNG data URI."""
    image = Image.fromarray(array, mode="L")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def _apply_layout(fig: go.Figure, *, title: str, full_w: int, full_h: int) -> None:
    """Apply the shared proxy-plot layout (axes, margins, transparent bg)."""
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


def plot_proxy_frame(
    array: np.ndarray,
    *,
    title: str,
    downscale_factor: int,
    show_pixels: bool = True,
    upscale_coordinates: bool = True,
) -> go.Figure:
    """Create a Plotly figure for a proxy frame.

    Args:
        array: Float16 proxy array of shape (H, W).
        title: Figure title.
        downscale_factor: Factor relating proxy pixels to full-res pixels.
        show_pixels: If True (default), render as a Heatmap with
            per-pixel hover data. If False, render as a static PNG
            layout image (faster for very large proxies).
        upscale_coordinates: If True (default), scale the axes by
            *downscale_factor* so hover positions are expressed in the
            full-resolution coordinate system.

    Returns:
        Plotly figure with the proxy rendered as a Heatmap or layout image.
    """
    h, w = array.shape
    step = downscale_factor if upscale_coordinates else 1
    full_h = h * step
    full_w = w * step

    fig = go.Figure()
    if show_pixels:
        fig.add_trace(
            go.Heatmap(
                z=array,
                x0=step / 2,
                dx=step,
                y0=step / 2,
                dy=step,
                colorscale="gray",
                showscale=False,
                hovertemplate="x=%{x}<br>y=%{y}<br>value=%{z:.4f}<extra></extra>",
            )
        )
    else:
        preview = _linear_scale_uint8(array)
        fig.add_layout_image(
            dict(
                source=_array_as_png_data_uri(preview),
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
    _apply_layout(fig, title=title, full_w=full_w, full_h=full_h)
    return fig


def plot_proxy_mask(
    array: np.ndarray,
    *,
    title: str,
    downscale_factor: int,
    show_pixels: bool = True,
    upscale_coordinates: bool = True,
) -> go.Figure:
    """Create a Plotly figure for a proxy mask.

    Args:
        array: Boolean proxy mask of shape (H, W).
        title: Figure title.
        downscale_factor: Factor relating proxy pixels to full-res pixels.
        show_pixels: If True (default), render as a Heatmap with
            per-pixel hover data. If False, render as a static PNG
            layout image.
        upscale_coordinates: If True (default), scale the axes by
            *downscale_factor* so hover positions are expressed in the
            full-resolution coordinate system.

    Returns:
        Plotly figure with the mask rendered as a Heatmap or layout image.
    """
    h, w = array.shape
    step = downscale_factor if upscale_coordinates else 1
    full_h = h * step
    full_w = w * step

    fig = go.Figure()
    if show_pixels:
        fig.add_trace(
            go.Heatmap(
                z=array.astype(np.uint8),
                x0=step / 2,
                dx=step,
                y0=step / 2,
                dy=step,
                zmin=0,
                zmax=1,
                colorscale="gray",
                showscale=False,
                hovertemplate="x=%{x}<br>y=%{y}<br>foreground=%{z}<extra></extra>",
            )
        )
    else:
        preview = (array.astype(np.uint8)) * 255
        fig.add_layout_image(
            dict(
                source=_array_as_png_data_uri(preview),
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
    _apply_layout(fig, title=title, full_w=full_w, full_h=full_h)
    return fig
