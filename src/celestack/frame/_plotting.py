"""Plotting helpers for frame visualization."""

from __future__ import annotations

import base64
import io

import numpy as np
import plotly.graph_objects as go
from PIL import Image

from celestack.frame._image_ops import scaled_preview_uint8


def as_png_data_uri(array: np.ndarray) -> str:
    """Serialize an array to a PNG data URI for Plotly layout images."""
    if array.ndim == 2:
        image = Image.fromarray(scaled_preview_uint8(array), mode="L")
    elif array.ndim == 3 and array.shape[2] >= 3:
        preview = scaled_preview_uint8(array[..., :3])
        image = Image.fromarray(preview, mode="RGB")
    else:
        msg = f"Unsupported array shape for plotting: {array.shape}"
        raise ValueError(msg)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def plot_frame(
    array: np.ndarray,
    *,
    title: str,
    bit_depth: int,
    downscale_factor: int,
    show_pixels: bool = False,
) -> go.Figure:
    """Create a Plotly figure for frame visualization.

    Args:
        array: Image pixel data.
        title: Figure title (typically the filename).
        bit_depth: Source bit depth for correct value scaling.
        downscale_factor: Ratio between full-res and proxy dimensions.
        show_pixels: Whether to plot with pixel hover data.

    Returns:
        A figure with axes in full-resolution pixel coordinates.
    """
    full_h = array.shape[0] * downscale_factor
    full_w = array.shape[1] * downscale_factor

    if show_pixels:
        fig = go.Figure()
        zmax = 2**bit_depth - 1
        if array.ndim == 2:
            fig.add_trace(
                go.Heatmap(
                    z=array,
                    x0=0,
                    dx=downscale_factor,
                    y0=0,
                    dy=downscale_factor,
                    zmin=0,
                    zmax=zmax,
                    colorscale="gray",
                    showscale=False,
                )
            )
        else:
            fig.add_trace(
                go.Image(
                    z=array,
                    x0=0,
                    dx=downscale_factor,
                    y0=0,
                    dy=downscale_factor,
                    zmax=[zmax, zmax, zmax, 1],
                )
            )
    else:
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
