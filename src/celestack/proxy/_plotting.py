"""Plotting helpers for proxy frame visualization."""

from __future__ import annotations

import base64
import io

import numpy as np
import plotly.graph_objects as go
from PIL import Image


def _percentile_stretch_uint8(array: np.ndarray) -> np.ndarray:
    """Stretch a float array to 8-bit using sky-pixel percentiles.

    Only non-zero pixels participate in the percentile computation — the
    foreground (masked) region of a :class:`ProxyFrame` is exactly
    ``0.0`` and should not pull the stretch down. When no sky pixels
    exist, the full array range is used.
    """
    data = array.astype(np.float32, copy=False)
    sky = data[data != 0.0]
    if sky.size > 0:
        vmin, vmax = np.percentile(sky, [1.0, 99.0])
    else:
        vmin, vmax = float(data.min()), float(data.max())
    if vmax <= vmin:
        return np.zeros(data.shape, dtype=np.uint8)
    normalized = np.clip((data - vmin) / (vmax - vmin), 0.0, 1.0)
    return (normalized * 255.0).astype(np.uint8)


def _array_as_png_data_uri(array: np.ndarray) -> str:
    """Encode a uint8 grayscale array as a PNG data URI."""
    image = Image.fromarray(array, mode="L")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def plot_proxy_frame(array: np.ndarray, *, title: str) -> go.Figure:
    """Create a Plotly figure for a proxy frame.

    The float16 array is percentile-stretched using only sky (non-zero)
    pixels and rendered as a single PNG layout image. Axes are in proxy
    pixel coordinates.

    Args:
        array: Float16 proxy array of shape (H, W).
        title: Figure title.

    Returns:
        Plotly figure with the proxy rendered as a layout image.
    """
    full_h, full_w = array.shape
    preview = _percentile_stretch_uint8(array)

    fig = go.Figure()
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
