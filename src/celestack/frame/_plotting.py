"""Plotting helpers for frame visualization."""

from __future__ import annotations

import base64
import io

import numpy as np
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
