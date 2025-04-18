from os import PathLike
from pathlib import Path

import exifread
import numpy as np
import tifffile
from PIL import Image
from plotly import graph_objects as go


def read_image(img_path: str | PathLike) -> np.ndarray:
    """
    Reads an image file and returns its pixel data as a NumPy array.

    It is designed to handle multiple image formats, predominantly the TIFF format
    (in both 8-bit and 16-bit color depth), and supports both grayscale and RGB images.

    Args:
        img_path: The path to the image file.

    Returns:
        A NumPy array representing the pixel data of the image.
    """
    img_path = Path(img_path)

    # Read the image to array - prefer tifffile for TIFF files and fall back to PIL
    # for other formats
    if img_path.suffix.lower() in {".tif", ".tiff"}:
        array = tifffile.imread(img_path)
    else:
        with Image.open(img_path) as img:
            array = np.array(img)

    return array


def save_tiff(
    array: np.ndarray, img_path: str | PathLike, overwrite: bool = False
) -> None:
    """
    Saves a NumPy array as a TIFF compressionless tiff.

    This function is used both for saving the compressed images and the created
    full-resolution full-depth images (such as the master dark).
    By default, it will raise an error if the file already exists, unless the
    `overwrite` parameter is set to True.

    Args:
        array: The NumPy array to save.
        img_path: The path to save the TIFF image.
        overwrite: If True, overwrites the file if it already exists. Optional, defaults
            to False.

    Raises:
        FileExistsError: If the file already exists and `overwrite` is False.

    TODO: Add support for metadata.
    """
    if not overwrite and Path(img_path).exists():
        raise FileExistsError(f"File '{img_path}' already exists.")

    tifffile.imwrite(img_path, array, compression=None)


def read_exif_data(img_path: str | PathLike) -> dict[str, str]:
    """
    Reads the selected EXIF data from an image file.

    Args:
        img_path: The path to the image file.

    Returns:
        A dictionary containing the EXIF data.
    """
    selected_tags = [
        "Image Make",
        "Image Model",
        "EXIF DateTimeOriginal",
        "EXIF ExposureTime",
        "EXIF FNumber",
        "EXIF ISOSpeedRatings",
        "EXIF ColorSpace",
        "EXIF FocalLength",
        "EXIF LensModel",
    ]

    with open(img_path, "rb") as img_file:
        exif_data_raw = exifread.process_file(img_file, details=False)

    exif_data = {
        tag.split()[1]: str(value).strip()
        for tag, value in exif_data_raw.items()
        if tag in selected_tags
    }

    return exif_data


def get_bit_depth(img_array: np.ndarray) -> int:
    """
    Determines the bit depth of an image array.

    Args:
        img_array: The image array.

    Returns:
        The bit depth of the image.
    """
    depth = {"uint8": 8, "uint16": 16}.get(img_array.dtype.name)
    if depth is None:
        raise ValueError(f"Unsupported array data type: '{img_array.dtype.name}'.")
    return depth


def compress_image(img_array: np.ndarray, downscale_factor: int = 1) -> np.ndarray:
    """
    Creates a compressed image array from the original image array.

    The compressed image is an 8-bit grayscale, optionally downscaled by a specified
    factor. The downscaling is done by averaging the pixel values in blocks of size
    `downscale_factor` x `downscale_factor` into a single pixel value.

    Args:
        img_array: The original image array. Normally this is a 16-bit 3-channel RGB.
        downscale_factor: The factor by which to downscale the image. Optional,
            defaults to 1 (no downscaling).

    Returns:
        A NumPy array representing the compressed image.
    """
    depth = get_bit_depth(img_array)

    # Convert grayscale using the standard luminance formula:
    if len(img_array.shape) == 3:
        img_array = np.dot(img_array[..., :3], [0.299, 0.587, 0.114])

    # Downscale the array - each NxN pixels will be averaged to one:
    if downscale_factor > 1:
        img_array = img_array[
            : img_array.shape[0] // downscale_factor * downscale_factor,
            : img_array.shape[1] // downscale_factor * downscale_factor,
        ]
        img_array = img_array.reshape(
            img_array.shape[0] // downscale_factor,
            downscale_factor,
            img_array.shape[1] // downscale_factor,
            downscale_factor,
        ).mean(axis=(1, 3))

    # Convert to 8-bit:
    img_array = img_array / (2**depth - 1) * 255
    img_array[img_array > 255] = 255
    img_array = img_array.astype(np.uint8)

    return img_array


def plot_image(array: np.ndarray) -> go.Figure:
    """
    Displays an image using matplotlib.

    Args:
        array: The image array to display.

    Returns:
        A Plotly figure object containing the image.
    """
    if array.ndim == 2:
        fig = go.Figure(data=go.Heatmap(z=array, colorscale="gray", showscale=False))
    else:
        raise NotImplementedError

    fig.update_layout(
        yaxis=dict(
            autorange="reversed",
            showgrid=False,
            zeroline=False,
            visible=False,
            scaleanchor="x",
        ),
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0)",
    )

    return fig
