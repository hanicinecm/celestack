import base64
import io
import warnings
from os import PathLike
from pathlib import Path

import exifread
import numpy as np
import pandas as pd
import tifffile
from astropy.stats import mad_std
from photutils.detection import DAOStarFinder
from photutils.detection.daofinder import NoDetectionsWarning
from PIL import Image
from plotly import graph_objects as go


def read_image(img_path: str | PathLike) -> np.ndarray:
    """Read an image file and return its pixels as a NumPy array.

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
    array: np.ndarray,
    img_path: str | PathLike,
    *,
    overwrite: bool = False,
) -> None:
    """Save a NumPy array as a compressionless TIFF.

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
        msg = f"File '{img_path}' already exists. Use `overwrite=True` to overwrite it."
        raise FileExistsError(msg)

    tifffile.imwrite(img_path, array, compression=None)


def read_exif_data(img_path: str | PathLike) -> dict[str, str]:
    """Read selected EXIF data from an image file.

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

    with Path(img_path).open("rb") as img_file:
        exif_data_raw = exifread.process_file(img_file, details=False)

    return {
        tag.split()[1]: str(value).strip()
        for tag, value in exif_data_raw.items()
        if tag in selected_tags
    }


def get_bit_depth(img_array: np.ndarray) -> int:
    """Determine the bit depth of an image array.

    Args:
        img_array: The image array.

    Returns:
        The bit depth of the image.
    """
    depth = {"uint8": 8, "uint16": 16}.get(img_array.dtype.name)
    if depth is None:
        msg = f"Unsupported array data type: '{img_array.dtype.name}'."
        raise ValueError(msg)
    return depth


def compress_image(img_array: np.ndarray, downscale_factor: int = 1) -> np.ndarray:
    """Create a compressed image array from the original image array.

    The compressed image is an 8-bit grayscale, optionally downscaled by a specified
    factor. The downscaling is done by averaging the pixel values in blocks of size
    `downscale_factor` x `downscale_factor` into a single pixel value.

    Args:
        img_array: The original image array. Normally this is a 16-bit 3-channel RGB.
        downscale_factor: The factor by which to downscale the image. Optional,
            defaults to 1 (no downscaling).

    Returns:
        A NumPy array representing the compressed image.

    TODO: Refactor this - should be called `create_grayscale_image` or similar...
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
    return img_array.astype(np.uint8)


def plot_image(
    array: np.ndarray,
    size: tuple[int, int] = (1200, 900),
    *,
    interactive: bool = False,
) -> go.Figure:
    """Plot an image as a Plotly figure.

    The figure pixel dimensions can be set using the `width` and `height` parameters
    and the image can be displayed either as a static image or as an interactive
    map where each pixel can be hovered over to see its coordinates and value (slower).

    Args:
        array: The image array to display.
        size: The size of the figure in pixels, as a tuple (width, height).
        interactive: If True, the image will be displayed in an interactive Plotly
            figure, where each pixel can be hovered over to see its coordinates and
            value. Optional, defaults to False.

    Returns:
        A Plotly figure object containing the image.

    TODO: Make the size parameter optional - this will require redefining the layout
        of the figure.
    """
    if array.ndim != 2:
        msg = "Only 2D arrays (grayscale images) are supported for plotting."
        raise NotImplementedError(
            msg,
        )

    # Get the image dimensions
    img_height, img_width = array.shape

    # Initialize the figure:
    if interactive:
        fig = go.Figure(data=go.Heatmap(z=array, colorscale="gray", showscale=False))
    else:
        fig = go.Figure()

        # Convert grayscale array to PIL image and then to base64 PNG
        img_pil = Image.fromarray(array.astype(np.uint8))
        buffer = io.BytesIO()
        img_pil.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        # Add the image to the figure as a background
        fig.update_layout(
            images=[
                {
                    "source": "data:image/png;base64," + img_base64,
                    "xref": "x",
                    "yref": "y",
                    "x": 0,
                    "y": 0,
                    "sizex": img_width,
                    "sizey": img_height,
                    "sizing": "stretch",
                    "opacity": 1.0,
                    "layer": "below",
                },
            ],
        )

    # Create a Plotly figure with the image in the background
    fig.update_layout(
        xaxis={
            "range": [0, img_width],
            "visible": False,
            "showgrid": False,
            "zeroline": False,
        },
        yaxis={
            "range": [img_height, 0],
            "visible": False,
            "showgrid": False,
            "zeroline": False,
            "scaleanchor": "x",
        },
        width=size[0],
        height=size[1],
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0)",
        margin={"l": 0, "r": 0, "t": 30, "b": 0},
    )

    return fig


def find_stars(
    array: np.ndarray,
    threshold: float,
    fwhm: float,
    *,
    max_roundness: float = 1.0,
    min_separation: float = 1.5,
    mask: np.ndarray | None = None,
    exclude_border: bool = True,
    show: bool = False,
) -> pd.DataFrame:
    """Find stars in an image array using the DAOStarFinder algorithm.

    This function uses the `photutils` library to detect stars in an array.
    It calculates the background noise using the median absolute deviation (MAD) and
    uses the DAOStarFinder algorithm to find sources in the image. The function
    returns a pandas DataFrame containing the detected stars.

    TODO: Change the table to a polars DataFrame.

    The stars with |roundness| > `max_roundness` are filtered out.
    Also, the stars close to the image border are filtered out, as are the stars within
    1.5 * fwhm of other stars.

    Args:
        array: The image array in which to find stars.
        threshold: The threshold for finding stars, in units of standard deviations of
            the background noise.
        fwhm: The full width at half maximum (FWHM) for the star finder.
        max_roundness: The maximum roundness of the stars. Optional, defaults to 1.0.
            More elongated stars will require a higher value not to be filtered out.
            1.0 is the default used by the DAOStarFinder algorithm.
        min_separation: The minimum separation between stars, in the units of fwhm.
            Optional, defaults to 1.5.
        mask: A mask to exclude certain pixels from the analysis. Optional, defaults to
            None (no mask).
        exclude_border: If True, excludes stars that are too close to the image border.
            Optional, defaults to True.
        show: If True, displays the plotly image with the detected stars.
            Optional, defaults to False.

    Returns:
        A pandas DataFrame containing the detected stars, as returned by the
        DAOStarFinder algorithm, indexed by their IDs.
        If no stars are found, an empty DataFrame is returned.
    """
    bkg_mad = mad_std(array)

    find = DAOStarFinder(
        fwhm=fwhm,
        threshold=threshold * bkg_mad,
        roundlo=-max_roundness,
        roundhi=max_roundness,
        exclude_border=exclude_border,
        min_separation=min_separation * fwhm,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NoDetectionsWarning)
        sources = find(data=array, mask=mask)

    if sources is None:
        return pd.DataFrame()
    sources = sources.to_pandas().set_index("id", drop=True)

    if show:
        fig = plot_image(array)
        fig.add_scatter(
            x=sources["xcentroid"],
            y=sources["ycentroid"],
            mode="markers",
            marker={
                "size": sources["flux"] / np.max(sources["flux"]) * 10,
                "color": sources["flux"],
                "colorscale": "Viridis",
            },
            name="Stars",
        )
        fig.update_layout(
            title="Stars",
            xaxis_title="X (pixels)",
            yaxis_title="Y (pixels)",
            showlegend=True,
        )
        fig.show()

    return sources
