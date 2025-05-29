import base64
import functools
import io
import warnings
from collections.abc import Callable
from os import PathLike
from pathlib import Path
from typing import cast

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


def compress_to_grayscale(img_array: np.ndarray) -> np.ndarray:
    """Create a grayscale 8bit image array from the original image array.

    Args:
        img_array: The original image array. Normally this is a 16-bit 3-channel RGB.

    Returns:
        A NumPy array representing the grayscale 8bit image.
    """
    depth = get_bit_depth(img_array)

    # Convert grayscale using the standard luminance formula:
    if len(img_array.shape) == 3:
        img_array = np.dot(img_array[..., :3], [0.299, 0.587, 0.114])

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


@functools.lru_cache(maxsize=128)
def _get_rough_prediction_model(
    x: tuple[float, ...], y: tuple[float, ...], t: tuple[float, ...]
) -> Callable[[float], tuple[float, float]]:
    """Get a model for rough prediction (x, y) based on time t.

    The function takes three tuples: x, y, and t, which represent the
    x-coordinates, y-coordinates, and time values of the known star positions.
    It returns a callable that takes an arbitrary time value and returns the
    predicted (x, y) coordinates of the star at that time.

    Args:
        x: A tuple of x-coordinates of the known star positions.
        y: A tuple of y-coordinates of the known star positions.
        t: A tuple of time values corresponding to the known star positions.

    Returns:
        A callable that takes a time value and returns the predicted (x, y)
        coordinates of the star at that time.
    """
    if len(x) != len(y) or len(x) != len(t) or not len(x):
        msg = "The lengths of x, y, and t must be the same and more than 0."
        raise ValueError(msg)

    if len(x) == 1:
        # The nearest neighbor model:
        return lambda t: (x[0], y[0])  # noqa: ARG005

    if len(x) == 2:
        # Trivial linear case:
        a_x = (x[1] - x[0]) / (t[1] - t[0])
        b_x = x[0] - a_x * t[0]
        a_y = (y[1] - y[0]) / (t[1] - t[0])
        b_y = y[0] - a_y * t[0]
        return lambda t: (a_x * t + b_x, a_y * t + b_y)

    # Fit a linear model to the points:
    a_x, b_x = np.polyfit(t, x, 1)
    a_y, b_y = np.polyfit(t, y, 1)

    return lambda t: (a_x * t + b_x, a_y * t + b_y)


def predict_star_position_in_frame(
    star_coordinates: pd.DataFrame,
    frame_name: str,
    n_closest: int = 7,
) -> tuple[float, float]:
    """Predict the star position in a frame, based on already know positions in others.

    The unknown position of a star in a frame with the name `frame_name` is predicted
    based on the `star_coordinates` DataFrame, which contains the coordinates of the
    star in other frames.
    The DataFrame must have the columns "t", "x" and "y" and the index must be
    the frame names. The "t" column contains the time of capture of the frame,
    relative to the reference frame (in seconds or any other arbitrary time unit) and
    it should be fully populated for all frames.
    The "x" and "y" columns contain the pixel coordinates of the star in the respective
    frame and it will be NaN for the frames where the star has not yet been detected.

    If only the position is known only for a single frame (the reference frame),
    then this position is used as the predicted position in the next frame.
    This is only possible, if the frame in question is the closest neighbor of the
    reference frame, otherwise the function returns tuple of NaNs.

    If more than one position is known, then the N closest positions in star_coordinates
    are fitted with a linear fit and the position is extrapolated to the frame in
    question.

    Args:
        star_coordinates: The DataFrame with the star coordinates.
            The DataFrame must have the columns "t", "x" and "y" and the index must be
            the frame names.
            The x, y are pixel coordinates of the star in the respective frame.
            The t is the time of capture of the frame, relative to the reference frame
            (in seconds or any other arbitrary time unit).
        frame_name: The name of the frame to predict the position for.
        n_closest: Only the N known closest stars to the frame in question are used for
            the fit to extrapolate the position of the star in the frame in question.

    Returns:
        The predicted x and y coordinates of the star in the frame with the name
        `frame_name`.
        The coordinates are in pixels in that frame.
        If the prediction failed, tuple of NaNs is returned.

    TODO: Return also the maximal allowed error of the prediction.
    """
    # Mask out the frames where the star coordinates are not known:
    known = star_coordinates.dropna(subset=["x", "y"])

    # The nearest-neighbor model is only allowed for the nearest neighbor :)
    if len(known) == 1:
        # What is the loc index of the frame with the known coordinates?
        known_name: str = known.index[0]
        i_known = star_coordinates.index.get_loc(known_name)
        i_known = cast("int", i_known)
        # Is the `frame_name` the closest neighbor of the known frame?
        i_frame = star_coordinates.index.get_loc(frame_name)
        i_frame = cast("int", i_frame)
        if abs(i_known - i_frame) > 1:
            # It's not!
            return np.nan, np.nan

    t_frame: float = star_coordinates.t[frame_name]

    # Select up to n_closest rows with t closest to t_frame
    if len(known) > n_closest:
        known = known.iloc[(known.t - t_frame).abs().argsort()[:n_closest]]

    model = _get_rough_prediction_model(
        x=tuple(known.x), y=tuple(known.y), t=tuple(known.t)
    )

    return model(t_frame)
