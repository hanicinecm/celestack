import base64
import io
import warnings
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any

import exifread
import numpy as np
import tifffile
from astropy.stats import mad_std
from photutils.detection import DAOStarFinder
from photutils.detection.daofinder import NoDetectionsWarning
from PIL import Image
from plotly import graph_objects as go


@dataclass
class StarsList:
    """A class to hold a list of stars with their coordinates and other essential data.

    This class is used to store the results of the star detection algorithm.

    All the arrays are of the same length, which is the number of detected stars.

    The `x` and `y` coordinates are in pixels always relative to the top-left
    corner of either the image, or the segment of the image, in which the stars
    were detected, depending on the context of the detection.

    The `flux` is the total flux of the star in the image, which is the sum of the
    pixel values in the star's area. It is an integer value, which can be used to
    determine the brightness of the star.

    The `threshold` is the threshold used for the star detection, in units of
    standard deviations of the background noise. It is a float value, and it's the
    parameter to the star detection algorithm, so it's normally the same for all
    the stars in the list.

    The `fwhm` is the full width at half maximum (FWHM) of the stars, in pixels.
    It is a float value, and it is the parameter to the star detection algorithm,
    so it's normally the same for all the stars in the list.
    """

    x: np.ndarray
    y: np.ndarray
    flux: np.ndarray
    threshold: np.ndarray
    fwhm: np.ndarray

    def __len__(self) -> int:
        """Return the number of stars in the list."""
        return len(self.x)

    def __bool__(self) -> bool:
        """Return True if the list is not empty."""
        return len(self) > 0

    def __repr__(self) -> str:
        return f"StarsList(len={len(self)})"

    @classmethod
    def empty(cls) -> "StarsList":
        """Return an empty StarsList."""
        _n = np.array([], dtype=float)
        return cls(x=_n, y=_n, flux=_n, threshold=_n, fwhm=_n)

    def sort(self) -> None:
        """Sort the stars list in-place, by their flux, brightest first."""
        indices = np.argsort(self.flux)[::-1]
        self.x = self.x[indices]
        self.y = self.y[indices]
        self.flux = self.flux[indices]
        self.threshold = self.threshold[indices]
        self.fwhm = self.fwhm[indices]

    def cap(self, max_stars: int) -> None:
        """Cap the number of stars in the list to `max_stars`.

        This is useful to limit the number of stars to a manageable number,
        for example, when displaying them in a plot.

        The number of stars will be capped to `max_stars`, whatever the current stars
        order is. It might make sense to call `sort()` before this method to ensure
        that only the brightest stars are kept in the list.

        Args:
            max_stars: The maximum number of stars to keep in the list.
        """
        if len(self) > max_stars:
            self.x = self.x[:max_stars]
            self.y = self.y[:max_stars]
            self.flux = self.flux[:max_stars]
            self.threshold = self.threshold[:max_stars]
            self.fwhm = self.fwhm[:max_stars]


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
    *,
    size: tuple[int, int] | None = None,
    interactive: bool = False,
) -> go.Figure:
    """Plot an image as a Plotly figure.

    The figure pixel dimensions can be set using the `width` and `height` parameters
    and the image can be displayed either as a static image or as an interactive
    map where each pixel can be hovered over to see its coordinates and value (slower).

    Assumptions:
    - The input array is a 2D grayscale image (single channel).

    Args:
        array: The image array to display. Must be a 2D array (grayscale image).
        size: The size of the figure in pixels, as a tuple (width, height). Optional,
            if not passed, the size will be left up to plotly.
        interactive: If True, the image will be displayed in an interactive Plotly
            figure, where each pixel can be hovered over to see its coordinates and
            value. Optional, defaults to False.

    Returns:
        A Plotly figure object containing the image.
    """
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
    size_kwargs: dict[str, Any] = {"width": size[0], "height": size[1]} if size else {}
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
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0)",
        margin={"l": 0, "r": 0, "t": 30, "b": 0},
        **size_kwargs,
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
) -> StarsList:
    """Find stars in an image array using the DAOStarFinder algorithm.

    This function uses the `photutils` library to detect stars in an array.
    It calculates the background noise using the median absolute deviation (MAD) and
    uses the DAOStarFinder algorithm to find sources in the image.

    The detected stars are returned packaged in the `StarsList` dataclass, which
    contains the x and y pixel coordinates of the stars (relative to the array),
    as well as their fluxes.

    The stars with |roundness| > `max_roundness` are filtered out.
    Also, the stars close to the image border are filtered out, as are the stars within
    (by default) 1.5 * fwhm of other stars.

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
        Stars detected in the image, packaged in a `StarsList` dataclass.
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
        # No stars found, return an empty StarsList
        return StarsList.empty()

    stars_list = StarsList(
        x=sources["xcentroid"].value,
        y=sources["ycentroid"].value,
        flux=sources["flux"].value,
        threshold=np.full(len(sources), threshold),
        fwhm=np.full(len(sources), fwhm),
    )

    if show:
        fig = plot_image(array)
        fig.add_scatter(
            x=stars_list.x,
            y=stars_list.y,
            mode="markers",
            marker={
                "size": stars_list.flux / np.max(stars_list.flux) * 10,
                "color": stars_list.flux,
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

    return stars_list
