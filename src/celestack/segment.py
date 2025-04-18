from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
from astropy.stats import mad_std
from astropy.table import Table
from photutils.detection import DAOStarFinder

import celestack._utils as utils


@dataclass(frozen=True)
class SegmentBox:
    """
    A class representing the rectangle, which will define the segment of a frame.

    The box is defined by its top-left coordinate (x1, y1) and bottom-right coordinate
    (x2, y2). All are integers - indices of the pixel in the image.
    The (x1, y1) coordinates are inclusive, while (x2, y2) coordinates are exclusive.
    """

    x1: int
    y1: int
    x2: int
    y2: int


class Segment:
    """
    A class representing a segment of a frame.

    The segment is defined by its box (x1, y1, x2, y2) and the pixel array.
    The pixel array is a 2D NumPy array representing the pixel values of the segment.

    The segment class defines all sorts of methods which are focused on recognition of
    stars in the segment. The Segment should therefore only be instantiated with the
    part of the image with night sky and stars, as opposed to the foreground or other
    objects, such as dark frames, etc.
    """

    def __init__(
        self, box: SegmentBox, array: np.ndarray, mask: np.ndarray | None = None
    ):
        """
        Initializes the Segment object with a box and an array.

        The array should be a 2D NumPy array in grayscale 8-bit format (uint8).
        The segments are normally instantiated only from the sections of the
        `compressed_array` of the `Frame` objects.

        Args:
            box: The SegmentBox object defining the segment.
            array: The pixel array representing the segment - must be 2D uint8.
            mask: Optional mask array for the segment. Defaults to None. If provided,
                it must have the same shape as the `array` and should be a boolean
                type.
        """
        self.box = box
        self._array = array
        self.mask = mask

        # Define stubs for the properties
        self.stars_table: Table | None = None

    @property
    def array(self) -> np.ndarray:
        """
        Returns the pixel array of the segment. If the mask is assigned, all the masked
        pixels are set to zero.
        """
        array = self._array
        if self.mask is not None:
            array[~self.mask] = 0
        return array

    def find_stars(self, n: int = 50) -> None:
        """
        Finds stars in the segment using the `array` and assigns the result to
        `self.stars_table`.
        """
        # Calculate the background noise using the median absolute deviation (MAD)
        bkg_mad = mad_std(self._array)

        # Find the stars:
        # TODO: I will want to iteratively home in on the correct threshold / fwhm to
        #   get the right number of sources - e.g. 2 or 3 times more than the final
        #   number I want
        daofind = DAOStarFinder(fwhm=3.0, threshold=3 * bkg_mad)
        stars = daofind(self.array)

        # Reject stars that are too non-round:
        MAX_ROUNDNESS = 0.85
        stars = stars[np.abs(stars["roundness2"]) <= MAX_ROUNDNESS]

        # Set the threshold flux to get the right number of stars:
        if len(stars) > n:
            threshold_flux = sorted(stars["flux"])[-n]
            stars = stars[stars["flux"] >= threshold_flux]

        # TODO: Reject all the sources which have another source too close (e.g. within
        #   2*FWHM) - I need this to guard against possible swaps in registering the
        #   stars across frames.

        # Set the stars table:
        self.stars_table = stars

    def plot(self) -> go.Figure:
        """
        Plots the segment using matplotlib.

        If the stars have been found, they are plotted as markers on the image, with
        their sizes and colors set to indicate their flux.

        Returns:
            A Plotly figure object containing the image.
        """
        # Plot the image into a figure:
        fig = utils.plot_image(self.array)  # this will have already applied the mask

        # Add the stars, if they have been found:
        if self.stars_table is not None:
            st = self.stars_table
            fig.add_trace(
                go.Scatter(
                    x=st["xcentroid"],
                    y=st["ycentroid"],
                    mode="markers",
                    marker=dict(
                        size=st["flux"] / np.max(st["flux"]) * 10,  # type: ignore
                        color=st["flux"],
                        colorscale="Viridis",
                    ),
                    name="Stars",
                )
            )

        fig.update_layout(showlegend=True)

        return fig
