from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from astropy.stats import mad_std
from photutils.detection import DAOStarFinder
from scipy.spatial import distance_matrix

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
        self.stars_table: pd.DataFrame | None = None

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

    def find_stars(
        self, n: int = 50, fwhm: float = 4.0, start_threshold: float = 4.0
    ) -> float:
        """
        Finds stars in the segment and save the result to `self.stars_table`.

        The stars are found using the DAOStarFinder algorithm from the photutils
        library, and the method intelligently adjusts the algorithm's threshold to find
        the specified number of stars `n`.

        Non-round stars are rejected from the table.
        Also, stars that are too close to other stars are rejected - this happens as
        the very last step, therefore the final number of stars may be (and most likely
        will be) less than `n`.

        Args:
            n: The target number of stars to find. Defaults to 50.
            fwhm: The full width at half maximum (FWHM) of the stars. Defaults to 4.0.
            start_threshold: The starting threshold for the star finding algorithm, in
                the units of the background noise. Defaults to 4.0. This is the value
                which is iteratively adjusted to find the target number of stars.

        Returns:
            The final threshold used for the star finding algorithm.
        """
        # Calculate the background noise using the median absolute deviation (MAD)
        bkg_mad = mad_std(self._array)

        # Find the stars:
        # Start with the `start_threshold * bkg_mad` threshold and iterate the threshold
        # until we find close to `2n` stars:
        stars = None
        target_n = 2 * n
        delta_n = None
        thresh = start_threshold
        thresh_incr = 0.2
        while True:
            # find the stars:
            daofind = DAOStarFinder(fwhm=fwhm, threshold=thresh * bkg_mad)
            new_stars = daofind(self.array)
            # how far are we from the target number of stars?
            new_delta_n = len(new_stars) - target_n
            if delta_n is not None and abs(new_delta_n) >= abs(delta_n):
                # we're further away from target than in the last iteration - stop...
                break
            delta_n = new_delta_n
            stars = new_stars
            # adjust the threshold for the next iteration:
            thresh += thresh_incr if new_delta_n > 0 else -thresh_incr

        # convert the optimized stars table to a pandas DataFrame:
        assert stars is not None
        stars = stars.to_pandas().set_index("id", drop=True)

        # Get rid of all the uninsteresing columns in the stars table:
        stars.drop(columns=["npix", "mag", "daofind_mag"], inplace=True)

        # Reject stars that are too non-round:
        MAX_ROUNDNESS = 0.85
        stars = stars[stars.roundness2.abs() <= MAX_ROUNDNESS]

        # Set the threshold flux to get exactly `n` stars:
        distance_thresh = 2 * fwhm
        if len(stars) > n:
            threshold_flux = sorted(stars["flux"])[-n]
            stars = stars[stars.flux >= threshold_flux]

        # Reject all the stars which have another star too close:
        dist_matrix = distance_matrix(
            stars[["xcentroid", "ycentroid"]].values,
            stars[["xcentroid", "ycentroid"]].values,
        )
        np.fill_diagonal(dist_matrix, np.inf)
        too_close_iloc = np.where(dist_matrix.min(axis=0) < distance_thresh)[0]
        too_close_ids = stars.index[too_close_iloc]
        stars = stars[~stars.index.isin(too_close_ids)]

        # Set the stars table:
        self.stars_table = stars

        return thresh

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
