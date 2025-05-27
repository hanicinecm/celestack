"""A module defining the functionality for representing a segment of a frame."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import celestack._utils as utils


@dataclass(frozen=True)
class SegmentBox:
    """A class representing the rectangle, which will define the segment of a frame.

    The box is defined by its top-left coordinate (x1, y1) and bottom-right coordinate
    (x2, y2). All are integers - indices of the pixel in the image.
    The (x1, y1) coordinates are inclusive, while (x2, y2) coordinates are exclusive.
    """

    x1: int
    y1: int
    x2: int
    y2: int


class Segment:
    """A class representing a segment of a frame.

    The segment is defined by its box (x1, y1, x2, y2) and the pixel array.
    The pixel array is a 2D NumPy array representing the pixel values of the segment.

    The segment class defines all sorts of methods which are focused on recognition of
    stars in the segment. The Segment should therefore only be instantiated with the
    part of the image with night sky and stars, as opposed to the foreground or other
    objects, such as dark frames, etc.
    """

    def __init__(
        self,
        box: SegmentBox,
        array: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> None:
        """Initialize the Segment object with a box and an array.

        The array should be a 2D NumPy array in grayscale 8-bit format (uint8).
        The segments are normally instantiated only from the sections of the
        `compressed_array` of the `Frame` objects.

        Args:
            box: The SegmentBox object defining the segment.
            array: The pixel array representing the segment - must be 2D uint8 and
                must have the shape consistent with the box.
            mask: Optional mask array for the segment. Defaults to None. If provided,
                it must have the same shape as the `array` and should be a boolean
                type, with True values indicating the foreground pixels to be masked
                out.
        """
        self.box = box
        self.array = array
        self.mask = mask

    def find_stars(
        self,
        density: float,
        fwhm: float,
        max_roundness: float,
        min_separation: float,
        init_thresh: float = 4.0,
    ) -> pd.DataFrame:
        """Find stars in the segment and return them in a pandas DataFrame table.

        The stars are found using the DAOStarFinder algorithm from the photutils
        library, and the method intelligently adjusts the algorithm's threshold to find
        the specified final stars density.
        The threshold will be optimized to find double the number of stars requested
        and the stars will then be capped to the `n` brightest ones.

        The final threshold which lead to the number of stars needed for the target N
        is recorded in the returned dataframe.
        It can be used as the `init_thresh` parameter for the next call of
        the `find_stars` in the neighboring segments (either spatially or in the stack).

        All stars that are too close to other stars or to to an edge, are rejected.

        The stars table is saved with the stars sorted by their flux (brightest first)
        and with unique IDs as index.

        TODO: Return a polars DataFrame instead of pandas DataFrame.

        Args:
            density: The final density of the found stars in the segment, in
                stars per 10,000 sky pixels.
            fwhm: The full width at half maximum (FWHM) of the stars in [px].
                This is one of the parameters of the star finding algorithm and might
                have to be adjusted for each project.
            max_roundness: The maximum roundness of the stars. This is fed to the
                DAOStarFinder algorithm - all the stars with |roundness| > max_roundness
                will be rejected by DAOStarFinder.
            min_separation: The minimum separation between stars, in the units of fwhm.
                This is fed to the DAOStarFinder algorithm - all the stars with
                separation < min_separation * fwhm will be rejected by DAOStarFinder.
            init_thresh: The starting threshold for the star finding algorithm, in the
                number of standard deviations of the typical bacground noise.
                This is treated only as an initial guess, and the algorithm will adjust
                this to get to the target `n`.
                Optional, if not passed, a sensible default is provided.

        Returns:
            A pandas DataFrame containing the stars found in the segment.
            The DataFrame contains the following columns:
                - x: The x coordinate of the star in the original image (in pixels).
                - y: The y coordinate of the star in the original image (in pixels).
                - flux: The flux of the star.
                - threshold: The threshold used for finding the stars (in units of
                    standard deviations of the background noise).
                - fwhm: The FWHM parameter used for the star finding algo (in pixels).
        """
        # Calculate the number of stars to find:
        n_pixels = self.array.size
        if self.mask is not None:
            n_pixels -= np.sum(self.mask)
        n = round(n_pixels * density / 10000)

        # Find the sources:
        # Start with the `init_thresh * bkg_mad` threshold and iterate the threshold
        # until we find close to required number of stars:
        _sources = None
        delta_n = None
        thresh = init_thresh
        thresh_incr = 0.2
        target_n = 2 * n  # target number of stars (will be capped later)
        n_iters_stagnating = 0
        # TODO: Optimize this with a binary search or something similar
        while True:
            # find the stars:
            new_sources = utils.find_stars(
                array=self.array,
                threshold=thresh,
                fwhm=fwhm,
                max_roundness=max_roundness,
                min_separation=min_separation,
                mask=self.mask,
            )
            # how far are we from the target number of sources?
            new_delta_n = len(new_sources) - target_n
            if delta_n is not None and abs(new_delta_n) > abs(delta_n):
                # we're further away from target than in the last iteration - stop...
                break
            if delta_n is not None and abs(new_delta_n) == abs(delta_n):
                # we're not getting closer to the target, but we don't want to stop
                # right away...
                n_iters_stagnating += 1
                if n_iters_stagnating > 3:
                    # ... so we stop after 3 iterations of stagnation
                    break
            else:
                n_iters_stagnating = 0

            delta_n = new_delta_n
            _sources = new_sources
            # adjust the threshold for the next iteration:
            thresh += thresh_incr if new_delta_n > 0 else -thresh_incr

        # If after all this we have no sources, return an empty DataFrame:
        if _sources is None or not len(_sources):
            return pd.DataFrame(columns=["x", "y", "flux", "fwhm", "threshold"])

        # Sort by flux:
        sources = _sources.sort_values(by="flux", ascending=False)

        # Cap the number of stars to `n`:
        if len(sources) > n:
            # The sources are already sorted by flux
            sources = sources.iloc[:n]

        # Turn this into a final table (x, y, x_seg, y_seg, flux, fwhm, threshold)
        stars_table: pd.DataFrame = sources[["xcentroid", "ycentroid", "flux"]].copy()
        stars_table["x"] = stars_table["xcentroid"] + self.box.x1
        stars_table["y"] = stars_table["ycentroid"] + self.box.y1
        stars_table["fwhm"] = fwhm
        stars_table["threshold"] = thresh

        return stars_table[["x", "y", "flux", "threshold", "fwhm"]].reset_index(
            drop=True,
        )

    def plot(self) -> go.Figure:
        """Plot the segment.

        Returns:
            A Plotly figure object containing the image.
        """
        return utils.plot_image(self.array, interactive=True)
