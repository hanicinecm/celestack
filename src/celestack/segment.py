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
            array: The pixel array representing the segment - must be 2D uint8 and
                must have the shape consistent with the box.
            mask: Optional mask array for the segment. Defaults to None. If provided,
                it must have the same shape as the `array` and should be a boolean
                type, with True values indicating the foreground pixels to be masked
                out.
        """
        self.box = box
        self.array = array
        self._mask = mask

    @property
    def mask(self) -> np.ndarray:
        """
        Returns the mask of the segment.

        If no mask was provided during initialization, a default mask is created
        with all pixels set to True (no masking).

        Returns:
            A boolean NumPy array representing the mask of the segment.
        """
        if self._mask is None:
            # Create a default mask with all pixels set to True (no masking)
            self._mask = np.ones_like(self.array, dtype=bool)
        return self._mask

    def find_stars(
        self,
        n: int,
        fwhm: float,
        init_thresh: float = 4.0,
    ) -> pd.DataFrame:
        """
        Finds stars in the segment and return them in a pandas DataFrame table.

        The stars are found using the DAOStarFinder algorithm from the photutils
        library, and the method intelligently adjusts the algorithm's threshold to find
        the specified number of stars given by the `n` parameter.

        The final threshold which lead to the number of stars needed for the target `n`
        is recorded in the returned dataframe, as well as the fwhm.
        These can be used as a `init_thresh` and `fwhm` parameters for the next call of
        the `find_stars` in the neighboring segments (either spatially or in the stack).

        All stars that are too close to other stars, or to an edge, are rejected.
        This is factored in when finding the appropriate threshold. However, in some
        rare cases, the final number of stars in the output table might be a little
        less than the target `n`.

        The stars table is saved with the stars sorted by their flux (brightest first)
        and with unique IDs as index.

        Args:
            n: The number of stars we want to find in the segment.
            fwhm: The full width at half maximum (FWHM) of the stars in [px].
                This is one of the parameters of the star finding algorithm and might
                have to be adjusted for each project.
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
                - fwhm: The FWHM setting for the star finding algorithm (in pixels).
                - threshold: The threshold used for finding the stars (in units of
                    standard deviations of the background noise).
        """
        # Calculate the background noise using the median absolute deviation (MAD)
        bkg_mad = mad_std(self.array)

        # Find the sources:
        # Start with the `init_thresh * bkg_mad` threshold and iterate the threshold
        # until we find close to required number of stars:
        _sources = None
        delta_n = None
        thresh = init_thresh
        thresh_incr = 0.2
        target_n = 2 * n  # target number of stars (will be capped later)
        n_iters_stagnating = 0
        while True:
            # find the stars:
            daofind = DAOStarFinder(fwhm=fwhm, threshold=thresh * bkg_mad)
            new_sources = daofind(data=self.array, mask=self._mask)
            # how far are we from the target number of sources?
            new_delta_n = len(new_sources) - target_n
            if delta_n is not None and abs(new_delta_n) > abs(delta_n):
                # we're further away from target than in the last iteration - stop...
                break
            elif delta_n is not None and abs(new_delta_n) == abs(delta_n):
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

        # Convert the optimized sources table to a pandas DataFrame and sort by flux:
        assert _sources is not None
        sources: pd.DataFrame = _sources.to_pandas().sort_values(
            by="flux", ascending=False
        )
        assert isinstance(sources, pd.DataFrame), "Just for the type checker"

        # Get rid of all the uninsteresing columns in the sources table:
        sources.drop(columns=["id", "npix"], inplace=True)

        # Reject all the sources which have another star too close:
        star_margin = 1.5 * fwhm
        dist_matrix = distance_matrix(
            sources[["xcentroid", "ycentroid"]].values,
            sources[["xcentroid", "ycentroid"]].values,
        )
        np.fill_diagonal(dist_matrix, np.inf)
        too_close_iloc = np.where(dist_matrix.min(axis=0) < star_margin)[0]
        too_close_ids = sources.index[too_close_iloc]
        sources = sources[~sources.index.isin(too_close_ids)]

        # Reject all the sources which are too close to the edges of the segment:
        edge_margin = 1.5 * fwhm
        edge_mask: pd.DataFrame = (
            (sources["xcentroid"] < edge_margin)
            | (sources["xcentroid"] > self.array.shape[1] - edge_margin)
            | (sources["ycentroid"] < edge_margin)
            | (sources["ycentroid"] > self.array.shape[0] - edge_margin)
        )
        sources = sources[~edge_mask]

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

        return stars_table[["x", "y", "flux", "fwhm", "threshold"]].reset_index(
            drop=True
        )

    def plot(self) -> go.Figure:
        """
        Plot the segment.

        Returns:
            A Plotly figure object containing the image.
        """
        fig = utils.plot_image(self.array)
        return fig
