from dataclasses import dataclass

import numpy as np
import pandas as pd
from astropy.stats import mad_std
from photutils.detection import DAOStarFinder
from scipy.spatial import distance_matrix


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
        target_density: float = 6.0,
        init_thresh: float = 4.0,
        fwhm: float = 4.0,
        cap_to: int | None = None,
    ) -> None:
        """
        Finds stars in the segment and return them in a pandas DataFrame table.

        The stars are found using the DAOStarFinder algorithm from the photutils
        library, and the method intelligently adjusts the algorithm's threshold to find
        the specified number of stars given by the target star density in the units
        of "stars per 10,000 sky pixels".

        The final threshold which lead to the number of stars closest to the target
        density is recorded in the returned dataframe, as well as the fwhm.
        These can be used as a `init_thresh` and `fwhm` parameters for the next call of
        the `find_stars` in the neighboring segments (either spatially or in the stack).

        All stars that are too close to other stars, or to an edge, are rejected.
        This happens as the very last step, therefore the final stars density may be
        (and most likely will be) less than what the target star density.

        The stars table is saved with the stars sorted by their flux (brightest first)
        and with unique IDs as index.

        Args:
            target_density: The target star density in [stars/10,000 pixels].
                Optional, if not passed, a sensible default is provided.
            init_thresh: The starting threshold for the star finding algorithm, in the
                number of standard deviations of the typical bacground noise.
                This is treated only as an initial guess, and the algorithm will adjust
                this to get close to the target star density.
                Optional, if not passed, a sensible default is provided.
            fwhm: The full width at half maximum (FWHM) of the stars in [px].
                Optional, if not passed, a sensible default is provided.
            cap_to: A cap for the number of stars. If provided, and if the number of
                stars found is greater than this value, the stars table will be capped
                to this value (the brightest stars will remain).
                Optional, if not passed, no capping is done.
        """
        # Calculate the background noise using the median absolute deviation (MAD)
        bkg_mad = mad_std(self.array)

        # Find the sources:
        # Start with the `init_thresh * bkg_mad` threshold and iterate the threshold
        # until we find close to `n` stars:
        sources = None
        delta_n = None
        thresh = init_thresh
        thresh_incr = 0.2
        target_n = (
            np.sum(~self.mask) / 10_000 * target_density
        )  # target number of stars
        while True:
            # find the stars:
            daofind = DAOStarFinder(fwhm=fwhm, threshold=thresh * bkg_mad)
            new_sources = daofind(data=self.array, mask=self._mask)
            # how far are we from the target number of sources?
            new_delta_n = len(new_sources) - target_n
            if delta_n is not None and abs(new_delta_n) >= abs(delta_n):
                # we're further away from target than in the last iteration - stop...
                break
            delta_n = new_delta_n
            sources = new_sources
            # adjust the threshold for the next iteration:
            thresh += thresh_incr if new_delta_n > 0 else -thresh_incr

        # Convert the optimized sources table to a pandas DataFrame and sort by flux:
        assert sources is not None
        sources = sources.to_pandas().sort_values(by="flux", ascending=False)

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
        edge_mask = (
            (sources["xcentroid"] < edge_margin)
            | (sources["xcentroid"] > self.array.shape[1] - edge_margin)
            | (sources["ycentroid"] < edge_margin)
            | (sources["ycentroid"] > self.array.shape[0] - edge_margin)
        )
        sources = sources[~edge_mask]

        # TODO: Cap the number of stars to `cap_to` if provided
        # TODO: Turn this into a final table (x, y, x_seg, y_seg, flux, fwhm, threshold)

    # def cap_stars(self, n: int) -> None:
    #     """
    #     Caps the number of stars in the segment to a maximum of `n`.

    #     If the number of stars found in the segment is greater than `n`, then only
    #     `n` stars with the highest flux are kept in the raw stars table and the rest is
    #     discarded.

    #     Args:
    #         n: The maximum number of stars to keep in the segment.
    #     """
    #     if self._stars_table_raw is None:
    #         raise ValueError("Stars have not been found yet.")
    #     if len(self._stars_table_raw) > n:
    #         self._stars_table_raw = self._stars_table_raw.iloc[:n].copy()

    # def plot(self) -> go.Figure:
    #     """
    #     Plots the segment using matplotlib.

    #     If the stars have been found, they are plotted as markers on the image,
    #     with their sizes and colors set to indicate their flux.

    #     Returns:
    #         A Plotly figure object containing the image.
    #     """
    #     # Plot the image into a figure:
    #     fig = utils.plot_image(self.array)  # this will have already applied the mask

    #     # Add the stars, if they have been found:
    #     if self._stars_table_raw is not None:
    #         st = self._stars_table_raw
    #         fig.add_trace(
    #             go.Scatter(
    #                 x=st["xcentroid"],
    #                 y=st["ycentroid"],
    #                 mode="markers",
    #                 marker=dict(
    #                     size=st["flux"] / np.max(st["flux"]) * 10,  # type: ignore
    #                     color=st["flux"],
    #                     colorscale="Viridis",
    #                 ),
    #                 name="Stars",
    #             )
    #         )

    #     fig.update_layout(showlegend=True)

    #     return fig

    # @property
    # def stars_table(self) -> pd.DataFrame:
    #     """
    #     Returns a table with all the stars found in this segment.

    #     Each line in the table corresponds to a star found in the segment and
    #     has a unique ID (the table index).

    #     The table contains the following columns:
    #         - x: The x coordinate of the star in the original image coordinate system.
    #         - y: The y coordinate of the star in the original image coordinate system.
    #         - x_segment: The x coordinate of the star in the segment coordinate system.
    #         - y_segment: The y coordinate of the star in the segment coordinate system.
    #         - flux: The flux (brightness) of the star.

    #     The stars are sorted by their flux (brightness) in descending order (brightest
    #     first).

    #     Returns:
    #         A pandas DataFrame containing the stars positions and brightness.

    #     Raises:
    #         ValueError: If the stars finding method has not been called yet.
    #     """
    #     if self._stars_table_raw is None:
    #         raise ValueError("Stars have not been found yet.")
    #     stars_table = self._stars_table_raw.copy()

    #     # Only keep the relevant columns:
    #     stars_table = stars_table[["xcentroid", "ycentroid", "flux"]]

    #     # Rename the columns:
    #     stars_table.rename(
    #         columns={
    #             "xcentroid": "x_segment",
    #             "ycentroid": "y_segment",
    #         },
    #         inplace=True,
    #     )

    #     # Add the x and y coordinates in the original image coordinate system:
    #     x = stars_table["x_segment"] + self.box.x1
    #     x.name = "x"
    #     y = stars_table["y_segment"] + self.box.y1
    #     y.name = "y"
    #     stars_table = pd.concat([x, y, stars_table], axis=1)

    #     return stars_table
