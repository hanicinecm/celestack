from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from astropy.stats import mad_std
from photutils.detection import DAOStarFinder
from scipy.spatial import KDTree, distance_matrix

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

    def shift(self, dx: int, dy: int) -> "SegmentBox":
        """
        Shifts the box by the given dx and dy.

        Args:
            dx: The shift in the x direction.
            dy: The shift in the y direction.

        Returns:
            A new SegmentBox object with the shifted coordinates.
        """
        return SegmentBox(self.x1 + dx, self.y1 + dy, self.x2 + dx, self.y2 + dy)


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
                type, with True values indicating the foreground pixels to be masked
                out.
        """
        self.box = box
        self.array = array
        self._mask = mask

        # Define stubs for the properties
        self.stars_fwhm: float = 4.0
        self.stars_thresh: float = 4.0
        self._stars_table_raw: pd.DataFrame | None = None

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

    def find_stars(self, n: float = 6.0, init_thresh: float | None = None) -> None:
        """
        Finds stars in the segment and save the result to `self._stars_table_raw`.

        The stars are found using the DAOStarFinder algorithm from the photutils
        library, and the method intelligently adjusts the algorithm's threshold to find
        the specified number of stars given by the target star density `n` in the units
        of "stars per 10,000 sky pixels".

        The final threshold which lead to the number of stars closest to `n` is
        then stored in `self.stars_thresh` and can be used as a `init_thresh`
        for the `find_stars` call in the neighboring segments (either spatially or in
        the stack).

        All stars that are too close to other stars, or to an edge, are rejected.
        This happens as the very last step, therefore the final number of stars may be
        (and most likely will be) less than what correspond to the target star density.

        The stars table is saved with the stars sorted by their flux (brightest first)
        and with unique IDs as index.
        See the `stars_table` property for more details.

        Args:
            n: The target star density in [stars/10,000 pixels]. Optional,
                if not passed, a sensible default is provided.
            init_thresh: The starting threshold for the star finding algorithm, in
                the units of the background noise. Optional, if not provided, the
                `self.stars_thresh` is used.
        """
        if init_thresh is None:
            init_thresh = self.stars_thresh
        fwhm = self.stars_fwhm

        # Calculate the background noise using the median absolute deviation (MAD)
        bkg_mad = mad_std(self.array)

        # Find the sources:
        # Start with the `init_thresh * bkg_mad` threshold and iterate the threshold
        # until we find close to `n` stars:
        sources = None
        delta_n = None
        thresh = init_thresh
        thresh_incr = 0.2
        target_n = np.sum(~self.mask) / 10_000 * n  # target number of stars
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

        # Store the sources and the final threshold in the instance:
        self._stars_table_raw = sources.reset_index(drop=True).copy()
        self.stars_thresh = thresh

    def cap_stars(self, n: int) -> None:
        """
        Caps the number of stars in the segment to a maximum of `n`.

        If the number of stars found in the segment is greater than `n`, then only
        `n` stars with the highest flux are kept in the raw stars table and the rest is
        discarded.

        Args:
            n: The maximum number of stars to keep in the segment.
        """
        if self._stars_table_raw is None:
            raise ValueError("Stars have not been found yet.")
        if len(self._stars_table_raw) > n:
            self._stars_table_raw = self._stars_table_raw.iloc[:n].copy()

    def plot(self) -> go.Figure:
        """
        Plots the segment using matplotlib.

        If the stars have been found, they are plotted as markers on the image,
        with their sizes and colors set to indicate their flux.

        Returns:
            A Plotly figure object containing the image.
        """
        # Plot the image into a figure:
        fig = utils.plot_image(self.array)  # this will have already applied the mask

        # Add the stars, if they have been found:
        if self._stars_table_raw is not None:
            st = self._stars_table_raw
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

    @property
    def stars_table(self) -> pd.DataFrame:
        """
        Returns a table with all the stars found in this segment.

        Each line in the table corresponds to a star found in the segment and
        has a unique ID (the table index).

        The table contains the following columns:
            - x: The x coordinate of the star in the original image coordinate system.
            - y: The y coordinate of the star in the original image coordinate system.
            - x_segment: The x coordinate of the star in the segment coordinate system.
            - y_segment: The y coordinate of the star in the segment coordinate system.
            - flux: The flux (brightness) of the star.

        The stars are sorted by their flux (brightness) in descending order (brightest
        first).

        Returns:
            A pandas DataFrame containing the stars positions and brightness.

        Raises:
            ValueError: If the stars finding method has not been called yet.
        """
        if self._stars_table_raw is None:
            raise ValueError("Stars have not been found yet.")
        stars_table = self._stars_table_raw.copy()

        # Only keep the relevant columns:
        stars_table = stars_table[["xcentroid", "ycentroid", "flux"]]

        # Rename the columns:
        stars_table.rename(
            columns={
                "xcentroid": "x_segment",
                "ycentroid": "y_segment",
            },
            inplace=True,
        )

        # Add the x and y coordinates in the original image coordinate system:
        x = stars_table["x_segment"] + self.box.x1
        x.name = "x"
        y = stars_table["y_segment"] + self.box.y1
        y.name = "y"
        stars_table = pd.concat([x, y, stars_table], axis=1)

        return stars_table

    def match_to(self, template: "Segment") -> None:
        """
        Match the stars to the (presumably the same) stars in the template segment.

        As result, the stars table in this segment will be re-indexed and capped, so
        that the stars sharing the same ID in both segments are presumably the same.

        The stars table of this segment is changed in place, while the template
        segment remains unchanged.

        The matching is done using the nearest-neighbor search in the space of the
        stars coordinates in their respective segments.

        After the nearest-neighbor search, the matches are also filtered to remove
        any outliers.

        Args:
            template: The template segment to match the stars to. The template segment
                will not be modified. The segment must have been already processed with
                `find_stars` method.
        """
        # Get the stars tables for both segments
        df_template = template.stars_table
        df_self = self.stars_table

        # The feature space for the nearest-neighbor search is the x and y coordinates
        # of the stars in their respective segments:
        feat_template = df_template[["x_segment", "y_segment"]].to_numpy()
        feat_self = df_self[["x_segment", "y_segment"]].to_numpy()

        # Create KDTree for self and query it with the template segment
        kdtree = KDTree(feat_self)
        distances, indices = kdtree.query(feat_template, k=1)

        # Create a DataFrame with dx/dy in the image coordinates and distance in the
        # feature space:
        new_index: np.ndarray = df_self.index.to_numpy()[indices]
        dx = df_self.loc[new_index, "x"].to_numpy() - df_template["x"].to_numpy()
        dy = df_self.loc[new_index, "y"].to_numpy() - df_template["y"].to_numpy()
        out_df = pd.DataFrame(
            {
                "new_index": new_index,
                "distance": distances,
                "dx": dx,
                "dy": dy,
            },
            index=df_template.index,
        )

        # Filter out the outliers in dx, dy, and distance
        for col in ["dx", "dy", "distance"]:
            median = out_df[col].median()
            mad = mad_std(out_df[col])
            threshold = 3 * mad
            out_df = out_df[(out_df[col] - median).abs() < threshold]

        # Remove any duplicates in the new index (this would mean that
        # more than one star in the template segment was matched to the same star in
        # this segment):
        out_df = out_df[~out_df["new_index"].duplicated(keep=False)]

        # Re-index the stars table in this instance, to match the template segment
        # (this will also cap the number of stars in this segment):
        assert self._stars_table_raw is not None  # type checker
        self._stars_table_raw = self._stars_table_raw.loc[out_df["new_index"]].copy()
        self._stars_table_raw.index = out_df.index
        self._stars_table_raw = self._stars_table_raw.sort_values(
            by="flux", ascending=False
        )

    def get_shift_from(self, template: "Segment") -> tuple[float, float]:
        """
        Evaluates the median shift between the stars of this segment and the matched
        stars in the template segment.

        The shift is evaluated as median shift of the stars with the same IDs in both
        segments, so it will only make sense, if this segment has already been matched
        to the template segment by `match_to` method.

        The shift is in the original image coordinates.

        Args:
            template: A template segment, to which this segment has been matched to.

        Returns:
            A tuple containing the average dx and dy values, defining the median shift
            of the matched stars in the original image coordinates.
        """
        # Get the stars tables for both segments
        df_template = template.stars_table
        df_self = self.stars_table

        stars_ids = df_self.index.intersection(df_template.index)
        if len(stars_ids) == 0:
            raise ValueError("No stars matched to the template segment.")

        # Get the dx and dy values for the matched stars
        dx = df_self.loc[stars_ids, "x"] - df_template.loc[stars_ids, "x"]
        dy = df_self.loc[stars_ids, "y"] - df_template.loc[stars_ids, "y"]

        # Return the median dx and dy values
        return float(dx.median()), float(dy.median())
