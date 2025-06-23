"""Stars table module."""

import functools
from collections.abc import Callable
from pathlib import Path
from types import MappingProxyType
from typing import cast

import numpy as np
import plotly.graph_objects as go
import polars as pl

from celestack._utils import StarsList


class StarsTable:
    """A class representing a table of stars registered and aligned across frames."""

    SCHEMA = MappingProxyType(
        {
            "frame": pl.Categorical,
            "id": pl.Int64,
            "t": pl.Float64,
            "x": pl.Float64,
            "y": pl.Float64,
            "flux": pl.Float64,
            "threshold": pl.Float64,
            "fwhm": pl.Float64,
        }
    )

    def __init__(
        self,
        frames_names: list[str],
        frames_times: list[float],
        ref_frame_name: str,
        ref_stars_list: StarsList,
    ) -> None:
        """Initialize the stars table object.

        The stars table is a data frame with each row representing a single star within
        a single frame.
        Various columns exist, populated with data useful not only for the stars
        registration and trasformation calculations, but also for various diagnostics
        and visualizations.

        The initialization method will seed the stars table with empty rows for all
        the frames in the stack, apart from the reference frame, which will have the
        stars from the `ref_stars_list` provided.

        Assumptions:
        - The frames_names and frames_times lists are of the same length.
        - The ref_frame_name is in the frames_names list.
        - The time belonging to the ref_frame_name is 0.0.

        Args:
            frames_names: The names of the frames in the stack.
            frames_times: The times of the frames in the stack, relative to the
                reference frame.
            ref_frame_name: The name of the reference frame (the frame for which the
                stars list is provided).
            ref_stars_list: The list of all the stars found in the reference frame.
        """
        # Seed the data with the frame names, ids and times:
        n_frames = len(frames_names)
        n_stars = len(ref_stars_list)
        data = {
            "frame": np.repeat(frames_names, n_stars),
            "id": np.tile(ref_stars_list.ids, n_frames),
            "t": np.repeat(frames_times, n_stars),
        }

        # Fill the rest of the columns with NaNs:
        n_rows = n_frames * n_stars
        nan_cols = [c for c in self.SCHEMA if c not in data]
        for col in nan_cols:
            data[col] = np.full(n_rows, np.nan)

        # Seed the data frame with the empty records:
        self._df = pl.DataFrame(data, schema=self.SCHEMA)

        # Fill in the values for the reference frame from the provided stars list:
        mask = pl.col("frame") == ref_frame_name
        ref_vals = {
            "x": ref_stars_list.x,
            "y": ref_stars_list.y,
            "flux": ref_stars_list.flux,
            "threshold": ref_stars_list.threshold,
            "fwhm": ref_stars_list.fwhm,
        }
        self._df = self._df.with_columns(
            [
                pl.when(mask).then(values).otherwise(pl.col(col)).alias(col)
                for col, values in ref_vals.items()
            ]
        )

    @classmethod
    def from_state(cls, stars_table_path: str | Path) -> "StarsTable":
        """Load the stars table from a CSV file.

        Assumptions:
        - The stars_df_path is an existing parquet file path.
        """
        # Create the instance of the class:
        instance = cls.__new__(cls)

        # Load the data frame from the parquet file:
        _df = pl.read_parquet(stars_table_path, schema=instance.SCHEMA)
        instance._df = _df  # noqa: SLF001

        # Return the instance:
        return instance

    def dump_state(self, stars_table_path: str | Path) -> None:
        """Save the stars table state to a file.

        Assumptions:
        - The stars_df_path is a parquet file path.
        """
        self._df.write_parquet(stars_table_path)

    def plot_to_figure(self, fig: go.Figure, stars_frame: str) -> None:
        """Plot the stars in the table into an existing figure.

        Each star present in the table for the given frame name is plotted as a scatter
        point, with its size and color determined by the star's brightness.

        Additionally, the star trails are plotted for each star, as a line of the same
        color as the star, connecting the star positions across all the frames.

        The stars and the trails are plotted into an already existing figure - this
        method will change the figure in place.

        Assumptions:
        - The stars_frame is a valid frame name, which exists in the stars table.
        - The stars_frame already has some stars registered in it. Either the reference
            frame is chosen, or the stars were already propagated across the frames.

        Args:
            fig: The figure to plot the stars into.
            stars_frame: The name of the frame to plot the stars for.
                The stars in the table for this frame will be plotted.
        """
        raise NotImplementedError

    def plot_star_trail(self, star_id: int) -> go.Figure:
        """Plot the trail of a star across all frames.

        The method will create a new figure with the fragment of all the frames across
        the stack, which contain the star with the given ID, stacked in the "lightest"
        mode.
        This way, the star data from all the frames are visible in the same figure.

        Additionally, scatter points are plotted for the star positions registered
        in each frame, with the size determining the star's brightness.

        Assumptions:
        - The star_id is a valid ID of a star in the stars table.

        Args:
            star_id: The ID of the star to plot the trail for. Must be a valid star ID
                existing in the stars table.

        Returns:
            A Plotly figure with the star trail plotted.
        """
        raise NotImplementedError


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
        i_known = cast("int", i_known)  # index assumed unique
        # Is the `frame_name` the closest neighbor of the known frame?
        i_frame = star_coordinates.index.get_loc(frame_name)
        i_frame = cast("int", i_frame)  # index assumed unique
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
