"""Stars table module."""

import functools
from collections.abc import Callable
from os import PathLike
from typing import cast

import numpy as np
import pandas as pd


class StarsTable:
    """A class representing a table of stars registered and aligned across frames."""

    @classmethod
    def from_csv(cls, stars_table_path: PathLike) -> "StarsTable":
        """Load the stars table from a CSV file."""
        raise NotImplementedError

    def to_csv(self, stars_table_path: PathLike) -> None:
        """Save the stars table to a CSV file."""
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
