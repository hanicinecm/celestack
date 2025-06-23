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
    
    def propagate_a_star(self, star_id: int, *, dry: bool = False) -> pl.DataFrame:
        """Propagate a single star from the reference frame to the whole stack.

        The method will try to match the star identified in the reference frame under
        the given `star_id` to the same star in all other frames in the stack.

        By default, the method will change the state of the StarsTable by populating
        the data for the given star across all the light frames, if it can be mapped
        to the reference frame star.

        The method will return a *star trail* data frame, which contains the data for
        the given star across all the light frames.
        
        If the `dry` flag is set to True, the method will not change the state of the
        StarsTable, but will return the same star trail data frame and it will show
        some diagnostics data and plots.
        This is useful for debugging and testing purposes, as it allows to inspect
        the star trail propagation procedure without changing the state of the table.

        Assumptions:
        - The `star_id` is a valid ID of a star, existing in the `id` column of
            the StarsTable's data frame.
        - The star with the given `star_id` has not yet been propagated to the
            frames other than the reference frame.

        Args:
            star_id: The ID of the star to propagate. This must be a valid ID from the
                `stars_table` in the stack, which is only known in the reference frame.
            dry: If True, the method will not change the state of the StarsTable, and
                it will show the star trail plot with some diagnostics graphs.

        Returns:
            TODO: Revise the columns in the returned data frame
            A DataFrame containing the star trail, with the following columns:
                - t: The time of capture of the frame, relative to the reference frame.
                - x_guess: The rough prediction of the x coordinate (in pixels).
                - y_guess: The rough prediction of the y coordinate (in pixels).
                - x: The x coordinate of the star in the frame (in pixels).
                - y: The y coordinate of the star in the frame (in pixels).
                - flux: The flux of the star in the frame.
                - threshold: The threshold used for the star finding in the frame.
                - fwhm: The full width at half maximum of the star in the frame.
                - guess_error: The error of the rough prediction of the star position.
                - guess_model_size: The number of samples used to calculate the
                    rough prediction of the star position.
                - in_sky: A boolean indicating if the star is in the sky. If False, by
                    definition the star will have rest of the columns as NaN.
        """
        raise NotImplementedError  # TODO: Implement the star propagation logic
        # # Start with the sanity checks:
        # if self.stars_table is None:
        #     msg = "Stars have not yet been detected in the reference frame."
        #     raise ValueError(msg)
        # if self.ref_frame is None:
        #     msg = "Reference frame has not yet been set."
        #     raise ValueError(msg)
        # star_data = self.stars_table[
        #     (self.stars_table["id"] == star_id)
        #     & (self.stars_table["frame"] == self.ref_frame.name)
        # ]
        # if len(star_data) != 1:
        #     msg = f"The star {star_id} must have already been found in the ref frame."
        #     raise ValueError(msg)

        # # Initialize the star trail table:
        # star_trail = pd.DataFrame(
        #     index=list(self.light_frames),
        #     columns=[
        #         "t",
        #         "x_guess",
        #         "y_guess",
        #         "x",
        #         "y",
        #         "flux",
        #         "threshold",
        #         "fwhm",
        #         "guess_error",
        #         "guess_model_size",
        #     ],
        #     dtype=float,
        # )
        # star_trail["in_sky"] = True
        # star_trail["t"] = self.t_series
        # star_trail = star_trail.sort_values(by="t", key=lambda x: x.abs())

        # # Fill in the star position in the reference frame:
        # star_trail.loc[self.ref_frame.name, ["x", "y", "flux", "threshold", "fwhm"]] = (
        #     star_data[["x", "y", "flux", "threshold", "fwhm"]].to_numpy()[0]
        # )
        # # Initialize the parameters for the star finder:
        # threshold: float = star_data.threshold.iloc[0]
        # fwhm: float = star_data.fwhm.iloc[0]
        # max_roundness = self.star_finder_params["max_roundness"]
        # min_separation = self.star_finder_params["min_separation"]
        # if not isinstance(max_roundness, float) or not isinstance(
        #     min_separation, float
        # ):
        #     msg = "The max_roundness and min_separation must be floats."
        #     raise TypeError(msg)

        # # In several passes, go from the reference frame up and down the stack, always
        # # trying to find the star in the next frame in the vicinity of the expected
        # # position predicted by the rough position prediction model.
        # # Filling the positions into the `star_trail` table as we go.
        # num_stars_found = 1  # running variable for exit condition
        # for _ in range(10):
        #     for frame_name in star_trail.index:
        #         if pd.notna(star_trail.loc[frame_name, "x"]):
        #             # The star has already been found in this frame, skip it:
        #             continue

        #         # Rough position prediction:
        #         use_n_closest = 15
        #         guess_model_size = min(star_trail.x.dropna().count(), use_n_closest)
        #         x_guess, y_guess = utils.predict_star_position_in_frame(
        #             star_trail[["t", "x", "y"]], frame_name, n_closest=use_n_closest
        #         )
        #         if np.isnan(x_guess) or np.isnan(y_guess):
        #             # The rough prediction failed, skip this frame:
        #             continue

        #         # If the point is outside the image or not in the sky, also skip it:
        #         frame = self.light_frames[frame_name]
        #         if not frame.is_in_sky((x_guess, y_guess)):
        #             star_trail.loc[frame_name, "in_sky"] = False
        #             continue

        #         # Attempt to find the star in the frame close to the predicted position:
        #         star = frame.find_star(
        #             pos=(x_guess, y_guess),
        #             margin=self._get_margin_for_star_finder(),
        #             fwhm=fwhm,
        #             init_thresh=threshold,
        #             max_roundness=max_roundness,
        #             min_separation=min_separation,
        #         )
        #         if star is not None:
        #             # If the star was found, we'll use its coordinates:
        #             star_trail.loc[
        #                 frame_name,
        #                 [
        #                     "x_guess",
        #                     "y_guess",
        #                     "x",
        #                     "y",
        #                     "flux",
        #                     "threshold",
        #                     "fwhm",
        #                     "guess_error",
        #                     "guess_model_size",
        #                 ],
        #             ] = [
        #                 round(x_guess, 2),
        #                 round(y_guess, 2),
        #                 star["x"],
        #                 star["y"],
        #                 star["flux"],
        #                 star["threshold"],
        #                 star["fwhm"],
        #                 round(
        #                     ((star["x"] - x_guess) ** 2 + (star["y"] - y_guess) ** 2)
        #                     ** 0.5,
        #                     2,
        #                 ),
        #                 guess_model_size,
        #             ]
        #     _num_stars_found = len(star_trail.x.dropna())
        #     if _num_stars_found == num_stars_found:
        #         # No new stars were found in this pass, stop the loop:
        #         break
        #     num_stars_found = _num_stars_found

        # # Final pimp-up of the star trail table:
        # star_trail.index.name = "frame"

        # if show_diagnostics:
        #     # Plot the star trail and show it:
        #     fig = self.plot_star_trail(trail_df=star_trail)
        #     fig.show()

        #     # Plot the guess error and guess model sample size as a function of `t`:
        #     fig = sp.make_subplots(
        #         rows=2,
        #         cols=1,
        #         shared_xaxes=True,
        #         vertical_spacing=0.1,
        #     )

        #     fig.add_trace(
        #         go.Scatter(
        #             x=star_trail["t"],
        #             y=star_trail["guess_error"],
        #             mode="markers",
        #             name="Guess Error",
        #         ),
        #         row=1,
        #         col=1,
        #     )

        #     fig.add_trace(
        #         go.Scatter(
        #             x=star_trail["t"],
        #             y=star_trail["guess_model_size"],
        #             mode="markers",
        #             name="Model Sample Size",
        #         ),
        #         row=2,
        #         col=1,
        #     )

        #     fig.update_xaxes(title_text="Capture Time", row=2, col=1)
        #     fig.update_yaxes(title_text="Guess error (px)", row=1, col=1)
        #     fig.update_yaxes(title_text="Guess model sample size", row=2, col=1)
        #     fig.show()

        # return star_trail

    def plot_to_figure(self, fig: go.Figure, stars_frame: str) -> None:
        """Plot the stars in the table into an existing figure.

        Each star present in the table for the given frame name is plotted as a scatter
        point, with its size and color determined by the star's brightness.

        Additionally, the star trails are plotted for each star, as a line of the same
        color as the star, connecting the star positions across all the frames.
        This only applies, if the stars have been propagated across the frames already.

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
        raise NotImplementedError  # TODO: Implement the plotting logic
        # if self.stars_table is not None:
        #     stars_table = self.stars_table.loc[
        #         (self.stars_table.x >= 0) & (self.stars_table.y >= 0)
        #     ]

        #     fig.update_layout(showlegend=True)
        #     colorscale = "Viridis"

        #     # Add the stars to the plot, if any are detected for the selected frame:
        #     stars_in_frame = stars_table[stars_table["frame"] == frame.name]
        #     fig.add_trace(
        #         go.Scatter(
        #             x=stars_in_frame["x"],
        #             y=stars_in_frame["y"],
        #             mode="markers",
        #             marker={
        #                 "size": (
        #                     stars_in_frame["flux"] / np.max(stars_in_frame["flux"]) * 10
        #                 ),
        #                 "color": stars_in_frame["flux"],
        #                 "colorscale": colorscale,
        #             },
        #             name="Stars",
        #             customdata=stars_in_frame[["id", "x", "y", "frame", "flux"]].values,
        #             hovertemplate=(
        #                 "ID: %{customdata[0]}<br>"
        #                 "x: %{customdata[1]:.2f}<br>"
        #                 "y: %{customdata[2]:.2f}<br>"
        #                 "frame: %{customdata[3]}<br>"
        #                 "flux: %{customdata[4]}<br>"
        #                 "<extra></extra>"
        #             ),
        #         ),
        #     )

        #     # If the stars have been already propagated to any other frames, add the
        #     # trails as separate traces:
        #     if len(stars_table.frame.unique()) > 1:
        #         stars_colors = stars_in_frame.set_index("id")["flux"]
        #         vmin, vmax = stars_colors.min(), stars_colors.max()
        #         stars_colors_norm = (stars_colors - vmin) / (vmax - vmin)
        #         rgb_dict = dict(
        #             zip(
        #                 stars_colors.index,
        #                 pc.sample_colorscale(
        #                     colorscale,
        #                     stars_colors_norm,
        #                     colortype="rgb",
        #                 ),
        #                 strict=False,
        #             ),
        #         )

        #         # Add a dummy trace to control the trails traces in one legend group:
        #         legend_group_name = "star_trails"
        #         fig.add_trace(
        #             go.Scatter(
        #                 x=[None],
        #                 y=[None],
        #                 mode="lines",
        #                 line={"width": 1, "color": "rgb(100, 100, 100)"},
        #                 name="Star Trails",  # Shown in the legend
        #                 legendgroup=legend_group_name,
        #                 showlegend=True,
        #             ),
        #         )
        #         star_trails = stars_table.sort_values(by=["t"])
        #         for star_id in sorted(star_trails["id"].unique()):
        #             star_trail = star_trails[star_trails["id"] == star_id]
        #             color = rgb_dict.get(star_id, "rgb(100, 100, 100)")
        #             fig.add_trace(
        #                 go.Scatter(
        #                     x=star_trail["x"],
        #                     y=star_trail["y"],
        #                     mode="lines",
        #                     line={
        #                         "width": 1,
        #                         "color": color,
        #                     },
        #                     customdata=star_trail[["id", "x", "y", "frame"]].values,
        #                     hovertemplate=(
        #                         "ID: %{customdata[0]}<br>"
        #                         "x: %{customdata[1]:.2f}<br>"
        #                         "y: %{customdata[2]:.2f}<br>"
        #                         "frame: %{customdata[3]}<br>"
        #                         "<extra></extra>"
        #                     ),
        #                     showlegend=False,
        #                     legendgroup=legend_group_name,
        #                 ),
        #             )

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
        raise NotImplementedError  # TODO: Implement the star trail plotting logic
        # """Plot the trail of a single star across all frames in the stack.

        # The method will plot the trail of the star across the frames in the stack.
        # The star to plot can either be specified by its ID (in which case the star
        # data will be queried from the `stars_table`), or by passing its star trail
        # as a DataFrame returned by the `propagate_a_star` method.

        # If the `trail_df` is provided, it will never be changed in place.

        # The star positions across the stack will be plotted on the background of the
        # frames stacked in the "brightest" mode, so that the star pixels from all the
        # frames are visible.

        # Args:
        #     star_id: The ID of the star to plot. Optional, if not passed, the `trail_df`
        #         must be provided.
        #     trail_df: The DataFrame containing the star trail. Optional, if not passed,
        #         the `star_id` must be provided.

        # Returns:
        #     A Plotly figure with the star trail plot.
        # """
        # # Basic sanity checks:
        # if sum([star_id is not None, trail_df is not None]) != 1:
        #     msg = "Exactly one of `star_id` or `trail_df` must be provided."
        #     raise ValueError(msg)

        # if trail_df is None:
        #     # Extract the star trail from the stars table:
        #     if self.stars_table is None:
        #         msg = "Stars have not yet been detected in the stack."
        #         raise ValueError(msg)
        #     trail_df = self.stars_table.loc[self.stars_table.id == star_id]
        #     trail_df = trail_df.copy().sort_values(by="t").set_index("frame", drop=True)
        # else:
        #     trail_df = trail_df.copy().sort_values(by="t")

        # if not len(trail_df.x.dropna()):
        #     msg = "No valid star positions found in the star trail."
        #     raise ValueError(msg)

        # # Initialize the background from one of the frames:
        # margin = self._get_margin_for_star_finder()
        # x_min = max(round(trail_df.x.min() - margin), 0)
        # x_max = min(round(trail_df.x.max() + margin), self.width)
        # y_min = max(round(trail_df.y.min() - margin), 0)
        # y_max = min(round(trail_df.y.max() + margin), self.height)
        # bkg_frame = (
        #     self.ref_frame if self.ref_frame else next(iter(self.light_frames.values()))
        # )
        # bkg_array = bkg_frame.array_gs[y_min:y_max, x_min:x_max]

        # # Add the pixels from all the frames in the stack into the background array:
        # for frame_name, row in trail_df.iterrows():
        #     if frame_name == bkg_frame.name:
        #         # Skip the background frame, it's already there:
        #         continue
        #     if np.isnan(row.x) or np.isnan(row.y):
        #         # Skip the star if its position is NaN:
        #         continue
        #     frame = self.light_frames[str(frame_name)]
        #     x1 = max(round(row.x - margin), x_min)
        #     x2 = min(round(row.x + margin), x_max)
        #     y1 = max(round(row.y - margin), y_min)
        #     y2 = min(round(row.y + margin), y_max)

        #     # Add the pixels from the frame to the background array:
        #     overlay = np.zeros_like(bkg_array)
        #     overlay[y1 - y_min : y2 - y_min, x1 - x_min : x2 - x_min] = frame.array_gs[
        #         y1:y2, x1:x2
        #     ]
        #     bkg_array = np.maximum(bkg_array, overlay)
        #     frame.clear_cache()  # Clear the cache to save memory

        # fig = utils.plot_image(bkg_array)

        # # Add the star positions to the plot:
        # trail_df_notna = trail_df.dropna(subset=["x", "y"])
        # colors = pd.Series("yellow", index=trail_df_notna.index, dtype="object")
        # if self.ref_frame is not None:
        #     # Highlight the star position in the reference frame:
        #     colors.loc[self.ref_frame.name] = "red"
        # hoverdata = trail_df_notna.reset_index()
        # hovertemplate = (
        #     "<br>".join(
        #         [
        #             f"{col}: %{{customdata[{i}]}}"
        #             for i, col in enumerate(hoverdata.columns)
        #             if col not in ["fwhm", "in_sky", "t"]
        #         ]
        #     )
        #     + "<extra></extra>"
        # )
        # fig.add_trace(
        #     go.Scatter(
        #         x=trail_df_notna["x"] - x_min,
        #         y=trail_df_notna["y"] - y_min,
        #         mode="markers",
        #         marker={
        #             "size": (
        #                 trail_df_notna["flux"] / trail_df_notna["flux"].max() * 10
        #             ),  # Scale the marker size by flux
        #             "color": colors,
        #             "line": {"width": 1, "color": "black"},  # Add a black border
        #         },
        #         name=f"Star ID {star_id} Positions",
        #         customdata=hoverdata.to_numpy(),
        #         hovertemplate=hovertemplate,
        #     ),
        # )
        # fig.update_layout(showlegend=True)

        # return fig


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
