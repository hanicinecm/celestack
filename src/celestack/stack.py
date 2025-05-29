"""A module containing the FrameStack class, which is used to manage a stack of frames.

Each stack is referenced only by the project name, as only a single stack is allowed
per project.
"""

from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.colors as pc
import plotly.graph_objects as go
import plotly.subplots as sp
import yaml

import celestack._discovery as discovery
import celestack._utils as utils
from celestack import PROGRESS_BAR
from celestack.frame import AverageLight, DarkFrame, Frame, LightFrame, Mask, MasterDark
from celestack.segment import SegmentBox


class FrameStack:
    """A class to manage a stack of frames, as the core object of the Celestack app."""

    def __init__(self, project: str) -> None:
        """Initialize the FrameStack object.

        The stack takes care of the dark-frame correction, foreground masking and
        ultimately the star detection.

        The stars (once detected) are stored in the `stars_table` attribute as a
        DataFrame.

        This initializer only applies if the stack does not yet exist in the project.
        For an already existing stack, use the `from_state` class method instead.

        Args:
            project: The name of the project.
        """
        # Instantiation is only allowed if the stack does not yet exist in the project:
        if discovery.get_stack_state_path(project).exists():
            msg = f"Stack already exists in project '{project}'."
            raise FileExistsError(msg)

        # Store the project name:
        self.project = project

        # Initialize all the other attributes:
        self.light_frames: dict[str, LightFrame] = {}
        self.dark_frames: dict[str, DarkFrame] = {}

        self.master_dark: MasterDark | None = None
        self.avg_light: AverageLight | None = None
        self.ref_frame: LightFrame | None = None
        self.mask: Mask | None = None

        self.segment_boxes: list[SegmentBox] = []

        self.star_finder_params: dict[str, float | None] = {
            "fwhm": None,  # optimum will be auto-detected.
            "max_roundness": 1.7,  # stars with higher |roundness| will be rejected.
            "min_separation": 1.5,  # minimum separation between stars in [fwhm].
        }

        self.stars_table: pd.DataFrame | None = None  # TODO: switch to polars

        # Dump the initial state of the stack to the state file:
        self.dump_state()

    def __repr__(self) -> str:
        """Return the string representation of the FrameStack object."""
        return f"{self.__class__.__name__}({self.project})"

    @cached_property
    def _px_resolution(self) -> tuple[int, int]:
        """The pixel width and height of the images in the stack."""
        if self.ref_frame is not None:
            frame = self.ref_frame
        elif self.light_frames:
            frame = next(iter(self.light_frames.values()))
        else:
            msg = "Stack does not contain any frames to determine its dimensions from."
            raise ValueError(msg)
        return frame.width, frame.height

    @property
    def width(self) -> int:
        """The pixel width of the images in the stack."""
        return self._px_resolution[0]

    @property
    def height(self) -> int:
        """The pixel height of the images in the stack."""
        return self._px_resolution[1]

    def load_frames(
        self,
        lf_paths: list[Path],
        df_paths: list[Path] | None = None,
    ) -> None:
        """Load all the light frames and the dark frames into the stack.

        The method will create the LightFrame and DarkFrame objects and store them in
        the project.

        Args:
            lf_paths: A list of paths to the light frames.
            df_paths: A list of paths to the dark frames.

        Raises:
            ValueError: If the stack already contains some frames.
            ValueError: If the light frame paths are not unique.
            ValueError: If the dark frame paths are not unique.
        """
        # Basic sanity checks:
        if self.light_frames or self.dark_frames:
            msg = "Stack contains some frames already."
            raise ValueError(msg)
        if len({pth.stem for pth in lf_paths}) != len(lf_paths):
            msg = "Light frame names must be unique."
            raise ValueError(msg)
        if df_paths and len({pth.stem for pth in df_paths}) != len(df_paths):
            msg = "Dark frame names must be unique."
            raise ValueError(msg)

        # Load the light frames and the dark frames into the project:
        for lf_path in PROGRESS_BAR(lf_paths, "Loading light frames"):
            lf = LightFrame(project=self.project, name=lf_path.stem, img_path=lf_path)
            self.light_frames[lf.name] = lf
            self.dump_state()

        for df_path in PROGRESS_BAR(df_paths or [], "Loading dark frames"):
            dark_frame = DarkFrame(
                project=self.project, name=df_path.stem, img_path=df_path
            )
            self.dark_frames[dark_frame.name] = dark_frame
            self.dump_state()

        self.light_frames = dict(sorted(self.light_frames.items()))
        self.dark_frames = dict(sorted(self.dark_frames.items()))
        self.dump_state()

    def apply_dark_frames_correction(self) -> None:
        """Create the master-dark frame and subtract it from the stack's light frames.

        Raises:
            ValueError: If the stack does not contain any dark frames.
            ValueError: If the stack contains the master-dark frame already.
            ValueError: If the stack contains the average-light frame already (which is
                supposed to be created from dark-frame-corrected light frames).
        """
        # Basic sanity checks:
        if not self.dark_frames:
            msg = "Stack does not contain any dark frames."
            raise ValueError(msg)
        if self.master_dark:
            msg = "Stack already contains the master-dark frame."
            raise ValueError(msg)
        if self.avg_light:
            msg = "Stack already contains the average-light frame."
            raise ValueError(msg)

        # Create the master-dark frame and save it into the project:
        self.master_dark = MasterDark(
            project=self.project,
            name="MasterDark",
            frames=list(self.dark_frames.values()),
        )
        self.dump_state()

        # Subtract the master-dark frame from all the light frames:
        for lf in PROGRESS_BAR(self.light_frames.values(), "Subtracting master dark"):
            lf.subtract_master_dark(self.master_dark)
        self.dump_state()

    def create_average_light_frame(self) -> None:
        """Create the average-light frame from the light frames in the stack.

        Raises:
            ValueError: If the stack does not contain any light frames.
            ValueError: If the stack contains the average-light frame already.
        """
        # Basic sanity checks:
        if not self.light_frames:
            msg = "Stack does not contain any light frames."
            raise ValueError(msg)
        if self.avg_light:
            msg = "Stack already contains the average-light frame."
            raise ValueError(msg)

        # Create the average-light frame and save it into the project:
        self.avg_light = AverageLight(
            project=self.project,
            name="AverageLight",
            frames=list(self.light_frames.values()),
        )
        self.dump_state()

    def add_mask(self, mask: Mask) -> None:
        """Add a mask to the stack.

        The mask passed must be an existing mask object, which has already been
        instantiated and saved into the same project.

        The mask name will be saved in the stack state file and the mask will also be
        assigned to every light frame in the stack.

        Args:
            mask: The mask to add to the stack.

        Raises:
            ValueError: If the stack already contains a mask.
            ValueError: If the mask does not belong to the same project.
        """
        # Basic sanity checks:
        if self.mask:
            msg = "Stack already contains a mask."
            raise ValueError(msg)
        if mask.project != self.project:
            msg = (
                "Mask does not belong to the same project: "
                f"{mask.project} != {self.project}"
            )
            raise ValueError(
                msg,
            )

        # Assign the mask to all the light frames in the stack:
        for lf in self.light_frames.values():
            lf.assign_mask(mask)

        # Add the mask to the stack and save it into the project:
        self.mask = mask
        self.dump_state()

    def segment_sky(self, n_segments: int = 50) -> None:
        """Segment the sky into N rectangular segments.

        The segments are defined by their bounding boxes and stored in the stack object
        as a list of SegmentBox objects.
        They are calculated to contain roughly the same number of sky pixels each.

        The segment boxes in the list are ordered in a continuous fashion,
        meaning that each two segments next to each other in the list are also negboring
        segments in the image.

        More specifically, the segments are ordered in the list column by column in a
        meandering fashion, with the first segment being the top-left corner of the
        image.

        Args:
            n_segments: The number of segments to create. Defaults to 50.

        Raises:
            ValueError: If the stack does not contain a mask.
            ValueError: If any stars have already been detected in the stack.
        """
        # Basic sanity checks:
        if self.mask is None:
            msg = "Stack does not contain a mask."
            raise ValueError(msg)
        if self.stars_table is not None:
            msg = "Stars have already been detected in the stack."
            raise ValueError(msg)

        segment_boxes: list[SegmentBox] = []

        # First, figure out how many pixels we have in the sky:
        ma = self.mask.mask_array  # This is a boolean array with False for sky pixels
        h, _ = self.mask.shape

        # Lets define the xmin, xmax of the bounding box of the sky:
        xmin, xmax = np.where(np.sum(ma, axis=0) < h)[0][[0, -1]]

        # Let's define the nominal segment size (the segments will be nominally square):
        n_px = np.sum(~ma)  # Total number of sky pixels
        size_nom = int(np.sqrt(n_px // n_segments))

        # The segments will be defined in columns - let's define the width of the
        # column as how many times the nominal size fits between the xmin and xmax:
        n_cols = round((xmax - xmin) / size_nom)
        width_nom = (xmax - xmin) // n_cols

        x1 = xmin
        for i in range(n_cols):
            x2 = x1 + width_nom if i < n_cols - 1 else xmax

            # Now we need to find the ymin and ymax of the bounding box of the sky in
            # the column:
            ymin, ymax = np.where(np.sum(ma[:, x1:x2], axis=1) < (x2 - x1))[0][[0, -1]]

            n_px_col = np.sum(
                ~ma[ymin:ymax, x1:x2],
            )  # Number of sky pixels in the column
            n_rows = round(n_px_col / width_nom**2)
            n_px_seg = n_px_col // n_rows  # Nominal number of pixels in the segment

            j = 1
            y1 = ymin
            y2 = ymin
            segment_boxes_col: list[SegmentBox] = []
            while True:
                # We'll increment y2 until we have enough pixels in the segment:
                y2 += 1

                # Catch the end of the column:
                if j == n_rows:
                    segment_boxes_col.append(
                        SegmentBox(x1=int(x1), y1=int(y1), x2=int(x2), y2=int(ymax)),
                    )
                    if i % 2 == 1:
                        segment_boxes_col.reverse()
                    segment_boxes.extend(segment_boxes_col)
                    segment_boxes_col = []
                    break

                # Check if we have enough pixels in the segment:
                if np.sum(~ma[ymin:y2, x1:x2]) >= j * n_px_seg:
                    segment_boxes_col.append(
                        SegmentBox(x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2)),
                    )
                    y1 = y2
                    j += 1

            x1 = x2

        # Store the segment boxes in the stack and dump the state:
        self.segment_boxes = segment_boxes
        self.dump_state()

    def set_reference_frame(self, ref_frame_name: str) -> None:
        """Set the reference frame for the stack.

        The reference frame is the special light frame from the stack, onto which all
        ther other frames will be aligned.

        It is also used for the initial stars detection after the sky is segmented.

        Args:
            ref_frame_name: The name of the reference frame. Must exist in the stack.

        Raises:
            ValueError: If the reference frame passed is not in the stack.
            ValueError: If any stars have already been detected.
        """
        # Basic sanity checks:
        if ref_frame_name not in self.light_frames:
            msg = f"Reference frame '{ref_frame_name}' does not belong to the stack."
            raise ValueError(
                msg,
            )
        if self.stars_table is not None:
            msg = "Stars have already been detected in the stack."
            raise ValueError(msg)

        # Set the reference frame and save it into the project:
        self.ref_frame = self.light_frames[ref_frame_name]
        self.dump_state()

        # Clear the `t_series` property cache by deleting the attribute:
        if hasattr(self, "t_series"):
            del self.t_series

    @cached_property
    def t_series(self) -> pd.Series:
        """Get the time coordinates of all the frames in the stack.

        Get the time-like distance of each frame from the reference frame.
        If the frames have exif data with the timestamp, we'll use that and the `t` will
        be in seconds.

        Note, that the

        Returns:
            The series of time of capture for each frame, relative to the reference
            frame. The t is in [seconds] and the t of the reference frame is 0.
            The output is a pandas Series with the frame names as index, times as values
            and the name "t".

        Raises:
            ValueError: If the reference frame has not yet been set.
            NotImplementedError: If the frames do not have exif data with the
                timestamp.
        """
        # Basic sanity checks:
        if self.ref_frame is None:
            msg = "Reference frame has not yet been set."
            raise ValueError(msg)

        # Get the timestamps from the light frames' exif data:
        try:
            timestamps: dict[str, str] = {
                frame_name: frame.exif_data["DateTimeOriginal"]
                for frame_name, frame in self.light_frames.items()
            }
        except KeyError as err:
            msg = (
                "The frames do not have exif data with the timestamp. "
                "Please implement a way to get the time of capture for each frame."
            )
            raise NotImplementedError(msg) from err

        times = pd.Series(timestamps)
        times = pd.to_datetime(times, format="%Y:%m:%d %H:%M:%S")

        # Convert to seconds relative to the reference frame
        times = (times - times[self.ref_frame.name]).dt.total_seconds()
        times.name = "t"

        return times

    def _find_optimal_fwhm(self) -> float:
        """Find the optimal star FWHM for the star finder.

        The method will pick 5 segments (closest to the center and each corner of the
        image) and find the FWHM which yields the most stars (with relatively high
        threshold).

        The average of these values will be considered the optimal FWHM for the
        star finder and it will be stored (persistently) in the stack's
        `star_finder_params` dictionary.

        Returns:
            The optimal FWHM for the star finder.

        Raises:
            ValueError: If the stack's sky has not yet been segmented.
            ValueError: If the stack's reference frame has not yet been set.
        """
        if not self.segment_boxes:
            msg = "Sky has not yet been segmented."
            raise ValueError(msg)
        if self.ref_frame is None:
            msg = "Reference frame has not yet been set."
            raise ValueError(msg)

        # Choose 5 segment boxes in representative locations:
        w, h = self.ref_frame.width, self.ref_frame.height
        boxes_x = [(b.x1 + b.x2) / 2 for b in self.segment_boxes]
        boxes_y = [(b.y1 + b.y2) / 2 for b in self.segment_boxes]
        boxes_df = pd.DataFrame({"x": boxes_x, "y": boxes_y})
        for pos, label in zip(
            [(0, 0), (0, w), (h, 0), (h, w), (h / 2, w / 2)],
            ["top-left", "top-right", "bottom-left", "bottom-right", "center"],
            strict=False,
        ):
            boxes_df[f"dist_{label}"] = (
                (boxes_df["x"] - pos[0]) ** 2 + (boxes_df["y"] - pos[1]) ** 2
            ) ** 0.5
        chosen_boxes_idx = [
            boxes_df["dist_top-left"].idxmin(),
            boxes_df["dist_top-right"].idxmin(),
            boxes_df["dist_bottom-left"].idxmin(),
            boxes_df["dist_bottom-right"].idxmin(),
            boxes_df["dist_center"].idxmin(),
        ]
        chosen_boxes = [self.segment_boxes[int(i)] for i in chosen_boxes_idx]

        # For each chosen segment, find the optimal FWHM:
        optimal_fwhms = []
        for box in PROGRESS_BAR(chosen_boxes, "Finding optimal FWHM for star finder"):
            # Get the segment and some parameters:
            segment = self.ref_frame.get_segment(box)
            max_roundness = self.star_finder_params["max_roundness"]
            if not isinstance(max_roundness, float):
                msg = "The max_roundness must be a float."
                raise TypeError(msg)
            threshold = 4.0  # reasonably high threshold to not detect noise...

            # Iterate over a range of FWHM values and find the one with the most stars:
            fwhm_values = np.linspace(2.0, 10.0, 17)
            n_stars_values = []
            for fwhm in fwhm_values:
                # Instantiate the star finder:
                stars = utils.find_stars(
                    array=segment.array,
                    threshold=threshold,
                    fwhm=fwhm,
                    max_roundness=max_roundness,
                    mask=segment.mask,
                )
                n_stars_values.append(len(stars))
            # Find the FWHM with the most stars:
            optimal_fwhms.append(fwhm_values[np.argmax(n_stars_values)])

        # Calculate the optimal FWHM as the average of the optimal FWHMs:
        fwhm = np.mean(optimal_fwhms)
        fwhm = round(float(fwhm), 2)

        # Store the optimal FWHM in the stack and return:
        self.star_finder_params["fwhm"] = fwhm
        self.dump_state()
        return fwhm

    def _get_margin_for_star_finder(self) -> int:
        """Get the pixel margin around the expected star position for the star finder.

        The margin will be calculated as (min_separation * FWHM) - both saved
        in the stack's `star_finder_params` dictionary.
        """
        min_separation = self.star_finder_params["min_separation"]
        fwhm = self.star_finder_params["fwhm"]
        if min_separation is None or fwhm is None:
            msg = "The min_separation and fwhm must be set in the star_finder_params."
            raise ValueError(msg)

        return round(min_separation * fwhm)

    def detect_stars_in_ref_frame(self, n: int = 1000) -> None:
        """Detect roughly N stars in a single frame.

        The algorithm will go from segment to segment (the sky must have been segmented
        already) and detect an appropriate number of stars in each segment, to reach
        the target `n` stars in the whole frame.

        The stars detection happens in a single frame only, typically in the reference
        frame, and it is the first step in the stars registration and alignment process.

        The stars are stored (persistently) in the `stars_table` DataFrame.

        Args:
            n: The target number of stars to detect in the whole frame. Optional, if
                not passed, a sensible default is provided.

        Raises:
            ValueError: If the stars table exists already and contains stars from more
                then one frame.
            ValueError: If the sky has not yet been segmented.
            ValueError: If the frame name passed does not belong to the stack.
            ValueError: If the frame name is not passed and the reference frame has
                not yet been set.
        """
        # Basic sanity checks:
        if self.stars_table is not None and len(self.stars_table.frame.unique()) > 1:
            msg = (
                "Stars have already been detected in the stack for more than one frame."
            )
            raise ValueError(
                msg,
            )
        if not self.segment_boxes:
            msg = "Sky has not yet been segmented."
            raise ValueError(msg)
        if self.ref_frame is None:
            msg = "Reference frame has not yet been set."
            raise ValueError(msg)

        frame = self.ref_frame

        # If the FWHM is not set, find the optimal one (and set it):
        fwhm = self.star_finder_params["fwhm"]
        if fwhm is None:
            fwhm = self._find_optimal_fwhm()

        # Retrieve the maximal roundness and minimal separation from the star finder
        # params:
        max_roundness = self.star_finder_params["max_roundness"]
        min_separation = self.star_finder_params["min_separation"]

        # Basic sanity checks, mainly for the type checker:
        if not isinstance(max_roundness, float):
            msg = "The max_roundness must be a float."
            raise TypeError(msg)
        if min_separation is None:
            msg = "The min_separation must not be None."
            raise ValueError(msg)

        # Calculate the star density (in stars per 10,000 sky pixels) to lead to the
        # target number of stars:
        n_pixels = frame.width * frame.height
        if frame.mask_array is not None:
            n_pixels -= np.sum(frame.mask_array)
        density = n / (n_pixels / 10000)

        # Detect the stars in the frame:
        threshold = 4.0  # initial threshold for the star finding algorithm
        segment_tables: list[pd.DataFrame] = []
        for box in PROGRESS_BAR(self.segment_boxes, f"Detecting stars in {frame.name}"):
            # Get the segment:
            segment = frame.get_segment(box)
            # Find stars in the segment:
            segment_table = segment.find_stars(
                density=density,
                fwhm=fwhm,
                max_roundness=max_roundness,
                min_separation=min_separation,
                init_thresh=threshold,
            )
            # Update the initial threshold for the next segment:
            threshold = round(float(segment_table["threshold"].iloc[0]), 9)
            # Add the segment table to the list:
            segment_tables.append(segment_table)

        # Concatenate the segment tables into a single table:
        stars_table = pd.concat(segment_tables, ignore_index=True)

        # Add the frame metadata to the stars table:
        stars_table["frame"] = frame.name

        # Assign unique IDs to the stars:
        stars_table["id"] = np.arange(len(stars_table))

        # Add the "t" column, which is a "time-like" distance from the reference frame:
        stars_table["t"] = 0  # this is the reference frame...

        # Sort the columns in the stars table:
        cols = ["id", "frame", "t", "x", "y", "flux", "threshold", "fwhm"]
        if set(cols) != set(stars_table.columns):
            msg = "Stars table columns do not match expected columns."
            raise ValueError(msg)
        stars_table = stars_table[cols]

        # round the columns to a reasonable precision:
        for col in ["threshold", "fwhm"]:  # floating point error...
            stars_table[col] = stars_table[col].astype(float).round(9)
        for col in ["x", "y"]:
            stars_table[col] = stars_table[col].astype(float).round(2)

        # Store the stars table in the stack:
        self.stars_table = stars_table
        self.dump_state()

    def propagate_a_star(
        self, star_id: int, *, show_diagnostics: bool = False
    ) -> pd.DataFrame:
        """Propagate a single star from the reference frame to the whole stack.

        The method will try to match the star identified in the reference frame under
        the given `star_id` to the same star in all other frames in the stack.

        The method will not change the state of the stack, but rather will return a
        *star trail* table, which contains the star's position in each frame of the
        stack.

        Args:
            star_id: The ID of the star to propagate. This must be a valid ID from the
                `stars_table` in the stack, which is only known in the reference frame.
            show_diagnostics: If True, the method will plot the star trail and show it
                and it will print some useful diagnostic data to the console.

        Returns:
            A pandas DataFrame containing the star trail, with the following columns:
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

        Raises:
            ValueError: If the reference frame has not yet been set.
            ValueError: If the star have not yet been detected in the reference frame.
        """
        # Start with the sanity checks:
        if self.stars_table is None:
            msg = "Stars have not yet been detected in the reference frame."
            raise ValueError(msg)
        if self.ref_frame is None:
            msg = "Reference frame has not yet been set."
            raise ValueError(msg)
        star_data = self.stars_table[
            (self.stars_table["id"] == star_id)
            & (self.stars_table["frame"] == self.ref_frame.name)
        ]
        if len(star_data) != 1:
            msg = f"The star {star_id} must have already been found in the ref frame."
            raise ValueError(msg)

        # Initialize the star trail table:
        star_trail = pd.DataFrame(
            index=list(self.light_frames),
            columns=[
                "t",
                "x_guess",
                "y_guess",
                "x",
                "y",
                "flux",
                "threshold",
                "fwhm",
                "guess_error",
                "guess_model_size",
            ],
            dtype=float,
        )
        star_trail["in_sky"] = True
        star_trail["t"] = self.t_series
        star_trail = star_trail.sort_values(by="t", key=lambda x: x.abs())

        # Fill in the star position in the reference frame:
        star_trail.loc[self.ref_frame.name, ["x", "y", "flux", "threshold", "fwhm"]] = (
            star_data[["x", "y", "flux", "threshold", "fwhm"]].to_numpy()[0]
        )
        # Initialize the parameters for the star finder:
        threshold: float = star_data.threshold.iloc[0]
        fwhm: float = star_data.fwhm.iloc[0]
        max_roundness = self.star_finder_params["max_roundness"]
        min_separation = self.star_finder_params["min_separation"]
        if not isinstance(max_roundness, float) or not isinstance(
            min_separation, float
        ):
            msg = "The max_roundness and min_separation must be floats."
            raise TypeError(msg)

        # In several passes, go from the reference frame up and down the stack, always
        # trying to find the star in the next frame in the vicinity of the expected
        # position predicted by the rough position prediction model.
        # Filling the positions into the `star_trail` table as we go.
        num_stars_found = 1  # running variable for exit condition
        for _ in range(10):
            for frame_name in star_trail.index:
                if pd.notna(star_trail.loc[frame_name, "x"]):
                    # The star has already been found in this frame, skip it:
                    continue

                # Rough position prediction:
                use_n_closest = 15
                guess_model_size = min(star_trail.x.dropna().count(), use_n_closest)
                x_guess, y_guess = utils.predict_star_position_in_frame(
                    star_trail[["t", "x", "y"]], frame_name, n_closest=use_n_closest
                )
                if np.isnan(x_guess) or np.isnan(y_guess):
                    # The rough prediction failed, skip this frame:
                    continue

                # If the point is outside the image or not in the sky, also skip it:
                frame = self.light_frames[frame_name]
                if not frame.is_in_sky((x_guess, y_guess)):
                    star_trail.loc[frame_name, "in_sky"] = False
                    continue

                # Attempt to find the star in the frame close to the predicted position:
                star = frame.find_star(
                    pos=(x_guess, y_guess),
                    margin=self._get_margin_for_star_finder(),
                    fwhm=fwhm,
                    init_thresh=threshold,
                    max_roundness=max_roundness,
                    min_separation=min_separation,
                )
                if star is not None:
                    # If the star was found, we'll use its coordinates:
                    star_trail.loc[
                        frame_name,
                        [
                            "x_guess",
                            "y_guess",
                            "x",
                            "y",
                            "flux",
                            "threshold",
                            "fwhm",
                            "guess_error",
                            "guess_model_size",
                        ],
                    ] = [
                        round(x_guess, 2),
                        round(y_guess, 2),
                        star["x"],
                        star["y"],
                        star["flux"],
                        star["threshold"],
                        star["fwhm"],
                        round(
                            ((star["x"] - x_guess) ** 2 + (star["y"] - y_guess) ** 2)
                            ** 0.5,
                            2,
                        ),
                        guess_model_size,
                    ]
            _num_stars_found = len(star_trail.x.dropna())
            if _num_stars_found == num_stars_found:
                # No new stars were found in this pass, stop the loop:
                break
            num_stars_found = _num_stars_found

        # Final pimp-up of the star trail table:
        star_trail.index.name = "frame"

        if show_diagnostics:
            # Plot the star trail and show it:
            fig = self.plot_star_trail(trail_df=star_trail)
            fig.show()

            # Plot the guess error and guess model sample size as a function of `t`:
            fig = sp.make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
            )

            fig.add_trace(
                go.Scatter(
                    x=star_trail["t"],
                    y=star_trail["guess_error"],
                    mode="markers",
                    name="Guess Error",
                ),
                row=1,
                col=1,
            )

            fig.add_trace(
                go.Scatter(
                    x=star_trail["t"],
                    y=star_trail["guess_model_size"],
                    mode="markers",
                    name="Model Sample Size",
                ),
                row=2,
                col=1,
            )

            fig.update_xaxes(title_text="Capture Time", row=2, col=1)
            fig.update_yaxes(title_text="Guess error (px)", row=1, col=1)
            fig.update_yaxes(title_text="Guess model sample size", row=2, col=1)
            fig.show()

        return star_trail

    def propagate_stars(self) -> None:
        """Propagate all the stars from the reference frame to the whole stack.

        The method will iterate over all the stars found in the reference frame and
        propagate each of them to all other frames in the stack.
        The star trails will be stored in the `stars_table` DataFrame.

        Raises:
            ValueError: If the stars have not yet been detected in the reference frame.
            ValueError: If the reference frame has not yet been set.
            ValueError: If the stars table contains any other frames than the ref frame.
        """
        # Basic sanity checks:
        if self.stars_table is None:
            msg = "Stars have not yet been detected in the reference frame."
            raise ValueError(msg)
        if self.ref_frame is None:
            msg = "Reference frame has not yet been set."
            raise ValueError(msg)
        if list(self.stars_table["frame"].unique()) != [self.ref_frame.name]:
            msg = "The stars table must only contain stars from the reference frame."
            raise ValueError(msg)

        # Iterate over all the stars in the reference frame and propagate them:
        for star_id in PROGRESS_BAR(self.stars_table.id.unique(), "Propagating stars"):
            # Propagate the star and get the star trail:
            star_trail = self.propagate_a_star(star_id)
            # Conform the table to the expected format in the stars table and drop the
            # additional columns and the row for the reference frame (already in):
            star_subtable = star_trail.drop(index=self.ref_frame.name)
            star_subtable = star_subtable.loc[star_trail.in_sky, :].reset_index()
            star_subtable["id"] = star_id
            # Add the star trail to the stars table:
            self.stars_table = pd.concat(
                [self.stars_table, star_subtable[self.stars_table.columns]],
                ignore_index=True,
            )
            self.dump_state()

        # After propagating all the stars, all the frames in the stack are cached, which
        # will take up a lot of memory - let's clear the cache:
        self.clear_cache()

    def dump_state(self) -> None:
        """Dump the current state of the stack to a YAML file.

        This method is responsible for persisting the stack state in the project.
        """
        # Create the state dictionary which can be serialized to YAML:
        state = {
            # if the frames are there, we will only store their names:
            "light_frames": list(self.light_frames.keys()),
            "dark_frames": list(self.dark_frames.keys()),
            "master_dark": self.master_dark.name if self.master_dark else None,
            "avg_light": self.avg_light.name if self.avg_light else None,
            "ref_frame": self.ref_frame.name if self.ref_frame else None,
            "mask": self.mask.name if self.mask else None,
            # segment boxes are stored as list of lists:
            "segment_boxes": [
                [box.x1, box.y1, box.x2, box.y2] for box in self.segment_boxes
            ],
            # star finder params are stored as a dictionary:
            "star_finder_params": self.star_finder_params,
        }

        # Dump the state to a YAML file:
        state_path = discovery.get_stack_state_path(self.project)
        with state_path.open("w") as state_file:
            yaml.dump(state, state_file, default_flow_style=False)

        # The stars table is stored as a CSV file, instead of the state file:
        if self.stars_table is not None:
            stars_path = discovery.get_stars_table_path(self.project)
            self.stars_table.to_csv(stars_path, index=False)

    @classmethod
    def from_state(cls, project: str) -> "FrameStack":
        """Load the stack state file and instantiates the FrameStack object from it.

        Args:
            project: The name of the project.

        Returns:
            A FrameStack object loaded from the state file.
        """
        # Load the state from the YAML file:
        state = {"project": project}
        state_path = discovery.get_stack_state_path(project)
        with state_path.open() as state_file:
            state |= yaml.safe_load(state_file)

        # Do the conversions from the values in the state file to the actual objects:
        state["light_frames"] = {
            name: LightFrame.from_state(project, name) for name in state["light_frames"]
        }
        state["dark_frames"] = {
            name: DarkFrame.from_state(project, name) for name in state["dark_frames"]
        }
        if state["master_dark"]:
            state["master_dark"] = DarkFrame.from_state(project, state["master_dark"])
        if state["avg_light"]:
            state["avg_light"] = AverageLight.from_state(project, state["avg_light"])
        if state["ref_frame"]:
            state["ref_frame"] = LightFrame.from_state(project, state["ref_frame"])
        if state["mask"]:
            state["mask"] = Mask.from_state(project, state["mask"])
        state["segment_boxes"] = [
            SegmentBox(x1, y1, x2, y2) for x1, y1, x2, y2 in state["segment_boxes"]
        ]
        # Add the stars table to the state:
        state["stars_table"] = None

        # Create the FrameStack object:
        stack = cls.__new__(cls)
        stack.__dict__.update(state)

        # Load the stars table from the CSV file, if it exists:
        stars_path = discovery.get_stars_table_path(project)
        if stars_path.exists():
            stack.stars_table = pd.read_csv(stars_path)

        return stack

    def clear(self) -> None:
        """Remove all the traces of the stack from the project folder."""
        discovery.get_stack_state_path(self.project).unlink(missing_ok=True)
        discovery.get_stars_table_path(self.project).unlink(missing_ok=True)

    def clear_cache(self) -> None:
        """Clear the cache of all the frames in the stack."""
        for frame in self.light_frames.values():
            frame.clear_cache()
        for frame in self.dark_frames.values():
            frame.clear_cache()
        for frame in [self.master_dark, self.avg_light, self.mask]:
            if frame is not None:
                frame.clear_cache()

    def plot(self, frame: Frame | None = None) -> go.Figure:
        """Make the stack plot and return it as a Plotly figure.

        Optionally, the underlying frame can be selected by passing any frame object
        belonging to the same project.
        If not passed, a frame will be selected automatically from the stack.

        If any stars have been detected for the selected frame, they will be plotted
        as well.
        Additionally, if any stars have been propagated to other frames, their
        trails will be plotted with the same colors as the stars in the selected
        frame.

        Args:
            frame: The frame to plot. If not passed, a frame will be selected
                automatically (and more or less intelligently) from the stack.

        Returns:
            A Plotly figure with the stack plot. The plot will contain the underlying
            frame, the bounding boxes of the sky segments and the stars (if they have
            already been detected).

        Raises:
            ValueError: If a frame is passed which does not belong to the stack.
            ValueError: If the stack does not contain any frames.
        """
        # Start with auto-selecting the frame to plot, if not passed:
        if frame is None:
            for f in [self.ref_frame, self.avg_light, self.mask]:
                if f is not None:
                    frame = f
                    break
            else:
                if not self.light_frames:
                    msg = "Stack does not contain any frames."
                    raise ValueError(msg)
                frame = self.light_frames[sorted(self.light_frames)[0]]
        if frame is None:
            msg = "Frame could not be determined for plotting."
            raise ValueError(msg)

        # Basic sanity checks:
        if frame.project != self.project:
            msg = f"Frame '{frame.name}' does not belong to the stack '{self.project}'."
            raise ValueError(
                msg,
            )

        # Plot the selected frame:
        fig = frame.plot()

        if self.stars_table is not None:
            stars_table = self.stars_table.loc[
                (self.stars_table.x >= 0) & (self.stars_table.y >= 0)
            ]

            fig.update_layout(showlegend=True)
            colorscale = "Viridis"

            # Add the stars to the plot, if any are detected for the selected frame:
            stars_in_frame = stars_table[stars_table["frame"] == frame.name]
            fig.add_trace(
                go.Scatter(
                    x=stars_in_frame["x"],
                    y=stars_in_frame["y"],
                    mode="markers",
                    marker={
                        "size": (
                            stars_in_frame["flux"] / np.max(stars_in_frame["flux"]) * 10
                        ),
                        "color": stars_in_frame["flux"],
                        "colorscale": colorscale,
                    },
                    name="Stars",
                    customdata=stars_in_frame[["id", "x", "y", "frame", "flux"]].values,
                    hovertemplate=(
                        "ID: %{customdata[0]}<br>"
                        "x: %{customdata[1]:.2f}<br>"
                        "y: %{customdata[2]:.2f}<br>"
                        "frame: %{customdata[3]}<br>"
                        "flux: %{customdata[4]}<br>"
                        "<extra></extra>"
                    ),
                ),
            )

            # If the stars have been already propagated to any other frames, add the
            # trails as separate traces:
            if len(stars_table.frame.unique()) > 1:
                stars_colors = stars_in_frame.set_index("id")["flux"]
                vmin, vmax = stars_colors.min(), stars_colors.max()
                stars_colors_norm = (stars_colors - vmin) / (vmax - vmin)
                rgb_dict = dict(
                    zip(
                        stars_colors.index,
                        pc.sample_colorscale(
                            colorscale,
                            stars_colors_norm,
                            colortype="rgb",
                        ),
                        strict=False,
                    ),
                )

                # Add a dummy trace to control the trails traces in one legend group:
                legend_group_name = "star_trails"
                fig.add_trace(
                    go.Scatter(
                        x=[None],
                        y=[None],
                        mode="lines",
                        line={"width": 1, "color": "rgb(100, 100, 100)"},
                        name="Star Trails",  # Shown in the legend
                        legendgroup=legend_group_name,
                        showlegend=True,
                    ),
                )
                star_trails = stars_table.sort_values(by=["t"])
                for star_id in sorted(star_trails["id"].unique()):
                    star_trail = star_trails[star_trails["id"] == star_id]
                    color = rgb_dict.get(star_id, "rgb(100, 100, 100)")
                    fig.add_trace(
                        go.Scatter(
                            x=star_trail["x"],
                            y=star_trail["y"],
                            mode="lines",
                            line={
                                "width": 1,
                                "color": color,
                            },
                            customdata=star_trail[["id", "x", "y", "frame"]].values,
                            hovertemplate=(
                                "ID: %{customdata[0]}<br>"
                                "x: %{customdata[1]:.2f}<br>"
                                "y: %{customdata[2]:.2f}<br>"
                                "frame: %{customdata[3]}<br>"
                                "<extra></extra>"
                            ),
                            showlegend=False,
                            legendgroup=legend_group_name,
                        ),
                    )

        # Plot the segment boxes:
        if self.segment_boxes:
            fig.update_layout(showlegend=True)
            segments_array = np.vstack(
                [
                    np.vstack([seg.to_array(), np.array([np.nan, np.nan])])
                    for seg in self.segment_boxes
                ],
            )
            fig.add_trace(
                go.Scatter(
                    x=segments_array[:, 0],
                    y=segments_array[:, 1],
                    mode="lines",
                    line={"width": 1, "color": "red"},
                    name="Segment Boxes",
                    visible="legendonly" if self.stars_table is not None else "legend",
                ),
            )

        return fig

    def plot_star_trail(
        self, *, star_id: int | None = None, trail_df: pd.DataFrame | None = None
    ) -> go.Figure:
        """Plot the trail of a single star across all frames in the stack.

        The method will plot the trail of the star across the frames in the stack.
        The star to plot can either be specified by its ID (in which case the star
        data will be queried from the `stars_table`), or by passing its star trail
        as a DataFrame returned by the `propagate_a_star` method.

        If the `trail_df` is provided, it will never be changed in place.

        The star positions across the stack will be plotted on the background of the
        frames stacked in the "brightest" mode, so that the star pixels from all the
        frames are visible.

        Args:
            star_id: The ID of the star to plot. Optional, if not passed, the `trail_df`
                must be provided.
            trail_df: The DataFrame containing the star trail. Optional, if not passed,
                the `star_id` must be provided.

        Returns:
            A Plotly figure with the star trail plot.
        """
        # Basic sanity checks:
        if sum([star_id is not None, trail_df is not None]) != 1:
            msg = "Exactly one of `star_id` or `trail_df` must be provided."
            raise ValueError(msg)

        if trail_df is None:
            # Extract the star trail from the stars table:
            if self.stars_table is None:
                msg = "Stars have not yet been detected in the stack."
                raise ValueError(msg)
            trail_df = self.stars_table.loc[self.stars_table.id == star_id]
            trail_df = trail_df.copy().sort_values(by="t").set_index("frame", drop=True)
        else:
            trail_df = trail_df.copy().sort_values(by="t")

        if not len(trail_df.x.dropna()):
            msg = "No valid star positions found in the star trail."
            raise ValueError(msg)

        # Initialize the background from one of the frames:
        margin = self._get_margin_for_star_finder()
        x_min = max(round(trail_df.x.min() - margin), 0)
        x_max = min(round(trail_df.x.max() + margin), self.width)
        y_min = max(round(trail_df.y.min() - margin), 0)
        y_max = min(round(trail_df.y.max() + margin), self.height)
        bkg_frame = (
            self.ref_frame if self.ref_frame else next(iter(self.light_frames.values()))
        )
        bkg_array = bkg_frame.array_gs[y_min:y_max, x_min:x_max]

        # Add the pixels from all the frames in the stack into the background array:
        for frame_name, row in trail_df.iterrows():
            if frame_name == bkg_frame.name:
                # Skip the background frame, it's already there:
                continue
            if np.isnan(row.x) or np.isnan(row.y):
                # Skip the star if its position is NaN:
                continue
            frame = self.light_frames[str(frame_name)]
            x1 = max(round(row.x - margin), x_min)
            x2 = min(round(row.x + margin), x_max)
            y1 = max(round(row.y - margin), y_min)
            y2 = min(round(row.y + margin), y_max)

            # Add the pixels from the frame to the background array:
            overlay = np.zeros_like(bkg_array)
            overlay[y1 - y_min : y2 - y_min, x1 - x_min : x2 - x_min] = frame.array_gs[
                y1:y2, x1:x2
            ]
            bkg_array = np.maximum(bkg_array, overlay)
            frame.clear_cache()  # Clear the cache to save memory

        fig = utils.plot_image(bkg_array)

        # Add the star positions to the plot:
        trail_df_notna = trail_df.dropna(subset=["x", "y"])
        colors = pd.Series("yellow", index=trail_df_notna.index, dtype="object")
        if self.ref_frame is not None:
            # Highlight the star position in the reference frame:
            colors.loc[self.ref_frame.name] = "red"
        hoverdata = trail_df_notna.reset_index()
        hovertemplate = (
            "<br>".join(
                [
                    f"{col}: %{{customdata[{i}]}}"
                    for i, col in enumerate(hoverdata.columns)
                    if col not in ["fwhm", "in_sky", "t"]
                ]
            )
            + "<extra></extra>"
        )
        fig.add_trace(
            go.Scatter(
                x=trail_df_notna["x"] - x_min,
                y=trail_df_notna["y"] - y_min,
                mode="markers",
                marker={
                    "size": (
                        trail_df_notna["flux"] / trail_df_notna["flux"].max() * 10
                    ),  # Scale the marker size by flux
                    "color": colors,
                    "line": {"width": 1, "color": "black"},  # Add a black border
                },
                name=f"Star ID {star_id} Positions",
                customdata=hoverdata.to_numpy(),
                hovertemplate=hovertemplate,
            ),
        )
        fig.update_layout(showlegend=True)

        return fig
