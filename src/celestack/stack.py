"""A module containing the FrameStack class, which is used to manage a stack of frames.

Each stack is referenced only by the project name, as only a single stack is allowed
per project.
"""

from functools import cached_property
from pathlib import Path
from typing import cast

import numpy as np
import plotly.graph_objects as go
import yaml

import celestack._discovery as discovery
import celestack._utils as utils
from celestack import PROGRESS_BAR
from celestack.frame import AverageLight, DarkFrame, LightFrame, Mask, MasterDark
from celestack.segment import SegmentBox
from celestack.stars_table import StarsTable


class FrameStack:
    """A class to manage a stack of frames, as the core object of the Celestack app."""

    def __init__(self, project: str) -> None:
        """Initialize the FrameStack object.

        The stack takes care of the dark-frame correction, foreground masking and
        ultimately the star detection.

        The stars (once detected) are stored in the `stars_table` attribute as a
        StarsTable instance.

        This initializer only applies if the stack does not yet exist in the project.
        For an already existing stack, use the `from_state` class method instead.

        Assumptions:
        - The stack has not yet been initialized in the project.
        - All the light frames and dark frames have the same pixel resolution and have
          been taken with the same camera and lens, as part of one continuous sequence.
        - The light frames have names which are in alphabetical order, so that that
          the first frame name after sorting is the frame taken first in the sequence.

        Args:
            project: The name of the project.
        """
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

        self.stars_table: StarsTable | None = None

        self.star_finder_params: dict[str, float | None] = {
            "fwhm": None,  # optimum will be auto-detected.
            "max_roundness": 1.7,  # stars with higher |roundness| will be rejected.
            "min_separation": 1.5,  # minimum separation between stars in [fwhm].
        }

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
        else:
            frame = next(iter(self.light_frames.values()))
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

        Assumptions:
        - The stack does not yet contain any light frames or dark frames.
        - All the frames have unique names - no two light frames or dark frames
          have the same name.

        Args:
            lf_paths: A list of paths to the light frames.
            df_paths: A list of paths to the dark frames.
        """
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

        Assumptions:
        - The stack contains at least one dark frame.
        - The dark frame correction has not yet been applied.
        - The dark frame correction is assumed to be the first processing step, so no
          subsequent processing steps must have been applied yet.
        """
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

        Assumptions:
        - The stack contains at least one light frame.
        - The average-light frame has not yet been created.
        """
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

        Assumptions:
        - The stack does not yet contain a mask.
        - The mask which is being applied belongs to the same project as the stack.
        - The mask has the same pixel dimensions as the light frames in the stack.

        Args:
            mask: The mask to add to the stack.
        """
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

        Assumptions:
        - No stars have yet been detected in the stack.

        Args:
            n_segments: The number of segments to create. Defaults to 50.
        """
        segment_boxes: list[SegmentBox] = []

        # First, figure out how many pixels we have in the sky:
        if self.mask is not None:
            ma = self.mask.mask_array  # Mask array, with False for sky
        else:
            ma = np.zeros((self.height, self.width), dtype=bool)
        h, _ = ma.shape

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

        Assumptions:
        - The reference frame must be one of the light frames in the stack.
        - No stars have yet been detected in the stack.

        Args:
            ref_frame_name: The name of the reference frame. Must exist in the stack.
        """
        # Set the reference frame and save it into the project:
        self.ref_frame = self.light_frames[ref_frame_name]
        self.dump_state()

    def _get_frames_times(self) -> list[float]:
        """Get the time coordinates of all the frames in the stack.

        Get the time-like distance of each frame from the reference frame.
        If the frames have exif data with the timestamp, we'll use that and the `t` will
        be in seconds.

        The reference frame will have `t` equal to 0, and the other frames will have
        `t` equal to the time difference from the reference frame, with frames taken
        before the reference frame having negative `t` values.

        The order of the returned array is the same as the order of the frames in the
        stack.

        Assumptions:
        - The reference frame has already been set.
        - The light frames in the stack have exif data with the timestamp.

        Returns:
            The list of times of capture for each frame, relative to the reference
            frame. The t is in [seconds] and the t of the reference frame is 0.
            The length of the list is equal to the number of light frames in the stack.

        Raises:
            NotImplementedError: If the frames do not have exif data with the timestamp.
        """
        ref_frame = cast("LightFrame", self.ref_frame)

        abs_times = np.array([lf.time() for lf in self.light_frames.values()])
        ref_time = ref_frame.time()

        t_array = abs_times - ref_time
        return list(t_array)

    def _find_optimal_fwhm(self) -> float:
        """Find the optimal star FWHM for the star finder.

        The method will pick 5 segments (closest to the center and each corner of the
        image) and find the FWHM which yields the most stars (with relatively high
        threshold).

        The average of these values will be considered the optimal FWHM for the
        star finder and it will be stored (persistently) in the stack's
        `star_finder_params` dictionary.

        Assumptions:
        - The sky has already been segmented into segment boxes.
        - The reference frame has already been set.

        Returns:
            The optimal FWHM for the star finder.
        """
        ref_frame = cast("LightFrame", self.ref_frame)  # assumed set

        # Choose 5 segment boxes in representative locations:
        chosen_boxes: list[SegmentBox] = []
        w, h = ref_frame.width, ref_frame.height
        for y, x in [(0, 0), (0, w), (h, 0), (h, w), (h / 2, w / 2)]:
            closest_box = min(
                self.segment_boxes,
                key=lambda b: (b.x - x) ** 2 + (b.y - y) ** 2,
            )
            chosen_boxes.append(closest_box)

        # For each chosen segment, find the optimal FWHM:
        optimal_fwhms = []
        for box in PROGRESS_BAR(chosen_boxes, "Finding optimal FWHM for star finder"):
            # Get the segment and some parameters:
            segment = ref_frame.get_segment(box)
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

        Assumptions:
        - The `min_separation` and `fwhm` parameters must be set (non-None) in the
          `star_finder_params` dictionary.

        Returns:
            The margin in pixels, which is the minimum separation between stars
            multiplied by the FWHM of the stars.
            This is used to define the search area for the star finder algorithm.
        """
        min_separation = cast("float", self.star_finder_params["min_separation"])
        fwhm = cast("float", self.star_finder_params["fwhm"])

        return round(min_separation * fwhm)

    def detect_stars_in_ref_frame(self, n: int = 1000) -> None:
        """Detect roughly N stars in the reference frame.

        The algorithm will go from segment to segment (the sky must have been segmented
        already) and detect an appropriate number of stars in each segment, to reach
        the target `n` stars in the whole frame.

        The stars detection happens in  the reference frame only,
        and it is the first step in the stars registration and alignment process.

        The stars are stored (persistently) in the `stars_table` object.

        Args:
            n: The target number of stars to detect in the whole frame. Optional, if
                not passed, a sensible default is provided.

        Assumptions:
            - The stars have not yet been propagated through the stack (the stars table
              either is not set, or it contains at most one frame - the reference
              frame).
            - The sky has already been segmented.
            - The reference frame has already been set.
            - The `max_roundness` and `min_separation` parameters are set in the
              star finder parameters.
        """
        frame = cast("LightFrame", self.ref_frame)  # assumed set

        # Find the optimal FWHM for the star finder, if not set yet:
        fwhm = self.star_finder_params["fwhm"]
        if fwhm is None:
            fwhm = self._find_optimal_fwhm()

        # Retrieve the other star finder parameters from the instance:
        max_roundness = cast("float", self.star_finder_params["max_roundness"])  # set
        min_separation = cast("float", self.star_finder_params["min_separation"])  # set

        # Calculate the star density (in stars per 10,000 sky pixels) to lead to the
        # target number of stars:
        n_pixels = frame.width * frame.height
        if frame.mask_array is not None:
            n_pixels -= np.sum(frame.mask_array)
        density = n / (n_pixels / 10000)

        # Detect the stars in the frame:
        threshold = 4.0  # initial threshold for the star finding algorithm
        stars_list_frame = utils.StarsList.empty()
        for box in PROGRESS_BAR(self.segment_boxes, f"Detecting stars in {frame.name}"):
            # Get the segment:
            segment = frame.get_segment(box)
            # Find stars in the segment:
            stars_list_segment = segment.find_stars(
                density=density,
                fwhm=fwhm,
                max_roundness=max_roundness,
                min_separation=min_separation,
                init_thresh=threshold,
            )
            # Update the initial threshold for the next segment:
            threshold = round(float(stars_list_segment.threshold[0]), 9)
            # Add the stars to the frame stars list:
            stars_list_frame.extend(stars_list_segment)

        # Instantiate the stars table:
        self.stars_table = StarsTable(
            frames_names=list(self.light_frames.keys()),
            frames_times=self._get_frames_times(),
            ref_frame_name=frame.name,
            ref_stars_list=stars_list_frame,
        )

        self.dump_state()

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

        # The stars table is stored as a CSV file, rather than in the state file:
        if self.stars_table is not None:
            stars_path = discovery.get_stars_table_path(self.project)
            self.stars_table.to_csv(stars_path)

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
            stack.stars_table = StarsTable.from_csv(stars_path)

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

    def plot(self, frame: LightFrame | None = None) -> go.Figure:
        """Make the stack plot and return it as a Plotly figure.

        Optionally, the underlying frame can be selected by passing any frame object
        belonging to the same project.
        If not passed, a frame will be selected automatically from the stack.

        If any stars have been detected for the selected frame, they will be plotted
        as well.
        Additionally, if any stars have been propagated to other frames, their
        trails will be plotted with the same colors as the stars in the selected
        frame.

        Assumptions:
        - The stack contains at least one frame light frame.
        - The frame passed (if any) belongs to the project and to the stack.

        Args:
            frame: The frame to plot. If not passed, a frame will be selected
                automatically (and more or less intelligently) from the stack.

        Returns:
            A Plotly figure with the stack plot. The plot will contain the underlying
            frame, the bounding boxes of the sky segments and the stars (if they have
            already been detected).
        """
        # Start with auto-selecting the frame to plot, if not passed:
        if frame is None:
            if self.ref_frame is not None:
                frame = self.ref_frame
            else:
                frame = self.light_frames[sorted(self.light_frames)[0]]

        # Plot the selected frame:
        fig = frame.plot()

        if self.stars_table is not None:
            self.stars_table.plot_to_figure(fig, stars_frame=frame.name)

        # Plot the segment boxes:
        if self.segment_boxes:
            fig.update_layout(showlegend=True)
            segments_array = np.vstack(
                [
                    np.vstack([seg.to_polygon(), np.array([np.nan, np.nan])])
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

    def plot_star_trail(self, star_id: int) -> go.Figure:
        """Plot the trail of a single star across all frames in the stack.

        The method will plot the trail of the star across the frames in the stack.
        The star to plot can either be specified by its ID (in which case the star
        data will be queried from the `stars_table`), or by passing its star trail
        as a DataFrame returned by the `propagate_a_star` method.

        If the `trail_df` is provided, it will never be changed in place.

        The star positions across the stack will be plotted on the background of the
        frames stacked in the "brightest" mode, so that the star pixels from all the
        frames are visible.

        Assumptions:
        - The stars table has already been fully populated and propagated across
          the stack frames.

        Args:
            star_id: The ID of the star to plot. Optional, if not passed, the `trail_df`
                must be provided.

        Returns:
            A Plotly figure with the star trail plot.
        """
        stars_table = cast("StarsTable", self.stars_table)  # assumed set

        return stars_table.plot_star_trail(star_id=star_id)
