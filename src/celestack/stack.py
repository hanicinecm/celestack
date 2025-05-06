"""
This module contains the FrameStack class, which is used to manage a stack of frames.

Each stack is referenced only by the project name, as only a single stack is allowed
per project.

TODO: Move segmentation and stars plotting from `project` module here...
"""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import yaml

import celestack._discovery as discovery
from celestack import PROGRESS_BAR
from celestack.frame import AverageLight, DarkFrame, Frame, LightFrame, Mask, MasterDark
from celestack.segment import SegmentBox


class FrameStack:
    """
    A class to manage a stack of frames, as the core object of the Celestack app.
    """

    def __init__(self, project: str):
        """
        Initializes the FrameStack object.

        Args:
            project: The name of the project.
        """
        # Instantiation is only allowed if the stack does not yet exist in the project:
        if discovery.get_stack_state_path(project).exists():
            raise FileExistsError(f"Stack already exists in project '{project}'.")

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

        # Dump the initial state of the stack to the state file:
        self.dump_state()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.project})"

    def load_frames(
        self, lf_paths: list[Path], df_paths: list[Path] | None = None
    ) -> None:
        """
        A method to load all the light frames and the dark frames into the stack.

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
            raise ValueError("Stack contains some frames already.")
        if len(set(pth.stem for pth in lf_paths)) != len(lf_paths):
            raise ValueError("Light frame names must be unique.")
        if df_paths and len(set(pth.stem for pth in df_paths)) != len(df_paths):
            raise ValueError("Dark frame names must be unique.")

        # Load the light frames and the dark frames into the project:
        for lf_path in PROGRESS_BAR(lf_paths, "Loading light frames"):
            lf = LightFrame(project=self.project, name=lf_path.stem, img_path=lf_path)
            self.light_frames[lf.name] = lf
            self.dump_state()

        for df_path in PROGRESS_BAR(df_paths or [], "Loading dark frames"):
            df = DarkFrame(project=self.project, name=df_path.stem, img_path=df_path)
            self.dark_frames[df.name] = df
            self.dump_state()

        self.light_frames = dict(sorted(self.light_frames.items()))
        self.dark_frames = dict(sorted(self.dark_frames.items()))
        self.dump_state()

    def apply_dark_frames_correction(self) -> None:
        """
        A method to create the master-dark frame and subtract it from all the light
        frames in the stack.

        Raises:
            ValueError: If the stack does not contain any dark frames.
            ValueError: If the stack contains the master-dark frame already.
            ValueError: If the stack contains the average-light frame already (which is
                supposed to be created from dark-frame-corrected light frames).
        """
        # Basic sanity checks:
        if not self.dark_frames:
            raise ValueError("Stack does not contain any dark frames.")
        if self.master_dark:
            raise ValueError("Stack already contains the master-dark frame.")
        if self.avg_light:
            raise ValueError("Stack already contains the average-light frame.")

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
        """
        A method to create the average-light frame from the light frames in the stack.

        Raises:
            ValueError: If the stack does not contain any light frames.
            ValueError: If the stack contains the average-light frame already.
        """
        # Basic sanity checks:
        if not self.light_frames:
            raise ValueError("Stack does not contain any light frames.")
        if self.avg_light:
            raise ValueError("Stack already contains the average-light frame.")

        # Create the average-light frame and save it into the project:
        self.avg_light = AverageLight(
            project=self.project,
            name="AverageLight",
            frames=list(self.light_frames.values()),
        )
        self.dump_state()

    def add_mask(self, mask: Mask) -> None:
        """
        A method to add a mask to the stack.

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
            raise ValueError("Stack already contains a mask.")
        if mask.project != self.project:
            raise ValueError(
                "Mask does not belong to the same project: "
                f"{mask.project} != {self.project}"
            )

        # Add the mask to the stack and save it into the project:
        self.mask = mask
        self.dump_state()

        # Assign the mask to all the light frames in the stack:
        for lf in self.light_frames.values():
            lf.assign_mask(mask)

    def segment_sky(self, n_segments: int = 50) -> None:
        """
        Segment the sky portion of the mask into N rectangular segments, with roughly
        the same number of sky pixels in each segment.

        The segments are defined by their bounding boxes and stored in the stack object
        as a list of SegmentBox objects.

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
            raise ValueError("Stack does not contain a mask.")
        # TODO: Check if any stars have already been detected.

        segment_boxes: list[SegmentBox] = []

        # First, figure out how many pixels we have in the sky:
        ma = self.mask.mask_array  # This is a boolean array with False for sky pixels
        h, w = self.mask.shape

        # Lets define the xmin, xmax of the bounding box of the sky:
        xmin, xmax = np.where(np.sum(ma, axis=0) < h)[0][[0, -1]]

        # Let's define the nominal segment size (the segments will be nominally square):
        n_px = np.sum(~ma)  # Total number of sky pixels
        size_nom = int(np.sqrt(n_px // n_segments))

        # The segments will be defined in columns - let's define the width of the
        # column as how many times the nominal size fits between the xmin and xmax:
        n_cols = int(round((xmax - xmin) / size_nom))
        width_nom = (xmax - xmin) // n_cols

        x1 = xmin
        for i in range(n_cols):
            x2 = x1 + width_nom if i < n_cols - 1 else xmax

            # Now we need to find the ymin and ymax of the bounding box of the sky in
            # the column:
            ymin, ymax = np.where(np.sum(ma[:, x1:x2], axis=1) < (x2 - x1))[0][[0, -1]]

            n_px_col = np.sum(
                ~ma[ymin:ymax, x1:x2]
            )  # Number of sky pixels in the column
            n_rows = int(round(n_px_col / width_nom**2))
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
                        SegmentBox(x1=int(x1), y1=int(y1), x2=int(x2), y2=int(ymax))
                    )
                    if i % 2 == 1:
                        segment_boxes_col.reverse()
                    segment_boxes.extend(segment_boxes_col)
                    segment_boxes_col = []
                    break

                # Check if we have enough pixels in the segment:
                if np.sum(~ma[ymin:y2, x1:x2]) >= j * n_px_seg:
                    segment_boxes_col.append(
                        SegmentBox(x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2))
                    )
                    y1 = y2
                    j += 1

            x1 = x2

        # Store the segment boxes in the stack and dump the state:
        self.segment_boxes = segment_boxes
        self.dump_state()

    def set_reference_frame(self, ref_frame_name: str) -> None:
        """
        A method to set the reference frame for the stack.

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
            raise ValueError(
                f"Reference frame '{ref_frame_name}' does not belong to the stack."
            )
        # TODO: Check if any stars have already been detected.

        # Set the reference frame and save it into the project:
        self.ref_frame = self.light_frames[ref_frame_name]
        self.dump_state()

    def detect_stars(self) -> None:
        raise NotImplementedError("Detecting stars is not implemented yet.")

    def dump_state(self) -> None:
        """
        Dumps the current state of the stack to a YAML file.

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
        }

        # Dump the state to a YAML file:
        state_path = discovery.get_stack_state_path(self.project)
        with open(state_path, "w") as state_file:
            yaml.dump(state, state_file, default_flow_style=False)

    @classmethod
    def from_state(cls, project: str) -> "FrameStack":
        """
        Loads the stack state file and instantiates the FrameStack object from it.

        Args:
            project: The name of the project.

        Returns:
            A FrameStack object loaded from the state file.
        """
        # Load the state from the YAML file:
        state = {"project": project}
        state_path = discovery.get_stack_state_path(project)
        with open(state_path, "r") as state_file:
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

        # Create the FrameStack object:
        stack = cls.__new__(cls)
        stack.__dict__.update(state)

        return stack

    def clear(self) -> None:
        """A method to remove all the traces of the stack from the project folder."""
        discovery.get_stack_state_path(self.project).unlink(missing_ok=True)

    def plot(self, frame: Frame | None = None) -> go.Figure:
        """
        Make the stack plot and return it as a Plotly figure.

        The stack plot will contain the bounding boxes of the sky segments (if the
        sky has already been segmented) and the stars (if they have already been
        detected).

        Optionally, the underlying frame can be selected by passing any frame object
        belonging to the same project.
        If not passed, a frame will be selected automatically from the stack.

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
                    raise ValueError("Stack does not contain any frames.")
                frame = self.light_frames[sorted(self.light_frames)[0]]
        assert frame is not None, "Defensive programming, mainly for the type checker."

        # Basic sanity checks:
        if frame.project != self.project:
            raise ValueError(
                f"Frame '{frame.name}' does not belong to the stack '{self.project}'."
            )

        # Plot the selected frame:
        fig = frame.plot()

        # Add the segment boxes to the plot:
        if self.segment_boxes:
            x, y = [], []
            for box in self.segment_boxes:
                x.extend([box.x1, box.x2, box.x2, box.x1, box.x1, np.nan])
                y.extend([box.y1, box.y1, box.y2, box.y2, box.y1, np.nan])
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    line=dict(color="blue", width=0.5),
                    name="Segment boxes",
                )
            )

        # Add the stars to the plot:
        # TODO: Implement the stars plotting.
        # if stars:
        #     fig.add_trace(
        #         go.Scatter(
        #             x=stars["x"],
        #             y=stars["y"],
        #             mode="markers",
        #             marker=dict(
        #                 size=stars["flux"] / np.max(stars["flux"]) * 10,
        #                 color=stars["flux"],
        #                 colorscale="Viridis",
        #             ),
        #             name="Stars",
        #         )
        #     )

        #     fig.update_layout(showlegend=True)

        return fig
