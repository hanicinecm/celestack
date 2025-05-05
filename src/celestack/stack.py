"""
This module contains the FrameStack class, which is used to manage a stack of frames.

Each stack is referenced only by the project name, as only a single stack is allowed
per project.

TODO: Move segmentation and stars plotting from `project` module here...
TODO: Implement the mask creation...

"""

from pathlib import Path

import yaml

import celestack._discovery as discovery
from celestack import PROGRESS_BAR
from celestack.frame import AverageLight, DarkFrame, LightFrame, Mask, MasterDark
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
        if len(set(pth.name for pth in lf_paths)) != len(lf_paths):
            raise ValueError("Light frame paths must be unique.")
        if df_paths and len(set(pth.name for pth in df_paths)) != len(df_paths):
            raise ValueError("Dark frame paths must be unique.")

        # Load the light frames and the dark frames into the project:
        for lf_path in PROGRESS_BAR(lf_paths, "Loading light frames"):
            self.light_frames[lf_path.name] = LightFrame(
                project=self.project, name=lf_path.name, img_path=lf_path
            )
        self.dump_state()
        for df_path in PROGRESS_BAR(df_paths or [], "Loading dark frames"):
            self.dark_frames[df_path.name] = DarkFrame(
                project=self.project, name=df_path.name, img_path=df_path
            )
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
