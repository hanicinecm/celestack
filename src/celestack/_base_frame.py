"""A module defining the Frame class.

The Frame class is a base class for all the concrete Frames in the Celestack project,
such as the `LightFrame`, `DarkFrame`, etc.
"""

from os import PathLike
from pathlib import Path
from typing import ClassVar, Literal, Self

import numpy as np
import yaml
from plotly import graph_objects as go

import celestack._discovery as discovery
import celestack._utils as utils
from celestack import PROGRESS_BAR


class Frame:
    """A base class representing a frame in a celestack project.

    This base class is designed to be used as a base class for multitude of concrete
    frame types, such as DarkFrame, LightFrame, AverageFrame, etc.

    The class is a wrapper around the image file and most notably it maintains its
    own copies of the 'full quality' image and the 'grayscale' 8bit image.

    Neither of the images' arrays are loaded in memory, but rather they read lazily when
    needed, with the grayscale image, however, being cached in memory after the first
    read.
    The `_cache` attribute is used to store the cached image arrays. It can be used
    by the subclasses to cache other image arrays as well.
    Calling the `clear_cache` method will invalidate the cache.

    As the image processing workflow progresses, both the images are being updated
    in the project folder (which does not affect the original image passed to the
    initializer). Each update will overwrite the existing images in the project
    folder and clear the cache.

    The state peristence is done by dumping the instance state into a YAML file
    (`dicscovery.get_frame_state_path`).
    """

    # The following attributes will not get dumped to the state file for any of the
    # subclasses of the Frame class, and they need to be assigned explicitly in the
    # `from_state` method.
    NON_STATE_ATTRS: ClassVar = {"project", "name"}

    # The following are the wrappings that need to be done on attributes when dumping
    # to the YAML state file and loading from it:
    NATIVE_TO_YAML: ClassVar = {"shape": list}
    YAML_TO_NATIVE: ClassVar = {"shape": tuple}

    def __init__(
        self,
        project: str,
        name: str,
        img_path: str | PathLike | None = None,
        img_array: np.ndarray | None = None,
    ) -> None:
        """Initialize the Frame.

        There are two modes of instantiating the Frame class - by passing either path
        to the original full-resolution image, or by passing the full-resolution,
        full-depth image array.

        In the first case, the original image will be copied to the project folder for
        further processing.
        In the later case, the image will be save to the project folder.
        In all cases, the compressed image will be created and saved together with the
        state file.

        The constructor is designed to be used for the very first initialization of the
        frame. If the frame has already been initialized (meaning it has a state file
        and the image copies in the project folder already), the instance should be
        created using the `from_state` class method.

        The class will work both with RGB images (light frames, dark frames, etc) and
        with grayscale images (used for instantiating the Mask class).

        Args:
            project: The name of the project to which this frame belongs.
            name: The name of the frame. This is assigned by the project and usually
                corresponds to the name of the original image.
            img_path: The path to the origina full-resolution full-depth image file.
                Optional, only required if the `img_array` is not passed.
            img_array: The full-resolution, full-depth image array. Optional, only
                required if the `img_path` is not passed. If this is passed, the image
                will be saved to the project folder.
        """
        # Arguments validation:
        if img_path is not None and img_array is not None:
            msg = "Only one of img_path or img_array must be provided."
            raise ValueError(msg)

        # Validate that the frame has not yet been initialized:
        if discovery.get_frame_state_path(project, name).exists():
            msg = f"Frame '{name}' already exists in project '{project}'."
            raise FileExistsError(
                msg,
            )

        # Read the full-quality image into array, if passed via path:
        if img_array is None:
            if img_path is None:
                raise ValueError  # just to make the type checker happy
            img_array = utils.read_image(img_path)

        # The basic attributes:
        self.project = project
        self.name = name
        self.original_path = str(img_path) if img_path else None

        # The attributes describing the image:
        self.exif_data = utils.read_exif_data(img_path) if img_path else {}
        self.shape: tuple[int, ...] = img_array.shape
        self.dtype = str(img_array.dtype.name)
        self.bit_depth = utils.get_bit_depth(img_array)

        # Save the full-quality and compressed image copies to the project folder:
        self.update_image(img_array)
        # Dump the state of the instance:
        self.dump_state()

        # A cache for the image arrays (it's private, so won't be dumped to the state):
        self._cache: dict[str, np.ndarray] = {}

    def __repr__(self) -> str:
        """Return a string representation of the Frame instance."""
        return f"{self.__class__.__name__}({self.name})"

    def clear_cache(self) -> None:
        """Clear the cache of the image arrays."""
        self._cache.clear()

    @property
    def path_fq(self) -> Path:
        """Path to the full-quality image copy."""
        full_quality_dir = (
            discovery.get_project_frames_dir(self.project) / "full_quality"
        )
        full_quality_dir.mkdir(parents=True, exist_ok=True)
        return full_quality_dir / f"{self.name}.tiff"

    @property
    def array_fq(self) -> np.ndarray:
        """The full-quality image copy as an array."""
        return utils.read_image(self.path_fq)

    @property
    def path_gs(self) -> Path:
        """Path to the grayscal 8bit image copy."""
        grayscale_dir = discovery.get_project_frames_dir(self.project) / "grayscale"
        grayscale_dir.mkdir(parents=True, exist_ok=True)
        return grayscale_dir / f"{self.name}.tiff"

    @property
    def array_gs(self) -> np.ndarray:
        """The grayscale 8bit image copy as an array.

        Note, that the array is cached, until the `clear_cache` method is called.
        """
        if "array_gs" not in self._cache:
            self._cache["array_gs"] = utils.read_image(self.path_gs)
        return self._cache["array_gs"]

    @property
    def width(self) -> int:
        """The width of the original image."""
        return self.shape[1]

    @property
    def height(self) -> int:
        """The height of the original image."""
        return self.shape[0]

    def update_image(self, img_array: np.ndarray) -> None:
        """Update all the image copies in the project folder.

        The method will overwrite the existing images in the project folder, but it will
        never touch the original image.

        Args:
            img_array: The new full-quality image array to save.
        """
        array_gs = utils.compress_to_grayscale(img_array)
        utils.save_tiff(img_array, self.path_fq, overwrite=True)
        utils.save_tiff(array_gs, self.path_gs, overwrite=True)
        self.clear_cache()

    def dump_state(self) -> None:
        """Dump the state of the frame into the frame's YAML file.

        This is the method responsible for persisting the state of the frame instance.
        """
        # Dump the state of the instance:
        state = {
            key: value
            for key, value in self.__dict__.items()
            if key not in self.NON_STATE_ATTRS and not key.startswith("_")
        }
        for key, func in self.NATIVE_TO_YAML.items():
            if key in state:
                state[key] = func(state[key])
        state_path = discovery.get_frame_state_path(self.project, self.name)
        with state_path.open("w") as state_file:
            yaml.dump(state, state_file, default_flow_style=False)

    @classmethod
    def from_state(cls, project: str, name: str) -> Self:
        """Initialize the Frame from its state file.

        Args:
            project: The name of the project to which this frame belongs.
            name: The name of the frame. This is assigned by the project and usually
                corresponds to the name of the original image.

        Returns:
            An instance of the Frame class.
        """
        # Load the state from the YAML file:
        state_path = discovery.get_frame_state_path(project, name)
        with state_path.open() as state_file:
            state = yaml.safe_load(state_file)
        state |= {"project": project, "name": name, "_cache": {}}
        for key, func in cls.YAML_TO_NATIVE.items():
            if key in state:
                state[key] = func(state[key])

        # Create the instance from the state:
        instance = cls.__new__(cls)
        instance.__dict__.update(state)

        return instance

    def clear(self) -> None:
        """Remove all the traces of the frame from the project folder."""
        self.path_fq.unlink(missing_ok=True)
        self.path_gs.unlink(missing_ok=True)
        discovery.get_frame_state_path(self.project, self.name).unlink(missing_ok=True)
        self.clear_cache()

    def plot(self) -> go.Figure:
        """Plot the compressed image into a standardly styled Plotly figure.

        Returns:
            A Plotly figure containing the compressed image.
        """
        return utils.plot_image(self.array_gs)


class AverageFrame(Frame):
    """A class used to average multiple frames together into a new frame.

    This is just a plain averaging, without taking into account any alignment or
    transformations.

    This class is used as a base class for the MasterDark and AverageLight classes.
    """

    def __init__(
        self,
        project: str,
        name: str,
        frames: list[Frame],
        avg_func: Literal["mean", "median"] = "median",
    ) -> None:
        """Initialize the AverageFrame.

        The constructor will create a new frame by averaging the passed frames together
        and saving the result to the project folder.

        Args:
            project: The name of the project to which this frame belongs.
            name: The name of the frame. This is assigned by the project and usually
                corresponds to the name of the original image.
            frames: A list of DarkFrames or LightFrames to average together.
            avg_func: The function to use for averaging. Can be either "mean" or
                "median". Defaults to "median".
        """
        # Validate that the frame has not yet been initialized:
        if discovery.get_frame_state_path(project, name).exists():
            msg = f"Frame '{name}' already exists in project '{project}'."
            raise FileExistsError(
                msg,
            )

        # Store some attributes specific to the AverageFrame:
        self.frames = [frame.name for frame in frames]
        self.avg_func = avg_func

        # Call the base class constructor to create the new frame with the averaged
        # full-resolution image:
        array_fq = self._average_frames(frames, avg_func)
        super().__init__(project=project, name=name, img_array=array_fq)

    @staticmethod
    def _average_frames(
        frames: list[Frame],
        avg_func: Literal["mean", "median"],
    ) -> np.ndarray:
        """Average multiple frames together into a new frame.

        I need to average together potentially large number of potentially very large
        images, so this cannot be done by loading all of them into memory at once.

        Args:
            frames: A list of frames, whose arrays are to be stacked together and
                averaged.
            avg_func: The function to use for averaging.

        Returns:
            A NumPy array representing the averaged image, with the same shape and
            data type as any of the original images.
        """
        # Basic arguments validation:
        if not frames:
            msg = "The list of frames cannot be empty."
            raise ValueError(msg)

        shape = frames[0].shape
        bit_depth = frames[0].bit_depth
        dtype = frames[0].dtype

        if avg_func == "mean":
            # Mean is easy - just keep the rolling sum and normalize at the end:
            sum_array = np.zeros(shape=shape, dtype=np.float32)

            for frame in PROGRESS_BAR(frames, "Averaging frames"):
                sum_array += frame.array_fq / (2**frame.bit_depth - 1)

            return (sum_array / len(frames) * (2**bit_depth - 1)).astype(dtype)

        if avg_func == "median":
            # Median is non-linear and cannot be build sequentially... we'll build it
            # chunk by chunk:
            avg_array = np.zeros(shape=shape, dtype=dtype)

            # How few chunks can we afford?
            memory_budget = 3000  # MB
            array_size = np.prod(shape) * np.dtype(dtype).itemsize / 1024**2
            total_size = array_size * len(frames)  # some overhead
            n = int(np.ceil(2 * total_size / memory_budget))  # factor 2 empirically

            # Build the average array chunk-by-chunk:
            slices = [slice(i, None, n) for i in range(n)]
            for slc in PROGRESS_BAR(slices, "Averaging frames"):
                # Get the chunks of each of the images and stack them together:
                chunk_stack = None
                # chunk_stack = np.empty((len(frames), *shape[slc]), dtype=dtype)
                for i, frame in enumerate(frames):
                    chunk = frame.array_fq[slc]
                    if chunk_stack is None:
                        chunk_stack = np.empty((len(frames), *chunk.shape), dtype=dtype)
                    chunk_stack[i] = frame.array_fq[slc]

                # Add the median of the chunk stack to the average array:
                if chunk_stack is None:
                    raise ValueError  # just to make the type checker happy

                chunk_median = np.median(chunk_stack, axis=0)
                avg_array[slc] = chunk_median

            return avg_array.astype(dtype)

        msg = f"Unknown average function: {avg_func}."
        raise ValueError(msg)
