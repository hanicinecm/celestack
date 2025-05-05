from os import PathLike
from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from plotly import graph_objects as go
from sklearn.cluster import KMeans

import celestack._discovery as discovery
import celestack._utils as utils
from celestack import PROGRESS_BAR
from celestack.segment import Segment, SegmentBox


class Frame:
    """
    A base class representing a frame in a celestack project.
    This base class is designed to be used as a base class for multitude of concrete
    frame types, such as DarkFrame, LightFrame, AverageFrame, etc.

    The class is a wrapper around the image file and most notably it maintains its
    own copies of the 'full' image and the 'compressed' image, which is a 8bit
    Grayscale version of the 'full' RGB full-resolution full-depth image.

    Neither of the images' arrays are loaded in memory, but rather they read lazily when
    needed.

    As the image processing workflow progresses, both the images are being updated
    in the project folder (which does not affect the original image passed to the
    initializer).

    The state peristence is done by dumping the instance state into a YAML file
    (`dicscovery.get_frame_state_path`).
    """

    # The following attributes will not get dumped to the state file for any of the
    # subclasses of the Frame class, and they need to be assigned explicitly in the
    # `from_state` method.
    NON_STATE_ATTRS = {"project", "name"}

    # The following are the wrappings that need to be done on attributes when dumping
    # to the YAML state file and loading from it:
    NATIVE_TO_YAML = {"shape": list}
    YAML_TO_NATIVE = {"shape": tuple}

    def __init__(
        self,
        project: str,
        name: str,
        img_path: str | PathLike | None = None,
        img_array: np.ndarray | None = None,
    ):
        """
        Initializes the Frame.

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
        and the compressed image), the instance should be created using the `from_state`
        class method.

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
            raise ValueError("Only one of img_path or img_array must be provided.")

        # Validate that the frame has not yet been initialized:
        if discovery.get_frame_state_path(project, name).exists():
            raise FileExistsError(
                f"Frame '{name}' already exists in project '{project}'."
            )

        # Read the full-quality image into array, if passed via path:
        if img_array is None:
            assert img_path is not None  # make type checker happy
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

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"

    @property
    def image_path(self) -> Path:
        """Returns the path to the full-quality image copy."""
        full_quality_dir = discovery.get_project_frames_dir(self.project) / "full"
        full_quality_dir.mkdir(parents=True, exist_ok=True)
        return full_quality_dir / f"{self.name}.tiff"

    @property
    def image_array(self) -> np.ndarray:
        """Returns the full-quality image copy as an array."""
        return utils.read_image(self.image_path)

    @property
    def compressed_path(self) -> Path:
        """Returns the path to the compressed image copy."""
        compressed_dir = discovery.get_project_frames_dir(self.project) / "compressed"
        compressed_dir.mkdir(parents=True, exist_ok=True)
        return compressed_dir / f"{self.name}.tiff"

    @property
    def compressed_array(self) -> np.ndarray:
        """Lazy read of the compressed image copy as an array."""
        return utils.read_image(self.compressed_path)

    @property
    def width(self) -> int:
        """Returns the width of the original image."""
        return self.shape[1]

    @property
    def height(self) -> int:
        """Returns the height of the original image."""
        return self.shape[0]

    def update_image(self, img_array: np.ndarray) -> None:
        """
        Updates the copies of the full-quality image and the compressed image in the
        project folder.

        The method will overwrite the existing images in the project folder, but it will
        never touch the original image.

        Args:
            img_array: The new full-quality image array to save.
        """
        compressed_array = utils.compress_image(img_array)
        utils.save_tiff(img_array, self.image_path, overwrite=True)
        utils.save_tiff(compressed_array, self.compressed_path, overwrite=True)

    def dump_state(self) -> None:
        """
        A function to dump the state of the frame into the frame's YAML file.

        This is the method responsible for persisting the state of the frame instance.
        """
        # Dump the state of the instance:
        state = {
            key: value
            for key, value in self.__dict__.items()
            if key not in self.NON_STATE_ATTRS
        }
        for key, func in self.NATIVE_TO_YAML.items():
            if key in state:
                state[key] = func(state[key])
        state_path = discovery.get_frame_state_path(self.project, self.name)
        with open(state_path, "w") as state_file:
            yaml.dump(state, state_file, default_flow_style=False)

    @classmethod
    def from_state(cls, project: str, name: str):
        """
        Initializes the Frame from its state file.

        Args:
            project: The name of the project to which this frame belongs.
            name: The name of the frame. This is assigned by the project and usually
                corresponds to the name of the original image.

        Returns:
            An instance of the Frame class.
        """
        # Load the state from the YAML file:
        state_path = discovery.get_frame_state_path(project, name)
        with open(state_path, "r") as state_file:
            state = yaml.safe_load(state_file)
        state |= {"project": project, "name": name}
        for key, func in cls.YAML_TO_NATIVE.items():
            if key in state:
                state[key] = func(state[key])

        # Create the instance from the state:
        instance = cls.__new__(cls)
        instance.__dict__.update(state)

        return instance

    def clear(self):
        """A method to remove all the traces of the frame from the project folder."""
        self.image_path.unlink(missing_ok=True)
        self.compressed_path.unlink(missing_ok=True)
        discovery.get_frame_state_path(self.project, self.name).unlink(missing_ok=True)

    def plot(self) -> go.Figure:
        """
        A method to plot the compressed image into a standardly styled Plotly figure.

        Returns:
            A Plotly figure containing the compressed image.
        """
        return utils.plot_image(self.compressed_array)


class DarkFrame(Frame):
    """
    A class representing a dark frame in a celestack project.

    A DarkFrame is a special kind of Frame, which wraps around the dark frame image -
    an image taken with the same camera settings as the light frames, but with the lens
    cap on.

    A DarkFrame is the building block of the MasterDark - a Stack of multiple DarkFrames
    averaged together and normally subracted from each an every LightFrame in the
    project.

    The DarkFrame class is a subclass of the Frame class, and it inherits all the
    attributes and methods of the Frame class.
    """

    def __init__(self, project: str, name: str, img_path: str | PathLike):
        super().__init__(project=project, name=name, img_path=img_path)


class LightFrame(Frame):
    """
    A class representing a light frame in a celestack project.

    A LightFrame is a special kind of Frame, which wraps around the light frame image -
    an image containing the actual sky and optionally the foreground.

    The LightFrame class is a subclass of the Frame class, and it inherits all the
    attributes and methods of the Frame class.
    """

    def __init__(self, project: str, name: str, img_path: str | PathLike):
        # Initialize some attributes specific to the LightFrame:
        self.master_dark_name: str | None = None
        self.mask_name: str | None = None

        super().__init__(project=project, name=name, img_path=img_path)

    def subtract_master_dark(self, master_dark: "MasterDark") -> None:
        """
        Subtracts the master dark frame from the light frame.

        The method will rewrite both the compressed image and the full-quality image
        in the project folder (the original images remain untouched).

        The method will also update the state of the frame in the YAML file.

        Args:
            master_dark: The master dark frame to subtract from the light frame.
        """
        # Check if the master dark is already set:
        if self.master_dark_name is not None:
            raise ValueError("Master dark is already set.")

        # Set the master dark:
        self.master_dark_name = master_dark.name

        # Subtract the master dark from the light frame (watch out for overflows):
        full_array = self.image_array.astype("float32") - master_dark.image_array
        full_array[full_array < 0] = 0
        full_array = full_array.astype(self.dtype)

        # Update the project files:
        self.update_image(full_array)
        self.dump_state()

    def assign_mask(self, mask: "Mask") -> None:
        """
        Adds a foreground mask to the light frame.

        The method will update the state of the frame in the YAML file with the mask
        frame name.

        Args:
            mask: The mask to add to the light frame.
        """
        self.mask_name = mask.name
        self.dump_state()

    @property
    def mask(self) -> "Mask":
        """
        Returns the mask frame associated with the light frame.

        If the mask is not set, it will raise an exception.
        """
        if self.mask_name is None:
            raise ValueError("Mask is not set.")
        return Mask.from_state(self.project, self.mask_name)

    def get_segment(self, box: SegmentBox) -> Segment:
        """
        Returns a Segment object representing the rectangular segment of the sky
        portion of the light frame.

        Args:
            box: The SegmentBox object defining the segment bounding box.

        Returns:
            A Segment object representing the rectangular segment of the light frame.
        """
        segment_array = self.compressed_array[box.y1 : box.y2, box.x1 : box.x2]
        mask = None
        if self.mask_name is not None:
            mask = self.mask.mask_array[box.y1 : box.y2, box.x1 : box.x2]
        return Segment(box=box, array=segment_array, mask=mask)


class Mask(Frame):
    """
    A class representing a mask frame in a celestack project.

    A Mask is a special kind of Frame, which expects the passed image to be a binary
    mask, where white pixels are the foreground, while the black pixels are the sky.
    """

    def __init__(
        self,
        project: str,
        name: str,
        img_path: str | PathLike | None = None,
        img_array: np.ndarray | None = None,
    ):
        if img_array is not None:
            # Validate that the mask is a 2D binary mask:
            if img_array.ndim != 2:
                raise ValueError("The mask array must be a 2D array.")
            if not np.array_equal(img_array, img_array.astype(bool)):
                raise ValueError("The mask array must be a boolean array.")
            # Make it into a 8bit grayscale image, to conform with the Frame class:
            img_array = img_array.astype(np.uint8) * 255

        super().__init__(
            project=project,
            name=name,
            img_path=img_path,
            img_array=img_array,
        )

    @property
    def mask_array(self) -> np.ndarray:
        """
        Returns the binary mask as a NumPy array with boolean datatype.

        The mask is a 2D array, where True values represent the foreground (to be masked
        out) and False values represent the sky.
        """
        return self.compressed_array.astype(bool)


class AverageFrame(Frame):
    """
    A class used to average multiple frames together into a new frame.

    This is just a plain averaging, without taking into account any alignment or
    transformations.

    This class is used as a base class for the MasterDark and AverageLight classes.
    """

    def __init__(
        self,
        project: str,
        name: str,
        frames: list[DarkFrame] | list[LightFrame],
        avg_func: Literal["mean", "median"] = "median",
    ):
        # Validate that the frame has not yet been initialized:
        if discovery.get_frame_state_path(project, name).exists():
            raise FileExistsError(
                f"Frame '{name}' already exists in project '{project}'."
            )

        # Store some attributes specific to the AverageFrame:
        self.frames = [frame.name for frame in frames]
        self.avg_func = avg_func

        # Call the base class constructor to create the new frame with the averaged
        # full-resolution image:
        full_array = self._average_frames(frames, avg_func)
        super().__init__(project=project, name=name, img_array=full_array)

    @staticmethod
    def _average_frames(
        frames: list[DarkFrame] | list[LightFrame], avg_func: Literal["mean", "median"]
    ) -> np.ndarray:
        """
        A static method to average multiple frames together into a new frame.

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
            raise ValueError("The list of frames cannot be empty.")

        shape = frames[0].shape
        bit_depth = frames[0].bit_depth
        dtype = frames[0].dtype
        if isinstance(frames[0], LightFrame):
            frame_type = "light frame"
        elif isinstance(frames[0], DarkFrame):
            frame_type = "dark frame"
        else:
            raise ValueError("The frames must be either LightFrame or DarkFrame.")

        if avg_func == "mean":
            # Mean is easy - just keep the rolling sum and normalize at the end:
            sum_array = np.zeros(shape=shape, dtype=np.float32)

            for frame in PROGRESS_BAR(frames, f"Averaging {frame_type}s"):
                sum_array += frame.image_array / (2**frame.bit_depth - 1)

            avg_array = (sum_array / len(frames) * (2**bit_depth - 1)).astype(dtype)

            return avg_array

        elif avg_func == "median":
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
            for slc in PROGRESS_BAR(slices, f"Averaging {frame_type}s"):
                # Get the chunks of each of the images and stack them together:
                chunk_stack = None
                # chunk_stack = np.empty((len(frames), *shape[slc]), dtype=dtype)
                for i, frame in enumerate(frames):
                    chunk = frame.image_array[slc]
                    if chunk_stack is None:
                        chunk_stack = np.empty((len(frames), *chunk.shape), dtype=dtype)
                    chunk_stack[i] = frame.image_array[slc]

                # Add the median of the chunk stack to the average array:
                assert chunk_stack is not None  # make type checker happy
                chunk_median = np.median(chunk_stack, axis=0)
                avg_array[slc] = chunk_median

            avg_array = avg_array.astype(dtype)
            return avg_array

        raise ValueError(f"Unknown average function: {avg_func}.")


class MasterDark(AverageFrame):
    """
    A class representing a master dark frame in a celestack project.

    A MasterDark is a special kind of AverageFrame, which wraps around the average of
    multiple DarkFrames.

    The MasterDark class is a subclass of the AverageFrame class, and it inherits all
    the attributes and methods of the AverageFrame and Frame classes.
    """

    def __init__(self, project: str, name: str, frames: list[DarkFrame]):
        super().__init__(project=project, name=name, frames=frames, avg_func="median")


class AverageLight(AverageFrame):
    """
    A class representing an average light frame in a celestack project.

    An AverageLight is a special kind of AverageFrame, which wraps around the average of
    multiple LightFrames.

    The AverageLight class is a subclass of the AverageFrame class, and it inherits all
    the attributes and methods of the AverageFrame and Frame classes.

    On top of that, it defines functionality to create a Mask from the average light
    frame, by the means of pixels clustering, and more manually by explicitly masking
    and/or unmasking rectangular areas of the image.
    """

    NON_STATE_ATTRS = AverageFrame.NON_STATE_ATTRS | {"clusters_array", "mask_array"}

    def __init__(self, project: str, name: str, frames: list[LightFrame]):
        """
        Initializes the AverageLight frame.

        The constructor will create a new frame by averaging the passed frames together
        and saving the result to the project folder.

        Args:
            project: The name of the project to which this frame belongs.
            name: The name of the frame. This is assigned by the project and usually
                corresponds to the name of the original image.
            frames: A list of LightFrames to average together.
        """
        # The mask-creating functionality will need some extra temporary attributes
        # which will not be persisted in the state file:
        self.clusters_array: np.ndarray | None = None  # uint8 array of int clusters
        self.mask_array: np.ndarray | None = None  # bool array - foreground is True

        super().__init__(project=project, name=name, frames=frames, avg_func="median")

    def cluster_pixels(self, n_clusters: int) -> None:
        """
        This method will cluster the pixels in the original image in the relevant
        feature space, to separate the sky pixels from the foreground pixels.

        The feature space is defined as the RGB values of the pixels, plus their
        coordinates in the image.
        The clustering is done using the K-Means algorithm.

        The output of the method is a uint8 array of the same widht and height as the
        original image, with integer indices as the cluster labels.
        The clusters are labeled in the way that the cluster with the lower labels
        should generally correspond to the foreground pixels.
        """
        array = self.image_array  # The full quality average array
        h, w, c = array.shape

        # Treat the RGB channels & pixel coordinates as features:
        features = array.reshape(-1, c)  # array of RGB values
        features = features.astype(np.float32)
        # Add the pixel coordinates:
        x_coords = np.tile(np.arange(w), h).astype(np.float32)
        y_coords = np.repeat(np.arange(h), w).astype(np.float32)
        features = np.column_stack((features, x_coords, y_coords))
        # Normalize the features:
        for i in range(features.shape[1]):
            features[:, i] -= np.mean(features[:, i])
            features[:, i] /= np.std(features[:, i])
        # Reverse the Y-axis coordinates feature, so that higher Y values (bottom of
        # the image) have smaller feature values (will help with interpretting the
        # clusters - lower cluster average means likely foreground):
        features[:, -1] = -features[:, -1]

        # Apply K-Means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(features)

        # Relabel the clusters so they are in order of their mean values:
        cluster_means = np.mean(kmeans.cluster_centers_, axis=1)
        sorted_indices = np.argsort(cluster_means)
        new_labels = np.zeros_like(labels)
        for new_label, old_label in enumerate(sorted_indices):
            new_labels[labels == old_label] = new_label
        labels = new_labels

        # Store the clusters array:
        self.clusters_array = labels.reshape(h, w).astype(np.uint8)

    def initialize_mask(self, foreground_cluster_labels: list[int]) -> None:
        """
        Initializes the mask array based on the clusters array.

        The method will take the list of foreground cluster labels and turn the
        clusters array into a boolean mask, with the foreground pixels set to True.

        The initialized mask will be stored in the `mask_array` attribute.

        Args:
            foreground_cluster_labels: A list of cluster labels that should be
                considered as foreground.
        """
        if self.clusters_array is None:
            raise ValueError("Clusters array is not initialized.")
        self.mask_array = np.isin(self.clusters_array, foreground_cluster_labels)

    def set_mask_in_box(
        self,
        value: bool,
        x1: int | None,
        x2: int | None,
        y1: int | None,
        y2: int | None,
    ) -> None:
        """
        Set the mask in a rectangular box.

        The method will set the pixels in the specified rectangular area to True in the
        mask array.

        Args:
            value: The value to set the mask to. If True, the pixels will be masked
                (foreground), if False, they will be unmasked (sky).
            x1: The left X coordinate of the box. Optional.
            x2: The right X coordinate of the box. Optional.
            y1: The top Y coordinate of the box. Optional.
            y2: The bottom Y coordinate of the box. Optional.
        """
        if self.mask_array is None:
            raise ValueError("Mask array is not initialized.")
        self.mask_array[slice(y1, y2), slice(x1, x2)] = value

    def create_mask(self, name: str) -> Mask:
        """
        Creates a mask frame from the average light frame.

        The method will create a new Mask object from the mask array and save it to the
        project folder.

        Args:
            name: The name of the mask frame. This is how the mask is referenced within
                the project.

        Returns:
            A Mask object representing the mask frame. The mask is also saved in the
            project
        """
        if self.mask_array is None:
            raise ValueError("Mask array is not initialized.")
        mask = Mask(
            project=self.project,
            name=name,
            img_array=self.mask_array,
        )
        return mask
