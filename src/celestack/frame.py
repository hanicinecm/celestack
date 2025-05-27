"""A module defining the Frame class and its subclasses."""

from os import PathLike
from pathlib import Path
from typing import ClassVar, Literal, Self

import numpy as np
import yaml
from plotly import graph_objects as go
from sklearn.cluster import KMeans

import celestack._discovery as discovery
import celestack._utils as utils
from celestack import PROGRESS_BAR
from celestack.segment import Segment, SegmentBox


class Frame:
    """A base class representing a frame in a celestack project.

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

    TODO: Rebrand the image copies - path_fq, path_gs, path_cmp for full quality,
        grayscale and compressed, respectively, the same with arrays.
        Currently, compressed is the 8bit grayscale image, but in the future, we will
        need downscaled copies for previews etc.
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

        # TODO: I might clean up the caching a bit at some point...
        self._cache = {}  # A cache for the image arrays

    def __repr__(self) -> str:
        """Return a string representation of the Frame instance."""
        return f"{self.__class__.__name__}({self.name})"

    @property
    def image_path(self) -> Path:
        """Path to the full-quality image copy."""
        full_quality_dir = discovery.get_project_frames_dir(self.project) / "full"
        full_quality_dir.mkdir(parents=True, exist_ok=True)
        return full_quality_dir / f"{self.name}.tiff"

    @property
    def image_array(self) -> np.ndarray:
        """The full-quality image copy as an array."""
        return utils.read_image(self.image_path)

    @property
    def compressed_path(self) -> Path:
        """Path to the compressed image copy."""
        compressed_dir = discovery.get_project_frames_dir(self.project) / "compressed"
        compressed_dir.mkdir(parents=True, exist_ok=True)
        return compressed_dir / f"{self.name}.tiff"

    @property
    def compressed_array(self) -> np.ndarray:
        """The compressed image copy as an array."""
        # TODO: clean up the caching...
        if "compressed_array" not in self._cache:
            self._cache["compressed_array"] = utils.read_image(self.compressed_path)
        return self._cache["compressed_array"]

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
        compressed_array = utils.compress_image(img_array)
        utils.save_tiff(img_array, self.image_path, overwrite=True)
        utils.save_tiff(compressed_array, self.compressed_path, overwrite=True)

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
        state |= {"project": project, "name": name, "_cache": {}}  # TODO: cache
        for key, func in cls.YAML_TO_NATIVE.items():
            if key in state:
                state[key] = func(state[key])

        # Create the instance from the state:
        instance = cls.__new__(cls)
        instance.__dict__.update(state)

        return instance

    def clear(self) -> None:
        """Remove all the traces of the frame from the project folder."""
        self.image_path.unlink(missing_ok=True)
        self.compressed_path.unlink(missing_ok=True)
        discovery.get_frame_state_path(self.project, self.name).unlink(missing_ok=True)

    def plot(self) -> go.Figure:
        """Plot the compressed image into a standardly styled Plotly figure.

        Returns:
            A Plotly figure containing the compressed image.
        """
        return utils.plot_image(self.compressed_array)


class DarkFrame(Frame):
    """A class representing a dark frame in a celestack project.

    A DarkFrame is a special kind of Frame, which wraps around the dark frame image -
    an image taken with the same camera settings as the light frames, but with the lens
    cap on.

    A DarkFrame is the building block of the MasterDark - a Stack of multiple DarkFrames
    averaged together and normally subracted from each an every LightFrame in the
    project.

    The DarkFrame class is a subclass of the Frame class, and it inherits all the
    attributes and methods of the Frame class.
    """

    def __init__(self, project: str, name: str, img_path: str | PathLike) -> None:
        """Initialize the DarkFrame.

        Args:
            project: The name of the project to which this frame belongs.
            name: The name of the frame. This is assigned by the project and usually
                corresponds to the name of the original image.
            img_path: The path to the original dark frame image file.
        """
        super().__init__(project=project, name=name, img_path=img_path)


class LightFrame(Frame):
    """A class representing a light frame in a celestack project.

    A LightFrame is a special kind of Frame, which wraps around the light frame image -
    an image containing the actual sky and optionally the foreground.

    The LightFrame class is a subclass of the Frame class, and it inherits all the
    attributes and methods of the Frame class.
    """

    def __init__(self, project: str, name: str, img_path: str | PathLike) -> None:
        """Initialize the LightFrame.

        Args:
            project: The name of the project to which this frame belongs.
            name: The name of the frame. This is assigned by the project and usually
                corresponds to the name of the original image.
            img_path: The path to the original light frame image file.
        """
        # Initialize some attributes specific to the LightFrame:
        self.master_dark_name: str | None = None
        self.mask_name: str | None = None

        super().__init__(project=project, name=name, img_path=img_path)

    def subtract_master_dark(self, master_dark: "MasterDark") -> None:
        """Subtract the master dark frame from the light frame.

        The method will rewrite both the compressed image and the full-quality image
        in the project folder (the original images remain untouched).

        The method will also update the state of the frame in the YAML file.

        Args:
            master_dark: The master dark frame to subtract from the light frame.
        """
        # Check if the master dark is already set:
        if self.master_dark_name is not None:
            msg = "Master dark is already set."
            raise ValueError(msg)

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
        """Add a foreground mask to the light frame.

        The method will update the state of the frame in the YAML file with the mask
        frame name.

        Args:
            mask: The mask to add to the light frame.
        """
        self.mask_name = mask.name
        self.dump_state()

    @property
    def mask(self) -> "Mask":
        """The mask frame associated with the light frame.

        If the mask is not set, it will raise an exception.
        """
        if self.mask_name is None:
            msg = "Mask is not set."
            raise ValueError(msg)
        return Mask.from_state(self.project, self.mask_name)

    @property
    def mask_array(self) -> np.ndarray | None:
        """The mask array associated with the light frame.

        If the mask is not set, None will be returned.
        """
        # TODO: clean up the caching...
        if "mask_array" not in self._cache:
            self._cache["mask_array"] = None
            if self.mask_name is not None:
                self._cache["mask_array"] = self.mask.mask_array
        return self._cache["mask_array"]

    def get_segment(self, box: SegmentBox) -> Segment:
        """Return a Segment object representing the rectangular chunk of the sky.

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

    def find_star(
        self,
        pos: tuple[float, float],
        fwhm: float,
        threshold: float,
        max_roundness: float,
        min_separation: float,
    ) -> dict[str, float] | None:
        """Find a single star in the light frame in the proximity of given point.

        If the expected pixel coordinates are outside the image or in the foreground,
        the method will return None.

        The method will slice the image array around the given coordinates with the
        size of 2 * min_separation * fwhm and run the star finder inside the slice.
        If a single star is found, its data will be returned as a dictionary.
        If more than one star is found, or if no stars are found, the method will return
        None.

        Args:
            pos: The (x, y) coordinates of the expected star position in the original
                image, in pixels.
            fwhm: The full width at half maximum (FWHM) for the star finder.
            threshold: The threshold for finding stars, in units of standard deviations
                of the background noise.
            max_roundness: The maximum roundness of the stars (see the DAOStarFinder
                algorithm).
            min_separation: The minimum separation between stars, in the units of fwhm.

        Returns:
            A dictionary containing the star data, or None if no star is found.
            The dictionary contains the following keys:
                - x: The X coordinate of the star in the original image (in pixels).
                - y: The Y coordinate of the star in the original image (in pixels).
                - flux: The flux of the star.
                - threshold: The threshold used for finding the stars (in units of
                    standard deviations of the background noise).
                - fwhm: The FWHM parameter used for the star finding algo (in pixels).
        """
        # Round the coordinates to the nearest pixel:
        x, y = (round(u) for u in pos)

        # Check if the coordinates are inside the image:
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return None

        # Check if the coordinates are in the foreground:
        mask_array = self.mask_array
        if mask_array is not None and mask_array[y, x]:
            return None

        # Slice the image array around the given coordinates:
        x1 = max(0, round(x - min_separation * fwhm))
        x2 = min(self.width, round(x + min_separation * fwhm))
        y1 = max(0, round(y - min_separation * fwhm))
        y2 = min(self.height, round(y + min_separation * fwhm))
        slice_array = self.compressed_array[y1:y2, x1:x2]

        # If the slice is empty, return None:
        if slice_array.size == 0:
            return None

        # Find the stars in the slice:
        sources = utils.find_stars(
            array=slice_array,
            threshold=threshold,
            fwhm=fwhm,
            max_roundness=max_roundness,
            min_separation=min_separation,
            exclude_border=False,
        )

        if sources is None or not len(sources):
            # If no stars are found, return None:
            return None

        if len(sources) == 1:
            # If exactly one star is found, return its data:
            star = sources.iloc[0]
        else:
            # If more than one star found, return the one closer to the center:
            dist = np.sqrt(
                (sources["xcentroid"] - slice_array.shape[1] / 2) ** 2
                + (sources["ycentroid"] - slice_array.shape[0] / 2) ** 2,
            )
            star = sources.iloc[np.argmin(dist)]

        return {
            "x": round(star["xcentroid"] + x1, 2),
            "y": round(star["ycentroid"] + y1, 2),
            "flux": int(star["flux"]),
            "threshold": round(threshold, 2),
            "fwhm": fwhm,
        }


class Mask(Frame):
    """A class representing a mask frame in a celestack project.

    A Mask is a special kind of Frame, which expects the passed image to be a binary
    mask, where white pixels are the foreground, while the black pixels are the sky.
    """

    def __init__(
        self,
        project: str,
        name: str,
        img_path: str | PathLike | None = None,
        img_array: np.ndarray | None = None,
    ) -> None:
        """Initialize the Mask frame.

        Args:
            project: The name of the project to which this frame belongs.
            name: The name of the frame. This is assigned by the project and usually
                corresponds to the name of the original image.
            img_path: The path to the original mask image file. Optional, only required
                if the `img_array` is not passed.
            img_array: The binary mask image array. Optional, only required if the
                `img_path` is not passed. The mask should be a 2D array of booleans,
                where True values represent the foreground (to be masked out) and False
                values represent the sky.
        """
        if img_array is not None:
            # Validate that the mask is a 2D binary mask:
            if img_array.ndim != 2:
                msg = "The mask array must be a 2D array."
                raise ValueError(msg)
            if not np.array_equal(img_array, img_array.astype(bool)):
                msg = "The mask array must be a boolean array."
                raise ValueError(msg)
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
        """Returns the binary mask as a NumPy array with boolean datatype.

        The mask is a 2D array, where True values represent the foreground (to be masked
        out) and False values represent the sky.
        """
        return self.compressed_array.astype(bool)


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
        frames: list[DarkFrame] | list[LightFrame],
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
        full_array = self._average_frames(frames, avg_func)
        super().__init__(project=project, name=name, img_array=full_array)

    @staticmethod
    def _average_frames(
        frames: list[DarkFrame] | list[LightFrame],
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

        TODO: All the arrays (appart from the full quality ones) are cached anyway!
        TODO: Refactor to utils.
        """
        # Basic arguments validation:
        if not frames:
            msg = "The list of frames cannot be empty."
            raise ValueError(msg)

        shape = frames[0].shape
        bit_depth = frames[0].bit_depth
        dtype = frames[0].dtype
        if isinstance(frames[0], LightFrame):
            frame_type = "light frame"
        elif isinstance(frames[0], DarkFrame):
            frame_type = "dark frame"
        else:
            msg = "The frames must be either LightFrame or DarkFrame."
            raise TypeError(msg)

        if avg_func == "mean":
            # Mean is easy - just keep the rolling sum and normalize at the end:
            sum_array = np.zeros(shape=shape, dtype=np.float32)

            for frame in PROGRESS_BAR(frames, f"Averaging {frame_type}s"):
                sum_array += frame.image_array / (2**frame.bit_depth - 1)

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
                if chunk_stack is None:
                    raise ValueError  # just to make the type checker happy

                chunk_median = np.median(chunk_stack, axis=0)
                avg_array[slc] = chunk_median

            return avg_array.astype(dtype)

        msg = f"Unknown average function: {avg_func}."
        raise ValueError(msg)


class MasterDark(AverageFrame):
    """A class representing a master dark frame in a celestack project.

    A MasterDark is a special kind of AverageFrame, which wraps around the average of
    multiple DarkFrames.

    The MasterDark class is a subclass of the AverageFrame class, and it inherits all
    the attributes and methods of the AverageFrame and Frame classes.
    """

    def __init__(self, project: str, name: str, frames: list[DarkFrame]) -> None:
        """Initialize the MasterDark frame.

        The constructor will create a new frame by averaging the passed frames together
        and saving the result to the project folder.

        Args:
            project: The name of the project to which this frame belongs.
            name: The name of the frame. This is assigned by the project and usually
                corresponds to the name of the original image.
            frames: A list of DarkFrames to average together.
        """
        super().__init__(project=project, name=name, frames=frames, avg_func="median")


class AverageLight(AverageFrame):
    """A class representing an average light frame in a celestack project.

    An AverageLight is a special kind of AverageFrame, which wraps around the average of
    multiple LightFrames.

    The AverageLight class is a subclass of the AverageFrame class, and it inherits all
    the attributes and methods of the AverageFrame and Frame classes.

    On top of that, it defines functionality to create a Mask from the average light
    frame, by the means of pixels clustering, and more manually by explicitly masking
    and/or unmasking rectangular areas of the image.
    """

    NON_STATE_ATTRS = AverageFrame.NON_STATE_ATTRS | {"clusters_array", "mask_array"}

    def __init__(self, project: str, name: str, frames: list[LightFrame]) -> None:
        """Initialize the AverageLight frame.

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
        """Cluster the pixels in the original image for sky-foreground separation.

        The clustering is done in a feature space relevant for the sky-foreground
        separation.
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

    def plot_clusters(self) -> go.Figure:
        """Plot the clusters array as a Plotly figure.

        The method will create a Plotly figure containing the clusters array, with
        different colors for each cluster.

        Returns:
            A Plotly figure containing the clusters array.
        """
        if self.clusters_array is None:
            msg = "Clusters array is not initialized."
            raise ValueError(msg)
        return utils.plot_image(self.clusters_array)

    def initialize_mask(self, foreground_cluster_labels: list[int]) -> None:
        """Initialize the mask array based on the clusters array.

        The method will take the list of foreground cluster labels and turn the
        clusters array into a boolean mask, with the foreground pixels set to True.

        The initialized mask will be stored in the `mask_array` attribute.

        Args:
            foreground_cluster_labels: A list of cluster labels that should be
                considered as foreground.
        """
        if self.clusters_array is None:
            msg = "Clusters array is not initialized."
            raise ValueError(msg)
        self.mask_array = np.isin(self.clusters_array, foreground_cluster_labels)

    def set_mask_in_box(
        self,
        *,
        value: bool,
        x1: int | None = None,
        x2: int | None = None,
        y1: int | None = None,
        y2: int | None = None,
    ) -> None:
        """Set the mask in a rectangular box.

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
            msg = "Mask array is not initialized."
            raise ValueError(msg)
        self.mask_array[slice(y1, y2), slice(x1, x2)] = value

    def create_mask(self, name: str = "Mask") -> Mask:
        """Create a mask frame from the average light frame.

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
            msg = "Mask array is not initialized."
            raise ValueError(msg)
        return Mask(
            project=self.project,
            name=name,
            img_array=self.mask_array,
        )
