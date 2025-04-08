from os import PathLike
from pathlib import Path

import numpy as np
import yaml

import celestack._discovery as discovery
import celestack._utils as utils


class _Frame:
    """
    A base class representing a frame in a celestack project.
    This base class is designed to be used as a base class for multitude of concrete
    frame types, such as DarkFrame, LightFrame, MasterDark, etc.

    The class is a wrapper around the image file and most notably it maintains the
    'compressed' image, which is a 8bit Grayscale version of the full image.
    The compressed image is not loaded up in memory, but rather it is created at
    instantiation (and modified as the processing goes on) and only read lazily when
    needed.

    All the bulk work done in the celestack project is done on the compressed images to
    save resources and time, such as registering the stars, aligning the frames, etc.
    Only after all of this is done, the original images are used to create the final
    stacked image product.

    The Frame peristence is done by dumping the instance state into a YAML file
    (`dicscovery.get_frame_state_path`).
    """

    def __init__(
        self,
        name: str,
        project: str,
        img_path: str | PathLike | None = None,
        img_array: np.ndarray | None = None,
    ):
        """
        Initializes the Frame.

        There are two modes of instantiating the Frame class - by passing either path
        to the full-resolution image, or by passing the full-resolution, full-depth
        image array. In the later case, the image will be save to disk under the `name`.
        In all cases, the compressed image will be created and saved together with the
        state file.

        The constructor is designed to be used for the very first initialization of the
        frame. If the frame has already been initialized (meaning it has a state file
        and the compressed image), the instance should be created using the `from_state`
        class method.

        Args:
            name: The name of the frame. This is assigned by the project and usually
                corresponds to the name of the original image.
            project: The name of the project to which this frame belongs.
            img_path: The path to the origina full-resolution full-depth image file.
                Optional, only required if the `img_array` is not passed.
            img_array: The full-resolution, full-depth image array. Optional, only
                required if the `img_path` is not passed. If this is passed, the image
                will be saved to the project folder.
        """
        # Arguments validation:
        if img_path is not None and img_array is not None:
            raise ValueError("Only one of img_path or img_array must be provided.")

        # Read the full-resolution image into memory, or save the full resolution image
        # to disk:
        if img_path is not None:
            _img_path = img_path
            _img_array = utils.read_image(img_path)
        elif img_array is not None:
            _img_array = img_array
            _img_path = discovery.get_project_dir(project) / f"{name}.tiff"
            utils.save_tiff(img_array, _img_path)
        else:
            raise ValueError("Either img_path or img_array must be provided.")

        # The basic attributes:
        self.name = name
        self.project = project
        self.img_path = str(_img_path)

        # The attributes describin the original image:
        self.exif_data = utils.read_exif_data(self.img_path) if img_path else {}
        self.width = _img_array.shape[1]
        self.height = _img_array.shape[0]
        self.dtype = _img_array.dtype.name
        self.bit_depth = utils.get_bit_depth(_img_array)

        # Save the compressed image:
        self.update_compressed_image(utils.compress_image(_img_array))
        # Dump the state of the instance:
        self.dump_state()

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name})"

    @property
    def image_path(self) -> Path:
        """Returns the path to the original image."""
        return Path(self.img_path)

    @property
    def compressed_path(self) -> Path:
        """Returns the path to the compressed image."""
        return discovery.get_project_frames_dir(self.project) / f"{self.name}_8bit.tiff"

    @property
    def compressed_array(self) -> np.ndarray:
        """Lazy read of the compressed image as an array."""
        # TODO: If the I/O is slow, consider caching the compressed image in memory.
        return utils.read_image(self.compressed_path)

    def update_compressed_image(self, array: np.ndarray) -> None:
        """Updates the compressed image of the frame on the disk."""
        utils.save_tiff(array, self.compressed_path, overwrite=True)

    def dump_state(self) -> None:
        """
        A function to dump the state of the frame into the frame's YAML file.

        This is the method responsible for persisting the state of the frame instance.
        """
        # Dump the state of the instance:
        state = {key: value for key, value in self.__dict__.items()}
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

        # Create the instance from the state:
        instance = cls.__new__(cls)
        instance.__dict__.update(state)

        return instance


class DarkFrame(_Frame):
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

    def __init__(self, name: str, project: str, img_path: str | PathLike):
        super().__init__(name=name, project=project, img_path=img_path)


class LightFrame(_Frame):
    """
    A class representing a light frame in a celestack project.

    A LightFrame is a special kind of Frame, which wraps around the light frame image -
    an image containing the actual sky and optionally the foreground.

    The LightFrame class is a subclass of the Frame class, and it inherits all the
    attributes and methods of the Frame class.
    """

    def __init__(self, name: str, project: str, img_path: str | PathLike):
        super().__init__(name=name, project=project, img_path=img_path)


class FameStack(_Frame):
    """A class used to average multiple frames together into a new frame."""
