from os import PathLike
from pathlib import Path

import exifread
import numpy as np
import tifffile
import yaml
from PIL import Image

import celestack._discovery as discovery


def _read_exif_data(img_path: str | PathLike) -> dict[str, str]:
    """
    Reads the selected EXIF data from an image file.

    Args:
        img_path: The path to the image file.

    Returns:
        A dictionary containing the EXIF data.
    """
    selected_tags = [
        "Image Make",
        "Image Model",
        "EXIF DateTimeOriginal",
        "EXIF ExposureTime",
        "EXIF FNumber",
        "EXIF ISOSpeedRatings",
        "EXIF ColorSpace",
        "EXIF FocalLength",
        "EXIF LensModel",
    ]

    with open(img_path, "rb") as img_file:
        exif_data_raw = exifread.process_file(img_file, details=False)

    exif_data = {
        tag.split()[1]: str(value).strip()
        for tag, value in exif_data_raw.items()
        if tag in selected_tags
    }

    return exif_data


def _read_image_array(img_path: str | PathLike) -> np.ndarray:
    """
    Reads an image file and returns its pixel data as a NumPy array.

    Args:
        img_path: The path to the image file.

    Returns:
        A NumPy array representing the pixel data of the image.
    """
    img_path = Path(img_path)

    # Read the image to array - prefer tifffile for TIFF files and fall back to PIL
    # for other formats
    if img_path.suffix.lower() in {".tif", ".tiff"}:
        array = tifffile.imread(img_path)
    else:
        with Image.open(img_path) as img:
            array = np.array(img)

    return array


class Frame:
    """
    A base class representing a frame.
    """

    PREVIEW_DOWNSCALE_FACTOR = 1  # how many pixels to bin to create a preview

    def __init__(self, img_path: str | PathLike, project: str):
        """
        Initializes the Frame.

        The constructor takes the path to the full-resolution image and the project
        name, and it will parse the image to extract the exif data, shape and bit
        depth, and generate a preview image.

        Args:
            img_path: The path to the full-resolution full-depth image file.
            project: The name of the project to which this frame belongs.
        """
        self.img_path = str(img_path)
        self.project = project

        # stubs for the attributes populated by the _load_image method
        self.exif_data: dict[str, str] = {}
        self.width: int = 0
        self.height: int = 0
        self.dtype: str = ""

        self._load_image()

    def __repr__(self):
        return f"{self.__class__.__name__}({self.orig_path.name})"

    def dump_state(self):
        """
        A function to dump the state of the frame into the project's frames directory.

        The state is dumped into a YAML file.
        """
        state = {key: value for key, value in self.__dict__.items()}
        with open(self.state_path, "w") as state_file:
            yaml.dump(state, state_file, default_flow_style=False)

    @property
    def orig_path(self) -> Path:
        """
        A property that returns the path to the original image.
        """
        return Path(self.img_path)

    @property
    def preview_path(self) -> Path:
        """
        A property that returns the path to the preview image.
        """
        return (
            discovery.get_project_frames_dir(self.project)
            / f"{self.orig_path.stem}.tif"
        )

    @property
    def state_path(self) -> Path:
        """
        A property that returns the path to the state file.
        """
        return (
            discovery.get_project_frames_dir(self.project)
            / f"{self.orig_path.stem}.yaml"
        )

    @property
    def bit_depth(self) -> int:
        """A property that returns the bit depth of the image."""
        depth = {"uint8": 8, "uint16": 16}.get(self.dtype)
        if depth is None:
            raise ValueError(f"Unsupported bit depth: {self.dtype}.")
        return depth

    def _load_image(self):
        """
        Process the original full-res image from the specified path.

        The method will read the full-resolution image from the specified path and
        do the following into the instance:
        - Save the exif data.
        - Save the image shape (height, width).
        - Save the image bit depth (dtype).
        - Generate a preview image (8-bit grayscale, with specified compression rate).
        - Save the preview image into the project directory.
        """
        # Read the EXIF data and image array
        self.exif_data = _read_exif_data(self.orig_path)
        array = _read_image_array(self.orig_path)

        # Save the image shape and bit depth
        self.height, self.width = array.shape[:2]
        self.dtype = array.dtype.name

        # Convert grayscale using the standard luminance formula:
        if len(array.shape) == 3:
            array = np.dot(array[..., :3], [0.299, 0.587, 0.114])

        # Downscale the array - each NxN pixels will be averaged to one:
        n = self.PREVIEW_DOWNSCALE_FACTOR
        if n > 1:
            array = array[: array.shape[0] // n * n, : array.shape[1] // n * n]
            array = array.reshape(
                array.shape[0] // n,
                n,
                array.shape[1] // n,
                n,
            ).mean(axis=(1, 3))

        # Convert to 8-bit:
        array = array / (2**self.bit_depth - 1) * 255
        array[array > 255] = 255
        array = array.astype(np.uint8)

        # Dump the preview image:
        tifffile.imwrite(self.preview_path, array, compression=None)

        # Dump the state:
        self.dump_state()
