"""A module defining all the concrete Frame sub-classes for Celestack."""

from os import PathLike

import numpy as np
import pandas as pd
from plotly import graph_objects as go
from sklearn.cluster import KMeans

import celestack._utils as utils
from celestack._base_frame import AverageFrame, Frame
from celestack.segment import Segment, SegmentBox


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
        array_fq = self.array_fq.astype("float32") - master_dark.array_fq
        array_fq[array_fq < 0] = 0
        array_fq = array_fq.astype(self.dtype)

        # Update the project files:
        self.update_image(array_fq)
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
    def mask_array(self) -> np.ndarray | None:
        """The mask array associated with the light frame.

        If the mask is not set, None will be returned.
        Note that if the mask is set, the mask array will be cached after the first
        access.
        """
        if self.mask_name is None:
            return None

        # The mask is set - return it via the cache:
        if "mask_array" not in self._cache:
            mask = Mask.from_state(self.project, self.mask_name)
            self._cache["mask_array"] = mask.mask_array

        return self._cache["mask_array"]

    def get_segment(self, box: SegmentBox) -> Segment:
        """Return a Segment object representing the rectangular chunk of the sky.

        Args:
            box: The SegmentBox object defining the segment bounding box.

        Returns:
            A Segment object representing the rectangular segment of the light frame.
        """
        segment_array = self.array_gs[box.y1 : box.y2, box.x1 : box.x2]
        mask = None
        if self.mask_array is not None:
            mask = self.mask_array[box.y1 : box.y2, box.x1 : box.x2]
        return Segment(box=box, array=segment_array, mask=mask)

    def is_in_sky(self, pos: tuple[float, float]) -> bool:
        """Check if the the given coordinates point to sky.

        Args:
            pos: A tuple (x, y) representing the coordinates in the image.

        Returns:
            True if the position is in sky, False if it is a foreground pixel or out
            of bounds of the image.
        """
        x, y = round(pos[0]), round(pos[1])

        return (
            0 <= x < self.width
            and 0 <= y < self.height
            and (self.mask_array is None or not self.mask_array[y, x])
        )

    def find_star(
        self,
        pos: tuple[float, float],
        margin: int,
        fwhm: float,
        init_thresh: float,
        max_roundness: float,
        min_separation: float,
    ) -> dict[str, float] | None:
        """Find a single star in the light frame in the proximity of given point.

        The method will slice the image array around the given coordinates with the
        size of 2 * min_separation * fwhm and run the star finder inside the slice.

        If the star can found, its data will be returned as a dictionary.

        If the star cannot be found, the method will return None.

        If the expected pixel coordinates are outside the image or in the foreground,
        the method will raise a ValueError.

        Args:
            pos: The (x, y) coordinates of the expected star position in the original
                image, in pixels.
            margin: The margin around the expected position, in pixels. This is used to
                determine the size of the slice around the expected position, where to
                search for the star.
            fwhm: The full width at half maximum (FWHM) for the star finder.
            init_thresh: The threshold for finding star, in units of standard deviations
                of the background noise. This is the initial guess, it will be tweaked
                until a single star is found.
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

        # Check if the coordinates belong to the sky:
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            msg = "Coordinates are outside the image bounds."
            raise ValueError(msg)
        if self.mask_array is not None and self.mask_array[y, x]:
            msg = "Coordinates are in the foreground (masked) area."
            raise ValueError(msg)

        # Slice the image array around the given coordinates:
        x1 = max(0, round(x - margin))
        x2 = min(self.width, round(x + margin))
        y1 = max(0, round(y - margin))
        y2 = min(self.height, round(y + margin))
        slice_array = self.array_gs[y1:y2, x1:x2]

        def _source_to_star_dict(
            source: pd.Series, threshold: float
        ) -> dict[str, float]:
            """Convert a source Series to a star dictionary."""
            return {
                "x": round(source["xcentroid"] + x1, 2),
                "y": round(source["ycentroid"] + y1, 2),
                "flux": int(source["flux"]),
                "threshold": round(threshold, 2),
                "fwhm": round(fwhm, 2),
            }

        # First, check if there is already exactly one star at the initial threshold
        sources = utils.find_stars(
            array=slice_array,
            threshold=init_thresh,
            fwhm=fwhm,
            max_roundness=max_roundness,
            min_separation=min_separation,
            exclude_border=False,
        )
        if sources is not None and len(sources) == 1:
            return _source_to_star_dict(sources.iloc[0], threshold=init_thresh)

        # Binary search for a threshold that yields exactly one star
        # TODO: Perhaps do not dwell on a single star, but allow more and select closest
        # TODO: Implement maximal allowed distance from the expected position
        if sources is None or not len(sources):
            min_thresh = 2.0  # Arbitrary lower limit for the search
            max_thresh = init_thresh
        else:
            min_thresh = init_thresh
            max_thresh = 20.0  # Arbitrary upper limit for the search

        while max_thresh - min_thresh > 0.05:
            mid_thresh = (min_thresh + max_thresh) / 2
            sources = utils.find_stars(
                array=slice_array,
                threshold=mid_thresh,
                fwhm=fwhm,
                max_roundness=max_roundness,
                min_separation=min_separation,
                exclude_border=False,
            )
            if sources is None or not len(sources):
                max_thresh = mid_thresh
            elif len(sources) > 1:
                min_thresh = mid_thresh
            else:
                # Found exactly one star
                return _source_to_star_dict(sources.iloc[0], threshold=mid_thresh)

        return None


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
        return self.array_gs.astype(bool)


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
        super().__init__(
            project=project,
            name=name,
            frames=frames,  # type: ignore[call-arg]
            avg_func="median",
        )


class AverageLight(AverageFrame):
    """A class representing an average light frame in a celestack project.

    An AverageLight is a special kind of AverageFrame, which wraps around the average of
    multiple LightFrames.

    The AverageLight class is a subclass of the AverageFrame class, and it inherits all
    the attributes and methods of the AverageFrame and Frame classes.

    On top of that, it defines functionality to create a Mask from the averaged
    array, by the means of pixels clustering, and more manually by explicitly masking
    and/or unmasking rectangular areas of the image.
    """

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
        self._clusters_array: np.ndarray | None = None  # uint8 array of int clusters
        self._mask_array: np.ndarray | None = None  # bool array - foreground is True

        super().__init__(
            project=project,
            name=name,
            frames=frames,  # type: ignore[call-arg]
            avg_func="median",
        )

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
        array = self.array_fq  # The full quality average array
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
        self._clusters_array = labels.reshape(h, w).astype(np.uint8)

    def plot_clusters(self) -> go.Figure:
        """Plot the clusters array as a Plotly figure.

        The method will create a Plotly figure containing the clusters array, with
        different colors for each cluster.

        Returns:
            A Plotly figure containing the clusters array.
        """
        if self._clusters_array is None:
            msg = "Clusters array is not initialized."
            raise ValueError(msg)
        return utils.plot_image(self._clusters_array, interactive=True)

    def initialize_mask(self, foreground_cluster_labels: list[int]) -> None:
        """Initialize the mask array based on the clusters array.

        The method will take the list of foreground cluster labels and turn the
        clusters array into a boolean mask, with the foreground pixels set to True.

        The initialized mask will be stored in the `mask_array` attribute.

        Args:
            foreground_cluster_labels: A list of cluster labels that should be
                considered as foreground.
        """
        if self._clusters_array is None:
            msg = "Clusters array is not initialized."
            raise ValueError(msg)
        self._mask_array = np.isin(self._clusters_array, foreground_cluster_labels)

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
        if self._mask_array is None:
            msg = "Mask array is not initialized."
            raise ValueError(msg)
        self._mask_array[slice(y1, y2), slice(x1, x2)] = value

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
        if self._mask_array is None:
            msg = "Mask array is not initialized."
            raise ValueError(msg)
        return Mask(
            project=self.project,
            name=name,
            img_array=self._mask_array,
        )
