"""
A module dedicated to discovering paths and resources for Celestack.
"""

from pathlib import Path


def get_projects_root() -> Path:
    """
    Returns the root directory for Celestack projects.

    The directory will be created if it does not already exist.

    Returns:
        The path to the projects root directory.
    """
    projects_root = Path.home() / ".celestack" / "projects"
    projects_root.mkdir(parents=True, exist_ok=True)
    return projects_root


def get_project_dir(project_name: str) -> Path:
    """
    Returns the directory for a specific Celestack project.

    This function will also create the directory (and any necessary parent directories)
    if it does not already exist.

    Args:
        project_name: The name of the project.

    Returns:
        The path to the project's directory.
    """
    project_dir = get_projects_root() / project_name
    project_dir.mkdir(parents=False, exist_ok=True)
    return project_dir


def get_project_frames_dir(project_name: str) -> Path:
    """
    Returns the path for the directory in a specific Celestack project, where the
    frames are stored.

    In the frames directory, the frame states are stored in configuration files, as well
    as the compressed images and any newly created full-res images.

    If the frames directory does not exist, it will be created.

    Args:
        project_name: The name of the project.

    Returns:
        The path to the frames directory within the project's directory.
    """
    frames_dir = get_project_dir(project_name) / "frames"
    frames_dir.mkdir(parents=False, exist_ok=True)
    return frames_dir


def get_frame_state_path(project_name: str, frame_name: str) -> Path:
    """
    Returns the path to the YAML file for a specific frame in a Celestack project.

    The YAML file contains the state of the frame, including its metadata and
    configuration and it has the same name as the original image, but with a `.yaml`
    extension.

    Args:
        project_name: The name of the project.
        frame_name: The name of the frame.

    Returns:
        The path to the YAML file for the specified frame.
    """
    return get_project_frames_dir(project_name) / f"{frame_name}_state.yaml"
