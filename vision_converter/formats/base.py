from abc import ABC, abstractmethod
from pathlib import Path
import shutil
from typing import Any, Generic, Optional, TypeVar, Union

from .bounding_box import BoundingBox, PascalVocBoundingBox

class Shape(ABC):
    """Abstract base class representing a Shape"""
    shape_type: str

    def __init__(self, type: str) -> None:
        self.shape_type = type

    @abstractmethod
    def getCoordinates(self) -> Any:
        pass

    @abstractmethod
    def getBoundingBox(self) -> PascalVocBoundingBox:
        pass

T = TypeVar("T", bound=Union[BoundingBox, Shape])

class Annotation(ABC, Generic[T]):
    """Abstract base class for object annotations with generic geometry type.
    
    Type Parameters:
        T (Union[BoundingBox, Shape]): Type of geometry implementation to use

    Attributes:
        geometry (T): Concrete geometry instance (BoundingBox or Shape)
    """
    geometry: T

    def __init__(self, geometry: T) -> None:
        self.geometry = geometry


K = TypeVar("K", bound=Annotation)

class FileFormat(ABC, Generic[K]):
    """Abstract base class representing a file format with annotations.
    
    Type Parameters:
        K (Annotation): Type of annotations contained in the file

    Attributes:
        filename (str): Name of the associated image file
        annotations (list[K]): List of annotations in the file
    """
    filename: str
    annotations: list[K]

    def __init__(self, filename: str, annotations: list[K]) -> None:
        self.filename = filename
        self.annotations = annotations


X = TypeVar("X", bound=FileFormat)

class DatasetFormat(ABC, Generic[X]):
    """Abstract base class representing a complete dataset format.
    
    Type Parameters:
        X (FileFormat): Type of files contained in the dataset

    Attributes:
        name (str): Name/identifier of the dataset
        files (list[X]): List of files in the dataset
        folder_path (Optional[str]): Optional filesystem path to dataset root
        images_path_list (Optional[list[str]]): Optional list of images paths
    """
    name: str
    files: list[X]
    folder_path: Optional[str]
    images_path_list: Optional[list[str]]

    def __init__(self, name: str, files: list[X], folder_path: Optional[str] = None, images_path_list: Optional[list[str]] = None) -> None:
        self.name = name
        self.files = files
        self.folder_path = folder_path
        self.images_path_list = images_path_list

    @staticmethod
    def get_image_paths(images_dir: str, extensions=('.jpg', '.jpeg', '.png', '.bmp', '.webp')) -> list[str]:
        """
        Retrieve image file paths from a specified directory with given file extensions.

        Args:
            images_dir (str): Path to the directory containing image files.
            extensions (tuple, optional): Tuple of file extensions to search for. Defaults to ('.jpg', '.jpeg', '.png', '.bmp', '.webp').

        Returns:
            list: List of image file paths as strings.

        Raises:
            FileNotFoundError: If no image files matching the extensions are found in the directory.
        """
        image_paths: list[str] = []
        for ext in extensions:
            image_paths.extend(str(p) for p in Path(images_dir).glob(f'*{ext}'))
        if not image_paths:
            raise FileNotFoundError(f"No images found to copy at {images_dir}")
        return image_paths

    def handle_images(self, images_path_list: Optional[list[str]], images_dir: str, copy_images=False, copy_as_links=False) -> None:
        """
        Copy or create symbolic links for a list of image files into a target directory.

        Args:
            images_path_list (list): List of paths to image files to handle.
            images_dir (str): Target directory where images will be copied or linked.
            copy_images (bool, optional): If True, images are copied to the target directory. Defaults to False.
            copy_as_links (bool, optional): If True, symbolic links to the images are created in the target directory. Defaults to False.

        Raises:
            FileNotFoundError: If the provided image path list is empty.
        """
        if not images_path_list:
            raise FileNotFoundError("Images not found to copy")
        for image_path in images_path_list:
            img_name = Path(image_path).name
            dest = Path(images_dir) / img_name
            if copy_images:
                shutil.copy(image_path, dest)
            elif copy_as_links:
                dest.symlink_to(Path(image_path).resolve())