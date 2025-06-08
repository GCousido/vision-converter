from abc import ABC, abstractmethod
from typing import Generic, Optional, TypeVar, Union

class BoundingBox(ABC):
    """Abstract base class representing a bounding box structure.
    
    Subclasses must implement the getBoundingBox method to provide
    coordinate values in a standardized format.
    """

    @abstractmethod
    def getBoundingBox(self) -> list:
        pass

class Shape(ABC):
    """Abstract base class representing a Shape"""
    shape_type: str

    def __init__(self, type: str) -> None:
        self.shape_type = type

    @abstractmethod
    def getCoordinates(self) -> list:
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
    """
    name: str
    files: list[X]
    folder_path: Optional[str]

    def __init__(self, name: str, files: list[X], folder_path: Optional[str] = None) -> None:
        self.name = name
        self.files = files
        self.folder_path = folder_path