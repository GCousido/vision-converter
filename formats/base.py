from abc import ABC, abstractmethod
from typing import Generic, Optional, TypeVar

class BoundingBox(ABC):
    """Abstract base class representing a bounding box structure.
    
    Subclasses must implement the getBoundingBox method to provide
    coordinate values in a standardized format.
    """

    @abstractmethod
    def getBoundingBox(self) -> list:
        pass


T = TypeVar("T", bound=BoundingBox)

class Annotation(ABC, Generic[T]):
    """Abstract base class for object annotations with generic bounding box type.
    
    Type Parameters:
        T (BoundingBox): Type of bounding box implementation to use

    Attributes:
        bbox (T): Concrete bounding box instance
    """
    bbox: T

    def __init__(self, bbox: T) -> None:
        self.bbox = bbox

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