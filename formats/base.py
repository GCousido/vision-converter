from abc import ABC, abstractmethod
from typing import Generic, TypeVar

class BoundingBox(ABC):

    @abstractmethod
    def getBoundingBox(self) -> list:
        pass


T = TypeVar("T", bound=BoundingBox)

class Annotation(ABC, Generic[T]):
    bbox: T

    def __init__(self, bbox: T) -> None:
        self.bbox = bbox

K = TypeVar("K", bound=Annotation)

class FileFormat(ABC, Generic[K]):
    filename: str
    annotations: list[K]

    def __init__(self, filename: str, annotations: list[K]) -> None:
        self.filename = filename
        self.annotations = annotations

    def getFilename(self) -> str:
        return self.filename
    
    def getAnnotations(self) -> list[K]:
        return self.annotations
    
    def addAnnotation(self, annotation: K) -> None:
        self.annotations.append(annotation)


X = TypeVar("X", bound=FileFormat)

class DatasetFormat(ABC, Generic[X]):
    name: str
    files: list[X]

    def __init__(self, name: str, files: list[X]) -> None:
        self.name = name
        self.files = files

    def getName(self) -> str:
        return self.name

    def getFiles(self) -> list[X]:
        return self.files
    
    def addFile(self, file: X) -> None:
        self.files.append(file)