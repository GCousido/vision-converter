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