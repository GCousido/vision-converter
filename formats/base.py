from abc import ABC, abstractmethod

class BoundingBox(ABC):

    @abstractmethod
    def getBoundingBox(self) -> list:
        pass