
from abc import ABC, abstractmethod

class BoundingBox(ABC):
    """Abstract base class representing a bounding box structure.
    
    Subclasses must implement the getBoundingBox method to provide
    coordinate values in a standardized format.
    """

    @abstractmethod
    def getBoundingBox(self) -> list:
        pass


class YoloBoundingBox(BoundingBox):
    """YOLO format bounding box implementation using normalized coordinates.
    
    Attributes:
        x_center (float): Normalized x-coordinate of center (0.0-1.0)
        y_center (float): Normalized y-coordinate of center (0.0-1.0)
        width (float): Normalized width (0.0-1.0)
        height (float): Normalized height (0.0-1.0)
    """
    x_center: float
    y_center: float
    width: float
    height: float

    def __init__(self, x_center: float, y_center: float, width: float, height: float) -> None:
        self.x_center = x_center
        self.y_center = y_center
        self.width = width
        self.height = height

    def getBoundingBox(self):
        """Returns YOLO format coordinates as [x_center, y_center, width, height]."""
        return [self.x_center, self.y_center, self.width, self.height]


class PascalVocBoundingBox(BoundingBox):
    """Bounding box in Pascal VOC format (absolute pixel coordinates).

    Attributes:
        x_min (int): Minimum x (left).
        y_min (int): Minimum y (top).
        x_max (int): Maximum x (right).
        y_max (int): Maximum y (bottom).
    """
    x_min: int
    y_min: int
    x_max: int
    y_max: int

    def __init__(self, x_min: int,  y_min: int, x_max: int, y_max: int) -> None:
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max

    def __eq__(self, other):
        """Compare two PascalVocBoundingBox objects for equality."""
        if not isinstance(other, PascalVocBoundingBox):
            return NotImplemented
        return (self.x_min == other.x_min and 
                self.y_min == other.y_min and 
                self.x_max == other.x_max and 
                self.y_max == other.y_max)

    def getBoundingBox(self):
        """Returns Pascal Voc coordinates as [x_min, y_min, x_max, y_max]."""
        return [self.x_min, self.y_min, self.x_max,  self.y_max]


class CocoBoundingBox(BoundingBox):
    """COCO format bounding box using absolute pixel coordinates.

    Attributes:
        x_min (float): Minimum x (left).
        y_min (float): Minimum y (top).
        width (float): Box width in pixels.
        height (float): Box height in pixels.
    """
    x_min: float
    y_min: float
    width: float
    height: float

    def __init__(self, x_min: float, y_min: float, width: float, height: float) -> None:
        self.x_min = x_min
        self.y_min = y_min
        self.width = width
        self.height = height

    def getBoundingBox(self):
        """Returns COCO format coordinates as [x_min, y_min, width, height]."""
        return [self.x_min, self.y_min, self.width, self.height]


class CreateMLBoundingBox(BoundingBox):
    """CreateML format bounding box implementation using absolute coordinates.
    
    Attributes:
        x_center (float): absolute x-coordinate of center
        y_center (float): absolute y-coordinate of center
        width (float): absolute width
        height (float): absolute height
    """
    x_center: float
    y_center: float
    width: float
    height: float

    def __init__(self, x_center: float, y_center: float, width: float, height: float) -> None:
        self.x_center = x_center
        self.y_center = y_center
        self.width = width
        self.height = height

    def getBoundingBox(self):
        """Returns CreateML format coordinates as [x_center, y_center, width, height]."""
        return [self.x_center, self.y_center, self.width, self.height]