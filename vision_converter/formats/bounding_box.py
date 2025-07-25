
from abc import ABC, abstractmethod

class BoundingBox(ABC):
    """Abstract base class representing a bounding box structure.
    
    Subclasses must implement the getBoundingBox method to provide
    coordinate values in a standardized format.
    """

    @abstractmethod
    def getBoundingBox(self) -> list:
        pass

    def __eq__(self, other):
        """Check equality based on bounding box representation and type."""
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.getBoundingBox() == other.getBoundingBox()

    def __repr__(self):
        cls_name = type(self).__name__
        attrs = ", ".join(f"{key}={value!r}" for key, value in self.__dict__.items())
        return f"{cls_name}({attrs})"

    def __str__(self):
        return f"{self.getBoundingBox()}"


class CenterNormalizedBoundingBox(BoundingBox):
    """Bounding box with normalized center coordinates and dimensions.
    
    This format represents the bounding box using values normalized
    between 0.0 and 1.0 for both the center position and size relative
    to the image width and height. Commonly used in formats like YOLO.

    Attributes:
        x_center (float): Normalized x-coordinate of the bounding box center (range: 0.0 to 1.0).
        y_center (float): Normalized y-coordinate of the bounding box center (range: 0.0 to 1.0).
        width (float): Normalized width of the bounding box (range: 0.0 to 1.0).
        height (float): Normalized height of the bounding box (range: 0.0 to 1.0).
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

    def getBoundingBox(self) -> list[float]:
        """Returns normalized center-format bounding box as [x_center, y_center, width, height]."""
        return [self.x_center, self.y_center, self.width, self.height]


class CenterAbsoluteBoundingBox(BoundingBox):
    """Bounding box with absolute center coordinates and dimensions.

    This format represents the bounding box using the absolute pixel coordinates
    of the center and the absolute width and height in pixels. Commonly used in
    formats such as Apple CreateML.

    Attributes:
        x_center (float): Absolute x-coordinate of the bounding box center (in pixels).
        y_center (float): Absolute y-coordinate of the bounding box center (in pixels).
        width (float): Absolute width of the bounding box (in pixels).
        height (float): Absolute height of the bounding box (in pixels).
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

    def getBoundingBox(self) -> list[float]:
        """Returns absolute center-format bounding box as [x_center, y_center, width, height]."""
        return [self.x_center, self.y_center, self.width, self.height]


class CornerAbsoluteBoundingBox(BoundingBox):
    """Bounding box with absolute top-left and bottom-right corners.

    This format specifies the absolute coordinates of the top-left (minimum x and y)
    and bottom-right (maximum x and y) corners in pixels. Commonly used in formats such
    as Pascal VOC.

    Attributes:
        x_min (int): Minimum x (left edge) coordinate of the bounding box (in pixels).
        y_min (int): Minimum y (top edge) coordinate of the bounding box (in pixels).
        x_max (int): Maximum x (right edge) coordinate of the bounding box (in pixels).
        y_max (int): Maximum y (bottom edge) coordinate of the bounding box (in pixels).
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

    def getBoundingBox(self) -> list[int]:
        """Returns absolute corner-format bounding box as [x_min, y_min, x_max, y_max]."""
        return [self.x_min, self.y_min, self.x_max,  self.y_max]


class TopLeftAbsoluteBoundingBox(BoundingBox):
    """Bounding box with absolute top-left corner and size.

    This format specifies the absolute coordinates of the top-left corner (x_min, y_min)
    along with the absolute width and height in pixels. Commonly used in formats such as COCO.

    Attributes:
        x_min (float): Absolute x-coordinate of the top-left corner (in pixels).
        y_min (float): Absolute y-coordinate of the top-left corner (in pixels).
        width (float): Absolute width of the bounding box (in pixels).
        height (float): Absolute height of the bounding box (in pixels).
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

    def getBoundingBox(self) -> list[float]:
        """Returns absolute bounding box coordinates as [x_min, y_min, width, height]."""
        return [self.x_min, self.y_min, self.width, self.height]