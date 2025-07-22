from abc import ABC, abstractmethod

# TODO: maybe add checks -> bounding box with allowed coordinates, (negative, out of image, points that forms a boundingbox)
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
        # Type checks
        for name, val in zip(["x_center", "y_center", "width", "height"], [x_center, y_center, width, height]):
            if not isinstance(val, float):
                raise TypeError(f"{name} must be a float. Got {val} ({type(val)})")
        # Range checks
        if not (0.0 <= x_center <= 1.0):
            raise ValueError(f"x_center must be in [0.0, 1.0]. Got {x_center}")
        if not (0.0 <= y_center <= 1.0):
            raise ValueError(f"y_center must be in [0.0, 1.0]. Got {y_center}")
        if not (0.0 < width <= 1.0):
            raise ValueError(f"width must be > 0 and <= 1. Got {width}")
        if not (0.0 < height <= 1.0):
            raise ValueError(f"height must be > 0 and <= 1. Got {height}")
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
        x_center (float | int): Absolute x-coordinate of the bounding box center (in pixels).
        y_center (float | int): Absolute y-coordinate of the bounding box center (in pixels).
        width (float | int): Absolute width of the bounding box (in pixels).
        height (float | int): Absolute height of the bounding box (in pixels).
    """
    x_center: float | int
    y_center: float | int
    width: float | int
    height: float | int

    def __init__(self, x_center: float | int, y_center: float | int, width: float | int, height: float | int) -> None:
        for name, val in zip(["x_center", "y_center", "width", "height"], [x_center, y_center, width, height]):
            if not isinstance(val, (float, int)):
                raise TypeError(f"{name} must be a float or int. Got {val} ({type(val)})")
        if width <= 0:
            raise ValueError(f"width must be > 0. Got {width}")
        if height <= 0:
            raise ValueError(f"height must be > 0. Got {height}")
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
        for name, val in zip(["x_min", "y_min", "x_max", "y_max"], [x_min, y_min, x_max, y_max]):
            if not isinstance(val, int):
                raise TypeError(f"{name} must be an int. Got {val} ({type(val)})")
        if x_min > x_max:
            raise ValueError(f"x_min ({x_min}) cannot be greater than x_max ({x_max})")
        if y_min > y_max:
            raise ValueError(f"y_min ({y_min}) cannot be greater than y_max ({y_max})")
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
        x_min (float | int): Absolute x-coordinate of the top-left corner (in pixels).
        y_min (float | int): Absolute y-coordinate of the top-left corner (in pixels).
        width (float | int): Absolute width of the bounding box (in pixels).
        height (float | int): Absolute height of the bounding box (in pixels).
    """
    x_min: float | int
    y_min: float | int
    width: float | int
    height: float | int

    def __init__(self, x_min: float | int, y_min: float | int, width: float | int, height: float | int) -> None:
        for name, val in zip(["x_min", "y_min", "width", "height"], [x_min, y_min, width, height]):
            if not isinstance(val, (float,int)):
                raise TypeError(f"{name} must be a float or int. Got {val} ({type(val)})")
        if width <= 0:
            raise ValueError(f"width must be > 0. Got {width}")
        if height <= 0:
            raise ValueError(f"height must be > 0. Got {height}")
        self.x_min = x_min
        self.y_min = y_min
        self.width = width
        self.height = height

    def getBoundingBox(self) -> list[float]:
        """Returns absolute bounding box coordinates as [x_min, y_min, width, height]."""
        return [self.x_min, self.y_min, self.width, self.height]