from dataclasses import dataclass
import math
from typing import Any, Optional
from .base import Shape, Annotation, DatasetFormat, FileFormat
from .bounding_box import CornerAbsoluteBoundingBox

# TODO: test and structure checks
class NeutralPolygon(Shape):
    """
    Polygon shape represented as a list of points with x,y float coordinates.
    """

    def __init__(self, points: list[list[float]]) -> None:
        super().__init__(type = "polygon")
        if not isinstance(points, list) or len(points) < 3:
            raise ValueError("A polygon must be a list with at least 3 points")
        for i, pt in enumerate(points):
            if not (isinstance(pt, list) and len(pt) == 2):
                raise ValueError(f"All points must have exactly 2 coordinates (x,y). Invalid at index {i}: {pt}")
            points[i] = [float(pt[0]), float(pt[1])]
        self.points: list[list[float]] = points

    def getCoordinates(self) -> list[list[float]]:
        return self.points

    def getBoundingBox(self) -> CornerAbsoluteBoundingBox:
        x_coords = [p[0] for p in self.points]
        y_coords = [p[1] for p in self.points]
        return CornerAbsoluteBoundingBox(
            int(min(x_coords)), int(min(y_coords)),
            int(max(x_coords)), int(max(y_coords))
        )


class NeutralPolyline(Shape):
    """
    Polyline (linestrip) shape represented as a list of points with x,y float coordinates.
    """
    def __init__(self, points: list[list[float]]) -> None:
        super().__init__(type = "polyline")
        if not isinstance(points, list) or len(points) < 2:
            raise ValueError("A polyline must be a list with at least 2 points")
        for i, pt in enumerate(points):
            if not (isinstance(pt, list) and len(pt) == 2):
                raise ValueError(f"All points must have exactly 2 coordinates. Invalid at index {i}: {pt}")
            points[i] = [float(pt[0]), float(pt[1])]
        self.points: list[list[float]] = points

    def getCoordinates(self) -> list[list[float]]:
        return self.points

    def getBoundingBox(self) -> CornerAbsoluteBoundingBox:
        x_coords = [p[0] for p in self.points]
        y_coords = [p[1] for p in self.points]
        return CornerAbsoluteBoundingBox(
            int(min(x_coords)), int(min(y_coords)),
            int(max(x_coords)), int(max(y_coords))
        )


class NeutralPoint(Shape):
    """
    Single point shape defined by x,y float coordinates.
    """

    def __init__(self, x: float, y: float) -> None:
        super().__init__(type = "point")
        self.x = float(x)
        self.y = float(y)

    def getCoordinates(self) -> list[list[float]]:
        return [[self.x, self.y]]

    def getBoundingBox(self) -> CornerAbsoluteBoundingBox:
        # FIXME: this is not a bounding box coordinates
        return CornerAbsoluteBoundingBox(int(self.x), int(self.y), int(self.x), int(self.y))

# TODO: check if this is necesary
class NeutralMultiPoint(Shape):
    """
    Group of points represented as a list of x,y float coordinates.
    """

    def __init__(self, points: list[list[float]]) -> None:
        super().__init__("multipoint")
        if not isinstance(points, list) or len(points) < 1:
            raise ValueError("MultiPoint must be a list with at least 1 point")
        for i, pt in enumerate(points):
            if not (isinstance(pt, list) and len(pt) == 2):
                raise ValueError(f"All points must have exactly 2 coordinates. Invalid at index {i}: {pt}")
            points[i] = [float(pt[0]), float(pt[1])]
        self.points: list[list[float]] = points

    def getCoordinates(self) -> list[list[float]]:
        return self.points

    def getBoundingBox(self) -> CornerAbsoluteBoundingBox:
        x_coords: list[float] = [p[0] for p in self.points]
        y_coords: list[float] = [p[1] for p in self.points]
        return CornerAbsoluteBoundingBox(
            int(min(x_coords)), int(min(y_coords)),
            int(max(x_coords)), int(max(y_coords))
        )


class NeutralLine(Shape):
    """
    Line segment shape defined by two points (x1, y1) and (x2, y2) as floats.
    """

    def __init__(self, x1: float, y1: float, x2: float, y2: float) -> None:
        super().__init__(type = "line")
        self.x1 = float(x1)
        self.y1 = float(y1)
        self.x2 = float(x2)
        self.y2 = float(y2)

    def getCoordinates(self) -> list[list[float]]:
        return [[self.x1, self.y1], [self.x2, self.y2]]

    def getBoundingBox(self) -> CornerAbsoluteBoundingBox:
        return CornerAbsoluteBoundingBox(
            int(min(self.x1, self.x2)), int(min(self.y1, self.y2)),
            int(max(self.x1, self.x2)), int(max(self.y1, self.y2))
        )


class NeutralCircle(Shape):
    """
    Circle shape with center coordinates and positive radius, all as floats.
    """

    def __init__(self, center_x: float, center_y: float, radius: float) -> None:
        super().__init__(type = "circle")
        if float(radius) <= 0:
            raise ValueError("Radius must be positive")
        self.center_x = float(center_x)
        self.center_y = float(center_y)
        self.radius = float(radius)

    def getCoordinates(self) -> list[list[float]]:
        return [[self.center_x, self.center_y], [self.radius]]

    def getBoundingBox(self) -> CornerAbsoluteBoundingBox:
        return CornerAbsoluteBoundingBox(
            int(self.center_x - self.radius), int(self.center_y - self.radius),
            int(self.center_x + self.radius), int(self.center_y + self.radius)
        )


class NeutralEllipse(Shape):
    """
    Ellipse shape defined by center coordinates, horizontal/vertical radii, and rotation angle in radians. All are floats.
    """

    def __init__(self, center_x: float, center_y: float, radius_x: float, radius_y: float, theta: float = 0.0) -> None:
        super().__init__(type = "ellipse")
        self.center_x = float(center_x)
        self.center_y = float(center_y)
        self.radius_x = float(radius_x)
        self.radius_y = float(radius_y)
        self.theta = float(theta)

    def getCoordinates(self) -> list[float]:
        return [self.center_x, self.center_y, self.radius_x, self.radius_y, self.theta]

    def getBoundingBox(self) -> CornerAbsoluteBoundingBox:
        cos_theta = math.cos(self.theta)
        sin_theta = math.sin(self.theta)
        extent_x = abs(self.radius_x * cos_theta) + abs(self.radius_y * sin_theta)
        extent_y = abs(self.radius_x * sin_theta) + abs(self.radius_y * cos_theta)
        return CornerAbsoluteBoundingBox(
            int(self.center_x - extent_x), int(self.center_y - extent_y),
            int(self.center_x + extent_x), int(self.center_y + extent_y)
        )

# TODO: checks and what is a real mask
class NeutralMask(Shape):
    """
    Mask shape represented with bounding box coordinates [x1, y1, x2, y2] as floats and optional mask data.
    """

    def __init__(self, bbox: list[float], mask_data: Optional[bytes] = None, type: str = "mask") -> None:
        super().__init__(type)
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError("bbox must be a list of four [x1, y1, x2, y2]")
        self.x1, self.y1, self.x2, self.y2 = [float(coord) for coord in bbox]
        self.mask_data = mask_data

    def getCoordinates(self) -> list[list[float]]:
        return [[self.x1, self.y1], [self.x2, self.y2]]

    def getBoundingBox(self) -> CornerAbsoluteBoundingBox:
        return CornerAbsoluteBoundingBox(
            int(min(self.x1, self.x2)), int(min(self.y1, self.y2)),
            int(max(self.x1, self.x2)), int(max(self.y1, self.y2))
        )


class NeutralAnnotation(Annotation[CornerAbsoluteBoundingBox]):
    """Generic annotation for objects in images with extended metadata.

    Inherits from:
        Annotation[CornerAbsoluteBoundingBox]: Base annotation class with Pascal VOC bounding box

    Attributes:
        class_name (str): Name of the object class (e.g., 'person', 'car').
        attributes (dict[str, Any]): Additional object metadata key-value pairs.
        bbox (CornerAbsoluteBoundingBox): Inherited attribute - bounding box coordinates
            in Pascal VOC format (xmin, ymin, xmax, ymax).
    """

    class_name: str
    attributes: dict[str, Any]

    def __init__(self, bbox: CornerAbsoluteBoundingBox, class_name: str, attributes: Optional[dict[str, Any]] = None) -> None:
        super().__init__(bbox)
        self.class_name = class_name
        self.attributes = attributes if attributes is not None else {}


@dataclass
class ImageOrigin:
    """Metadata container describing the origin and provenance of an image.

    Attributes:
        extension (str): File extension with leading dot (e.g., ".jpg", ".png")
        
        source_type (Optional[list[str]]): Types of original sources, must be aligned with
            source_id and image_url lists. Typical values: ["flickr", "synthetic", "web"]
        source_id (Optional[list[str]]): Unique identifiers from original sources
        image_url (Optional[list[str]]): Original URLs where the image was obtained
        
        image_provider (Optional[str]): Current provider/service hosting the image
            (e.g., "flickr", "user_upload", "stock")
        source_dataset (Optional[str]): Original dataset identifier
            (e.g., "VOC2007", "COCO2017")
        
        date_captured (Optional[str]): Capture date in YYYY/MM/DD format
        image_license (Optional[str]): License type (e.g., "CC BY 4.0", "proprietary")
        license_url (Optional[str]): URL to full license text

    Note:
        The lists source_type, source_id, and image_url must be index-aligned
    """

    extension: str                              # ".jpg", ".png"

    # This lists have to be aligned
    source_type: Optional[list[str]] = None     # "flickr", "synthetic", "web"
    source_id: Optional[list[str]] = None       # flickrid, etc
    image_url: Optional[list[str]] = None

    image_provider: Optional[str] = None        # "flickr", "user_upload", "stock"
    source_dataset: Optional[str] = None        # "VOC2007", "COCO2017"

    date_captured: Optional[str] = None         # 2025/05/02

    image_license: Optional[str] = None         # "CC BY 4.0", "proprietary"
    license_url: Optional[str] = None           # URL



class NeutralFile(FileFormat[NeutralAnnotation]):
    """Container for image file data and annotations in neutral format.

    Inherits from:
        FileFormat[NeutralAnnotation]: Base file format class with a list neutral annotations

    Attributes:
        width (int): Image width in pixels
        height (int): Image height in pixels
        depth (int): Color depth (typically 3 for RGB)
        image_origin (ImageOrigin): Metadata about image provenance
        params (dict[str, Any]): Additional processing parameters
        filename (str): Inherited attribute - name of the image file
        annotations (list[NeutralAnnotation]): Inherited attribute - list of annotations
    """

    # image information
    width: int
    height: int
    depth: int

    # image metadata
    image_origin: ImageOrigin

    params: dict[str, Any]

    def __init__(self, filename: str, annotations: list[NeutralAnnotation], width: int, height: int, depth: int, image_origin: ImageOrigin, params: Optional[dict[str, Any]] = None) -> None:
        super().__init__(filename, annotations)
        self.width = width
        self.height = height
        self.depth = depth
        self.image_origin = image_origin
        self.params = params if params is not None else {}


class NeutralFormat(DatasetFormat[NeutralFile]):
    """Container for a dataset in neutral format. This is designed to be a unified 

    Inherits from:
        DatasetFormat[NeutralFile]: Base dataset format with neutral file type

    Attributes:
        metadata (dict[str, Any]): Global dataset metadata (e.g., version, creator)
        class_map (dict[int, str]): Mapping of numeric IDs to class names
            (e.g., {0: 'person', 1: 'car'})
        original_format (str): Original dataset format name (e.g., "PascalVOC")
        name (str): Inherited attribute - dataset name
        files (list[NeutralFile]): Inherited attribute - list of image files of NeutralFile class
        images_path_list (Optional[list[str]]): Inherited attribute - List of images paths
    """
    metadata: dict[str, Any]
    class_map: dict[int, str]
    original_format: str

    def __init__(self, name: str, files: list[NeutralFile], original_format: str,
                metadata: Optional[dict[str, Any]] = None, 
                class_map: Optional[dict[int, str]] = None, 
                images_path_list: Optional[list[str]] = None) -> None:
        super().__init__(name, files)
        self.metadata = metadata if metadata is not None else {}
        self.class_map = class_map if class_map is not None else {}
        self.original_format = original_format
        self.images_path_list = images_path_list