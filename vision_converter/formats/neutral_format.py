from dataclasses import dataclass
from typing import Any, Optional
from .base import Shape, Annotation, DatasetFormat, FileFormat
from .bounding_box import PascalVocBoundingBox

# TODO: implementar todo esto
class NeutralPolygon(Shape):
    def __init__(self, type: str) -> None:
        raise NotImplementedError("Implementation in Process")

class NeutralCircle(Shape):
    def __init__(self, type: str) -> None:
        raise NotImplementedError("Implementation in Process")

class NeutralEllipse(Shape):
    def __init__(self, type: str) -> None:
        raise NotImplementedError("Implementation in Process")

class NeutralPointOrPoints(Shape):
    def __init__(self, type: str) -> None:
        raise NotImplementedError("Implementation in Process")

class NeutralLineOrPolyline(Shape):
    def __init__(self, type: str) -> None:
        raise NotImplementedError("Implementation in Process")

class NeutralAnnotation(Annotation[PascalVocBoundingBox]):
    """Generic annotation for objects in images with extended metadata.

    Inherits from:
        Annotation[PascalVocBoundingBox]: Base annotation class with Pascal VOC bounding box

    Attributes:
        class_name (str): Name of the object class (e.g., 'person', 'car').
        attributes (dict[str, Any]): Additional object metadata key-value pairs.
        bbox (PascalVocBoundingBox): Inherited attribute - bounding box coordinates
            in Pascal VOC format (xmin, ymin, xmax, ymax).
    """

    class_name: str
    attributes: dict[str, Any]

    def __init__(self, bbox: PascalVocBoundingBox, class_name: str, attributes: Optional[dict[str, Any]] = None) -> None:
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