from dataclasses import dataclass
from typing import Any, Optional
from formats.base import Annotation, DatasetFormat, FileFormat
from formats.pascal_voc import PascalVocBoundingBox


class NeutralAnnotation(Annotation[PascalVocBoundingBox]):
    class_name: str
    attributes: dict[str, Any]

    def __init__(self, bbox: PascalVocBoundingBox, class_name: str, attributes: Optional[dict[str, Any]] = None) -> None:
        super().__init__(bbox)
        self.class_name = class_name
        self.attributes = attributes if attributes is not None else {}

    def addAttribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value


@dataclass
class ImageOrigin:

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

    def addParam(self, key: str, value: Any) -> None:
        self.params[key] = value


class NeutralFormat(DatasetFormat[NeutralFile]):
    metadata: dict[str, Any]
    class_map: dict[int, str]
    original_format: str

    def __init__(self, name: str, files: list[NeutralFile], original_format: str,
                 metadata: Optional[dict[str, Any]] = None, 
                 class_map: Optional[dict[int, str]] = None) -> None:
        super().__init__(name, files)
        self.metadata = metadata if metadata is not None else {}
        self.class_map = class_map if class_map is not None else {}
        self.original_format = original_format

    def addMetadata(self, key: str, value: Any) -> None:
        self.metadata[key] = value

    def addClass(self, class_id: int, class_label: str) -> None:
        self.class_map[class_id] = class_label