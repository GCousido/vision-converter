from typing import Any, Optional
from formats.base import Annotation, DatasetFormat, FileFormat
from formats.pascal_voc import PascalVocBoundingBox


class NeutralAnnotation(Annotation[PascalVocBoundingBox]):
    class_name: str
    params: dict[str, Any]

    def __init__(self, bbox: PascalVocBoundingBox, class_name: str, params: Optional[dict[str, Any]] = None) -> None:
        super().__init__(bbox)
        self.class_name = class_name
        self.params = params if params is not None else {}

    def addParam(self, key: str, value: Any) -> None:
        self.params[key] = value


class NeutralFile(FileFormat[NeutralAnnotation]):

    # image information
    width: int
    height: int
    depth: int

    params: dict[str, Any]

    def __init__(self, filename: str, annotations: list[NeutralAnnotation], width: int, height: int, depth: int, params: Optional[dict[str, Any]] = None) -> None:
        super().__init__(filename, annotations)
        self.width = width
        self.height = height
        self.depth = depth
        self.params = params if params is not None else {}

    def addParam(self, key: str, value: Any) -> None:
        self.params[key] = value


class NeutralFormat(DatasetFormat[NeutralFile]):
    metadata: dict[str, Any]
    class_map: dict[int, str]

    def __init__(self, name: str, files: list[NeutralFile], 
                 metadata: Optional[dict[str, Any]] = None, 
                 class_map: Optional[dict[int, str]] = None) -> None:
        super().__init__(name, files)
        self.metadata = metadata if metadata is not None else {}
        self.class_map = class_map if class_map is not None else {}

    def addMetadata(self, key: str, value: Any) -> None:
        self.metadata[key] = value

    def addClass(self, class_id: int, class_label: str) -> None:
        self.class_map[class_id] = class_label