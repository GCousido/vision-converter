from typing import Optional
from formats.base import Annotation, DatasetFormat, FileFormat
from formats.pascal_voc import PascalVocBoundingBox


class NeutralAnnotation(Annotation[PascalVocBoundingBox]):
    class_name: str
    params: dict[str, list[str] | str]

    def __init__(self, bbox: PascalVocBoundingBox, class_name: str, params: Optional[dict[str, list[str] | str]] = None) -> None:
        super().__init__(bbox)
        self.class_name = class_name
        self.params = params if params is not None else {}

    def getParams(self) -> dict[str, list[str] | str]:
        return self.params

    def addParam(self, key: str, value: list[str] | str) -> None:
        self.params[key] = value

    def setParams(self, params: dict[str, list[str] | str]) -> None:
        self.params = params


class NeutralFile(FileFormat[NeutralAnnotation]):

    # image information
    image_path: str
    width: int
    height: int
    depth: int

    params: dict[str, list[str] | str]

    def __init__(self, filename: str, annotations: list[NeutralAnnotation], image_path: str, width: int, height: int, depth: int, params: Optional[dict[str, list[str] | str]] = None) -> None:
        super().__init__(filename, annotations)
        self.image_path = image_path
        self.width = width
        self.height = height
        self.depth = depth
        self.params = params if params is not None else {}

    def getParams(self) -> dict[str, list[str] | str]:
        return self.params

    def addParam(self, key: str, value: list[str] | str) -> None:
        self.params[key] = value

    def setParams(self, params: dict[str, list[str] | str]) -> None:
        self.params = params


class NeutralFormat(DatasetFormat[NeutralFile]):

    metadata: dict[str, list[str] | str]
    class_list: list[str]

    def __init__(self, name: str, files: list[NeutralFile], metadata: Optional[dict[str, list[str] | str]] = None, class_list: Optional[list[str]] = None) -> None:
        super().__init__(name, files)
        self.metadata = metadata if metadata is not None else {}
        self.class_list = class_list if class_list is not None else []

    def getMetadata(self) -> dict[str, list[str] | str]:
        return self.metadata

    def addMetadata(self, key: str, value: list[str] | str) -> None:
        self.metadata[key] = value

    def setMetadata(self, metadata: dict[str, list[str] | str]) -> None:
        self.metadata = metadata

    def getClassList(self) -> list[str]:
        return self.class_list

    def addClassList(self, class_label: str) -> None:
        self.class_list.append(class_label)

    def setClassList(self, class_list: list[str]) -> None:
        self.class_list = class_list