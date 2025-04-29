from formats.base import Annotation, DatasetFormat, FileFormat
from formats.pascal_voc import PascalVocBoundingBox


class NeutralAnnotation(Annotation[PascalVocBoundingBox]):
    
    class_name: str
    params: dict[str, list[str] | str]

    def __init__(self, bbox: PascalVocBoundingBox, class_name: str, params: dict[str, list[str] | str]) -> None:
        super().__init__(bbox)
        self.class_name = class_name
        self.params = params

    def getParams(self) -> dict[str, list[str] | str]:
        return self.params

    def addParam(self, key: str, value: list[str] | str) -> None:
        self.params[key] = value


class NeutralFile(FileFormat[NeutralAnnotation]):

    # image information
    image_path: str
    width: int
    height: int
    depth: int

    params: dict[str, list[str] | str]

    def __init__(self, filename: str, annotations: list[NeutralAnnotation], image_path: str, width: int, height: int, depth: int, params: dict[str, list[str] | str]) -> None:
        super().__init__(filename, annotations)
        self.image_path = image_path
        self.width = width
        self.height = height
        self.depth = depth
        self.params = params

    def getParams(self) -> dict[str, list[str] | str]:
        return self.params

    def addParam(self, key: str, value: list[str] | str) -> None:
        self.params[key] = value


class NeutralFormat(DatasetFormat[NeutralFile]):

    metadata: dict[str, list[str] | str]
    class_list: list[str]

    def __init__(self, name: str, files: list[NeutralFile], metadata: dict[str, list[str] | str], class_list: list[str]) -> None:
        super().__init__(name, files)
        self.metadata = metadata
        self.class_list = class_list

    def getMetadata(self) -> dict[str, list[str] | str]:
        return self.metadata

    def addMetadata(self, key: str, value: list[str] | str) -> None:
        self.metadata[key] = value