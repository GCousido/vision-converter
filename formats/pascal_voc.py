from .base import Annotation, BoundingBox, DatasetFormat, FileFormat

class PascalVocBoundingBox(BoundingBox):
    x_min: int
    y_min: int
    x_max: int
    y_max: int

    def __init__(self, x_min: int,  y_min: int, x_max: int, y_max: int) -> None:
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max

    def getBoundingBox(self):
        return [self.x_min, self.y_min, self.x_max,  self.y_max]
    

class PascalVocObject(Annotation[PascalVocBoundingBox]):
    name: str
    pose: str
    truncated: bool
    difficult: bool

    def __init__(self, bbox: PascalVocBoundingBox, name: str, pose: str, truncated: bool, difficult: bool) -> None:
        super().__init__(bbox)
        self.name = name
        self.pose = pose
        self.truncated = truncated
        self.difficult = difficult


class PascalVocFile(FileFormat[PascalVocObject]):
    folder: str
    # filename: str - en principio está en el base
    path: str

    # forman la etiqueta de size
    width: int
    height: int
    depth: int

    segmented: int

    def __init__(self, filename: str, annotations: list[PascalVocObject], folder: str, path: str, width: int, height: int, depth: int, segmented: int) -> None:
        super().__init__(filename, annotations)
        self.folder = folder
        self.path = path
        self.width = width
        self.height = height
        self.depth = depth
        self.segmented = segmented


class PascalVocFormat(DatasetFormat[PascalVocFile]):

    def __init__(self, name: str, files: list[PascalVocFile]) -> None:
        super().__init__(name, files)