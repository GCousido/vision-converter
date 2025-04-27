from dataclasses import dataclass
from .base import Annotation, BoundingBox, FileFormat

class CocoBoundingBox(BoundingBox):
    x_min: int
    y_min: int
    width: int
    height: int

    def __init__(self, x_min: int, y_min: int, width: int, height: int) -> None:
        self.x_min = x_min
        self.y_min = y_min
        self.width = width
        self.height = height

    def getBoundingBox(self):
        return [self.x_min, self.y_min, self.width, self.height]

@dataclass
class RLESegmentation:
    alto: int
    ancho: int
    counts: str

class CocoLabel(Annotation[CocoBoundingBox]):
    id: int
    image_id: int
    category_id: int
    segmentation: list[list[float] | RLESegmentation]
    area: float
    iscrowd: bool

    def __init__(self, bbox: CocoBoundingBox, id: int, image_id: int, category_id: int, segmentation: list[list[float] | RLESegmentation], area: float, iscrowd: bool) -> None:
        super().__init__(bbox)
        self.id = id
        self.image_id = image_id
        self.category_id = category_id
        self.segmentation = segmentation
        self.area = area
        self.iscrowd = iscrowd


@dataclass
class Info:
    description: str
    url: str
    version: str
    year: int
    contributor: str
    date_created: str

@dataclass
class License:
    id: int
    name: str
    url: str

@dataclass
class CocoImages:
    id: int
    width: int
    height: int
    file_name: str
    license: int
    flicker_url: str
    coco_url: str
    date_captured: str

@dataclass
class Category:
    id: int
    name: str
    supercategory: str


class CocoFile(FileFormat[CocoLabel]):
    info: Info
    licenses: list[License]
    images: list[CocoImages]
    categories: list[Category]

    def __init__(self, filename: str, annotations: list[CocoLabel], info: Info, licenses: list[License], images: list[CocoImages], categories: list[Category]) -> None:
        super().__init__(filename, annotations)
        self.info = info
        self.licenses = licenses
        self.images = images
        self.categories = categories