from unicodedata import category
from .base import Annotation, BoundingBox

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