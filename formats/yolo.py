from .base import Annotation, BoundingBox, FileFormat

class YoloBoundingBox(BoundingBox):
    x_center: int
    y_center: int
    width: int
    height: int

    def __init__(self, x_center: int, y_center: int, width: int, height: int) -> None:
        self.x_center = x_center
        self.y_center = y_center
        self.width = width
        self.height = height

    def getBoundingBox(self):
        return [self.x_center, self.y_center, self.width, self.height]


class YoloAnnotation(Annotation[YoloBoundingBox]):
    id_class: int

    def __init__(self, bbox: YoloBoundingBox, id_class: int) -> None:
        super().__init__(bbox)
        self.id_class = id_class

class YoloFile(FileFormat[YoloAnnotation]):

    def __init__(self, filename: str, annotations: list[YoloAnnotation]) -> None:
        super().__init__(filename, annotations)
