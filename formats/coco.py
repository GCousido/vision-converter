from .base import BoundingBox

class Coco_BoundingBox(BoundingBox):
    
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