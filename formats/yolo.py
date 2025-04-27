from .base import BoundingBox

class Yolo_BoundingBox(BoundingBox):
    
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