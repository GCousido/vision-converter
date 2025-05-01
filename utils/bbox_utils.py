from formats.coco import CocoBoundingBox
from formats.pascal_voc import PascalVocBoundingBox
from formats.yolo import YoloBoundingBox



def YoloBBox_to_PascalVocBBox(bbox: YoloBoundingBox, image_width: int, image_height: int) -> PascalVocBoundingBox:
    x_center_abs = bbox.x_center * image_width
    y_center_abs = bbox.y_center * image_height
    width_abs = bbox.width * image_width
    height_abs = bbox.height * image_height

    x_min = round(x_center_abs - width_abs / 2)
    y_min = round(y_center_abs - height_abs / 2)
    x_max = round(x_center_abs + width_abs / 2)
    y_max = round(y_center_abs + height_abs / 2)

    return PascalVocBoundingBox(
        x_min=x_min,
        y_min=y_min,
        x_max=x_max,
        y_max=y_max
    )

def CocoBBox_to_PascalVocBBox(bbox: CocoBoundingBox) -> PascalVocBoundingBox:
    x_max_raw = bbox.x_min + bbox.width
    y_max_raw = bbox.y_min + bbox.height

    x_min = round(bbox.x_min)
    y_min = round(bbox.y_min)
    x_max = round(x_max_raw)
    y_max = round(y_max_raw)

    return PascalVocBoundingBox(
        x_min=x_min,
        y_min=y_min,
        x_max=x_max,
        y_max=y_max
    )



def PascalVocBBox_to_YoloBBox(bbox: PascalVocBoundingBox, image_width: int, image_height: int) -> YoloBoundingBox:
    x_center = ((bbox.x_min + bbox.x_max) / 2) / image_width
    y_center = ((bbox.y_min + bbox.y_max) / 2) / image_height
    width = (bbox.x_max - bbox.x_min) / image_width
    height = (bbox.y_max - bbox.y_min) / image_height

    return YoloBoundingBox(
        x_center=x_center,
        y_center=y_center,
        width=width,
        height=height
    )

def PascalVocBBox_to_CocoBBox(bbox: PascalVocBoundingBox) -> CocoBoundingBox:
    x_min = bbox.x_min
    y_min = bbox.y_min
    width = bbox.x_max - bbox.x_min
    height = bbox.y_max - bbox.y_min

    return CocoBoundingBox(
        x_min=x_min,
        y_min=y_min,
        width=width,
        height=height
    )