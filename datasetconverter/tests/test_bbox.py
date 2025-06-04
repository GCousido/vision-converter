from datasetconverter.formats.coco import CocoBoundingBox
from datasetconverter.formats.pascal_voc import PascalVocBoundingBox
from datasetconverter.formats.yolo import YoloBoundingBox
from datasetconverter.utils.bbox_utils import (
    YoloBBox_to_PascalVocBBox,
    CocoBBox_to_PascalVocBBox,
    PascalVocBBox_to_YoloBBox,
    PascalVocBBox_to_CocoBBox
)

def test_YoloBBox_to_PascalVocBBox():
    yolo_bbox1 = YoloBoundingBox(0.5, 0.5, 0.2, 0.2)
    yolo_bbox2 = YoloBoundingBox(0.0, 0.0, 0.1, 0.1)
    yolo_bbox3 = YoloBoundingBox(1.0, 1.0, 0.2, 0.2)
    yolo_bbox4 = YoloBoundingBox(0.33, 0.66, 0.1, 0.1)

    expected_bbox1 = PascalVocBoundingBox(40, 40, 60, 60)
    expected_bbox2 = PascalVocBoundingBox(-5, -5, 5, 5)
    expected_bbox3 = PascalVocBoundingBox(90, 90, 110, 110)
    expected_bbox4 = PascalVocBoundingBox(28, 61, 38, 71)

    # Case 1
    new_bbox1 = YoloBBox_to_PascalVocBBox(yolo_bbox1, 100, 100)
    assert new_bbox1.x_min == expected_bbox1.x_min
    assert new_bbox1.y_min == expected_bbox1.y_min
    assert new_bbox1.x_max == expected_bbox1.x_max
    assert new_bbox1.y_max == expected_bbox1.y_max

    # Case 2
    new_bbox2 = YoloBBox_to_PascalVocBBox(yolo_bbox2, 100, 100)
    assert new_bbox2.x_min == expected_bbox2.x_min
    assert new_bbox2.y_min == expected_bbox2.y_min
    assert new_bbox2.x_max == expected_bbox2.x_max
    assert new_bbox2.y_max == expected_bbox2.y_max

    # Case 3
    new_bbox3 = YoloBBox_to_PascalVocBBox(yolo_bbox3, 100, 100)
    assert new_bbox3.x_min == expected_bbox3.x_min
    assert new_bbox3.y_min == expected_bbox3.y_min
    assert new_bbox3.x_max == expected_bbox3.x_max
    assert new_bbox3.y_max == expected_bbox3.y_max

    # Case 4
    new_bbox4 = YoloBBox_to_PascalVocBBox(yolo_bbox4, 100, 100)
    assert new_bbox4.x_min == expected_bbox4.x_min
    assert new_bbox4.y_min == expected_bbox4.y_min
    assert new_bbox4.x_max == expected_bbox4.x_max
    assert new_bbox4.y_max == expected_bbox4.y_max

def test_CocoBBox_to_PascalVocBBox():
    # Test cases
    coco_bbox1 = CocoBoundingBox(10.0, 20.0, 30.0, 40.0)
    coco_bbox2 = CocoBoundingBox(10.3, 20.7, 30.2, 40.4)
    coco_bbox3 = CocoBoundingBox(10.5, 20.5, 20.5, 30.5)
    coco_bbox4 = CocoBoundingBox(10.6, 20.4, 20.4, 30.6)

    expected_bbox1 = PascalVocBoundingBox(10, 20, 40, 60)
    expected_bbox2 = PascalVocBoundingBox(10, 21, 40, 61)
    expected_bbox3 = PascalVocBoundingBox(10, 20, 31, 51)
    expected_bbox4 = PascalVocBoundingBox(11, 20, 31, 51)

    # Case 1: Exact integer values
    result1 = CocoBBox_to_PascalVocBBox(coco_bbox1)
    assert result1.x_min == expected_bbox1.x_min
    assert result1.y_min == expected_bbox1.y_min
    assert result1.x_max == expected_bbox1.x_max
    assert result1.y_max == expected_bbox1.y_max

    # Case 2: Standard decimal rounding
    result2 = CocoBBox_to_PascalVocBBox(coco_bbox2)
    assert result2.x_min == expected_bbox2.x_min
    assert result2.y_min == expected_bbox2.y_min
    assert result2.x_max == expected_bbox2.x_max
    assert result2.y_max == expected_bbox2.y_max

    # Case 3: .5 values (bankers rounding)
    result3 = CocoBBox_to_PascalVocBBox(coco_bbox3)
    assert result3.x_min == expected_bbox3.x_min
    assert result3.y_min == expected_bbox3.y_min
    assert result3.x_max == expected_bbox3.x_max
    assert result3.y_max == expected_bbox3.y_max

    # Case 4: Decimal accumulation
    result4 = CocoBBox_to_PascalVocBBox(coco_bbox4)
    assert result4.x_min == expected_bbox4.x_min
    assert result4.y_min == expected_bbox4.y_min
    assert result4.x_max == expected_bbox4.x_max
    assert result4.y_max == expected_bbox4.y_max

def test_PascalVocBBox_to_YoloBBox():
    bbox1 = PascalVocBoundingBox(40, 40, 60, 60)
    bbox2 = PascalVocBoundingBox(-5, -5, 5, 5)
    bbox3 = PascalVocBoundingBox(90, 90, 110, 110)
    bbox4 = PascalVocBoundingBox(28, 61, 38, 71)

    expected_bbox1 = YoloBoundingBox(0.5, 0.5, 0.2, 0.2)
    expected_bbox2 = YoloBoundingBox(0.0, 0.0, 0.1, 0.1)
    expected_bbox3 = YoloBoundingBox(1.0, 1.0, 0.2, 0.2)
    expected_bbox4 = YoloBoundingBox(0.33, 0.66, 0.1, 0.1)

    # Case 1
    new_bbox1 = PascalVocBBox_to_YoloBBox(bbox1, 100, 100)
    assert abs(new_bbox1.x_center - expected_bbox1.x_center) < 1e-2
    assert abs(new_bbox1.y_center - expected_bbox1.y_center) < 1e-2
    assert abs(new_bbox1.width - expected_bbox1.width) < 1e-2
    assert abs(new_bbox1.height - expected_bbox1.height) < 1e-2

    # Case 2
    new_bbox2 = PascalVocBBox_to_YoloBBox(bbox2, 100, 100)
    assert abs(new_bbox2.x_center - expected_bbox2.x_center) < 1e-2
    assert abs(new_bbox2.y_center - expected_bbox2.y_center) < 1e-2
    assert abs(new_bbox2.width - expected_bbox2.width) < 1e-2
    assert abs(new_bbox2.height - expected_bbox2.height) < 1e-2

    # Case 3
    new_bbox3 = PascalVocBBox_to_YoloBBox(bbox3, 100, 100)
    assert abs(new_bbox3.x_center - expected_bbox3.x_center) < 1e-2
    assert abs(new_bbox3.y_center - expected_bbox3.y_center) < 1e-2
    assert abs(new_bbox3.width - expected_bbox3.width) < 1e-2
    assert abs(new_bbox3.height - expected_bbox3.height) < 1e-2

    # Case 4
    new_bbox4 = PascalVocBBox_to_YoloBBox(bbox4, 100, 100)
    assert abs(new_bbox4.x_center - expected_bbox4.x_center) < 1e-2
    assert abs(new_bbox4.y_center - expected_bbox4.y_center) < 1e-2
    assert abs(new_bbox4.width - expected_bbox4.width) < 1e-2
    assert abs(new_bbox4.height - expected_bbox4.height) < 1e-2


def test_PascalVocBBox_to_CocoBBox():
    bbox1 = PascalVocBoundingBox(5, 10, 25, 30)       
    bbox2 = PascalVocBoundingBox(100, 150, 300, 400)  
    bbox3 = PascalVocBoundingBox(7, 21, 57, 99)       
    bbox4 = PascalVocBoundingBox(200, 50, 350, 120)  
    bbox5 = PascalVocBoundingBox(60, 80, 90, 200)     

    expected_bbox1 = CocoBoundingBox(5, 10, 20, 20)
    expected_bbox2 = CocoBoundingBox(100, 150, 200, 250)
    expected_bbox3 = CocoBoundingBox(7, 21, 50, 78)
    expected_bbox4 = CocoBoundingBox(200, 50, 150, 70)
    expected_bbox5 = CocoBoundingBox(60, 80, 30, 120)

    # Case 1: Small box
    new_bbox1 = PascalVocBBox_to_CocoBBox(bbox1)
    assert new_bbox1.x_min == expected_bbox1.x_min
    assert new_bbox1.y_min == expected_bbox1.y_min
    assert new_bbox1.width == expected_bbox1.width
    assert new_bbox1.height == expected_bbox1.height

    # Case 2: Medium box
    new_bbox2 = PascalVocBBox_to_CocoBBox(bbox2)
    assert new_bbox2.x_min == expected_bbox2.x_min
    assert new_bbox2.y_min == expected_bbox2.y_min
    assert new_bbox2.width == expected_bbox2.width
    assert new_bbox2.height == expected_bbox2.height

    # Case 3: Non-rounded values
    new_bbox3 = PascalVocBBox_to_CocoBBox(bbox3)
    assert new_bbox3.x_min == expected_bbox3.x_min
    assert new_bbox3.y_min == expected_bbox3.y_min
    assert new_bbox3.width == expected_bbox3.width
    assert new_bbox3.height == expected_bbox3.height

    # Case 4: Large horizontal box
    new_bbox4 = PascalVocBBox_to_CocoBBox(bbox4)
    assert new_bbox4.x_min == expected_bbox4.x_min
    assert new_bbox4.y_min == expected_bbox4.y_min
    assert new_bbox4.width == expected_bbox4.width
    assert new_bbox4.height == expected_bbox4.height

    # Case 5: Large vertical box
    new_bbox5 = PascalVocBBox_to_CocoBBox(bbox5)
    assert new_bbox5.x_min == expected_bbox5.x_min
    assert new_bbox5.y_min == expected_bbox5.y_min
    assert new_bbox5.width == expected_bbox5.width
    assert new_bbox5.height == expected_bbox5.height

