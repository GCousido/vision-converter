from vision_converter.formats.bounding_box import TopLeftAbsoluteBoundingBox, CenterAbsoluteBoundingBox, CornerAbsoluteBoundingBox, CenterNormalizedBoundingBox
from vision_converter.utils.bbox_utils import (
    CenterAbsolute_to_CornerAbsolute,
    CornerAbsolute_to_CenterAbsolute,
    CenterNormalized_to_CornerAbsolute,
    TopLeftAbsolute_to_CornerAbsolute,
    CornerAbsolute_to_CenterNormalized,
    CornerAbsolute_to_TopLeftAbsolute
)

def test_CenterNormalized_to_CornerAbsolute():
    yolo_bbox1 = CenterNormalizedBoundingBox(0.5, 0.5, 0.2, 0.2)
    yolo_bbox2 = CenterNormalizedBoundingBox(0.0, 0.0, 0.1, 0.1)
    yolo_bbox3 = CenterNormalizedBoundingBox(1.0, 1.0, 0.2, 0.2)
    yolo_bbox4 = CenterNormalizedBoundingBox(0.33, 0.66, 0.1, 0.1)

    expected_bbox1 = CornerAbsoluteBoundingBox(40, 40, 60, 60)
    expected_bbox2 = CornerAbsoluteBoundingBox(-5, -5, 5, 5)
    expected_bbox3 = CornerAbsoluteBoundingBox(90, 90, 110, 110)
    expected_bbox4 = CornerAbsoluteBoundingBox(28, 61, 38, 71)

    # Case 1
    new_bbox1 = CenterNormalized_to_CornerAbsolute(yolo_bbox1, 100, 100)
    assert new_bbox1 == expected_bbox1

    # Case 2
    new_bbox2 = CenterNormalized_to_CornerAbsolute(yolo_bbox2, 100, 100)
    assert new_bbox2 == expected_bbox2

    # Case 3
    new_bbox3 = CenterNormalized_to_CornerAbsolute(yolo_bbox3, 100, 100)
    assert new_bbox3 == expected_bbox3

    # Case 4
    new_bbox4 = CenterNormalized_to_CornerAbsolute(yolo_bbox4, 100, 100)
    assert new_bbox4 == expected_bbox4

def test_TopLeftAbsolute_to_CornerAbsolute():
    # Test cases
    coco_bbox1 = TopLeftAbsoluteBoundingBox(10.0, 20.0, 30.0, 40.0)
    coco_bbox2 = TopLeftAbsoluteBoundingBox(10.3, 20.7, 30.2, 40.4)
    coco_bbox3 = TopLeftAbsoluteBoundingBox(10.5, 20.5, 20.5, 30.5)
    coco_bbox4 = TopLeftAbsoluteBoundingBox(10.6, 20.4, 20.4, 30.6)

    expected_bbox1 = CornerAbsoluteBoundingBox(10, 20, 40, 60)
    expected_bbox2 = CornerAbsoluteBoundingBox(10, 21, 40, 61)
    expected_bbox3 = CornerAbsoluteBoundingBox(10, 20, 31, 51)
    expected_bbox4 = CornerAbsoluteBoundingBox(11, 20, 31, 51)

    # Case 1: Exact integer values
    result1 = TopLeftAbsolute_to_CornerAbsolute(coco_bbox1)
    assert result1 == expected_bbox1

    # Case 2: Standard decimal rounding
    result2 = TopLeftAbsolute_to_CornerAbsolute(coco_bbox2)
    assert result2 == expected_bbox2

    # Case 3: .5 values (bankers rounding)
    result3 = TopLeftAbsolute_to_CornerAbsolute(coco_bbox3)
    assert result3 == expected_bbox3

    # Case 4: Decimal accumulation
    result4 = TopLeftAbsolute_to_CornerAbsolute(coco_bbox4)
    assert result4 == expected_bbox4

def test_CornerAbsolute_to_CenterNormalized():
    bbox1 = CornerAbsoluteBoundingBox(40, 40, 60, 60)
    bbox2 = CornerAbsoluteBoundingBox(-5, -5, 5, 5)
    bbox3 = CornerAbsoluteBoundingBox(90, 90, 110, 110)
    bbox4 = CornerAbsoluteBoundingBox(28, 61, 38, 71)

    expected_bbox1 = CenterNormalizedBoundingBox(0.5, 0.5, 0.2, 0.2)
    expected_bbox2 = CenterNormalizedBoundingBox(0.0, 0.0, 0.1, 0.1)
    expected_bbox3 = CenterNormalizedBoundingBox(1.0, 1.0, 0.2, 0.2)
    expected_bbox4 = CenterNormalizedBoundingBox(0.33, 0.66, 0.1, 0.1)

    # Case 1
    new_bbox1 = CornerAbsolute_to_CenterNormalized(bbox1, 100, 100)
    assert new_bbox1 == expected_bbox1

    # Case 2
    new_bbox2 = CornerAbsolute_to_CenterNormalized(bbox2, 100, 100)
    assert new_bbox2 == expected_bbox2

    # Case 3
    new_bbox3 = CornerAbsolute_to_CenterNormalized(bbox3, 100, 100)
    assert new_bbox3 == expected_bbox3

    # Case 4
    new_bbox4 = CornerAbsolute_to_CenterNormalized(bbox4, 100, 100)
    assert new_bbox4 == expected_bbox4


def test_CornerAbsolute_to_TopLeftAbsolute():
    bbox1 = CornerAbsoluteBoundingBox(5, 10, 25, 30)       
    bbox2 = CornerAbsoluteBoundingBox(100, 150, 300, 400)  
    bbox3 = CornerAbsoluteBoundingBox(7, 21, 57, 99)       
    bbox4 = CornerAbsoluteBoundingBox(200, 50, 350, 120)  
    bbox5 = CornerAbsoluteBoundingBox(60, 80, 90, 200)     

    expected_bbox1 = TopLeftAbsoluteBoundingBox(5, 10, 20, 20)
    expected_bbox2 = TopLeftAbsoluteBoundingBox(100, 150, 200, 250)
    expected_bbox3 = TopLeftAbsoluteBoundingBox(7, 21, 50, 78)
    expected_bbox4 = TopLeftAbsoluteBoundingBox(200, 50, 150, 70)
    expected_bbox5 = TopLeftAbsoluteBoundingBox(60, 80, 30, 120)

    # Case 1: Small box
    new_bbox1 = CornerAbsolute_to_TopLeftAbsolute(bbox1)
    assert new_bbox1 == expected_bbox1

    # Case 2: Medium box
    new_bbox2 = CornerAbsolute_to_TopLeftAbsolute(bbox2)
    assert new_bbox2 == expected_bbox2

    # Case 3: Non-rounded values
    new_bbox3 = CornerAbsolute_to_TopLeftAbsolute(bbox3)
    assert new_bbox3 == expected_bbox3

    # Case 4: Large horizontal box
    new_bbox4 = CornerAbsolute_to_TopLeftAbsolute(bbox4)
    assert new_bbox4 == expected_bbox4

    # Case 5: Large vertical box
    new_bbox5 = CornerAbsolute_to_TopLeftAbsolute(bbox5)
    assert new_bbox5 == expected_bbox5


def test_CenterAbsolute_to_CornerAbsolute():
    """Test conversion from CreateML bounding box format to Pascal VOC format."""
    
    # Test cases with center coordinates + dimensions -> corner coordinates
    createml_bbox1 = CenterAbsoluteBoundingBox(50, 50, 20, 30)    # Center at (50,50), size 20x30
    createml_bbox2 = CenterAbsoluteBoundingBox(100, 150, 60, 80)  # Center at (100,150), size 60x80
    createml_bbox3 = CenterAbsoluteBoundingBox(25, 75, 40, 50)    # Center at (25,75), size 40x50
    createml_bbox4 = CenterAbsoluteBoundingBox(200, 300, 100, 200) # Center at (200,300), size 100x200
    createml_bbox5 = CenterAbsoluteBoundingBox(15, 25, 10, 20)    # Center at (15,25), size 10x20

    # Expected Pascal VOC format: (x_min, y_min, x_max, y_max)
    expected_bbox1 = CornerAbsoluteBoundingBox(40, 35, 60, 65)   # 50±10, 50±15
    expected_bbox2 = CornerAbsoluteBoundingBox(70, 110, 130, 190) # 100±30, 150±40
    expected_bbox3 = CornerAbsoluteBoundingBox(5, 50, 45, 100)   # 25±20, 75±25
    expected_bbox4 = CornerAbsoluteBoundingBox(150, 200, 250, 400) # 200±50, 300±100
    expected_bbox5 = CornerAbsoluteBoundingBox(10, 15, 20, 35)   # 15±5, 25±10

    # Case 1: Standard center conversion
    result1 = CenterAbsolute_to_CornerAbsolute(createml_bbox1)
    assert result1 == expected_bbox1

    # Case 2: Medium sized box
    result2 = CenterAbsolute_to_CornerAbsolute(createml_bbox2)
    assert result2 == expected_bbox2

    # Case 3: Box near image edge
    result3 = CenterAbsolute_to_CornerAbsolute(createml_bbox3)
    assert result3 == expected_bbox3

    # Case 4: Large box
    result4 = CenterAbsolute_to_CornerAbsolute(createml_bbox4)
    assert result4 == expected_bbox4

    # Case 5: Small box
    result5 = CenterAbsolute_to_CornerAbsolute(createml_bbox5)
    assert result5 == expected_bbox5


def test_CornerAbsolute_to_CenterAbsolute():
    """Test conversion from Pascal VOC bounding box format to CreateML format."""
    
    # Test cases with corner coordinates -> center coordinates + dimensions
    pascal_bbox1 = CornerAbsoluteBoundingBox(40, 35, 60, 65) 
    pascal_bbox2 = CornerAbsoluteBoundingBox(70, 110, 130, 190)  
    pascal_bbox3 = CornerAbsoluteBoundingBox(5, 50, 45, 100)     
    pascal_bbox4 = CornerAbsoluteBoundingBox(150, 200, 250, 400) 
    pascal_bbox5 = CornerAbsoluteBoundingBox(10, 15, 20, 35)     

    # Expected CreateML format: center coordinates + dimensions
    expected_bbox1 = CenterAbsoluteBoundingBox(50, 50, 20, 30)
    expected_bbox2 = CenterAbsoluteBoundingBox(100, 150, 60, 80)
    expected_bbox3 = CenterAbsoluteBoundingBox(25, 75, 40, 50)
    expected_bbox4 = CenterAbsoluteBoundingBox(200, 300, 100, 200)
    expected_bbox5 = CenterAbsoluteBoundingBox(15, 25, 10, 20)

    # Case 1: Standard corner to center conversion
    result1 = CornerAbsolute_to_CenterAbsolute(pascal_bbox1)
    assert result1 == expected_bbox1

    # Case 2: Medium sized box
    result2 = CornerAbsolute_to_CenterAbsolute(pascal_bbox2)
    assert result2 == expected_bbox2

    # Case 3: Box with edge coordinates
    result3 = CornerAbsolute_to_CenterAbsolute(pascal_bbox3)
    assert result3 == expected_bbox3

    # Case 4: Large box
    result4 = CornerAbsolute_to_CenterAbsolute(pascal_bbox4)
    assert result4 == expected_bbox4

    # Case 5: Small box
    result5 = CornerAbsolute_to_CenterAbsolute(pascal_bbox5)
    assert result5 == expected_bbox5

def test_representation_methods():
    # __str__ and __repr__ tests for all box classes
    boxes = [
        CenterNormalizedBoundingBox(0.5, 0.5, 0.2, 0.2),
        CenterAbsoluteBoundingBox(50, 50, 20, 30),
        CornerAbsoluteBoundingBox(40, 40, 60, 60),
        TopLeftAbsoluteBoundingBox(10, 20, 30, 40),
    ]

    expected_strs = [
        "[0.5, 0.5, 0.2, 0.2]",
        "[50, 50, 20, 30]",
        "[40, 40, 60, 60]",
        "[10, 20, 30, 40]",
    ]
    expected_reprs = [
        "CenterNormalizedBoundingBox(x_center=0.5, y_center=0.5, width=0.2, height=0.2)",
        "CenterAbsoluteBoundingBox(x_center=50, y_center=50, width=20, height=30)",
        "CornerAbsoluteBoundingBox(x_min=40, y_min=40, x_max=60, y_max=60)",
        "TopLeftAbsoluteBoundingBox(x_min=10, y_min=20, width=30, height=40)",
    ]

    for box, exp_str, exp_repr in zip(boxes, expected_strs, expected_reprs):
        assert str(box) == exp_str, f"__str__ failed for {type(box).__name__}"
        assert repr(box) == exp_repr, f"__repr__ failed for {type(box).__name__}"