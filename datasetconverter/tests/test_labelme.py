from pathlib import Path
from PIL import Image
import pytest
import json
import base64

from datasetconverter.formats.labelme import (
    LabelMePolygon, LabelMeRectangle, LabelMeCircle, LabelMePoint, 
    LabelMeLine, LabelMeLinestrip, LabelMePoints, LabelMeMask,
    LabelMeAnnotation, LabelMeFile, LabelMeFormat
)

# Fixture for LabelMe dataset
@pytest.fixture
def sample_labelme_dataset(tmp_path):
    """Create a sample LabelMe dataset with various shape types"""
    
    # Create images directory
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    
    # Create sample images
    image1 = Image.new('RGB', (640, 480), color='red')
    image2 = Image.new('RGB', (800, 600), color='blue')
    
    image1.save(images_dir / 'image1.jpg')
    image2.save(images_dir / 'image2.jpg')
    
    # Create first JSON annotation file
    json_data1 = {
        "version": "5.0.1",
        "flags": {"reviewed": True},
        "shapes": [
            {
                "label": "person",
                "points": [[100, 100], [200, 100], [200, 200], [100, 200]],
                "group_id": 1,
                "shape_type": "polygon",
                "flags": {"difficult": False}
            },
            {
                "label": "car",
                "points": [[300, 300], [400, 400]],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {}
            },
            {
                "label": "wheel",
                "points": [[350, 350], [370, 370]],
                "group_id": None,
                "shape_type": "circle",
                "flags": {}
            }
        ],
        "imagePath": "images/image1.jpg",
        "imageData": None,
        "imageHeight": 480,
        "imageWidth": 640
    }
    
    # Create second JSON annotation file
    json_data2 = {
        "version": "5.0.1",
        "flags": {},
        "shapes": [
            {
                "label": "landmark",
                "points": [[400, 300]],
                "group_id": None,
                "shape_type": "point",
                "flags": {}
            },
            {
                "label": "road",
                "points": [[100, 500], [200, 450], [300, 400]],
                "group_id": 2,
                "shape_type": "linestrip",
                "flags": {}
            }
        ],
        "imagePath": "images/image2.jpg",
        "imageData": None,
        "imageHeight": 600,
        "imageWidth": 800
    }
    
    # Save JSON files
    with open(tmp_path / "image1.json", 'w') as f:
        json.dump(json_data1, f, indent=2)
    
    with open(tmp_path / "image2.json", 'w') as f:
        json.dump(json_data2, f, indent=2)
    
    return tmp_path


def test_labelme_shapes_construction():
    """Test construction of all LabelMe shape types"""
    
    # Test Polygon
    polygon = LabelMePolygon([[0, 0], [10, 0], [10, 10], [0, 10]])
    assert polygon.shape_type == "polygon"
    assert len(polygon.points) == 4
    assert polygon.getCoordinates() == [[0, 0], [10, 0], [10, 10], [0, 10]]
    
    # Test invalid polygon (less than 3 points)
    with pytest.raises(ValueError):
        LabelMePolygon([[0, 0], [10, 0]])
    
    # Test Rectangle
    rectangle = LabelMeRectangle(10, 20, 50, 60)
    assert rectangle.shape_type == "rectangle"
    assert rectangle.getCoordinates() == [[10, 20], [50, 60]]
    
    # Test Circle
    circle = LabelMeCircle(100, 100, 110, 100)
    assert circle.shape_type == "circle"
    assert circle.radius == 10.0
    assert circle.getCoordinates() == [[100, 100], [110, 100]]
    
    # Test Point
    point = LabelMePoint(50, 75)
    assert point.shape_type == "point"
    assert point.getCoordinates() == [[50, 75]]
    
    # Test Line
    line = LabelMeLine(0, 0, 100, 100)
    assert line.shape_type == "line"
    assert line.getCoordinates() == [[0, 0], [100, 100]]
    
    # Test Linestrip
    linestrip = LabelMeLinestrip([[0, 0], [50, 50], [100, 0]])
    assert linestrip.shape_type == "linestrip"
    assert len(linestrip.points) == 3
    
    # Test invalid linestrip (less than 2 points)
    with pytest.raises(ValueError):
        LabelMeLinestrip([[0, 0]])
    
    # Test Points collection
    points = LabelMePoints([[10, 10], [20, 20], [30, 30]])
    assert points.shape_type == "points"
    assert len(points.points) == 3
    
    # Test Mask
    mask_data = b"binary_mask_data"
    mask = LabelMeMask(0, 0, 100, 100, mask_data)
    assert mask.shape_type == "mask"
    assert mask.mask_data == mask_data


def test_bounding_box_calculations():
    """Test bounding box calculations for all shape types"""
    
    # Test Polygon bounding box
    polygon = LabelMePolygon([[10, 20], [50, 10], [60, 40], [20, 50]])
    bbox = polygon.get_bounding_box()
    assert bbox.x_min == 10
    assert bbox.y_min == 10
    assert bbox.x_max == 60
    assert bbox.y_max == 50
    
    # Test Circle bounding box
    circle = LabelMeCircle(100, 100, 110, 100)  # radius = 10
    bbox = circle.get_bounding_box()
    assert bbox.x_min == 90
    assert bbox.y_min == 90
    assert bbox.x_max == 110
    assert bbox.y_max == 110
    
    # Test Point bounding box (zero area)
    point = LabelMePoint(25, 35)
    bbox = point.get_bounding_box()
    assert bbox.x_min == 25
    assert bbox.y_min == 35
    assert bbox.x_max == 25
    assert bbox.y_max == 35


def test_labelme_format_construction(sample_labelme_dataset):
    """Test LabelMe format construction from folder"""
    
    labelme_format = LabelMeFormat.read_from_folder(str(sample_labelme_dataset))
    
    # Check basic structure
    assert labelme_format.folder_path == str(sample_labelme_dataset)
    assert labelme_format.name == sample_labelme_dataset.name
    assert isinstance(labelme_format.files, list)
    
    # Check files
    assert len(labelme_format.files) == 2
    filenames = {f.filename for f in labelme_format.files}
    assert "image1.jpg" in filenames
    assert "image2.jpg" in filenames
    
    # Check first file details
    file1 = next(f for f in labelme_format.files if f.filename == "image1.jpg")
    assert file1.version == "5.0.1"
    assert file1.imageHeight == 480
    assert file1.imageWidth == 640
    assert file1.flags
    assert file1.flags["reviewed"] == True
    assert len(file1.annotations) == 3
    
    # Check polygon annotation
    polygon_ann = next(a for a in file1.annotations if a.label == "person")
    assert isinstance(polygon_ann.geometry, LabelMePolygon)
    assert polygon_ann.group_id == 1
    assert polygon_ann.flags
    assert polygon_ann.flags["difficult"] == False
    
    # Check rectangle annotation
    rect_ann = next(a for a in file1.annotations if a.label == "car")
    assert isinstance(rect_ann.geometry, LabelMeRectangle)
    assert rect_ann.geometry.x_min == 300
    assert rect_ann.geometry.y_min == 300
    assert rect_ann.geometry.x_max == 400
    assert rect_ann.geometry.y_max == 400
    
    # Check circle annotation
    circle_ann = next(a for a in file1.annotations if a.label == "wheel")
    assert isinstance(circle_ann.geometry, LabelMeCircle)
    assert circle_ann.geometry.center_x == 350
    assert circle_ann.geometry.center_y == 350


def test_labelme_annotations_subfolder(tmp_path):
    """Test reading from annotations subfolder structure"""
    
    # Create annotations subfolder
    annotations_dir = tmp_path / "annotations"
    annotations_dir.mkdir()
    
    json_data = {
        "version": "5.0.1",
        "flags": {},
        "shapes": [
            {
                "label": "test",
                "points": [[10, 10]],
                "shape_type": "point",
                "flags": {}
            }
        ],
        "imagePath": "test.jpg",
        "imageHeight": 100,
        "imageWidth": 100
    }
    
    with open(annotations_dir / "test.json", 'w') as f:
        json.dump(json_data, f)
    
    labelme_format = LabelMeFormat.read_from_folder(str(tmp_path))
    assert len(labelme_format.files) == 1
    assert labelme_format.files[0].filename == "test.jpg"


def test_invalid_dataset_structure(tmp_path):
    """Test error handling for invalid dataset structures"""
    
    # Case: folder doesn't exist
    with pytest.raises(FileNotFoundError):
        LabelMeFormat.read_from_folder(str(tmp_path / "nonexistent"))
    
    # Case: no JSON files
    with pytest.raises(FileNotFoundError):
        LabelMeFormat.read_from_folder(str(tmp_path))


def test_labelme_format_save_full(tmp_path):
    """Test complete save functionality"""
    
    # Create test data with various shape types
    polygon = LabelMePolygon([[10, 10], [50, 10], [50, 50], [10, 50]])
    rectangle = LabelMeRectangle(100, 100, 200, 150)
    circle = LabelMeCircle(300, 300, 310, 300)
    point = LabelMePoint(400, 400)
    line = LabelMeLine(500, 500, 550, 550)
    
    annotations = [
        LabelMeAnnotation(polygon, "building", group_id=1, flags={"verified": True}),
        LabelMeAnnotation(rectangle, "car", description="Red car"),
        LabelMeAnnotation(circle, "wheel"),
        LabelMeAnnotation(point, "landmark"),
        LabelMeAnnotation(line, "boundary")
    ]
    
    labelme_file = LabelMeFile(
        filename="test_image.jpg",
        annotations=annotations,
        version="5.0.1",
        imagePath="images/test_image.jpg",
        imageHeight=600,
        imageWidth=800,
        flags={"quality": "high"}
    )
    
    labelme_format = LabelMeFormat("test_dataset", [labelme_file])
    
    # Execute save
    labelme_format.save(str(tmp_path))
    
    # Check saved file
    json_file = tmp_path / "test_image.json"
    assert json_file.exists()
    
    # Load and verify content
    with open(json_file, 'r') as f:
        saved_data = json.load(f)
    
    assert saved_data["version"] == "5.0.1"
    assert saved_data["flags"]["quality"] == "high"
    assert saved_data["imageHeight"] == 600
    assert saved_data["imageWidth"] == 800
    assert len(saved_data["shapes"]) == 5
    
    # Check polygon shape
    polygon_shape = next(s for s in saved_data["shapes"] if s["label"] == "building")
    assert polygon_shape["shape_type"] == "polygon"
    assert polygon_shape["group_id"] == 1
    assert polygon_shape["flags"]["verified"] == True
    assert len(polygon_shape["points"]) == 4
    
    # Check rectangle shape
    rect_shape = next(s for s in saved_data["shapes"] if s["label"] == "car")
    assert rect_shape["shape_type"] == "rectangle"
    assert rect_shape["points"] == [[100, 100], [200, 150]]
    assert rect_shape["description"] == "Red car"
    
    # Check circle shape
    circle_shape = next(s for s in saved_data["shapes"] if s["label"] == "wheel")
    assert circle_shape["shape_type"] == "circle"
    assert circle_shape["points"] == [[300, 300], [310, 300]]


def test_mask_with_base64_data(tmp_path):
    """Test mask handling with base64 encoded data"""
    
    # Create mask with binary data
    mask_data = b"binary_mask_data_example"
    mask = LabelMeMask(0, 0, 100, 100, mask_data)
    
    annotation = LabelMeAnnotation(mask, "segmentation_mask")
    labelme_file = LabelMeFile(
        filename="mask_test.jpg",
        annotations=[annotation],
        version="5.0.1",
        imagePath="mask_test.jpg",
        imageHeight=100,
        imageWidth=100
    )
    
    labelme_format = LabelMeFormat("mask_test", [labelme_file])
    
    # Save and reload
    labelme_format.save(str(tmp_path))
    reloaded_format = LabelMeFormat.read_from_folder(str(tmp_path))
    
    # Verify mask data is preserved
    reloaded_mask = reloaded_format.files[0].annotations[0].geometry
    assert isinstance(reloaded_mask, LabelMeMask)
    assert reloaded_mask.mask_data == mask_data
