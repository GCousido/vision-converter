import os
from pathlib import Path
from PIL import Image
import pytest
import json
import math

from datasetconverter.formats.vgg import (
    VGGRect, VGGCircle, VGGEllipse, VGGPolygon, VGGPolyline, VGGPoint,
    VGGAnnotation, VGGFile, VGGFormat
)
from datasetconverter.tests.utils_for_tests import normalize_path

# Fixture para dataset VGG
@pytest.fixture
def sample_vgg_dataset(tmp_path):
    """Create a sample VGG dataset with various shape types"""
    
    # Create images directory
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    
    # Create sample images
    image1 = Image.new('RGB', (640, 480), color='red')
    image2 = Image.new('RGB', (800, 600), color='blue')
    
    image1.save(images_dir / 'image1.jpg')
    image2.save(images_dir / 'image2.jpg')
    
    # Create first VGG JSON annotation file
    json_data1 = {
        "image1.jpg123456": {
            "filename": "image1.jpg",
            "size": 123456,
            "regions": [
                {
                    "shape_attributes": {
                        "name": "polygon",
                        "all_points_x": [100, 200, 200, 100],
                        "all_points_y": [100, 100, 200, 200]
                    },
                    "region_attributes": {
                        "type": "person",
                        "difficult": False
                    }
                },
                {
                    "shape_attributes": {
                        "name": "rect",
                        "x": 300,
                        "y": 300,
                        "width": 100,
                        "height": 100
                    },
                    "region_attributes": {
                        "type": "car"
                    }
                },
                {
                    "shape_attributes": {
                        "name": "circle",
                        "cx": 350,
                        "cy": 350,
                        "r": 20
                    },
                    "region_attributes": {
                        "type": "wheel"
                    }
                }
            ],
            "file_attributes": {
                "scene": "urban"
            }
        }
    }
    
    # Create second VGG JSON annotation file
    json_data2 = {
        "image2.jpg789012": {
            "filename": "image2.jpg",
            "size": 789012,
            "regions": [
                {
                    "shape_attributes": {
                        "name": "point",
                        "cx": 400,
                        "cy": 300
                    },
                    "region_attributes": {
                        "type": "landmark"
                    }
                },
                {
                    "shape_attributes": {
                        "name": "polyline",
                        "all_points_x": [100, 200, 300],
                        "all_points_y": [500, 450, 400]
                    },
                    "region_attributes": {
                        "type": "road"
                    }
                },
                {
                    "shape_attributes": {
                        "name": "ellipse",
                        "cx": 500,
                        "cy": 300,
                        "rx": 50,
                        "ry": 30,
                        "theta": 0.5
                    },
                    "region_attributes": {
                        "type": "building"
                    }
                }
            ],
            "file_attributes": {}
        }
    }
    
    # Save JSON files
    with open(tmp_path / "annotations1.json", 'w') as f:
        json.dump(json_data1, f, indent=2)
    
    with open(tmp_path / "annotations2.json", 'w') as f:
        json.dump(json_data2, f, indent=2)
    
    return tmp_path


def test_vgg_shapes_construction():
    """Test construction of all VGG shape types"""
    
    # Test Rectangle
    rect = VGGRect(10, 20, 50, 60)
    assert rect.shape_type == "rect"
    assert rect.x == 10
    assert rect.y == 20
    assert rect.width == 50
    assert rect.height == 60
    coords = rect.getCoordinates()
    assert coords == {"x": 10, "y": 20, "width": 50, "height": 60}
    
    # Test Circle
    circle = VGGCircle(100, 100, 20)
    assert circle.shape_type == "circle"
    assert circle.cx == 100
    assert circle.cy == 100
    assert circle.r == 20
    coords = circle.getCoordinates()
    assert coords == {"cx": 100, "cy": 100, "r": 20}
    
    # Test Ellipse
    ellipse = VGGEllipse(150, 150, 30, 20, 0.5)
    assert ellipse.shape_type == "ellipse"
    assert ellipse.cx == 150
    assert ellipse.cy == 150
    assert ellipse.rx == 30
    assert ellipse.ry == 20
    assert ellipse.theta == 0.5
    coords = ellipse.getCoordinates()
    assert coords == {"cx": 150, "cy": 150, "rx": 30, "ry": 20, "theta": 0.5}
    
    # Test Polygon
    polygon = VGGPolygon([10, 20, 30], [40, 50, 60])
    assert polygon.shape_type == "polygon"
    assert polygon.all_points_x == [10, 20, 30]
    assert polygon.all_points_y == [40, 50, 60]
    coords = polygon.getCoordinates()
    assert coords == {"all_points_x": [10, 20, 30], "all_points_y": [40, 50, 60]}
    
    # Test invalid polygon (less than 3 points)
    with pytest.raises(ValueError):
        VGGPolygon([10, 20], [30, 40])
    
    # Test invalid polygon (mismatched arrays)
    with pytest.raises(ValueError):
        VGGPolygon([10, 20, 30], [40, 50])
    
    # Test Polyline
    polyline = VGGPolyline([100, 200], [300, 400])
    assert polyline.shape_type == "polyline"
    assert polyline.all_points_x == [100, 200]
    assert polyline.all_points_y == [300, 400]
    coords = polyline.getCoordinates()
    assert coords == {"all_points_x": [100, 200], "all_points_y": [300, 400]}
    
    # Test invalid polyline (less than 2 points)
    with pytest.raises(ValueError):
        VGGPolyline([10], [20])
    
    # Test Point
    point = VGGPoint(50, 75)
    assert point.shape_type == "point"
    assert point.cx == 50
    assert point.cy == 75
    coords = point.getCoordinates()
    assert coords == {"cx": 50, "cy": 75}


def test_bounding_box_calculations():
    """Test bounding box calculations for all VGG shape types"""
    
    # Test Rectangle bounding box (direct conversion)
    rect = VGGRect(10, 20, 50, 60)
    bbox = rect.getBoundingBox()
    assert bbox.x_min == 10
    assert bbox.y_min == 20
    assert bbox.x_max == 60  # x + width
    assert bbox.y_max == 80  # y + height
    
    # Test Circle bounding box
    circle = VGGCircle(100, 100, 20)
    bbox = circle.getBoundingBox()
    assert bbox.x_min == 80   # cx - r
    assert bbox.y_min == 80   # cy - r
    assert bbox.x_max == 120  # cx + r
    assert bbox.y_max == 120  # cy + r
    
    # Test Ellipse bounding box (no rotation)
    ellipse = VGGEllipse(100, 100, 30, 20, 0)
    bbox = ellipse.getBoundingBox()
    assert bbox.x_min == 70   # cx - rx
    assert bbox.y_min == 80   # cy - ry
    assert bbox.x_max == 130  # cx + rx
    assert bbox.y_max == 120  # cy + ry
    
    # Test Ellipse bounding box (no rotation)
    ellipse = VGGEllipse(100, 100, 30, 20, 0)
    bbox = ellipse.getBoundingBox()
    assert bbox.x_min == 70   # cx - rx
    assert bbox.y_min == 80   # cy - ry
    assert bbox.x_max == 130  # cx + rx
    assert bbox.y_max == 120  # cy + ry
    
    # Test Ellipse bounding box (with rotation)
    ellipse_rotated = VGGEllipse(100, 100, 30, 20, math.pi/4)
    bbox = ellipse_rotated.getBoundingBox()
    # For rotated elipse by 45°, bounding box should be smaller that not rotated
    assert abs(bbox.x_min - 74.5) < 1  # ~74.50
    assert abs(bbox.y_min - 74.5) < 1  # ~74.50  
    assert abs(bbox.x_max - 125.5) < 1  # ~125.49
    assert abs(bbox.y_max - 125.5) < 1  # ~125.49
    
    # Check rotated boundingbox its different that not rotated
    assert bbox.x_min > 70  # Bigger
    assert bbox.y_min < 80  # Smaller
    
    # Test Polygon bounding box
    polygon = VGGPolygon([10, 50, 60, 20], [20, 10, 40, 50])
    bbox = polygon.getBoundingBox()
    assert bbox.x_min == 10  # min(all_points_x)
    assert bbox.y_min == 10  # min(all_points_y)
    assert bbox.x_max == 60  # max(all_points_x)
    assert bbox.y_max == 50  # max(all_points_y)
    
    # Test Polyline bounding box
    polyline = VGGPolyline([100, 200, 150], [300, 250, 400])
    bbox = polyline.getBoundingBox()
    assert bbox.x_min == 100
    assert bbox.y_min == 250
    assert bbox.x_max == 200
    assert bbox.y_max == 400
    
    # Test Point bounding box (minimal area)
    point = VGGPoint(25, 35)
    bbox = point.getBoundingBox()
    assert bbox.x_min == 25
    assert bbox.y_min == 35
    assert bbox.x_max == 26  # cx + 1
    assert bbox.y_max == 36  # cy + 1


def test_vgg_format_construction(sample_vgg_dataset):
    """Test VGG format construction from JSON file"""
    
    # Test reading first annotation file
    vgg_format = VGGFormat.read_from_folder(str(sample_vgg_dataset / "annotations1.json"))
    
    # Check basic structure
    assert vgg_format.folder_path == str(sample_vgg_dataset)
    assert vgg_format.name == "annotations1"
    assert isinstance(vgg_format.files, list)
    
    # Check files
    assert len(vgg_format.files) == 1
    file1 = vgg_format.files[0]
    assert file1.filename == "image1.jpg"
    assert file1.size == 123456
    assert len(file1.annotations) == 3
    
    # Check file attributes
    assert file1.file_attributes["scene"] == "urban"
    
    # Check polygon annotation
    polygon_ann = next(a for a in file1.annotations if a.region_attributes.get("type") == "person")
    assert isinstance(polygon_ann.geometry, VGGPolygon)
    assert polygon_ann.geometry.all_points_x == [100, 200, 200, 100]
    assert polygon_ann.geometry.all_points_y == [100, 100, 200, 200]
    assert polygon_ann.region_attributes["difficult"] == False
    
    # Check rectangle annotation
    rect_ann = next(a for a in file1.annotations if a.region_attributes.get("type") == "car")
    assert isinstance(rect_ann.geometry, VGGRect)
    assert rect_ann.geometry.x == 300
    assert rect_ann.geometry.y == 300
    assert rect_ann.geometry.width == 100
    assert rect_ann.geometry.height == 100
    
    # Check circle annotation
    circle_ann = next(a for a in file1.annotations if a.region_attributes.get("type") == "wheel")
    assert isinstance(circle_ann.geometry, VGGCircle)
    assert circle_ann.geometry.cx == 350
    assert circle_ann.geometry.cy == 350
    assert circle_ann.geometry.r == 20


def test_vgg_format_all_shapes(sample_vgg_dataset):
    """Test VGG format with all shape types"""
    
    # Test reading second annotation file with all remaining shapes
    vgg_format = VGGFormat.read_from_folder(str(sample_vgg_dataset / "annotations2.json"))
    
    assert len(vgg_format.files) == 1
    file2 = vgg_format.files[0]
    assert file2.filename == "image2.jpg"
    assert file2.size == 789012
    assert len(file2.annotations) == 3
    
    # Check point annotation
    point_ann = next(a for a in file2.annotations if a.region_attributes.get("type") == "landmark")
    assert isinstance(point_ann.geometry, VGGPoint)
    assert point_ann.geometry.cx == 400
    assert point_ann.geometry.cy == 300
    
    # Check polyline annotation
    polyline_ann = next(a for a in file2.annotations if a.region_attributes.get("type") == "road")
    assert isinstance(polyline_ann.geometry, VGGPolyline)
    assert polyline_ann.geometry.all_points_x == [100, 200, 300]
    assert polyline_ann.geometry.all_points_y == [500, 450, 400]
    
    # Check ellipse annotation
    ellipse_ann = next(a for a in file2.annotations if a.region_attributes.get("type") == "building")
    assert isinstance(ellipse_ann.geometry, VGGEllipse)
    assert ellipse_ann.geometry.cx == 500
    assert ellipse_ann.geometry.cy == 300
    assert ellipse_ann.geometry.rx == 50
    assert ellipse_ann.geometry.ry == 30
    assert ellipse_ann.geometry.theta == 0.5


def test_invalid_dataset_structure(tmp_path):
    """Test error handling for invalid dataset structures"""
    
    # Case: file doesn't exist
    with pytest.raises(FileNotFoundError):
        VGGFormat.read_from_folder(str(tmp_path / "nonexistent.json"))
    
    # Case: invalid JSON
    invalid_json = tmp_path / "invalid.json"
    with open(invalid_json, 'w') as f:
        f.write("invalid json content")
    
    with pytest.raises(json.JSONDecodeError):
        VGGFormat.read_from_folder(str(invalid_json))


def test_vgg_format_save_full(tmp_path):
    """Test complete save functionality"""
    
    # Create test data with various shape types
    rect = VGGRect(10, 10, 50, 50)
    circle = VGGCircle(100, 100, 25)
    ellipse = VGGEllipse(200, 200, 40, 30, 0.5)
    polygon = VGGPolygon([300, 350, 350, 300], [300, 300, 350, 350])
    polyline = VGGPolyline([400, 450, 500], [400, 450, 400])
    point = VGGPoint(550, 550)
    
    annotations = [
        VGGAnnotation(rect, {"type": "building", "verified": True}),
        VGGAnnotation(circle, {"type": "wheel", "color": "black"}),
        VGGAnnotation(ellipse, {"type": "pool", "depth": "deep"}),
        VGGAnnotation(polygon, {"type": "garden"}),
        VGGAnnotation(polyline, {"type": "path"}),
        VGGAnnotation(point, {"type": "landmark"})
    ]
    
    vgg_file = VGGFile(
        filename="test_image.jpg",
        size=987654,
        annotations=annotations,
        file_attributes={"quality": "high", "weather": "sunny"}
    )
    
    vgg_format = VGGFormat("test_dataset", [vgg_file])
    
    # Execute save
    vgg_format.save(str(tmp_path))
    
    # Check saved file
    json_file = tmp_path / "annotations.json"
    assert json_file.exists()
    
    # Load and verify content
    with open(json_file, 'r') as f:
        saved_data = json.load(f)
    
    # Check structure
    assert "_via_img_metadata" in saved_data
    img_key = "test_image.jpg987654"
    assert img_key in saved_data["_via_img_metadata"]
    
    img_data = saved_data["_via_img_metadata"][img_key]
    assert img_data["filename"] == "test_image.jpg"
    assert img_data["size"] == 987654
    assert img_data["file_attributes"]["quality"] == "high"
    assert img_data["file_attributes"]["weather"] == "sunny"
    assert len(img_data["regions"]) == 6
    
    # Check rectangle shape
    rect_region = next(r for r in img_data["regions"] if r["region_attributes"].get("type") == "building")
    assert rect_region["shape_attributes"]["name"] == "rect"
    assert rect_region["shape_attributes"]["x"] == 10
    assert rect_region["shape_attributes"]["y"] == 10
    assert rect_region["shape_attributes"]["width"] == 50
    assert rect_region["shape_attributes"]["height"] == 50
    assert rect_region["region_attributes"]["verified"] == True
    
    # Check circle shape
    circle_region = next(r for r in img_data["regions"] if r["region_attributes"].get("type") == "wheel")
    assert circle_region["shape_attributes"]["name"] == "circle"
    assert circle_region["shape_attributes"]["cx"] == 100
    assert circle_region["shape_attributes"]["cy"] == 100
    assert circle_region["shape_attributes"]["r"] == 25
    assert circle_region["region_attributes"]["color"] == "black"
    
    # Check polygon shape
    polygon_region = next(r for r in img_data["regions"] if r["region_attributes"].get("type") == "garden")
    assert polygon_region["shape_attributes"]["name"] == "polygon"
    assert polygon_region["shape_attributes"]["all_points_x"] == [300, 350, 350, 300]
    assert polygon_region["shape_attributes"]["all_points_y"] == [300, 300, 350, 350]


def test_vgg_format_real_data_compatibility(tmp_path):
    """Test compatibility with real VGG Image Annotator data format"""
    
    # Create a JSON file similar to the Potatodata_json.json structure
    real_data = {
        "001.jpg71584": {
            "filename": "001.jpg",
            "size": 71584,
            "regions": [
                {
                    "shape_attributes": {
                        "name": "polygon",
                        "all_points_x": [163, 147, 156, 142, 149, 160],
                        "all_points_y": [32, 54, 56, 68, 79, 83]
                    },
                    "region_attributes": {
                        "names": "crop"
                    }
                }
            ],
            "file_attributes": {}
        }
    }
    
    json_file = tmp_path / "real_data.json"
    with open(json_file, 'w') as f:
        json.dump(real_data, f)
    
    # Load and verify
    vgg_format = VGGFormat.read_from_folder(str(json_file))
    
    assert len(vgg_format.files) == 1
    file = vgg_format.files[0]
    assert file.filename == "001.jpg"
    assert file.size == 71584
    assert len(file.annotations) == 1
    
    annotation = file.annotations[0]
    assert isinstance(annotation.geometry, VGGPolygon)
    assert annotation.geometry.all_points_x == [163, 147, 156, 142, 149, 160]
    assert annotation.geometry.all_points_y == [32, 54, 56, 68, 79, 83]
    assert annotation.region_attributes["names"] == "crop"


def test_vgg_format_via_metadata_structure(tmp_path):
    """Test VGG format with full VIA metadata structure"""
    
    # Create JSON with full VIA structure
    via_data = {
        "_via_settings": {
            "ui": {"annotation_editor_height": 25}
        },
        "_via_img_metadata": {
            "test.jpg12345": {
                "filename": "test.jpg",
                "size": 12345,
                "regions": [
                    {
                        "shape_attributes": {
                            "name": "rect",
                            "x": 10,
                            "y": 10,
                            "width": 100,
                            "height": 100
                        },
                        "region_attributes": {
                            "label": "object"
                        }
                    }
                ],
                "file_attributes": {}
            }
        },
        "_via_attributes": {
            "region": {
                "label": {
                    "type": "text",
                    "description": "Object label"
                }
            }
        },
        "_via_data_format_version": "2.0.10",
        "_via_image_id_list": ["test.jpg12345"]
    }
    
    json_file = tmp_path / "via_full.json"
    with open(json_file, 'w') as f:
        json.dump(via_data, f)
    
    # Load and verify it works with full VIA structure
    vgg_format = VGGFormat.read_from_folder(str(json_file))
    
    assert len(vgg_format.files) == 1
    file = vgg_format.files[0]
    assert file.filename == "test.jpg"
    assert file.size == 12345
    assert len(file.annotations) == 1
    
    annotation = file.annotations[0]
    assert isinstance(annotation.geometry, VGGRect)
    assert annotation.region_attributes["label"] == "object"

# Fixture for VGG dataset with images
@pytest.fixture()
def vgg_image_dataset(tmp_path):
    """Fixture for tests with images in VGG format"""
    dataset_dir = tmp_path / "vgg_dataset"
    dataset_dir.mkdir()
    
    # Create annotations.json
    annotations_data = {
        "_via_img_metadata": {
            "img1.jpg12345": {
                "filename": "img1.jpg",
                "size": 12345,
                "regions": [],
                "file_attributes": {}
            },
            "img2.png67890": {
                "filename": "img2.png",
                "size": 67890,
                "regions": [],
                "file_attributes": {}
            }
        }
    }
    
    annotations_file = dataset_dir / "annotations.json"
    with open(annotations_file, 'w', encoding='utf-8') as f:
        json.dump(annotations_data, f, indent=4)
    
    # Create images
    images_dir = dataset_dir / "images"
    images_dir.mkdir()
    (images_dir / "img1.jpg").write_bytes(b"image_data")
    (images_dir / "img2.png").write_bytes(b"image_data")
    
    return dataset_dir

# Fixture for VGGFormat instance
@pytest.fixture()
def vgg_format_image_instance(tmp_path):
    """Fixture for VGGFormat instance with images"""
    img_dir = tmp_path / "source_imgs"
    img_dir.mkdir()
    img1 = img_dir / "test_img1.jpg"
    img2 = img_dir / "test_img2.png"
    img1.write_bytes(b"source1")
    img2.write_bytes(b"source2")
    
    # Create VGGFile instances
    files = [
        VGGFile(filename="test_img1.jpg", size=12345, annotations=[]),
        VGGFile(filename="test_img2.png", size=67890, annotations=[])
    ]
    
    return VGGFormat(
        name="vgg_test",
        files=files,
        folder_path=str(tmp_path),
        images_path_list=[str(img1), str(img2)]
    )

# Read tests
def test_vgg_read_with_copy_images(vgg_image_dataset):
    """Test reading VGG dataset with image copying enabled"""
    vgg = VGGFormat.read_from_folder(
        str(vgg_image_dataset),
        copy_images=True,
        copy_as_links=False
    )
    assert vgg.images_path_list is not None
    assert len(vgg.images_path_list) == 2
    assert all(
        any(img_name in p for p in vgg.images_path_list)
        for img_name in ['img1.jpg', 'img2.png']
    )

def test_vgg_read_with_links(vgg_image_dataset):
    """Test reading VGG dataset with symbolic links enabled"""
    vgg = VGGFormat.read_from_folder(
        str(vgg_image_dataset),
        copy_images=False,
        copy_as_links=True
    )
    assert vgg.images_path_list is not None
    assert len(vgg.images_path_list) == 2
    assert all(
        any(img_name in p for p in vgg.images_path_list)
        for img_name in ['img1.jpg', 'img2.png']
    )

def test_vgg_read_without_copy(vgg_image_dataset):
    """Test reading VGG dataset without image handling"""
    vgg = VGGFormat.read_from_folder(
        str(vgg_image_dataset),
        copy_images=False,
        copy_as_links=False
    )
    assert vgg.images_path_list is None

# Save tests
def test_vgg_save_with_copy_images(vgg_format_image_instance, tmp_path):
    """Test saving VGG dataset with image copying"""
    output_dir = tmp_path / "output"
    vgg_format_image_instance.save(
        str(output_dir),
        copy_images=True,
        copy_as_links=False
    )
    
    # Verify that images were copied
    output_images = list((output_dir / "images").iterdir())
    assert len(output_images) == 2
    assert (output_dir / "images" / "test_img1.jpg").exists()
    assert (output_dir / "images" / "test_img2.png").exists()

def test_vgg_save_with_links(vgg_format_image_instance, tmp_path):
    """Test saving VGG dataset with symbolic links (with Windows permission check)"""
    # Windows symlink permission check
    if os.name == "nt":
        try:
            test_link = tmp_path / "test_link"
            test_target = tmp_path / "test_target.txt"
            test_target.write_text("test")
            test_link.symlink_to(test_target)
        except OSError as e:
            if e.winerror == 1314:
                pytest.skip("Symlinks require admin privileges on Windows")
            else:
                raise
                
    output_dir = tmp_path / "output"
    vgg_format_image_instance.save(
        str(output_dir),
        copy_images=False,
        copy_as_links=True
    )
    
    # Verify that symbolic links were created
    img1 = output_dir / "images" / "test_img1.jpg"
    img2 = output_dir / "images" / "test_img2.png"
    assert img1.is_symlink()
    assert Path(normalize_path(os.readlink(img1))).resolve() == Path(vgg_format_image_instance.images_path_list[0]).resolve()
    assert img2.is_symlink()
    assert Path(normalize_path(os.readlink(img2))).resolve() == Path(vgg_format_image_instance.images_path_list[1]).resolve()

def test_vgg_save_without_copy(vgg_format_image_instance, tmp_path):
    """Test saving VGG dataset without image handling"""
    output_dir = tmp_path / "output"
    vgg_format_image_instance.save(
        str(output_dir),
        copy_images=False,
        copy_as_links=False
    )
    
    images_dir = output_dir / "images"
    assert not any(images_dir.iterdir()) if images_dir.exists() else True
