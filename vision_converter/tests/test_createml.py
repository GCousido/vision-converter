from pathlib import Path
from PIL import Image
import pytest
import json
import os

from vision_converter.formats.createml import CreateMLAnnotation, CreateMLBoundingBox, CreateMLFile, CreateMLFormat
from vision_converter.tests.utils_for_tests import normalize_path

# Fixture for CreateML dataset
@pytest.fixture
def sample_createml_dataset(tmp_path):
    """Creates a sample CreateML dataset structure for testing."""
    
    # Creating file structure
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    # Creating sample images
    image1 = Image.new('RGB', (640, 480), color='red')
    image2 = Image.new('RGB', (800, 600), color='blue')

    image1.save(images_dir / 'image1.jpg')
    image2.save(images_dir / 'image2.png')
    
    # Creating annotations.json file
    annotations_data = [
        {
            "image": "image1.jpg",
            "annotations": [
                {
                    "label": "person",
                    "coordinates": {
                        "x": 320,
                        "y": 240,
                        "width": 100,
                        "height": 150
                    }
                },
                {
                    "label": "car",
                    "coordinates": {
                        "x": 500,
                        "y": 300,
                        "width": 80,
                        "height": 60
                    }
                }
            ]
        },
        {
            "image": "image2.png",
            "annotations": [
                {
                    "label": "truck",
                    "coordinates": {
                        "x": 400,
                        "y": 300,
                        "width": 120,
                        "height": 180
                    }
                }
            ]
        }
    ]
    
    annotations_file = tmp_path / "annotations.json"
    with open(annotations_file, 'w', encoding='utf-8') as f:
        json.dump(annotations_data, f, indent=4)
    
    return annotations_file


def test_createml_format_construction(sample_createml_dataset):
    
    createml_format = CreateMLFormat.read_from_folder(sample_createml_dataset)
    
    # 1. Check basic structure
    assert createml_format.folder_path == str(sample_createml_dataset.parent)
    assert createml_format.name == "CreateMLDataset"
    assert isinstance(createml_format.files, list)
    
    # 2. Check files count
    assert len(createml_format.files) == 2
    filenames = {f.filename for f in createml_format.files}
    assert "image1.jpg" in filenames
    assert "image2.png" in filenames
    
    # 3. Check annotations for first image
    file1 = next(f for f in createml_format.files if f.filename == "image1.jpg")
    assert len(file1.annotations) == 2
    
    # First annotation - person
    ann1 = file1.annotations[0]
    assert ann1.label == "person"
    assert ann1.geometry.x_center == 320
    assert ann1.geometry.y_center == 240
    assert ann1.geometry.width == 100
    assert ann1.geometry.height == 150
    
    # Second annotation - car
    ann2 = file1.annotations[1]
    assert ann2.label == "car"
    assert ann2.geometry.x_center == 500
    assert ann2.geometry.y_center == 300
    assert ann2.geometry.width == 80
    assert ann2.geometry.height == 60
    
    # 4. Check annotations for second image
    file2 = next(f for f in createml_format.files if f.filename == "image2.png")
    assert len(file2.annotations) == 1
    
    ann3 = file2.annotations[0]
    assert ann3.label == "truck"
    assert ann3.geometry.x_center == 400
    assert ann3.geometry.y_center == 300
    assert ann3.geometry.width == 120
    assert ann3.geometry.height == 180


def test_createml_bounding_box():
    """Test CreateML bounding box functionality."""
    
    bbox = CreateMLBoundingBox(x_center=100, y_center=150, width=50, height=75)
    
    assert bbox.x_center == 100
    assert bbox.y_center == 150
    assert bbox.width == 50
    assert bbox.height == 75
    
    # Test getBoundingBox method
    coords = bbox.getBoundingBox()
    assert coords == [100, 150, 50, 75]


def test_createml_annotation():
    """Test CreateML annotation functionality."""
    
    bbox = CreateMLBoundingBox(x_center=200, y_center=250, width=60, height=80)
    annotation = CreateMLAnnotation(bbox=bbox, label="bicycle")
    
    assert annotation.label == "bicycle"
    assert annotation.geometry.x_center == 200
    assert annotation.geometry.y_center == 250
    assert annotation.geometry.width == 60
    assert annotation.geometry.height == 80


def test_invalid_dataset_structure(tmp_path):
    """Test error handling for invalid dataset structures."""
    
    # Case: folder doesn't exist
    with pytest.raises(FileNotFoundError, match="Invalid path:"):
        CreateMLFormat.read_from_folder(str(tmp_path / "nonexistent"))
    
    ( tmp_path / "annotations.json" ).touch()
    
    # Case: missing images directory
    with pytest.raises(FileNotFoundError, match="Folder .* was not found"):
        CreateMLFormat.read_from_folder(str(tmp_path))


def test_invalid_json_format(tmp_path):
    """Test error handling for invalid JSON format."""

    # Create directory structure
    (tmp_path / "images").mkdir()
    
    # Create invalid JSON file
    annotations_file = tmp_path / "annotations.json"
    annotations_file.write_text("invalid json content")
    
    with pytest.raises(ValueError, match="Invalid JSON format"):
        CreateMLFormat.read_from_folder(str(annotations_file))


def test_missing_image_files(tmp_path):
    """Test error handling when image files are missing."""

    path = tmp_path.parent
    # Create directory structure
    images_dir = path / "images"
    images_dir.mkdir()
    
    # Create annotations.json with reference to non-existent image
    annotations_data = [
        {
            "image": "missing_image.jpg",
            "annotations": []
        }
    ]
    
    annotations_file = path / "annotations.json"
    with open(annotations_file, 'w') as f:
        json.dump(annotations_data, f)
    
    with pytest.raises(Exception, match="Dataset structure error: Image file .* not found"):
        CreateMLFormat.read_from_folder(str(annotations_file))


def test_createml_format_save_full(tmp_path: Path):
    """Test complete CreateML dataset save functionality."""
    
    # Prepare test data
    annotations1 = [
        CreateMLAnnotation(CreateMLBoundingBox(320, 240, 100, 150), "person"),
        CreateMLAnnotation(CreateMLBoundingBox(500, 300, 80, 60), "car")
    ]

    annotations2 = [
        CreateMLAnnotation(CreateMLBoundingBox(400, 300, 120, 180), "truck"),
        CreateMLAnnotation(CreateMLBoundingBox(200, 150, 60, 80), "bicycle")
    ]

    createml_files = [
        CreateMLFile("test_image1.jpg", annotations1),
        CreateMLFile("test_image2.png", annotations2)
    ]
    
    createml_format = CreateMLFormat(
        name="test_dataset",
        files=createml_files,
        folder_path=None
    )

    # Execute save
    createml_format.save(str(tmp_path.resolve()))

    # Check file structure
    assert (tmp_path / "images").is_dir()
    assert (tmp_path / "annotations.json").is_file()

    # Check annotations.json content
    annotations_file = tmp_path / "annotations.json"
    with open(annotations_file, 'r', encoding='utf-8') as f:
        saved_data = json.load(f)
    
    assert len(saved_data) == 2
    
    # Check first image annotations
    image1_data = next(item for item in saved_data if item["image"] == "test_image1.jpg")
    assert len(image1_data["annotations"]) == 2
    
    ann1 = image1_data["annotations"][0]
    assert ann1["label"] == "person"
    assert ann1["coordinates"]["x"] == 320
    assert ann1["coordinates"]["y"] == 240
    assert ann1["coordinates"]["width"] == 100
    assert ann1["coordinates"]["height"] == 150
    
    ann2 = image1_data["annotations"][1]
    assert ann2["label"] == "car"
    assert ann2["coordinates"]["x"] == 500
    assert ann2["coordinates"]["y"] == 300
    assert ann2["coordinates"]["width"] == 80
    assert ann2["coordinates"]["height"] == 60
    
    # Check second image annotations
    image2_data = next(item for item in saved_data if item["image"] == "test_image2.png")
    assert len(image2_data["annotations"]) == 2
    
    ann3 = image2_data["annotations"][0]
    assert ann3["label"] == "truck"
    assert ann3["coordinates"]["x"] == 400
    assert ann3["coordinates"]["y"] == 300
    assert ann3["coordinates"]["width"] == 120
    assert ann3["coordinates"]["height"] == 180


def test_empty_annotations(tmp_path):
    """Test handling of images with no annotations."""
    
    # Create directory structure
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    
    # Create sample image
    image = Image.new('RGB', (640, 480), color='green')
    image.save(images_dir / 'empty_image.jpg')
    
    # Create annotations.json with empty annotations
    annotations_data = [
        {
            "image": "empty_image.jpg",
            "annotations": []
        }
    ]
    
    annotations_file = tmp_path / "annotations.json"
    with open(annotations_file, 'w') as f:
        json.dump(annotations_data, f)
    
    # Should work without errors
    createml_format = CreateMLFormat.read_from_folder(str(annotations_file))
    assert len(createml_format.files) == 1
    assert len(createml_format.files[0].annotations) == 0


def test_create_files_from_jsondata():
    """Test the static method create_files_from_jsondata."""
    
    json_data = [
        {
            "image": "test1.jpg",
            "annotations": [
                {
                    "label": "dog",
                    "coordinates": {"x": 100, "y": 200, "width": 50, "height": 75}
                }
            ]
        },
        {
            "image": "test2.png",
            "annotations": []
        }
    ]
    
    files = CreateMLFormat.create_files_from_jsondata(json_data)
    
    assert len(files) == 2
    assert files[0].filename == "test1.jpg"
    assert len(files[0].annotations) == 1
    assert files[0].annotations[0].label == "dog"
    
    assert files[1].filename == "test2.png"
    assert len(files[1].annotations) == 0

# Fixture for CreateML dataset with images
@pytest.fixture()
def createml_image_dataset(tmp_path):
    """Fixture for tests with images in CreateML format"""
    dataset_dir = tmp_path / "image_handling"
    dataset_dir.mkdir()
    
    # Create annotations.json
    annotations_data = [
        {
            "image": "img1.jpg",
            "annotations": []
        },
        {
            "image": "img2.png",
            "annotations": []
        }
    ]
    
    annotations_file = dataset_dir / "annotations.json"
    with open(annotations_file, 'w', encoding='utf-8') as f:
        json.dump(annotations_data, f, indent=4)
    
    # Create images
    images_dir = dataset_dir / "images"
    images_dir.mkdir()
    (images_dir / "img1.jpg").write_bytes(b"image_data")
    (images_dir / "img2.png").write_bytes(b"image_data")
    
    return dataset_dir

# Fixture for CreateMLFormat instance
@pytest.fixture()
def createml_format_image_instance(tmp_path):
    """Fixture for CreateMLFormat instance with images"""
    img_dir = tmp_path / "source_imgs"
    img_dir.mkdir()
    img1 = img_dir / "test_img1.jpg"
    img2 = img_dir / "test_img2.png"
    img1.write_bytes(b"source1")
    img2.write_bytes(b"source2")
    
    # Create CreateMLFile instances
    files = [
        CreateMLFile(filename="test_img1.jpg", annotations=[]),
        CreateMLFile(filename="test_img2.png", annotations=[])
    ]
    
    return CreateMLFormat(
        name="image_test",
        files=files,
        folder_path=str(tmp_path),
        images_path_list=[str(img1), str(img2)]
    )

# Read tests
def test_createml_read_with_copy_images(createml_image_dataset):
    createml = CreateMLFormat.read_from_folder(
        str(createml_image_dataset),
        copy_images=True,
        copy_as_links=False
    )
    assert createml.images_path_list is not None
    assert len(createml.images_path_list) == 2
    assert all(
        any(img_name in p for p in createml.images_path_list)
        for img_name in ['img1.jpg', 'img2.png']
    )

def test_createml_read_with_links(createml_image_dataset):
    createml = CreateMLFormat.read_from_folder(
        str(createml_image_dataset),
        copy_images=False,
        copy_as_links=True
    )
    assert createml.images_path_list is not None
    assert len(createml.images_path_list) == 2
    assert all(
        any(img_name in p for p in createml.images_path_list)
        for img_name in ['img1.jpg', 'img2.png']
    )

def test_createml_read_without_copy(createml_image_dataset):
    createml = CreateMLFormat.read_from_folder(
        str(createml_image_dataset),
        copy_images=False,
        copy_as_links=False
    )
    assert createml.images_path_list is None

# Save tests
def test_createml_save_with_copy_images(createml_format_image_instance, tmp_path):
    output_dir = tmp_path / "output"
    createml_format_image_instance.save(
        str(output_dir),
        copy_images=True,
        copy_as_links=False
    )
    
    output_images = list((output_dir / "images").iterdir())
    assert len(output_images) == 2
    assert (output_dir / "images" / "test_img1.jpg").read_bytes() == b"source1"
    assert (output_dir / "images" / "test_img2.png").read_bytes() == b"source2"

def test_createml_save_with_links(createml_format_image_instance, tmp_path):
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
    createml_format_image_instance.save(
        str(output_dir),
        copy_images=False,
        copy_as_links=True
    )
    
    img1 = output_dir / "images" / "test_img1.jpg"
    img2 = output_dir / "images" / "test_img2.png"
    assert img1.is_symlink()
    assert Path(normalize_path(os.readlink(img1))).resolve()== Path(createml_format_image_instance.images_path_list[0]).resolve()
    assert img2.is_symlink()
    assert Path(normalize_path(os.readlink(img2))).resolve()== Path(createml_format_image_instance.images_path_list[1]).resolve()

def test_createml_save_without_copy(createml_format_image_instance, tmp_path):
    output_dir = tmp_path / "output"
    createml_format_image_instance.save(
        str(output_dir),
        copy_images=False,
        copy_as_links=False
    )
    images_dir = output_dir / "images"
    assert not any(images_dir.iterdir()) if images_dir.exists() else True