from pathlib import Path
from PIL import Image
import pytest
import json

from datasetconverter.formats.createml import CreateMLAnnotation, CreateMLBoundingBox, CreateMLFile, CreateMLFormat

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
    
    return tmp_path


def test_createml_format_construction(sample_createml_dataset):
    
    createml_format = CreateMLFormat.read_from_folder(sample_createml_dataset)
    
    # 1. Check basic structure
    assert createml_format.folder_path == sample_createml_dataset
    assert createml_format.name == sample_createml_dataset.name
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
    assert ann1.bbox.x_center == 320
    assert ann1.bbox.y_center == 240
    assert ann1.bbox.width == 100
    assert ann1.bbox.height == 150
    
    # Second annotation - car
    ann2 = file1.annotations[1]
    assert ann2.label == "car"
    assert ann2.bbox.x_center == 500
    assert ann2.bbox.y_center == 300
    assert ann2.bbox.width == 80
    assert ann2.bbox.height == 60
    
    # 4. Check annotations for second image
    file2 = next(f for f in createml_format.files if f.filename == "image2.png")
    assert len(file2.annotations) == 1
    
    ann3 = file2.annotations[0]
    assert ann3.label == "truck"
    assert ann3.bbox.x_center == 400
    assert ann3.bbox.y_center == 300
    assert ann3.bbox.width == 120
    assert ann3.bbox.height == 180


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
    assert annotation.bbox.x_center == 200
    assert annotation.bbox.y_center == 250
    assert annotation.bbox.width == 60
    assert annotation.bbox.height == 80


def test_invalid_dataset_structure(tmp_path):
    """Test error handling for invalid dataset structures."""
    
    # Case: folder doesn't exist
    with pytest.raises(FileNotFoundError, match="Folder .* was not found"):
        CreateMLFormat.read_from_folder(str(tmp_path / "nonexistent"))
    
    # Case: missing images directory
    with pytest.raises(FileNotFoundError, match="Folder .* was not found"):
        CreateMLFormat.read_from_folder(str(tmp_path))
    
    # Create images directory
    (tmp_path / "images").mkdir()
    
    # Case: missing annotations.json
    with pytest.raises(FileNotFoundError, match="File 'annotations.json' was not found"):
        CreateMLFormat.read_from_folder(str(tmp_path))


def test_invalid_json_format(tmp_path):
    """Test error handling for invalid JSON format."""
    
    # Create directory structure
    (tmp_path / "images").mkdir()
    
    # Create invalid JSON file
    annotations_file = tmp_path / "annotations.json"
    annotations_file.write_text("invalid json content")
    
    with pytest.raises(ValueError, match="Invalid JSON format"):
        CreateMLFormat.read_from_folder(str(tmp_path))


def test_missing_image_files(tmp_path):
    """Test error handling when image files are missing."""
    
    # Create directory structure
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    
    # Create annotations.json with reference to non-existent image
    annotations_data = [
        {
            "image": "missing_image.jpg",
            "annotations": []
        }
    ]
    
    annotations_file = tmp_path / "annotations.json"
    with open(annotations_file, 'w') as f:
        json.dump(annotations_data, f)
    
    with pytest.raises(Exception, match="Dataset structure error: Image file .* not found"):
        CreateMLFormat.read_from_folder(str(tmp_path))


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
    createml_format = CreateMLFormat.read_from_folder(str(tmp_path))
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
