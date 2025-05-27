from pathlib import Path
from PIL import Image
import pytest

from formats.yolo import YoloFormat

# Fixture for YOLO dataset
@pytest.fixture
def sample_yolo_dataset(tmp_path):

    # Creating file structure
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    # Creating images
    image1 = Image.new('RGB', (100, 100), color='red')
    image2 = Image.new('RGB', (100, 100), color='blue')

    image1.save(images_dir / 'image1.png')
    image2.save(images_dir / 'image2.png')
    
    # Classes files
    (labels_dir / "classes.txt").write_text("person\ncar\ntruck")
    
    # First Annotation file
    (labels_dir / "image1.txt").write_text(
        "0 0.5 0.5 0.3 0.3\n"
        "1 0.2 0.2 0.1 0.1"
    )
    
    # Second Annotations file
    (labels_dir / "image2.txt").write_text(
        "2 0.7 0.7 0.4 0.4"
    )
    
    return tmp_path

def test_yolo_format_construction(sample_yolo_dataset):

    yolo_format = YoloFormat.read_from_folder(sample_yolo_dataset)
    
    # 1. Checking basic structure
    assert yolo_format.folder_path == sample_yolo_dataset
    assert yolo_format.name == sample_yolo_dataset.name
    assert isinstance(yolo_format.files, list)
    
    # 2. Checking class labels
    assert yolo_format.class_labels == {
                                            0: "person", 
                                            1: "car", 
                                            2: "truck"
                                        }
    assert len(yolo_format.class_labels) == 3
    
    # 3. Checking files
    assert len(yolo_format.files) == 2
    filenames = {f.filename for f in yolo_format.files}
    assert "image1.png" in filenames
    assert "image2.png" in filenames
    
    # 4. Checking bounding boxes
    file1 = next(f for f in yolo_format.files if f.filename == "image1.png")
    assert len(file1.annotations) == 2
    
    # First annotation
    ann1 = file1.annotations[0]
    assert ann1.id_class == 0
    assert ann1.bbox.x_center == 0.5
    assert ann1.bbox.y_center == 0.5
    assert ann1.bbox.width == 0.3
    assert ann1.bbox.height == 0.3
    
    # Second annotation
    ann2 = file1.annotations[1]
    assert ann2.id_class == 1
    assert ann2.bbox.x_center == 0.2
    assert ann2.bbox.y_center == 0.2
    assert ann2.bbox.width == 0.1
    assert ann2.bbox.height == 0.1


def test_invalid_dataset_structure(tmp_path):
    # Case without labels directory
    with pytest.raises(FileNotFoundError):
        YoloFormat.read_from_folder(tmp_path)
    
    # Create empty labels directory
    (tmp_path / "labels").mkdir()
    
    # Case without classes.txt
    with pytest.raises(FileNotFoundError):
        YoloFormat.read_from_folder(tmp_path)

    # Creating classes.txt
    (tmp_path / "labels" / "classes.txt").write_text("person\ncar\ntruck")

    # Case with classes.txt
    assert (tmp_path / "labels" / "classes.txt").exists()