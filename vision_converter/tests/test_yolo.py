from pathlib import Path
from PIL import Image
import pytest

from vision_converter.formats.bounding_box import YoloBoundingBox
from vision_converter.formats.yolo import YoloAnnotation, YoloFile, YoloFormat
from vision_converter.tests.utils_for_tests import normalize_path

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
    assert ann1.geometry.x_center == 0.5
    assert ann1.geometry.y_center == 0.5
    assert ann1.geometry.width == 0.3
    assert ann1.geometry.height == 0.3
    
    # Second annotation
    ann2 = file1.annotations[1]
    assert ann2.id_class == 1
    assert ann2.geometry.x_center == 0.2
    assert ann2.geometry.y_center == 0.2
    assert ann2.geometry.width == 0.1
    assert ann2.geometry.height == 0.1

@pytest.mark.parametrize("location", ["labels", "root"])
def test_classes_txt_locations(tmp_path, location):
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    image1 = Image.new('RGB', (100, 100), color='red')
    image1.save(images_dir / 'image1.png')

    (labels_dir / "image1.txt").write_text("0 0.5 0.5 0.3 0.3\n")

    if location == "labels":
        (labels_dir / "classes.txt").write_text("person\ncar\ntruck")
    else:
        (tmp_path / "classes.txt").write_text("person\ncar\ntruck")

    yolo_format = YoloFormat.read_from_folder(tmp_path)
    assert yolo_format.class_labels == {0: "person", 1: "car", 2: "truck"}
    assert len(yolo_format.files) == 1
    assert yolo_format.files[0].filename == "image1.png"

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


def test_yolo_format_save_full(tmp_path: Path):
    # Prepare test data
    class_labels = {1: "dog", 0: "cat", 2: "car"}
    annotations1 = [
        YoloAnnotation(YoloBoundingBox(0.5, 0.5, 0.1, 0.1), 0),
        YoloAnnotation(YoloBoundingBox(0.2, 0.3, 0.4, 0.5), 1)
    ]

    annotations2 = [
        YoloAnnotation(YoloBoundingBox(0.25, 0.25, 0.15, 0.8), 2),
        YoloAnnotation(YoloBoundingBox(0.75, 0.3, 0.5, 0.1), 1)
    ]

    yolo_files = [
        YoloFile("test_image.jpg", annotations1),
        YoloFile("test2_image.jpg", annotations2)
    ]
    yolo = YoloFormat(
        name="test_dataset",
        files=yolo_files,
        class_labels=class_labels,
        folder_path=None
    )

    # Execute save
    yolo.save(str(tmp_path.resolve()))

    # Check file structure
    assert (tmp_path / "labels").is_dir()
    assert (tmp_path / "images").is_dir()

    # Check classes.txt content
    classes_file = tmp_path / "labels" / "classes.txt"
    assert classes_file.exists()
    assert classes_file.read_text().splitlines() == ["cat", "dog", "car"]

    # Check annotations files and content
    test_image = tmp_path / "labels" / "test_image.txt"
    assert test_image.exists()
    assert test_image.read_text() == (
        "0 0.5 0.5 0.1 0.1\n"
        "1 0.2 0.3 0.4 0.5\n"
    )

    test2_image = tmp_path / "labels" / "test2_image.txt"
    assert test2_image.exists()
    assert test2_image.read_text() == (
        "2 0.25 0.25 0.15 0.8\n"
        "1 0.75 0.3 0.5 0.1\n"
    )

# Fixture for YOLO dataset with images
@pytest.fixture()
def yolo_image_dataset(tmp_path):
    """Fixture for tests with images in YOLO format"""
    dataset_dir = tmp_path / "yolo_image_handling"
    dataset_dir.mkdir()
    
    # Create images directory and images
    images_dir = dataset_dir / "images"
    images_dir.mkdir()
    (images_dir / "img1.jpg").write_bytes(b"image_data")
    (images_dir / "img2.png").write_bytes(b"image_data")
    
    # Create labels directory and annotation files
    labels_dir = dataset_dir / "labels"
    labels_dir.mkdir()
    # Create classes.txt
    (labels_dir / "classes.txt").write_text("person\ncar\ntruck")
    # Create annotation files
    (labels_dir / "img1.txt").write_text("0 0.5 0.5 0.3 0.3\n")
    (labels_dir / "img2.txt").write_text("1 0.2 0.2 0.1 0.1\n")
    
    return dataset_dir

# Fixture for YoloFormat instance with images
@pytest.fixture()
def yolo_format_image_instance(tmp_path):
    """Fixture for YoloFormat instance with images"""
    img_dir = tmp_path / "source_imgs"
    img_dir.mkdir()
    img1 = img_dir / "test_img1.jpg"
    img2 = img_dir / "test_img2.png"
    img1.write_bytes(b"source1")
    img2.write_bytes(b"source2")
    
    files = [
        YoloFile(filename="test_img1.jpg", annotations=[
            YoloAnnotation(YoloBoundingBox(0.5, 0.5, 0.3, 0.3), 0)
        ]),
        YoloFile(filename="test_img2.png", annotations=[
            YoloAnnotation(YoloBoundingBox(0.2, 0.2, 0.1, 0.1), 1)
        ])
    ]
    class_labels = {0: "person", 1: "car"}
    
    return YoloFormat(
        name="image_test",
        files=files,
        class_labels=class_labels,
        folder_path=str(tmp_path),
        images_path_list=[str(img1), str(img2)]
    )

# Read tests
def test_yolo_read_with_copy_images(yolo_image_dataset):
    """Test reading YOLO dataset with image copying enabled"""
    yolo = YoloFormat.read_from_folder(
        str(yolo_image_dataset),
        copy_images=True,
        copy_as_links=False
    )
    assert yolo.images_path_list is not None
    assert len(yolo.images_path_list) == 2
    assert all(
        any(img_name in p for p in yolo.images_path_list)
        for img_name in ['img1.jpg', 'img2.png']
    )

def test_yolo_read_with_links(yolo_image_dataset):
    """Test reading YOLO dataset with symbolic links enabled"""
    yolo = YoloFormat.read_from_folder(
        str(yolo_image_dataset),
        copy_images=False,
        copy_as_links=True
    )
    assert yolo.images_path_list is not None
    assert len(yolo.images_path_list) == 2
    assert all(
        any(img_name in p for p in yolo.images_path_list)
        for img_name in ['img1.jpg', 'img2.png']
    )

def test_yolo_read_without_copy(yolo_image_dataset):
    """Test reading YOLO dataset without image handling"""
    yolo = YoloFormat.read_from_folder(
        str(yolo_image_dataset),
        copy_images=False,
        copy_as_links=False
    )
    assert yolo.images_path_list is None

# Save tests
def test_yolo_save_with_copy_images(yolo_format_image_instance, tmp_path):
    """Test saving YOLO dataset with image copying"""
    output_dir = tmp_path / "output"
    yolo_format_image_instance.save(
        str(output_dir),
        copy_images=True,
        copy_as_links=False
    )
    output_images = list((output_dir / "images").iterdir())
    assert len(output_images) == 2
    assert (output_dir / "images" / "test_img1.jpg").read_bytes() == b"source1"
    assert (output_dir / "images" / "test_img2.png").read_bytes() == b"source2"

def test_yolo_save_with_links(yolo_format_image_instance, tmp_path):
    """Test saving YOLO dataset with symbolic links (with Windows permission check)"""
    # Windows symlink permission check
    import os
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
    yolo_format_image_instance.save(
        str(output_dir),
        copy_images=False,
        copy_as_links=True
    )
    img1 = output_dir / "images" / "test_img1.jpg"
    img2 = output_dir / "images" / "test_img2.png"
    assert img1.is_symlink()
    assert Path(normalize_path(os.readlink(img1))).resolve() == Path(yolo_format_image_instance.images_path_list[0]).resolve()
    assert img2.is_symlink()
    assert Path(normalize_path(os.readlink(img2))).resolve() == Path(yolo_format_image_instance.images_path_list[1]).resolve()

def test_yolo_save_without_copy(yolo_format_image_instance, tmp_path):
    """Test saving YOLO dataset without image handling"""
    output_dir = tmp_path / "output"
    yolo_format_image_instance.save(
        str(output_dir),
        copy_images=False,
        copy_as_links=False
    )
    images_dir = output_dir / "images"
    assert not any(images_dir.iterdir()) if images_dir.exists() else True
