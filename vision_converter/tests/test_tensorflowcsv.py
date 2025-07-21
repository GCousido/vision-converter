import os
from pathlib import Path
from PIL import Image
import pytest
import csv
import tensorflow as tf

from vision_converter.formats.tensorflow_csv import TensorflowCsvAnnotation, TensorflowCsvFile, TensorflowCsvFormat
from vision_converter.formats.bounding_box import CornerAbsoluteBoundingBox
from vision_converter.tests.utils_for_tests import normalize_path

# Fixture for TensorFlow CSV dataset
@pytest.fixture
def sample_tensorflow_csv_dataset(tmp_path):
    # Creating file structure
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    # Creating images
    image1 = Image.new('RGB', (800, 600), color='red')
    image2 = Image.new('RGB', (1000, 800), color='blue')

    image1.save(images_dir / 'image1.jpg')
    image2.save(images_dir / 'image2.jpg')
    
    # Creating CSV annotation file
    csv_content = (
        "filename,width,height,class,xmin,ymin,xmax,ymax\n"
        "image1.jpg,800,600,person,100,150,200,300\n"
        "image1.jpg,800,600,car,300,200,450,350\n"
        "image2.jpg,1000,800,truck,400,300,600,500\n"
    )

    csv_path = tmp_path / "annotations.csv"
    csv_path.write_text(csv_content)
    
    return tmp_path

def test_tensorflow_csv_format_construction(sample_tensorflow_csv_dataset):
    dataset_path = sample_tensorflow_csv_dataset / "annotations.csv"
    tf_format = TensorflowCsvFormat.read_from_folder(str(dataset_path))
    
    # 1. Checking basic structure
    assert tf_format.folder_path == str(sample_tensorflow_csv_dataset)
    assert tf_format.name == "annotations"
    assert isinstance(tf_format.files, list)
    
    # 2. Checking files
    assert len(tf_format.files) == 2
    filenames = {f.filename for f in tf_format.files}
    assert "image1.jpg" in filenames
    assert "image2.jpg" in filenames
    
    # 3. Checking image dimensions
    file1 = next(f for f in tf_format.files if f.filename == "image1.jpg")
    assert file1.width == 800
    assert file1.height == 600
    
    file2 = next(f for f in tf_format.files if f.filename == "image2.jpg")
    assert file2.width == 1000
    assert file2.height == 800
    
    # 4. Checking annotations for image1
    assert len(file1.annotations) == 2
    
    # First annotation
    ann1 = file1.annotations[0]
    assert ann1.class_name == "person"
    assert ann1.geometry.x_min == 100
    assert ann1.geometry.y_min == 150
    assert ann1.geometry.x_max == 200
    assert ann1.geometry.y_max == 300
    
    # Second annotation
    ann2 = file1.annotations[1]
    assert ann2.class_name == "car"
    assert ann2.geometry.x_min == 300
    assert ann2.geometry.y_min == 200
    assert ann2.geometry.x_max == 450
    assert ann2.geometry.y_max == 350
    
    # 5. Checking annotations for image2
    assert len(file2.annotations) == 1
    
    ann3 = file2.annotations[0]
    assert ann3.class_name == "truck"
    assert ann3.geometry.x_min == 400
    assert ann3.geometry.y_min == 300
    assert ann3.geometry.x_max == 600
    assert ann3.geometry.y_max == 500

def test_invalid_csv_file(tmp_path):
    # Case 1: CSV file does not exist
    with pytest.raises(FileNotFoundError):
        TensorflowCsvFormat.read_from_folder(str(tmp_path / "nonexistent.csv"))
    
    # Case 2: CSV file with no headers
    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text("")
    
    with pytest.raises(ValueError, match="CSV file has no headers"):
        TensorflowCsvFormat.read_from_folder(str(empty_csv))
    
    # Case 3: CSV file missing required columns
    invalid_csv = tmp_path / "invalid.csv"
    invalid_csv.write_text("filename,width,height,class,xmin,ymin\n")  # Missing xmax, ymax
    
    with pytest.raises(KeyError, match="CSV must contain columns"):
        TensorflowCsvFormat.read_from_folder(str(invalid_csv))
    
    # Case 4: CSV file with wrong order but correct columns
    reordered_csv = tmp_path / "reordered.csv"
    reordered_csv.write_text(
        "class,filename,ymax,xmax,ymin,xmin,height,width\n"
        "person,test.jpg,300,200,150,100,600,800\n"
    )
    
    # Should work fine - order doesn't matter in CSV
    tf_format = TensorflowCsvFormat.read_from_folder(str(reordered_csv))
    assert len(tf_format.files) == 1

def test_tensorflow_csv_format_save(tmp_path):
    # Prepare test data
    annotations1 = [
        TensorflowCsvAnnotation(CornerAbsoluteBoundingBox(100, 150, 200, 300), "person"),
        TensorflowCsvAnnotation(CornerAbsoluteBoundingBox(300, 200, 450, 350), "car")
    ]

    annotations2 = [
        TensorflowCsvAnnotation(CornerAbsoluteBoundingBox(400, 300, 600, 500), "truck")
    ]

    tf_files = [
        TensorflowCsvFile("image1.jpg", annotations1, 800, 600),
        TensorflowCsvFile("image2.jpg", annotations2, 1000, 800)
    ]

    tf_format = TensorflowCsvFormat(
        name="test_dataset",
        files=tf_files,
        folder_path=None
    )

    # Execute save
    output_folder = tmp_path / "output"
    tf_format.save(str(output_folder))

    # Check file structure
    assert (output_folder / "images").is_dir()
    csv_file = output_folder / "tensorflow.csv"
    assert csv_file.exists()

    # Check CSV content
    with open(csv_file, newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 3

    # Check first row (person annotation)
    row1 = rows[0]
    assert row1['filename'] == "image1.jpg"
    assert int(row1['width']) == 800
    assert int(row1['height']) == 600
    assert row1['class'] == "person"
    assert int(row1['xmin']) == 100
    assert int(row1['ymin']) == 150
    assert int(row1['xmax']) == 200
    assert int(row1['ymax']) == 300

    # Check second row (car annotation)
    row2 = rows[1]
    assert row2['filename'] == "image1.jpg"
    assert row2['class'] == "car"
    assert int(row2['xmin']) == 300
    assert int(row2['ymin']) == 200
    assert int(row2['xmax']) == 450
    assert int(row2['ymax']) == 350

    # Check third row (truck annotation)
    row3 = rows[2]
    assert row3['filename'] == "image2.jpg"
    assert int(row3['width']) == 1000
    assert int(row3['height']) == 800
    assert row3['class'] == "truck"
    assert int(row3['xmin']) == 400
    assert int(row3['ymin']) == 300
    assert int(row3['xmax']) == 600
    assert int(row3['ymax']) == 500

def test_get_unique_classes(sample_tensorflow_csv_dataset):
    """Test helper method to get unique class names."""
    dataset_path = sample_tensorflow_csv_dataset / "annotations.csv"
    tf_format = TensorflowCsvFormat.read_from_folder(str(dataset_path))
    
    # Add get_unique_classes method to your class
    unique_classes = tf_format.get_unique_classes()
    expected_classes = {"person", "car", "truck"}
    
    assert unique_classes == expected_classes
    assert len(unique_classes) == 3

# Fixture for TensorFlow CSV dataset with images
@pytest.fixture
def tensorflow_csv_with_images(tmp_path):
    # Create directory structure
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    
    # Create real images
    img1 = images_dir / "image1.jpg"
    img2 = images_dir / "image2.jpg"
    img1.write_bytes(b"image1_data")
    img2.write_bytes(b"image2_data")
    
    # Create CSV file
    csv_path = tmp_path / "annotations.csv"
    csv_content = (
        "filename,width,height,class,xmin,ymin,xmax,ymax\n"
        "image1.jpg,800,600,person,100,150,200,300\n"
        "image2.jpg,1000,800,truck,400,300,600,500\n"
    )
    csv_path.write_text(csv_content)
    
    return tmp_path


# Fixture for TensorflowCsvFormat instance
@pytest.fixture()
def tensorflow_csv_format_instance(tmp_path):
    """Fixture para instancia TensorflowCsvFormat con imágenes"""
    img_dir = tmp_path / "source_imgs"
    img_dir.mkdir()
    
    img1 = img_dir / "image1.jpg"
    img2 = img_dir / "image2.jpg"
    img1.write_bytes(b"dummy_image_data")
    img2.write_bytes(b"dummy_image_data")
    
    bbox = CornerAbsoluteBoundingBox(10, 20, 30, 40)
    annotation = TensorflowCsvAnnotation(bbox, "test_class")
    
    files = [
        TensorflowCsvFile(
            filename="image1.jpg",
            annotations=[annotation],
            width=100,
            height=100
        ),
        TensorflowCsvFile(
            filename="image2.jpg",
            annotations=[annotation],
            width=100,
            height=100
        )
    ]
    
    return TensorflowCsvFormat(
        name="tensorflow_test",
        files=files,
        folder_path=str(tmp_path),
        images_path_list=[str(img1), str(img2)]
    )

# Tests for read_from_folder
def test_read_tensorflow_with_copy_images(tensorflow_csv_with_images):
    tf_format = TensorflowCsvFormat.read_from_folder(
        str(tensorflow_csv_with_images),
        copy_images=True,
        copy_as_links=False
    )
    
    assert tf_format.images_path_list is not None
    assert len(tf_format.images_path_list) == 2
    assert all(
        any(img_name in p for p in tf_format.images_path_list)
        for img_name in ['image1.jpg', 'image2.jpg']
    )

def test_read_tensorflow_with_links(tensorflow_csv_with_images):
    tf_format = TensorflowCsvFormat.read_from_folder(
        str(tensorflow_csv_with_images),
        copy_images=False,
        copy_as_links=True
    )
    
    assert tf_format.images_path_list is not None
    assert len(tf_format.images_path_list) == 2

def test_read_tensorflow_without_copy(tensorflow_csv_with_images):
    tf_format = TensorflowCsvFormat.read_from_folder(
        str(tensorflow_csv_with_images),
        copy_images=False,
        copy_as_links=False
    )
    
    assert tf_format.images_path_list is None

# Tests for save
def test_save_tensorflow_with_copy_images(tensorflow_csv_format_instance, tmp_path):
    output_dir = tmp_path / "output"
    tensorflow_csv_format_instance.save(
        str(output_dir),
        copy_images=True,
        copy_as_links=False
    )
    
    # Verify that images were copied
    output_images = list((output_dir / "images").iterdir())
    assert len(output_images) == 2
    assert (output_dir / "images" / "image1.jpg").exists()
    assert (output_dir / "images" / "image2.jpg").exists()

def test_save_tensorflow_with_links(tensorflow_csv_format_instance, tmp_path):
    if os.name == "nt":
        try:
            test_link = tmp_path / "test_link"
            test_target = tmp_path / "test_target.txt"
            test_target.write_text("test")
            test_link.symlink_to(test_target)
        except OSError as e:
            if e.winerror == 1314:
                pytest.skip("Symlinks require administrator privileges on Windows")
            else:
                raise
    
    output_dir = tmp_path / "output"
    tensorflow_csv_format_instance.save(
        str(output_dir),
        copy_images=False,
        copy_as_links=True
    )
    
    img1 = output_dir / "images" / "image1.jpg"
    img2 = output_dir / "images" / "image2.jpg"
    assert img1.is_symlink()
    assert img2.is_symlink()
    source_img1 = Path(tensorflow_csv_format_instance.images_path_list[0])
    source_img2 = Path(tensorflow_csv_format_instance.images_path_list[1])
    assert Path(normalize_path(os.readlink(img1))).resolve() == source_img1.resolve()
    assert Path(normalize_path(os.readlink(img2))).resolve() == source_img2.resolve()

def test_save_tensorflow_without_copy(tensorflow_csv_format_instance, tmp_path):
    output_dir = tmp_path / "output"
    tensorflow_csv_format_instance.save(
        str(output_dir),
        copy_images=False,
        copy_as_links=False
    )
    
    images_dir = output_dir / "images"
    assert not any(images_dir.iterdir()) if images_dir.exists() else True

def test_save_tensorflow_binary(tensorflow_csv_format_instance, tmp_path):
    output_dir = tmp_path / "output"

    tensorflow_csv_format_instance.save(
        str(output_dir),
        copy_images = False,
        copy_as_links = False,
        tfrecord = True
    )

    # Define the expected TFRecord path
    tfrecord_path = output_dir / "dataset.tfrecord"
    
    # Check that the TFRecord file was created
    assert tfrecord_path.exists(), "TFRecord file was not created"

    # Define feature schema to parse the TFRecord
    feature_description = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/filename': tf.io.FixedLenFeature([], tf.string),
        'image/width': tf.io.FixedLenFeature([], tf.int64),
        'image/height': tf.io.FixedLenFeature([], tf.int64),
        'image/object/class/text': tf.io.FixedLenFeature([], tf.string),
        'image/object/bbox/xmin': tf.io.FixedLenFeature([], tf.float32),
        'image/object/bbox/ymin': tf.io.FixedLenFeature([], tf.float32),
        'image/object/bbox/xmax': tf.io.FixedLenFeature([], tf.float32),
        'image/object/bbox/ymax': tf.io.FixedLenFeature([], tf.float32),
    }

    def parse_example(example_proto):
        return tf.io.parse_single_example(example_proto, feature_description)

    # Read and parse the TFRecord file
    raw_dataset = tf.data.TFRecordDataset(str(tfrecord_path))
    parsed_dataset = raw_dataset.map(parse_example)

    # Collect all parsed records
    records = list(parsed_dataset)

    # Ensure at least one record exists
    assert len(records) > 0, "TFRecord file is empty"

    # Validate each record
    for record in records:
        filename = record['image/filename'].numpy().decode("utf-8")
        width = record['image/width'].numpy()
        height = record['image/height'].numpy()
        class_text = record['image/object/class/text'].numpy().decode("utf-8")
        xmin = record['image/object/bbox/xmin'].numpy()
        ymin = record['image/object/bbox/ymin'].numpy()
        xmax = record['image/object/bbox/xmax'].numpy()
        ymax = record['image/object/bbox/ymax'].numpy()

        # ✅ Find matching file from original dataset
        matching_file = next((f for f in tensorflow_csv_format_instance.files if f.filename == filename), None)
        assert matching_file is not None, f"Filename {filename} not found in original dataset"

        # ✅ Check width and height
        assert matching_file.width == width, f"Width mismatch for {filename}: {width} != {matching_file.width}"
        assert matching_file.height == height, f"Height mismatch for {filename}: {height} != {matching_file.height}"

        # ✅ Find matching annotation
        matching_ann = next((ann for ann in matching_file.annotations if ann.class_name == class_text and
                        ann.geometry.x_min == xmin and
                        ann.geometry.y_min == ymin and
                        ann.geometry.x_max == xmax and
                        ann.geometry.y_max == ymax), None)

        assert matching_ann is not None, f"Annotation mismatch for {filename}, class {class_text}"