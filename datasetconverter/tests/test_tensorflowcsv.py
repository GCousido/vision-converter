from pathlib import Path
from PIL import Image
import pytest
import csv

from datasetconverter.formats.tensorflow_csv import TensorflowCsvAnnotation, TensorflowCsvFile, TensorflowCsvFormat
from datasetconverter.formats.pascal_voc import PascalVocBoundingBox

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
        TensorflowCsvAnnotation(PascalVocBoundingBox(100, 150, 200, 300), "person"),
        TensorflowCsvAnnotation(PascalVocBoundingBox(300, 200, 450, 350), "car")
    ]

    annotations2 = [
        TensorflowCsvAnnotation(PascalVocBoundingBox(400, 300, 600, 500), "truck")
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