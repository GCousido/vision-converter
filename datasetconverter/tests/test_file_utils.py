import os
import pytest
import tempfile

from PIL import UnidentifiedImageError
from pathlib import Path
from unittest.mock import patch, PropertyMock
from datasetconverter.tests.utils_for_tests import normalize_path
from datasetconverter.utils.file_utils import find_all_images_folders, find_annotation_file, get_image_info_from_file, get_image_path, estimate_file_size

########################################################
#          Tests for get_image_info_from_file 
########################################################

@pytest.mark.parametrize("mode,expected_depth", [
    ('1', 1),
    ('L', 1),
    ('P', 1),
    ('RGB', 3),
    ('RGBA', 4),
    ('CMYK', 4),
    ('YCbCr', 3),
    ('I', 1),
    ('F', 1),
    ('XYZ', 3),  # Unknown mode should default to 3
])
def test_get_image_info_from_file(tmp_path, mode, expected_depth):
    """Test extraction of image width, height, and color depth for various modes."""
    from PIL import Image
    
    # Use TIFF format which supports more modes
    image_path = tmp_path / f"test_{mode}.tiff"
    
    # Create with supported base mode
    base_mode = 'RGB' if mode in ['CMYK', 'YCbCr', 'F'] else mode
    if base_mode not in Image.MODES:
        base_mode = 'RGB'
    
    try:
        img = Image.new(base_mode, (100, 200))
        img.save(image_path, format='TIFF')
    except Exception:
        pytest.skip(f"Mode {base_mode} not supported for TIFF creation")

    with patch('PIL.Image.Image.mode', new_callable=PropertyMock) as mock_mode:
        mock_mode.return_value = mode
        width, height, depth = get_image_info_from_file(str(image_path))
    
    assert width == 100
    assert height == 200
    assert depth == expected_depth


def test_get_image_info_file_not_found():
    """Test that FileNotFoundError is raised for missing file."""
    with pytest.raises(FileNotFoundError):
        get_image_info_from_file("non_existent_file.png")

def test_get_image_info_invalid_image(tmp_path):
    """Test that PIL.UnidentifiedImageError is raised for non-image files."""
    invalid_path = tmp_path / "not_an_image.txt"
    invalid_path.write_text("This is not an image.")
    with pytest.raises(UnidentifiedImageError):
        get_image_info_from_file(str(invalid_path))

########################################################
#            Tests for get_image_path 
########################################################

def test_get_image_path_found(tmp_path):
    """Test that the correct image path is returned for a matching base name."""
    images_folder = tmp_path / "images"
    images_folder.mkdir()
    img_file = images_folder / "sample_image.JPG"
    img_file.write_bytes(b"fake")
    result = get_image_path(str(tmp_path), "images", "sample_image.txt")
    assert result is not None
    assert Path(result).name == "sample_image.JPG"

def test_get_image_path_not_found(tmp_path):
    """Test that None is returned when no matching image is found."""
    images_folder = tmp_path / "images"
    images_folder.mkdir()
    (images_folder / "other_image.png").write_bytes(b"fake")
    result = get_image_path(str(tmp_path), "images", "missing_image.txt")
    assert result is None

def test_get_image_path_case_insensitive(tmp_path):
    """Test that extension matching is case-insensitive."""
    images_folder = tmp_path / "images"
    images_folder.mkdir()
    img_file = images_folder / "photo.PnG"
    img_file.write_bytes(b"fake")
    result = get_image_path(str(tmp_path), "images", "photo.txt")
    assert result is not None
    assert Path(result).name == "photo.PnG"

def test_get_image_path_supported_extensions(tmp_path):
    """Test that only supported extensions are considered."""
    images_folder = tmp_path / "images"
    images_folder.mkdir()
    (images_folder / "img1.jpg").write_bytes(b"fake")
    (images_folder / "img1.tiff").write_bytes(b"fake")
    result = get_image_path(str(tmp_path), "images", "img1.txt")
    # Should match .jpg, not .tiff
    assert result is not None
    assert Path(result).suffix.lower() == ".jpg"


########################################################
#          Tests for estimate_file_size 
########################################################

def test_jpeg_estimate():
    """Test JPEG file size estimation (high compression, 10% of uncompressed)."""
    width, height, depth = 1000, 1000, 3
    expected = int(width * height * depth * 0.1)
    assert estimate_file_size(width, height, depth, '.jpg') == expected
    assert estimate_file_size(width, height, depth, '.jpeg') == expected

def test_png_estimate():
    """Test PNG file size estimation (lossless, 30% of uncompressed)."""
    width, height, depth = 800, 600, 3
    expected = int(width * height * depth * 0.3)
    assert estimate_file_size(width, height, depth, '.png') == expected

def test_bmp_estimate():
    """Test BMP file size estimation (no compression, 100% of uncompressed)."""
    width, height, depth = 500, 500, 1
    expected = int(width * height * depth * 1.0)
    assert estimate_file_size(width, height, depth, '.bmp') == expected

def test_tiff_estimate():
    """Test TIFF file size estimation (moderate compression, 50% of uncompressed)."""
    width, height, depth = 200, 300, 3
    expected = int(width * height * depth * 0.5)
    assert estimate_file_size(width, height, depth, '.tiff') == expected

def test_webp_estimate():
    """Test WebP file size estimation (very high compression, 8% of uncompressed)."""
    width, height, depth = 400, 400, 3
    expected = int(width * height * depth * 0.08)
    assert estimate_file_size(width, height, depth, '.webp') == expected

def test_unknown_extension():
    """Test estimation for unknown file extension (default 20% compression)."""
    width, height, depth = 100, 100, 3
    expected = int(width * height * depth * 0.2)
    assert estimate_file_size(width, height, depth, '.xyz') == expected

def test_case_insensitivity():
    """Test that extension matching is case-insensitive."""
    width, height, depth = 50, 50, 3
    expected = int(width * height * depth * 0.3)
    assert estimate_file_size(width, height, depth, '.PNG') == expected

def test_grayscale_depth():
    """Test estimation with grayscale images (depth=1)."""
    width, height, depth = 256, 256, 1
    expected = int(width * height * depth * 0.1)
    assert estimate_file_size(width, height, depth, '.jpg') == expected


########################################################
#          Tests for find_annotation_file 
########################################################

def test_find_annotation_file_valid_file(tmp_path):
    """Test with valid direct file path matching extension"""
    test_file = tmp_path / "annotations.xml"
    test_file.touch()
    result = find_annotation_file(str(test_file), "xml")
    assert result == str(test_file)

def test_find_annotation_file_wrong_extension(tmp_path):
    """Test direct file path with incorrect extension"""
    test_file = tmp_path / "wrong.txt"
    test_file.touch()
    with pytest.raises(ValueError) as excinfo:
        find_annotation_file(str(test_file), "xml")
    assert "extension mismatch" in str(excinfo.value)

def test_find_single_file_in_directory(tmp_path):
    """Test finding single annotation file in directory"""
    subdir = tmp_path / "dataset"
    subdir.mkdir()
    ann_file = subdir / "coco.json"
    ann_file.touch()
    
    result = find_annotation_file(str(subdir), "json")
    assert result == str(ann_file)

def test_multiple_files_error(tmp_path):
    """Test error when multiple annotation files found"""
    files = [tmp_path / f"ann{i}.json" for i in range(3)]
    for f in files:
        f.touch()
    
    with pytest.raises(ValueError) as excinfo:
        find_annotation_file(str(tmp_path), "json")
    
    assert "3 '.json' files" in str(excinfo.value)
    for f in files[:3]:
        assert str(f) in str(excinfo.value)

def test_no_files_found_error(tmp_path):
    """Test error when no annotation files exist"""
    with pytest.raises(FileNotFoundError):
        find_annotation_file(str(tmp_path), "xml")

def test_case_insensitive_extension(tmp_path):
    """Test case-insensitive extension handling"""
    test_file = tmp_path / "ANNOTATIONS.XML"
    test_file.touch()
    
    # With lowercase extension
    result = find_annotation_file(str(tmp_path), "xml")
    assert result == str(test_file)
    
    # With uppercase extension
    result = find_annotation_file(str(tmp_path), "XML")
    assert result == str(test_file)

def test_normalization_edge_cases(tmp_path):
    """Test extension normalization with edge cases"""
    test_file = tmp_path / "file.json"
    test_file.touch()
    
    # Extension with spaces and dots
    assert find_annotation_file(str(tmp_path), " .JSON ") == str(test_file)
    assert find_annotation_file(str(tmp_path), "json.") == str(test_file)

def test_invalid_path_error():
    """Test error for non-existent path"""
    with pytest.raises(FileNotFoundError):
        find_annotation_file("/invalid/path/123", "json")

def test_nested_directory_search(tmp_path):
    """Test recursive search in nested directories"""
    (tmp_path / "subdir1/subdir2").mkdir(parents=True)
    ann_file = tmp_path / "subdir1/subdir2/annotations.json"
    ann_file.touch()
    
    result = find_annotation_file(str(tmp_path), "json")
    assert result == str(ann_file)

def test_mixed_extensions_in_directory(tmp_path):
    """Test handling of mixed valid/invalid extensions"""
    valid = tmp_path / "correct.xml"
    valid.touch()
    invalid = tmp_path / "wrong.txt"
    invalid.touch()
    
    result = find_annotation_file(str(tmp_path), "xml")
    assert result == str(valid)

########################################################
#          Tests for find_all_images_folders
########################################################

def test_find_all_images_folders_basic():
    with tempfile.TemporaryDirectory() as tmpdir:
        img_folder = Path(tmpdir) / "images"
        img_folder.mkdir()
        (img_folder / "test.jpg").write_text("fake image content")
        result = find_all_images_folders(tmpdir)
        assert str(img_folder.resolve()) in result

def test_find_all_images_folders_multiple_folders():
    with tempfile.TemporaryDirectory() as tmpdir:
        folder1 = Path(tmpdir) / "folder1"
        folder2 = Path(tmpdir) / "folder2"
        folder1.mkdir()
        folder2.mkdir()
        (folder1 / "img1.png").write_text("fake image content")
        (folder2 / "img2.jpeg").write_text("fake image content")
        result = find_all_images_folders(tmpdir)
        assert str(folder1.resolve()) in result
        assert str(folder2.resolve()) in result

def test_find_all_images_folders_no_images():
    with tempfile.TemporaryDirectory() as tmpdir:
        empty_folder = Path(tmpdir) / "empty"
        empty_folder.mkdir()
        with pytest.raises(FileNotFoundError):
            find_all_images_folders(tmpdir)

def test_find_all_images_folders_custom_extensions():
    with tempfile.TemporaryDirectory() as tmpdir:
        folder = Path(tmpdir) / "folder"
        folder.mkdir()
        (folder / "file.tiff").write_text("fake image content")
        with pytest.raises(FileNotFoundError):
            find_all_images_folders(tmpdir)
        result = find_all_images_folders(tmpdir, exts=(".tiff",))
        assert str(folder.resolve()) in result

def test_find_all_images_folders_nested_folders():
    with tempfile.TemporaryDirectory() as tmpdir:
        parent = Path(tmpdir) / "parent"
        child = parent / "child"
        child.mkdir(parents=True)
        (child / "pic.bmp").write_text("fake image content")
        result = find_all_images_folders(tmpdir)
        assert str(child.resolve()) in result
        assert str(parent.resolve()) not in result

def test_find_all_images_folders_link():
    with tempfile.TemporaryDirectory() as tmpdir:
        target = Path(tmpdir) / "target"
        target.mkdir()
        (target / "image.webp").write_text("fake image content")
        link = Path(tmpdir) / "link"
        try:
            link.symlink_to(target, target_is_directory=True)
        except OSError as e:
            if os.name == "nt" and getattr(e, "winerror", None) == 1314:
                pytest.skip("Symlinks require admin privileges on Windows")
            else:
                raise

        result = find_all_images_folders(tmpdir)
        assert str(target.resolve()) in result
        assert normalize_path(str(link.resolve())) not in result