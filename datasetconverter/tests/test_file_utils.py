import pytest
from PIL import UnidentifiedImageError
from pathlib import Path
from unittest.mock import patch, PropertyMock

from datasetconverter.utils.file_utils import get_image_info_from_file, get_image_path, estimate_file_size

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
