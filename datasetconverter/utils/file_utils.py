from PIL import Image
from pathlib import Path


def find_annotation_file(directory_or_file: str, extension: str) -> str:
    """Locates the unique annotation file with specified extension in a directory structure.
    
    Args:
        directory_or_file (str): Path to either a directory (searched recursively) 
            or a direct file candidate.
        extension (str): File extension to search for (case-insensitive)

    Returns:
        str: Full path to the unique matching file

    Raises:
        FileNotFoundError: If input path doesn't exist or no files found
        ValueError: For multiple files or extension mismatch
    """
    path = Path(directory_or_file)
    if not path.exists():
        raise FileNotFoundError(f"Invalid path: {directory_or_file}")

    # Normalize extension
    extension = f".{extension.strip('. ').lower()}"

    # Validate direct path to the annotation file
    if path.is_file():
        if path.suffix.lower() == extension:
            return str(path)
        else:
            raise ValueError(f"File extension mismatch. Expected {extension}")

    # Find files in the directory
    matched_files = [
        str(p) for p in path.rglob(f"*{extension}") 
        if p.suffix.lower() == extension and p.is_file()
    ]
    
    if len(matched_files) == 0:
        raise FileNotFoundError(f"Annotations file not found in the folder: {directory_or_file}")

    if len(matched_files) > 1:
        raise ValueError(
            f"Found {len(matched_files)} '{extension}' files. "
            f"Please check your dataset structure. First 3:\n" +
            "\n".join(f"• {f}" for f in matched_files[:3])
        )

    return matched_files[0]


def get_image_info_from_file(image_path: str):
    """Retrieves image dimensions and color depth from an image file.

    Args:
        image_path (str): Full path to the image file.

    Returns:
        tuple: (width, height, depth) where:
            width (int): Image width in pixels.
            height (int): Image height in pixels.
            depth (int): Number of color channels.

    Raises:
        FileNotFoundError: If the file does not exist.
        PIL.UnidentifiedImageError: If the file is not a valid image.

    Note:
        Maps PIL image modes to color depth:
        - '1', 'L', 'P', 'I', 'F' → 1 channel
        - 'RGB', 'YCbCr' → 3 channels
        - 'RGBA', 'CMYK' → 4 channels
        - Unknown modes default to 3 channels.
    """
    with Image.open(image_path) as img:
        width, height = img.size
        mode = img.mode

        mode_to_depth = {
            '1': 1,
            'L': 1,
            'P': 1,
            'RGB': 3,
            'RGBA': 4,
            'CMYK': 4,
            'YCbCr': 3,
            'I': 1,
            'F': 1
        }
        depth = mode_to_depth.get(mode, 3) 

        return width, height, depth


def get_image_path(folder_path: str, image_folder_route: str , filename: str):
    """Returns the path of the image corresponding to the name of an annotation file 
    (annotation files have the same name as the image they refer to, e.g.: image1.png -> image1.txt)

    Args:
        folder_path (str): Base path.
        image_folder_route (str): Subfolder within folder_path where the images are stored.
        filename (str): File name

    Returns:
        str | None: Full path if it exists, None if not found.

    Note:
        - Supported extensions: .jpg, .jpeg, .png, .bmp, .webp
        - The match is by exact base name, without extension.
        - Extension comparison is case-insensitive.
    """
    base_name = Path(filename).stem
    images_folder = Path(folder_path) / image_folder_route
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    for file in images_folder.iterdir():
        if file.is_file() and file.stem == base_name and file.suffix.lower() in image_exts:
            return str(file.resolve())

    return None

def estimate_file_size(width: int, height: int, depth: int, extension: str) -> int:
    """Estimate file size based on image dimensions and format.
    
    Calculates an approximate file size by applying typical compression factors
    for different image formats to the uncompressed pixel data size.
    
    Args:
        width (int): Image width in pixels
        height (int): Image height in pixels  
        depth (int): Color depth (typically 3 for RGB, 1 for grayscale)
        extension (str): File extension including dot (e.g., '.jpg', '.png')
        
    Returns:
        int: Estimated file size in bytes
        
    Note:
        Compression factors are approximations based on typical usage:
        - JPEG: High compression (10% of uncompressed)
        - PNG: Lossless compression (30% of uncompressed)
        - BMP: No compression (100% of uncompressed)
        - TIFF: Moderate compression (50% of uncompressed)
        - WebP: Very high compression (8% of uncompressed)
    """
    uncompressed_size = width * height * depth
    
    # Typical compression factors by image format
    compression_factors = {
        '.jpg': 0.1,   # JPEG high compression
        '.jpeg': 0.1,
        '.png': 0.3,   # PNG lossless compression
        '.bmp': 1.0,   # No compression
        '.tiff': 0.5,  # TIFF moderate compression
        '.webp': 0.08  # WebP very high compression
    }
    
    factor = compression_factors.get(extension.lower(), 0.2)  # Default 20% for unknown formats
    return int(uncompressed_size * factor)