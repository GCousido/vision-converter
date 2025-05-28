from PIL import Image
from pathlib import Path

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
