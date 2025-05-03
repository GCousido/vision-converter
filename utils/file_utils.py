from PIL import Image
from pathlib import Path

def get_image_info_from_file(image_path: str):
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


def get_image_path(folder_path: str, folder_route: str , filename: str):
    base_name = Path(filename).stem

    images_folder = Path(folder_path) / folder_route

    matches = list(images_folder.glob(f"{base_name}.*"))
    
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    for match in matches:
        if match.suffix.lower() in image_exts:
            return str(match)
        
    return None
