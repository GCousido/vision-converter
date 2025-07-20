import glob
import json
import os
from typing import Optional

from pathlib import Path

from .bounding_box import CreateMLBoundingBox

from ..utils.file_utils import find_annotation_file
from .base import Annotation, DatasetFormat, FileFormat

class CreateMLAnnotation(Annotation[CreateMLBoundingBox]):
    """CreateML format annotation with label name and bounding box in CreateMLBoundingBox format.
    
    Attributes:
        label (str): label of the annotated object
        bbox (CreateMLBoundingBox): Inherited attribute - CreateML format bounding box
    """
    label: str

    def __init__(self, bbox: CreateMLBoundingBox, label: str) -> None:
        super().__init__(bbox)
        self.label = label


class CreateMLFile(FileFormat[CreateMLAnnotation]):
    """Represents a CreateML format image file with annotations in CreateMLAnnotation format.
    
    Attributes:
        filename (str): Inherited - Image filename
        annotations (list[CreateMLAnnotation]): Inherited - List of CreateML annotations
        width (Optional[int]): Image width in pixels (optional)
        height (Optional[int]): Image height in pixels (optional)
        depth (Optional[int]): Color channels (optional, typically 3 for RGB)
    """

    width: Optional[int]
    height: Optional[int]
    depth: Optional[int]

    def __init__(self, filename: str, annotations: list[CreateMLAnnotation], width: Optional[int] = None, height: Optional[int] = None, depth: Optional[int] = None) -> None:
        super().__init__(filename, annotations)
        self.width = width
        self.height = height
        self.depth = depth


class CreateMLFormat(DatasetFormat[CreateMLFile]):
    """CreateML format dataset container.
    
    Attributes:
        name (str): Inherited - Dataset name
        files (list[CreateMLFile]): Inherited - List of CreateML files
        folder_path (Optional[str]): Inherited - Dataset root path
        images_path_list (Optional[list[str]]): Inherited - List of images paths
    """

    def __init__(self, name: str, files: list[CreateMLFile], folder_path: Optional[str] = None, images_path_list: Optional[list[str]] = None) -> None:
        super().__init__(name,files, folder_path, images_path_list)

    @staticmethod
    def build(name: str, files: list[CreateMLFile], folder_path: Optional[str] = None, images_path_list: Optional[list[str]] = None) -> 'CreateMLFormat':
        return CreateMLFormat(name, files, folder_path, images_path_list)
    
    @staticmethod
    def create_files_from_jsondata(json_data) -> list[CreateMLFile]:
        files: list[CreateMLFile] = []
        for entry in json_data:
            filename = entry["image"]
            
            # Process annotations for this image
            annotations = []
            for ann in entry.get("annotations", []):
                bbox = CreateMLBoundingBox(
                    x_center=ann["coordinates"]["x"],
                    y_center=ann["coordinates"]["y"],
                    width=ann["coordinates"]["width"],
                    height=ann["coordinates"]["height"]
                )
                annotation = CreateMLAnnotation(bbox=bbox, label=ann["label"])
                annotations.append(annotation)

            create_ml_file = CreateMLFile(filename=filename, annotations=annotations)
            files.append(create_ml_file)
        
        return files
    
    @staticmethod
    def read_from_folder(path: str, copy_images: bool = False, copy_as_links: bool = False) -> 'CreateMLFormat':
        """Constructs CreateML dataset from standard folder structure.

        Expected structure:
        ``` 
        - dataset/ 
            ├── images/            # Contains image files  
            └── annotations.json   # File with all the annotations
        ```

        Args:
            path (str): Path to the dataset folder or annotation file
            copy_images (bool, default False): If True, loads and stores the image file paths in the dataset object; if False, image paths are not loaded.
            copy_as_links (bool, default False): If True, loads and stores the image file paths in the dataset object; if False, image paths are not loaded.
            
        Returns:
            CreateMLFormat: Dataset object
            
        Raises:
            FileNotFoundError: If required folders/files are missing
            Exception: If image-annotation name mismatch occurs
        """
        files: list[CreateMLFile] = []

        annotations_path = Path(find_annotation_file(path, "json"))

        images_path = annotations_path.parent / "images"
        if not Path(images_path).exists():
            raise FileNotFoundError(f"Folder {images_path} was not found")

        # 1. Read annotations
        try:
            with open(annotations_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in annotations.json: {e}")
        except Exception as e:
            raise FileNotFoundError(f"Error reading annotations.json: {e}")

        files = CreateMLFormat.create_files_from_jsondata(json_data)
        
        image_files = set()

        images_path_str = str(images_path)
        if os.path.exists(images_path_str) and os.path.isdir(images_path_str):
            # Image names
            image_patterns = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
            for pattern in image_patterns:
                for img_path in glob.glob(os.path.join(images_path, pattern)):
                    image_files.add(os.path.abspath(img_path))
        else:
            raise FileNotFoundError(f"Images directory {images_path} does not exist")

        image_basenames = {os.path.basename(path) for path in image_files}
        # 3. Validate image-annotation correspondence
        annotated_filenames = {entry["image"] for entry in json_data}
        for filename in annotated_filenames:
            if filename not in image_basenames:
                raise Exception(f'Dataset structure error: Image file {filename} not found in images folder')


        image_paths = list(image_files)
        return CreateMLFormat.build(
            name = "CreateMLDataset",
            files = files,
            folder_path = str(annotations_path.parent),
            images_path_list= image_paths if len(image_paths) > 0 and (copy_images or copy_as_links) else None
        )
    
    def save(self, folder: str, copy_images: bool = False, copy_as_links: bool = False) -> None:
        """Saves CreateML dataset to standard folder structure.
        
        ```
        {folder}/  
            ├── images/      # (Note: copies images if path exists)  
            └── annotations.json
        ```
        Args:
            folder: Output directory path
            copy_images (bool, default False): If True, copies image files to the output directory. If False, images are not copied.
            copy_as_links (bool, default False): If True, creates links to the original images in the output directory instead of copying them. If False, no links are created.
        """
        folder_path = Path(folder)

        # Create any folder if necesary
        folder_path.mkdir(parents=True, exist_ok=True)
        
        # Create subfolders images
        images_dir = folder_path / "images"
        images_dir.mkdir(exist_ok=True)

        # Create annotations.json
        annotations_json = []
        for file in self.files:
            annotations_list = []
            for ann in file.annotations:
                bbox: CreateMLBoundingBox = ann.geometry
                annotations_list.append({
                    "label": ann.label,
                    "coordinates": {
                        "x": bbox.x_center,
                        "y": bbox.y_center,
                        "width": bbox.width,
                        "height": bbox.height
                    }
                })
            
            annotations_json.append({
                "image": file.filename,
                "annotations": annotations_list
            })

        # Save annotations.json
        annotations_path = folder_path / "annotations.json"
        try:
            with open(annotations_path, 'w', encoding='utf-8') as f:
                json.dump(annotations_json, f, indent=4, ensure_ascii=False)
        except Exception as e:
            raise IOError(f"Error writing annotations.json: {e}")
        
        if copy_images or copy_as_links:
            self.handle_images(self.images_path_list, str(images_dir), copy_images, copy_as_links)