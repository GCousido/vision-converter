import glob
import json
import os
from typing import Optional

from pathlib import Path

from ..utils.file_utils import get_image_path
from .base import Annotation, BoundingBox, DatasetFormat, FileFormat

class CreateMLBoundingBox(BoundingBox):
    """CreateML format bounding box implementation using absolute coordinates.
    
    Attributes:
        x_center (float): absolute x-coordinate of center
        y_center (float): absolute y-coordinate of center
        width (float): absolute width
        height (float): absolute height
    """
    x_center: float
    y_center: float
    width: float
    height: float

    def __init__(self, x_center: float, y_center: float, width: float, height: float) -> None:
        self.x_center = x_center
        self.y_center = y_center
        self.width = width
        self.height = height

    def getBoundingBox(self):
        """Returns CreateML format coordinates as [x_center, y_center, width, height]."""
        return [self.x_center, self.y_center, self.width, self.height]


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
    """

    def __init__(self, name: str, files: list[CreateMLFile], folder_path: Optional[str] = None) -> None:
        super().__init__(name,files, folder_path)

    @staticmethod
    def build(name: str, files: list[CreateMLFile], folder_path: Optional[str] = None) -> 'CreateMLFormat':
        return CreateMLFormat(name, files, folder_path)
    
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
    def read_from_folder(folder_path: str) -> 'CreateMLFormat':
        """Constructs CreateML dataset from standard folder structure.

        Expected structure:
        ``` 
        - {folder_path}/ 
            ├── images/             # Contains image files  
            └── annotations.json    # File with all the annotations
        ```
        Args:
            folder_path (str): Root directory of CreateML dataset
            
        Returns:
            CreateMLFormat: Dataset object
            
        Raises:
            FileNotFoundError: If required folders/files are missing
            Exception: If image-annotation name mismatch occurs
        """
        files: list[CreateMLFile] = []

        if not Path(folder_path).exists():
            raise FileNotFoundError(f"Folder {folder_path} was not found")


        images_path = Path(folder_path) / "images"
        if not Path(images_path).exists():
            raise FileNotFoundError(f"Folder {images_path} was not found")

        # 1. Read annotations
        annotations_path = Path(folder_path) / "annotations.json"
        if not annotations_path.exists():
            raise FileNotFoundError(f"File 'annotations.json' was not found in {folder_path}")
        
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
                    image_files.add(os.path.basename(img_path))
        else:
            raise FileNotFoundError(f"Images directory {images_path} does not exist")


        # 3. Validate image-annotation correspondence
        annotated_filenames = {entry["image"] for entry in json_data}
        for filename in annotated_filenames:
            if filename not in image_files:
                raise Exception(f'Dataset structure error: Image file {filename} not found in images folder')

        return CreateMLFormat.build(
            name=Path(folder_path).name,
            files=files,
            folder_path=folder_path
        )
    
    def save(self, folder: str) -> None:
        """Saves CreateML dataset to standard folder structure.
        
        ```
        {folder}/  
            ├── images/      # (Note: copies images if path exists)  
            └── annotations.json
        ```
        Args:
            folder: Output directory path
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