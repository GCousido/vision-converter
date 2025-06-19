from pathlib import Path
from typing import Optional

from ..utils.file_utils import get_image_path

from .base import Annotation, BoundingBox, DatasetFormat, FileFormat

class YoloBoundingBox(BoundingBox):
    """YOLO format bounding box implementation using normalized coordinates.
    
    Attributes:
        x_center (float): Normalized x-coordinate of center (0.0-1.0)
        y_center (float): Normalized y-coordinate of center (0.0-1.0)
        width (float): Normalized width (0.0-1.0)
        height (float): Normalized height (0.0-1.0)
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
        """Returns YOLO format coordinates as [x_center, y_center, width, height]."""
        return [self.x_center, self.y_center, self.width, self.height]


class YoloAnnotation(Annotation[YoloBoundingBox]):
    """YOLO format annotation with class ID and bounding box in YoloBoundingBox format.
    
    Attributes:
        id_class (int): Numeric class ID corresponding to class_labels
        bbox (YoloBoundingBox): Inherited attribute - YOLO format bounding box
    """
    id_class: int

    def __init__(self, bbox: YoloBoundingBox, id_class: int) -> None:
        super().__init__(bbox)
        self.id_class = id_class

class YoloFile(FileFormat[YoloAnnotation]):
    """Represents a YOLO format image file with annotations in YoloAnnotation format.
    
    Attributes:
        filename (str): Inherited - Image filename
        annotations (list[YoloAnnotation]): Inherited - List of YOLO annotations
        width (Optional[int]): Image width in pixels (optional)
        height (Optional[int]): Image height in pixels (optional)
        depth (Optional[int]): Color channels (optional, typically 3 for RGB)
    """

    width: Optional[int]
    height: Optional[int]
    depth: Optional[int]

    def __init__(self, filename: str, annotations: list[YoloAnnotation], width: Optional[int] = None, height: Optional[int] = None, depth: Optional[int] = None) -> None:
        super().__init__(filename, annotations)
        self.width = width
        self.height = height
        self.depth = depth


class YoloFormat(DatasetFormat[YoloFile]):
    """YOLO format dataset container.
    
    Attributes:
        class_labels (dict[int, str]): Mapping of class IDs to names
        name (str): Inherited - Dataset name
        files (list[YoloFile]): Inherited - List of YOLO files
        folder_path (Optional[str]): Inherited - Dataset root path
        images_path_list (Optional[list[str]]): Inherited - List of images paths
    """
    class_labels: dict[int, str]

    def __init__(self, name: str, files: list[YoloFile], class_labels: dict[int, str], folder_path: Optional[str] = None, images_path_list: Optional[list[str]] = None) -> None:
        super().__init__(name,files, folder_path, images_path_list)
        self.class_labels = class_labels

    
    @staticmethod
    def build(name: str, files: list[YoloFile], class_labels: dict[int, str], folder_path: Optional[str] = None, images_path_list: Optional[list[str]] = None) -> 'YoloFormat':
        return YoloFormat(name, files, class_labels, folder_path, images_path_list)

    @staticmethod
    def read_from_folder(folder_path: str, copy_images: bool = False, copy_as_links: bool = False) -> 'YoloFormat':
        """Constructs YOLO dataset from standard folder structure.

        Expected structure:
        ``` 
        - {folder_path}/  
            ├── images/      # Contains image files  
            └── labels/      # Contains .txt annotations and classes.txt
        ```
        Args:
            folder_path (str): Root directory of YOLO dataset
            copy_images (bool, default False): If True, loads and stores the image file paths in the dataset object; if False, image paths are not loaded.
            copy_as_links (bool, default False): If True, loads and stores the image file paths in the dataset object; if False, image paths are not loaded.
            
        Returns:
            YoloFormat: Dataset object
            
        Raises:
            FileNotFoundError: If required folders/files are missing
            Exception: If image-annotation name mismatch occurs
        """
        files = []
        class_labels = {}

        if not Path(folder_path).exists():
            raise FileNotFoundError(f"Folder {folder_path} was not found")

        labels_dir = Path(folder_path) / "labels"

        if not labels_dir.exists():
            raise FileNotFoundError(f"Folder 'labels' was not found in {folder_path}")
        
        images_dir = Path(folder_path) / "images"

        if not images_dir.exists():
            raise FileNotFoundError(f"Folder 'images' was not found in {folder_path}")
        
        image_paths = []

        # 1. Read classes
        if (labels_dir / "classes.txt").exists():
            classes_file = labels_dir / "classes.txt"
        elif (Path(folder_path) / "classes.txt").exists():
            classes_file = Path(folder_path) / "classes.txt"
        else:
            raise FileNotFoundError(f"File 'classes.txt' was not found in {folder_path}")

        with open(classes_file, 'r') as f:
            class_labels = {i: line.strip() for i, line in enumerate(f.readlines())}

        # 2. Read annotations (archivos .txt)
        for ann_file in labels_dir.glob("*.txt"):
            if ann_file.name == "classes.txt":
                continue
                
            annotations = []
            with open(ann_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        bbox = YoloBoundingBox(
                            x_center=float(parts[1]),
                            y_center=float(parts[2]),
                            width=float(parts[3]),
                            height=float(parts[4])
                        )
                        annotations.append(YoloAnnotation(bbox, class_id))

            image_path = get_image_path(folder_path, "images", ann_file.name)

            if not image_path:
                raise Exception("Dataset structure error in the YOLO Dataset, annotations file must have the same name as the image")
            
            if copy_images or copy_as_links:
                image_paths.append(image_path)

            files.append(YoloFile(Path(image_path).name, annotations))

        return YoloFormat.build(
            name=Path(folder_path).name,
            files=files,
            folder_path=folder_path,
            class_labels=class_labels,
            images_path_list=image_paths if len(image_paths) > 0 else None
        )


    def save(self, folder: str, copy_images: bool = False, copy_as_links: bool = False) -> None:
        """
        Saves YOLO dataset to standard folder structure.

        Structure:
            {folder}/  
                ├── images/
                └── labels/  
                    ├── classes.txt
                    └── *.txt    # Annotation files  

        Args:
            folder (str): Output directory path.
            copy_images (bool, default False): If True, copies image files to the output directory. If False, images are not copied.
            copy_as_links (bool, default False): If True, creates links to the original images in the output directory instead of copying them. If False, no links are created.

        Returns:
            None

        Raises:
            Exception: If image paths are not found or an error occurs during copy/symlink.
        """
        folder_path = Path(folder)
        
        # Create any folder if necesary
        folder_path.mkdir(parents=True, exist_ok=True)
        
        # Create subfolders labels and images
        labels_dir = folder_path / "labels"
        images_dir = folder_path / "images"
        labels_dir.mkdir(exist_ok=True)
        images_dir.mkdir(exist_ok=True)
        
        # Save classes.txt
        classes_file = labels_dir / "classes.txt"
        with open(classes_file, "w") as f:
            for class_id in sorted(self.class_labels):
                f.write(f"{self.class_labels[class_id]}\n")
        
        # Save annotations in labels folder
        for yolo_file in self.files:
            filename = Path(yolo_file.filename).stem + ".txt"
            file_path = labels_dir / filename
            with open(file_path, "w") as f:
                for ann in yolo_file.annotations:
                    bbox = ann.geometry
                    line = f"{ann.id_class} {bbox.x_center} {bbox.y_center} {bbox.width} {bbox.height}\n"
                    f.write(line)

        if copy_images or copy_as_links:
            self.handle_images(self.images_path_list, str(images_dir), copy_images, copy_as_links)