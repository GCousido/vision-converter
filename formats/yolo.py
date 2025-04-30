from pathlib import Path
from typing import Optional

from .base import Annotation, BoundingBox, DatasetFormat, FileFormat

class YoloBoundingBox(BoundingBox):
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
        return [self.x_center, self.y_center, self.width, self.height]


class YoloAnnotation(Annotation[YoloBoundingBox]):
    id_class: int

    def __init__(self, bbox: YoloBoundingBox, id_class: int) -> None:
        super().__init__(bbox)
        self.id_class = id_class

class YoloFile(FileFormat[YoloAnnotation]):

    def __init__(self, filename: str, annotations: list[YoloAnnotation]) -> None:
        super().__init__(filename, annotations)


class YoloFormat(DatasetFormat[YoloFile]):
    class_labels: list[str]

    def __init__(self, name: str, files: list[YoloFile], class_labels: list[str], folder_path: Optional[str] = None, ) -> None:
        super().__init__(name,files, folder_path)
        self.class_labels = class_labels

    def addClassLabel(self, class_label: str) -> None:
        self.class_labels.append(class_label)

    def getClassLabels(self) -> list[str]:
        return self.class_labels
    
    @staticmethod
    def build(name: str, files: list[YoloFile], class_labels: list[str], folder_path: Optional[str] = None) -> 'YoloFormat':
        return YoloFormat(name, files, class_labels, folder_path)

    @staticmethod
    def read_from_folder(folder_path: str) -> 'YoloFormat':
        """
        Create a dataset in YOLO format from folder.

        A standar YOLO format consist of:
        - A images folder
        - A labels folder with text files with the annotations

        Args:
            folder_path (str): Path to the folder

        Returns:
            YoloFormat: Object with the YOLO dataset
        """
        files = []
        class_labels = []

        if not Path(folder_path).exists():
            raise FileNotFoundError(f"Folder {folder_path} was not found")

        labels_dir = Path(folder_path) / "labels"

        if not labels_dir.exists():
            raise FileNotFoundError(f"Folder 'labels' was not found in {folder_path}")
        
        # 1. Read classes
        classes_file = labels_dir / "classes.txt"
        if classes_file.exists():
            with open(classes_file, 'r') as f:
                class_labels = [line.strip() for line in f.readlines()]
        else:
            raise FileNotFoundError(f"File 'classes.txt' was not found in {labels_dir}")
        
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
            
            files.append(YoloFile(ann_file.name, annotations))
        
        return YoloFormat.build(
            name=Path(folder_path).name,
            files=files,
            folder_path=folder_path,
            class_labels=class_labels
        )