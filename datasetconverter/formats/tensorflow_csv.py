from pathlib import Path
from typing import Optional
import csv

from .pascal_voc import PascalVocBoundingBox

from .base import Annotation, DatasetFormat, FileFormat

class TensorflowCsvAnnotation(Annotation[PascalVocBoundingBox]):
    """TensorFlow CSV format annotation with class name and bounding box in PascalVocBoundingBox format.
    
    Attributes:
        class_name (str): String class name corresponding to class_labels
        bbox (PascalVocBoundingBox): Inherited attribute - Pascal VOC format bounding box
    """
    class_name: str

    def __init__(self, bbox: PascalVocBoundingBox, class_name: str) -> None:
        super().__init__(bbox)
        self.class_name = class_name


class TensorflowCsvFile(FileFormat[TensorflowCsvAnnotation]):
    """Represents a TensorFlow CSV format image file with annotations in TensorflowCsvAnnotation format.
    
    Attributes:
        filename (str): Inherited - Image filename
        annotations (list[TensorflowCsvAnnotation]): Inherited - List of TensorFlow CSV annotations
        width (int): Image width in pixels
        height (int): Image height in pixels
    """
    width: int
    height: int

    def __init__(self, filename: str, annotations: list[TensorflowCsvAnnotation], width: int, height: int) -> None:
        super().__init__(filename, annotations)
        self.width = width
        self.height = height


class TensorflowCsvFormat(DatasetFormat[TensorflowCsvFile]):
    """TensorFlow CSV format dataset container.
    
    Attributes:
        name (str): Inherited - Dataset name
        files (list[TensorflowCsvFile]): Inherited - List of TensorFlow CSV files
        folder_path (Optional[str]): Inherited - Dataset root path
    """

    def __init__(self, name: str, files: list[TensorflowCsvFile], folder_path: Optional[str] = None) -> None:
        super().__init__(name, files, folder_path)

    def get_unique_classes(self) -> set[str]:
        """Get all unique class names in the dataset."""
        classes = set()
        for file in self.files:
            for ann in file.annotations:
                classes.add(ann.class_name)
        return classes

    @staticmethod
    def build(name: str, files: list[TensorflowCsvFile], folder_path: Optional[str] = None) -> 'TensorflowCsvFormat':
        return TensorflowCsvFormat(name, files, folder_path)

    @staticmethod
    def read_from_folder(csv_path: str) -> 'TensorflowCsvFormat':
        """Constructs TensorFlow CSV dataset from CSV file.

        Expected CSV format:
        ```
        csv_path
            filename,width,height,class,xmin,ymin,xmax,ymax
            image1.jpg,800,600,person,100,150,200,300
            image1.jpg,800,600,car,300,200,450,350
        ```
        
        Args:
            csv_path (str): Path to CSV annotation file
            
        Returns:
            TensorflowCsvFormat: Dataset object
            
        Raises:
            FileNotFoundError: If CSV file is missing
            KeyError: If required CSV columns are missing
        """
        if not Path(csv_path).exists():
            raise FileNotFoundError(f"CSV file {csv_path} was not found")

        files_dict = {}
        
        with open(csv_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            
            # Validate required columns
            required_columns = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']

            fieldnames = reader.fieldnames
            if fieldnames is None:
                raise ValueError("CSV file has no headers")
            if not all(col in fieldnames for col in required_columns):
                raise KeyError(f"CSV must contain columns: {required_columns}")
            
            for row in reader:
                filename = row['filename']
                width = int(row['width'])
                height = int(row['height'])
                class_name = row['class']
                xmin = int(row['xmin'])
                ymin = int(row['ymin'])
                xmax = int(row['xmax'])
                ymax = int(row['ymax'])

                bbox = PascalVocBoundingBox(xmin, ymin, xmax, ymax)
                annotation = TensorflowCsvAnnotation(bbox, class_name)
                
                # Store file info with first occurrence dimensions
                if filename not in files_dict:
                    files_dict[filename] = {
                        'annotations': [],
                        'width': width,
                        'height': height
                    }
                
                files_dict[filename]['annotations'].append(annotation)

        files = []
        for filename, file_data in files_dict.items():
            files.append(TensorflowCsvFile(
                filename=filename,
                annotations=file_data['annotations'],
                width=file_data['width'],
                height=file_data['height']
            ))

        return TensorflowCsvFormat.build(
            name = Path(csv_path).stem,
            files = files,
            folder_path = str(Path(csv_path).parent)
        )

    def save(self, folder_path: str) -> None:
        """Saves TensorFlow CSV dataset to CSV file.
        
        Output format:
        ```
        folder_path/
                |-- images/             # image files (not written)
                └── tensorflow.csv      # annotations file
                        filename,width,height,class,xmin,ymin,xmax,ymax
                        ->
        ```
        
        Args:
            csv_path (str): Output CSV file path
        """
        # Ensure output directory exists
        Path(folder_path).mkdir(parents=True, exist_ok=True)

        if not Path(folder_path + "/images").exists():
            Path(folder_path + "/images").mkdir()

        csv_path = Path(folder_path) / "tensorflow.csv"
        
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for file in self.files:
                for ann in file.annotations:
                    writer.writerow({
                        'filename': file.filename,
                        'width': file.width,
                        'height': file.height,
                        'class': ann.class_name,
                        'xmin': ann.geometry.x_min,
                        'ymin': ann.geometry.y_min,
                        'xmax': ann.geometry.x_max,
                        'ymax': ann.geometry.y_max
                    })
