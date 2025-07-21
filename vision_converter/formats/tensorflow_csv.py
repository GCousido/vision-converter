from pathlib import Path
from typing import Optional
import csv
import tensorflow as tf

from vision_converter.utils.file_utils import find_all_images_folders, find_annotation_file

from .bounding_box import CornerAbsoluteBoundingBox

from .base import Annotation, DatasetFormat, FileFormat

class TensorflowCsvAnnotation(Annotation[CornerAbsoluteBoundingBox]):
    """TensorFlow CSV format annotation with class name and bounding box in CornerAbsoluteBoundingBox format.
    
    Attributes:
        class_name (str): String class name corresponding to class_labels
        bbox (CornerAbsoluteBoundingBox): Inherited attribute - Pascal VOC format bounding box
    """
    class_name: str

    def __init__(self, bbox: CornerAbsoluteBoundingBox, class_name: str) -> None:
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
        images_path_list (Optional[list[str]]): Inherited - List of images paths
    """

    def __init__(self, name: str, files: list[TensorflowCsvFile], folder_path: Optional[str] = None, images_path_list: Optional[list[str]] = None) -> None:
        super().__init__(name, files, folder_path, images_path_list)

    def get_unique_classes(self) -> set[str]:
        """Get all unique class names in the dataset."""
        classes = set()
        for file in self.files:
            for ann in file.annotations:
                classes.add(ann.class_name)
        return classes

    @staticmethod
    def build(name: str, files: list[TensorflowCsvFile], folder_path: Optional[str] = None, images_path_list: Optional[list[str]] = None) -> 'TensorflowCsvFormat':
        return TensorflowCsvFormat(name, files, folder_path, images_path_list)

    @staticmethod
    def read_from_folder(path: str, copy_images: bool = False, copy_as_links: bool = False, tfrecord: bool = False) -> 'TensorflowCsvFormat':
        """Constructs TensorFlow CSV dataset from CSV file.

        Expected CSV format:
        ```
        csv_path
            filename,width,height,class,xmin,ymin,xmax,ymax
            image1.jpg,800,600,person,100,150,200,300
            image1.jpg,800,600,car,300,200,450,350
        ```
        
        Args:
            path (str): Path to the dataset folder or annotation file
            copy_images (bool, default False): If True, loads and stores the image file paths in the dataset object; if False, image paths are not loaded.
            copy_as_links (bool, default False): If True, loads and stores the image file paths in the dataset object; if False, image paths are not loaded.
            tfrecord (bool, default False): If True, loads and stores the image file paths in the dataset object; if False, image paths are not loaded.
            
        Returns:
            TensorflowCsvFormat: Dataset object
            
        Raises:
            FileNotFoundError: If CSV file is missing
            KeyError: If required CSV columns are missing
        """

        annotations_path = Path(find_annotation_file(path, "csv"))

        files_dict = {}
        
        with open(annotations_path, newline='') as csvfile:
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

                bbox = CornerAbsoluteBoundingBox(xmin, ymin, xmax, ymax)
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

        # Save images path
        image_paths = []
        if copy_images or copy_as_links or tfrecord:
            # Search for images folders
            list_images_dir = find_all_images_folders(annotations_path.parent) 
            for images_dir in list_images_dir:
                image_paths += TensorflowCsvFormat.get_image_paths(images_dir)

        return TensorflowCsvFormat.build(
            name = Path(annotations_path).stem,
            files = files,
            folder_path = str(Path(annotations_path).parent),
            images_path_list = image_paths if len(image_paths) > 0 else None
        )

    def save(self, folder_path: str, copy_images: bool = False, copy_as_links: bool = False, tfrecord: bool = False) -> None:
        """Saves TensorFlow CSV dataset to CSV file.
        
        Output format:
        ```
        folder_path/
                |-- images/             # image files
                └── tensorflow.csv      # annotations file
                        filename,width,height,class,xmin,ymin,xmax,ymax
                        ->
        ```
        
        Args:
            csv_path (str): Output CSV file path
            copy_images (bool, default False): If True, copies image files to the output directory. If False, images are not copied.
            copy_as_links (bool, default False): If True, creates links to the original images in the output directory instead of copying them. If False, no links are created.
            tfrecord (bool, default False): If True, creates the TFRecord binary file with the information of the dataset.
        """
        # Ensure output directory exists
        Path(folder_path).mkdir(parents=True, exist_ok=True)

        images_dir = Path(folder_path) / "images"
        images_dir.mkdir(exist_ok=True)

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

        if copy_images or copy_as_links:
            self.handle_images(self.images_path_list, str(images_dir), copy_images, copy_as_links)

        if tfrecord:
            _generate_tfrecord(self, folder_path)


def _generate_tfrecord(dataset: TensorflowCsvFormat, folder_path: str):
    record_file = str(Path(folder_path) / "dataset.tfrecord")
    with tf.io.TFRecordWriter(record_file) as writer:
        for file in dataset.files:
            img_path = _find_image_path_in_list(file.filename, dataset.images_path_list)
            if img_path:
                with open(img_path, 'rb') as img_f:
                    img_bytes = img_f.read()
            else:
                raise FileNotFoundError(f"Image {file.filename} not found")
            
            for ann in file.annotations:
                feature = {
                    'image/encoded': _bytes_feature(img_bytes),
                    'image/filename': _bytes_feature(file.filename.encode('utf-8')),
                    'image/width': _int64_feature(file.width),
                    'image/height': _int64_feature(file.height),
                    'image/object/class/text': _bytes_feature(ann.class_name.encode('utf-8')),
                    'image/object/bbox/xmin': _float_feature(ann.geometry.x_min),
                    'image/object/bbox/ymin': _float_feature(ann.geometry.y_min),
                    'image/object/bbox/xmax': _float_feature(ann.geometry.x_max),
                    'image/object/bbox/ymax': _float_feature(ann.geometry.y_max)
                }
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _find_image_path_in_list(filename, images_path_list) -> str | None:
    for path in images_path_list:
        if path.endswith(filename):
            return str(path)
    return None