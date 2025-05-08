from dataclasses import dataclass
from pathlib import Path
import json
from typing import Optional

from .base import Annotation, BoundingBox, DatasetFormat, FileFormat

class CocoBoundingBox(BoundingBox):
    x_min: float
    y_min: float
    width: float
    height: float

    def __init__(self, x_min: float, y_min: float, width: float, height: float) -> None:
        self.x_min = x_min
        self.y_min = y_min
        self.width = width
        self.height = height

    def getBoundingBox(self):
        return [self.x_min, self.y_min, self.width, self.height]

@dataclass
class RLESegmentation:
    size: list[int] # [height, width]
    counts: str

    def __post_init__(self):
        # Validate that size have only 2 elements
        if len(self.size) != 2 or not all(isinstance(x, int) for x in self.size):
            raise ValueError("'size' has to be a list of 2 ints [height, width].")

class CocoLabel(Annotation[CocoBoundingBox]):
    id: int
    image_id: int
    category_id: int
    segmentation: Optional[list[list[float]] | RLESegmentation]
    area: Optional[float]
    iscrowd: Optional[bool]

    def __init__(self, bbox: CocoBoundingBox, id: int, image_id: int, category_id: int, segmentation: Optional[list[list[float]] | RLESegmentation] = None, area: Optional[float] = None, iscrowd: Optional[bool] = None) -> None:
        super().__init__(bbox)
        self.id = id
        self.image_id = image_id
        self.category_id = category_id
        self.segmentation = segmentation
        self.area = area
        self.iscrowd = iscrowd


@dataclass
class Info:
    description: str
    url: str
    version: str
    year: int
    contributor: str
    date_created: str

@dataclass
class License:
    id: int
    name: str
    url: str

@dataclass
class CocoImage:
    id: int
    width: int
    height: int
    file_name: str
    date_captured: str
    flickr_url: Optional[str] = None
    coco_url: Optional[str] = None
    license: Optional[int] = None

@dataclass
class Category:
    id: int
    name: str
    supercategory: Optional[str] = None


class CocoFile(FileFormat[CocoLabel]):
    info: Optional[Info]
    licenses: Optional[list[License]]
    images: list[CocoImage]
    categories: list[Category]

    def __init__(self, filename: str, annotations: list[CocoLabel], images: list[CocoImage], categories: list[Category], info: Optional[Info] = None, licenses: Optional[list[License]] = None) -> None:
        super().__init__(filename, annotations)
        self.info = info
        self.licenses = licenses
        self.images = images
        self.categories = categories


class CocoFormat(DatasetFormat[CocoFile]):

    def __init__(self, name: str, files: list[CocoFile], folder_path: Optional[str] = None) -> None:
        super().__init__(name, files, folder_path)

    @staticmethod
    def build(name: str, files: list[CocoFile], folder_path: Optional[str] = None) -> 'CocoFormat':
        return CocoFormat(name, files, folder_path)
    
    @staticmethod
    def create_coco_file_from_json(coco_data, name: str) -> CocoFile:

        # Extract info
        info_data = coco_data.get('info', {})
        info = Info(
            description=info_data.get('description', ''),
            url=info_data.get('url', ''),
            version=info_data.get('version', ''),
            year=info_data.get('year', 0),
            contributor=info_data.get('contributor', ''),
            date_created=info_data.get('date_created', '')
        )

        # Extract licenses
        licenses = []
        for license_data in coco_data.get('licenses', []):
            licenses.append(License(
                id=license_data.get('id', 0),
                name=license_data.get('name', ''),
                url=license_data.get('url', '')
            ))
        
        # Extract images
        images = []
        for image_data in coco_data.get('images', []):
            images.append(CocoImage(
                id=image_data.get('id', 0),
                width=image_data.get('width', 0),
                height=image_data.get('height', 0),
                file_name=image_data.get('file_name', ''),
                license=image_data.get('license', None),
                flickr_url=image_data.get('flickr_url', None),
                coco_url=image_data.get('coco_url', None),
                date_captured=image_data.get('date_captured', '')
            ))
        
        # Extract categories
        categories = []
        for category_data in coco_data.get('categories', []):
            categories.append(Category(
                id=category_data.get('id', 0),
                name=category_data.get('name', ''),
                supercategory=category_data.get('supercategory', None)
            ))
        
        # Extract annotations
        annotations = []
        for ann_data in coco_data.get('annotations', []):
            bbox_data = ann_data.get('bbox', [0, 0, 0, 0])
            bbox = CocoBoundingBox(
                x_min=bbox_data[0] if len(bbox_data) > 0 else 0,
                y_min=bbox_data[1] if len(bbox_data) > 1 else 0,
                width=bbox_data[2] if len(bbox_data) > 2 else 0,
                height=bbox_data[3] if len(bbox_data) > 3 else 0
            )
            
            # Procesing segmentation data
            segmentation_data = ann_data.get('segmentation', [])
            if isinstance(segmentation_data, dict) and 'counts' in segmentation_data:
                # RLE: dict
                segmentation = RLESegmentation(
                    size=segmentation_data.get('size', [0, 0]),
                    counts=segmentation_data.get('counts', '')
                )
            elif isinstance(segmentation_data, list):
                # Polygon: list
                segmentation = segmentation_data
            else:
                segmentation = []

            annotations.append(CocoLabel(
                bbox=bbox,
                id=ann_data.get('id', 0),
                image_id=ann_data.get('image_id', 0),
                category_id=ann_data.get('category_id', 0),
                segmentation=segmentation,
                area=ann_data.get('area', 0.0),
                iscrowd=bool(ann_data.get('iscrowd', 0))
            ))

        return CocoFile(
                filename=name,
                annotations=annotations,
                info=info,
                licenses=licenses,
                images=images,
                categories=categories
            )

    @staticmethod
    def read_from_folder(folder_path: str) -> 'CocoFormat':
        """
        Create a dataset in COCO format from folder.

        A standar COCO format consist of:
        - A images folder
        - One or more JSON files with annotations in an 'annotations' folder

        Args:
            folder_path (str): Path to the folder

        Returns:
            CocoFormat: Object with the COCO dataset
        """

        folder = Path(folder_path)

        if not folder.exists():
            raise FileNotFoundError(f"Folder {folder_path} was not found")

        json_files = list(folder.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"JSON files were not found in {folder}")

        coco_files = []

        for json_file in json_files:
            with open(json_file, 'r') as f:
                coco_data = json.load(f)

            coco_files.append(CocoFormat.create_coco_file_from_json(coco_data, json_file.name))

        # Construir y devolver un CocoFormat
        return CocoFormat.build(
            name=folder.name,
            files=coco_files,
            folder_path=folder_path
        )