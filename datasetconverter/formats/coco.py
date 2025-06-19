from dataclasses import dataclass
from pathlib import Path
import json
from typing import Optional

from datasetconverter.utils.file_utils import find_all_images_folders, find_annotation_file

from .base import Annotation, BoundingBox, DatasetFormat, FileFormat

class CocoBoundingBox(BoundingBox):
    """COCO format bounding box using absolute pixel coordinates.

    Attributes:
        x_min (float): Minimum x (left).
        y_min (float): Minimum y (top).
        width (float): Box width in pixels.
        height (float): Box height in pixels.
    """
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
        """Returns COCO format coordinates as [x_min, y_min, width, height]."""
        return [self.x_min, self.y_min, self.width, self.height]

@dataclass
class RLESegmentation:
    """Run-Length Encoding (RLE) segmentation for COCO.

    Attributes:
        size (list[int]): [height, width] of the mask.
        counts (str): RLE-encoded mask.
    """
    size: list[int] # [height, width]
    counts: str

    def __post_init__(self):
        """Validates that size is a list of two integers.

        Raises:
            ValueError: If size is not [height, width].
        """
        if len(self.size) != 2 or not all(isinstance(x, int) for x in self.size):
            raise ValueError("'size' has to be a list of 2 ints [height, width].")

class CocoLabel(Annotation[CocoBoundingBox]):
    """COCO annotation label for an object instance.

    Attributes:
        id (int): Annotation ID.
        image_id (int): ID of the image this annotation belongs to.
        category_id (int): Category ID.
        segmentation (Optional[list[list[float]] | RLESegmentation]): Polygon or RLE segmentation.
        area (Optional[float]): Area of the object.
        iscrowd (Optional[bool]): Whether the annotation is a crowd region.
        bbox (CocoBoundingBox): Inherited. Bounding box for the object.
    """
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
    """General information about the COCO dataset.

    Attributes:
        description (str): Description of the dataset.
        url (str): URL with more information.
        version (str): Version string.
        year (int): Year of release.
        contributor (str): Dataset contributor.
        date_created (str): Creation date.
    """
    description: str
    url: str
    version: str
    year: int
    contributor: str
    date_created: str

@dataclass
class License:
    """License information for the COCO dataset.

    Attributes:
        id (int): License ID.
        name (str): License name.
        url (str): URL to the license text.
    """
    id: int
    name: str
    url: str

@dataclass
class CocoImage:
    """Image metadata for COCO.

    Attributes:
        id (int): Image ID.
        width (int): Image width in pixels.
        height (int): Image height in pixels.
        file_name (str): Filename of the image.
        date_captured (str): Date the image was captured.
        flickr_url (Optional[str]): Optional Flickr URL.
        coco_url (Optional[str]): Optional COCO URL.
        license (Optional[int]): License ID.
    """
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
    """Object category for COCO.

    Attributes:
        id (int): Category ID.
        name (str): Category name.
        supercategory (Optional[str]): Supercategory name.
    """
    id: int
    name: str
    supercategory: Optional[str] = None


class CocoFile(FileFormat[CocoLabel]):
    """Represents a single COCO annotation file and its metadata.

    Attributes:
        info (Optional[Info]): Dataset information.
        licenses (Optional[list[License]]): List of licenses.
        images (list[CocoImage]): List of images.
        categories (list[Category]): List of categories.
        filename (str): Inherited. Name of the annotation file.
        annotations (list[CocoLabel]): Inherited. List of annotation labels.
    """
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
    """Dataset container for COCO format.

    Attributes:
        name (str): Inherited. Name of the dataset.
        files (list[CocoFile]): Inherited. List of COCO annotation files.
        folder_path (Optional[str]): Inherited. Path to the dataset folder.
        images_path_list (Optional[list[str]]): Inherited - List of images paths
    """

    def __init__(self, name: str, files: list[CocoFile], folder_path: Optional[str] = None, images_path_list: Optional[list[str]] = None) -> None:
        super().__init__(name, files, folder_path, images_path_list)

    @staticmethod
    def build(name: str, files: list[CocoFile], folder_path: Optional[str] = None, images_path_list: Optional[list[str]] = None) -> 'CocoFormat':
        return CocoFormat(name, files, folder_path, images_path_list)
    
    @staticmethod
    def create_coco_file_from_json(coco_data, name: str) -> CocoFile:
        """Creates a CocoFile object from a COCO-format JSON dictionary.

        Args:
            coco_data (dict): Dictionary loaded from COCO JSON.
            name (str): Name for the annotation file.

        Returns:
            CocoFile: Parsed COCO annotation file.
        """

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
    def read_from_folder(path: str, copy_images: bool = False, copy_as_links: bool = False) -> 'CocoFormat':
        """Loads a COCO dataset from a folder.

        Args:
            path (str): Path to the dataset folder or annotation file
            copy_images (bool, default False): If True, loads and stores the image file paths in the dataset object; if False, image paths are not loaded.
            copy_as_links (bool, default False): If True, loads and stores the image file paths in the dataset object; if False, image paths are not loaded.

        Returns:
            CocoFormat: Loaded COCO dataset.

        Raises:
            FileNotFoundError: If the folder or JSON files are missing.
        """

        file = Path(find_annotation_file(path, "json"))

        with open(file, 'r') as f:
            coco_data = json.load(f)

        # Save images path
        image_paths = []
        if copy_images or copy_as_links:
            # Search for images folders
            list_images_dir = find_all_images_folders(file.parent) 
            for images_dir in list_images_dir:
                image_paths += CocoFormat.get_image_paths(images_dir)

        return CocoFormat.build(
            name = "CocoDataset",
            files=[CocoFormat.create_coco_file_from_json(coco_data, file.name)],
            folder_path=str(file.parent),
            images_path_list=image_paths  if len(image_paths) > 0 else None
        )
    
    @staticmethod
    def read_from_json(json_data) -> "CocoFormat":
        """Creates a CocoFormat from a single COCO-format JSON dictionary.

        Args:
            json_data (dict): COCO-format annotation dictionary.

        Returns:
            CocoFormat: Loaded dataset.
        """
        return CocoFormat.build(
            name = "COCO_DATASET",
            files = [CocoFormat.create_coco_file_from_json(json_data, "annotations.json")],
        )


    def save(self, output_folder: str, copy_images: bool = False, copy_as_links: bool = False) -> None:
        """Saves the COCO dataset to the specified output folder.

        ```
        {folder}/  
            ├── images/      # Image files
            └── annotations.json
        ```

        Args:
            output_folder (str): Path to the output directory.
            copy_images (bool, default False): If True, copies image files to the output directory. If False, images are not copied.
            copy_as_links (bool, default False): If True, creates links to the original images in the output directory instead of copying them. If False, no links are created.
        """
        # Save the dataset in output_folder and create a folder with the dataset name
        folder_path = Path(output_folder)

        # Create any folder if necesary
        folder_path.mkdir(parents=True, exist_ok=True)

        # Create images folder
        images_dir = Path(output_folder) / "images"
        images_dir.mkdir(exist_ok=True)

        # Path to create the json file
        json_path = folder_path / "annotations.json"

        # 1 file = Coco Dataset
        coco_file = self.files[0]

        # Info dictionary
        info = vars(coco_file.info) if coco_file.info else {}

        # Licenses dictionary
        licenses = [vars(lic) for lic in coco_file.licenses] if coco_file.licenses else []

        # Images dictionary
        images = [vars(img) for img in coco_file.images]

        # Categories dictionary
        categories = [vars(cat) for cat in coco_file.categories]

        # Annotations dictionary
        annotations = []
        for ann in coco_file.annotations:
            ann_dict = {
                "id": ann.id,
                "image_id": ann.image_id,
                "category_id": ann.category_id,
                "bbox": ann.geometry.getBoundingBox(),
                "area": ann.area if ann.area is not None else 0.0,
                "iscrowd": int(ann.iscrowd) if ann.iscrowd is not None else 0,
            }

            # Segmentation
            if isinstance(ann.segmentation, list):
                ann_dict["segmentation"] = ann.segmentation
            elif isinstance(ann.segmentation, RLESegmentation):
                ann_dict["segmentation"] = {
                    "size": ann.segmentation.size,
                    "counts": ann.segmentation.counts,
                }
            else:
                ann_dict["segmentation"] = []
            annotations.append(ann_dict)

        # Build COCO dictionary
        coco_dict = {
            "info": info,
            "licenses": licenses,
            "images": images,
            "categories": categories,
            "annotations": annotations,
        }

        # Save as json
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(coco_dict, f, ensure_ascii=False, indent=4)

        if copy_images or copy_as_links:
            self.handle_images(self.images_path_list, str(images_dir), copy_images, copy_as_links)