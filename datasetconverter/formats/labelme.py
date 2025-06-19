from abc import abstractmethod
import math
from typing import Any, Optional
import json
import base64
from pathlib import Path

from datasetconverter.utils.file_utils import find_all_images_folders

from .base import Annotation, DatasetFormat, FileFormat, Shape
from .pascal_voc import PascalVocBoundingBox

class LabelMeShape(Shape):

    def __init__(self, type: str) -> None:
        super().__init__(type)
    
    @abstractmethod
    def getBoundingBox(self) -> PascalVocBoundingBox:
        pass

class LabelMePolygon(LabelMeShape):
    """Polygon shape with multiple vertices in LabelMe format.

    Attributes:
        points (list[list[float]]): List of polygon vertices as [[x1, y1], [x2, y2], ...].
        shape_type (str): Inherited. Type of shape ('polygon').
    """

    points: list[list[float]]
    
    def __init__(self, points: list[list[float]]) -> None:
        super().__init__("polygon")
        if len(points) < 3:
            raise ValueError("Polygon must have at least 3 points")
        self.points = points
    
    def getCoordinates(self) -> list[list[float]]:
        """Returns all polygon vertices as [[x1, y1], [x2, y2], ...]"""
        return self.points
    
    def getBoundingBox(self) -> PascalVocBoundingBox:
        """Calculate bounding box as PascalVocBoundingBox object"""
        x_coords: list[float] = [point[0] for point in self.points]
        y_coords: list[float] = [point[1] for point in self.points]
        return PascalVocBoundingBox(
            int(min(x_coords)), 
            int(min(y_coords)), 
            int(max(x_coords)), 
            int(max(y_coords))
        )


class LabelMeRectangle(LabelMeShape):
    """Rectangle shape defined by two diagonal points in LabelMe format.

    Attributes:
        x_min (float): Minimum x coordinate (left).
        y_min (float): Minimum y coordinate (top).
        x_max (float): Maximum x coordinate (right).
        y_max (float): Maximum y coordinate (bottom).
        shape_type (str): Inherited. Type of shape ('rectangle').
    """
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __init__(self, x_min: float, y_min: float, x_max: float, y_max: float) -> None:
        super().__init__("rectangle")
        self.x_min: float = x_min
        self.y_min: float = y_min
        self.x_max: float = x_max
        self.y_max: float = y_max
    
    def getCoordinates(self) -> list[list[float]]:
        """Returns two diagonal points [[x_min, y_min], [x_max, y_max]]"""
        return [[self.x_min, self.y_min], [self.x_max, self.y_max]]
    
    def getBoundingBox(self) -> PascalVocBoundingBox:
        """Returns bounding box as PascalVocBoundingBox object"""
        return PascalVocBoundingBox(
            int(self.x_min), 
            int(self.y_min), 
            int(self.x_max), 
            int(self.y_max)
        )


class LabelMeCircle(LabelMeShape):
    """Circle shape defined by center point and edge point in LabelMe format.

    Attributes:
        center_x (float): X coordinate of circle center.
        center_y (float): Y coordinate of circle center.
        edge_x (float): X coordinate of edge point.
        edge_y (float): Y coordinate of edge point.
        radius (float): Calculated radius of the circle.
        shape_type (str): Inherited. Type of shape ('circle').
    """
    
    center_x: float
    center_y: float
    edge_x: float
    edge_y: float
    radius: float
    
    def __init__(self, center_x: float, center_y: float, edge_x: float, edge_y: float) -> None:
        super().__init__("circle")
        self.center_x = center_x
        self.center_y = center_y
        self.edge_x = edge_x
        self.edge_y = edge_y
        self.radius = math.sqrt((edge_x - center_x) ** 2 + (edge_y - center_y) ** 2)
    
    def getCoordinates(self) -> list[list[float]]:
        """Returns center and edge points [[center_x, center_y], [edge_x, edge_y]]"""
        return [[self.center_x, self.center_y], [self.edge_x, self.edge_y]]
    
    def getBoundingBox(self) -> PascalVocBoundingBox:
        """Calculate bounding box that encloses the circle as PascalVocBoundingBox object"""
        return PascalVocBoundingBox(
            int(self.center_x - self.radius),
            int(self.center_y - self.radius),
            int(self.center_x + self.radius),
            int(self.center_y + self.radius)
        )


class LabelMePoint(LabelMeShape):
    """Point shape with single coordinate in LabelMe format.

    Attributes:
        x (float): X coordinate of the point.
        y (float): Y coordinate of the point.
        shape_type (str): Inherited. Type of shape ('point').
    """
    
    x: float
    y: float
    
    def __init__(self, x: float, y: float) -> None:
        super().__init__("point")
        self.x = x
        self.y = y
    
    def getCoordinates(self) -> list[list[float]]:
        """Returns single point [[x, y]]"""
        return [[self.x, self.y]]
    
    def getBoundingBox(self) -> PascalVocBoundingBox:
        """Returns point as zero-area bounding box as PascalVocBoundingBox object"""
        return PascalVocBoundingBox(int(self.x), int(self.y), int(self.x), int(self.y))


class LabelMeLine(LabelMeShape):
    """Line shape defined by start and end points in LabelMe format.

    Attributes:
        x1 (float): X coordinate of start point.
        y1 (float): Y coordinate of start point.
        x2 (float): X coordinate of end point.
        y2 (float): Y coordinate of end point.
        shape_type (str): Inherited. Type of shape ('line').
    """

    
    x1: float
    y1: float
    x2: float
    y2: float
    
    def __init__(self, x1: float, y1: float, x2: float, y2: float) -> None:
        super().__init__("line")
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
    
    def getCoordinates(self) -> list[list[float]]:
        """Returns start and end points [[x1, y1], [x2, y2]]"""
        return [[self.x1, self.y1], [self.x2, self.y2]]
    
    def getBoundingBox(self) -> PascalVocBoundingBox:
        """Calculate bounding box that encloses the line as PascalVocBoundingBox object"""
        return PascalVocBoundingBox(
            int(min(self.x1, self.x2)),
            int(min(self.y1, self.y2)),
            int(max(self.x1, self.x2)),
            int(max(self.y1, self.y2))
        )


class LabelMeLinestrip(LabelMeShape):
    """Linestrip shape with connected line segments in LabelMe format.

    Attributes:
        points (list[list[float]]): List of connected points as [[x1, y1], [x2, y2], ...].
        shape_type (str): Inherited. Type of shape ('linestrip').
    """
    
    points: list[list[float]]
    
    def __init__(self, points: list[list[float]]) -> None:
        super().__init__("linestrip")
        if len(points) < 2:
            raise ValueError("Linestrip must have at least 2 points")
        self.points = points
    
    def getCoordinates(self) -> list[list[float]]:
        """Returns all connected points [[x1, y1], [x2, y2], ...]"""
        return self.points
    
    def getBoundingBox(self) -> PascalVocBoundingBox:
        """Calculate bounding box that encloses all line segments as PascalVocBoundingBox object"""
        x_coords = [point[0] for point in self.points]
        y_coords = [point[1] for point in self.points]
        return PascalVocBoundingBox(
            int(min(x_coords)), 
            int(min(y_coords)), 
            int(max(x_coords)), 
            int(max(y_coords))
        )


class LabelMePoints(LabelMeShape):
    """Multiple points shape (collection of individual points) in LabelMe format.

    Attributes:
        points (list[list[float]]): List of individual points as [[x1, y1], [x2, y2], ...].
        shape_type (str): Inherited. Type of shape ('points').
    """
    
    points: list[list[float]]
    
    def __init__(self, points: list[list[float]]) -> None:
        super().__init__("points")
        if len(points) < 1:
            raise ValueError("Points must have at least 1 point")
        self.points = points
    
    def getCoordinates(self) -> list[list[float]]:
        """Returns all points [[x1, y1], [x2, y2], ...]"""
        return self.points
    
    def getBoundingBox(self) -> PascalVocBoundingBox:
        """Calculate bounding box that encloses all points as PascalVocBoundingBox object"""
        x_coords = [point[0] for point in self.points]
        y_coords = [point[1] for point in self.points]
        return PascalVocBoundingBox(
            int(min(x_coords)), 
            int(min(y_coords)), 
            int(max(x_coords)), 
            int(max(y_coords))
        )


class LabelMeMask(LabelMeShape):
    """Mask shape defined by rectangular region with binary mask data in LabelMe format.

    Attributes:
        x1 (float): X coordinate of first corner.
        y1 (float): Y coordinate of first corner.
        x2 (float): X coordinate of opposite corner.
        y2 (float): Y coordinate of opposite corner.
        mask_data (Optional[bytes]): Binary mask data for the region.
        shape_type (str): Inherited. Type of shape ('mask').
    """
    
    x1: float
    y1: float
    x2: float
    y2: float
    mask_data: Optional[bytes]
    
    def __init__(self, x1: float, y1: float, x2: float, y2: float, mask_data: Optional[bytes] = None) -> None:
        super().__init__("mask")
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.mask_data = mask_data
    
    def getCoordinates(self) -> list[list[float]]:
        """Returns rectangular bounds [[x1, y1], [x2, y2]]"""
        return [[self.x1, self.y1], [self.x2, self.y2]]
    
    def getBoundingBox(self) -> PascalVocBoundingBox:
        """Returns the mask bounding box as PascalVocBoundingBox object"""
        return PascalVocBoundingBox(
            int(min(self.x1, self.x2)),
            int(min(self.y1, self.y2)),
            int(max(self.x1, self.x2)),
            int(max(self.y1, self.y2))
        )

class LabelMeAnnotation(Annotation[Shape]):
    """Annotation for a single LabelMe object, with shape geometry and metadata.

    Attributes:
        label (str): Class name or label of the annotated object.
        group_id (Optional[int]): Group identifier for related annotations.
        flags (Optional[dict[str, Any]]): Annotation-level flags and metadata.
        description (Optional[str]): Text description of the annotation.
        geometry (Shape): Inherited. Shape geometry of the annotation.
    """
    label: str
    group_id: Optional[int]
    flags: Optional[dict[str,Any]]
    description: Optional[str]

    def __init__(self, shape: Shape, label: str, group_id: Optional[int] = None, flags: Optional[dict[str,Any]] = None, description: Optional[str] = None) -> None:
        super().__init__(geometry = shape)
        self.label = label
        self.group_id = group_id
        self.flags = flags
        self.description = description


class LabelMeFile(FileFormat[LabelMeAnnotation]):
    """Represents a LabelMe annotated file with metadata and shape annotations.

    Attributes:
        version (str): LabelMe version used for annotation.
        flags (Optional[dict[str, Any]]): Image-level flags/metadata.
        imagePath (str): Path to the image file.
        imageData (Optional[str]): Base64 encoded image data.
        imageHeight (int): Image height in pixels.
        imageWidth (int): Image width in pixels.
        filename (str): Inherited. Name of the image file.
        annotations (list[LabelMeAnnotation]): Inherited. List of shape annotations.
    """
    version: str
    flags: Optional[dict[str, Any]]
    imagePath: str
    imageData: Optional[str]
    imageHeight: int
    imageWidth: int

    def __init__(self, filename: str, annotations: list[LabelMeAnnotation], version: str, 
                imagePath: str, imageHeight: int, imageWidth: int, 
                flags: Optional[dict[str, Any]] = None, imageData: Optional[str] = None) -> None:
        super().__init__(filename, annotations)
        self.version = version
        self.flags = flags if flags is not None else {}
        self.imagePath = imagePath
        self.imageData = imageData
        self.imageHeight = imageHeight
        self.imageWidth = imageWidth


class LabelMeFormat(DatasetFormat[LabelMeFile]):
    """Dataset in LabelMe format, including files and folder structure.

    Attributes:
        name (str): Inherited. Name of the dataset.
        files (list[LabelMeFile]): Inherited. List of LabelMeFile objects.
        folder_path (Optional[str]): Inherited. Path to the dataset folder.
        images_path_list (Optional[list[str]]): Inherited - List of images paths
    """
    def __init__(self, name: str, files: list[LabelMeFile], folder_path: str | None = None, images_path_list: Optional[list[str]] = None) -> None:
        super().__init__(name, files, folder_path, images_path_list)

    @staticmethod
    def build(name: str, files: list[LabelMeFile], folder_path: Optional[str] = None, images_path_list: Optional[list[str]] = None) -> 'LabelMeFormat':
        return LabelMeFormat(name, files, folder_path, images_path_list)
    
    @staticmethod
    def create_labelme_file(json_file_name: str, json_data):
        # Read file metadata
        version = json_data.get('version', '5.0.0')
        flags = json_data.get('flags', {})
        image_path = json_data.get('imagePath', '')
        image_data = json_data.get('imageData', None)
        image_height = json_data.get('imageHeight', 0)
        image_width = json_data.get('imageWidth', 0)
        
        # Extract filename from imagePath or use JSON filename
        if image_path:
            filename = Path(image_path).name
        else:
            filename = json_file_name + '.jpg'  # Default extension

        # Read shape annotations
        annotations = []
        shapes_data = json_data.get('shapes', [])
        
        for shape_data in shapes_data:
            label = shape_data.get('label', '')
            shape_type = shape_data.get('shape_type', '')
            points = shape_data.get('points', [])
            group_id = shape_data.get('group_id', None)
            description = shape_data.get('description', None)
            flags_shape = shape_data.get('flags', {})

            # Create appropriate shape based on shape_type
            shape = None
            
            if shape_type == 'polygon':
                shape = LabelMePolygon(points)
            elif shape_type == 'rectangle':
                if len(points) >= 2:
                    x1, y1 = points[0]
                    x2, y2 = points[1]
                    shape = LabelMeRectangle(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
            elif shape_type == 'circle':
                if len(points) >= 2:
                    center_x, center_y = points[0]
                    edge_x, edge_y = points[1]
                    shape = LabelMeCircle(center_x, center_y, edge_x, edge_y)
            elif shape_type == 'point':
                if len(points) >= 1:
                    x, y = points[0]
                    shape = LabelMePoint(x, y)
            elif shape_type == 'line':
                if len(points) >= 2:
                    x1, y1 = points[0]
                    x2, y2 = points[1]
                    shape = LabelMeLine(x1, y1, x2, y2)
            elif shape_type == 'linestrip':
                shape = LabelMeLinestrip(points)
            elif shape_type == 'points':
                shape = LabelMePoints(points)
            elif shape_type == 'mask':
                if len(points) >= 2:
                    x1, y1 = points[0]
                    x2, y2 = points[1]
                    mask_data = shape_data.get('mask', None)
                    if isinstance(mask_data, str):
                        mask_data = base64.b64decode(mask_data)
                    shape = LabelMeMask(x1, y1, x2, y2, mask_data)

            if shape is not None:
                annotation = LabelMeAnnotation(
                    shape=shape,
                    label=label,
                    group_id=group_id,
                    flags=flags_shape if flags_shape else None,
                    description=description
                )
                annotations.append(annotation)

        return LabelMeFile(
            filename=filename,
            annotations=annotations,
            version=version,
            imagePath=image_path,
            imageHeight=image_height,
            imageWidth=image_width,
            flags=flags if flags else None,
            imageData=image_data
        )


    @staticmethod
    def read_from_folder(folder_path: str, copy_images: bool = False, copy_as_links: bool = False) -> 'LabelMeFormat':
        """Create a dataset in LabelMe format from a folder.

        Expecting JSON annotation files in the root folder.
        ```
        folder_path/
                *.json        # JSON annotation files
                *.jpg         # Image files
        ```

        Args:
            folder_path (str): Path to the LabelMe dataset root folder.
            copy_images (bool, default False): If True, loads and stores the image file paths in the dataset object; if False, image paths are not loaded.
            copy_as_links (bool, default False): If True, loads and stores the image file paths in the dataset object; if False, image paths are not loaded.

        Returns:
            LabelMeFormat: Object with the LabelMe dataset.

        Raises:
            FileNotFoundError: If the required folders or files are missing.
        """
        path = Path(folder_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Folder {folder_path} was not found")
        
        # Look for JSON files
        json_files = list(path.glob("*.json"))
        
        if not json_files:
            raise FileNotFoundError(f"No JSON annotation files found in {folder_path}")

        labelme_files = []

        for json_file in json_files:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            labelme_files.append(LabelMeFormat.create_labelme_file(json_file.stem, data))

        # Save images path
        image_paths = []
        if copy_images or copy_as_links:
            # Search for images folders
            list_images_dir = find_all_images_folders(path) 
            for images_dir in list_images_dir:
                image_paths += LabelMeFormat.get_image_paths(images_dir)

        return LabelMeFormat.build(
            name = path.name,
            files = labelme_files,
            folder_path = str(folder_path),
            images_path_list = image_paths if len(image_paths) > 0 else None
        )

    def save(self, folder: str, copy_images: bool = False, copy_as_links: bool = False) -> None:
        """Save the LabelMe dataset to the specified folder.

        The JSON annotation files will be saved in the root folder with the same
        name as the image file but with .json extension.

        Args:
            folder (str): Output directory path.
            copy_images (bool, default False): If True, copies image files to the output directory. If False, images are not copied.
            copy_as_links (bool, default False): If True, creates links to the original images in the output directory instead of copying them. If False, no links are created.
        """
        folder_path = Path(folder)
        
        # Create output folder if necessary
        folder_path.mkdir(parents=True, exist_ok=True)
        
        # Save all JSON annotation files
        for file in self.files:
            
            # Prepare shapes data
            shapes_data = []
            for annotation in file.annotations:
                shape_data = {
                    'label': annotation.label,
                    'shape_type': annotation.geometry.shape_type,
                    'flags': annotation.flags or {}
                }
                
                # Add optional fields
                if annotation.group_id is not None:
                    shape_data['group_id'] = annotation.group_id
                if annotation.description is not None:
                    shape_data['description'] = annotation.description
                
                # Add points based on shape type
                if isinstance(annotation.geometry, (LabelMePolygon, LabelMeLinestrip, LabelMePoints)):
                    shape_data['points'] = annotation.geometry.points
                elif isinstance(annotation.geometry, LabelMeRectangle):
                    shape_data['points'] = [[annotation.geometry.x_min, annotation.geometry.y_min],
                                        [annotation.geometry.x_max, annotation.geometry.y_max]]
                elif isinstance(annotation.geometry, LabelMeCircle):
                    shape_data['points'] = [[annotation.geometry.center_x, annotation.geometry.center_y],
                                        [annotation.geometry.edge_x, annotation.geometry.edge_y]]
                elif isinstance(annotation.geometry, LabelMePoint):
                    shape_data['points'] = [[annotation.geometry.x, annotation.geometry.y]]
                elif isinstance(annotation.geometry, LabelMeLine):
                    shape_data['points'] = [[annotation.geometry.x1, annotation.geometry.y1],
                                        [annotation.geometry.x2, annotation.geometry.y2]]
                elif isinstance(annotation.geometry, LabelMeMask):
                    shape_data['points'] = [[annotation.geometry.x1, annotation.geometry.y1],
                                        [annotation.geometry.x2, annotation.geometry.y2]]
                    if annotation.geometry.mask_data is not None:
                        shape_data['mask'] = base64.b64encode(annotation.geometry.mask_data).decode('utf-8')
                
                shapes_data.append(shape_data)
            
            # Prepare JSON data
            json_data = {
                'version': file.version,
                'flags': file.flags if file.flags is not None else {},
                'shapes': shapes_data,
                'imagePath': file.imagePath,
                'imageHeight': file.imageHeight,
                'imageWidth': file.imageWidth
            }
            
            # Add imageData if present
            if file.imageData is not None:
                json_data['imageData'] = file.imageData
            
            # Save JSON file
            filename = Path(file.filename).stem + ".json"
            json_path = folder_path / filename
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)

        if copy_images or copy_as_links:
            self.handle_images(self.images_path_list, folder, copy_images, copy_as_links)