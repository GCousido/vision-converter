import math
from typing import Any, Optional, Dict, List
import json
from pathlib import Path

from datasetconverter.utils.file_utils import find_annotation_file

from .base import Annotation, DatasetFormat, FileFormat, Shape
from .pascal_voc import PascalVocBoundingBox


class VGGRect(Shape):
    """Rectangle shape in VGG format.
    
    Attributes:
        x (float): X coordinate of top-left corner.
        y (float): Y coordinate of top-left corner.
        width (float): Width of the rectangle.
        height (float): Height of the rectangle.
        shape_type (str): Type of shape ('rect').
    """
    x: float
    y: float
    width: float
    height: float
    
    def __init__(self, x: float, y: float, width: float, height: float) -> None:
        super().__init__("rect")
        self.x = x
        self.y = y
        self.width = width
        self.height = height
    
    def getCoordinates(self) -> Dict[str, float]:
        """Returns rectangle coordinates as VGG format."""
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height
        }
    
    def getBoundingBox(self) -> PascalVocBoundingBox:
        """Convert to Pascal VOC bounding box format."""
        return PascalVocBoundingBox(
            int(self.x),
            int(self.y),
            int(self.x + self.width),
            int(self.y + self.height)
        )


class VGGCircle(Shape):
    """Circle shape in VGG format.
    
    Attributes:
        cx (float): X coordinate of center.
        cy (float): Y coordinate of center.
        r (float): Radius of the circle.
        shape_type (str): Type of shape ('circle').
    """
    
    cx: float
    cy: float
    r: float
    
    def __init__(self, cx: float, cy: float, r: float) -> None:
        super().__init__("circle")
        self.cx = cx
        self.cy = cy
        self.r = r
    
    def getCoordinates(self) -> Dict[str, float]:
        """Returns circle coordinates as VGG format."""
        return {
            "cx": self.cx,
            "cy": self.cy,
            "r": self.r
        }
    
    def getBoundingBox(self) -> PascalVocBoundingBox:
        """Convert circle to bounding box."""
        return PascalVocBoundingBox(
            int(self.cx - self.r),
            int(self.cy - self.r),
            int(self.cx + self.r),
            int(self.cy + self.r)
        )


class VGGEllipse(Shape):
    """Ellipse shape in VGG format.
    
    Attributes:
        cx (float): X coordinate of center.
        cy (float): Y coordinate of center.
        rx (float): Horizontal radius.
        ry (float): Vertical radius.
        theta (float): Rotation angle in radians.
        shape_type (str): Type of shape ('ellipse').
    """
    
    cx: float
    cy: float
    rx: float
    ry: float
    theta: float
    
    def __init__(self, cx: float, cy: float, rx: float, ry: float, theta: float = 0) -> None:
        super().__init__("ellipse")
        self.cx = cx
        self.cy = cy
        self.rx = rx
        self.ry = ry
        self.theta = theta
    
    def getCoordinates(self) -> Dict[str, float]:
        """Returns ellipse coordinates as VGG format."""
        return {
            "cx": self.cx,
            "cy": self.cy,
            "rx": self.rx,
            "ry": self.ry,
            "theta": self.theta
        }
    
    def getBoundingBox(self) -> PascalVocBoundingBox:
        """Convert ellipse to bounding box considering rotation."""
        # Calculate bounding box for rotated ellipse
        cos_theta = math.cos(self.theta)
        sin_theta = math.sin(self.theta)
        
        # Calculate the extent of the rotated ellipse
        extent_x = math.sqrt((self.rx * cos_theta) ** 2 + (self.ry * sin_theta) ** 2)
        extent_y = math.sqrt((self.rx * sin_theta) ** 2 + (self.ry * cos_theta) ** 2)
        
        return PascalVocBoundingBox(
            int(self.cx - extent_x),
            int(self.cy - extent_y),
            int(self.cx + extent_x),
            int(self.cy + extent_y)
        )


class VGGPolygon(Shape):
    """Polygon shape in VGG format.
    
    Attributes:
        all_points_x (List[float]): List of X coordinates.
        all_points_y (List[float]): List of Y coordinates.
        shape_type (str): Type of shape ('polygon').
    """
    
    all_points_x: List[float]
    all_points_y: List[float]
    
    def __init__(self, all_points_x: List[float], all_points_y: List[float]) -> None:
        super().__init__("polygon")
        if len(all_points_x) != len(all_points_y) or len(all_points_x) < 3:
            raise ValueError("Polygon must have at least 3 points and equal X,Y arrays")
        self.all_points_x = all_points_x
        self.all_points_y = all_points_y
    
    def getCoordinates(self) -> Dict[str, List[float]]:
        """Returns polygon coordinates as VGG format."""
        return {
            "all_points_x": self.all_points_x,
            "all_points_y": self.all_points_y
        }
    
    def getBoundingBox(self) -> PascalVocBoundingBox:
        """Convert polygon to bounding box."""
        return PascalVocBoundingBox(
            int(min(self.all_points_x)),
            int(min(self.all_points_y)),
            int(max(self.all_points_x)),
            int(max(self.all_points_y))
        )


class VGGPolyline(Shape):
    """Polyline shape in VGG format.
    
    Attributes:
        all_points_x (List[float]): List of X coordinates.
        all_points_y (List[float]): List of Y coordinates.
        shape_type (str): Type of shape ('polyline').
    """
    
    all_points_x: List[float]
    all_points_y: List[float]
    
    def __init__(self, all_points_x: List[float], all_points_y: List[float]) -> None:
        super().__init__("polyline")
        if len(all_points_x) != len(all_points_y) or len(all_points_x) < 2:
            raise ValueError("Polyline must have at least 2 points and equal X,Y arrays")
        self.all_points_x = all_points_x
        self.all_points_y = all_points_y
    
    def getCoordinates(self) -> Dict[str, List[float]]:
        """Returns polyline coordinates as VGG format."""
        return {
            "all_points_x": self.all_points_x,
            "all_points_y": self.all_points_y
        }
    
    def getBoundingBox(self) -> PascalVocBoundingBox:
        """Convert polyline to bounding box."""
        return PascalVocBoundingBox(
            int(min(self.all_points_x)),
            int(min(self.all_points_y)),
            int(max(self.all_points_x)),
            int(max(self.all_points_y))
        )


class VGGPoint(Shape):
    """Point shape in VGG format.
    
    Attributes:
        cx (float): X coordinate of the point.
        cy (float): Y coordinate of the point.
        shape_type (str): Type of shape ('point').
    """
    
    cx: float
    cy: float
    
    def __init__(self, cx: float, cy: float) -> None:
        super().__init__("point")
        self.cx = cx
        self.cy = cy
    
    def getCoordinates(self) -> Dict[str, float]:
        """Returns point coordinates as VGG format."""
        return {
            "cx": self.cx,
            "cy": self.cy
        }
    
    def getBoundingBox(self) -> PascalVocBoundingBox:
        """Convert point to minimal bounding box."""
        return PascalVocBoundingBox(
            int(self.cx),
            int(self.cy),
            int(self.cx + 1),
            int(self.cy + 1)
        )


class VGGAnnotation(Annotation[Shape]):
    """Annotation for a single VGG object with shape and attributes.
    
    Attributes:
        region_attributes (Dict[str, Any]): Custom attributes for the region.
        geometry (Shape): Shape geometry of the annotation.
    """
    
    region_attributes: Dict[str, Any]
    
    def __init__(self, shape: Shape, region_attributes: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(geometry=shape)
        self.region_attributes = region_attributes if region_attributes is not None else {}


class VGGFile(FileFormat[VGGAnnotation]):
    """Represents a VGG annotated file with metadata and regions.
    
    Attributes:
        size (int): File size in bytes.
        regions (List[VGGAnnotation]): List of region annotations.
        file_attributes (Dict[str, Any]): Image-level attributes.
        filename (str): Name of the image file.
        annotations (List[VGGAnnotation]): List of annotations.
    """
    
    size: int
    file_attributes: Dict[str, Any]

    image_width: Optional[int]
    image_height: Optional[int]
    
    def __init__(self, filename: str, size: int, annotations: List[VGGAnnotation], 
                file_attributes: Optional[Dict[str, Any]] = None, image_width: Optional[int] = None, image_height: Optional[int] = None) -> None:
        super().__init__(filename, annotations)
        self.size = size
        self.file_attributes = file_attributes if file_attributes is not None else {}
        self.image_width = image_width
        self.image_height = image_height


class VGGFormat(DatasetFormat[VGGFile]):
    """Dataset in VGG Image Annotator format.
    
    Attributes:
        name (str): Name of the dataset.
        files (List[VGGFile]): List of VGGFile objects.
        folder_path (Optional[str]): Path to the dataset folder.
    """

    def __init__(self, name: str, files: List[VGGFile], folder_path: Optional[str] = None) -> None:
        super().__init__(name, files, folder_path)

    @staticmethod
    def build(name: str, files: List[VGGFile], folder_path: Optional[str] = None) -> 'VGGFormat':
        return VGGFormat(name, files, folder_path)
    
    @staticmethod
    def create_vgg_file(image_data: Dict[str, Any]) -> VGGFile:
        """Create a VGGFile from VIA JSON image data."""
        filename = image_data.get('filename', '')
        size = image_data.get('size', -1)
        file_attributes = image_data.get('file_attributes', {})
        regions_data = image_data.get('regions', [])
        
        annotations = []
        
        for region_data in regions_data:
            shape_attributes = region_data.get('shape_attributes', {})
            region_attributes = region_data.get('region_attributes', {})
            
            shape_name = shape_attributes.get('name', '')
            shape = None
            
            if shape_name == 'rect':
                shape = VGGRect(
                    shape_attributes.get('x', 0),
                    shape_attributes.get('y', 0),
                    shape_attributes.get('width', 0),
                    shape_attributes.get('height', 0)
                )
            elif shape_name == 'circle':
                shape = VGGCircle(
                    shape_attributes.get('cx', 0),
                    shape_attributes.get('cy', 0),
                    shape_attributes.get('r', 0)
                )
            elif shape_name == 'ellipse':
                shape = VGGEllipse(
                    shape_attributes.get('cx', 0),
                    shape_attributes.get('cy', 0),
                    shape_attributes.get('rx', 0),
                    shape_attributes.get('ry', 0),
                    shape_attributes.get('theta', 0)
                )
            elif shape_name == 'polygon':
                shape = VGGPolygon(
                    shape_attributes.get('all_points_x', []),
                    shape_attributes.get('all_points_y', [])
                )
            elif shape_name == 'polyline':
                shape = VGGPolyline(
                    shape_attributes.get('all_points_x', []),
                    shape_attributes.get('all_points_y', [])
                )
            elif shape_name == 'point':
                shape = VGGPoint(
                    shape_attributes.get('cx', 0),
                    shape_attributes.get('cy', 0)
                )
            
            if shape is not None:
                annotation = VGGAnnotation(shape, region_attributes)
                annotations.append(annotation)
        
        return VGGFile(filename, size, annotations, file_attributes)

    @staticmethod
    def read_from_folder(path: str) -> 'VGGFormat':
        """Load VGG dataset from a single JSON file.
        
        Args:
            path (str): Path to the dataset folder or annotation file
            
        Returns:
            VGGFormat: VGG dataset object.

        Raises:
            FileNotFoundError: If the folder or JSON files are missing.
        """
        annotations_path = Path(find_annotation_file(path, "json"))
        
        with open(annotations_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract VIA image metadata
        via_img_metadata = data.get('_via_img_metadata', {})
        
        # If no _via_img_metadata, assume the whole JSON is image metadata
        if not via_img_metadata:
            via_img_metadata = {k: v for k, v in data.items() if not k.startswith('_via_')}
        
        vgg_files = []
        for image_key, image_data in via_img_metadata.items():
            vgg_file = VGGFormat.create_vgg_file(image_data)
            vgg_files.append(vgg_file)
        
        return VGGFormat(
            name=annotations_path.stem,
            files=vgg_files,
            folder_path=str(annotations_path.parent)
        )


    def save(self, output_path: str) -> None:
        """Save VGG dataset to JSON file.
        
        ```text
            dataset/
                |-- images/
                |     img1.jpg
                |     img2.jpg 
                ├── annotations.json
        '''
        Args:
            output_path (str): Output folder path for the VIA JSON.
        """
        # Create output directory if needed
        Path(output_path).mkdir(parents=True, exist_ok=True)

        Path(output_path + "/images").mkdir(exist_ok=True)
        output_file = Path(output_path) / "annotations.json"

        
        # Build VIA JSON structure
        via_img_metadata = {}
        
        for file in self.files:
            image_key = f"{file.filename}{file.size}"
            
            # Build regions data
            regions = []
            for annotation in file.annotations:
                region_data = {
                    'shape_attributes': {
                        'name': annotation.geometry.shape_type,
                        **annotation.geometry.getCoordinates()
                    },
                    'region_attributes': annotation.region_attributes
                }
                regions.append(region_data)
            
            # Build image data
            image_data = {
                'filename': file.filename,
                'size': file.size,
                'regions': regions,
                'file_attributes': file.file_attributes
            }
            
            via_img_metadata[image_key] = image_data
        
        # Build complete VIA JSON
        via_json = {
            '_via_img_metadata': via_img_metadata
        }
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(via_json, f, indent=2, ensure_ascii=False)


def extract_class_name(region_attributes: dict[str, Any]) -> str:
    """Extract class name from VGG region attributes.
    
    Args:
        region_attributes: Dictionary of region attributes
        
    Returns:
        str: Extracted class name or 'object' as default
    """
    # Common attribute keys that might contain class names
    class_keys = ['type', 'class', 'label', 'name', 'names', 'category', 'object_type']
    
    for key in class_keys:
        if key in region_attributes and isinstance(region_attributes[key], str):
            return region_attributes[key]
    
    # If no class found, return first string value or default
    for value in region_attributes.values():
        if isinstance(value, str) and value.strip():
            return value.strip()
    
    return 'object'