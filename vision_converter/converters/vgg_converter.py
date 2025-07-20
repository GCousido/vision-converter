from pathlib import Path
from typing import Optional
from .dataset_converter import DatasetConverter
from ..formats.neutral_format import ImageOrigin, NeutralAnnotation, NeutralFile, NeutralFormat
from ..formats.vgg import (
    VGGFile, VGGFormat, VGGAnnotation,
    VGGRect, VGGCircle, VGGEllipse, VGGPolygon, VGGPolyline, VGGPoint, extract_class_name
)
from ..utils.file_utils import estimate_file_size, get_image_path, get_image_info_from_file
from ..formats.bounding_box import CornerAbsoluteBoundingBox

class VGGConverter(DatasetConverter[VGGFormat]):
    """Converter between VGGFormat and NeutralFormat
    
    Provides bidirectional conversion capabilities between VGGFormat and NeutralFormat
    """
    
    @staticmethod
    def toNeutral(df: VGGFormat) -> NeutralFormat:
        """Convert a VGG dataset to the Neutral format

        Args:
            df (VGGFormat): object that represents a VGG dataset to convert

        Returns:
            NeutralFormat: object with the converted dataset
        """
        neutral_files: list[NeutralFile] = [VGGFile_to_NeutralFile(i, df.folder_path) for i in df.files]

        # Unique class list from region_attributes
        unique_classes = list({
            # Extract class names from various possible attribute keys
            attr_value
            for file in df.files 
            for annotation in file.annotations
            for attr_key, attr_value in annotation.region_attributes.items()
            if isinstance(attr_value, str) and attr_key in ['type', 'class', 'label', 'name', 'names', 'category']
        })

        # If no classes found, use a default
        if not unique_classes:
            unique_classes = ['object']

        class_map = {i: class_name for i, class_name in enumerate(unique_classes)}

        return NeutralFormat(
            name=df.name, 
            files=neutral_files,
            original_format="vgg",
            metadata={
                "vgg_version": "2.0.10",
                "total_regions": sum(len(file.annotations) for file in df.files),
                "shape_types_used": list({
                    annotation.geometry.shape_type
                    for file in df.files 
                    for annotation in file.annotations
                })
            },
            class_map=class_map,
            images_path_list=df.images_path_list
        )
    

    @staticmethod
    def fromNeutral(nf: NeutralFormat) -> VGGFormat:
        """Convert the Neutral dataset to the VGG format.

        Args:
            nf (NeutralFormat): object that represents a Neutral dataset to convert

        Returns:
            VGGFormat: object with the converted dataset
        """
        vgg_files: list[VGGFile] = [NeutralFile_to_VGGFile(i) for i in nf.files]

        return VGGFormat(
            name=nf.name,
            files=vgg_files,
            folder_path=nf.folder_path,
            images_path_list=nf.images_path_list
        )


def VGGFile_to_NeutralFile(file: VGGFile, folder_path: Optional[str] = None) -> NeutralFile:
    """Convert a VGG file to the Neutral format

    Args:
        file (VGGFile): object that represents a VGG file to convert
        folder_path (str, optional): path to the dataset folder for image dimension extraction

    Returns:
        NeutralFile: object with the converted file
    """
    neutral_annotations: list[NeutralAnnotation] = [
        VGGAnnotation_to_NeutralAnnotation(annotation) 
        for annotation in file.annotations
    ] 

    image_depth = 3 # Default

    # Get image dimensions
    if folder_path:
        # If the format was constructed from a folder, get image dimension from the image
        image_path = get_image_path(folder_path, "images", file.filename)
        if not image_path:
            raise FileNotFoundError(f"Image {file.filename} not found")

        image_width, image_height, image_depth = get_image_info_from_file(image_path)
    else:
        if file.image_width  and file.image_height:
            image_width = file.image_width 
            image_height = file.image_height
        else:
            raise Exception(f"Error during conversion: Dimensions not found for image{file.filename}")

    image_origin = ImageOrigin(
        extension=Path(file.filename).suffix or ".jpg"
    )

    return NeutralFile(
        filename=Path(file.filename).stem,
        annotations=neutral_annotations,
        width=image_width,
        height=image_height,
        depth=image_depth,
        image_origin=image_origin,
        params={
            "vgg_file_size": file.size,
            "vgg_file_attributes": file.file_attributes,
            "shape_count_by_type": _count_shapes_by_type(file.annotations)
        }
    )


def VGGAnnotation_to_NeutralAnnotation(annotation: VGGAnnotation) -> NeutralAnnotation:
    """Convert a VGG annotation to the Neutral format

    Args:
        annotation (VGGAnnotation): object that represents a VGG annotation to convert

    Returns:
        NeutralAnnotation: object with the converted annotation
    """
    # Get bounding box from any shape type
    bbox = annotation.geometry.getBoundingBox()
    
    # Extract class name from region_attributes
    class_name = extract_class_name(annotation.region_attributes)
    
    # Prepare attributes with shape-specific information
    attributes = {
        "shape_type": annotation.geometry.shape_type,
        "coordinates": annotation.geometry.getCoordinates(),
        "original_region_attributes": annotation.region_attributes
    }

    if isinstance(bbox, CornerAbsoluteBoundingBox):
        return NeutralAnnotation(
            bbox=bbox,
            class_name=class_name,
            attributes=attributes
        )
    else:
        raise TypeError(f"Expected CornerAbsoluteBoundingBox, got {type(bbox)}")


def NeutralFile_to_VGGFile(file: NeutralFile) -> VGGFile:
    """Converts a NeutralFile to a VGG format.

    Args:
        file (NeutralFile): object with the file to convert

    Returns:
        VGGFile: object with the converted file
    """
    vgg_annotations: list[VGGAnnotation] = [
        NeutralAnnotation_to_VGGAnnotation(annotation) 
        for annotation in file.annotations
    ]

    # Extract VGG-specific parameters
    file_size = file.params.get("vgg_file_size", -1)
    
    # If it was not found then estimate
    if file_size == -1:
        file_size = estimate_file_size(file.width, file.height, file.depth, file.image_origin.extension)

    file_attributes = file.params.get("vgg_file_attributes", {})
    
    return VGGFile(
        filename=file.filename + file.image_origin.extension,
        size=file_size,
        annotations=vgg_annotations,
        file_attributes=file_attributes
    )


def NeutralAnnotation_to_VGGAnnotation(annotation: NeutralAnnotation) -> VGGAnnotation:
    """Converts a NeutralAnnotation to a VGG format.

    Args:
        annotation (NeutralAnnotation): object with the annotation to convert

    Returns:
        VGGAnnotation: object with the converted annotation
    """

    shape = VGGRect(
        annotation.geometry.x_min,
        annotation.geometry.y_min,
        annotation.geometry.x_max - annotation.geometry.x_min,
        annotation.geometry.y_max - annotation.geometry.y_min
    )
    
    region_attributes = {
        "type": annotation.class_name
    }
    
    return VGGAnnotation(
        shape=shape,
        region_attributes=region_attributes
    )


def _count_shapes_by_type(annotations: list[VGGAnnotation]) -> dict[str, int]:
    """Count annotations by shape type.
    
    Args:
        annotations: List of VGG annotations
        
    Returns:
        dict: Count of each shape type
    """
    shape_counts = {}
    for annotation in annotations:
        shape_type = annotation.geometry.shape_type
        shape_counts[shape_type] = shape_counts.get(shape_type, 0) + 1
    return shape_counts