from pathlib import Path
from typing import Any
from .dataset_converter import DatasetConverter
from ..formats.neutral_format import ImageOrigin, NeutralAnnotation, NeutralFile, NeutralFormat
from ..formats.labelme import (
    LabelMeFile, LabelMeFormat, LabelMeAnnotation,
    LabelMePolygon, LabelMeRectangle, LabelMeCircle, LabelMePoint, 
    LabelMeLine, LabelMeLinestrip, LabelMePoints, LabelMeMask
)
from ..formats.pascal_voc import PascalVocBoundingBox


class LabelMeConverter(DatasetConverter[LabelMeFormat]):
    """Converter between LabelMeFormat and NeutralFormat
    
    Provides bidirectional conversion capabilities between LabelMeFormat and NeutralFormat
    """
    
    @staticmethod
    def toNeutral(df: LabelMeFormat) -> NeutralFormat:
        """Convert a LabelMe dataset to the Neutral format

        Args:
            df (LabelMeFormat): object that represents a LabelMe dataset to convert

        Returns:
            NeutralFormat: object with the converted dataset
        """
        neutral_files: list[NeutralFile] = [LabelMeFile_to_NeutralFile(i) for i in df.files]

        # Unique class list (first creating a set(unique elements) and then a list)
        unique_classes = list({
            annotation.label
            for file in df.files 
            for annotation in file.annotations
        }) 

        class_map = {i: class_name for i, class_name in enumerate(unique_classes)}

        return NeutralFormat(
            name = df.name, 
            files = neutral_files,
            original_format = "labelme",
            metadata = {
                "labelme_version": df.files[0].version if df.files else "5.0.0"
            },
            class_map = class_map
        )
    

    @staticmethod
    def fromNeutral(nf: NeutralFormat) -> LabelMeFormat:
        """Convert the Neutral dataset to the LabelMe format.

        Args:
            nf (NeutralFormat): object that represents a Neutral dataset to convert

        Returns:
            LabelMeFormat: object with the converted dataset
        """
        labelme_files: list[LabelMeFile] = [NeutralFile_to_LabelMeFile(i, nf.metadata) for i in nf.files]

        return LabelMeFormat(
            name = nf.name,
            files = labelme_files
        )


def LabelMeFile_to_NeutralFile(file: LabelMeFile) -> NeutralFile:
    """Convert a LabelMe file to the Neutral format

    Args:
        file (LabelMeFile): object that represents a LabelMe file to convert

    Returns:
        NeutralFile: object with the converted file
    """
    neutral_annotations: list[NeutralAnnotation] = [
        LabelMeAnnotation_to_NeutralAnnotation(annotation) 
        for annotation in file.annotations
    ] 

    image_origin = ImageOrigin(
        extension = Path(file.filename).suffix or ".jpg"
    )

    return NeutralFile(
        filename = Path(file.filename).stem,
        annotations = neutral_annotations,
        width = file.imageWidth,
        height = file.imageHeight,
        depth=3,  # Default RGB depth
        image_origin = image_origin,
        params={
            "labelme_version": file.version,
            "labelme_flags": file.flags,
            "has_image_data": file.imageData is not None
        }
    )


def LabelMeAnnotation_to_NeutralAnnotation(annotation: LabelMeAnnotation) -> NeutralAnnotation:
    """Convert a LabelMe annotation to the Neutral format

    Args:
        annotation (LabelMeAnnotation): object that represents a LabelMe annotation to convert

    Returns:
        NeutralAnnotation: object with the converted annotation
    """
    # Get bounding box from any shape type
    bbox = annotation.geometry.getBoundingBox()
    
    # Prepare attributes with shape-specific information
    attributes = {
        "shape_type": annotation.geometry.shape_type,
        "coordinates": annotation.geometry.getCoordinates()
    }
    
    # Add optional LabelMe-specific attributes
    if annotation.group_id is not None:
        attributes["group_id"] = annotation.group_id
    if annotation.description is not None:
        attributes["description"] = annotation.description
    if annotation.flags is not None:
        attributes["flags"] = annotation.flags
    
    # Add shape-specific attributes
    if isinstance(annotation.geometry, LabelMeCircle):
        attributes["radius"] = annotation.geometry.radius
    elif isinstance(annotation.geometry, LabelMeMask):
        attributes["has_mask_data"] = annotation.geometry.mask_data is not None

    if isinstance(bbox, PascalVocBoundingBox):
        return NeutralAnnotation(
            bbox = bbox,
            class_name = annotation.label,
            attributes = attributes
        )
    else:
        raise TypeError(f"Expected PascalVocBoundingBox, got {type(bbox)}")


def NeutralFile_to_LabelMeFile(file: NeutralFile, metadata: dict[str, Any]) -> LabelMeFile:
    """Converts a NeutralFile to a LabelMe format.

    Args:
        file (NeutralFile): object with the file to convert
        metadata (dict[str, Any]): global metadata from the neutral format

    Returns:
        LabelMeFile: object with the converted file
    """
    labelme_annotations: list[LabelMeAnnotation] = [
        NeutralAnnotation_to_LabelMeAnnotation(annotation) 
        for annotation in file.annotations
    ]

    # Extract LabelMe-specific parameters
    version = file.params.get("labelme_version", metadata.get("labelme_version", "5.0.0"))
    flags = file.params.get("labelme_flags", {})
    
    return LabelMeFile(
        filename = file.filename + file.image_origin.extension,
        annotations = labelme_annotations,
        version = version,
        imagePath = file.filename + file.image_origin.extension,
        imageHeight = file.height,
        imageWidth = file.width,
        flags = flags,
        imageData = None  # Not preserved in neutral format
    )


def NeutralAnnotation_to_LabelMeAnnotation(annotation: NeutralAnnotation) -> LabelMeAnnotation:
    """Converts a NeutralAnnotation to a LabelMe format.

    Args:
        annotation (NeutralAnnotation): object with the annotation to convert

    Returns:
        LabelMeAnnotation: object with the converted annotation
    """
    
    # Fallback to rectangle using bounding box
    shape = LabelMeRectangle(
        annotation.geometry.x_min, 
        annotation.geometry.y_min, 
        annotation.geometry.x_max, 
        annotation.geometry.y_max
    )
    
    # Extract LabelMe-specific attributes
    group_id = annotation.attributes.get("group_id", None)
    description = annotation.attributes.get("description", None)
    flags = annotation.attributes.get("flags", None)
    
    return LabelMeAnnotation(
        shape = shape,
        label = annotation.class_name,
        group_id = group_id,
        flags = flags,
        description = description
    )
