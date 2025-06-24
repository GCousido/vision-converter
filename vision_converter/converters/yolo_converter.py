from typing import Optional

from .dataset_converter import DatasetConverter
from ..formats.neutral_format import ImageOrigin, NeutralAnnotation, NeutralFile, NeutralFormat
from ..formats.pascal_voc import PascalVocBoundingBox
from ..formats.yolo import YoloAnnotation, YoloBoundingBox, YoloFile, YoloFormat
from ..utils.bbox_utils import PascalVocBBox_to_YoloBBox, YoloBBox_to_PascalVocBBox
from ..utils.file_utils import get_image_path, get_image_info_from_file


class YoloConverter(DatasetConverter[YoloFormat]):
    """Converter between YoloFormat and NeutralFormat
    
    Provides bidirectional conversion capabilities between YoloFormat and NeutralFormat
    """
    
    @staticmethod
    def toNeutral(df: YoloFormat) -> NeutralFormat:
        """Convert a YoloFormat dataset to Neutral format.
        
        Args:
            df (YoloFormat): Source dataset in YOLO format
            
        Returns:
            NeutralFormat: Converted dataset in neutral annotation format
            
        Raises:
            FileNotFoundError: If image files are missing when constructed from folder
            ValueError: If image dimensions are missing in YoloFile when not constructed from folder
        """

        neutral_files: list[NeutralFile] = [YoloFile_to_NeutralFile(i, df.class_labels, df.folder_path) for i in df.files]

        return NeutralFormat(
            name = df.name, 
            files = neutral_files,
            original_format = "yolo",
            metadata = None,
            class_map = df.class_labels,
            images_path_list = df.images_path_list
        )
    
    
    @staticmethod
    def fromNeutral(nf: NeutralFormat) -> YoloFormat:
        """Convert a NeutralFormat dataset to YOLO format.
        
        Args:
            nf (NeutralFormat): Source dataset in neutral format
            
        Returns:
            YoloFormat: Converted dataset in YOLO format
            
        Raises:
            ValueError: If the class name of an annotation was not found in the class map
        """

        # Invert the class map to get the ids
        inverse_class_map: dict[str, int] = {v: k for k, v in nf.class_map.items()}

        yolo_files: list[YoloFile] = [NeutralFile_to_YoloFile(i, inverse_class_map) for i in nf.files]

        return YoloFormat(
            name = nf.name,
            files = yolo_files,
            class_labels = nf.class_map,
            images_path_list = nf.images_path_list
        )



def YoloFile_to_NeutralFile(file: YoloFile, class_labels: dict[int, str], folder_path: Optional[str] = None) -> NeutralFile:
    """Convert a YoloFile representation to Neutral format.
    
    Args:
        file (YoloFile): YOLO file object to convert
        class_labels (dict[int, str]): Mapping from class IDs to class names
        folder_path (Optional[str]): Optional path to folder for dimension validation
        
    Returns:
        NeutralFile: Converted file representation
        
    Raises:
        FileNotFoundError: If image file is missing when folder_path is provided
        ValueError: If image dimensions are missing and can't be retrieved
    """

    image_width: int
    image_height: int
    image_depth: int
    
    # If the format was constructed from a folder, get image dimension from the image
    if folder_path:
        image_path = get_image_path(folder_path, "images", file.filename)
        if not image_path:
            raise FileNotFoundError(f"Image {file.filename} not found")

        image_width, image_height, image_depth = get_image_info_from_file(image_path)
    else:
        # Check all dimensions exist
        if not (file.width and file.height and file.depth):
            raise ValueError("Missing image dimensions in YoloFile when dataset was not created from folder")
    
        image_width = file.width
        image_height = file.height
        image_depth = file.depth

    neutral_annotations: list[NeutralAnnotation] = [
        YoloAnnotation_to_NeutralAnnotation(i, class_labels, image_width, image_height)
        for i in file.annotations
    ]

    image_origin = ImageOrigin(
        extension = ""
    )

    return NeutralFile(
        filename = file.filename,
        annotations = neutral_annotations,
        width = image_width,
        height = image_height,
        depth = image_depth,
        image_origin = image_origin
    )


def YoloAnnotation_to_NeutralAnnotation(annotation: YoloAnnotation, class_labels: dict[int, str], image_width: int, image_height: int) -> NeutralAnnotation:
    """Convert a YoloAnnotation to Neutral format.
    
    Args:
        annotation (YoloAnnotation): Source YOLO annotation
        class_labels (dict[int, str]): Mapping from class IDs to class names
        image_width (int): Width of the associated image
        image_height (int): Height of the associated image
        
    Returns:
        NeutralAnnotation: Converted annotation
        
    Raises:
        KeyError: If class ID doesn't exist in class_labels
    """
    
    bbox: PascalVocBoundingBox = YoloBBox_to_PascalVocBBox(annotation.geometry, image_width, image_height)

    class_name: str = class_labels[annotation.id_class]

    return NeutralAnnotation(bbox, class_name)



def NeutralFile_to_YoloFile(file: NeutralFile, inverse_class_list: dict[str, int]) -> YoloFile:
    """Convert a NeutralFile to YOLO format.
    
    Args:
        file (NeutralFile): Neutral format file to convert
        inverse_class_map (dict[str, int]): Mapping from class names to class IDs
        
    Returns:
        YoloFile: Converted YOLO file
        
    Raises:
        ValueError: If class names don't exist in inverse_class_map
    """

    yolo_annotations: list[YoloAnnotation] = [NeutralAnnotation_to_YoloAnnotation(i, inverse_class_list, file.width, file.height) for i in file.annotations]

    return YoloFile(
        filename = file.filename + file.image_origin.extension,
        annotations = yolo_annotations,
        width = file.width,
        height = file.height,
        depth = file.depth
    )


def NeutralAnnotation_to_YoloAnnotation(annotation: NeutralAnnotation, inverse_class_list: dict[str, int], image_width: int, image_height: int) -> YoloAnnotation:
    """Convert a NeutralAnnotation to YOLO format.
    
    Args:
        annotation (NeutralAnnotation): Source neutral annotation
        inverse_class_map (dict[str, int]): Mapping from class names to class IDs
        image_width (int): Width of the associated image
        image_height (int): Height of the associated image
        
    Returns:
        YoloAnnotation: Converted YOLO annotation
        
    Raises:
        ValueError: If class name doesn't exist in inverse_class_map
    """

    bbox: YoloBoundingBox = PascalVocBBox_to_YoloBBox(annotation.geometry, image_width, image_height)

    try:
        id_class = inverse_class_list[annotation.class_name]
    except KeyError:
        raise ValueError(f"Class '{annotation.class_name}' not found in class map") from None

    return YoloAnnotation(
        bbox=bbox,
        id_class=id_class
    )