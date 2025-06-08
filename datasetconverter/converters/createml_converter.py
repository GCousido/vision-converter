from pathlib import Path
from typing import Optional

from .dataset_converter import DatasetConverter
from ..formats.neutral_format import ImageOrigin, NeutralAnnotation, NeutralFile, NeutralFormat
from ..formats.pascal_voc import PascalVocBoundingBox
from ..formats.createml import CreateMLAnnotation, CreateMLBoundingBox, CreateMLFile, CreateMLFormat
from ..utils.bbox_utils import PascalVocBBox_to_CreateMLBBox, CreateMLBBox_to_PascalVocBBox
from ..utils.file_utils import get_image_path, get_image_info_from_file


class CreateMLConverter(DatasetConverter[CreateMLFormat]):
    """Converter between CreateMLFormat and NeutralFormat
    
    Provides bidirectional conversion capabilities between CreateMLFormat and NeutralFormat
    """
    
    @staticmethod
    def toNeutral(df: CreateMLFormat) -> NeutralFormat:
        """Convert a CreateMLFormat dataset to Neutral format.
        
        Args:
            df (CreateMLFormat): Source dataset in CreateML format
            
        Returns:
            NeutralFormat: Converted dataset in neutral annotation format
            
        Raises:
            FileNotFoundError: If image files are missing when constructed from folder
            ValueError: If image dimensions are missing in CreateMLFile when not constructed from folder
        """

        neutral_files: list[NeutralFile] = [CreateMLFile_to_NeutralFile(i, df.folder_path) for i in df.files]

        # Extract unique class names from all annotations
        class_names = set()
        for file in df.files:
            for annotation in file.annotations:
                class_names.add(annotation.label)
        
        # Create class map with sorted class names for consistency
        sorted_classes = sorted(class_names)
        class_map = {i: class_name for i, class_name in enumerate(sorted_classes)}

        return NeutralFormat(
                name = df.name, 
                files = neutral_files,
                original_format = "createml",
                class_map = class_map
            )
    
    
    @staticmethod
    def fromNeutral(nf: NeutralFormat) -> CreateMLFormat:
        """Convert a NeutralFormat dataset to CreateML format.
        
        Args:
            nf (NeutralFormat): Source dataset in neutral format
            
        Returns:
            CreateMLFormat: Converted dataset in CreateML format
        """

        createml_files: list[CreateMLFile] = [NeutralFile_to_CreateMLFile(i) for i in nf.files]

        return CreateMLFormat(
                name = nf.name,
                files = createml_files
            )



def CreateMLFile_to_NeutralFile(file: CreateMLFile, folder_path: Optional[str] = None) -> NeutralFile:
    """Convert a CreateMLFile representation to Neutral format.
    
    Args:
        file (CreateMLFile): CreateML file object to convert
        folder_path (Optional[str]): Optional path to image folder for dimension validation
        
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
            raise ValueError("Missing image dimensions in CreateMLFile when dataset was not created from folder")
    
        image_width = file.width
        image_height = file.height
        image_depth = file.depth

    neutral_annotations: list[NeutralAnnotation] = [
        CreateMLAnnotation_to_NeutralAnnotation(i)
        for i in file.annotations
    ]

    image_origin = ImageOrigin(
        extension = Path(file.filename).suffix
    )

    return NeutralFile(
        filename = Path(file.filename).stem,
        annotations = neutral_annotations,
        width = image_width,
        height = image_height,
        depth = image_depth,
        image_origin = image_origin
    )


def CreateMLAnnotation_to_NeutralAnnotation(annotation: CreateMLAnnotation) -> NeutralAnnotation:
    """Convert a CreateMLAnnotation to Neutral format.
    
    Args:
        annotation (CreateMLAnnotation): Source CreateML annotation
        
    Returns:
        NeutralAnnotation: Converted annotation
    """
    
    bbox: PascalVocBoundingBox = CreateMLBBox_to_PascalVocBBox(annotation.bbox)

    return NeutralAnnotation(
        bbox = bbox, 
        class_name = annotation.label
    )



def NeutralFile_to_CreateMLFile(file: NeutralFile) -> CreateMLFile:
    """Convert a NeutralFile to CreateML format.
    
    Args:
        file (NeutralFile): Neutral format file to convert
        
    Returns:
        CreateMLFile: Converted CreateML file
    """

    createml_annotations: list[CreateMLAnnotation] = [
        NeutralAnnotation_to_CreateMLAnnotation(i) for i in file.annotations
    ]

    return CreateMLFile(
        filename = file.filename + file.image_origin.extension,
        annotations = createml_annotations,
        width = file.width,
        height = file.height,
        depth = file.depth
    )


def NeutralAnnotation_to_CreateMLAnnotation(annotation: NeutralAnnotation) -> CreateMLAnnotation:
    """Convert a NeutralAnnotation to CreateML format.
    
    Args:
        annotation (NeutralAnnotation): Source neutral annotation
        
    Returns:
        CreateMLAnnotation: Converted CreateML annotation
    """

    bbox: CreateMLBoundingBox = PascalVocBBox_to_CreateMLBBox(annotation.bbox)

    return CreateMLAnnotation(
        bbox = bbox,
        label = annotation.class_name
    )
