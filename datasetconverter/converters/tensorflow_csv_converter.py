from pathlib import Path
from typing import Optional

from datasetconverter.utils.file_utils import get_image_info_from_file, get_image_path

from .dataset_converter import DatasetConverter
from ..formats.neutral_format import ImageOrigin, NeutralAnnotation, NeutralFile, NeutralFormat
from ..formats.tensorflow_csv import TensorflowCsvAnnotation, TensorflowCsvFile, TensorflowCsvFormat


class TensorflowCsvConverter(DatasetConverter[TensorflowCsvFormat]):
    """Converter between TensorflowCsvFormat and NeutralFormat
    
    Provides bidirectional conversion capabilities between TensorflowCsvFormat and NeutralFormat
    """
    
    @staticmethod
    def toNeutral(df: TensorflowCsvFormat) -> NeutralFormat:
        """Convert a TensorflowCsvFormat dataset to Neutral format.
        
        Args:
            df (TensorflowCsvFormat): Source dataset in TensorFlow CSV format
            
        Returns:
            NeutralFormat: Converted dataset in neutral annotation format
        """

        neutral_files: list[NeutralFile] = [TensorflowCsvFile_to_NeutralFile(i, df.folder_path) for i in df.files]

        # Generate class_map from unique classes (TensorFlow CSV doesn't have explicit class mapping)
        unique_classes = df.get_unique_classes()
        class_map = {i: class_name for i, class_name in enumerate(sorted(unique_classes))}

        return NeutralFormat(
            name = df.name,
            files = neutral_files,
            original_format = "tensorflow_csv",
            metadata = None,
            class_map = class_map
        )
    
    @staticmethod
    def fromNeutral(nf: NeutralFormat) -> TensorflowCsvFormat:
        """Convert a NeutralFormat dataset to TensorflowCsvFormat.
        
        Args:
            nf (NeutralFormat): Source dataset in neutral format
            
        Returns:
            TensorflowCsvFormat: Converted dataset in TensorFlow CSV format
        """

        tf_files: list[TensorflowCsvFile] = [NeutralFile_to_TensorflowCsvFile(i) for i in nf.files]

        return TensorflowCsvFormat(
            name = nf.name,
            files = tf_files,
            folder_path = nf.folder_path
        )


def TensorflowCsvFile_to_NeutralFile(file: TensorflowCsvFile, folder_path: Optional[str] = None) -> NeutralFile:
    """Convert a TensorflowCsvFile representation to Neutral format.
    
    Args:
        file (TensorflowCsvFile): TensorFlow CSV file object to convert
        folder_path (Optional[str]): Optional path path to the folder containing the original dataset
        
    Returns:
        NeutralFile: Converted file representation
    """

    # Depth info not stored in TensorFlow CSV format, default 3
    image_depth = 3
    # If the format was constructed from a folder, get image depth from the image
    if folder_path:
        image_path = get_image_path(folder_path, "images", file.filename)
        if image_path:
            _, _, image_depth = get_image_info_from_file(image_path)


    neutral_annotations: list[NeutralAnnotation] = [TensorflowCsvAnnotation_to_NeutralAnnotation(i) for i in file.annotations]

    image_origin = ImageOrigin(extension = Path(file.filename).suffix)

    return NeutralFile(
        filename = Path(file.filename).stem,
        annotations = neutral_annotations,
        width = file.width,
        height = file.height,
        depth = image_depth,  
        image_origin = image_origin
    )


def TensorflowCsvAnnotation_to_NeutralAnnotation(annotation: TensorflowCsvAnnotation) -> NeutralAnnotation:
    """Convert a TensorflowCsvAnnotation to Neutral format.
    
    Args:
        annotation (TensorflowCsvAnnotation): Source TensorFlow CSV annotation
        
    Returns:
        NeutralAnnotation: Converted annotation
    """

    # No conversion needed - both use PascalVocBoundingBox and class_name
    return NeutralAnnotation(annotation.bbox, annotation.class_name)


def NeutralFile_to_TensorflowCsvFile(file: NeutralFile) -> TensorflowCsvFile:
    """Convert a NeutralFile to TensorflowCsvFile format.
    
    Args:
        file (NeutralFile): Neutral format file to convert
        
    Returns:
        TensorflowCsvFile: Converted TensorFlow CSV file
    """

    tf_annotations: list[TensorflowCsvAnnotation] = [NeutralAnnotation_to_TensorflowCsvAnnotation(i) for i in file.annotations]

    return TensorflowCsvFile(
        filename = file.filename + file.image_origin.extension,
        annotations = tf_annotations,
        width = file.width,
        height = file.height
    )


def NeutralAnnotation_to_TensorflowCsvAnnotation(annotation: NeutralAnnotation) -> TensorflowCsvAnnotation:
    """Convert a NeutralAnnotation to TensorflowCsvAnnotation format.
    
    Args:
        annotation (NeutralAnnotation): Source neutral annotation
        
    Returns:
        TensorflowCsvAnnotation: Converted TensorFlow CSV annotation
    """

    # No conversion needed - both use PascalVocBoundingBox and class_name
    return TensorflowCsvAnnotation(bbox = annotation.bbox, class_name = annotation.class_name)
