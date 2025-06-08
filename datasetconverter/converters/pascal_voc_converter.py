from pathlib import Path
from .dataset_converter import DatasetConverter
from ..formats.neutral_format import ImageOrigin, NeutralAnnotation, NeutralFile, NeutralFormat
from ..formats.pascal_voc import PascalVocFile, PascalVocFormat, PascalVocObject, PascalVocSource


class PascalVocConverter(DatasetConverter[PascalVocFormat]):
    """Converter between PascalVocFormat and NeutralFormat
    
    Provides bidirectional conversion capabilities between PascalVocFormat and NeutralFormat
    """
    
    @staticmethod
    def toNeutral(df: PascalVocFormat) -> NeutralFormat:
        """Convert a Pascal Voc dataset to the Neutral format

        Args:
            df (PascalVocFormat): object that represents a Pascal Voc dataset to convert

        Returns:
            NeutralFormat: object with the converted dataset
        """
        neutral_files: list[NeutralFile] = [PascalFile_to_NeutralFile(i) for i in df.files]

        # Unique class list (first creating a set(unique elements) and then a list)
        unique_classes = list({
            obj.name
            for file in df.files 
            for obj in file.annotations
        }) 

        class_map = {i: class_name for i, class_name in enumerate(unique_classes)}

        return NeutralFormat(
                name = df.name, 
                files = neutral_files,
                original_format = "pascal_voc",
                metadata = None,
                class_map = class_map
            )
    

    @staticmethod
    def fromNeutral(nf: NeutralFormat) -> PascalVocFormat:
        """Convert the Neutral dataset to the Pascal Voc format.

        Args:
            nf (NeutralFormat): object that represents a Neutral dataset to convert

        Returns:
            PascalVocFormat: object with the converted dataset
        """
        pascal_files: list[PascalVocFile] = [NeutralFile_to_PascalFile(i, nf.name) for i in nf.files]

        return PascalVocFormat(
            name = nf.name,
            files = pascal_files
        )


def PascalFile_to_NeutralFile(file: PascalVocFile) -> NeutralFile:
    """Convert a Pascal Voc file to the Neutral format

    Args:
        file (PascalVocFile): object that represents a Pascal Voc file to convert

    Returns:
        NeutralFile: object with the converted file
    """
    neutral_annotations: list[NeutralAnnotation] = [PascalAnnotation_to_NeutralAnnotation(i) for i in file.annotations] 

    image_origin = ImageOrigin(
        source_type = [file.source.image] if file.source.image else None,
        source_dataset = file.source.database if file.source.database else None,
        extension = Path(file.filename).suffix
    )

    return NeutralFile(
        filename = Path(file.filename).stem,
        annotations = neutral_annotations,
        width = file.width,
        height = file.height,
        depth = file.depth,
        image_origin = image_origin,
        params = {
            "segmented": file.segmented
        }
    )


def PascalAnnotation_to_NeutralAnnotation(annotation: PascalVocObject) -> NeutralAnnotation:
    """Convert a Pascal Voc annotation to the Neutral format

    Args:
        annotation (PascalVocObject): object that represents a Pascal Voc object to convert

    Returns:
        NeutralAnnotation: object with the converted annotation
    """
    return NeutralAnnotation(
        bbox = annotation.geometry,
        class_name = annotation.name,
        attributes = {
            "pose": annotation.pose,
            "truncated": annotation.truncated,
            "difficult": annotation.difficult
        }
    )



def NeutralFile_to_PascalFile(file: NeutralFile, original_dataset_name: str) -> PascalVocFile:
    """Converts a NeutralFile to a Pascal Voc format.

    Args:
        file (NeutralFile): object with the file to convert
        original_dataset_name (str): name of the original dataset

    Returns:
        PascalVocFile: object with the converted file
    """
    pascal_annotations: list[PascalVocObject] = [NeutralAnnotation_to_PascalAnnotation(i) for i in file.annotations]


    source: PascalVocSource = PascalVocSource(
            database = file.image_origin.source_dataset if file.image_origin.source_dataset else original_dataset_name,
            annotation = "Pascal Voc", # annotation standard
            image = file.image_origin.source_type[0] if file.image_origin.source_type else ""
    )


    return PascalVocFile(
        filename = file.filename + file.image_origin.extension,
        annotations = pascal_annotations,
        folder = "Annotations",
        path = "Annotations/" + file.filename + file.image_origin.extension,
        source = source,
        width = file.width,
        height = file.height,
        depth = file.depth,
        segmented = file.params.get("segmented", 0)
    )


def NeutralAnnotation_to_PascalAnnotation(annotation: NeutralAnnotation) -> PascalVocObject:
    """Converts a NeutralAnnotation to a Pascal Voc format.

    Args:
        annotation (NeutralAnnotation): object with the annotation to convert

    Returns:
        PascalVocObject: object with the converted annotation
    """
    pose = annotation.attributes.get("pose", "")

    truncated = annotation.attributes.get("truncated", False)

    difficult = annotation.attributes.get("difficult", False)

    return PascalVocObject(
        bbox = annotation.geometry,
        name = annotation.class_name,
        pose = pose,  
        truncated = truncated,  
        difficult = difficult  
    )