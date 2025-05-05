from pathlib import Path
from typing import Any, Optional
import xml.etree.ElementTree as ET

from .base import Annotation, BoundingBox, DatasetFormat, FileFormat

class PascalVocBoundingBox(BoundingBox):
    x_min: int
    y_min: int
    x_max: int
    y_max: int

    def __init__(self, x_min: int,  y_min: int, x_max: int, y_max: int) -> None:
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max

    def getBoundingBox(self):
        return [self.x_min, self.y_min, self.x_max,  self.y_max]


class PascalVocObject(Annotation[PascalVocBoundingBox]):
    name: str
    pose: str
    truncated: bool
    difficult: bool

    def __init__(self, bbox: PascalVocBoundingBox, name: str, pose: str, truncated: bool, difficult: bool) -> None:
        super().__init__(bbox)
        self.name = name
        self.pose = pose
        self.truncated = truncated
        self.difficult = difficult


class PascalVocSource:
    database: str
    annotation: str
    image: str

    def __init__(self, database: str = "", annotation: str = "", image: str = "") -> None:
        self.database = database
        self.annotation = annotation
        self.image = image

    def to_dict(self) -> dict[str, Any]:
        return {
            "database": self.database,
            "annotation": self.annotation,
            "image": self.image
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PascalVocSource":
        return cls(
            database=data.get("database", ""),
            annotation=data.get("annotation", ""),
            image=data.get("image", "")
        )


class PascalVocFile(FileFormat[PascalVocObject]):
    folder: str
    path: str
    source: PascalVocSource 

    # size tag
    width: int
    height: int
    depth: int

    segmented: int

    def __init__(self, filename: str, annotations: list[PascalVocObject], folder: str, path: str, source: PascalVocSource, width: int, height: int, depth: int, segmented: int) -> None:
        super().__init__(filename, annotations)
        self.folder = folder
        self.path = path
        self.source = source
        self.width = width
        self.height = height
        self.depth = depth
        self.segmented = segmented


class PascalVocFormat(DatasetFormat[PascalVocFile]):

    def __init__(self, name: str, files: list[PascalVocFile], folder_path: Optional[str] = None) -> None:
        super().__init__(name, files, folder_path)

    @staticmethod
    def build(name: str, files: list[PascalVocFile], folder_path: Optional[str] = None) -> 'PascalVocFormat':
        return PascalVocFormat(name, files, folder_path)

    @staticmethod
    def read_from_folder(folder_path: str) -> 'PascalVocFormat':
        """
        Create a dataset in Pascal Voc format from folder.

        A standar Pascal Voc format consist of:
        - A images folder
        - A folder with text files that have the different sets of images for training
        - XML files with annotations in an 'annotations' folder

        Args:
            folder_path (str): Path to the folder

        Returns:
            PascalVocFormat: Object with the Pascal Voc dataset
        """
        if not Path(folder_path).exists():
            raise FileNotFoundError(f"Folder {folder_path} was not found")

        annotations_folder = Path(folder_path) / "Annotations"
        if not Path(annotations_folder).exists():
            raise FileNotFoundError(f"Subfolder Annotations was not found in {annotations_folder}")

        pascal_files = []

        for xml_file in annotations_folder.glob("*.xml"):
            tree = ET.parse(xml_file)
            root = tree.getroot()

            # Read file metadata
            folder_tag = root.findtext('folder', default="")
            path_tag = root.findtext('path', default="")
            size_tag = root.find('size')
            width = int(size_tag.findtext('width', default="0")) if size_tag is not None else 0
            height = int(size_tag.findtext('height', default="0")) if size_tag is not None else 0
            depth = int(size_tag.findtext('depth', default="0")) if size_tag is not None else 0
            segmented = int(root.findtext('segmented', default="0"))

            # Read source tag
            source_tag = root.find('source')
            if source_tag is not None:
                source = PascalVocSource(
                    database=source_tag.findtext('database', default=""),
                    annotation=source_tag.findtext('annotation', default=""),
                    image=source_tag.findtext('image', default="")
                )
            else:
                source = PascalVocSource()  # Empty instance

            # Read annotation objects
            annotations = []
            for obj in root.findall('object'):
                name = obj.findtext('name', default="")
                pose = obj.findtext('pose', default="")
                truncated = bool(int(obj.findtext('truncated', default="0")))
                difficult = bool(int(obj.findtext('difficult', default="0")))
                bndbox = obj.find('bndbox')
                if bndbox is not None:
                    x_min = int(bndbox.findtext('xmin', default="0"))
                    y_min = int(bndbox.findtext('ymin', default="0"))
                    x_max = int(bndbox.findtext('xmax', default="0"))
                    y_max = int(bndbox.findtext('ymax', default="0"))
                    bbox = PascalVocBoundingBox(x_min, y_min, x_max, y_max)
                    annotations.append(PascalVocObject(bbox, name, pose, truncated, difficult))

            pascal_files.append(
                PascalVocFile(
                    filename=xml_file.name,
                    annotations=annotations,
                    folder=folder_tag,
                    path=path_tag,
                    source=source,
                    width=width,
                    height=height,
                    depth=depth,
                    segmented=segmented
                )
            )

        return PascalVocFormat.build(
            name=Path(folder_path).name,
            files=pascal_files,
            folder_path=folder_path
        )