from pathlib import Path
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


class PascalVocFile(FileFormat[PascalVocObject]):
    folder: str
    # filename: str - en principio está en el base
    path: str

    # forman la etiqueta de size
    width: int
    height: int
    depth: int

    segmented: int

    def __init__(self, filename: str, annotations: list[PascalVocObject], folder: str, path: str, width: int, height: int, depth: int, segmented: int) -> None:
        super().__init__(filename, annotations)
        self.folder = folder
        self.path = path
        self.width = width
        self.height = height
        self.depth = depth
        self.segmented = segmented


class PascalVocFormat(DatasetFormat[PascalVocFile]):

    def __init__(self, name: str, files: list[PascalVocFile]) -> None:
        super().__init__(name, files)

    @staticmethod
    def build(name: str, files: list[PascalVocFile]) -> 'PascalVocFormat':
        """Construye un objeto PascalVocFormat con parámetros específicos"""
        return PascalVocFormat(name, files)

    @staticmethod
    def read_from_folder(folder_path: str) -> 'PascalVocFormat':
        """
        Lee archivos Pascal VOC XML desde una carpeta y construye el formato.
        """
        if not Path(folder_path).exists():
            raise FileNotFoundError(f"La carpeta {folder_path} no se encontró")

        annotations_folder = Path(folder_path) / "Annotations"
        if not Path(annotations_folder).exists():
            raise FileNotFoundError(f"La subcarpeta Annotations no se encontró en {annotations_folder}")

        pascal_files = []

        for xml_file in annotations_folder.glob("*.xml"):
            tree = ET.parse(xml_file)
            root = tree.getroot()

            # Leer metadatos del archivo
            folder_tag = root.findtext('folder', default="")
            path_tag = root.findtext('path', default="")
            size_tag = root.find('size')
            width = int(size_tag.findtext('width', default="0")) if size_tag is not None else 0
            height = int(size_tag.findtext('height', default="0")) if size_tag is not None else 0
            depth = int(size_tag.findtext('depth', default="0")) if size_tag is not None else 0
            segmented = int(root.findtext('segmented', default="0"))

            # Leer objetos anotados
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
                    width=width,
                    height=height,
                    depth=depth,
                    segmented=segmented
                )
            )

        return PascalVocFormat.build(
            name=Path(folder_path).name,
            files=pascal_files
        )