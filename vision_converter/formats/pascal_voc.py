from pathlib import Path
from typing import Optional
import xml.etree.ElementTree as ET

from vision_converter.utils.file_utils import find_all_images_folders

from .base import Annotation, BoundingBox, DatasetFormat, FileFormat

class PascalVocBoundingBox(BoundingBox):
    """Bounding box in Pascal VOC format (absolute pixel coordinates).

    Attributes:
        x_min (int): Minimum x (left).
        y_min (int): Minimum y (top).
        x_max (int): Maximum x (right).
        y_max (int): Maximum y (bottom).
    """
    x_min: int
    y_min: int
    x_max: int
    y_max: int

    def __init__(self, x_min: int,  y_min: int, x_max: int, y_max: int) -> None:
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max

    def __eq__(self, other):
        """Compare two PascalVocBoundingBox objects for equality."""
        if not isinstance(other, PascalVocBoundingBox):
            return NotImplemented
        return (self.x_min == other.x_min and 
                self.y_min == other.y_min and 
                self.x_max == other.x_max and 
                self.y_max == other.y_max)

    def getBoundingBox(self):
        """Returns Pascal Voc coordinates as [x_min, y_min, x_max, y_max]."""
        return [self.x_min, self.y_min, self.x_max,  self.y_max]



class PascalVocObject(Annotation[PascalVocBoundingBox]):
    """Annotation for a single Pascal VOC Object, with bounding box and metadata.

    Attributes:
        name (str): Class name of the object.
        pose (str): Pose label (e.g., 'Unspecified').
        truncated (bool): True if object is truncated in the image.
        difficult (bool): True if object is difficult to detect.
        bbox (PascalVocBoundingBox): Inherited. Bounding box of the object.
    """
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
    """Metadata for the 'source' tag in Pascal VOC XML files.

    Attributes:
        database (str): Name of the database.
        annotation (str): Annotation type.
        image (str): Image source.
    """
    database: str
    annotation: str
    image: str

    def __init__(self, database: str = "", annotation: str = "", image: str = "") -> None:
        self.database = database
        self.annotation = annotation
        self.image = image


class PascalVocFile(FileFormat[PascalVocObject]):
    """Represents a Pascal VOC annotated file with metadata and object annotations.

    Attributes:
        folder (str): Name of the folder containing the image.
        path (str): Path to the image file.
        source (PascalVocSource): Metadata about the data source.
        width (int): Image width in pixels.
        height (int): Image height in pixels.
        depth (int): Number of color channels (e.g., 3 for RGB).
        segmented (int): Segmentation flag (0 or 1).
        filename (str): Inherited. Name of the image file.
        annotations (list[PascalVocObject]): Inherited. List of object annotations.
    """
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
    """Dataset in Pascal VOC format, including files and folder structure.

    Attributes:
        name (str): Inherited. Name of the dataset.
        files (list[PascalVocFile]): Inherited. List of PascalVocFile objects.
        folder_path (Optional[str]): Inherited. Path to the dataset folder.
        images_path_list (Optional[list[str]]): Inherited - List of images paths
    """

    def __init__(self, name: str, files: list[PascalVocFile], folder_path: Optional[str] = None, images_path_list: Optional[list[str]] = None) -> None:
        super().__init__(name, files, folder_path, images_path_list)

    @staticmethod
    def build(name: str, files: list[PascalVocFile], folder_path: Optional[str] = None, images_path_list: Optional[list[str]] = None) -> 'PascalVocFormat':
        return PascalVocFormat(name, files, folder_path, images_path_list)

    @staticmethod
    def read_from_folder(folder_path: str, copy_images: bool = False, copy_as_links: bool = False) -> 'PascalVocFormat':
        """Create a dataset in Pascal VOC format from a folder.

        Expecting annotations in Annotations folder.

        Args:
            folder_path (str): Path to the Pascal VOC dataset root folder.
            copy_images (bool, default False): If True, loads and stores the image file paths in the dataset object; if False, image paths are not loaded.
            copy_as_links (bool, default False): If True, loads and stores the image file paths in the dataset object; if False, image paths are not loaded.

        Returns:
            PascalVocFormat: Object with the Pascal VOC dataset.

        Raises:
            FileNotFoundError: If the required folders or files are missing.
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
            filename = root.findtext('filename', default="")
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
                    filename=filename,
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

        # Save images path
        image_paths = []
        if copy_images or copy_as_links:
            # Search for images folders
            list_images_dir = find_all_images_folders(folder_path)
            for images_dir in list_images_dir:
                image_paths += PascalVocFormat.get_image_paths(images_dir)

        return PascalVocFormat.build(
            name=Path(folder_path).name,
            files=pascal_files,
            folder_path=folder_path,
            images_path_list=image_paths  if len(image_paths) > 0 else None
        )


    def save(self, folder: str, copy_images: bool = False, copy_as_links: bool = False) -> None:
        """Save the Pascal VOC dataset to the specified folder, creating the standard structure.

        The following structure will be created:
        ```
        {folder}/
            ├── Annotations/    # XML annotation files
            ├── JPEGImages/     # Image files
            └── ImageSets/      # Image set text files (not written here)
        ```

        Args:
            folder (str): Output directory path.
            copy_images (bool, default False): If True, copies image files to the output directory. If False, images are not copied.
            copy_as_links (bool, default False): If True, creates links to the original images in the output directory instead of copying them. If False, no links are created.
        """
        folder_path = Path(folder)
        
        # Create any folder if necesary
        folder_path.mkdir(parents=True, exist_ok=True)
        
        # Create folder structure for PascalVoc
        annotations_dir = folder_path / "Annotations"
        imagesets_dir = folder_path / "ImageSets"
        images_dir = folder_path / "JPEGImages"
        
        annotations_dir.mkdir(exist_ok=True)
        imagesets_dir.mkdir(exist_ok=True)
        images_dir.mkdir(exist_ok=True)
        
        
        # Save all XML Annotations files
        for file in self.files:
            
            root = ET.Element("annotation")
            
            # Basic metadata
            ET.SubElement(root, "folder").text = file.folder
            ET.SubElement(root, "filename").text = file.filename
            ET.SubElement(root, "path").text = file.path
            
            # Source tag
            source = ET.SubElement(root, "source")
            ET.SubElement(source, "database").text = file.source.database
            ET.SubElement(source, "annotation").text = file.source.annotation
            ET.SubElement(source, "image").text = file.source.image
            
            # Size tag
            size = ET.SubElement(root, "size")
            ET.SubElement(size, "width").text = str(file.width)
            ET.SubElement(size, "height").text = str(file.height)
            ET.SubElement(size, "depth").text = str(file.depth)
            
            # Segmentation tag
            ET.SubElement(root, "segmented").text = str(file.segmented)
            
            # Pascal Voc objects
            for obj in file.annotations:
                obj_elem = ET.SubElement(root, "object")
                ET.SubElement(obj_elem, "name").text = obj.name
                ET.SubElement(obj_elem, "pose").text = obj.pose
                ET.SubElement(obj_elem, "truncated").text = str(int(obj.truncated))
                ET.SubElement(obj_elem, "difficult").text = str(int(obj.difficult))
                
                bndbox = ET.SubElement(obj_elem, "bndbox")
                ET.SubElement(bndbox, "xmin").text = str(obj.geometry.x_min)
                ET.SubElement(bndbox, "ymin").text = str(obj.geometry.y_min)
                ET.SubElement(bndbox, "xmax").text = str(obj.geometry.x_max)
                ET.SubElement(bndbox, "ymax").text = str(obj.geometry.y_max)
            
            filename = Path(file.filename).stem + ".xml"
            xml_path = annotations_dir / filename

            # Save XML
            tree = ET.ElementTree(root)
            tree.write(xml_path, encoding="utf-8", xml_declaration=True)

        if copy_images or copy_as_links:
            self.handle_images(self.images_path_list, str(images_dir), copy_images, copy_as_links)
