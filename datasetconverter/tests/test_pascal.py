import os
from pathlib import Path
import pytest
import xml.etree.ElementTree as ET
from datasetconverter.formats.pascal_voc import PascalVocBoundingBox, PascalVocFile, PascalVocFormat, PascalVocObject, PascalVocSource
from datasetconverter.tests.utils_for_tests import normalize_path

# Helper for creating XML Pascal VOC files
def create_pascalvoc_xml(path, filename, objects, width=800, height=600, depth=3, folder="VOC", segmented=0, source=None):
    if source is None:
        source = {
            "database": "Unknown",
            "annotation": "",
            "image": ""
        }
    source_xml = f'''
    <source>
        <database>{source.get("database", "")}</database>
        <annotation>{source.get("annotation", "")}</annotation>
        <image>{source.get("image", "")}</image>
    </source>
    '''
    content = f'''<annotation>
    <folder>{folder}</folder>
    <filename>{filename}</filename>
    <path>{path / filename}</path>
    {source_xml}
    <size>
        <width>{width}</width>
        <height>{height}</height>
        <depth>{depth}</depth>
    </size>
    <segmented>{segmented}</segmented>
    {''.join(objects)}
</annotation>'''
    return content

def make_object(name, xmin, ymin, xmax, ymax, pose="Unspecified", truncated=0, difficult=0):
    return f'''
    <object>
        <name>{name}</name>
        <pose>{pose}</pose>
        <truncated>{truncated}</truncated>
        <difficult>{difficult}</difficult>
        <bndbox>
            <xmin>{xmin}</xmin>
            <ymin>{ymin}</ymin>
            <xmax>{xmax}</xmax>
            <ymax>{ymax}</ymax>
        </bndbox>
    </object>'''

@pytest.fixture
def sample_pascalvoc_dataset(tmp_path):
    ann_dir = tmp_path / "Annotations"
    ann_dir.mkdir()

    # File 1 with 2 objects, custom source
    xml1 = create_pascalvoc_xml(
        ann_dir, "image1.xml",
        [
            make_object("person", 10, 20, 110, 220, pose="Left", truncated=0, difficult=0),
            make_object("car", 30, 40, 130, 240, pose="Right", truncated=1, difficult=1)
        ],
        width=800, height=600, depth=3,
        source={"database": "TestDB", "annotation": "VOC2007", "image": "synthetic"}
    )
    (ann_dir / "image1.xml").write_text(xml1)

    # File 2 with 1 object, default source
    xml2 = create_pascalvoc_xml(
        ann_dir, "image2.xml",
        [
            make_object("dog", 50, 60, 150, 260, pose="Unspecified", truncated=0, difficult=0)
        ],
        width=1024, height=768, depth=3
    )
    (ann_dir / "image2.xml").write_text(xml2)
    return tmp_path

def test_pascalvoc_format_construction(sample_pascalvoc_dataset):
    pascalvoc_format = PascalVocFormat.read_from_folder(sample_pascalvoc_dataset)

    # 1. Checking normal structure
    assert pascalvoc_format.folder_path == sample_pascalvoc_dataset
    assert pascalvoc_format.name == sample_pascalvoc_dataset.name
    assert isinstance(pascalvoc_format.files, list)

    # 2. Checking files
    assert len(pascalvoc_format.files) == 2
    filenames = {f.filename for f in pascalvoc_format.files}
    assert "image1.xml" in filenames
    assert "image2.xml" in filenames

    # 3. Checking bounding boxes, objects, and source
    file1 = next(f for f in pascalvoc_format.files if f.filename == "image1.xml")
    assert file1.width == 800
    assert file1.height == 600
    assert file1.depth == 3
    assert len(file1.annotations) == 2

    # Check source fields for file1 (custom)
    assert isinstance(file1.source, PascalVocSource)
    assert file1.source.database == "TestDB"
    assert file1.source.annotation == "VOC2007"
    assert file1.source.image == "synthetic"

    ann1 = file1.annotations[0]
    assert ann1.name == "person"
    assert ann1.pose == "Left"
    assert ann1.truncated is False
    assert ann1.difficult is False
    assert ann1.geometry.x_min == 10
    assert ann1.geometry.y_max == 220

    ann2 = file1.annotations[1]
    assert ann2.name == "car"
    assert ann2.pose == "Right"
    assert ann2.truncated is True
    assert ann2.difficult is True
    assert ann2.geometry.x_max == 130
    assert ann2.geometry.y_min == 40

    # Checking second file
    file2 = next(f for f in pascalvoc_format.files if f.filename == "image2.xml")
    assert file2.width == 1024
    assert file2.height == 768
    assert file2.depth == 3
    assert len(file2.annotations) == 1

    # Check source fields for file2 (default)
    assert isinstance(file2.source, PascalVocSource)
    assert file2.source.database == "Unknown"
    assert file2.source.annotation == ""
    assert file2.source.image == ""

    ann3 = file2.annotations[0]
    assert ann3.name == "dog"
    assert ann3.pose == "Unspecified"
    assert ann3.truncated is False
    assert ann3.difficult is False
    assert ann3.geometry.x_min == 50
    assert ann3.geometry.x_max == 150

def test_invalid_pascalvoc_structure(tmp_path):
    # Case with a path that doesnt exist
    with pytest.raises(FileNotFoundError):
        PascalVocFormat.read_from_folder("incorrect_path")

    # Case without an Annotations folder
    with pytest.raises(FileNotFoundError):
        PascalVocFormat.read_from_folder(tmp_path)

    ann_dir = tmp_path / "Annotations"
    ann_dir.mkdir()

    # Case with an Annotations folder
    try:
        PascalVocFormat.read_from_folder(tmp_path)
    except Exception as exc:
        assert False, f"Excepcion: {exc}"


def test_pascalvoc_format_save_full_no_empty_annotations(tmp_path):
    # Prepare test data
    # ================================================
    source = PascalVocSource(database="MyDB", annotation="PascalVOC", image="flickr")
    
    # File 1: 2 objects
    bbox1 = PascalVocBoundingBox(10, 20, 50, 60)
    bbox2 = PascalVocBoundingBox(15, 25, 55, 65)
    obj1 = PascalVocObject(bbox1, name="cat", pose="Unspecified", truncated=False, difficult=False)
    obj2 = PascalVocObject(bbox2, name="dog", pose="Left", truncated=True, difficult=True)
    
    # File 2: 2 objects
    bbox3 = PascalVocBoundingBox(5, 15, 25, 35)
    bbox4 = PascalVocBoundingBox(30, 40, 70, 80)
    obj3 = PascalVocObject(bbox3, name="bird", pose="Back", truncated=False, difficult=True)
    obj4 = PascalVocObject(bbox4, name="horse", pose="Frontal", truncated=False, difficult=False)
    
    # File 3: 1 objects
    bbox5 = PascalVocBoundingBox(100, 120, 150, 160)
    obj5 = PascalVocObject(bbox5, name="sheep", pose="Right", truncated=True, difficult=False)

    pascal_files = [
        PascalVocFile(
            filename="image1.jpg",
            annotations=[obj1, obj2],
            folder="Images",
            path="/test/path/JPEGImages/image1.jpg",
            source=source,
            width=100,
            height=200,
            depth=3,
            segmented=0
        ),
        PascalVocFile(
            filename="image2.jpg",
            annotations=[obj3, obj4],
            folder="Images",
            path="/test/path/JPEGImages/image2.jpg",
            source=source,
            width=80,
            height=120,
            depth=3,
            segmented=1
        ),
        PascalVocFile(
            filename="image3.jpg",
            annotations=[obj5],
            folder="Images",
            path="/test/path/JPEGImages/image3.jpg",
            source=source,
            width=640,
            height=480,
            depth=3,
            segmented=0
        )
    ]

    pascal_dataset = PascalVocFormat(
        name="test_dataset",
        files=pascal_files,
        folder_path=None
    )

    # Execute save
    pascal_dataset.save(str(tmp_path))

    # Check file structure
    assert (tmp_path / "Annotations").is_dir()
    assert (tmp_path / "ImageSets").is_dir()
    assert (tmp_path / "JPEGImages").is_dir()

    # Check xml files
    verify_xml_content(
        tmp_path / "Annotations" / "image1.xml",
        expected_objects=2,
        expected_size=(100, 200, 3),
        expected_objects_data=[
            {"name": "cat", "xmin": 10, "ymin": 20, "xmax": 50, "ymax": 60, "truncated": "0"},
            {"name": "dog", "xmin": 15, "ymin": 25, "xmax": 55, "ymax": 65, "truncated": "1"}
        ]
    )

    verify_xml_content(
        tmp_path / "Annotations" / "image2.xml",
        expected_objects=2,
        expected_size=(80, 120, 3),
        expected_objects_data=[
            {"name": "bird", "xmin": 5, "ymin": 15, "xmax": 25, "ymax": 35, "difficult": "1"},
            {"name": "horse", "xmin": 30, "ymin": 40, "xmax": 70, "ymax": 80, "pose": "Frontal"}
        ]
    )

    verify_xml_content(
        tmp_path / "Annotations" / "image3.xml",
        expected_objects=1,
        expected_size=(640, 480, 3),
        expected_objects_data=[
            {"name": "sheep", "xmin": 100, "ymin": 120, "xmax": 150, "ymax": 160, "truncated": "1"}
        ]
    )

def verify_xml_content(xml_path: Path, expected_objects: int, expected_size: tuple, expected_objects_data: list) -> None:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 1. Check basic metadata
    assert root.findtext("folder") == "Images", f"Folder wrong in {xml_path.name}"
    assert root.findtext("filename") == xml_path.stem + ".jpg", f"Filename wrong in {xml_path.name}"
    assert root.findtext("path") == f"/test/path/JPEGImages/{xml_path.stem}.jpg", f"Path wrong in {xml_path.name}"

    # 2. Check source tag
    source = root.find("source")
    assert source is not None, f"Missing node <source> in {xml_path.name}"
    assert source.findtext("database") == "MyDB", f"Database wrong in {xml_path.name}"
    
    # 3. Check size tag
    size = root.find("size")
    assert size is not None, f"Missing node <size> in {xml_path.name}"
    assert size.findtext("width") == str(expected_size[0]), f"Width wrong in {xml_path.name}"
    assert size.findtext("height") == str(expected_size[1]), f"Height wrong in {xml_path.name}"
    assert size.findtext("depth") == str(expected_size[2]), f"Depth wrong in {xml_path.name}"

    # 4. Check object data
    objects = root.findall("object")
    assert len(objects) == expected_objects, f"Wrong number of objects in {xml_path.name}"

    for i, obj in enumerate(objects):
        expected = expected_objects_data[i]
        
        # Name
        assert obj.findtext("name") == expected["name"], f"Name wrong in object {i} of {xml_path.name}"
        
        # Pascal Voc BoundingBox
        bndbox = obj.find("bndbox")
        assert bndbox is not None, f"Falta nodo <bndbox> en object {i} de {xml_path.name}"
        assert bndbox.findtext("xmin") == str(expected["xmin"]), f"xmin wrong in object {i} of {xml_path.name}"
        assert bndbox.findtext("ymin") == str(expected["ymin"]), f"ymin wrong in object {i} of {xml_path.name}"
        assert bndbox.findtext("xmax") == str(expected["xmax"]), f"xmax wrong in object {i} of {xml_path.name}"
        assert bndbox.findtext("ymax") == str(expected["ymax"]), f"ymax wrong in object {i} of {xml_path.name}"
        
        # Optional Attributes
        if "truncated" in expected:
            assert obj.findtext("truncated") == expected["truncated"], f"Truncated wrong in object {i} of {xml_path.name}"
        if "difficult" in expected:
            assert obj.findtext("difficult") == expected["difficult"], f"Difficult wrong in object {i} of {xml_path.name}"
        if "pose" in expected:
            assert obj.findtext("pose") == expected["pose"], f"Pose wrong in object {i} of {xml_path.name}"

@pytest.fixture()
def pascalvoc_image_handling_dataset(tmp_path):
    """Fixture para tests con imágenes en Pascal VOC"""
    dataset_dir = tmp_path / "pascalvoc_image_handling"
    annotations_dir = dataset_dir / "Annotations"
    images_dir = dataset_dir / "JPEGImages"

    annotations_dir.mkdir(parents=True)
    images_dir.mkdir(parents=True)

    # Crear imágenes
    (images_dir / "img1.jpg").write_bytes(b"image_data")
    (images_dir / "img2.png").write_bytes(b"image_data")

    # Crear archivos XML mínimos
    for img_name in ["img1.jpg", "img2.png"]:
        root = ET.Element("annotation")
        ET.SubElement(root, "folder").text = "JPEGImages"
        ET.SubElement(root, "filename").text = img_name
        ET.SubElement(root, "path").text = str(images_dir / img_name)
        source = ET.SubElement(root, "source")
        ET.SubElement(source, "database").text = "Unknown"
        ET.SubElement(source, "annotation").text = "Unknown"
        ET.SubElement(source, "image").text = "Unknown"
        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = "100"
        ET.SubElement(size, "height").text = "100"
        ET.SubElement(size, "depth").text = "3"
        ET.SubElement(root, "segmented").text = "0"
        tree = ET.ElementTree(root)
        tree.write(annotations_dir / (img_name.replace('.', '_') + ".xml"), encoding="utf-8", xml_declaration=True)

    return dataset_dir

@pytest.fixture()
def pascalvoc_format_image_instance(tmp_path):
    """Fixture para instancia PascalVocFormat"""
    img_dir = tmp_path / "source_imgs"
    img_dir.mkdir()
    img1 = img_dir / "test_img1.jpg"
    img2 = img_dir / "test_img2.png"
    img1.write_bytes(b"source1")
    img2.write_bytes(b"source2")

    files = [
        PascalVocFile(
            filename="test_img1.jpg",
            annotations=[],
            folder="JPEGImages",
            path=str(img1),
            source=PascalVocSource("Unknown", "Unknown", "Unknown"),
            width=100, height=100, depth=3, segmented=0
        ),
        PascalVocFile(
            filename="test_img2.png",
            annotations=[],
            folder="JPEGImages",
            path=str(img2),
            source=PascalVocSource("Unknown", "Unknown", "Unknown"),
            width=100, height=100, depth=3, segmented=0
        )
    ]
    return PascalVocFormat(
        name="image_test",
        files=files,
        folder_path=str(tmp_path),
        images_path_list=[str(img1), str(img2)]
    )

def test_pascalvoc_read_with_copy_images(pascalvoc_image_handling_dataset):
    pascal = PascalVocFormat.read_from_folder(
        str(pascalvoc_image_handling_dataset),
        copy_images=True,
        copy_as_links=False
    )
    assert pascal.images_path_list is not None
    assert len(pascal.images_path_list) == 2
    assert all(
        any(img_name in p for p in pascal.images_path_list)
        for img_name in ['img1.jpg', 'img2.png']
    )

def test_pascalvoc_read_with_links(pascalvoc_image_handling_dataset):
    pascal = PascalVocFormat.read_from_folder(
        str(pascalvoc_image_handling_dataset),
        copy_images=False,
        copy_as_links=True
    )
    assert pascal.images_path_list is not None
    assert len(pascal.images_path_list) == 2
    assert all(
        any(img_name in p for p in pascal.images_path_list)
        for img_name in ['img1.jpg', 'img2.png']
    )

def test_pascalvoc_read_without_copy(pascalvoc_image_handling_dataset):
    pascal = PascalVocFormat.read_from_folder(
        str(pascalvoc_image_handling_dataset),
        copy_images=False,
        copy_as_links=False
    )
    assert pascal.images_path_list is None

def test_pascalvoc_save_with_copy_images(pascalvoc_format_image_instance, tmp_path):
    output_dir = tmp_path / "output"
    pascalvoc_format_image_instance.save(
        str(output_dir),
        copy_images=True,
        copy_as_links=False
    )
    images_dir = output_dir / "JPEGImages"
    output_images = list(images_dir.iterdir())
    assert len(output_images) == 2
    assert (images_dir / "test_img1.jpg").read_bytes() == b"source1"
    assert (images_dir / "test_img2.png").read_bytes() == b"source2"

def test_pascalvoc_save_with_links(pascalvoc_format_image_instance, tmp_path):
    if os.name == "nt":
        try:
            test_link = tmp_path / "test_link"
            test_target = tmp_path / "test_target.txt"
            test_target.write_text("test")
            test_link.symlink_to(test_target)
        except OSError as e:
            if getattr(e, "winerror", None) == 1314:
                pytest.skip("Symlinks requieren privilegios de administrador en Windows")
            else:
                raise

    output_dir = tmp_path / "output"
    pascalvoc_format_image_instance.save(
        str(output_dir),
        copy_images=False,
        copy_as_links=True
    )
    images_dir = output_dir / "JPEGImages"
    img1 = images_dir / "test_img1.jpg"
    img2 = images_dir / "test_img2.png"
    assert img1.is_symlink()
    assert Path(normalize_path(os.readlink(img1))).resolve()== Path(pascalvoc_format_image_instance.images_path_list[0]).resolve()
    assert img2.is_symlink()
    assert Path(normalize_path(os.readlink(img2))).resolve()== Path(pascalvoc_format_image_instance.images_path_list[1]).resolve()

def test_pascalvoc_save_without_copy(pascalvoc_format_image_instance, tmp_path):
    output_dir = tmp_path / "output"
    pascalvoc_format_image_instance.save(
        str(output_dir),
        copy_images=False,
        copy_as_links=False
    )
    images_dir = output_dir / "JPEGImages"
    assert not any(images_dir.iterdir()) if images_dir.exists() else True
