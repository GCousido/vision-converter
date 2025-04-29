from pathlib import Path
import pytest
from formats.pascal_voc import PascalVocFormat

# Helper for creating XML Pascal VOC files
def create_pascalvoc_xml(path, filename, objects, width=800, height=600, depth=3, folder="VOC", segmented=0):
    content = f'''<annotation>
    <folder>{folder}</folder>
    <filename>{filename}</filename>
    <path>{path / filename}</path>
    <source>
        <database>Unknown</database>
    </source>
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

    # File 1 with 2 objects
    xml1 = create_pascalvoc_xml(
        ann_dir, "image1.xml",
        [
            make_object("person", 10, 20, 110, 220, pose="Left", truncated=0, difficult=0),
            make_object("car", 30, 40, 130, 240, pose="Right", truncated=1, difficult=1)
        ],
        width=800, height=600, depth=3
    )
    (ann_dir / "image1.xml").write_text(xml1)

    # File 2 with 1 object
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
    assert pascalvoc_format.name == sample_pascalvoc_dataset.name
    assert isinstance(pascalvoc_format.files, list)


    # 2. Checking files
    assert len(pascalvoc_format.files) == 2
    filenames = {f.filename for f in pascalvoc_format.files}
    assert "image1.xml" in filenames
    assert "image2.xml" in filenames


    # 3. Checking bounding boxes and objects
    file1 = next(f for f in pascalvoc_format.files if f.filename == "image1.xml")
    assert file1.width == 800
    assert file1.height == 600
    assert file1.depth == 3
    assert len(file1.annotations) == 2
    
    ann1 = file1.annotations[0]
    assert ann1.name == "person"
    assert ann1.pose == "Left"
    assert ann1.truncated is False
    assert ann1.difficult is False
    assert ann1.bbox.x_min == 10
    assert ann1.bbox.y_max == 220

    ann2 = file1.annotations[1]
    assert ann2.name == "car"
    assert ann2.pose == "Right"
    assert ann2.truncated is True
    assert ann2.difficult is True
    assert ann2.bbox.x_max == 130
    assert ann2.bbox.y_min == 40


    # Checking second file
    file2 = next(f for f in pascalvoc_format.files if f.filename == "image2.xml")
    assert file2.width == 1024
    assert file2.height == 768
    assert file2.depth == 3
    assert len(file2.annotations) == 1

    ann3 = file2.annotations[0]
    assert ann3.name == "dog"
    assert ann3.pose == "Unspecified"
    assert ann3.truncated is False
    assert ann3.difficult is False
    assert ann3.bbox.x_min == 50
    assert ann3.bbox.x_max == 150

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
