import pytest
from pathlib import Path
import json
import os

from vision_converter.formats.bounding_box import TopLeftAbsoluteBoundingBox
from vision_converter.formats.coco import Category, CocoFile, CocoFormat, CocoImage, CocoLabel, Info, License, RLESegmentation
from vision_converter.tests.utils_for_tests import normalize_path

def test_coco_format_creation(sample_coco_dataset):
    # Load dataset
    coco_format = CocoFormat.read_from_folder(sample_coco_dataset)
    
    # 1. Checking basic structure
    assert coco_format.folder_path == str(Path(sample_coco_dataset).parent)
    assert coco_format.name == "CocoDataset"
    assert len(coco_format.files) == 1
    
    # 2. Checking metadata
    coco_file = coco_format.files[0]
    assert coco_file.info
    assert coco_file.info.description == "Test dataset"
    assert coco_file.info.version == "1.0"
    
    # 3. Checking licenses
    assert coco_file.licenses
    assert len(coco_file.licenses) == 1
    assert coco_file.licenses[0].name == "CC-BY-4.0"
    
    # 4. Checking coco_images
    assert len(coco_file.images) == 2
    img_filenames = {img.file_name for img in coco_file.images}
    assert "image1.jpg" in img_filenames
    assert "image2.jpg" in img_filenames
    
    # 5. Checking categories
    assert len(coco_file.categories) == 2
    category_names = {c.name for c in coco_file.categories}
    assert "person" in category_names
    assert "car" in category_names
    
    # 6. Checking annotations
    assert len(coco_file.annotations) == 3
    
    # Checking bounding boxes
    person_ann = [a for a in coco_file.annotations if a.category_id == 1]
    assert len(person_ann) == 1
    assert person_ann[0].geometry.x_min == 100
    assert person_ann[0].geometry.width == 50
    
    # Checking polygon segmentation
    polygon_seg1 = next(a for a in coco_file.annotations if a.id == 1).segmentation
    assert isinstance(polygon_seg1, list)
    assert isinstance(polygon_seg1[0], list)
    assert len(polygon_seg1[0]) == 8

    polygon_seg2 = next(a for a in coco_file.annotations if a.id == 2).segmentation
    assert not polygon_seg2
    
    # Checking RLE segmentation
    rle_seg = next(a for a in coco_file.annotations if a.id == 3).segmentation
    assert isinstance(rle_seg, RLESegmentation)
    assert rle_seg.size[0] == 600
    assert rle_seg.size[1] == 800
    assert rle_seg.counts == "abc123XYZ"


def test_invalid_coco_structure(tmp_path):
    # Case without JSON archive
    with pytest.raises(FileNotFoundError):
        CocoFormat.read_from_folder(tmp_path / "invalid.json")
    
    (tmp_path / "invalid.json").write_text("{invalid_json}")
    with pytest.raises(json.JSONDecodeError):
        CocoFormat.read_from_folder(tmp_path / "invalid.json")

def test_rle_segmentation_handling():
    rle_data = {
        "size": [640, 480],
        "counts": "ABCDE1234"
    }
    
    segmentation = CocoFormat.create_coco_file_from_json(
        {"annotations": [{"segmentation": rle_data}]}, 
        "test"
    ).annotations[0].segmentation
    
    assert isinstance(segmentation, RLESegmentation)
    assert segmentation.size[0] == 640
    assert segmentation.size[1] == 480
    assert segmentation.counts == "ABCDE1234"

def test_polygon_segmentation_handling():
    poly_data = [[10.5, 20.3, 30.1, 40.7, 50.9, 60.2]]
    
    segmentation = CocoFormat.create_coco_file_from_json(
        {"annotations": [{"segmentation": poly_data}]}, 
        "test"
    ).annotations[0].segmentation
    assert isinstance(segmentation, list)
    assert isinstance(segmentation[0], list)
    assert len(segmentation[0]) == 6
    assert all(isinstance(x, float) for x in segmentation[0])

def test_invalid_rle_segmentation():
    with pytest.raises(ValueError):
        RLESegmentation(size=[640], counts="invalid")

# Fixture para dataset COCO de prueba
@pytest.fixture
def sample_coco_dataset(tmp_path):
    # Create file structure
    dataset_dir = tmp_path / "test_coco"
    dataset_dir.mkdir()
    dataset_file = dataset_dir / "annotations.json"

    # Create COCO file
    coco_data = {
        "info": {
            "description": "Test dataset",
            "url": "http://example.com",
            "version": "1.0",
            "year": 2024,
            "contributor": "Tester",
            "date_created": "2024-01-01"
        },
        "licenses": [{
            "id": 1,
            "name": "CC-BY-4.0",
            "url": "https://creativecommons.org/licenses/by/4.0/"
        }],
        "images": [
            {
                "id": 1,
                "file_name": "image1.jpg",
                "width": 640,
                "height": 480,
                "license": 1,
                "flickr_url": "",
                "coco_url": "",
                "date_captured": ""
            },
            {
                "id": 2,
                "file_name": "image2.jpg",
                "width": 800,
                "height": 600,
                "license": 1,
                "flickr_url": "",
                "coco_url": "",
                "date_captured": ""
            }
        ],
        "categories": [
            {"id": 1, "name": "person", "supercategory": "human"},
            {"id": 2, "name": "car", "supercategory": "vehicle"}
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [100, 150, 50, 80],
                "area": 4000,
                "iscrowd": 0,
                "segmentation": [[100.5, 150.3, 150.2, 150.7, 149.9, 230.1, 100.1, 230.5]]
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 2,
                "bbox": [400, 200, 100, 150],
                "area": 15000,
                "iscrowd": 0,
                "segmentation": []
            },
            {
                "id": 3,
                "image_id": 2,
                "category_id": 2,
                "bbox": [50, 50, 80, 120],
                "area": 9600,
                "iscrowd": 1,
                "segmentation": {
                    "size": [600, 800],
                    "counts": "abc123XYZ"
                }
            }
        ]
    }

    (dataset_file).write_text(json.dumps(coco_data))
    return dataset_file

def test_coco_format_save(tmp_path):
    # Prepare test data
    categories = [
        Category(id=1, name="cat", supercategory="animal"),
        Category(id=2, name="dog", supercategory="animal"),
    ]

    images = [
        CocoImage(
            id=1, 
            width=100, 
            height=200, 
            file_name="image1.jpg",
            date_captured="2023-01-01",
            license=1,
            flickr_url="http://flickr.com/image1",
            coco_url="http://cocodataset.org/image1"
        ),
        CocoImage(
            id=2,
            width=150,
            height=300,
            file_name="image2.jpg",
            date_captured="2023-01-02"
        )
    ]

    annotations = [
        CocoLabel(
            bbox=TopLeftAbsoluteBoundingBox(10, 20, 30, 40),
            id=1,
            image_id=1,
            category_id=1,
            segmentation=[[10,20,40,20,40,60,10,60]],
            area=1200.0,
            iscrowd=False
        ),
        CocoLabel(
            bbox=TopLeftAbsoluteBoundingBox(50, 60, 70, 80),
            id=2,
            image_id=2,
            category_id=2,
            segmentation=RLESegmentation(size=[300,150], counts="abcd123"),
            area=5600.0,
            iscrowd=True
        )
    ]

    info = Info(
        description="Test Dataset",
        url="http://example.com",
        version="1.0",
        year=2023,
        contributor="Tester",
        date_created="2023-01-01"
    )

    licenses = [
        License(id=1, name="CC-BY", url="http://creativecommons.org/licenses/by/4.0/")
    ]

    coco_file = CocoFile(
        filename="annotations.json",
        annotations=annotations,
        images=images,
        categories=categories,
        info=info,
        licenses=licenses
    )

    coco_dataset = CocoFormat(
        name="test_coco",
        files=[coco_file],
        folder_path=None
    )

    # Execute save
    output_path = tmp_path / "output"
    coco_dataset.save(str(output_path))
    
    # Check save file
    json_path = output_path / "annotations.json"
    assert json_path.exists()

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Check basic structure
    assert set(data.keys()) == {"info", "licenses", "images", "categories", "annotations"}
    
    # 2. Check info
    assert data["info"] == {
        "description": "Test Dataset",
        "url": "http://example.com",
        "version": "1.0",
        "year": 2023,
        "contributor": "Tester",
        "date_created": "2023-01-01"
    }

    # 3. Check licenses
    assert data["licenses"] == [{
        "id": 1,
        "name": "CC-BY",
        "url": "http://creativecommons.org/licenses/by/4.0/"
    }]

    # 4. Check images
    assert len(data["images"]) == 2
    img1 = next(img for img in data["images"] if img["id"] == 1)
    assert img1 == {
        "id": 1,
        "width": 100,
        "height": 200,
        "file_name": "image1.jpg",
        "license": 1,
        "flickr_url": "http://flickr.com/image1",
        "coco_url": "http://cocodataset.org/image1",
        "date_captured": "2023-01-01"
    }

    img2 = next(img for img in data["images"] if img["id"] == 2)
    assert img2 == {
        "id": 2,
        "width": 150,
        "height": 300,
        "file_name": "image2.jpg",
        "license": None,
        "flickr_url": None,
        "coco_url": None,
        "date_captured": "2023-01-02"
    }

    # 5. Check categories
    assert data["categories"] == [
        {"id": 1, "name": "cat", "supercategory": "animal"},
        {"id": 2, "name": "dog", "supercategory": "animal"}
    ]

    # 6. Check annotations
    assert len(data["annotations"]) == 2
    
    # Annotation 1 (Polygon)
    ann1 = next(ann for ann in data["annotations"] if ann["id"] == 1)
    assert ann1 == {
        "id": 1,
        "image_id": 1,
        "category_id": 1,
        "bbox": [10.0, 20.0, 30.0, 40.0],
        "area": 1200.0,
        "iscrowd": 0,
        "segmentation": [[10,20,40,20,40,60,10,60]]
    }

    # Annotation 2 (RLE)
    ann2 = next(ann for ann in data["annotations"] if ann["id"] == 2)
    assert ann2 == {
        "id": 2,
        "image_id": 2,
        "category_id": 2,
        "bbox": [50.0, 60.0, 70.0, 80.0],
        "area": 5600.0,
        "iscrowd": 1,
        "segmentation": {
            "size": [300, 150],
            "counts": "abcd123"
        }
    }

    # 7. Check relations with the ids
    image_ids = {img["id"] for img in data["images"]}
    for ann in data["annotations"]:
        assert ann["image_id"] in image_ids, f"Image ID {ann['image_id']} do not exist"
    
    category_ids = {cat["id"] for cat in data["categories"]}
    for ann in data["annotations"]:
        assert ann["category_id"] in category_ids, f"Category ID {ann['category_id']} do not exist"

    # 8. Check data type
    for img in data["images"]:
        assert isinstance(img["id"], int)
        assert isinstance(img["width"], int)
        assert isinstance(img["height"], int)
    
    for ann in data["annotations"]:
        assert isinstance(ann["bbox"], list)
        assert len(ann["bbox"]) == 4
        assert all(isinstance(x, float) or isinstance(x, int) for x in ann["bbox"])


@pytest.fixture()
def image_handling_dataset(tmp_path):
    """Fixture for tests with images"""
    dataset_dir = tmp_path / "image_handling"
    dataset_dir.mkdir()
    
    annotations = dataset_dir / "annotations.json"
    annotations.write_text(json.dumps({
        "info": {"description": "Image Handling Test"},
        "images": [{"id": 1, "file_name": "img1.jpg"}],
        "categories": [{"id": 1, "name": "test"}],
        "annotations": []
    }))
    
    images_dir = dataset_dir / "images"
    images_dir.mkdir()
    (images_dir / "img1.jpg").write_bytes(b"image_data")
    (images_dir / "img2.png").write_bytes(b"image_data")
    
    return dataset_dir

@pytest.fixture()
def coco_format_image_instance(tmp_path):
    """Fixture for CocoFormat instance"""
    img_dir = tmp_path / "source_imgs"
    img_dir.mkdir()
    img1 = img_dir / "test_img1.jpg"
    img2 = img_dir / "test_img2.png"
    img1.write_bytes(b"source1")
    img2.write_bytes(b"source2")
    
    return CocoFormat(
        name="image_test",
        files=[CocoFile(
            filename="annotations.json",
            annotations=[],
            images=[CocoImage(id=1, file_name="test_img1.jpg", width=480, height=480, date_captured="test")],
            categories=[]
        )],
        folder_path=str(tmp_path),
        images_path_list=[str(img1), str(img2)]
    )

def test_read_with_copy_images(image_handling_dataset):
    coco = CocoFormat.read_from_folder(
        str(image_handling_dataset),
        copy_images=True,
        copy_as_links=False
    )
    assert coco.images_path_list is not None
    assert len(coco.images_path_list) == 2
    assert all(
        any(img_name in p for p in coco.images_path_list)
        for img_name in ['img1.jpg', 'img2.png']
    )

def test_read_with_links(image_handling_dataset):
    coco = CocoFormat.read_from_folder(
        str(image_handling_dataset),
        copy_images=False,
        copy_as_links=True
    )
    assert coco.images_path_list is not None
    assert len(coco.images_path_list) == 2
    assert all(
        any(img_name in p for p in coco.images_path_list)
        for img_name in ['img1.jpg', 'img2.png']
    )

def test_read_without_copy(image_handling_dataset):
    coco = CocoFormat.read_from_folder(
        str(image_handling_dataset),
        copy_images=False,
        copy_as_links=False
    )
    assert coco.images_path_list is None

def test_save_with_copy_images(coco_format_image_instance, tmp_path):
    output_dir = tmp_path / "output"
    coco_format_image_instance.save(
        str(output_dir),
        copy_images=True,
        copy_as_links=False
    )
    
    output_images = list((output_dir / "images").iterdir())
    assert len(output_images) == 2
    assert (output_dir / "images" / "test_img1.jpg").read_bytes() == b"source1"
    assert (output_dir / "images" / "test_img2.png").read_bytes() == b"source2"

def test_save_with_links(coco_format_image_instance, tmp_path):
    if os.name == "nt":
        try:
            test_link = tmp_path / "test_link"
            test_target = tmp_path / "test_target.txt"
            test_target.write_text("test")
            test_link.symlink_to(test_target)
        except OSError as e:
            if e.winerror == 1314:
                pytest.skip("Symlinks requieren privilegios de administrador en Windows")
            else:
                raise
    
    output_dir = tmp_path / "output"
    coco_format_image_instance.save(
        str(output_dir),
        copy_images=False,
        copy_as_links=True
    )
    
    img1 = output_dir / "images" / "test_img1.jpg"
    img2 = output_dir / "images" / "test_img2.png"
    assert img1.is_symlink()
    assert Path(normalize_path(os.readlink(img1))).resolve()== Path(coco_format_image_instance.images_path_list[0]).resolve()
    assert img2.is_symlink()
    assert Path(normalize_path(os.readlink(img2))).resolve()== Path(coco_format_image_instance.images_path_list[1]).resolve()

def test_save_without_copy(coco_format_image_instance, tmp_path):
    output_dir = tmp_path / "output"
    coco_format_image_instance.save(
        str(output_dir),
        copy_images=False,
        copy_as_links=False
    )
    images_dir = output_dir / "images"
    assert not any(images_dir.iterdir()) if images_dir.exists() else True