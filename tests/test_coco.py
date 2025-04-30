import pytest
from pathlib import Path
import json

from formats.coco import CocoFormat, RLESegmentation

def test_coco_format_creation(sample_coco_dataset):
    # Load dataset
    coco_format = CocoFormat.read_from_folder(sample_coco_dataset)
    
    # 1. Checking basic structure
    assert coco_format.folder_path == sample_coco_dataset
    assert coco_format.name == "test_coco"
    assert len(coco_format.files) == 1
    
    # 2. Checking metadata
    coco_file = coco_format.files[0]
    assert coco_file.info.description == "Test dataset"
    assert coco_file.info.version == "1.0"
    
    # 3. Checking licenses
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
    assert person_ann[0].bbox.x_min == 100
    assert person_ann[0].bbox.width == 50
    
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
    assert rle_seg.alto == 600
    assert rle_seg.ancho == 800
    assert rle_seg.counts == "abc123XYZ"


def test_invalid_coco_structure(tmp_path):
    # Case without Annotations directory
    with pytest.raises(FileNotFoundError):
        CocoFormat.read_from_folder(tmp_path)
    
    ann_dir = tmp_path / "Annotations"
    ann_dir.mkdir()
    
    # Case without JSON archive
    with pytest.raises(FileNotFoundError):
        CocoFormat.read_from_folder(tmp_path)
    
    # Archivo JSON inválido
    (ann_dir / "invalid.json").write_text("{invalid_json}")
    with pytest.raises(json.JSONDecodeError):
        CocoFormat.read_from_folder(tmp_path)

def test_rle_segmentation_handling():
    # Crear datos de prueba RLE
    rle_data = {
        "size": [640, 480],
        "counts": "ABCDE1234"
    }
    
    # Procesar a través de la clase
    segmentation = CocoFormat.create_coco_file_from_json(
        {"annotations": [{"segmentation": rle_data}]}, 
        "test"
    ).annotations[0].segmentation
    
    assert isinstance(segmentation, RLESegmentation)
    assert segmentation.alto == 640
    assert segmentation.ancho == 480
    assert segmentation.counts == "ABCDE1234"

def test_polygon_segmentation_handling():
    # Crear datos de prueba polígono
    poly_data = [[10.5, 20.3, 30.1, 40.7, 50.9, 60.2]]
    
    # Procesar a través de la clase
    segmentation = CocoFormat.create_coco_file_from_json(
        {"annotations": [{"segmentation": poly_data}]}, 
        "test"
    ).annotations[0].segmentation
    assert isinstance(segmentation, list)
    assert isinstance(segmentation[0], list)
    assert len(segmentation[0]) == 6
    assert all(isinstance(x, float) for x in segmentation[0])

# Fixture para dataset COCO de prueba
@pytest.fixture
def sample_coco_dataset(tmp_path):
    # Crear estructura de directorios
    dataset_dir = tmp_path / "test_coco"
    dataset_dir.mkdir()
    ann_dir = dataset_dir / "annotations"
    ann_dir.mkdir()

    # Crear archivo COCO JSON de prueba
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

    (ann_dir / "instances_train.json").write_text(json.dumps(coco_data))
    return dataset_dir
