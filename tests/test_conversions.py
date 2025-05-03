import pytest
import os

from converters.coco_converter import CocoConverter
from converters.pascal_converter import PascalVocConverter
from converters.yolo_converter import YoloConverter
from formats.coco import CocoFile, CocoFormat, CocoImage
from formats.neutral_format import NeutralFormat
from formats.pascal_voc import PascalVocFormat
from formats.yolo import YoloFormat


def test_yolo_to_pascalvoc():
    base_dir = os.path.dirname(__file__)
    yolo_path = os.path.join(base_dir, "test_resources/YOLO_TEST")

    # 1. Load YOLO Dataset
    yolo_original: YoloFormat = YoloFormat.read_from_folder(yolo_path)
    
    # 2. YOLO → Neutral → Pascal VOC
    neutral_from_yolo: NeutralFormat = YoloConverter.toNeutral(yolo_original)
    pascal_converted: PascalVocFormat = PascalVocConverter.fromNeutral(neutral_from_yolo)
    
    # Check Pascal VOC
    assert isinstance(pascal_converted, PascalVocFormat)
    assert len(pascal_converted.files) == 2
    assert len(pascal_converted.files[0].annotations) == 1
    assert pascal_converted.files[0].width > 0
    assert pascal_converted.files[0].height > 0
    assert not pascal_converted.files[0].segmented
    assert pascal_converted.files[0].filename == yolo_original.files[0].filename
    assert pascal_converted.files[0].annotations[0].name == yolo_original.class_labels[yolo_original.files[0].annotations[0].id_class]


def test_yolo_original_and_yolo_reconverted():
    base_dir = os.path.dirname(__file__)
    yolo_path = os.path.join(base_dir, "test_resources/YOLO_TEST")

    # 1. Load YOLO Dataset
    yolo_original: YoloFormat = YoloFormat.read_from_folder(yolo_path)
    
    # 2. YOLO → Neutral → YOLO
    neutral_from_yolo: NeutralFormat = YoloConverter.toNeutral(yolo_original)
    yolo_reconverted = YoloConverter.fromNeutral(neutral_from_yolo)
    
    # Check idempotence
    assert yolo_original.name == yolo_reconverted.name
    assert len(yolo_original.files) == len(yolo_reconverted.files)
    assert len(yolo_original.class_labels) == len(yolo_reconverted.class_labels)

    for orig_file, reconv_file in zip(yolo_original.files, yolo_reconverted.files):

        assert len(orig_file.annotations) == len(reconv_file.annotations)

        for i, (orig_ann, reconv_ann) in enumerate(zip(orig_file.annotations, reconv_file.annotations)):

            # Check bbox
            orig_bbox = orig_file.annotations[0].bbox.getBoundingBox()
            reconv_bbox = reconv_file.annotations[0].bbox.getBoundingBox()
            assert orig_bbox == pytest.approx(reconv_bbox, rel=1e-3, abs=1e-3)


            # Get class ID
            orig_class_id = orig_ann.id_class
            reconv_class_id = reconv_ann.id_class
            
            # Get name from the map
            orig_class_name = yolo_original.class_labels[orig_class_id]
            reconv_class_name = yolo_reconverted.class_labels[reconv_class_id]
            
            # Check name matches
            assert orig_class_name == reconv_class_name, f"Annotation {i}: class '{orig_class_name}' != '{reconv_class_name}'"


def test_pascalvoc_to_yolo():
    base_dir = os.path.dirname(__file__)
    pascal_path = os.path.join(base_dir, "test_resources/PASCAL_VOC_TEST")

    # 1. Load PASCAL Dataset
    pascal_original: PascalVocFormat = PascalVocFormat.read_from_folder(pascal_path)
    
    # 2. Pascal Voc → Neutral → YOLO
    neutral_from_pascal: NeutralFormat = PascalVocConverter.toNeutral(pascal_original)
    yolo_converted: YoloFormat = YoloConverter.fromNeutral(neutral_from_pascal)

    # Check YOLO
    assert isinstance(yolo_converted, YoloFormat)
    assert len(yolo_converted.files) == 2
    assert len(yolo_converted.files[0].annotations) == 2
    assert len(yolo_converted.files[1].annotations) == 1
    assert yolo_converted.files[0].filename == pascal_original.files[0].filename
    assert yolo_converted.files[1].filename == pascal_original.files[1].filename
    assert yolo_converted.class_labels[yolo_converted.files[0].annotations[0].id_class] == pascal_original.files[0].annotations[0].name
    assert yolo_converted.class_labels[yolo_converted.files[0].annotations[1].id_class] == pascal_original.files[0].annotations[1].name
    assert yolo_converted.class_labels[yolo_converted.files[1].annotations[0].id_class] == pascal_original.files[1].annotations[0].name


def test_pascalvoc_original_and_pascalvoc_reconverted():
    base_dir = os.path.dirname(__file__)
    pascal_path = os.path.join(base_dir, "test_resources/PASCAL_VOC_TEST")

    # 1. Load PASCAL Dataset
    pascal_original: PascalVocFormat = PascalVocFormat.read_from_folder(pascal_path)
    
    # 2. Pascal Voc → Neutral → Pascal Voc
    neutral_from_pascal: NeutralFormat = PascalVocConverter.toNeutral(pascal_original)
    pascal_reconverted: PascalVocFormat = PascalVocConverter.fromNeutral(neutral_from_pascal)
    
    # Check idempotence
    assert pascal_original.name == pascal_reconverted.name
    assert len(pascal_original.files) == len(pascal_reconverted.files)
    
    for orig_file, reconv_file in zip(pascal_original.files, pascal_reconverted.files):

        assert len(orig_file.annotations) == len(reconv_file.annotations)

        for i, (orig_ann, reconv_ann) in enumerate(zip(orig_file.annotations, reconv_file.annotations)):
            # Check name matches
            assert orig_ann.name == reconv_ann.name, f"Annotation {i}: class '{orig_ann.name}' != '{reconv_ann.name}'"

            # Check bbox
            orig_bbox = orig_file.annotations[0].bbox.getBoundingBox()
            reconv_bbox = reconv_file.annotations[0].bbox.getBoundingBox()
            assert orig_bbox == pytest.approx(reconv_bbox, rel=1e-6, abs=1e-6)


def test_coco_to_yolo():
    base_dir = os.path.dirname(__file__)
    coco_path = os.path.join(base_dir, "test_resources/COCO_TEST")

    # 1. Load COCO Dataset
    coco_original: CocoFormat = CocoFormat.read_from_folder(coco_path)
    
    # 2. COCO → Neutral → YOLO
    neutral_from_coco: NeutralFormat = CocoConverter.toNeutral(coco_original)
    yolo_converted: YoloFormat = YoloConverter.fromNeutral(neutral_from_coco)
    
    # Check YOLO
    assert isinstance(yolo_converted, YoloFormat)
    assert len(yolo_converted.files) == 4
    assert sum(len(file.annotations) for file in yolo_converted.files) == 48
    assert len(yolo_converted.files[0].annotations) == 11
    assert len(yolo_converted.files[1].annotations) == 7
    assert len(yolo_converted.files[2].annotations) == 7
    assert len(yolo_converted.files[3].annotations) == 23
    assert yolo_converted.files[0].filename == coco_original.files[0].images[0].file_name
    assert yolo_converted.files[1].filename == coco_original.files[0].images[1].file_name
    assert yolo_converted.files[2].filename == coco_original.files[0].images[2].file_name
    assert yolo_converted.files[3].filename == coco_original.files[0].images[3].file_name
    assert all(v in [c.name for c in coco_original.files[0].categories] for v in yolo_converted.class_labels.values())


def test_yolo_to_coco():
    base_dir = os.path.dirname(__file__)
    yolo_path = os.path.join(base_dir, "test_resources/YOLO_TEST")

    # 1. Load YOLO Dataset
    yolo_original: YoloFormat = YoloFormat.read_from_folder(yolo_path)
    
    # 2. YOLO → Neutral → COCO
    neutral_from_yolo: NeutralFormat = YoloConverter.toNeutral(yolo_original)
    coco_converted: CocoFormat = CocoConverter.fromNeutral(neutral_from_yolo)
    
    # Check COCO
    assert isinstance(coco_converted, CocoFormat)
    assert len(coco_converted.files) == 1
    assert len(coco_converted.files[0].annotations) == 2
    
    for file in coco_converted.files:
        assert isinstance(file, CocoFile)
        for image in file.images:
            assert isinstance(image, CocoImage)
            assert image.width > 0
            assert image.height > 0

        for annotation in file.annotations:
            assert annotation.area == (annotation.bbox.width * annotation.bbox.height)


def test_coco_original_and_coco_reconverted():
    base_dir = os.path.dirname(__file__)
    coco_path = os.path.join(base_dir, "test_resources/COCO_TEST")

    # 1. Load COCO Dataset
    coco_original: CocoFormat = CocoFormat.read_from_folder(coco_path)
    
    # 2. COCO → Neutral → COCO
    neutral_from_coco: NeutralFormat = CocoConverter.toNeutral(coco_original)
    coco_reconverted: CocoFormat = CocoConverter.fromNeutral(neutral_from_coco)

    # Check idempotence
    assert coco_original.name == coco_reconverted.name
    assert len(coco_original.files) == len(coco_reconverted.files)
    assert len(coco_original.files[0].annotations) == len(coco_reconverted.files[0].annotations)

    # Check classes in annotations
    for orig_file, reconv_file in zip(coco_original.files, coco_reconverted.files):

        # Group annotations by image_id
        orig_anns_by_image = {}
        for ann in orig_file.annotations:
            image_id = ann.image_id
            if image_id not in orig_anns_by_image:
                orig_anns_by_image[image_id] = []
            orig_anns_by_image[image_id].append(ann)
        
        reconv_anns_by_image = {}
        for ann in reconv_file.annotations:
            image_id = ann.image_id
            if image_id not in reconv_anns_by_image:
                reconv_anns_by_image[image_id] = []
            reconv_anns_by_image[image_id].append(ann)
        
        # Check by image
        for image in orig_file.images:
            image_name = image.file_name
            orig_anns = orig_anns_by_image.get(image.id, [])
            
            # Find the other image
            reconv_image = next((img for img in reconv_file.images if img.file_name == image_name), None)

            # Check image exist
            assert reconv_image is not None, f"Image {image_name} not found in reconverted"

            reconv_anns = reconv_anns_by_image.get(reconv_image.id, [])
            
            # Sort annotations by bbox
            orig_anns_sorted = sorted(orig_anns, key=lambda a: (a.bbox.x_min, a.bbox.y_min))
            reconv_anns_sorted = sorted(reconv_anns, key=lambda a: (a.bbox.x_min, a.bbox.y_min))
            
            # Check annotations per image
            for i, (orig_ann, reconv_ann) in enumerate(zip(orig_anns_sorted, reconv_anns_sorted)):
                name_orig = next(c.name for c in orig_file.categories if c.id == orig_ann.category_id)
                name_reconv = next(c.name for c in reconv_file.categories if c.id == reconv_ann.category_id)
                assert name_orig == name_reconv, f"Image {image_name}, annotation {i}: class '{name_orig}' != '{name_reconv}'"
                
                orig_bbox = orig_ann.bbox.getBoundingBox()
                reconv_bbox = reconv_ann.bbox.getBoundingBox()
                assert orig_bbox == pytest.approx(reconv_bbox, rel=0.5, abs=0.5), f"Image {image_name}, annotation {i}: bbox do not match"