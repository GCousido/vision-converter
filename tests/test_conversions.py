import pytest
import os

from converters.coco_converter import CocoConverter
from converters.pascal_voc_converter import PascalVocConverter
from converters.yolo_converter import YoloConverter
from formats.coco import CocoFile, CocoFormat, CocoImage
from formats.neutral_format import NeutralFormat
from formats.pascal_voc import PascalVocFormat
from formats.yolo import YoloFormat
from utils.bbox_utils import PascalVocBBox_to_CocoBBox, PascalVocBBox_to_YoloBBox, YoloBBox_to_PascalVocBBox
from utils.file_utils import get_image_info_from_file, get_image_path


def bbox_almost_equal(bbox1, bbox2, epsilon=1e-3) -> bool:
    return all(abs(a - b) < epsilon for a, b in zip(bbox1, bbox2))


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
            orig_bbox = orig_ann.bbox.getBoundingBox()
            reconv_bbox = reconv_ann.bbox.getBoundingBox()
            assert orig_bbox == pytest.approx(reconv_bbox, rel=1e-3, abs=1e-3)

            # Get name from the map
            orig_class_name = yolo_original.class_labels[orig_ann.id_class]
            reconv_class_name = yolo_reconverted.class_labels[reconv_ann.id_class]
            
            # Check name matches
            assert orig_class_name == reconv_class_name, f"Annotation {i}: class '{orig_class_name}' != '{reconv_class_name}'"


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
    assert len(pascal_converted.files) == len(yolo_original.files)

    # Check file values
    for pascal_file, yolo_file in zip(pascal_converted.files, yolo_original.files):
        assert len(pascal_file.annotations) == len(yolo_file.annotations)
        assert pascal_file.width > 0
        assert pascal_file.height > 0
        assert not pascal_file.segmented
        assert pascal_file.filename == yolo_file.filename
        assert pascal_file.source.database == yolo_original.name
        assert pascal_file.source.annotation == "Pascal Voc"
        assert pascal_file.source.image == ""

        # Check annotation values
        for pascal_annotation, yolo_annotation in zip(pascal_file.annotations, yolo_file.annotations):
            assert pascal_annotation.name == yolo_original.class_labels[yolo_annotation.id_class]
            assert pytest.approx(PascalVocBBox_to_YoloBBox(pascal_annotation.bbox, pascal_file.width, pascal_file.height).getBoundingBox(), rel=1e-3, abs=1e-3) == yolo_annotation.bbox.getBoundingBox() 


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
    assert len(coco_converted.files[0].annotations) == 5
    
    i = 1
    for file in coco_converted.files:
        assert isinstance(file, CocoFile)
        for image in file.images:
            assert isinstance(image, CocoImage)
            assert image.width > 0
            assert image.height > 0

        for coco_annotation in file.annotations:
            # Check annotation values
            assert coco_annotation.area == (coco_annotation.bbox.width * coco_annotation.bbox.height)
            coco_category_name = next((c.name for c in file.categories if c.id == coco_annotation.category_id), None)
            
            yolo_class_name = None
            found = False

            for yolo_file in yolo_original.files:
                if found:
                    break
                if yolo_original.folder_path:
                    image_path = get_image_path(yolo_original.folder_path, "images", yolo_file.filename)
                    if image_path:
                        width, height, depth = get_image_info_from_file(image_path)
                        for yolo_annotation in yolo_file.annotations:
                            bbox1 = PascalVocBBox_to_CocoBBox(YoloBBox_to_PascalVocBBox(yolo_annotation.bbox, width, height)).getBoundingBox()
                            bbox2 = coco_annotation.bbox.getBoundingBox()
                            if bbox_almost_equal(bbox1, bbox2, 2):
                                yolo_class_name = yolo_original.class_labels[yolo_annotation.id_class]
                                found = True
                                break
            
            # Check category name is the same category as the one in the yolo annotation
            assert coco_category_name == yolo_class_name


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
        # Check source metadata
        assert orig_file.source.database == reconv_file.source.database
        assert reconv_file.source.annotation == "Pascal Voc"
        assert orig_file.source.image == reconv_file.source.image

        assert len(orig_file.annotations) == len(reconv_file.annotations)

        for i, (orig_ann, reconv_ann) in enumerate(zip(orig_file.annotations, reconv_file.annotations)):
            # Check name matches
            assert orig_ann.name == reconv_ann.name, f"Annotation {i}: class '{orig_ann.name}' != '{reconv_ann.name}'"

            # Check bbox
            orig_bbox = orig_file.annotations[0].bbox.getBoundingBox()
            reconv_bbox = reconv_file.annotations[0].bbox.getBoundingBox()
            assert orig_bbox == pytest.approx(reconv_bbox, rel=1e-6, abs=1e-6)


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
    assert len(yolo_converted.files) == len(pascal_original.files)

    # Check file values
    for yolo_file, pascal_file in zip(yolo_converted.files, pascal_original.files):
        assert len(yolo_file.annotations) == len(pascal_file.annotations)
        assert yolo_file.filename == pascal_file.filename
        assert yolo_file.height == pascal_file.height
        assert yolo_file.width == pascal_file.width
        assert yolo_file.depth == pascal_file.depth
        
        # Check annotation values
        for yolo_annotation, pascal_annotation in zip(yolo_file.annotations, pascal_file.annotations):
            assert yolo_converted.class_labels[yolo_annotation.id_class] == pascal_annotation.name
            assert yolo_annotation.bbox.getBoundingBox() == pytest.approx(PascalVocBBox_to_YoloBBox(pascal_annotation.bbox, pascal_file.width, pascal_file.height).getBoundingBox(), rel=1e-9, abs=1e-9)


def test_pascalvoc_to_coco():
    base_dir = os.path.dirname(__file__)
    pascal_path = os.path.join(base_dir, "test_resources/PASCAL_VOC_TEST")

    # 1. Load PASCAL Dataset
    pascal_original: PascalVocFormat = PascalVocFormat.read_from_folder(pascal_path)
    
    # 2. Pascal Voc → Neutral → COCO
    neutral_from_pascal: NeutralFormat = PascalVocConverter.toNeutral(pascal_original)
    coco_converted: CocoFormat = CocoConverter.fromNeutral(neutral_from_pascal)

    # Check COCO
    assert isinstance(coco_converted, CocoFormat)
    assert len(coco_converted.files) == 1

    for coco_file in coco_converted.files:
        pascal_number_of_annotations = 0

        # Check image metadata
        for pascal_file, coco_images in zip(pascal_original.files, coco_file.images):
            pascal_number_of_annotations += len(pascal_file.annotations)
            assert coco_images.file_name == pascal_file.filename
            assert coco_images.height == pascal_file.height
            assert coco_images.width == pascal_file.width

        # Check same number of annotations
        assert len(coco_file.annotations) == pascal_number_of_annotations

        found = False
        pascal_class_name = None

        for coco_annotation in coco_file.annotations:
            for pascal_file in pascal_original.files:
                if found:
                    break
                for pascal_annotation in pascal_file.annotations:
                    bbox1 = PascalVocBBox_to_CocoBBox(pascal_annotation.bbox).getBoundingBox()
                    bbox2 = coco_annotation.bbox.getBoundingBox()
                    if bbox_almost_equal(bbox1, bbox2, 2):
                        pascal_class_name = pascal_annotation.name
                        found = True
                        break

            # Check coco category in an annotation is the same class name as in the pascal annotation
            assert next((c.name for c in coco_file.categories if c.id == coco_annotation.category_id), None) == pascal_class_name


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
        for orig_image in orig_file.images:
            image_name = orig_image.file_name
            
            # Find the other image
            reconv_image = next((img for img in reconv_file.images if img.file_name == image_name), None)

            # Check image exist
            assert reconv_image is not None, f"Image {image_name} not found in reconverted"

            # Get licenses
            license_orig = next(
                (l for l in (orig_file.licenses or []) if l.id == orig_image.license), 
                None
            )
            license_reconv = next(
                (l for l in (reconv_file.licenses or []) if l.id == reconv_image.license), 
                None
            )

            # Check both licenses exist
            assert license_orig is not None, f"Original license ID {orig_image.license} not found in original dataset for {image_name}"
            assert license_reconv is not None, f"Reconverted license ID {reconv_image.license} not found in reconverted dataset for {image_name}"

            # Check license details
            assert license_orig.name == license_reconv.name, f"License name mismatch in {image_name} ({license_orig.name} vs {license_reconv.name})"
            assert license_orig.url == license_reconv.url, f"License URL mismatch in {image_name} ({license_orig.url} vs {license_reconv.url})"

            i = 1
            # Check image metadata
            assert orig_image.date_captured == reconv_image.date_captured, f"Date captured mismatch in {image_name}"
            assert orig_image.flickr_url == reconv_image.flickr_url, f"Flickr URL mismatch in {image_name}"
            assert orig_image.coco_url == reconv_image.coco_url, f"COCO URL mismatch in {image_name}"
            i += 1

            # Get annotations
            orig_anns = orig_anns_by_image.get(orig_image.id, [])
            reconv_anns = reconv_anns_by_image.get(reconv_image.id, [])
            
            # Sort annotations by bbox
            orig_anns_sorted = sorted(orig_anns, key=lambda a: (a.bbox.x_min, a.bbox.y_min))
            reconv_anns_sorted = sorted(reconv_anns, key=lambda a: (a.bbox.x_min, a.bbox.y_min))
            
            # Check annotations per image
            for i, (orig_ann, reconv_ann) in enumerate(zip(orig_anns_sorted, reconv_anns_sorted)):
                name_orig = next(c.name for c in orig_file.categories if c.id == orig_ann.category_id)
                name_reconv = next(c.name for c in reconv_file.categories if c.id == reconv_ann.category_id)
                assert name_orig == name_reconv, f"Image {image_name}, annotation {i}: class '{name_orig}' != '{name_reconv}'"
                
                # Check bbox aprox values
                orig_bbox = orig_ann.bbox.getBoundingBox()
                reconv_bbox = reconv_ann.bbox.getBoundingBox()
                assert orig_bbox == pytest.approx(reconv_bbox, rel=0.5, abs=0.5), f"Image {image_name}, annotation {i}: bbox do not match"


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

    # Check every yolo class is in categories
    assert all(v in [c.name for c in coco_original.files[0].categories] for v in yolo_converted.class_labels.values())


def test_coco_to_pascalvoc():
    base_dir = os.path.dirname(__file__)
    coco_path = os.path.join(base_dir, "test_resources/COCO_TEST")

    # 1. Load COCO Dataset
    coco_original: CocoFormat = CocoFormat.read_from_folder(coco_path)
    
    # 2. COCO → Neutral → Pascal Voc
    neutral_from_coco: NeutralFormat = CocoConverter.toNeutral(coco_original)
    pascal_converted: PascalVocFormat = PascalVocConverter.fromNeutral(neutral_from_coco)
    
    # Check Pascal Voc
    assert isinstance(pascal_converted, PascalVocFormat)
    assert len(pascal_converted.files) == 4
    assert sum(len(file.annotations) for file in pascal_converted.files) == 48
    assert len(pascal_converted.files[0].annotations) == 11
    assert len(pascal_converted.files[1].annotations) == 7
    assert len(pascal_converted.files[2].annotations) == 7
    assert len(pascal_converted.files[3].annotations) == 23
    assert pascal_converted.files[0].filename == coco_original.files[0].images[0].file_name
    assert pascal_converted.files[1].filename == coco_original.files[0].images[1].file_name
    assert pascal_converted.files[2].filename == coco_original.files[0].images[2].file_name
    assert pascal_converted.files[3].filename == coco_original.files[0].images[3].file_name
    
    # Check that every class name is in the coco original
    assert all(
        ann.name in [c.name for c in coco_original.files[0].categories]
        for file in pascal_converted.files
        for ann in file.annotations
    )