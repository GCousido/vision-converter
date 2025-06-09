import pytest
import os

from datasetconverter.converters.coco_converter import CocoConverter
from datasetconverter.converters.createml_converter import CreateMLConverter
from datasetconverter.converters.labelme_converter import LabelMeConverter
from datasetconverter.converters.pascal_voc_converter import PascalVocConverter
from datasetconverter.converters.tensorflow_csv_converter import TensorflowCsvConverter
from datasetconverter.converters.vgg_converter import VGGConverter
from datasetconverter.converters.yolo_converter import YoloConverter
from datasetconverter.formats.coco import CocoFile, CocoFormat, CocoImage
from datasetconverter.formats.createml import CreateMLFormat
from datasetconverter.formats.labelme import LabelMeCircle, LabelMeFormat
from datasetconverter.formats.neutral_format import NeutralFormat
from datasetconverter.formats.pascal_voc import PascalVocBoundingBox, PascalVocFormat
from datasetconverter.formats.tensorflow_csv import TensorflowCsvFormat
from datasetconverter.formats.vgg import VGGFormat
from datasetconverter.formats.yolo import YoloFormat
from datasetconverter.utils.bbox_utils import CreateMLBBox_to_PascalVocBBox, PascalVocBBox_to_CocoBBox, PascalVocBBox_to_CreateMLBBox, PascalVocBBox_to_YoloBBox, YoloBBox_to_PascalVocBBox
from datasetconverter.utils.file_utils import get_image_info_from_file, get_image_path


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
            orig_bbox = orig_ann.geometry.getBoundingBox()
            reconv_bbox = reconv_ann.geometry.getBoundingBox()
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
            assert pytest.approx(PascalVocBBox_to_YoloBBox(pascal_annotation.geometry, pascal_file.width, pascal_file.height).getBoundingBox(), rel=1e-3, abs=1e-3) == yolo_annotation.geometry.getBoundingBox() 


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
            assert coco_annotation.area == (coco_annotation.geometry.width * coco_annotation.geometry.height)
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
                            bbox1 = PascalVocBBox_to_CocoBBox(YoloBBox_to_PascalVocBBox(yolo_annotation.geometry, width, height)).getBoundingBox()
                            bbox2 = coco_annotation.geometry.getBoundingBox()
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
            orig_bbox = orig_file.annotations[0].geometry.getBoundingBox()
            reconv_bbox = reconv_file.annotations[0].geometry.getBoundingBox()
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
            assert yolo_annotation.geometry.getBoundingBox() == pytest.approx(PascalVocBBox_to_YoloBBox(pascal_annotation.geometry, pascal_file.width, pascal_file.height).getBoundingBox(), rel=1e-9, abs=1e-9)


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
                    bbox1 = PascalVocBBox_to_CocoBBox(pascal_annotation.geometry).getBoundingBox()
                    bbox2 = coco_annotation.geometry.getBoundingBox()
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
            orig_anns_sorted = sorted(orig_anns, key=lambda a: (a.geometry.x_min, a.geometry.y_min))
            reconv_anns_sorted = sorted(reconv_anns, key=lambda a: (a.geometry.x_min, a.geometry.y_min))
            
            # Check annotations per image
            for i, (orig_ann, reconv_ann) in enumerate(zip(orig_anns_sorted, reconv_anns_sorted)):
                name_orig = next(c.name for c in orig_file.categories if c.id == orig_ann.category_id)
                name_reconv = next(c.name for c in reconv_file.categories if c.id == reconv_ann.category_id)
                assert name_orig == name_reconv, f"Image {image_name}, annotation {i}: class '{name_orig}' != '{name_reconv}'"
                
                # Check bbox aprox values
                orig_bbox = orig_ann.geometry.getBoundingBox()
                reconv_bbox = reconv_ann.geometry.getBoundingBox()
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

def test_createml_original_and_createml_reconverted():
    """Test CreateML format idempotence through neutral conversion."""
    base_dir = os.path.dirname(__file__)
    createml_path = os.path.join(base_dir, "test_resources/CREATEML_TEST")

    # 1. Load CreateML Dataset
    createml_original: CreateMLFormat = CreateMLFormat.read_from_folder(createml_path)
    
    # 2. CreateML → Neutral → CreateML
    neutral_from_createml: NeutralFormat = CreateMLConverter.toNeutral(createml_original)
    createml_reconverted: CreateMLFormat = CreateMLConverter.fromNeutral(neutral_from_createml)
    
    # Check idempotence
    assert createml_original.name == createml_reconverted.name
    assert len(createml_original.files) == len(createml_reconverted.files)

    for orig_file, reconv_file in zip(createml_original.files, createml_reconverted.files):
        assert orig_file.filename == reconv_file.filename
        assert len(orig_file.annotations) == len(reconv_file.annotations)

        for i, (orig_ann, reconv_ann) in enumerate(zip(orig_file.annotations, reconv_file.annotations)):
            # Check bbox coordinates (CreateML uses center + dimensions)
            orig_bbox = orig_ann.geometry.getBoundingBox()
            reconv_bbox = reconv_ann.geometry.getBoundingBox()
            assert orig_bbox == pytest.approx(reconv_bbox, rel=1e-3, abs=1e-3)

            # Check label matches
            assert orig_ann.label == reconv_ann.label, f"Annotation {i}: label '{orig_ann.label}' != '{reconv_ann.label}'"


def test_createml_to_pascalvoc():
    """Test conversion from CreateML to Pascal VOC format."""
    base_dir = os.path.dirname(__file__)
    createml_path = os.path.join(base_dir, "test_resources/CREATEML_TEST")

    # 1. Load CreateML Dataset
    createml_original: CreateMLFormat = CreateMLFormat.read_from_folder(createml_path)
    
    # 2. CreateML → Neutral → Pascal VOC
    neutral_from_createml: NeutralFormat = CreateMLConverter.toNeutral(createml_original)
    pascal_converted: PascalVocFormat = PascalVocConverter.fromNeutral(neutral_from_createml)
    
    # Check Pascal VOC structure
    assert isinstance(pascal_converted, PascalVocFormat)
    assert len(pascal_converted.files) == len(createml_original.files)

    # Check file values
    for pascal_file, createml_file in zip(pascal_converted.files, createml_original.files):
        assert len(pascal_file.annotations) == len(createml_file.annotations)
        assert pascal_file.width > 0
        assert pascal_file.height > 0
        assert not pascal_file.segmented
        assert pascal_file.filename == createml_file.filename
        assert pascal_file.source.database == createml_original.name
        assert pascal_file.source.annotation == "Pascal Voc"
        assert pascal_file.source.image == ""

        # Check annotation values
        for pascal_annotation, createml_annotation in zip(pascal_file.annotations, createml_file.annotations):
            assert pascal_annotation.name == createml_annotation.label
            
            # Convert Pascal VOC back to CreateML format and compare
            converted_bbox = PascalVocBBox_to_CreateMLBBox(pascal_annotation.geometry)
            assert pytest.approx(converted_bbox.getBoundingBox(), rel=1e-3, abs=1e-3) == createml_annotation.geometry.getBoundingBox()


def test_pascalvoc_to_createml():
    """Test conversion from Pascal VOC to CreateML format."""
    base_dir = os.path.dirname(__file__)
    pascal_path = os.path.join(base_dir, "test_resources/PASCAL_VOC_TEST")

    # 1. Load Pascal VOC Dataset
    pascal_original: PascalVocFormat = PascalVocFormat.read_from_folder(pascal_path)
    
    # 2. Pascal VOC → Neutral → CreateML
    neutral_from_pascal: NeutralFormat = PascalVocConverter.toNeutral(pascal_original)
    createml_converted: CreateMLFormat = CreateMLConverter.fromNeutral(neutral_from_pascal)

    # Check CreateML structure
    assert isinstance(createml_converted, CreateMLFormat)
    assert len(createml_converted.files) == len(pascal_original.files)

    # Check file values
    for createml_file, pascal_file in zip(createml_converted.files, pascal_original.files):
        assert len(createml_file.annotations) == len(pascal_file.annotations)
        assert createml_file.filename == pascal_file.filename
        assert createml_file.height == pascal_file.height
        assert createml_file.width == pascal_file.width
        assert createml_file.depth == pascal_file.depth
        
        # Check annotation values
        for createml_annotation, pascal_annotation in zip(createml_file.annotations, pascal_file.annotations):
            assert createml_annotation.label == pascal_annotation.name
            
            # Convert Pascal VOC to CreateML and compare
            expected_createml_bbox = PascalVocBBox_to_CreateMLBBox(pascal_annotation.geometry)
            assert createml_annotation.geometry.getBoundingBox() == pytest.approx(expected_createml_bbox.getBoundingBox(), rel=1e-9, abs=1e-9)


def test_createml_to_coco():
    """Test conversion from CreateML to COCO format."""
    base_dir = os.path.dirname(__file__)
    createml_path = os.path.join(base_dir, "test_resources/CREATEML_TEST")

    # 1. Load CreateML Dataset
    createml_original: CreateMLFormat = CreateMLFormat.read_from_folder(createml_path)
    
    # 2. CreateML → Neutral → COCO
    neutral_from_createml: NeutralFormat = CreateMLConverter.toNeutral(createml_original)
    coco_converted: CocoFormat = CocoConverter.fromNeutral(neutral_from_createml)
    
    # Check COCO structure
    assert isinstance(coco_converted, CocoFormat)
    assert len(coco_converted.files) == 1
    
    total_annotations = sum(len(file.annotations) for file in createml_original.files)
    assert len(coco_converted.files[0].annotations) == total_annotations
    
    for coco_file in coco_converted.files:
        # Check image metadata
        for createml_file, coco_image in zip(createml_original.files, coco_file.images):
            assert coco_image.file_name == createml_file.filename
            assert coco_image.height > 0
            assert coco_image.width > 0

        # Check annotations
        for coco_annotation in coco_file.annotations:
            # Check annotation values
            assert coco_annotation.area == (coco_annotation.geometry.width * coco_annotation.geometry.height)
            coco_category_name = next((c.name for c in coco_file.categories if c.id == coco_annotation.category_id), None)
            
            # Find corresponding CreateML annotation
            createml_class_name = None
            found = False

            for createml_file in createml_original.files:
                if found:
                    break
                if createml_original.folder_path:
                    image_path = get_image_path(createml_original.folder_path, "images", createml_file.filename)
                    if image_path:
                        width, height, depth = get_image_info_from_file(image_path)
                        for createml_annotation in createml_file.annotations:
                            # Convert CreateML to Pascal VOC, then to COCO
                            pascal_bbox = CreateMLBBox_to_PascalVocBBox(createml_annotation.geometry)
                            expected_coco_bbox = PascalVocBBox_to_CocoBBox(pascal_bbox)
                            
                            bbox1 = expected_coco_bbox.getBoundingBox()
                            bbox2 = coco_annotation.geometry.getBoundingBox()
                            if bbox_almost_equal(bbox1, bbox2, 2):
                                createml_class_name = createml_annotation.label
                                found = True
                                break
            
            # Check category name matches
            assert coco_category_name == createml_class_name


def test_coco_to_createml():
    """Test conversion from COCO to CreateML format."""
    base_dir = os.path.dirname(__file__)
    coco_path = os.path.join(base_dir, "test_resources/COCO_TEST")

    # 1. Load COCO Dataset
    coco_original: CocoFormat = CocoFormat.read_from_folder(coco_path)
    
    # 2. COCO → Neutral → CreateML
    neutral_from_coco: NeutralFormat = CocoConverter.toNeutral(coco_original)
    createml_converted: CreateMLFormat = CreateMLConverter.fromNeutral(neutral_from_coco)
    
    # Check CreateML structure
    assert isinstance(createml_converted, CreateMLFormat)
    assert len(createml_converted.files) == len(coco_original.files[0].images)
    
    total_annotations = sum(len(file.annotations) for file in createml_converted.files)
    assert total_annotations == len(coco_original.files[0].annotations)
    
    # Check that every CreateML class is in COCO categories
    all_createml_labels = set()
    for file in createml_converted.files:
        for ann in file.annotations:
            all_createml_labels.add(ann.label)
    
    coco_category_names = {c.name for c in coco_original.files[0].categories}
    assert all_createml_labels.issubset(coco_category_names)


def test_tensorflow_csv_original_and_tensorflow_csv_reconverted():
    """Test TensorFlow CSV format idempotence through neutral conversion."""
    base_dir = os.path.dirname(__file__)
    tf_csv_path = os.path.join(base_dir, "test_resources/TENSORFLOW_CSV_TEST/annotations.csv")

    # 1. Load TensorFlow CSV Dataset
    tf_csv_original: TensorflowCsvFormat = TensorflowCsvFormat.read_from_folder(tf_csv_path)
    
    # 2. TensorFlow CSV → Neutral → TensorFlow CSV
    neutral_from_tf_csv: NeutralFormat = TensorflowCsvConverter.toNeutral(tf_csv_original)
    tf_csv_reconverted: TensorflowCsvFormat = TensorflowCsvConverter.fromNeutral(neutral_from_tf_csv)
    
    # Check idempotence
    assert tf_csv_original.name == tf_csv_reconverted.name
    assert len(tf_csv_original.files) == len(tf_csv_reconverted.files)

    for orig_file, reconv_file in zip(tf_csv_original.files, tf_csv_reconverted.files):
        assert orig_file.filename == reconv_file.filename
        assert orig_file.width == reconv_file.width
        assert orig_file.height == reconv_file.height
        assert len(orig_file.annotations) == len(reconv_file.annotations)

        for i, (orig_ann, reconv_ann) in enumerate(zip(orig_file.annotations, reconv_file.annotations)):
            # Check bbox coordinates
            orig_bbox = orig_ann.geometry.getBoundingBox()
            reconv_bbox = reconv_ann.geometry.getBoundingBox()
            assert orig_bbox == pytest.approx(reconv_bbox, rel=1e-6, abs=1e-6)

            # Check class name matches
            assert orig_ann.class_name == reconv_ann.class_name, f"Annotation {i}: class '{orig_ann.class_name}' != '{reconv_ann.class_name}'"


def test_tensorflow_csv_to_yolo():
    """Test conversion from TensorFlow CSV to YOLO format."""
    base_dir = os.path.dirname(__file__)
    tf_csv_path = os.path.join(base_dir, "test_resources/TENSORFLOW_CSV_TEST/annotations.csv")

    # 1. Load TensorFlow CSV Dataset
    tf_csv_original: TensorflowCsvFormat = TensorflowCsvFormat.read_from_folder(tf_csv_path)
    
    # 2. TensorFlow CSV → Neutral → YOLO
    neutral_from_tf_csv: NeutralFormat = TensorflowCsvConverter.toNeutral(tf_csv_original)
    yolo_converted: YoloFormat = YoloConverter.fromNeutral(neutral_from_tf_csv)
    
    # Check YOLO structure
    assert isinstance(yolo_converted, YoloFormat)
    assert len(yolo_converted.files) == len(tf_csv_original.files)

    # Check file values
    for yolo_file, tf_csv_file in zip(yolo_converted.files, tf_csv_original.files):
        assert len(yolo_file.annotations) == len(tf_csv_file.annotations)
        assert yolo_file.filename == tf_csv_file.filename
        assert yolo_file.width == tf_csv_file.width
        assert yolo_file.height == tf_csv_file.height
        
        # Check annotation values
        for yolo_annotation, tf_csv_annotation in zip(yolo_file.annotations, tf_csv_file.annotations):
            # Get class name from YOLO class_labels mapping
            yolo_class_name = yolo_converted.class_labels[yolo_annotation.id_class]
            assert yolo_class_name == tf_csv_annotation.class_name
            
            # Check bbox conversion (Pascal VOC to YOLO coordinates)
            expected_yolo_bbox = PascalVocBBox_to_YoloBBox(tf_csv_annotation.geometry, tf_csv_file.width, tf_csv_file.height)
            assert yolo_annotation.geometry.getBoundingBox() == pytest.approx(expected_yolo_bbox.getBoundingBox(), rel=1e-9, abs=1e-9)


def test_tensorflow_csv_to_pascalvoc():
    """Test conversion from TensorFlow CSV to Pascal VOC format."""
    base_dir = os.path.dirname(__file__)
    tf_csv_path = os.path.join(base_dir, "test_resources/TENSORFLOW_CSV_TEST/annotations.csv")

    # 1. Load TensorFlow CSV Dataset
    tf_csv_original: TensorflowCsvFormat = TensorflowCsvFormat.read_from_folder(tf_csv_path)
    
    # 2. TensorFlow CSV → Neutral → Pascal VOC
    neutral_from_tf_csv: NeutralFormat = TensorflowCsvConverter.toNeutral(tf_csv_original)
    pascal_converted: PascalVocFormat = PascalVocConverter.fromNeutral(neutral_from_tf_csv)
    
    # Check Pascal VOC structure
    assert isinstance(pascal_converted, PascalVocFormat)
    assert len(pascal_converted.files) == len(tf_csv_original.files)

    # Check file values
    for pascal_file, tf_csv_file in zip(pascal_converted.files, tf_csv_original.files):
        assert len(pascal_file.annotations) == len(tf_csv_file.annotations)
        assert pascal_file.width == tf_csv_file.width
        assert pascal_file.height == tf_csv_file.height
        assert not pascal_file.segmented
        assert pascal_file.filename == tf_csv_file.filename
        assert pascal_file.source.database == tf_csv_original.name
        assert pascal_file.source.annotation == "Pascal Voc"
        assert pascal_file.source.image == ""

        # Check annotation values
        for pascal_annotation, tf_csv_annotation in zip(pascal_file.annotations, tf_csv_file.annotations):
            assert pascal_annotation.name == tf_csv_annotation.class_name
            
            # Both use PascalVocBoundingBox, so coordinates should be identical
            orig_bbox = tf_csv_annotation.geometry.getBoundingBox()
            pascal_bbox = pascal_annotation.geometry.getBoundingBox()
            assert orig_bbox == pytest.approx(pascal_bbox, rel=1e-6, abs=1e-6)


def test_tensorflow_csv_to_coco():
    """Test conversion from TensorFlow CSV to COCO format."""
    base_dir = os.path.dirname(__file__)
    tf_csv_path = os.path.join(base_dir, "test_resources/TENSORFLOW_CSV_TEST/annotations.csv")

    # 1. Load TensorFlow CSV Dataset
    tf_csv_original: TensorflowCsvFormat = TensorflowCsvFormat.read_from_folder(tf_csv_path)
    
    # 2. TensorFlow CSV → Neutral → COCO
    neutral_from_tf_csv: NeutralFormat = TensorflowCsvConverter.toNeutral(tf_csv_original)
    coco_converted: CocoFormat = CocoConverter.fromNeutral(neutral_from_tf_csv)
    
    # Check COCO structure
    assert isinstance(coco_converted, CocoFormat)
    assert len(coco_converted.files) == 1
    
    # Calculate total annotations
    total_annotations = sum(len(file.annotations) for file in tf_csv_original.files)
    assert len(coco_converted.files[0].annotations) == total_annotations
    
    for coco_file in coco_converted.files:
        # Check image metadata
        for tf_csv_file, coco_image in zip(tf_csv_original.files, coco_file.images):
            assert coco_image.file_name == tf_csv_file.filename
            assert coco_image.height == tf_csv_file.height
            assert coco_image.width == tf_csv_file.width

        # Check annotations
        for coco_annotation in coco_file.annotations:
            # Check annotation values
            assert coco_annotation.area == (coco_annotation.geometry.width * coco_annotation.geometry.height)
            coco_category_name = next((c.name for c in coco_file.categories if c.id == coco_annotation.category_id), None)
            
            # Find corresponding TensorFlow CSV annotation by bbox matching
            tf_csv_class_name = None
            found = False

            for tf_csv_file in tf_csv_original.files:
                if found:
                    break
                for tf_csv_annotation in tf_csv_file.annotations:
                    # Convert Pascal VOC to COCO format for comparison
                    expected_coco_bbox = PascalVocBBox_to_CocoBBox(tf_csv_annotation.geometry)
                    bbox1 = expected_coco_bbox.getBoundingBox()
                    bbox2 = coco_annotation.geometry.getBoundingBox()
                    if bbox_almost_equal(bbox1, bbox2, 2):
                        tf_csv_class_name = tf_csv_annotation.class_name
                        found = True
                        break
            
            # Check category name matches
            assert coco_category_name == tf_csv_class_name


def test_yolo_to_tensorflow_csv():
    """Test conversion from YOLO to TensorFlow CSV format."""
    base_dir = os.path.dirname(__file__)
    yolo_path = os.path.join(base_dir, "test_resources/YOLO_TEST")

    # 1. Load YOLO Dataset
    yolo_original: YoloFormat = YoloFormat.read_from_folder(yolo_path)
    
    # 2. YOLO → Neutral → TensorFlow CSV
    neutral_from_yolo: NeutralFormat = YoloConverter.toNeutral(yolo_original)
    tf_csv_converted: TensorflowCsvFormat = TensorflowCsvConverter.fromNeutral(neutral_from_yolo)
    
    # Check TensorFlow CSV structure
    assert isinstance(tf_csv_converted, TensorflowCsvFormat)
    assert len(tf_csv_converted.files) == len(yolo_original.files)

    # Check file values
    for tf_csv_file, yolo_file in zip(tf_csv_converted.files, yolo_original.files):
        assert len(tf_csv_file.annotations) == len(yolo_file.annotations)
        assert tf_csv_file.filename == yolo_file.filename
        
        # Check annotation values
        for tf_csv_annotation, yolo_annotation in zip(tf_csv_file.annotations, yolo_file.annotations):
            # Get class name from YOLO class_labels mapping
            yolo_class_name = yolo_original.class_labels[yolo_annotation.id_class]
            assert tf_csv_annotation.class_name == yolo_class_name


def test_pascalvoc_to_tensorflow_csv():
    """Test conversion from Pascal VOC to TensorFlow CSV format."""
    base_dir = os.path.dirname(__file__)
    pascal_path = os.path.join(base_dir, "test_resources/PASCAL_VOC_TEST")

    # 1. Load Pascal VOC Dataset
    pascal_original: PascalVocFormat = PascalVocFormat.read_from_folder(pascal_path)
    
    # 2. Pascal VOC → Neutral → TensorFlow CSV
    neutral_from_pascal: NeutralFormat = PascalVocConverter.toNeutral(pascal_original)
    tf_csv_converted: TensorflowCsvFormat = TensorflowCsvConverter.fromNeutral(neutral_from_pascal)

    # Check TensorFlow CSV structure
    assert isinstance(tf_csv_converted, TensorflowCsvFormat)
    assert len(tf_csv_converted.files) == len(pascal_original.files)

    # Check file values
    for tf_csv_file, pascal_file in zip(tf_csv_converted.files, pascal_original.files):
        assert len(tf_csv_file.annotations) == len(pascal_file.annotations)
        assert tf_csv_file.filename == pascal_file.filename
        assert tf_csv_file.width == pascal_file.width
        assert tf_csv_file.height == pascal_file.height
        
        # Check annotation values
        for tf_csv_annotation, pascal_annotation in zip(tf_csv_file.annotations, pascal_file.annotations):
            assert tf_csv_annotation.class_name == pascal_annotation.name
            
            # Both use PascalVocBoundingBox, so coordinates should be identical
            tf_csv_bbox = tf_csv_annotation.geometry.getBoundingBox()
            pascal_bbox = pascal_annotation.geometry.getBoundingBox()
            assert tf_csv_bbox == pytest.approx(pascal_bbox, rel=1e-6, abs=1e-6)


def test_coco_to_tensorflow_csv():
    """Test conversion from COCO to TensorFlow CSV format."""
    base_dir = os.path.dirname(__file__)
    coco_path = os.path.join(base_dir, "test_resources/COCO_TEST")

    # 1. Load COCO Dataset
    coco_original: CocoFormat = CocoFormat.read_from_folder(coco_path)
    
    # 2. COCO → Neutral → TensorFlow CSV
    neutral_from_coco: NeutralFormat = CocoConverter.toNeutral(coco_original)
    tf_csv_converted: TensorflowCsvFormat = TensorflowCsvConverter.fromNeutral(neutral_from_coco)
    
    # Check TensorFlow CSV structure
    assert isinstance(tf_csv_converted, TensorflowCsvFormat)
    assert len(tf_csv_converted.files) == len(coco_original.files[0].images)
    
    # Calculate total annotations
    total_annotations = sum(len(file.annotations) for file in tf_csv_converted.files)
    assert total_annotations == len(coco_original.files[0].annotations)
    
    # Check that every TensorFlow CSV class is in COCO categories
    unique_tf_csv_classes = tf_csv_converted.get_unique_classes()
    coco_category_names = {c.name for c in coco_original.files[0].categories}
    assert unique_tf_csv_classes.issubset(coco_category_names)
    
    # Check file correspondence
    for i, tf_csv_file in enumerate(tf_csv_converted.files):
        corresponding_coco_image = coco_original.files[0].images[i]
        assert tf_csv_file.filename == corresponding_coco_image.file_name
        assert tf_csv_file.width == corresponding_coco_image.width
        assert tf_csv_file.height == corresponding_coco_image.height


def test_labelme_original_and_labelme_reconverted():
    """Test LabelMe format idempotence through neutral conversion."""
    base_dir = os.path.dirname(__file__)
    labelme_path = os.path.join(base_dir, "test_resources/LABELME_TEST")

    # 1. Load LabelMe Dataset
    labelme_original: LabelMeFormat = LabelMeFormat.read_from_folder(labelme_path)
    
    # 2. LabelMe → Neutral → LabelMe
    neutral_from_labelme: NeutralFormat = LabelMeConverter.toNeutral(labelme_original)
    labelme_reconverted: LabelMeFormat = LabelMeConverter.fromNeutral(neutral_from_labelme)
    
    # Check idempotence
    assert labelme_original.name == labelme_reconverted.name
    assert len(labelme_original.files) == len(labelme_reconverted.files)

    for orig_file, reconv_file in zip(labelme_original.files, labelme_reconverted.files):
        assert orig_file.filename == reconv_file.filename
        assert orig_file.imageWidth == reconv_file.imageWidth
        assert orig_file.imageHeight == reconv_file.imageHeight
        assert len(orig_file.annotations) == len(reconv_file.annotations)

        for i, (orig_ann, reconv_ann) in enumerate(zip(orig_file.annotations, reconv_file.annotations)):
            # Check label matches
            assert orig_ann.label == reconv_ann.label, f"Annotation {i}: label '{orig_ann.label}' != '{reconv_ann.label}'"
            
            # Check shape type matches
            assert orig_ann.geometry.shape_type == reconv_ann.geometry.shape_type, f"Annotation {i}: shape_type mismatch"
            
            # Check coordinates match (allowing for small floating point differences)
            orig_coords = orig_ann.geometry.getCoordinates()
            reconv_coords = reconv_ann.geometry.getCoordinates()
            
            for j, (orig_coord, reconv_coord) in enumerate(zip(orig_coords, reconv_coords)):
                assert orig_coord == pytest.approx(reconv_coord, rel=1e-3, abs=1e-3), f"Annotation {i}, coordinate {j} mismatch"
            
            # Check optional attributes
            assert orig_ann.group_id == reconv_ann.group_id
            assert orig_ann.description == reconv_ann.description


def test_labelme_to_pascalvoc():
    """Test conversion from LabelMe to Pascal VOC format."""
    base_dir = os.path.dirname(__file__)
    labelme_path = os.path.join(base_dir, "test_resources/LABELME_TEST")

    # 1. Load LabelMe Dataset
    labelme_original: LabelMeFormat = LabelMeFormat.read_from_folder(labelme_path)
    
    # 2. LabelMe → Neutral → Pascal VOC
    neutral_from_labelme: NeutralFormat = LabelMeConverter.toNeutral(labelme_original)
    pascal_converted: PascalVocFormat = PascalVocConverter.fromNeutral(neutral_from_labelme)
    
    # Check Pascal VOC structure
    assert isinstance(pascal_converted, PascalVocFormat)
    assert len(pascal_converted.files) == len(labelme_original.files)

    # Check file values
    for pascal_file, labelme_file in zip(pascal_converted.files, labelme_original.files):
        assert len(pascal_file.annotations) == len(labelme_file.annotations)
        assert pascal_file.width == labelme_file.imageWidth
        assert pascal_file.height == labelme_file.imageHeight
        assert not pascal_file.segmented
        assert pascal_file.filename == labelme_file.filename
        assert pascal_file.source.database == labelme_original.name
        assert pascal_file.source.annotation == "Pascal Voc"
        assert pascal_file.source.image == ""

        # Check annotation values
        for pascal_annotation, labelme_annotation in zip(pascal_file.annotations, labelme_file.annotations):
            assert pascal_annotation.name == labelme_annotation.label
            
            # Check that bounding boxes match (LabelMe shapes converted to bounding boxes)
            labelme_bbox = labelme_annotation.geometry.getBoundingBox()
            pascal_bbox = pascal_annotation.geometry
            assert isinstance(labelme_bbox, PascalVocBoundingBox)
            assert labelme_bbox.x_min == pytest.approx(pascal_bbox.x_min, rel=1e-3, abs=1e-3)
            assert labelme_bbox.y_min == pytest.approx(pascal_bbox.y_min, rel=1e-3, abs=1e-3)
            assert labelme_bbox.x_max == pytest.approx(pascal_bbox.x_max, rel=1e-3, abs=1e-3)
            assert labelme_bbox.y_max == pytest.approx(pascal_bbox.y_max, rel=1e-3, abs=1e-3)


def test_labelme_to_yolo():
    """Test conversion from LabelMe to YOLO format."""
    base_dir = os.path.dirname(__file__)
    labelme_path = os.path.join(base_dir, "test_resources/LABELME_TEST")

    # 1. Load LabelMe Dataset
    labelme_original: LabelMeFormat = LabelMeFormat.read_from_folder(labelme_path)
    
    # 2. LabelMe → Neutral → YOLO
    neutral_from_labelme: NeutralFormat = LabelMeConverter.toNeutral(labelme_original)
    yolo_converted: YoloFormat = YoloConverter.fromNeutral(neutral_from_labelme)
    
    # Check YOLO structure
    assert isinstance(yolo_converted, YoloFormat)
    assert len(yolo_converted.files) == len(labelme_original.files)

    # Check file values
    for yolo_file, labelme_file in zip(yolo_converted.files, labelme_original.files):
        assert len(yolo_file.annotations) == len(labelme_file.annotations)
        assert yolo_file.filename == labelme_file.filename
        assert yolo_file.width == labelme_file.imageWidth
        assert yolo_file.height == labelme_file.imageHeight
        
        # Check annotation values
        for yolo_annotation, labelme_annotation in zip(yolo_file.annotations, labelme_file.annotations):
            # Get class name from YOLO class_labels mapping
            yolo_class_name = yolo_converted.class_labels[yolo_annotation.id_class]
            assert yolo_class_name == labelme_annotation.label
            
            # Check bbox conversion (LabelMe bounding box to YOLO coordinates)
            labelme_bbox = labelme_annotation.geometry.getBoundingBox()
            assert isinstance(labelme_bbox, PascalVocBoundingBox)
            expected_yolo_bbox = PascalVocBBox_to_YoloBBox(labelme_bbox, labelme_file.imageWidth, labelme_file.imageHeight)
            assert yolo_annotation.geometry.getBoundingBox() == pytest.approx(expected_yolo_bbox.getBoundingBox(), rel=1e-3, abs=1e-3)


def test_labelme_to_coco():
    """Test conversion from LabelMe to COCO format."""
    base_dir = os.path.dirname(__file__)
    labelme_path = os.path.join(base_dir, "test_resources/LABELME_TEST")

    # 1. Load LabelMe Dataset
    labelme_original: LabelMeFormat = LabelMeFormat.read_from_folder(labelme_path)
    
    # 2. LabelMe → Neutral → COCO
    neutral_from_labelme: NeutralFormat = LabelMeConverter.toNeutral(labelme_original)
    coco_converted: CocoFormat = CocoConverter.fromNeutral(neutral_from_labelme)
    
    # Check COCO structure
    assert isinstance(coco_converted, CocoFormat)
    assert len(coco_converted.files) == 1
    
    # Calculate total annotations
    total_annotations = sum(len(file.annotations) for file in labelme_original.files)
    assert len(coco_converted.files[0].annotations) == total_annotations
    
    for coco_file in coco_converted.files:
        # Check image metadata
        for labelme_file, coco_image in zip(labelme_original.files, coco_file.images):
            assert coco_image.file_name == labelme_file.filename
            assert coco_image.height == labelme_file.imageHeight
            assert coco_image.width == labelme_file.imageWidth

        # Check annotations
        for coco_annotation in coco_file.annotations:
            # Check annotation values
            assert coco_annotation.area == (coco_annotation.geometry.width * coco_annotation.geometry.height)
            coco_category_name = next((c.name for c in coco_file.categories if c.id == coco_annotation.category_id), None)
            
            # Find corresponding LabelMe annotation by bbox matching
            labelme_class_name = None
            found = False

            for labelme_file in labelme_original.files:
                if found:
                    break
                for labelme_annotation in labelme_file.annotations:
                    # Convert LabelMe bounding box to COCO format for comparison
                    labelme_bbox = labelme_annotation.geometry.getBoundingBox()
                    assert isinstance(labelme_bbox, PascalVocBoundingBox)
                    expected_coco_bbox = PascalVocBBox_to_CocoBBox(labelme_bbox)
                    bbox1 = expected_coco_bbox.getBoundingBox()
                    bbox2 = coco_annotation.geometry.getBoundingBox()
                    if bbox_almost_equal(bbox1, bbox2, 2):
                        labelme_class_name = labelme_annotation.label
                        found = True
                        break
            
            # Check category name matches
            assert coco_category_name == labelme_class_name


def test_pascalvoc_to_labelme():
    """Test conversion from Pascal VOC to LabelMe format."""
    base_dir = os.path.dirname(__file__)
    pascal_path = os.path.join(base_dir, "test_resources/PASCAL_VOC_TEST")

    # 1. Load Pascal VOC Dataset
    pascal_original: PascalVocFormat = PascalVocFormat.read_from_folder(pascal_path)
    
    # 2. Pascal VOC → Neutral → LabelMe
    neutral_from_pascal: NeutralFormat = PascalVocConverter.toNeutral(pascal_original)
    labelme_converted: LabelMeFormat = LabelMeConverter.fromNeutral(neutral_from_pascal)

    # Check LabelMe structure
    assert isinstance(labelme_converted, LabelMeFormat)
    assert len(labelme_converted.files) == len(pascal_original.files)

    # Check file values
    for labelme_file, pascal_file in zip(labelme_converted.files, pascal_original.files):
        assert len(labelme_file.annotations) == len(pascal_file.annotations)
        assert labelme_file.filename == pascal_file.filename
        assert labelme_file.imageHeight == pascal_file.height
        assert labelme_file.imageWidth == pascal_file.width
        
        # Check annotation values
        for labelme_annotation, pascal_annotation in zip(labelme_file.annotations, pascal_file.annotations):
            assert labelme_annotation.label == pascal_annotation.name
            
            # Check that shapes are rectangles (converted from Pascal VOC bounding boxes)
            assert labelme_annotation.geometry.shape_type == "rectangle"
            
            # Check bounding box coordinates match
            labelme_bbox = labelme_annotation.geometry.getBoundingBox()
            pascal_bbox = pascal_annotation.geometry
            assert isinstance(labelme_bbox, PascalVocBoundingBox)
            assert labelme_bbox.x_min == pytest.approx(pascal_bbox.x_min, rel=1e-6, abs=1e-6)
            assert labelme_bbox.y_min == pytest.approx(pascal_bbox.y_min, rel=1e-6, abs=1e-6)
            assert labelme_bbox.x_max == pytest.approx(pascal_bbox.x_max, rel=1e-6, abs=1e-6)
            assert labelme_bbox.y_max == pytest.approx(pascal_bbox.y_max, rel=1e-6, abs=1e-6)


def test_yolo_to_labelme():
    """Test conversion from YOLO to LabelMe format."""
    base_dir = os.path.dirname(__file__)
    yolo_path = os.path.join(base_dir, "test_resources/YOLO_TEST")

    # 1. Load YOLO Dataset
    yolo_original: YoloFormat = YoloFormat.read_from_folder(yolo_path)
    
    # 2. YOLO → Neutral → LabelMe
    neutral_from_yolo: NeutralFormat = YoloConverter.toNeutral(yolo_original)
    labelme_converted: LabelMeFormat = LabelMeConverter.fromNeutral(neutral_from_yolo)
    
    # Check LabelMe structure
    assert isinstance(labelme_converted, LabelMeFormat)
    assert len(labelme_converted.files) == len(yolo_original.files)

    # Check file values
    for labelme_file, yolo_file in zip(labelme_converted.files, yolo_original.files):
        assert len(labelme_file.annotations) == len(yolo_file.annotations)
        assert labelme_file.filename == yolo_file.filename
        
        # Check annotation values
        for labelme_annotation, yolo_annotation in zip(labelme_file.annotations, yolo_file.annotations):
            # Get class name from YOLO class_labels mapping
            yolo_class_name = yolo_original.class_labels[yolo_annotation.id_class]
            assert labelme_annotation.label == yolo_class_name
            
            # Check that shapes are rectangles (converted from YOLO bounding boxes)
            assert labelme_annotation.geometry.shape_type == "rectangle"


def test_coco_to_labelme():
    """Test conversion from COCO to LabelMe format."""
    base_dir = os.path.dirname(__file__)
    coco_path = os.path.join(base_dir, "test_resources/COCO_TEST")

    # 1. Load COCO Dataset
    coco_original: CocoFormat = CocoFormat.read_from_folder(coco_path)
    
    # 2. COCO → Neutral → LabelMe
    neutral_from_coco: NeutralFormat = CocoConverter.toNeutral(coco_original)
    labelme_converted: LabelMeFormat = LabelMeConverter.fromNeutral(neutral_from_coco)
    
    # Check LabelMe structure
    assert isinstance(labelme_converted, LabelMeFormat)
    assert len(labelme_converted.files) == len(coco_original.files[0].images)
    
    # Calculate total annotations
    total_annotations = sum(len(file.annotations) for file in labelme_converted.files)
    assert total_annotations == len(coco_original.files[0].annotations)
    
    # Check that every LabelMe class is in COCO categories
    all_labelme_labels = set()
    for file in labelme_converted.files:
        for ann in file.annotations:
            all_labelme_labels.add(ann.label)
    
    coco_category_names = {c.name for c in coco_original.files[0].categories}
    assert all_labelme_labels.issubset(coco_category_names)
    
    # Check file correspondence
    for i, labelme_file in enumerate(labelme_converted.files):
        corresponding_coco_image = coco_original.files[0].images[i]
        assert labelme_file.filename == corresponding_coco_image.file_name
        assert labelme_file.imageWidth == corresponding_coco_image.width
        assert labelme_file.imageHeight == corresponding_coco_image.height


def test_labelme_shape_preservation():
    """Test that LabelMe shape information is preserved through neutral conversion."""
    base_dir = os.path.dirname(__file__)
    labelme_path = os.path.join(base_dir, "test_resources/LABELME_TEST")

    # 1. Load LabelMe Dataset
    labelme_original: LabelMeFormat = LabelMeFormat.read_from_folder(labelme_path)
    
    # 2. LabelMe → Neutral → LabelMe
    neutral_from_labelme: NeutralFormat = LabelMeConverter.toNeutral(labelme_original)
    labelme_reconverted: LabelMeFormat = LabelMeConverter.fromNeutral(neutral_from_labelme)
    
    # Check that shape-specific attributes are preserved
    for orig_file, reconv_file in zip(labelme_original.files, labelme_reconverted.files):
        for orig_ann, reconv_ann in zip(orig_file.annotations, reconv_file.annotations):
            # Check shape type preservation
            assert orig_ann.geometry.shape_type == reconv_ann.geometry.shape_type
            
            # Check coordinates preservation for different shape types
            if orig_ann.geometry.shape_type == "circle":
                # Check radius preservation
                assert isinstance(orig_ann.geometry, LabelMeCircle)
                orig_radius = orig_ann.geometry.radius
                assert isinstance(reconv_ann.geometry, LabelMeCircle)
                reconv_radius = reconv_ann.geometry.radius
                assert orig_radius == pytest.approx(reconv_radius, rel=1e-3, abs=1e-3)
            
            # Check that complex shapes maintain their coordinate structure
            orig_coords = orig_ann.geometry.getCoordinates()
            reconv_coords = reconv_ann.geometry.getCoordinates()
            assert len(orig_coords) == len(reconv_coords)


def test_labelme_metadata_preservation():
    """Test that LabelMe metadata is preserved through neutral conversion."""
    base_dir = os.path.dirname(__file__)
    labelme_path = os.path.join(base_dir, "test_resources/LABELME_TEST")

    # 1. Load LabelMe Dataset
    labelme_original: LabelMeFormat = LabelMeFormat.read_from_folder(labelme_path)
    
    # 2. LabelMe → Neutral → LabelMe
    neutral_from_labelme: NeutralFormat = LabelMeConverter.toNeutral(labelme_original)
    labelme_reconverted: LabelMeFormat = LabelMeConverter.fromNeutral(neutral_from_labelme)
    
    # Check that LabelMe-specific metadata is preserved
    for orig_file, reconv_file in zip(labelme_original.files, labelme_reconverted.files):
        # Check version preservation
        assert orig_file.version == reconv_file.version
        
        # Check flags preservation
        assert orig_file.flags == reconv_file.flags
        
        for orig_ann, reconv_ann in zip(orig_file.annotations, reconv_file.annotations):
            # Check annotation-level metadata
            assert orig_ann.group_id == reconv_ann.group_id
            assert orig_ann.description == reconv_ann.description
            assert orig_ann.flags == reconv_ann.flags


def test_vgg_original_and_vgg_reconverted():
    """Test VGG format idempotence through neutral conversion."""
    base_dir = os.path.dirname(__file__)
    vgg_path = os.path.join(base_dir, "test_resources/VGG_TEST/annotations.json")

    # 1. Load VGG Dataset
    vgg_original: VGGFormat = VGGFormat.read_from_file(vgg_path)
    
    # 2. VGG → Neutral → VGG
    neutral_from_vgg: NeutralFormat = VGGConverter.toNeutral(vgg_original)
    vgg_reconverted: VGGFormat = VGGConverter.fromNeutral(neutral_from_vgg)
    
    # Check idempotence
    assert vgg_original.name == vgg_reconverted.name
    assert len(vgg_original.files) == len(vgg_reconverted.files)

    for orig_file, reconv_file in zip(vgg_original.files, vgg_reconverted.files):
        assert orig_file.filename == reconv_file.filename
        assert orig_file.size == reconv_file.size
        assert len(orig_file.annotations) == len(reconv_file.annotations)
        assert orig_file.file_attributes == reconv_file.file_attributes

        for i, (orig_ann, reconv_ann) in enumerate(zip(orig_file.annotations, reconv_file.annotations)):
            # Check shape type matches
            assert orig_ann.geometry.shape_type == reconv_ann.geometry.shape_type, f"Annotation {i}: shape_type mismatch"
            
            # Check coordinates match
            orig_coords = orig_ann.geometry.getCoordinates()
            reconv_coords = reconv_ann.geometry.getCoordinates()
            assert orig_coords == reconv_coords, f"Annotation {i}: coordinates mismatch"
            
            # Check region attributes
            assert orig_ann.region_attributes == reconv_ann.region_attributes, f"Annotation {i}: region_attributes mismatch"
            
            # Check bbox conversion
            orig_bbox = orig_ann.geometry.getBoundingBox()
            reconv_bbox = reconv_ann.geometry.getBoundingBox()
            assert orig_bbox == pytest.approx(reconv_bbox, rel=1e-3, abs=1e-3)


def test_vgg_to_pascalvoc():
    """Test conversion from VGG to Pascal VOC format."""
    base_dir = os.path.dirname(__file__)
    vgg_path = os.path.join(base_dir, "test_resources/VGG_TEST/annotations.json")

    # 1. Load VGG Dataset
    vgg_original: VGGFormat = VGGFormat.read_from_file(vgg_path)
    
    # 2. VGG → Neutral → Pascal VOC
    neutral_from_vgg: NeutralFormat = VGGConverter.toNeutral(vgg_original)
    pascal_converted: PascalVocFormat = PascalVocConverter.fromNeutral(neutral_from_vgg)
    
    # Check Pascal VOC structure
    assert isinstance(pascal_converted, PascalVocFormat)
    assert len(pascal_converted.files) == len(vgg_original.files)

    # Check file values
    for pascal_file, vgg_file in zip(pascal_converted.files, vgg_original.files):
        assert len(pascal_file.annotations) == len(vgg_file.annotations)
        assert pascal_file.width > 0
        assert pascal_file.height > 0
        assert not pascal_file.segmented
        assert pascal_file.filename == vgg_file.filename
        assert pascal_file.source.database == vgg_original.name
        assert pascal_file.source.annotation == "Pascal Voc"
        assert pascal_file.source.image == ""

        # Check annotation values
        for pascal_annotation, vgg_annotation in zip(pascal_file.annotations, vgg_file.annotations):
            # Extract class name from VGG region attributes
            expected_class = _extract_class_from_vgg_attributes(vgg_annotation.region_attributes)
            assert pascal_annotation.name == expected_class
            
            # Check bbox matches VGG shape bounding box
            vgg_bbox = vgg_annotation.geometry.getBoundingBox()
            pascal_bbox = pascal_annotation.geometry
            assert isinstance(vgg_bbox, PascalVocBoundingBox)
            assert vgg_bbox.x_min == pytest.approx(pascal_bbox.x_min, rel=1e-3, abs=1e-3)
            assert vgg_bbox.y_min == pytest.approx(pascal_bbox.y_min, rel=1e-3, abs=1e-3)
            assert vgg_bbox.x_max == pytest.approx(pascal_bbox.x_max, rel=1e-3, abs=1e-3)
            assert vgg_bbox.y_max == pytest.approx(pascal_bbox.y_max, rel=1e-3, abs=1e-3)


def test_vgg_to_yolo():
    """Test conversion from VGG to YOLO format."""
    base_dir = os.path.dirname(__file__)
    vgg_path = os.path.join(base_dir, "test_resources/VGG_TEST/annotations.json")

    # 1. Load VGG Dataset
    vgg_original: VGGFormat = VGGFormat.read_from_file(vgg_path)
    
    # 2. VGG → Neutral → YOLO
    neutral_from_vgg: NeutralFormat = VGGConverter.toNeutral(vgg_original)
    yolo_converted: YoloFormat = YoloConverter.fromNeutral(neutral_from_vgg)
    
    # Check YOLO structure
    assert isinstance(yolo_converted, YoloFormat)
    assert len(yolo_converted.files) == len(vgg_original.files)

    # Check file values
    for yolo_file, vgg_file in zip(yolo_converted.files, vgg_original.files):
        assert len(yolo_file.annotations) == len(vgg_file.annotations)
        assert yolo_file.filename == vgg_file.filename
        assert yolo_file.width and yolo_file.height
        assert yolo_file.width > 0
        assert yolo_file.height > 0
        
        # Check annotation values
        for yolo_annotation, vgg_annotation in zip(yolo_file.annotations, vgg_file.annotations):
            # Get class name from YOLO class_labels mapping
            yolo_class_name = yolo_converted.class_labels[yolo_annotation.id_class]
            expected_class = _extract_class_from_vgg_attributes(vgg_annotation.region_attributes)
            assert yolo_class_name == expected_class
            
            # Check bbox conversion (VGG bounding box to YOLO coordinates)
            vgg_bbox = vgg_annotation.geometry.getBoundingBox()
            assert isinstance(vgg_bbox, PascalVocBoundingBox)
            expected_yolo_bbox = PascalVocBBox_to_YoloBBox(vgg_bbox, yolo_file.width, yolo_file.height)
            assert yolo_annotation.geometry.getBoundingBox() == pytest.approx(expected_yolo_bbox.getBoundingBox(), rel=1e-3, abs=1e-3)


def test_vgg_to_coco():
    """Test conversion from VGG to COCO format."""
    base_dir = os.path.dirname(__file__)
    vgg_path = os.path.join(base_dir, "test_resources/VGG_TEST/annotations.json")

    # 1. Load VGG Dataset
    vgg_original: VGGFormat = VGGFormat.read_from_file(vgg_path)
    
    # 2. VGG → Neutral → COCO
    neutral_from_vgg: NeutralFormat = VGGConverter.toNeutral(vgg_original)
    coco_converted: CocoFormat = CocoConverter.fromNeutral(neutral_from_vgg)
    
    # Check COCO structure
    assert isinstance(coco_converted, CocoFormat)
    assert len(coco_converted.files) == 1
    
    # Calculate total annotations
    total_annotations = sum(len(file.annotations) for file in vgg_original.files)
    assert len(coco_converted.files[0].annotations) == total_annotations
    
    for coco_file in coco_converted.files:
        # Check image metadata
        for vgg_file, coco_image in zip(vgg_original.files, coco_file.images):
            assert coco_image.file_name == vgg_file.filename
            assert coco_image.height > 0
            assert coco_image.width > 0

        # Check annotations
        for coco_annotation in coco_file.annotations:
            # Check annotation values
            assert coco_annotation.area == (coco_annotation.geometry.width * coco_annotation.geometry.height)
            coco_category_name = next((c.name for c in coco_file.categories if c.id == coco_annotation.category_id), None)
            
            # Find corresponding VGG annotation by bbox matching
            vgg_class_name = None
            found = False

            for vgg_file in vgg_original.files:
                if found:
                    break
                for vgg_annotation in vgg_file.annotations:
                    # Convert VGG bounding box to COCO format for comparison
                    vgg_bbox = vgg_annotation.geometry.getBoundingBox()
                    assert isinstance(vgg_bbox, PascalVocBoundingBox)
                    expected_coco_bbox = PascalVocBBox_to_CocoBBox(vgg_bbox)
                    bbox1 = expected_coco_bbox.getBoundingBox()
                    bbox2 = coco_annotation.geometry.getBoundingBox()
                    if bbox_almost_equal(bbox1, bbox2, 2):
                        vgg_class_name = _extract_class_from_vgg_attributes(vgg_annotation.region_attributes)
                        found = True
                        break
            
            # Check category name matches
            assert coco_category_name == vgg_class_name


def test_pascalvoc_to_vgg():
    """Test conversion from Pascal VOC to VGG format."""
    base_dir = os.path.dirname(__file__)
    pascal_path = os.path.join(base_dir, "test_resources/PASCAL_VOC_TEST")

    # 1. Load Pascal VOC Dataset
    pascal_original: PascalVocFormat = PascalVocFormat.read_from_folder(pascal_path)
    
    # 2. Pascal VOC → Neutral → VGG
    neutral_from_pascal: NeutralFormat = PascalVocConverter.toNeutral(pascal_original)
    vgg_converted: VGGFormat = VGGConverter.fromNeutral(neutral_from_pascal)

    # Check VGG structure
    assert isinstance(vgg_converted, VGGFormat)
    assert len(vgg_converted.files) == len(pascal_original.files)

    # Check file values
    for vgg_file, pascal_file in zip(vgg_converted.files, pascal_original.files):
        assert len(vgg_file.annotations) == len(pascal_file.annotations)
        assert vgg_file.filename == pascal_file.filename
        
        # Check annotation values
        for vgg_annotation, pascal_annotation in zip(vgg_file.annotations, pascal_file.annotations):
            # Check that VGG shapes are rectangles (converted from Pascal VOC bounding boxes)
            assert vgg_annotation.geometry.shape_type == "rect"
            
            # Check region attributes contain class name
            assert pascal_annotation.name in vgg_annotation.region_attributes.values()
            
            # Check bounding box coordinates match
            vgg_bbox = vgg_annotation.geometry.getBoundingBox()
            pascal_bbox = pascal_annotation.geometry
            assert isinstance(vgg_bbox, PascalVocBoundingBox)
            assert vgg_bbox.x_min == pytest.approx(pascal_bbox.x_min, rel=1e-6, abs=1e-6)
            assert vgg_bbox.y_min == pytest.approx(pascal_bbox.y_min, rel=1e-6, abs=1e-6)
            assert vgg_bbox.x_max == pytest.approx(pascal_bbox.x_max, rel=1e-6, abs=1e-6)
            assert vgg_bbox.y_max == pytest.approx(pascal_bbox.y_max, rel=1e-6, abs=1e-6)


def test_yolo_to_vgg():
    """Test conversion from YOLO to VGG format."""
    base_dir = os.path.dirname(__file__)
    yolo_path = os.path.join(base_dir, "test_resources/YOLO_TEST")

    # 1. Load YOLO Dataset
    yolo_original: YoloFormat = YoloFormat.read_from_folder(yolo_path)
    
    # 2. YOLO → Neutral → VGG
    neutral_from_yolo: NeutralFormat = YoloConverter.toNeutral(yolo_original)
    vgg_converted: VGGFormat = VGGConverter.fromNeutral(neutral_from_yolo)
    
    # Check VGG structure
    assert isinstance(vgg_converted, VGGFormat)
    assert len(vgg_converted.files) == len(yolo_original.files)

    # Check file values
    for vgg_file, yolo_file in zip(vgg_converted.files, yolo_original.files):
        assert len(vgg_file.annotations) == len(yolo_file.annotations)
        assert vgg_file.filename == yolo_file.filename
        
        # Check annotation values
        for vgg_annotation, yolo_annotation in zip(vgg_file.annotations, yolo_file.annotations):
            # Check that VGG shapes are rectangles (converted from YOLO bounding boxes)
            assert vgg_annotation.geometry.shape_type == "rect"
            
            # Get class name from YOLO class_labels mapping
            yolo_class_name = yolo_original.class_labels[yolo_annotation.id_class]
            assert yolo_class_name in vgg_annotation.region_attributes.values()


def test_coco_to_vgg():
    """Test conversion from COCO to VGG format."""
    base_dir = os.path.dirname(__file__)
    coco_path = os.path.join(base_dir, "test_resources/COCO_TEST")

    # 1. Load COCO Dataset
    coco_original: CocoFormat = CocoFormat.read_from_folder(coco_path)
    
    # 2. COCO → Neutral → VGG
    neutral_from_coco: NeutralFormat = CocoConverter.toNeutral(coco_original)
    vgg_converted: VGGFormat = VGGConverter.fromNeutral(neutral_from_coco)
    
    # Check VGG structure
    assert isinstance(vgg_converted, VGGFormat)
    assert len(vgg_converted.files) == len(coco_original.files[0].images)
    
    # Calculate total annotations
    total_annotations = sum(len(file.annotations) for file in vgg_converted.files)
    assert total_annotations == len(coco_original.files[0].annotations)
    
    # Check that every VGG class is in COCO categories
    all_vgg_classes = set()
    for file in vgg_converted.files:
        for ann in file.annotations:
            vgg_class = _extract_class_from_vgg_attributes(ann.region_attributes)
            all_vgg_classes.add(vgg_class)
    
    coco_category_names = {c.name for c in coco_original.files[0].categories}
    assert all_vgg_classes.issubset(coco_category_names)


def test_vgg_shape_preservation():
    """Test that VGG shape information is preserved through neutral conversion."""
    base_dir = os.path.dirname(__file__)
    vgg_path = os.path.join(base_dir, "test_resources/VGG_TEST/annotations.json")

    # 1. Load VGG Dataset
    vgg_original: VGGFormat = VGGFormat.read_from_file(vgg_path)
    
    # 2. VGG → Neutral → VGG
    neutral_from_vgg: NeutralFormat = VGGConverter.toNeutral(vgg_original)
    vgg_reconverted: VGGFormat = VGGConverter.fromNeutral(neutral_from_vgg)
    
    # Check that shape-specific attributes are preserved
    for orig_file, reconv_file in zip(vgg_original.files, vgg_reconverted.files):
        for orig_ann, reconv_ann in zip(orig_file.annotations, reconv_file.annotations):
            # Check shape type preservation
            assert orig_ann.geometry.shape_type == reconv_ann.geometry.shape_type
            
            # Check coordinates preservation for different shape types
            orig_coords = orig_ann.geometry.getCoordinates()
            reconv_coords = reconv_ann.geometry.getCoordinates()
            
            if orig_ann.geometry.shape_type == "circle":
                assert orig_coords["cx"] == pytest.approx(reconv_coords["cx"], rel=1e-3, abs=1e-3)
                assert orig_coords["cy"] == pytest.approx(reconv_coords["cy"], rel=1e-3, abs=1e-3)
                assert orig_coords["r"] == pytest.approx(reconv_coords["r"], rel=1e-3, abs=1e-3)
            elif orig_ann.geometry.shape_type == "ellipse":
                assert orig_coords["cx"] == pytest.approx(reconv_coords["cx"], rel=1e-3, abs=1e-3)
                assert orig_coords["cy"] == pytest.approx(reconv_coords["cy"], rel=1e-3, abs=1e-3)
                assert orig_coords["rx"] == pytest.approx(reconv_coords["rx"], rel=1e-3, abs=1e-3)
                assert orig_coords["ry"] == pytest.approx(reconv_coords["ry"], rel=1e-3, abs=1e-3)
                assert orig_coords["theta"] == pytest.approx(reconv_coords["theta"], rel=1e-3, abs=1e-3)
            elif orig_ann.geometry.shape_type in ["polygon", "polyline"]:
                assert orig_coords["all_points_x"] == reconv_coords["all_points_x"]
                assert orig_coords["all_points_y"] == reconv_coords["all_points_y"]


def test_vgg_metadata_preservation():
    """Test that VGG metadata is preserved through neutral conversion."""
    base_dir = os.path.dirname(__file__)
    vgg_path = os.path.join(base_dir, "test_resources/VGG_TEST/annotations.json")

    # 1. Load VGG Dataset
    vgg_original: VGGFormat = VGGFormat.read_from_file(vgg_path)
    
    # 2. VGG → Neutral → VGG
    neutral_from_vgg: NeutralFormat = VGGConverter.toNeutral(vgg_original)
    vgg_reconverted: VGGFormat = VGGConverter.fromNeutral(neutral_from_vgg)
    
    # Check that VGG-specific metadata is preserved
    for orig_file, reconv_file in zip(vgg_original.files, vgg_reconverted.files):
        # Check file attributes preservation
        assert orig_file.file_attributes == reconv_file.file_attributes
        
        # Check file size preservation
        assert orig_file.size == reconv_file.size
        
        for orig_ann, reconv_ann in zip(orig_file.annotations, reconv_file.annotations):
            # Check region attributes preservation
            assert orig_ann.region_attributes == reconv_ann.region_attributes


def test_vgg_class_extraction():
    """Test that class names are correctly extracted from VGG region attributes."""
    base_dir = os.path.dirname(__file__)
    vgg_path = os.path.join(base_dir, "test_resources/VGG_TEST/annotations.json")

    # 1. Load VGG Dataset
    vgg_original: VGGFormat = VGGFormat.read_from_file(vgg_path)
    
    # 2. VGG → Neutral
    neutral_from_vgg: NeutralFormat = VGGConverter.toNeutral(vgg_original)
    
    # Check that class map is correctly built
    assert len(neutral_from_vgg.class_map) > 0
    
    # Check that all classes are strings
    for class_id, class_name in neutral_from_vgg.class_map.items():
        assert isinstance(class_id, int)
        assert isinstance(class_name, str)
        assert len(class_name) > 0
    
    # Check that neutral annotations have correct class names
    for neutral_file in neutral_from_vgg.files:
        for neutral_ann in neutral_file.annotations:
            assert neutral_ann.class_name in neutral_from_vgg.class_map.values()


def _extract_class_from_vgg_attributes(region_attributes: dict) -> str:
    """Helper function to extract class name from VGG region attributes."""
    class_keys = ['type', 'class', 'label', 'name', 'names', 'category', 'object_type']
    
    for key in class_keys:
        if key in region_attributes and isinstance(region_attributes[key], str):
            return region_attributes[key]
    
    # If no class found, return first string value or default
    for value in region_attributes.values():
        if isinstance(value, str) and value.strip():
            return value.strip()
    
    return 'object'