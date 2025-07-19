import os
import pytest

from vision_converter.converters.coco_converter import CocoConverter
from vision_converter.converters.createml_converter import CreateMLConverter
from vision_converter.converters.labelme_converter import LabelMeConverter
from vision_converter.converters.pascal_voc_converter import PascalVocConverter
from vision_converter.converters.tensorflow_csv_converter import TensorflowCsvConverter
from vision_converter.converters.vgg_converter import VGGConverter
from vision_converter.converters.yolo_converter import YoloConverter

from vision_converter.formats.coco import CocoFormat
from vision_converter.formats.createml import CreateMLFormat
from vision_converter.formats.labelme import LabelMeFormat
from vision_converter.formats.neutral_format import NeutralFormat
from vision_converter.formats.pascal_voc import PascalVocFormat
from vision_converter.formats.tensorflow_csv import TensorflowCsvFormat
from vision_converter.formats.vgg import VGGFormat
from vision_converter.formats.yolo import YoloFormat
from vision_converter.utils.bbox_utils import CreateMLBBox_to_PascalVocBBox, PascalVocBBox_to_YoloBBox


def bbox_almost_equal(bbox1, bbox2, epsilon=1e-3) -> bool:
    """Helper function to compare bounding boxes with tolerance."""
    return all(abs(a - b) < epsilon for a, b in zip(bbox1, bbox2))


# ============================================================================
# YOLO FORMAT TESTS
# ============================================================================

def test_yolo_to_neutral():
    """Test conversion from YOLO format to NeutralFormat."""
    base_dir = os.path.dirname(__file__)
    yolo_path = os.path.join(base_dir, "test_resources/YOLO_TEST")
    
    # Load YOLO dataset
    yolo_original: YoloFormat = YoloFormat.read_from_folder(yolo_path)
    
    # Convert to neutral
    neutral = YoloConverter.toNeutral(yolo_original)
    
    # Basic structure checks
    assert neutral is not None
    assert isinstance(neutral, NeutralFormat)
    assert neutral.original_format == "yolo"
    assert len(neutral.files) == len(yolo_original.files)
    
    # Check class mapping preservation
    assert len(neutral.class_map) == len(yolo_original.class_labels)
    neutral_classes = set(neutral.class_map.values())
    yolo_classes = set(yolo_original.class_labels.values())
    assert neutral_classes.issubset(yolo_classes)
    
    # Check annotation count preservation
    total_yolo_annotations = sum(len(f.annotations) for f in yolo_original.files)
    total_neutral_annotations = sum(len(f.annotations) for f in neutral.files)
    assert total_neutral_annotations == total_yolo_annotations

    for yolo_file, neutral_file in zip(yolo_original.files, neutral.files):

        assert len(yolo_file.annotations) == len(neutral_file.annotations)
        assert yolo_file.filename == neutral_file.filename + neutral_file.image_origin.extension

        for i, (yolo_ann, neutral_ann) in enumerate(zip(yolo_file.annotations, neutral_file.annotations)):

            # Check bbox
            yolo_bbox = yolo_ann.geometry.getBoundingBox()
            neutral_bbox = PascalVocBBox_to_YoloBBox(neutral_ann.geometry, neutral_file.width, neutral_file.height).getBoundingBox()
            assert yolo_bbox == pytest.approx(neutral_bbox, rel=1e-3, abs=1e-3)

            # Get name from the map
            yolo_class_name = yolo_original.class_labels[yolo_ann.id_class]
            neutral_class_name = neutral_ann.class_name
            
            # Check name matches
            assert yolo_class_name == neutral_class_name, f"Annotation {i}: class '{yolo_class_name}' != '{neutral_class_name}'"


def test_neutral_to_yolo():
    """Test conversion from NeutralFormat to YOLO format."""
    base_dir = os.path.dirname(__file__)
    yolo_path = os.path.join(base_dir, "test_resources/YOLO_TEST")
    
    # Load and convert through neutral
    neutral = YoloConverter.toNeutral(YoloFormat.read_from_folder(yolo_path))
    yolo_reconverted = YoloConverter.fromNeutral(neutral)
    
    # Basic structure checks
    assert isinstance(yolo_reconverted, YoloFormat)
    assert len(yolo_reconverted.files) == len(neutral.files)
    assert len(yolo_reconverted.class_labels) == len(neutral.class_map.values())
    
    # Check class labels preservation
    assert set(neutral.class_map.values()) == set(yolo_reconverted.class_labels.values())
    
    # Check annotation preservation
    for neutral_file, reconv_file in zip(neutral.files, yolo_reconverted.files):
        assert neutral_file.filename + neutral_file.image_origin.extension == reconv_file.filename 
        assert len(neutral_file.annotations) == len(reconv_file.annotations)

        for i, (neutral_ann, reconv_ann) in enumerate(zip(neutral_file.annotations, reconv_file.annotations)):

            # Check bbox
            yolo_bbox = reconv_ann.geometry.getBoundingBox()
            neutral_bbox = PascalVocBBox_to_YoloBBox(neutral_ann.geometry, neutral_file.width, neutral_file.height).getBoundingBox()
            assert yolo_bbox == pytest.approx(neutral_bbox, rel=1e-3, abs=1e-3)

            # Get name from the map
            yolo_class_name = yolo_reconverted.class_labels[reconv_ann.id_class]
            neutral_class_name = neutral_ann.class_name
            
            # Check name matches
            assert yolo_class_name == neutral_class_name, f"Annotation {i}: class '{yolo_class_name}' != '{neutral_class_name}'"


# ============================================================================
# PASCAL VOC FORMAT TESTS
# ============================================================================

def test_pascalvoc_to_neutral():
    """Test conversion from Pascal VOC format to NeutralFormat."""
    base_dir = os.path.dirname(__file__)
    pascal_path = os.path.join(base_dir, "test_resources/PASCAL_VOC_TEST")
    
    # Load Pascal VOC dataset
    pascal_original: PascalVocFormat = PascalVocFormat.read_from_folder(pascal_path)
    
    # Convert to neutral
    neutral = PascalVocConverter.toNeutral(pascal_original)
    
    # Basic structure checks
    assert neutral is not None
    assert isinstance(neutral, NeutralFormat)
    assert neutral.original_format == "pascal_voc"
    assert len(neutral.files) == len(pascal_original.files)
    
    # Check unique classes extraction
    pascal_classes = {ann.name for file in pascal_original.files for ann in file.annotations}
    neutral_classes = set(neutral.class_map.values())
    assert neutral_classes == pascal_classes
    
    # Check annotation count preservation
    total_pascal_annotations = sum(len(f.annotations) for f in pascal_original.files)
    total_neutral_annotations = sum(len(f.annotations) for f in neutral.files)
    assert total_neutral_annotations == total_pascal_annotations

    for pascal_file, neutral_file in zip(pascal_original.files, neutral.files):
        # Check source metadata
        assert pascal_file.filename == neutral_file.filename + neutral_file.image_origin.extension
        assert len(pascal_file.annotations) == len(neutral_file.annotations)

        for i, (pascal_ann, neutral_ann) in enumerate(zip(pascal_file.annotations, neutral_file.annotations)):
            # Check name matches
            assert pascal_ann.name == neutral_ann.class_name, f"Annotation {i}: class '{pascal_ann.name}' != '{neutral_ann.class_name}'"

            # Check bbox
            pascal_bbox = pascal_ann.geometry.getBoundingBox()
            neutral_bbox = neutral_ann.geometry.getBoundingBox()
            assert pascal_bbox == pytest.approx(neutral_bbox, rel=1e-6, abs=1e-6)


def test_neutral_to_pascalvoc():
    """Test conversion from NeutralFormat to Pascal VOC format."""
    base_dir = os.path.dirname(__file__)
    pascal_path = os.path.join(base_dir, "test_resources/PASCAL_VOC_TEST")
    
    # Load and convert through neutral
    neutral = PascalVocConverter.toNeutral(PascalVocFormat.read_from_folder(pascal_path))
    pascal_reconverted = PascalVocConverter.fromNeutral(neutral)
    
    # Basic structure checks
    assert isinstance(pascal_reconverted, PascalVocFormat)
    assert len(pascal_reconverted.files) == len(neutral.files)
    
    # Check source metadata consistency
    for neutral_file, reconv_file in zip(neutral.files, pascal_reconverted.files):

        assert reconv_file.source.annotation == "Pascal Voc"
        assert len(neutral_file.annotations) == len(reconv_file.annotations)
        
        for i, (neutral_ann, pascal_ann) in enumerate(zip(neutral_file.annotations, reconv_file.annotations)):
            assert neutral_ann.class_name == pascal_ann.name, f"Annotation {i}: class '{pascal_ann.name}' != '{neutral_ann.class_name}'"

            # Check bbox
            neutral_bbox = neutral_ann.geometry.getBoundingBox()
            pascal_bbox = pascal_ann.geometry.getBoundingBox()
            assert pascal_bbox == pytest.approx(neutral_bbox, rel=1e-6, abs=1e-6)


# ============================================================================
# COCO FORMAT TESTS
# ============================================================================

def test_coco_to_neutral():
    """Test conversion from COCO format to NeutralFormat."""
    base_dir = os.path.dirname(__file__)
    coco_path = os.path.join(base_dir, "test_resources/COCO_TEST/annotations.json")
    
    # Load COCO dataset
    coco_original: CocoFormat = CocoFormat.read_from_folder(coco_path)
    
    # Convert to neutral
    neutral = CocoConverter.toNeutral(coco_original)
    
    # Basic structure checks
    assert neutral is not None
    assert isinstance(neutral, NeutralFormat)
    assert neutral.original_format == "coco"
    assert len(neutral.files) == sum(len(f.images) for f in coco_original.files)
    
    # Check class extraction
    original_classes = {cat.name for f in coco_original.files for cat in f.categories}
    neutral_classes = set(neutral.class_map.values())
    assert neutral_classes.issubset(original_classes)
    
    # Check annotation count preservation
    total_coco_annotations = sum(len(f.annotations) for f in coco_original.files)
    total_neutral_annotations = sum(len(f.annotations) for f in neutral.files)
    assert total_neutral_annotations == total_coco_annotations

    for coco_file in coco_original.files:
        neutral_number_of_annotations = 0

        # Check image metadata
        for neutral_file, coco_images in zip(neutral.files, coco_file.images):
            neutral_number_of_annotations += len(neutral_file.annotations)
            assert coco_images.file_name == neutral_file.filename + neutral_file.image_origin.extension
            assert coco_images.height == neutral_file.height
            assert coco_images.width == neutral_file.width

        # Check same number of annotations
        assert len(coco_file.annotations) == neutral_number_of_annotations


def test_neutral_to_coco():
    """Test conversion from NeutralFormat to COCO format."""
    base_dir = os.path.dirname(__file__)
    coco_path = os.path.join(base_dir, "test_resources/COCO_TEST/annotations.json")
    
    # Load and convert through neutral
    neutral = CocoConverter.toNeutral(CocoFormat.read_from_folder(coco_path))
    coco_reconverted = CocoConverter.fromNeutral(neutral)
    
    # Basic structure checks
    assert isinstance(coco_reconverted, CocoFormat)
    assert len(neutral.files) == sum(len(f.images) for f in coco_reconverted.files)

    # Check annotation count preservation
    total_coco_annotations = sum(len(f.annotations) for f in coco_reconverted.files)
    total_neutral_annotations = sum(len(f.annotations) for f in neutral.files)
    assert total_neutral_annotations == total_coco_annotations
    
    # Check categories and annotations preservation
    for coco_file in coco_reconverted.files:
        neutral_number_of_annotations = 0

        # Check image metadata
        for neutral_file, coco_images in zip(neutral.files, coco_file.images):
            neutral_number_of_annotations += len(neutral_file.annotations)
            assert coco_images.file_name == neutral_file.filename + neutral_file.image_origin.extension
            assert coco_images.height == neutral_file.height
            assert coco_images.width == neutral_file.width

        # Check same number of annotations
        assert len(coco_file.annotations) == neutral_number_of_annotations


# ============================================================================
# CREATE ML FORMAT TESTS
# ============================================================================

def test_createml_to_neutral():
    """Test conversion from CreateML format to NeutralFormat."""
    base_dir = os.path.dirname(__file__)
    createml_path = os.path.join(base_dir, "test_resources/CREATEML_TEST/annotations.json")
    
    # Load CreateML dataset
    createml_original: CreateMLFormat = CreateMLFormat.read_from_folder(createml_path)
    
    # Convert to neutral
    neutral = CreateMLConverter.toNeutral(createml_original)
    
    # Basic structure checks
    assert neutral is not None
    assert isinstance(neutral, NeutralFormat)
    assert neutral.original_format == "createml"
    assert len(neutral.files) == len(createml_original.files)
    
    # Check class extraction
    createml_labels = {ann.label for file in createml_original.files for ann in file.annotations}
    neutral_classes = set(neutral.class_map.values())
    assert neutral_classes == createml_labels
    
    # Check annotation count preservation
    total_createml_annotations = sum(len(f.annotations) for f in createml_original.files)
    total_neutral_annotations = sum(len(f.annotations) for f in neutral.files)
    assert total_neutral_annotations == total_createml_annotations

    # Check labels and annotations preservation
    for createml_file, neutral_file in zip(createml_original.files, neutral.files):
        assert len(createml_file.annotations) == len(neutral_file.annotations)

        # Check image metadata
        assert createml_file.filename == neutral_file.filename + neutral_file.image_origin.extension
        
        for createml_ann, neutral_ann in zip(createml_file.annotations, neutral_file.annotations):

            assert createml_ann.label == neutral_ann.class_name
            neutral_bbox = neutral_ann.geometry.getBoundingBox()
            createml_bbox = CreateMLBBox_to_PascalVocBBox(createml_ann.geometry).getBoundingBox()
            assert createml_bbox == pytest.approx(neutral_bbox, rel=1e-6, abs=1e-6)



def test_neutral_to_createml():
    """Test conversion from NeutralFormat to CreateML format."""
    base_dir = os.path.dirname(__file__)
    createml_path = os.path.join(base_dir, "test_resources/CREATEML_TEST/annotations.json")
    
    # Load and convert through neutral
    neutral = CreateMLConverter.toNeutral(CreateMLFormat.read_from_folder(createml_path))
    createml_reconverted = CreateMLConverter.fromNeutral(neutral)
    
    # Basic structure checks
    assert isinstance(createml_reconverted, CreateMLFormat)
    assert len(createml_reconverted.files) == len(neutral.files)

    # Check annotation count preservation
    total_neutral_annotations = sum(len(f.annotations) for f in neutral.files)
    total_createml_annotations = sum(len(f.annotations) for f in createml_reconverted.files)
    assert total_neutral_annotations == total_createml_annotations
    
    # Check labels and annotations preservation
    for neutral_file, createml_file  in zip(neutral.files, createml_reconverted.files):
        assert len(createml_file.annotations) == len(neutral_file.annotations)

        # Check image metadata
        assert createml_file.filename == neutral_file.filename + neutral_file.image_origin.extension
        
        for createml_ann, neutral_ann in zip(createml_file.annotations, neutral_file.annotations):
            assert createml_ann.label == neutral_ann.class_name
            neutral_bbox = neutral_ann.geometry.getBoundingBox()
            createml_bbox = CreateMLBBox_to_PascalVocBBox(createml_ann.geometry).getBoundingBox()
            assert createml_bbox == pytest.approx(neutral_bbox, rel=1e-6, abs=1e-6)


# ============================================================================
# TENSORFLOW CSV FORMAT TESTS
# ============================================================================

def test_tensorflow_csv_to_neutral():
    """Test conversion from TensorFlow CSV format to NeutralFormat."""
    base_dir = os.path.dirname(__file__)
    tf_csv_path = os.path.join(base_dir, "test_resources/TENSORFLOW_CSV_TEST/annotations.csv")
    
    # Load TensorFlow CSV dataset
    tf_csv_original: TensorflowCsvFormat = TensorflowCsvFormat.read_from_folder(tf_csv_path)
    
    # Convert to neutral
    neutral = TensorflowCsvConverter.toNeutral(tf_csv_original)
    
    # Basic structure checks
    assert neutral is not None
    assert isinstance(neutral, NeutralFormat)
    assert neutral.original_format == "tensorflow_csv"
    assert len(neutral.files) == len(tf_csv_original.files)
    
    # Check class extraction
    tf_csv_classes = tf_csv_original.get_unique_classes()
    neutral_classes = set(neutral.class_map.values())
    assert neutral_classes == tf_csv_classes
    
    # Check annotation count preservation
    total_tf_csv_annotations = sum(len(f.annotations) for f in tf_csv_original.files)
    total_neutral_annotations = sum(len(f.annotations) for f in neutral.files)
    assert total_neutral_annotations == total_tf_csv_annotations

    for tf_file, neutral_file in zip(tf_csv_original.files, neutral.files):
        assert tf_file.filename == neutral_file.filename + neutral_file.image_origin.extension
        assert len(tf_file.annotations) == len(neutral_file.annotations)
        assert tf_file.height == neutral_file.height
        assert tf_file.width == neutral_file.width
        
        for tf_ann, neutral_ann in zip(tf_file.annotations, neutral_file.annotations):
            assert tf_ann.class_name == neutral_ann.class_name
            neutral_bbox = neutral_ann.geometry.getBoundingBox()
            tf_bbox = tf_ann.geometry.getBoundingBox()
            assert tf_bbox == pytest.approx(neutral_bbox, rel=1e-6, abs=1e-6)


def test_neutral_to_tensorflow_csv():
    """Test conversion from NeutralFormat to TensorFlow CSV format."""
    base_dir = os.path.dirname(__file__)
    tf_csv_path = os.path.join(base_dir, "test_resources/TENSORFLOW_CSV_TEST/annotations.csv")
    
    # Load and convert through neutral
    neutral = TensorflowCsvConverter.toNeutral(TensorflowCsvFormat.read_from_folder(tf_csv_path))
    tf_csv_reconverted = TensorflowCsvConverter.fromNeutral(neutral)
    
    # Basic structure checks
    assert isinstance(tf_csv_reconverted, TensorflowCsvFormat)
    assert len(tf_csv_reconverted.files) == len(neutral.files)
    
    # Check classes and annotations preservation
    neutral_classes = set(neutral.class_map.values())
    reconv_classes = tf_csv_reconverted.get_unique_classes()
    assert neutral_classes.issubset(reconv_classes) 
    
    # Check annotation count preservation
    total_tf_csv_annotations = sum(len(f.annotations) for f in tf_csv_reconverted.files)
    total_neutral_annotations = sum(len(f.annotations) for f in neutral.files)
    assert total_neutral_annotations == total_tf_csv_annotations

    for tf_file, neutral_file in zip(tf_csv_reconverted.files, neutral.files):
        assert tf_file.filename == neutral_file.filename + neutral_file.image_origin.extension
        assert len(tf_file.annotations) == len(neutral_file.annotations)
        assert tf_file.height == neutral_file.height
        assert tf_file.width == neutral_file.width
        
        for tf_ann, neutral_ann in zip(tf_file.annotations, neutral_file.annotations):
            assert tf_ann.class_name == neutral_ann.class_name
            neutral_bbox = neutral_ann.geometry.getBoundingBox()
            tf_bbox = tf_ann.geometry.getBoundingBox()
            assert tf_bbox == pytest.approx(neutral_bbox, rel=1e-6, abs=1e-6)


# ============================================================================
# LABELME FORMAT TESTS
# ============================================================================

def test_labelme_to_neutral():
    """Test conversion from LabelMe format to NeutralFormat."""
    base_dir = os.path.dirname(__file__)
    labelme_path = os.path.join(base_dir, "test_resources/LABELME_TEST")
    
    # Load LabelMe dataset
    labelme_original: LabelMeFormat = LabelMeFormat.read_from_folder(labelme_path)
    
    # Convert to neutral
    neutral = LabelMeConverter.toNeutral(labelme_original)
    
    # Basic structure checks
    assert neutral is not None
    assert isinstance(neutral, NeutralFormat)
    assert neutral.original_format == "labelme"
    assert len(neutral.files) == len(labelme_original.files)
    
    # Check class extraction
    labelme_labels = {ann.label for file in labelme_original.files for ann in file.annotations}
    neutral_classes = set(neutral.class_map.values())
    assert neutral_classes == labelme_labels
    
    # Check annotation count preservation
    total_labelme_annotations = sum(len(f.annotations) for f in labelme_original.files)
    total_neutral_annotations = sum(len(f.annotations) for f in neutral.files)
    assert total_neutral_annotations == total_labelme_annotations

    for labelme_file, neutral_file in zip(labelme_original.files, neutral.files):
        assert labelme_file.filename == neutral_file.filename + neutral_file.image_origin.extension
        assert len(labelme_file.annotations) == len(neutral_file.annotations)
        assert labelme_file.imageHeight == neutral_file.height
        assert labelme_file.imageWidth == neutral_file.width
        
        for labelme_ann, neutral_ann in zip(labelme_file.annotations, neutral_file.annotations):
            assert labelme_ann.label == neutral_ann.class_name
            neutral_bbox = neutral_ann.geometry.getBoundingBox()
            label_bbox = labelme_ann.geometry.getBoundingBox().getBoundingBox()
            assert label_bbox == pytest.approx(neutral_bbox, rel=1e-6, abs=1e-6)


def test_neutral_to_labelme():
    """Test conversion from NeutralFormat to LabelMe format."""
    base_dir = os.path.dirname(__file__)
    labelme_path = os.path.join(base_dir, "test_resources/LABELME_TEST")
    
    # Load original and convert through neutral
    neutral = LabelMeConverter.toNeutral(LabelMeFormat.read_from_folder(labelme_path))
    labelme_reconverted = LabelMeConverter.fromNeutral(neutral)
    
    # Basic structure checks
    assert isinstance(labelme_reconverted, LabelMeFormat)
    assert len(labelme_reconverted.files) == len(neutral.files)

    # Check class extraction
    labelme_labels = {ann.label for file in labelme_reconverted.files for ann in file.annotations}
    neutral_classes = set(neutral.class_map.values())
    assert neutral_classes == labelme_labels
    
    # Check annotation count preservation
    total_labelme_annotations = sum(len(f.annotations) for f in labelme_reconverted.files)
    total_neutral_annotations = sum(len(f.annotations) for f in neutral.files)
    assert total_neutral_annotations == total_labelme_annotations

    for labelme_file, neutral_file in zip(labelme_reconverted.files, neutral.files):
        assert labelme_file.filename == neutral_file.filename + neutral_file.image_origin.extension
        assert len(labelme_file.annotations) == len(neutral_file.annotations)
        assert labelme_file.imageHeight == neutral_file.height
        assert labelme_file.imageWidth == neutral_file.width
        
        for labelme_ann, neutral_ann in zip(labelme_file.annotations, neutral_file.annotations):
            assert labelme_ann.label == neutral_ann.class_name
            neutral_bbox = neutral_ann.geometry.getBoundingBox()
            label_bbox = labelme_ann.geometry.getBoundingBox().getBoundingBox()
            assert label_bbox == pytest.approx(neutral_bbox, rel=1e-6, abs=1e-6)


# ============================================================================
# VGG FORMAT TESTS
# ============================================================================

def test_vgg_to_neutral():
    """Test conversion from VGG format to NeutralFormat."""
    base_dir = os.path.dirname(__file__)
    vgg_path = os.path.join(base_dir, "test_resources/VGG_TEST/annotations.json")
    
    # Load VGG dataset
    vgg_original: VGGFormat = VGGFormat.read_from_folder(vgg_path)
    
    # Convert to neutral
    neutral = VGGConverter.toNeutral(vgg_original)
    
    # Basic structure checks
    assert neutral is not None
    assert isinstance(neutral, NeutralFormat)
    assert neutral.original_format == "vgg"
    assert len(neutral.files) == len(vgg_original.files)
    
    # Check metadata preservation
    assert "vgg_version" in neutral.metadata
    assert "total_regions" in neutral.metadata
    assert "shape_types_used" in neutral.metadata
    
    # Check annotation count preservation
    total_vgg_annotations = sum(len(f.annotations) for f in vgg_original.files)
    total_neutral_annotations = sum(len(f.annotations) for f in neutral.files)
    assert total_neutral_annotations == total_vgg_annotations

    for vgg_file, neutral_file in zip(vgg_original.files, neutral.files):
        assert vgg_file.filename == neutral_file.filename + neutral_file.image_origin.extension
        assert len(vgg_file.annotations) == len(neutral_file.annotations)
        
        for vgg_ann, neutral_ann in zip(vgg_file.annotations, neutral_file.annotations):
            neutral_bbox = neutral_ann.geometry.getBoundingBox()
            vgg_bbox = vgg_ann.geometry.getBoundingBox().getBoundingBox()
            assert vgg_bbox == pytest.approx(neutral_bbox, rel=1e-6, abs=1e-6)



def test_neutral_to_vgg():
    """Test conversion from NeutralFormat to VGG format."""
    base_dir = os.path.dirname(__file__)
    vgg_path = os.path.join(base_dir, "test_resources/VGG_TEST/annotations.json")
    
    # Load original and convert through neutral
    neutral = VGGConverter.toNeutral(VGGFormat.read_from_folder(vgg_path))
    vgg_reconverted = VGGConverter.fromNeutral(neutral)
    
    # Basic structure checks
    assert isinstance(vgg_reconverted, VGGFormat)
    assert len(vgg_reconverted.files) == len(neutral.files)
    
    # Check annotation count preservation
    total_vgg_annotations = sum(len(f.annotations) for f in vgg_reconverted.files)
    total_neutral_annotations = sum(len(f.annotations) for f in neutral.files)
    assert total_neutral_annotations == total_vgg_annotations

    for vgg_file, neutral_file in zip(vgg_reconverted.files, neutral.files):
        assert vgg_file.filename == neutral_file.filename + neutral_file.image_origin.extension
        assert len(vgg_file.annotations) == len(neutral_file.annotations)
        
        for vgg_ann, neutral_ann in zip(vgg_file.annotations, neutral_file.annotations):
            neutral_bbox = neutral_ann.geometry.getBoundingBox()
            vgg_bbox = vgg_ann.geometry.getBoundingBox().getBoundingBox()
            assert vgg_ann.geometry.shape_type == "rect"
            assert vgg_bbox == pytest.approx(neutral_bbox, rel=1e-6, abs=1e-6)