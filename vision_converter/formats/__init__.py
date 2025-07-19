from .yolo import YoloFormat
from .coco import CocoFormat
from .pascal_voc import PascalVocFormat
from .neutral_format import NeutralFormat
from .createml import CreateMLFormat
from .tensorflow_csv import TensorflowCsvFormat
from .labelme import LabelMeFormat
from .vgg import VGGFormat

FORMATS_NAME: dict[str, str] = {
    'coco': 'Coco',
    'pascal_voc': 'PascalVoc',
    'yolo': 'Yolo',
    'createml': 'CreateML',
    'tensorflow_csv': 'TensorflowCsv',
    'labelme': 'LabelMe',
    'vgg': 'VGG'
}

__all__ = ['YoloFormat', 'CocoFormat', 'PascalVocFormat', 'CreateMLFormat', 'TensorflowCsvFormat', 'LabelMeFormat', 'VGGFormat', 'NeutralFormat']