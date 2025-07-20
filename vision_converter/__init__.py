import importlib
import os

# Formats
from .formats import FORMATS_NAME, CocoFormat, PascalVocFormat, YoloFormat, CreateMLFormat, TensorflowCsvFormat, LabelMeFormat, VGGFormat, NeutralFormat

# Converters
from .converters import CocoConverter, PascalVocConverter, YoloConverter, CreateMLConverter, TensorflowCsvConverter, LabelMeConverter, VGGConverter

__all__ = [
    # Formats
    'CocoFormat',
    'PascalVocFormat',
    'YoloFormat',
    'NeutralFormat',
    'CreateMLFormat',
    'TensorflowCsvFormat', 
    'LabelMeFormat', 
    'VGGFormat',
    # Converters
    'CocoConverter',
    'PascalVocConverter',
    'YoloConverter',
    'CreateMLConverter',
    'TensorflowCsvConverter',
    'LabelMeConverter',
    'VGGConverter',
    # Convert function
    'convert'
]


def convert(input_format: str, input_path: str, output_format: str, output_path: str):
    """
    Converts a dataset from the input format to the output format.

    Args:
        input_format (str): Input dataset format.
        input_path (str): Path to the input dataset.
        output_format (str): Output dataset format.
        output_path (str): Path to save the converted dataset.

    Raise:
        ImportError, AttributeError, PermissionError, Exception
    """
    if not os.access(os.path.dirname(output_path), os.W_OK):
        raise PermissionError("Output path is not writable")

    # Dynamic import of format classes
    input_format_module = importlib.import_module(f'vision_converter.formats.{input_format}')
    input_format_class_name = f"{FORMATS_NAME.get(input_format)}Format"
    input_format_class = getattr(input_format_module, input_format_class_name)

    output_format_module = importlib.import_module(f'vision_converter.formats.{output_format}')
    output_format_class_name = f"{FORMATS_NAME.get(output_format)}Format"
    output_format_class = getattr(output_format_module, output_format_class_name)

    if not input_format_class:
        raise ImportError(f"Format not found for format: {input_format}")
    if not output_format_class:
        raise ImportError(f"Format not found for format: {output_format}")
        
    # Dynamic import of converters
    input_converter_module = importlib.import_module(f'vision_converter.converters.{input_format}_converter')
    input_converter_class_name = f"{FORMATS_NAME.get(input_format)}Converter"
    input_converter_class = getattr(input_converter_module,input_converter_class_name)

    output_converter_module = importlib.import_module(f'vision_converter.converters.{output_format}_converter')
    output_converter_class_name = f"{FORMATS_NAME.get(output_format)}Converter"
    output_converter_class = getattr(output_converter_module, output_converter_class_name)

    if not input_converter_class:
        raise ImportError(f"Converter not found for format: {input_format}")
    if not output_converter_class:
        raise ImportError(f"Converter not found for format: {output_format}")

    # Load input dataset
    input_dataset = input_format_class.read_from_folder(input_path)

    # Convert to NeutralFormat
    neutral_format = input_converter_class.toNeutral(input_dataset)

    # Convert to output format
    output_dataset = output_converter_class.fromNeutral(neutral_format)

    # Save output dataset 
    output_dataset.save(output_path)