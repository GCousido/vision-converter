import click
import importlib
import sys
import os
from pathlib import Path

# Añadir directorio padre al path para importar desde otros módulos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from formats.neutral_format import NeutralFormat

FORMATS = ['coco', 'pascal_voc', 'yolo']

@click.command()
@click.option('--input-format', '-if', 
                required=True, 
                type=click.Choice(FORMATS), 
                help='Input dataset format')
@click.option('--input-path', '-ip', 
                required=True, 
                type=click.Path(exists=True), 
                help='Path to the Input dataset')
@click.option('--output-format', '-of', 
                required=True, 
                type=click.Choice(FORMATS), 
                help='Output dataset format')
@click.option('--output-path', '-op', 
                required=True, 
                type=click.Path(), 
                help='Path to the Output dataset')
def dconverter(input_format, input_path, output_format, output_path):
    """Convert one dataset from one format to another
    
    This command takes as input one dataset in a specific format, and converts it to the neutral format. Then it converts the neutral format to the output wanted format.

    FORMATS:
    \b
        - coco
        - yolo
        - pascal_voc
    """

    # Check if it has permissions to write
    if not os.access(os.path.dirname(output_path), os.W_OK):
        raise click.ClickException("Output path is not writable")

    try:
        # Dynamic import of format classes
        input_format_module = importlib.import_module(f'formats.{input_format}')
        input_format_class_name = f"{to_camel_case(input_format)}Format"
        input_format_class = getattr(input_format_module, input_format_class_name)

        output_format_module = importlib.import_module(f'formats.{output_format}')
        output_format_class_name = f"{to_camel_case(output_format)}Format"
        output_format_class = getattr(output_format_module, output_format_class_name)
        
        # Dynamic import of converters
        input_converter_module = importlib.import_module(f'converters.{input_format}_converter')
        input_converter_class_name = f"{to_camel_case(input_format)}Converter"
        input_converter_class = getattr(input_converter_module,input_converter_class_name)

        output_converter_module = importlib.import_module(f'converters.{output_format}_converter')
        output_converter_class_name = f"{to_camel_case(output_format)}Converter"
        output_converter_class = getattr(output_converter_module, output_converter_class_name)
        
        # Load input dataset
        click.echo(f"Loading dataset {input_format} from {input_path}...")
        input_dataset = input_format_class.read_from_folder(folder_path = input_path)
        
        # Convert to NeutralFormat
        click.echo(f"Converting from {input_format} to neutral format...")
        neutral_format = input_converter_class.toNeutral(input_dataset)
        
        # Convert to output format
        click.echo(f"Converting from neutral format to {output_format}...")
        output_dataset = output_converter_class.fromNeutral(neutral_format)
        
        # Save output dataset 
        click.echo(f"Saving dataset {output_format} in {output_path}...")
        output_dataset.save(output_path)
        
    except ImportError as e:
        click.echo(f"Error: Could not import necessary modules: {e}", err=True)
        sys.exit(1)
    except AttributeError as e:
        click.echo(f"Error: Lacking class or required method: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error while converting: {e}", err=True)
        sys.exit(1)

def to_camel_case(snake_str: str) -> str:
    return "".join(word.capitalize() for word in snake_str.split("_"))


if __name__ == "__main__":
    dconverter()
