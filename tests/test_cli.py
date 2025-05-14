import pytest
import sys

from click.testing import CliRunner
from pathlib import Path

# Add root directory to the path
sys.path.append(str(Path(__file__).parent.parent))

# Import
from cli.main import dconverter, FORMATS

# Fixture for Click Runner
@pytest.fixture
def runner():
    return CliRunner()

# Fixture for mocking modules and classes
@pytest.fixture(autouse=True)
def mock_imports(mocker):
    # Create mock for format and converter modules
    format_mocks = {}
    converter_mocks = {}
    
    for fmt in FORMATS:
        # 1. Mock for one format with a load() method that returns the format itself
        fmt_mock = mocker.MagicMock()
        fmt_class_mock = mocker.MagicMock()
        fmt_class_mock.load.return_value = mocker.MagicMock()
        
        #  Configure the format to return a mock class when accesing with getattr
        setattr(fmt_mock, f"{fmt.capitalize()}Format", fmt_class_mock)
        format_mocks[f'formats.{fmt}'] = fmt_mock
        
        # 2. Mock for converter
        conv_mock = mocker.MagicMock()
        conv_class_mock = mocker.MagicMock()
        
        # Output dataset have a module with save() method and return
        output_dataset = mocker.MagicMock()
        output_dataset.save.return_value = None
        
        # Configure static method for each converter
        conv_class_mock.toNeutral.return_value = mocker.MagicMock()  # neutral format
        conv_class_mock.fromNeutral.return_value = output_dataset
        
        # Asign mock class to the converter
        setattr(conv_mock, f"{fmt.capitalize()}Converter", conv_class_mock)
        converter_mocks[f'converters.{fmt}_converter'] = conv_mock
    
    # All mocks in a single dictionary
    all_mocks = {**format_mocks, **converter_mocks}
    
    # Aply mocks
    mocker.patch.dict('sys.modules', all_mocks)
    
    return all_mocks


# Succesful conversion cases
@pytest.mark.parametrize("input_fmt, output_fmt", [
    ('coco', 'yolo'),
    ('yolo', 'pascal_voc'),
    ('pascal_voc', 'coco')
])
def test_successful_conversion(runner, input_fmt, output_fmt ,  tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"

    result = runner.invoke(
        dconverter,
        ['-if', input_fmt, '-ip', str(input_dir), '-of', output_fmt, '-op', str(output_dir)]
    )
    
    assert result.exit_code == 0
    assert f"Loading dataset {input_fmt}" in result.output
    assert f"Converting from {input_fmt} to neutral format" in result.output
    assert f"Converting from neutral format to {output_fmt}" in result.output
    assert f"Saving dataset {output_fmt}" in result.output


# Case with an output that cant be written
def test_non_writable_output(runner, mocker, tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    
    output_path = "/readonly/output"
    
    # Verify permissions at the parent directory
    def mock_access(path, mode):
        parent_dir = str(Path(output_path).parent).replace('\\', '/')
        return not (Path(path).resolve() == Path(parent_dir).resolve())
    
    mocker.patch('os.access', mock_access)
    
    result = runner.invoke(
        dconverter,
        ['-if', 'coco', '-ip', str(input_dir), '-of', 'yolo', '-op', output_path]
    )
    
    assert result.exit_code == 1
    assert "Output path is not writable" in result.output



# Case wtih a missing converter class
def test_missing_converter_class(runner, mocker, tmp_path):
    # Create actual input directory
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    
    mocker.patch.dict('sys.modules', {'converters.coco_converter': None})
    
    result = runner.invoke(
        dconverter,
        ['-if', 'coco', '-ip', str(input_dir), '-of', 'yolo', '-op', 'output']
    )
    assert result.exit_code == 1


# Case with an error during conversion
def test_general_conversion_error(runner, mocker, tmp_path):
    # Create actual input directory
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    
    mocker.patch(
        'converters.coco_converter.CocoConverter.toNeutral',
        side_effect=Exception("Conversion failed")
    )
    
    result = runner.invoke(
        dconverter,
        ['-if', 'coco', '-ip', str(input_dir), '-of', 'yolo', '-op', 'output']
    )
    assert result.exit_code == 1


# Case where there are missing required params
def test_missing_required_parameters(runner):
    result = runner.invoke(dconverter)
    assert result.exit_code == 2
    assert "Missing option" in result.output


# Case with an invalid input format
def test_invalid_input_format(runner):
    result = runner.invoke(
        dconverter,
        ['-if', 'invalid', '-ip', 'input', '-of', 'yolo', '-op', 'output']
    )
    assert result.exit_code == 2
    assert "Invalid value for '--input-format' / '-if'" in result.output



# Case with an invalid output format
def test_invalid_output_format(runner, tmp_path):
    # Create actual input directory
    input_dir = tmp_path / "input"
    input_dir.mkdir()


    result = runner.invoke(
        dconverter,
        ['-if', 'yolo', '-ip', str(input_dir), '-of', 'invalid', '-op', 'output']
    )
    assert result.exit_code == 2
    assert "Invalid value for '--output-format' / '-of'" in result.output


# Case in which it is necesary to create output directory
def test_output_directory_creation(runner, mocker, tmp_path):
    # Create actual input directory
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    
    output_path = tmp_path / "new_directory" / "dataset"
    mocker.patch('os.access', return_value=True)
    
    result = runner.invoke(
        dconverter,
        ['-if', 'coco', '-ip', str(input_dir), '-of', 'yolo', '-op', str(output_path)]
    )
    assert result.exit_code == 0

