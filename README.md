# DatasetConverter

![License](https://img.shields.io/github/license/GCousido/TFG-DatasetConverter)
![Status](https://img.shields.io/badge/status-in%20development-yellow)
![Last Commit](https://img.shields.io/github/last-commit/GCousido/TFG-DatasetConverter)

## Index

* [Description](#description)
* [Installation](#installation)
* [How to Use](#how-to-use)
* [Supported Formats](#supported-formats)
* [License](#license)

## Description

DatasetConverter is a **library** that also includes a **CLI tool** for converting object detection annotation datasets between popular formats. It simplifies dataset interoperability for machine learning and computer vision projects.

Key Features:

* **Bidirectional conversion** between supported formats
* **Unified internal representation** ensures consistent and reliable transformations

Conversion Process:

1. **Load** the input dataset from the specified path
2. **Transforms** to internal representation
3. **Convert** from internal representation to target output format
4. **Save** the converted dataset to the desired output location

---

## Installation

### Install from Source

Clone the repository and install the package:

```bash
git clone https://github.com/GCousido/TFG-DatasetConverter.git
cd TFG-DatasetConverter
pip install  .
```

### Development Installation

For development (including dependencies for testing) and in editable mode:

```bash
git clone https://github.com/GCousido/TFG-DatasetConverter.git
cd TFG-DatasetConverter
pip install -e ".[dev]"
```

---

## How to Use

### Library Usage

You can use DatasetConverter as a Python library to convert datasets programmatically.

#### Example

```python
from datasetconverter import YoloFormat, YoloConverter, CocoFormat, CocoConverter, NeutralFormat

yolo_dataset: YoloFormat = YoloFormat.read_from_folder(input_path)

internal_dataset: NeutralFormat = YoloConverter.toNeutral(yolo_dataset)

coco_dataset: CocoFormat = CocoConverter.fromNeutral(internal_dataset)

coco_dataset.save(output_path)
```

### Command Line Interface

The CLI provides a simple interface for converting datasets:

#### Basic Usage

```bash
dconverter --input-format <INPUT_FORMAT> --input-path <INPUT_PATH> --output-format <OUTPUT_FORMAT> --output-path <OUTPUT_PATH> [OPTIONS]
```

#### Required Arguments

* `--input-format`: Source format
* `--input-path`: Path to the folder containing the input dataset
* `--output-format`: Target format
* `--output-path`: Path to save the converted dataset

#### Examples

Convert a **YOLO** dataset to **COCO**:

```bash
dconverter --input-format yolo --input-path ./datasets/yolo --output-format coco --output-path ./datasets/coco
```

Convert **Pascal VOC** to **YOLO**:

```bash
dconverter --input-format pascal_voc --input-path ./datasets/pascalvoc --output-format yolo --output-path ./datasets/yolo
```

Convert **COCO** to **Pascal VOC**:

```bash
dconverter --input-format coco --input-path ./datasets/coco --output-format pascal_voc --output-path ./datasets/pascalvoc
```

---

## Supported Formats

| Format | Input | Output | Description |
|--------|-------|--------|-------------|
| **YOLO** | ✅ | ✅ | YOLO format (.txt files with normalized coordinates and classes.txt for class names) |
| **COCO** | ✅ | ✅ | Microsoft COCO format (.json with absolute coordinates) |
| **Pascal VOC** | ✅ | ✅ | Pascal Visual Object Classes format (.xml files with absolute coordinates) |

### Format Specifications

#### YOLO Format

* **File Structure**: One `.txt` file per image with same basename as the image
* **Annotation Format**: `<class_id> <x_center> <y_center> <width> <height>`
* **Coordinates**: Normalized values between 0 and 1 (relatives to the image size)
* **Additional Files**: `classes.txt` containing class names, one per line

#### COCO Format

* **File Structure**: Single `.json` file containing all annotations
* **Annotation Format**: JSON with images, annotations and categories arrays
* **Coordinates**: Absolute pixel values `[x, y, width, height]`
* **Metadata**: Includes dataset `info`, `licenses`, and `category` definitions

#### Pascal VOC Format

* **File Structure**: One `.xml` file per image, sharing the basename with the image file
* **Annotation Format**: XML structure with bounding box coordinates and class names
* **Coordinates**: Absolute pixel values `<xmin>, <ymin>, <xmax>, <ymax>`
* **Metadata**: Rich annotation metadata, including image `size`, object attributes (`difficult`, `truncated`, `occluded`), and `source` info

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
