# VisionConverter

![License](https://img.shields.io/github/license/GCousido/VisionConverter)
![Status](https://img.shields.io/badge/status-in%20development-yellow)
![Last Commit](https://img.shields.io/github/last-commit/GCousido/VisionConverter)

## Index

* [Description](#description)
* [Installation](#installation)
* [How to Use](#how-to-use)
* [Supported Formats](#supported-formats)
* [License](#license)

## Description

VisionConverter is a **library** for converting object detection annotation datasets between popular formats. It simplifies dataset interoperability for machine learning and computer vision projects.

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

### Requirements

* Python >= 3.12
* [Click](https://pypi.org/project/click/) >= 8.1.8
* [Pillow](https://pypi.org/project/Pillow/) >= 11.2.1

### Install from PyPI

```bash
pip install vision-converter
```

### Install from Source

Clone the repository and install the package:

```bash
git clone https://github.com/GCousido/VisionConverter.git
cd VisionConverter
pip install  .
```

### Development Installation

For development including dependencies for testing and in editable mode:

```bash
git clone https://github.com/GCousido/VisionConverter.git
cd VisionConverter
pip install -e ".[dev]"
```

---

## How to Use

### Library Usage

You can use VisionConverter as a Python library to convert datasets programmatically.

#### Example

```python
from vision_converter import YoloFormat, YoloConverter, CocoFormat, CocoConverter, NeutralFormat

yolo_dataset: YoloFormat = YoloFormat.read_from_folder("./dataset/yolo")

internal_dataset: NeutralFormat = YoloConverter.toNeutral(yolo_dataset)

coco_dataset: CocoFormat = CocoConverter.fromNeutral(internal_dataset)

coco_dataset.save("./dataset/coco")
```

### Command Line Interface

The CLI provides a simple interface for converting datasets:

#### Basic Usage

```bash
vconverter --input-format <INPUT_FORMAT> --input-path <INPUT_PATH> --output-format <OUTPUT_FORMAT> --output-path <OUTPUT_PATH> <OPTIONS>
```

#### Required Arguments

* `--input-format`: Source format
* `--input-path`: Path to the folder containing the input dataset
* `--output-format`: Target format
* `--output-path`: Path to save the converted dataset

#### Options

* `--copy-images`: Copy images files to the output directory.
* `--symlink-images`: Creates symbolic links to the original images in the output directory.

#### Examples

Convert a **YOLO** dataset to **COCO**:

```bash
vconverter --input-format yolo --input-path ./datasets/yolo --output-format coco --output-path ./datasets/coco
```

Convert **Pascal VOC** to **YOLO**:

```bash
vconverter --input-format pascal_voc --input-path ./datasets/pascalvoc --output-format yolo --output-path ./datasets/yolo
```

Convert **COCO** to **Pascal VOC** with images:

```bash
vconverter --input-format coco --input-path ./datasets/coco --output-format pascal_voc --output-path ./datasets/pascalvoc --copy-images
```

---

## Supported Formats

| Format | Input | Output | Parameter Value | Description |
|--------|-------|--------|-----------------|-------------|
| **YOLO** | ✅ | ✅ | yolo | YOLO format (.txt files with normalized coordinates and classes.txt for class names) |
| **COCO** | ✅ | ✅ | coco | Microsoft COCO format (.json with absolute coordinates) |
| **Pascal VOC** | ✅ | ✅ | pascal_voc | Pascal Visual Object Classes format (.xml files with absolute coordinates) |
| **CreateML**  | ✅ | ✅ | createml | Apple CreateML format (.json with centered bounding boxes and absolute coordinates)|
| **TensorFlow CSV** | ✅ | ✅ | tensorflow_csv | TensorFlow Object Detection CSV format (.csv with absolute coordinates) |
| **LabelMe** | ✅ | ✅ | labelme | LabelMe JSON format (.json files with shape annotations and optional embedded image data)|
| **VGG** | ✅ | ✅ | vgg | VGG Image Annotator format (.json with multiple shape types and region attributes) |

### Format Specifications

#### YOLO Format

* **File Structure**: One `.txt` file per image with same basename as the image
* **Annotation Format**: `<class_id> <x_center> <y_center> <width> <height>`
* **Coordinates**: Normalized values between 0 and 1 (relatives to the image size)
* **Additional Files**: `classes.txt` containing class names, one per line

```text
EXPECTED INPUT FILE STRUCTURE                      GENERATED OUTPUT FILE STRUCTURE
      dataset/                                           dataset/
        ├── images/                                        ├── images/
        │     img1.jpg                                     │
        │     img2.jpg                                     │
        ├── labels/                                        ├── labels/
        │     img1.txt                                     │     img1.txt
        │     img2.txt                                     │     img2.txt
        │     classes.txt                                  │     classes.txt
```

#### COCO Format

* **File Structure**: Single `.json` file containing all annotations
* **Annotation Format**: JSON with images, annotations and categories arrays
* **Coordinates**: Absolute pixel values `[x, y, width, height]`
* **Metadata**: Includes dataset `info`, `licenses`, and `category` definitions

```text
EXPECTED INPUT FILE STRUCTURE                      GENERATED OUTPUT FILE STRUCTURE
      dataset/                                           dataset/
        ├── images/                                        ├── images/
        │     img1.jpg                                     |
        │     img2.jpg                                     |   
        ├── annotations.json                               ├── annotations.json   
```

#### Pascal VOC Format

* **File Structure**: One `.xml` file per image, sharing the basename with the image file
* **Annotation Format**: XML structure with bounding box coordinates and class names
* **Coordinates**: Absolute pixel values `<xmin>, <ymin>, <xmax>, <ymax>`
* **Metadata**: Rich annotation metadata, including image `size`, object attributes (`difficult`, `truncated`, `occluded`), and `source` info

```text
EXPECTED INPUT FILE STRUCTURE                      GENERATED OUTPUT FILE STRUCTURE
      dataset/                                           dataset/
        ├── JPEGImages/                                    ├── JPEGImages/
        │     img1.jpg                                     │     
        │     img2.jpg                                     │     
        ├── Annotations/                                   ├── Annotations/
        │     img1.xml                                     │     img1.xml
        │     img2.xml                                     │     img2.xml
        |-- ImageSets/                                     |-- ImageSets/
```

#### CreateML Format

* **File Structure**: Single `.json` file containing all annotations and an images/ folder with image files
* **Annotation Format**: JSON array with entries for each image, each containing image filename and annotations array
* **Coordinates**: Absolute pixel values with bounding boxes defined by center coordinates and dimensions `{x_center, y_center, width, height}`

```text
EXPECTED INPUT FILE STRUCTURE                      GENERATED OUTPUT FILE STRUCTURE
      dataset/                                           dataset/
        ├── images/                                        ├── images/
        │     img1.jpg                                     │     
        │     img2.jpg                                     │     
        ├── annotations.json                               ├── annotations.json
```

#### TensorFlow Object Detection CSV Format

* **File Structure**: Single `.csv` file containing all annotations
* **Annotation Format**: CSV structure with specific columns for image metadata and bounding box coordinates
* **Coordinates**: Absolute pixel values `<xmin>, <ymin>, <xmax>, <ymax>`
* **Required Columns**: `filename`, `width`, `height`, `class`, `xmin`, `ymin`, `xmax`, `ymax`
* **Features**: Human-readable format, direct compatibility with TensorFlow Object Detection API, supports multiple objects per image

```text
EXPECTED INPUT FILE STRUCTURE                      GENERATED OUTPUT FILE STRUCTURE
      dataset/                                           dataset/
        ├── images/                                        ├── images/
        │     img1.jpg                                     │     
        │     img2.jpg                                     │     
        ├── annotations.csv                                ├── annotations.csv
```

#### LabelMe JSON Format

* **File Structure**: One `.json` file per image containing annotations and image metadata
* **Annotation Format**: JSON with shapes array, each shape having `label`, `points`, `shape_type`, `group_id`, `flags`, and optional `description`
* **Coordinates**: Absolute pixel values for `points` defining `shapes` (e.g., polygons, rectangles)
* **Image Data**: Optional `base64` encoded image data embedded in `imageData` field
* **Metadata**: Includes dataset `version`, `flags`, `imagePath`, `imageHeight`, `imageWidth`

```text
EXPECTED INPUT FILE STRUCTURE                      GENERATED OUTPUT FILE STRUCTURE
      dataset/                                           dataset/
        ├── img1.jpg                                       ├── img1.jpg
        ├── img1.json                                      ├── img1.json
        ├── img2.jpg                                       ├── img2.jpg
        ├── img2.json                                      ├── img2.json
```

#### VGG Image Annotator Format

* **File Structure**: Single `.json` file containing all annotations with VIA metadata structure
* **Annotation Format**: JSON with `_via_img_metadata` containing image entries, each with `regions` array for shape annotations
* **Coordinates**: Absolute pixel values with support for 6 shape types: `rect`, `circle`, `ellipse`, `polygon`, `polyline`, `point`
* **Shape Types**:
  * **Rectangle**: `{x, y, width, height}` - top-left corner and dimensions
  * **Circle**: `{cx, cy, r}` - center coordinates and radius
  * **Ellipse**: `{cx, cy, rx, ry, theta}` - center, radii, and rotation angle
  * **Polygon**: `{all_points_x[], all_points_y[]}` - arrays of vertex coordinates
  * **Polyline**: `{all_points_x[], all_points_y[]}` - arrays of line point coordinates
  * **Point**: `{cx, cy}` - single point coordinates
* **Metadata**: Includes `file_attributes` for image-level data, `region_attributes` for annotation-level data, and optional VIA project settings

```text
EXPECTED INPUT FILE STRUCTURE                      GENERATED OUTPUT FILE STRUCTURE
      dataset/                                           dataset/
        |-- images/                                        ├── images/
        |     img1.jpg                                     |
        |     img2.jpg                                     | 
        ├── annotations.json                               ├── annotations.json 
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
