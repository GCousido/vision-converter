from pathlib import Path
import pytest

from formats.yolo import YoloFormat

# Fixture para dataset YOLO de prueba
@pytest.fixture
def sample_yolo_dataset(tmp_path):
    # Crear estructura de directorios
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    
    # Archivo de clases
    (labels_dir / "classes.txt").write_text("person\ncar\ntruck")
    
    # Archivo de anotaciones 1
    (labels_dir / "image1.txt").write_text(
        "0 0.5 0.5 0.3 0.3\n"
        "1 0.2 0.2 0.1 0.1"
    )
    
    # Archivo de anotaciones 2
    (labels_dir / "image2.txt").write_text(
        "2 0.7 0.7 0.4 0.4"
    )
    
    return tmp_path

def test_yolo_format_construction(sample_yolo_dataset):
    # Ejecutar método bajo prueba
    yolo_format = YoloFormat.read_from_folder(sample_yolo_dataset)
    
    # 1. Verificación básica de estructura
    assert yolo_format.name == sample_yolo_dataset.name
    assert isinstance(yolo_format.files, list)
    
    # 2. Verificación de clases
    assert yolo_format.class_labels == ["person", "car", "truck"]
    assert len(yolo_format.class_labels) == 3
    
    # 3. Verificación de archivos procesados
    assert len(yolo_format.files) == 2
    filenames = {f.filename for f in yolo_format.files}
    assert "image1.txt" in filenames
    assert "image2.txt" in filenames
    
    # 4. Verificación detallada de bounding boxes
    file1 = next(f for f in yolo_format.files if f.filename == "image1.txt")
    assert len(file1.annotations) == 2
    
    # Primera anotación
    ann1 = file1.annotations[0]
    assert ann1.id_class == 0
    assert ann1.bbox.x_center == 0.5
    assert ann1.bbox.y_center == 0.5
    assert ann1.bbox.width == 0.3
    assert ann1.bbox.height == 0.3
    
    # Segunda anotación
    ann2 = file1.annotations[1]
    assert ann2.id_class == 1
    assert ann2.bbox.x_center == 0.2
    assert ann2.bbox.y_center == 0.2
    assert ann2.bbox.width == 0.1
    assert ann2.bbox.height == 0.1


def test_invalid_dataset_structure(tmp_path):
    # Caso sin directorio labels
    with pytest.raises(FileNotFoundError):
        YoloFormat.read_from_folder(tmp_path)
    
    # Crear directorio labels vacío
    (tmp_path / "labels").mkdir()
    
    # Caso sin classes.txt
    with pytest.raises(FileNotFoundError):
        YoloFormat.read_from_folder(tmp_path)

    # Archivo de clases
    (tmp_path / "labels" / "classes.txt").write_text("person\ncar\ntruck")

    # Caso con classes.txt
    assert (tmp_path / "labels" / "classes.txt").exists()