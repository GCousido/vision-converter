from datetime import datetime
import re

from pathlib import Path

from converters.dataset_converter import DatasetConverter
from formats.coco import Category, CocoBoundingBox, CocoFile, CocoFormat, CocoImage, CocoLabel, Info, License
from formats.neutral_format import ImageOrigin, NeutralAnnotation, NeutralFile, NeutralFormat
from formats.pascal_voc import PascalVocBoundingBox
from utils.bbox_utils import CocoBBox_to_PascalVocBBox, PascalVocBBox_to_CocoBBox


class CocoConverter(DatasetConverter[CocoFormat]):
    """Converter between CocoFormat and NeutralFormat
    
    Provides bidirectional conversion capabilities between CocoFormat and NeutralFormat
    """
    
    @staticmethod
    def toNeutral(df: CocoFormat) -> NeutralFormat:
        """Convert a CocoFormat dataset to Neutral format.

        Args:
            df (CocoFormat): Source dataset in COCO format

        Returns:
            NeutralFormat: Converted dataset in neutral annotation format
        """
        neutral_files = []
        class_names = set()

        for file in df.files:
            # Map category_id -> category_name
            category_map: dict[int, str] = {cat.id: cat.name for cat in file.categories}
            
            # Group annotations by image_id
            annotations_by_image = {}
            for ann in file.annotations:
                if ann.image_id not in annotations_by_image:
                    annotations_by_image[ann.image_id] = []
                annotations_by_image[ann.image_id].append(ann)
            
            # Create NeutralFile per image
            for image in file.images:
                image_anns = annotations_by_image.get(image.id, [])
                
                # Convert annotations
                neutral_anns: list[NeutralAnnotation] = [COCOLabel_to_NeutralAnnotation(ann, category_map) for ann in image_anns]
                
                # Add classes to the set
                class_names.update([ann.class_name for ann in neutral_anns])


                source_types = []
                source_ids = []
                urls = []

                if image.flickr_url is not None:
                    m = re.search(r'/(\d+)_', image.flickr_url)
                    source_types.append("flickr")
                    flickrid = m.group(1) if m else None
                    source_ids.append(flickrid)
                    urls.append(image.flickr_url)

                if image.coco_url is not None:
                    m = re.search(r'/(\d+)\.jpg', image.coco_url)
                    source_types.append("coco")
                    cocoid = m.group(1) if m else None
                    source_ids.append(cocoid)
                    urls.append(image.coco_url)

                image_license = next((lic for lic in file.licenses if lic.id == image.license), None) if file.licenses else None

                image_origin = ImageOrigin(
                    source_type = source_types,
                    source_id = source_ids,
                    image_url = urls,
                    image_license = image_license.name if image_license is not None else None,
                    license_url = image_license.url if image_license is not None else None,
                    date_captured = image.date_captured,
                    extension = Path(image.file_name).suffix
                )
                
                neutral_files.append(NeutralFile(
                    filename = Path(image.file_name).stem,
                    annotations = neutral_anns,
                    width = image.width,
                    height = image.height,
                    depth = 3,  # Asuming RGB images
                    image_origin = image_origin,
                    params = None
                ))

        # Convert class name set to a dictionary with ids
        class_map = {i: class_name for i, class_name in enumerate(sorted(class_names))}
    
        return NeutralFormat(
            name = df.name,
            files = neutral_files,
            original_format = "coco",
            metadata = None,
            class_map = class_map
        )

    
    @staticmethod
    def fromNeutral(nf: NeutralFormat) -> CocoFormat:
        """Convert a NeutralFormat dataset to CocoFormat.

        Args:
            nf (NeutralFormat): Source dataset in neutral format

        Returns:
            CocoFormat: Converted dataset in COCO format
        """

        # Group NeutralFiles per COCO filename (Assuming 1 file per dataset)
        category_map = {}

        next_cat_id = 1
        categories = []

        next_img_id = 1
        images = []

        next_ann_id = 1
        annotations = []

        next_license_id = 1
        license_map = {}
        licenses = []

        actual_date = datetime.now()
        
        for neutral_file in nf.files:

            # Managing Licenses
            if neutral_file.image_origin and neutral_file.image_origin.image_license:
                image_license: str = neutral_file.image_origin.image_license 
                if image_license not in license_map:
                    license_map[image_license] = next_license_id

                    license = License(
                            id = next_license_id,
                            name = image_license,
                            url = neutral_file.image_origin.license_url if neutral_file.image_origin.license_url is not None else ""
                    )
                    licenses.append(license)

                    next_license_id += 1

                license_id = license_map[image_license]
            else:
                license_id = None  # default value

            flickr_url = None
            coco_url = None
            # Managing flickr_url
            if neutral_file.image_origin.source_type and "flickr" in neutral_file.image_origin.source_type and neutral_file.image_origin.image_url:
                try:
                    flickr_url = neutral_file.image_origin.image_url[neutral_file.image_origin.source_type.index("flickr")]
                except ValueError:
                    pass

            # Managing coco_url
            if neutral_file.image_origin.source_type and "coco" in neutral_file.image_origin.source_type and neutral_file.image_origin.image_url:
                try:
                    coco_url = neutral_file.image_origin.image_url[neutral_file.image_origin.source_type.index("coco")]
                except ValueError:
                    pass

            # Managing date_captured
            if neutral_file.image_origin.date_captured is not None:
                date_captured = neutral_file.image_origin.date_captured
            else:
                date_captured = ""

            image = CocoImage(
                id = next_img_id,
                width = neutral_file.width,
                height = neutral_file.height,
                file_name = neutral_file.filename + neutral_file.image_origin.extension,
                license = license_id,
                flickr_url = flickr_url,
                coco_url = coco_url,
                date_captured = date_captured
            )
            images.append(image)
            
            # Procces annotations
            for ann in neutral_file.annotations:
                # Map class_name -> id
                if ann.class_name not in category_map:
                    category_map[ann.class_name] = next_cat_id
                    categories.append(Category(
                        id = next_cat_id,
                        name = ann.class_name,
                        supercategory = "" # Default to empty
                    ))
                    next_cat_id += 1
                
                # Convert bbox
                coco_bbox: CocoBoundingBox = PascalVocBBox_to_CocoBBox(ann.bbox)
                
                # Create CocoLabel
                coco_ann = CocoLabel(
                    bbox = coco_bbox,
                    id = next_ann_id,
                    image_id = next_img_id,
                    category_id = category_map[ann.class_name],
                    segmentation = ann.attributes.get("segmentation", []), # at this moment, only supports bbox conversions
                    area = float(coco_bbox.width * coco_bbox.height), # not supporting segmentation
                    iscrowd = ann.attributes.get("iscrowd", False)
                )
                annotations.append(coco_ann)
                next_ann_id += 1
            
            next_img_id += 1
        
        coco_file = CocoFile(
            filename = "annotations.json",
            annotations = annotations,
            info=Info(description="COCO Dataset created after conversion from " + nf.original_format.replace('_',' ').title(), url="", version="1.0", year=actual_date.year, contributor="DatasetConverter", date_created=actual_date.strftime("%Y/%m/%d")),
            licenses = licenses,
            images = images,
            categories = categories
        )

        return CocoFormat(
            name=nf.name,
            files=[coco_file]
        )


def COCOLabel_to_NeutralAnnotation(annotation: CocoLabel, category_map: dict[int, str]) -> NeutralAnnotation:
    """Convert a CocoLabel to Neutral format.

    Args:
        annotation (CocoLabel): Source annotation in COCO format
        category_map (dict[int, str]): Map with ids to category_names

    Returns:
        NeutralAnnotation: Converted annotation in Neutral format
    """
    bbox: PascalVocBoundingBox = CocoBBox_to_PascalVocBBox(annotation.bbox)

    return NeutralAnnotation(
        bbox = bbox,
        class_name = category_map[annotation.category_id],
        attributes = {
            "iscrowd": annotation.iscrowd,
            "segmentation": annotation.segmentation
        }
    )