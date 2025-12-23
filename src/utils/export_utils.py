"""
Export utilities for annotation data.

This module provides:
- COCO format dataset export
- YOLO format dataset export
- VOC/Pascal format dataset export
- Dataset conversion utilities
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import numpy as np

from src.logger import get_logger
from src.utils.mask_utils import binary_mask_to_rle, mask_to_bbox, mask_to_polygon
from src.utils.file_utils import ensure_dir, save_json

logger = get_logger(__name__)


class COCOExporter:
    """
    Export annotations to COCO JSON format.
    
    COCO format structure:
    {
        "info": {...},
        "licenses": [...],
        "images": [...],
        "annotations": [...],
        "categories": [...]
    }
    """
    
    def __init__(self, 
                 dataset_name: str = "Industrial Defect Dataset",
                 description: str = "Dataset for industrial defect segmentation",
                 version: str = "1.0"):
        """
        Initialize COCO exporter.
        
        Args:
            dataset_name: Name of the dataset
            description: Dataset description
            version: Dataset version
        """
        self.dataset_name = dataset_name
        self.description = description
        self.version = version
        
        # Initialize COCO structure
        self.coco_data = {
            "info": {
                "description": description,
                "version": version,
                "year": datetime.now().year,
                "contributor": "Industrial AI Team",
                "date_created": datetime.now().isoformat()
            },
            "licenses": [
                {
                    "id": 1,
                    "name": "Custom License",
                    "url": ""
                }
            ],
            "images": [],
            "annotations": [],
            "categories": []
        }
        
        self.image_id_counter = 1
        self.annotation_id_counter = 1
        self.category_dict = {}  # category_name -> category_id
    
    def add_category(self, category_name: str, supercategory: str = "defect") -> int:
        """
        Add a category to the dataset.
        
        Args:
            category_name: Name of the category
            supercategory: Parent category name
            
        Returns:
            Category ID
        """
        if category_name in self.category_dict:
            return self.category_dict[category_name]
        
        category_id = len(self.category_dict) + 1
        self.category_dict[category_name] = category_id
        
        self.coco_data["categories"].append({
            "id": category_id,
            "name": category_name,
            "supercategory": supercategory
        })
        
        logger.debug(f"Added category: {category_name} (id={category_id})")
        return category_id
    
    def add_image(self, 
                  image_path: str,
                  width: int,
                  height: int,
                  image_id: Optional[int] = None) -> int:
        """
        Add an image to the dataset.
        
        Args:
            image_path: Path to the image file
            width: Image width
            height: Image height
            image_id: Optional custom image ID
            
        Returns:
            Image ID
        """
        if image_id is None:
            image_id = self.image_id_counter
            self.image_id_counter += 1
        
        image_filename = Path(image_path).name
        
        self.coco_data["images"].append({
            "id": image_id,
            "file_name": image_filename,
            "width": width,
            "height": height,
            "license": 1,
            "date_captured": datetime.now().isoformat()
        })
        
        logger.debug(f"Added image: {image_filename} (id={image_id})")
        return image_id
    
    def add_annotation(self,
                       image_id: int,
                       category_id: int,
                       mask: np.ndarray,
                       annotation_id: Optional[int] = None) -> int:
        """
        Add an annotation to the dataset.
        
        Args:
            image_id: ID of the image
            category_id: ID of the category
            mask: Binary mask (H, W)
            annotation_id: Optional custom annotation ID
            
        Returns:
            Annotation ID
        """
        if annotation_id is None:
            annotation_id = self.annotation_id_counter
            self.annotation_id_counter += 1
        
        # Get bbox
        bbox = mask_to_bbox(mask)
        if bbox is None:
            logger.warning(f"Empty mask for annotation {annotation_id}, skipping")
            return annotation_id
        
        # Get RLE segmentation
        rle = binary_mask_to_rle(mask)
        
        # Calculate area
        area = int(np.sum(mask > 0))
        
        annotation = {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": bbox,  # [x, y, width, height]
            "area": area,
            "segmentation": rle,
            "iscrowd": 0
        }
        
        self.coco_data["annotations"].append(annotation)
        logger.debug(f"Added annotation: id={annotation_id}, image_id={image_id}, area={area}")
        return annotation_id
    
    def save(self, output_path: str):
        """
        Save COCO dataset to JSON file.
        
        Args:
            output_path: Path to save JSON file
        """
        ensure_dir(Path(output_path).parent)
        save_json(self.coco_data, output_path, indent=2)
        logger.info(f"Saved COCO dataset to: {output_path}")
        logger.info(f"  Images: {len(self.coco_data['images'])}")
        logger.info(f"  Annotations: {len(self.coco_data['annotations'])}")
        logger.info(f"  Categories: {len(self.coco_data['categories'])}")
    
    def get_statistics(self) -> Dict:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            "num_images": len(self.coco_data["images"]),
            "num_annotations": len(self.coco_data["annotations"]),
            "num_categories": len(self.coco_data["categories"]),
            "category_names": [cat["name"] for cat in self.coco_data["categories"]]
        }
        
        # Count annotations per category
        category_counts = {}
        for ann in self.coco_data["annotations"]:
            cat_id = ann["category_id"]
            category_counts[cat_id] = category_counts.get(cat_id, 0) + 1
        
        stats["annotations_per_category"] = category_counts
        
        return stats


class YOLOExporter:
    """
    Export annotations to YOLO format.
    
    YOLO format:
    - One txt file per image
    - Each line: class_id x1 y1 x2 y2 x3 y3 ... (normalized polygon coordinates)
    - classes.txt file with class names
    """
    
    def __init__(self, output_dir: str, class_names: List[str]):
        """
        Initialize YOLO exporter.
        
        Args:
            output_dir: Output directory for YOLO files
            class_names: List of class names in order
        """
        self.output_dir = Path(output_dir)
        self.class_names = class_names
        
        # Create output directory
        ensure_dir(self.output_dir)
        
        # Save classes.txt
        classes_file = self.output_dir / "classes.txt"
        with open(classes_file, 'w') as f:
            for class_name in class_names:
                f.write(f"{class_name}\n")
        
        logger.info(f"Created YOLO exporter: {output_dir}")
        logger.info(f"Classes: {class_names}")
    
    def export_annotation(self,
                         image_path: str,
                         masks: List[np.ndarray],
                         class_ids: List[int],
                         image_width: int,
                         image_height: int):
        """
        Export annotations for one image to YOLO format.
        
        Args:
            image_path: Path to the image file
            masks: List of binary masks
            class_ids: List of class IDs (0-indexed)
            image_width: Image width
            image_height: Image height
        """
        image_name = Path(image_path).stem
        output_file = self.output_dir / f"{image_name}.txt"
        
        lines = []
        
        for mask, class_id in zip(masks, class_ids):
            # Get polygons from mask
            polygons = mask_to_polygon(mask)
            
            if not polygons:
                logger.warning(f"No polygons found in mask for {image_name}")
                continue
            
            for polygon in polygons:
                if len(polygon) < 6:  # Need at least 3 points
                    continue
                
                # Normalize coordinates to [0, 1]
                normalized_polygon = []
                for i in range(0, len(polygon), 2):
                    x = polygon[i] / image_width
                    y = polygon[i + 1] / image_height
                    normalized_polygon.extend([x, y])
                
                # Format: class_id x1 y1 x2 y2 x3 y3 ...
                line = f"{class_id} " + " ".join([f"{coord:.6f}" for coord in normalized_polygon])
                lines.append(line)
        
        # Write to file
        with open(output_file, 'w') as f:
            f.write("\n".join(lines))
        
        logger.debug(f"Exported YOLO annotation: {output_file} ({len(lines)} objects)")
    
    def create_data_yaml(self, 
                        train_path: str,
                        val_path: str,
                        test_path: Optional[str] = None):
        """
        Create data.yaml for YOLO training.
        
        Args:
            train_path: Path to training images directory
            val_path: Path to validation images directory
            test_path: Optional path to test images directory
        """
        data_yaml = {
            'path': str(self.output_dir.absolute()),
            'train': train_path,
            'val': val_path,
            'names': {i: name for i, name in enumerate(self.class_names)}
        }
        
        if test_path:
            data_yaml['test'] = test_path
        
        yaml_path = self.output_dir / "data.yaml"
        
        # Write YAML manually (simple format)
        with open(yaml_path, 'w') as f:
            f.write(f"path: {data_yaml['path']}\n")
            f.write(f"train: {data_yaml['train']}\n")
            f.write(f"val: {data_yaml['val']}\n")
            if test_path:
                f.write(f"test: {data_yaml['test']}\n")
            f.write("\nnames:\n")
            for idx, name in data_yaml['names'].items():
                f.write(f"  {idx}: {name}\n")
        
        logger.info(f"Created data.yaml: {yaml_path}")


class VOCExporter:
    """
    Export annotations to Pascal VOC XML format.
    
    VOC format:
    - One XML file per image
    - Contains image info and bounding boxes
    - Segmentation masks stored separately as PNG
    """
    
    def __init__(self, output_dir: str):
        """
        Initialize VOC exporter.
        
        Args:
            output_dir: Output directory for VOC files
        """
        self.output_dir = Path(output_dir)
        self.annotations_dir = self.output_dir / "Annotations"
        self.segmentation_dir = self.output_dir / "SegmentationClass"
        
        # Create directories
        ensure_dir(self.annotations_dir)
        ensure_dir(self.segmentation_dir)
        
        logger.info(f"Created VOC exporter: {output_dir}")
    
    def export_annotation(self,
                         image_path: str,
                         masks: List[np.ndarray],
                         class_names: List[str],
                         image_width: int,
                         image_height: int):
        """
        Export annotations for one image to VOC format.
        
        Args:
            image_path: Path to the image file
            masks: List of binary masks
            class_names: List of class names for each mask
            image_width: Image width
            image_height: Image height
        """
        image_name = Path(image_path).stem
        
        # Create XML annotation
        annotation = ET.Element("annotation")
        
        # Add folder
        folder = ET.SubElement(annotation, "folder")
        folder.text = "VOC2012"
        
        # Add filename
        filename = ET.SubElement(annotation, "filename")
        filename.text = Path(image_path).name
        
        # Add source
        source = ET.SubElement(annotation, "source")
        database = ET.SubElement(source, "database")
        database.text = "Industrial Defect Dataset"
        
        # Add size
        size = ET.SubElement(annotation, "size")
        width_elem = ET.SubElement(size, "width")
        width_elem.text = str(image_width)
        height_elem = ET.SubElement(size, "height")
        height_elem.text = str(image_height)
        depth_elem = ET.SubElement(size, "depth")
        depth_elem.text = "3"
        
        # Add segmented flag
        segmented = ET.SubElement(annotation, "segmented")
        segmented.text = "1"
        
        # Add objects
        for mask, class_name in zip(masks, class_names):
            bbox = mask_to_bbox(mask)
            if bbox is None:
                continue
            
            x, y, w, h = bbox
            xmin, ymin = int(x), int(y)
            xmax, ymax = int(x + w), int(y + h)
            
            obj = ET.SubElement(annotation, "object")
            
            name_elem = ET.SubElement(obj, "name")
            name_elem.text = class_name
            
            pose = ET.SubElement(obj, "pose")
            pose.text = "Unspecified"
            
            truncated = ET.SubElement(obj, "truncated")
            truncated.text = "0"
            
            difficult = ET.SubElement(obj, "difficult")
            difficult.text = "0"
            
            bndbox = ET.SubElement(obj, "bndbox")
            xmin_elem = ET.SubElement(bndbox, "xmin")
            xmin_elem.text = str(xmin)
            ymin_elem = ET.SubElement(bndbox, "ymin")
            ymin_elem.text = str(ymin)
            xmax_elem = ET.SubElement(bndbox, "xmax")
            xmax_elem.text = str(xmax)
            ymax_elem = ET.SubElement(bndbox, "ymax")
            ymax_elem.text = str(ymax)
        
        # Write XML file
        tree = ET.ElementTree(annotation)
        xml_path = self.annotations_dir / f"{image_name}.xml"
        tree.write(xml_path, encoding='utf-8', xml_declaration=True)
        
        # Save segmentation mask
        if masks:
            # Combine all masks into one image (each class with different pixel value)
            combined_mask = np.zeros((image_height, image_width), dtype=np.uint8)
            for i, mask in enumerate(masks, start=1):
                combined_mask[mask > 0] = i
            
            from src.utils.mask_utils import save_mask
            mask_path = self.segmentation_dir / f"{image_name}.png"
            save_mask(combined_mask, str(mask_path))
        
        logger.debug(f"Exported VOC annotation: {xml_path}")


def batch_export_coco(image_paths: List[str],
                     mask_paths: List[str],
                     category_names: List[str],
                     output_path: str,
                     dataset_name: str = "Industrial Defect Dataset") -> Dict:
    """
    Batch export annotations to COCO format.
    
    Args:
        image_paths: List of image file paths
        mask_paths: List of mask file paths (same order as images)
        category_names: List of category names for each mask
        output_path: Output JSON file path
        dataset_name: Name of the dataset
        
    Returns:
        Export statistics
    """
    exporter = COCOExporter(dataset_name=dataset_name)
    
    # Add categories
    unique_categories = list(set(category_names))
    for cat_name in unique_categories:
        exporter.add_category(cat_name)
    
    # Process each image
    from src.utils.image_utils import get_image_info
    from src.utils.mask_utils import load_mask
    
    for image_path, mask_path, category_name in zip(image_paths, mask_paths, category_names):
        # Get image info
        info = get_image_info(image_path)
        if info is None:
            logger.warning(f"Failed to get info for: {image_path}")
            continue
        
        width, height = info['width'], info['height']
        
        # Add image
        image_id = exporter.add_image(image_path, width, height)
        
        # Load and add mask
        mask = load_mask(mask_path)
        if mask is None:
            logger.warning(f"Failed to load mask: {mask_path}")
            continue
        
        category_id = exporter.category_dict[category_name]
        exporter.add_annotation(image_id, category_id, mask)
    
    # Save
    exporter.save(output_path)
    
    return exporter.get_statistics()


def batch_export_yolo(image_paths: List[str],
                     mask_paths: List[str],
                     class_ids: List[int],
                     class_names: List[str],
                     output_dir: str) -> int:
    """
    Batch export annotations to YOLO format.
    
    Args:
        image_paths: List of image file paths
        mask_paths: List of mask file paths
        class_ids: List of class IDs for each mask
        class_names: List of all class names
        output_dir: Output directory
        
    Returns:
        Number of exported annotations
    """
    exporter = YOLOExporter(output_dir, class_names)
    
    from src.utils.image_utils import get_image_info
    from src.utils.mask_utils import load_mask
    
    count = 0
    
    for image_path, mask_path, class_id in zip(image_paths, mask_paths, class_ids):
        # Get image info
        info = get_image_info(image_path)
        if info is None:
            continue
        
        width, height = info['width'], info['height']
        
        # Load mask
        mask = load_mask(mask_path)
        if mask is None:
            continue
        
        # Export
        exporter.export_annotation(
            image_path,
            [mask],
            [class_id],
            width,
            height
        )
        count += 1
    
    logger.info(f"Exported {count} YOLO annotations")
    return count
