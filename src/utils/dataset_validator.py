"""
Dataset validator for checking annotation format correctness.

This module provides:
- COCO format validation
- YOLO format validation
- VOC format validation
- Annotation integrity checks
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np

from src.logger import get_logger
from src.utils.file_utils import load_json

logger = get_logger(__name__)


class ValidationResult:
    """Container for validation results."""
    
    def __init__(self):
        self.is_valid = True
        self.errors = []
        self.warnings = []
        self.stats = {}
    
    def add_error(self, message: str):
        """Add an error message."""
        self.errors.append(message)
        self.is_valid = False
        logger.error(f"Validation error: {message}")
    
    def add_warning(self, message: str):
        """Add a warning message."""
        self.warnings.append(message)
        logger.warning(f"Validation warning: {message}")
    
    def add_stat(self, key: str, value):
        """Add a statistic."""
        self.stats[key] = value
    
    def get_report(self) -> str:
        """
        Get validation report as string.
        
        Returns:
            Formatted validation report
        """
        lines = []
        lines.append("=" * 60)
        lines.append("VALIDATION REPORT")
        lines.append("=" * 60)
        lines.append(f"Status: {'✅ PASSED' if self.is_valid else '❌ FAILED'}")
        lines.append("")
        
        if self.stats:
            lines.append("Statistics:")
            for key, value in self.stats.items():
                lines.append(f"  {key}: {value}")
            lines.append("")
        
        if self.errors:
            lines.append(f"Errors ({len(self.errors)}):")
            for error in self.errors:
                lines.append(f"  ❌ {error}")
            lines.append("")
        
        if self.warnings:
            lines.append(f"Warnings ({len(self.warnings)}):")
            for warning in self.warnings:
                lines.append(f"  ⚠️  {warning}")
            lines.append("")
        
        lines.append("=" * 60)
        return "\n".join(lines)


class COCOValidator:
    """
    Validator for COCO format datasets.
    
    Checks:
    - JSON structure validity
    - Required fields presence
    - Image file existence
    - Annotation consistency
    - Segmentation format
    """
    
    def __init__(self, coco_json_path: str, images_dir: Optional[str] = None):
        """
        Initialize COCO validator.
        
        Args:
            coco_json_path: Path to COCO JSON file
            images_dir: Optional directory containing images (for file checks)
        """
        self.coco_json_path = Path(coco_json_path)
        self.images_dir = Path(images_dir) if images_dir else None
        self.result = ValidationResult()
    
    def validate(self) -> ValidationResult:
        """
        Validate COCO dataset.
        
        Returns:
            ValidationResult object
        """
        logger.info(f"Validating COCO dataset: {self.coco_json_path}")
        
        # Check file exists
        if not self.coco_json_path.exists():
            self.result.add_error(f"COCO JSON file not found: {self.coco_json_path}")
            return self.result
        
        # Load JSON
        try:
            coco_data = load_json(self.coco_json_path)
        except Exception as e:
            self.result.add_error(f"Failed to load JSON: {e}")
            return self.result
        
        # Validate structure
        self._validate_structure(coco_data)
        
        if not self.result.is_valid:
            return self.result
        
        # Validate components
        self._validate_images(coco_data.get("images", []))
        self._validate_annotations(coco_data.get("annotations", []))
        self._validate_categories(coco_data.get("categories", []))
        
        # Check consistency
        self._check_consistency(coco_data)
        
        # Add statistics
        self.result.add_stat("num_images", len(coco_data.get("images", [])))
        self.result.add_stat("num_annotations", len(coco_data.get("annotations", [])))
        self.result.add_stat("num_categories", len(coco_data.get("categories", [])))
        
        logger.info("COCO validation complete")
        return self.result
    
    def _validate_structure(self, coco_data: Dict):
        """Validate top-level structure."""
        required_fields = ["images", "annotations", "categories"]
        
        for field in required_fields:
            if field not in coco_data:
                self.result.add_error(f"Missing required field: {field}")
        
        optional_fields = ["info", "licenses"]
        for field in optional_fields:
            if field not in coco_data:
                self.result.add_warning(f"Missing optional field: {field}")
    
    def _validate_images(self, images: List[Dict]):
        """Validate images section."""
        if not images:
            self.result.add_error("No images found in dataset")
            return
        
        image_ids = set()
        
        for idx, image in enumerate(images):
            # Check required fields
            required = ["id", "file_name", "width", "height"]
            for field in required:
                if field not in image:
                    self.result.add_error(f"Image {idx}: missing field '{field}'")
            
            # Check ID uniqueness
            if "id" in image:
                if image["id"] in image_ids:
                    self.result.add_error(f"Duplicate image ID: {image['id']}")
                image_ids.add(image["id"])
            
            # Check dimensions
            if "width" in image and "height" in image:
                if image["width"] <= 0 or image["height"] <= 0:
                    self.result.add_error(f"Image {idx}: invalid dimensions")
            
            # Check file existence
            if self.images_dir and "file_name" in image:
                image_path = self.images_dir / image["file_name"]
                if not image_path.exists():
                    self.result.add_warning(f"Image file not found: {image['file_name']}")
    
    def _validate_annotations(self, annotations: List[Dict]):
        """Validate annotations section."""
        if not annotations:
            self.result.add_warning("No annotations found in dataset")
            return
        
        annotation_ids = set()
        
        for idx, ann in enumerate(annotations):
            # Check required fields
            required = ["id", "image_id", "category_id", "bbox", "area"]
            for field in required:
                if field not in ann:
                    self.result.add_error(f"Annotation {idx}: missing field '{field}'")
            
            # Check ID uniqueness
            if "id" in ann:
                if ann["id"] in annotation_ids:
                    self.result.add_error(f"Duplicate annotation ID: {ann['id']}")
                annotation_ids.add(ann["id"])
            
            # Validate bbox
            if "bbox" in ann:
                bbox = ann["bbox"]
                if not isinstance(bbox, list) or len(bbox) != 4:
                    self.result.add_error(f"Annotation {idx}: invalid bbox format")
                elif any(v < 0 for v in bbox):
                    self.result.add_error(f"Annotation {idx}: negative bbox values")
            
            # Validate area
            if "area" in ann:
                if ann["area"] <= 0:
                    self.result.add_warning(f"Annotation {idx}: zero or negative area")
            
            # Check segmentation (optional)
            if "segmentation" in ann:
                seg = ann["segmentation"]
                # Can be RLE or polygon
                if isinstance(seg, dict):
                    # RLE format
                    if "counts" not in seg or "size" not in seg:
                        self.result.add_error(f"Annotation {idx}: invalid RLE format")
                elif isinstance(seg, list):
                    # Polygon format
                    for poly in seg:
                        if len(poly) < 6:  # At least 3 points
                            self.result.add_warning(f"Annotation {idx}: polygon too small")
    
    def _validate_categories(self, categories: List[Dict]):
        """Validate categories section."""
        if not categories:
            self.result.add_error("No categories found in dataset")
            return
        
        category_ids = set()
        
        for idx, cat in enumerate(categories):
            # Check required fields
            required = ["id", "name"]
            for field in required:
                if field not in cat:
                    self.result.add_error(f"Category {idx}: missing field '{field}'")
            
            # Check ID uniqueness
            if "id" in cat:
                if cat["id"] in category_ids:
                    self.result.add_error(f"Duplicate category ID: {cat['id']}")
                category_ids.add(cat["id"])
    
    def _check_consistency(self, coco_data: Dict):
        """Check consistency between sections."""
        images = coco_data.get("images", [])
        annotations = coco_data.get("annotations", [])
        categories = coco_data.get("categories", [])
        
        # Get valid IDs
        image_ids = {img["id"] for img in images if "id" in img}
        category_ids = {cat["id"] for cat in categories if "id" in cat}
        
        # Check annotation references
        for ann in annotations:
            if "image_id" in ann and ann["image_id"] not in image_ids:
                self.result.add_error(f"Annotation references non-existent image ID: {ann['image_id']}")
            
            if "category_id" in ann and ann["category_id"] not in category_ids:
                self.result.add_error(f"Annotation references non-existent category ID: {ann['category_id']}")


class YOLOValidator:
    """
    Validator for YOLO format datasets.
    
    Checks:
    - Label file format
    - Class ID validity
    - Coordinate validity
    - File structure
    """
    
    def __init__(self, 
                 labels_dir: str,
                 classes_file: str,
                 images_dir: Optional[str] = None):
        """
        Initialize YOLO validator.
        
        Args:
            labels_dir: Directory containing label txt files
            classes_file: Path to classes.txt
            images_dir: Optional directory containing images
        """
        self.labels_dir = Path(labels_dir)
        self.classes_file = Path(classes_file)
        self.images_dir = Path(images_dir) if images_dir else None
        self.result = ValidationResult()
    
    def validate(self) -> ValidationResult:
        """
        Validate YOLO dataset.
        
        Returns:
            ValidationResult object
        """
        logger.info(f"Validating YOLO dataset: {self.labels_dir}")
        
        # Check directories exist
        if not self.labels_dir.exists():
            self.result.add_error(f"Labels directory not found: {self.labels_dir}")
            return self.result
        
        # Check classes file
        if not self.classes_file.exists():
            self.result.add_error(f"Classes file not found: {self.classes_file}")
            return self.result
        
        # Load classes
        with open(self.classes_file, 'r') as f:
            classes = [line.strip() for line in f if line.strip()]
        
        num_classes = len(classes)
        self.result.add_stat("num_classes", num_classes)
        
        # Validate label files
        label_files = list(self.labels_dir.glob("*.txt"))
        self.result.add_stat("num_label_files", len(label_files))
        
        total_annotations = 0
        
        for label_file in label_files:
            count = self._validate_label_file(label_file, num_classes)
            total_annotations += count
            
            # Check corresponding image
            if self.images_dir:
                image_name = label_file.stem
                image_found = False
                for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    image_path = self.images_dir / f"{image_name}{ext}"
                    if image_path.exists():
                        image_found = True
                        break
                
                if not image_found:
                    self.result.add_warning(f"No image found for label: {label_file.name}")
        
        self.result.add_stat("total_annotations", total_annotations)
        
        logger.info("YOLO validation complete")
        return self.result
    
    def _validate_label_file(self, label_file: Path, num_classes: int) -> int:
        """
        Validate a single label file.
        
        Returns:
            Number of annotations in file
        """
        count = 0
        
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            for line_num, line in enumerate(lines, start=1):
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                
                if len(parts) < 5:  # class_id + at least 2 points (4 coords)
                    self.result.add_error(
                        f"{label_file.name}:{line_num} - Too few values (need class_id + coordinates)"
                    )
                    continue
                
                # Check class ID
                try:
                    class_id = int(parts[0])
                    if class_id < 0 or class_id >= num_classes:
                        self.result.add_error(
                            f"{label_file.name}:{line_num} - Invalid class ID: {class_id}"
                        )
                except ValueError:
                    self.result.add_error(
                        f"{label_file.name}:{line_num} - Invalid class ID (not an integer)"
                    )
                    continue
                
                # Check coordinates
                coords = parts[1:]
                if len(coords) % 2 != 0:
                    self.result.add_error(
                        f"{label_file.name}:{line_num} - Odd number of coordinates"
                    )
                    continue
                
                # Validate coordinate values
                for coord in coords:
                    try:
                        val = float(coord)
                        if val < 0 or val > 1:
                            self.result.add_warning(
                                f"{label_file.name}:{line_num} - Coordinate out of range [0,1]: {val}"
                            )
                    except ValueError:
                        self.result.add_error(
                            f"{label_file.name}:{line_num} - Invalid coordinate value: {coord}"
                        )
                
                count += 1
                
        except Exception as e:
            self.result.add_error(f"Error reading {label_file.name}: {e}")
        
        return count


def validate_coco_dataset(coco_json_path: str, 
                         images_dir: Optional[str] = None) -> ValidationResult:
    """
    Validate COCO format dataset.
    
    Args:
        coco_json_path: Path to COCO JSON file
        images_dir: Optional directory containing images
        
    Returns:
        ValidationResult object
    """
    validator = COCOValidator(coco_json_path, images_dir)
    return validator.validate()


def validate_yolo_dataset(labels_dir: str,
                         classes_file: str,
                         images_dir: Optional[str] = None) -> ValidationResult:
    """
    Validate YOLO format dataset.
    
    Args:
        labels_dir: Directory containing label txt files
        classes_file: Path to classes.txt
        images_dir: Optional directory containing images
        
    Returns:
        ValidationResult object
    """
    validator = YOLOValidator(labels_dir, classes_file, images_dir)
    return validator.validate()
