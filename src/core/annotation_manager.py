"""
Annotation manager for handling mask annotations.

This module provides:
- Mask creation and editing
- Undo/redo functionality
- Mask persistence (save/load)
- Export to various formats (COCO, YOLO, VOC)
"""

from pathlib import Path
from typing import Optional, List, Dict, Tuple
import numpy as np
import json
from datetime import datetime

from src.logger import get_logger
from src.utils.mask_utils import (
    save_mask, load_mask, binary_mask_to_rle,
    rle_to_binary_mask, mask_to_bbox, mask_to_polygon
)
from src.utils.file_utils import ensure_dir, save_json, load_json

logger = get_logger(__name__)


class AnnotationManager:
    """
    Manages annotations including masks, bboxes, and metadata.
    
    Provides functionality for:
    - Creating and editing masks
    - Undo/redo operations
    - Saving and loading annotations
    - Exporting to COCO/YOLO/VOC formats
    
    Attributes:
        image_path: Current image path
        image_shape: Shape of current image (H, W, C)
        current_mask: Current mask being edited
        history: List of mask history for undo/redo
        history_index: Current position in history
    """
    
    def __init__(self, max_history: int = 50):
        """
        Initialize AnnotationManager.
        
        Args:
            max_history: Maximum number of history states to keep
        """
        self.max_history = max_history
        
        # Current state
        self.image_path: Optional[str] = None
        self.image_shape: Optional[Tuple[int, int]] = None
        self.current_mask: Optional[np.ndarray] = None
        
        # History for undo/redo
        self.history: List[np.ndarray] = []
        self.history_index: int = -1
        
        # Annotation metadata
        self.metadata: Dict = {
            'image_id': None,
            'category_id': 1,
            'category_name': 'defect',
            'created_at': None,
            'modified_at': None
        }
        
        logger.info("AnnotationManager initialized")
    
    def set_image(self, image_path: str, image_shape: Tuple[int, int]):
        """
        Set the current image to annotate.
        
        Args:
            image_path: Path to the image file
            image_shape: Shape of image (H, W) or (H, W, C)
        """
        self.image_path = image_path
        
        if len(image_shape) == 3:
            self.image_shape = image_shape[:2]
        else:
            self.image_shape = image_shape
        
        # Initialize empty mask
        self.current_mask = np.zeros(self.image_shape, dtype=np.uint8)
        
        # Reset history
        self.clear_history()
        self._save_state()
        
        # Update metadata
        self.metadata['image_id'] = Path(image_path).stem
        self.metadata['created_at'] = datetime.now().isoformat()
        self.metadata['modified_at'] = None
        
        logger.info(f"Set image: {image_path}, shape: {image_shape}")
    
    def get_current_mask(self) -> Optional[np.ndarray]:
        """
        Get the current mask.
        
        Returns:
            Current mask as numpy array or None
        """
        return self.current_mask.copy() if self.current_mask is not None else None
    
    def set_mask(self, mask: np.ndarray):
        """
        Set the current mask (replaces existing mask).
        
        Args:
            mask: Mask as numpy array (H, W)
        """
        if mask.shape != self.image_shape:
            logger.error(f"Mask shape {mask.shape} does not match image shape {self.image_shape}")
            return
        
        self.current_mask = mask.copy()
        self._save_state()
        self._update_modified_time()
        
        logger.debug("Mask updated")
    
    def update_mask(self, mask: np.ndarray, operation: str = 'replace'):
        """
        Update the current mask with a new mask.
        
        Args:
            mask: Mask to apply (H, W)
            operation: Update operation ('replace', 'add', 'subtract', 'intersect')
        """
        if mask.shape != self.image_shape:
            logger.error(f"Mask shape {mask.shape} does not match image shape {self.image_shape}")
            return
        
        if operation == 'replace':
            self.current_mask = mask.copy()
        elif operation == 'add':
            self.current_mask = np.maximum(self.current_mask, mask)
        elif operation == 'subtract':
            self.current_mask = np.where(mask > 0, 0, self.current_mask)
        elif operation == 'intersect':
            self.current_mask = np.minimum(self.current_mask, mask)
        else:
            logger.error(f"Unknown operation: {operation}")
            return
        
        self._save_state()
        self._update_modified_time()
        
        logger.debug(f"Mask updated with operation: {operation}")
    
    def paint_mask(self, points: List[Tuple[int, int]], brush_size: int, 
                   value: int = 255, operation: str = 'paint'):
        """
        Paint mask at given points (brush tool).
        
        Args:
            points: List of (x, y) coordinates
            brush_size: Brush radius in pixels
            value: Pixel value to paint (0-255)
            operation: 'paint' or 'erase'
        """
        if self.current_mask is None:
            logger.error("No mask initialized")
            return
        
        import cv2
        
        for x, y in points:
            # Ensure coordinates are within bounds
            if 0 <= x < self.image_shape[1] and 0 <= y < self.image_shape[0]:
                if operation == 'paint':
                    cv2.circle(self.current_mask, (x, y), brush_size, value, -1)
                elif operation == 'erase':
                    cv2.circle(self.current_mask, (x, y), brush_size, 0, -1)
        
        # Don't save state for every paint stroke (too many states)
        # State will be saved when brush is released
        logger.debug(f"Painted {len(points)} points with brush_size={brush_size}")
    
    def paint_polygon(self, points: List[Tuple[int, int]], value: int = 255):
        """
        Paint a filled polygon.
        
        Args:
            points: List of (x, y) vertices
            value: Pixel value to fill
        """
        if self.current_mask is None:
            logger.error("No mask initialized")
            return
        
        if len(points) < 3:
            logger.warning("Polygon needs at least 3 points")
            return
        
        import cv2
        
        # Convert to numpy array
        pts = np.array(points, dtype=np.int32)
        
        # Fill polygon
        cv2.fillPoly(self.current_mask, [pts], value)
        
        self._save_state()
        self._update_modified_time()
        
        logger.debug(f"Painted polygon with {len(points)} vertices")
    
    def clear_mask(self):
        """Clear the current mask."""
        if self.current_mask is not None:
            self.current_mask.fill(0)
            self._save_state()
            self._update_modified_time()
            logger.debug("Mask cleared")
    
    def undo(self) -> bool:
        """
        Undo last operation.
        
        Returns:
            True if undo was successful, False otherwise
        """
        if self.history_index > 0:
            self.history_index -= 1
            self.current_mask = self.history[self.history_index].copy()
            logger.debug(f"Undo: history_index={self.history_index}")
            return True
        
        logger.debug("Cannot undo: at beginning of history")
        return False
    
    def redo(self) -> bool:
        """
        Redo last undone operation.
        
        Returns:
            True if redo was successful, False otherwise
        """
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            self.current_mask = self.history[self.history_index].copy()
            logger.debug(f"Redo: history_index={self.history_index}")
            return True
        
        logger.debug("Cannot redo: at end of history")
        return False
    
    def can_undo(self) -> bool:
        """Check if undo is available."""
        return self.history_index > 0
    
    def can_redo(self) -> bool:
        """Check if redo is available."""
        return self.history_index < len(self.history) - 1
    
    def save_mask(self, output_path: str) -> bool:
        """
        Save the current mask to file.
        
        Args:
            output_path: Path to save mask file
            
        Returns:
            True if successful, False otherwise
        """
        if self.current_mask is None:
            logger.error("No mask to save")
            return False
        
        try:
            from src.utils.mask_utils import save_mask
            save_mask(self.current_mask, output_path)
            logger.info(f"Saved mask to: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save mask: {e}")
            return False
    
    def load_mask(self, mask_path: str) -> bool:
        """
        Load mask from file.
        
        Args:
            mask_path: Path to mask file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            from src.utils.mask_utils import load_mask
            mask = load_mask(mask_path)
            
            if mask is None:
                logger.error(f"Failed to load mask: {mask_path}")
                return False
            
            # Ensure mask matches image shape
            if mask.shape != self.image_shape:
                logger.warning(f"Mask shape {mask.shape} does not match image shape {self.image_shape}")
                import cv2
                mask = cv2.resize(mask, (self.image_shape[1], self.image_shape[0]), 
                                 interpolation=cv2.INTER_NEAREST)
            
            self.current_mask = mask
            self._save_state()
            logger.info(f"Loaded mask from: {mask_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load mask: {e}")
            return False
    
    def export_coco_annotation(self) -> Dict:
        """
        Export annotation to COCO format.
        
        Returns:
            COCO annotation dictionary
        """
        if self.current_mask is None:
            return {}
        
        # Get bbox from mask
        bbox = mask_to_bbox(self.current_mask)
        
        if bbox is None:
            logger.warning("Empty mask, cannot export annotation")
            return {}
        
        # Get RLE encoding
        rle = binary_mask_to_rle(self.current_mask)
        
        # Calculate area
        area = int(np.sum(self.current_mask > 0))
        
        annotation = {
            'id': hash(self.image_path) % (10 ** 8),  # Generate unique ID
            'image_id': self.metadata['image_id'],
            'category_id': self.metadata['category_id'],
            'bbox': bbox,  # [x, y, width, height]
            'area': area,
            'segmentation': rle,
            'iscrowd': 0
        }
        
        return annotation
    
    def export_yolo_annotation(self, class_id: int = 0) -> List[str]:
        """
        Export annotation to YOLO format (polygon format).
        
        Args:
            class_id: Class ID for YOLO format
            
        Returns:
            List of YOLO annotation strings
        """
        if self.current_mask is None or self.image_shape is None:
            return []
        
        # Get polygons from mask
        polygons = mask_to_polygon(self.current_mask)
        
        if not polygons:
            logger.warning("No polygons found in mask")
            return []
        
        annotations = []
        h, w = self.image_shape
        
        for polygon in polygons:
            if len(polygon) < 6:  # Need at least 3 points (x,y pairs)
                continue
            
            # Normalize coordinates to [0, 1]
            normalized_polygon = []
            for i in range(0, len(polygon), 2):
                x = polygon[i] / w
                y = polygon[i + 1] / h
                normalized_polygon.extend([x, y])
            
            # Format: class_id x1 y1 x2 y2 x3 y3 ...
            annotation = f"{class_id} " + " ".join([f"{coord:.6f}" for coord in normalized_polygon])
            annotations.append(annotation)
        
        return annotations
    
    def clear_history(self):
        """Clear undo/redo history."""
        self.history.clear()
        self.history_index = -1
        logger.debug("History cleared")
    
    def _save_state(self):
        """Save current mask state to history."""
        if self.current_mask is None:
            return
        
        # Remove any states after current index (when new action after undo)
        if self.history_index < len(self.history) - 1:
            self.history = self.history[:self.history_index + 1]
        
        # Add current state
        self.history.append(self.current_mask.copy())
        self.history_index += 1
        
        # Limit history size
        if len(self.history) > self.max_history:
            self.history.pop(0)
            self.history_index -= 1
        
        logger.debug(f"State saved: history_index={self.history_index}, history_len={len(self.history)}")
    
    def _update_modified_time(self):
        """Update the modified timestamp."""
        self.metadata['modified_at'] = datetime.now().isoformat()
    
    def finish_paint_stroke(self):
        """
        Finish a paint stroke and save state.
        
        Call this when user releases mouse button after painting.
        """
        self._save_state()
        self._update_modified_time()
        logger.debug("Paint stroke finished, state saved")
