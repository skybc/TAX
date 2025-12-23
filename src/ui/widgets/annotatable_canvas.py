"""
Annotatable canvas for image annotation with drawing capabilities.

Extends ImageCanvas with annotation features:
- Brush and eraser tools
- Polygon drawing
- Mask overlay display
"""

import numpy as np
from typing import Optional, List, Tuple
from PyQt5.QtCore import Qt, QPoint, QPointF, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QBrush
from PyQt5.QtWidgets import QGraphicsPixmapItem, QGraphicsEllipseItem

from src.ui.widgets.image_canvas import ImageCanvas
from src.logger import get_logger

logger = get_logger(__name__)


class AnnotatableCanvas(ImageCanvas):
    """
    Image canvas with annotation capabilities.
    
    Additional Signals:
        annotation_changed: Emitted when annotation is modified
        paint_stroke_finished: Emitted when a paint stroke is completed
    """
    
    annotation_changed = pyqtSignal()
    paint_stroke_finished = pyqtSignal()
    
    def __init__(self, parent=None):
        """
        Initialize AnnotatableCanvas.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        
        # Annotation mask
        self.annotation_mask: Optional[np.ndarray] = None
        self.mask_overlay_item: Optional[QGraphicsPixmapItem] = None
        self.mask_opacity = 0.5
        
        # Drawing state
        self.current_tool = "select"  # 'select', 'brush', 'eraser', 'polygon'
        self.brush_size = 10
        self.is_drawing = False
        self.last_draw_point: Optional[QPoint] = None
        
        # Polygon points
        self.polygon_points: List[Tuple[int, int]] = []
        self.temp_polygon_lines = []
        
        # Brush preview
        self.brush_preview: Optional[QGraphicsEllipseItem] = None
        
        logger.info("AnnotatableCanvas initialized")
    
    def load_image(self, image: np.ndarray, image_path: Optional[str] = None):
        """
        Load and display an image.
        
        Args:
            image: Image as numpy array (HxWxC) in RGB format
            image_path: Optional path to the image file
        """
        super().load_image(image, image_path)
        
        # Initialize annotation mask
        if image is not None:
            h, w = image.shape[:2]
            self.annotation_mask = np.zeros((h, w), dtype=np.uint8)
            self._update_mask_overlay()
    
    def set_annotation_mask(self, mask: np.ndarray):
        """
        Set the annotation mask.
        
        Args:
            mask: Mask as numpy array (H, W)
        """
        if self.current_image is None:
            logger.error("No image loaded")
            return
        
        h, w = self.current_image.shape[:2]
        
        if mask.shape != (h, w):
            logger.error(f"Mask shape {mask.shape} does not match image shape ({h}, {w})")
            return
        
        self.annotation_mask = mask.copy()
        self._update_mask_overlay()
        self.annotation_changed.emit()
    
    def get_annotation_mask(self) -> Optional[np.ndarray]:
        """
        Get the current annotation mask.
        
        Returns:
            Annotation mask or None
        """
        return self.annotation_mask.copy() if self.annotation_mask is not None else None
    
    def set_tool(self, tool: str):
        """
        Set the current annotation tool.
        
        Args:
            tool: Tool name ('select', 'brush', 'eraser', 'polygon')
        """
        self.current_tool = tool
        
        # Reset polygon if switching tools
        if tool != 'polygon':
            self._clear_temp_polygon()
        
        # Update cursor
        if tool in ['brush', 'eraser']:
            self.setCursor(Qt.CrossCursor)
        else:
            self.setCursor(Qt.ArrowCursor)
        
        logger.debug(f"Tool set to: {tool}")
    
    def set_brush_size(self, size: int):
        """Set brush size."""
        self.brush_size = size
    
    def set_mask_opacity(self, opacity: float):
        """Set mask overlay opacity."""
        self.mask_opacity = opacity
        self._update_mask_overlay()
    
    def clear_annotation(self):
        """Clear all annotations."""
        if self.annotation_mask is not None:
            self.annotation_mask.fill(0)
            self._update_mask_overlay()
            self.annotation_changed.emit()
    
    def mousePressEvent(self, event):
        """Handle mouse press event."""
        if self.image_item is None or self.annotation_mask is None:
            super().mousePressEvent(event)
            return
        
        scene_pos = self.mapToScene(event.pos())
        
        if not self.image_item.contains(scene_pos):
            super().mousePressEvent(event)
            return
        
        # Get image coordinates
        image_pos = self.image_item.mapFromScene(scene_pos)
        x, y = int(image_pos.x()), int(image_pos.y())
        
        # Handle tool-specific actions
        if self.current_tool == 'brush' and event.button() == Qt.LeftButton:
            self.is_drawing = True
            self.last_draw_point = QPoint(x, y)
            self._draw_point(x, y, value=255)
            
        elif self.current_tool == 'eraser' and event.button() == Qt.LeftButton:
            self.is_drawing = True
            self.last_draw_point = QPoint(x, y)
            self._draw_point(x, y, value=0)
            
        elif self.current_tool == 'polygon':
            if event.button() == Qt.LeftButton:
                # Add point to polygon
                self.polygon_points.append((x, y))
                self._update_temp_polygon()
            elif event.button() == Qt.RightButton:
                # Complete polygon
                if len(self.polygon_points) >= 3:
                    self._complete_polygon()
                else:
                    self._clear_temp_polygon()
        else:
            super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        """Handle mouse move event."""
        if self.image_item is None:
            super().mouseMoveEvent(event)
            return
        
        scene_pos = self.mapToScene(event.pos())
        
        # Update brush preview
        if self.current_tool in ['brush', 'eraser']:
            self._update_brush_preview(scene_pos)
        
        if not self.image_item.contains(scene_pos):
            super().mouseMoveEvent(event)
            return
        
        # Get image coordinates
        image_pos = self.image_item.mapFromScene(scene_pos)
        x, y = int(image_pos.x()), int(image_pos.y())
        
        # Handle drawing
        if self.is_drawing and self.last_draw_point is not None:
            if self.current_tool == 'brush':
                self._draw_line(self.last_draw_point.x(), self.last_draw_point.y(), 
                              x, y, value=255)
            elif self.current_tool == 'eraser':
                self._draw_line(self.last_draw_point.x(), self.last_draw_point.y(),
                              x, y, value=0)
            self.last_draw_point = QPoint(x, y)
        
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release event."""
        if self.is_drawing:
            self.is_drawing = False
            self.last_draw_point = None
            self.paint_stroke_finished.emit()
        
        super().mouseReleaseEvent(event)
    
    def _draw_point(self, x: int, y: int, value: int):
        """Draw a single point with brush."""
        if self.annotation_mask is None:
            return
        
        import cv2
        
        h, w = self.annotation_mask.shape
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(self.annotation_mask, (x, y), self.brush_size, value, -1)
            self._update_mask_overlay()
            self.annotation_changed.emit()
    
    def _draw_line(self, x1: int, y1: int, x2: int, y2: int, value: int):
        """Draw a line with brush."""
        if self.annotation_mask is None:
            return
        
        import cv2
        
        # Draw line with circular brush
        h, w = self.annotation_mask.shape
        
        # Interpolate points along line
        num_points = max(abs(x2 - x1), abs(y2 - y1)) + 1
        
        for i in range(num_points):
            t = i / max(num_points - 1, 1)
            x = int(x1 + t * (x2 - x1))
            y = int(y1 + t * (y2 - y1))
            
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(self.annotation_mask, (x, y), self.brush_size, value, -1)
        
        self._update_mask_overlay()
        self.annotation_changed.emit()
    
    def _update_mask_overlay(self):
        """Update the mask overlay display."""
        if self.annotation_mask is None or self.image_item is None:
            return
        
        # Create colored mask overlay
        h, w = self.annotation_mask.shape
        mask_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        
        # Set color (red) where mask is non-zero
        mask_rgba[self.annotation_mask > 0] = [255, 0, 0, int(self.mask_opacity * 255)]
        
        # Convert to QPixmap
        qimage = QImage(mask_rgba.data, w, h, w * 4, QImage.Format_RGBA8888)
        pixmap = QPixmap.fromImage(qimage)
        
        # Update or create overlay item
        if self.mask_overlay_item is None:
            self.mask_overlay_item = self.scene.addPixmap(pixmap)
            self.mask_overlay_item.setZValue(1)  # Above image
        else:
            self.mask_overlay_item.setPixmap(pixmap)
    
    def _update_brush_preview(self, scene_pos: QPointF):
        """Update brush preview circle."""
        # Remove old preview
        if self.brush_preview is not None:
            self.scene.removeItem(self.brush_preview)
            self.brush_preview = None
        
        # Add new preview
        if self.image_item is not None and self.image_item.contains(scene_pos):
            pen = QPen(QColor(255, 255, 0), 2)  # Yellow border
            brush = QBrush(QColor(255, 255, 0, 50))  # Semi-transparent yellow
            
            radius = self.brush_size
            self.brush_preview = self.scene.addEllipse(
                scene_pos.x() - radius,
                scene_pos.y() - radius,
                radius * 2,
                radius * 2,
                pen,
                brush
            )
            self.brush_preview.setZValue(2)  # Above everything
    
    def _update_temp_polygon(self):
        """Update temporary polygon lines."""
        # Clear old lines
        for line in self.temp_polygon_lines:
            self.scene.removeItem(line)
        self.temp_polygon_lines.clear()
        
        if len(self.polygon_points) < 2:
            return
        
        # Draw lines between points
        pen = QPen(QColor(0, 255, 0), 2)  # Green lines
        
        for i in range(len(self.polygon_points) - 1):
            x1, y1 = self.polygon_points[i]
            x2, y2 = self.polygon_points[i + 1]
            
            # Convert to scene coordinates
            p1 = self.image_to_scene_coords(x1, y1)
            p2 = self.image_to_scene_coords(x2, y2)
            
            line = self.scene.addLine(p1.x(), p1.y(), p2.x(), p2.y(), pen)
            line.setZValue(2)
            self.temp_polygon_lines.append(line)
    
    def _complete_polygon(self):
        """Complete and fill the polygon."""
        if len(self.polygon_points) < 3 or self.annotation_mask is None:
            return
        
        import cv2
        
        # Convert points to numpy array
        pts = np.array(self.polygon_points, dtype=np.int32)
        
        # Fill polygon in mask
        cv2.fillPoly(self.annotation_mask, [pts], 255)
        
        # Update display
        self._update_mask_overlay()
        self.annotation_changed.emit()
        self.paint_stroke_finished.emit()
        
        # Clear polygon
        self._clear_temp_polygon()
    
    def _clear_temp_polygon(self):
        """Clear temporary polygon."""
        self.polygon_points.clear()
        
        for line in self.temp_polygon_lines:
            self.scene.removeItem(line)
        self.temp_polygon_lines.clear()
