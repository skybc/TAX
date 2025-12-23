"""
Image canvas widget for displaying and interacting with images.

This widget provides:
- Image display with zoom and pan
- Mouse interaction (click, drag, draw)
- Coordinate display
- Annotation overlay
"""

import numpy as np
from typing import Optional, Tuple, List
from PyQt5.QtCore import Qt, QPoint, QPointF, QRectF, pyqtSignal
from PyQt5.QtGui import (
    QImage, QPixmap, QPainter, QPen, QColor, 
    QWheelEvent, QMouseEvent, QPaintEvent
)
from PyQt5.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QWidget, QVBoxLayout, QLabel
)

from src.logger import get_logger

logger = get_logger(__name__)


class ImageCanvas(QGraphicsView):
    """
    Image canvas widget based on QGraphicsView.
    
    Provides image display with zoom, pan, and mouse interaction capabilities.
    
    Signals:
        image_loaded: Emitted when image is loaded
        mouse_moved: Emitted when mouse moves over image (x, y coordinates)
        mouse_clicked: Emitted when mouse is clicked (x, y, button)
        zoom_changed: Emitted when zoom level changes (zoom_factor)
    """
    
    # Signals
    image_loaded = pyqtSignal()
    mouse_moved = pyqtSignal(int, int)  # x, y in image coordinates
    mouse_clicked = pyqtSignal(int, int, int)  # x, y, button (1=left, 2=right, 4=middle)
    zoom_changed = pyqtSignal(float)
    
    def __init__(self, parent: Optional[QWidget] = None):
        """
        Initialize ImageCanvas.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        
        # Create graphics scene
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        
        # Image item
        self.image_item: Optional[QGraphicsPixmapItem] = None
        self.current_image: Optional[np.ndarray] = None
        self.image_path: Optional[str] = None
        
        # Zoom settings
        self.zoom_factor = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 10.0
        self.zoom_step = 1.2
        
        # Pan settings
        self.is_panning = False
        self.pan_start_pos = QPoint()
        
        # Mouse tracking
        self.setMouseTracking(True)
        
        # View settings
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Background color
        self.setBackgroundBrush(QColor("#2b2b2b"))
        
        logger.info("ImageCanvas initialized")
    
    def load_image(self, image: np.ndarray, image_path: Optional[str] = None):
        """
        Load and display an image.
        
        Args:
            image: Image as numpy array (HxWxC) in RGB format
            image_path: Optional path to the image file
        """
        if image is None or image.size == 0:
            logger.error("Cannot load empty or None image")
            return
        
        try:
            # Store image
            self.current_image = image.copy()
            self.image_path = image_path
            
            # Convert numpy array to QImage
            height, width = image.shape[:2]
            
            if len(image.shape) == 2:  # Grayscale
                qimage = QImage(
                    image.data, width, height, width,
                    QImage.Format_Grayscale8
                )
            elif image.shape[2] == 3:  # RGB
                bytes_per_line = 3 * width
                qimage = QImage(
                    image.data, width, height, bytes_per_line,
                    QImage.Format_RGB888
                )
            elif image.shape[2] == 4:  # RGBA
                bytes_per_line = 4 * width
                qimage = QImage(
                    image.data, width, height, bytes_per_line,
                    QImage.Format_RGBA8888
                )
            else:
                logger.error(f"Unsupported image format with {image.shape[2]} channels")
                return
            
            # Convert to QPixmap
            pixmap = QPixmap.fromImage(qimage)
            
            # Clear scene and add image
            self.scene.clear()
            self.image_item = self.scene.addPixmap(pixmap)
            
            # Reset zoom
            self.reset_zoom()
            
            # Emit signal
            self.image_loaded.emit()
            
            logger.info(f"Loaded image: shape={image.shape}, path={image_path}")
            
        except Exception as e:
            logger.error(f"Error loading image: {e}", exc_info=True)
    
    def clear(self):
        """Clear the canvas."""
        self.scene.clear()
        self.image_item = None
        self.current_image = None
        self.image_path = None
        self.zoom_factor = 1.0
        logger.debug("Canvas cleared")
    
    def get_image(self) -> Optional[np.ndarray]:
        """
        Get the current image.
        
        Returns:
            Current image as numpy array or None
        """
        return self.current_image.copy() if self.current_image is not None else None
    
    def zoom_in(self):
        """Zoom in."""
        self.zoom(self.zoom_step)
    
    def zoom_out(self):
        """Zoom out."""
        self.zoom(1.0 / self.zoom_step)
    
    def zoom(self, factor: float):
        """
        Apply zoom factor.
        
        Args:
            factor: Zoom multiplication factor
        """
        new_zoom = self.zoom_factor * factor
        
        # Clamp zoom level
        if new_zoom < self.min_zoom or new_zoom > self.max_zoom:
            return
        
        self.scale(factor, factor)
        self.zoom_factor = new_zoom
        
        self.zoom_changed.emit(self.zoom_factor)
        logger.debug(f"Zoom changed: {self.zoom_factor:.2f}x")
    
    def reset_zoom(self):
        """Reset zoom to fit image in view."""
        if self.image_item is None:
            return
        
        # Reset transform
        self.resetTransform()
        self.zoom_factor = 1.0
        
        # Fit image in view
        self.fitInView(self.image_item, Qt.KeepAspectRatio)
        
        # Get actual zoom factor
        transform = self.transform()
        self.zoom_factor = transform.m11()
        
        self.zoom_changed.emit(self.zoom_factor)
        logger.debug(f"Zoom reset to fit: {self.zoom_factor:.2f}x")
    
    def wheelEvent(self, event: QWheelEvent):
        """
        Handle mouse wheel event for zooming.
        
        Args:
            event: Wheel event
        """
        if self.image_item is None:
            return
        
        # Get wheel delta
        delta = event.angleDelta().y()
        
        if delta > 0:
            self.zoom_in()
        else:
            self.zoom_out()
    
    def mousePressEvent(self, event: QMouseEvent):
        """
        Handle mouse press event.
        
        Args:
            event: Mouse event
        """
        if self.image_item is None:
            super().mousePressEvent(event)
            return
        
        # Middle button or Ctrl+Left for panning
        if (event.button() == Qt.MiddleButton or 
            (event.button() == Qt.LeftButton and 
             event.modifiers() & Qt.ControlModifier)):
            self.is_panning = True
            self.pan_start_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        
        # Convert to scene coordinates
        scene_pos = self.mapToScene(event.pos())
        
        # Check if click is on image
        if self.image_item.contains(scene_pos):
            # Convert to image coordinates
            image_pos = self.image_item.mapFromScene(scene_pos)
            x, y = int(image_pos.x()), int(image_pos.y())
            
            # Emit signal
            button = event.button()
            self.mouse_clicked.emit(x, y, button)
            logger.debug(f"Mouse clicked: ({x}, {y}), button={button}")
        
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event: QMouseEvent):
        """
        Handle mouse move event.
        
        Args:
            event: Mouse event
        """
        if self.image_item is None:
            super().mouseMoveEvent(event)
            return
        
        # Handle panning
        if self.is_panning:
            delta = event.pos() - self.pan_start_pos
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )
            self.pan_start_pos = event.pos()
            event.accept()
            return
        
        # Convert to scene coordinates
        scene_pos = self.mapToScene(event.pos())
        
        # Check if mouse is on image
        if self.image_item.contains(scene_pos):
            # Convert to image coordinates
            image_pos = self.image_item.mapFromScene(scene_pos)
            x, y = int(image_pos.x()), int(image_pos.y())
            
            # Emit signal
            self.mouse_moved.emit(x, y)
        
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event: QMouseEvent):
        """
        Handle mouse release event.
        
        Args:
            event: Mouse event
        """
        if self.is_panning:
            self.is_panning = False
            self.setCursor(Qt.ArrowCursor)
            event.accept()
            return
        
        super().mouseReleaseEvent(event)
    
    def get_visible_rect(self) -> QRectF:
        """
        Get the visible rectangle in scene coordinates.
        
        Returns:
            Visible rectangle
        """
        return self.mapToScene(self.viewport().rect()).boundingRect()
    
    def get_image_rect(self) -> Optional[QRectF]:
        """
        Get the image rectangle in scene coordinates.
        
        Returns:
            Image rectangle or None if no image loaded
        """
        if self.image_item is None:
            return None
        return self.image_item.boundingRect()
    
    def scene_to_image_coords(self, scene_pos: QPointF) -> Tuple[int, int]:
        """
        Convert scene coordinates to image coordinates.
        
        Args:
            scene_pos: Position in scene coordinates
            
        Returns:
            (x, y) in image coordinates
        """
        if self.image_item is None:
            return (0, 0)
        
        image_pos = self.image_item.mapFromScene(scene_pos)
        return (int(image_pos.x()), int(image_pos.y()))
    
    def image_to_scene_coords(self, x: int, y: int) -> QPointF:
        """
        Convert image coordinates to scene coordinates.
        
        Args:
            x: X coordinate in image
            y: Y coordinate in image
            
        Returns:
            Position in scene coordinates
        """
        if self.image_item is None:
            return QPointF(0, 0)
        
        return self.image_item.mapToScene(QPointF(x, y))


class ImageCanvasWithInfo(QWidget):
    """
    ImageCanvas with additional information display.
    
    Combines ImageCanvas with a label showing image info and mouse coordinates.
    """
    
    def __init__(self, parent: Optional[QWidget] = None):
        """
        Initialize ImageCanvasWithInfo.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        
        # Create layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Create canvas
        self.canvas = ImageCanvas(self)
        layout.addWidget(self.canvas)
        
        # Create info label
        self.info_label = QLabel("No image loaded")
        self.info_label.setStyleSheet(
            "QLabel { background-color: #3c3c3c; color: #ffffff; "
            "padding: 4px; font-family: monospace; }"
        )
        layout.addWidget(self.info_label)
        
        # Connect signals
        self.canvas.image_loaded.connect(self._update_info)
        self.canvas.mouse_moved.connect(self._update_coords)
        self.canvas.zoom_changed.connect(self._update_zoom)
    
    def load_image(self, image: np.ndarray, image_path: Optional[str] = None):
        """
        Load and display an image.
        
        Args:
            image: Image as numpy array
            image_path: Optional path to the image file
        """
        self.canvas.load_image(image, image_path)
    
    def _update_info(self):
        """Update info label with image information."""
        if self.canvas.current_image is None:
            self.info_label.setText("No image loaded")
            return
        
        image = self.canvas.current_image
        h, w = image.shape[:2]
        channels = image.shape[2] if len(image.shape) > 2 else 1
        
        info_text = f"Size: {w}x{h} | Channels: {channels} | Zoom: {self.canvas.zoom_factor:.2f}x"
        
        if self.canvas.image_path:
            info_text = f"{self.canvas.image_path} | " + info_text
        
        self.info_label.setText(info_text)
    
    def _update_coords(self, x: int, y: int):
        """Update info label with mouse coordinates."""
        if self.canvas.current_image is None:
            return
        
        image = self.canvas.current_image
        h, w = image.shape[:2]
        
        # Get pixel value if coordinates are valid
        pixel_info = ""
        if 0 <= x < w and 0 <= y < h:
            if len(image.shape) == 2:  # Grayscale
                value = image[y, x]
                pixel_info = f" | Value: {value}"
            elif image.shape[2] == 3:  # RGB
                r, g, b = image[y, x]
                pixel_info = f" | RGB: ({r}, {g}, {b})"
        
        info_text = f"Position: ({x}, {y}){pixel_info} | Zoom: {self.canvas.zoom_factor:.2f}x"
        
        self.info_label.setText(info_text)
    
    def _update_zoom(self, zoom: float):
        """Update info label with zoom level."""
        self._update_info()
