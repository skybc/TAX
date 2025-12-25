"""
用于显示和与图像交互的图像画布小部件。

此小部件提供：
- 带有缩放和平移的图像显示
- 鼠标交互（点击、拖动、绘制）
- 坐标显示
- 标注叠加
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
    基于 QGraphicsView 的图像画布小部件。
    
    提供带有缩放、平移和鼠标交互功能的图像显示。
    
    信号:
        image_loaded: 图像加载时发出
        mouse_moved: 鼠标在图像上移动时发出（x, y 坐标）
        mouse_clicked: 鼠标点击时发出（x, y, 按钮）
        zoom_changed: 缩放级别更改时发出（zoom_factor）
    """
    
    # 信号
    image_loaded = pyqtSignal()
    mouse_moved = pyqtSignal(int, int)  # 图像坐标中的 x, y
    mouse_clicked = pyqtSignal(int, int, int)  # x, y, 按钮 (1=左键, 2=右键, 4=中键)
    zoom_changed = pyqtSignal(float)
    
    def __init__(self, parent: Optional[QWidget] = None):
        """
        初始化 ImageCanvas。
        
        参数:
            parent: 父小部件
        """
        super().__init__(parent)
        
        # 创建图形场景
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        
        # 图像项
        self.image_item: Optional[QGraphicsPixmapItem] = None
        self.current_image: Optional[np.ndarray] = None
        self.image_path: Optional[str] = None
        
        # 缩放设置
        self.zoom_factor = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 10.0
        self.zoom_step = 1.2
        
        # 平移设置
        self.is_panning = False
        self.pan_start_pos = QPoint()
        
        # 鼠标追踪
        self.setMouseTracking(True)
        
        # 视图设置
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # 背景颜色
        self.setBackgroundBrush(QColor("#2b2b2b"))
        
        logger.info("ImageCanvas 已初始化")
    
    def load_image(self, image: np.ndarray, image_path: Optional[str] = None):
        """
        加载并显示图像。
        
        参数:
            image: RGB 格式的 numpy 数组 (HxWxC) 图像
            image_path: 图像文件的可选路径
        """
        if image is None or image.size == 0:
            logger.error("无法加载空或 None 图像")
            return
        
        try:
            # 存储图像（确保内存连续）
            self.current_image = np.ascontiguousarray(image).copy()
            self.image_path = image_path
            
            # 将 numpy 数组转换为 QImage
            height, width = self.current_image.shape[:2]
            
            if len(self.current_image.shape) == 2:  # 灰度图
                bytes_per_line = width
                qimage = QImage(
                    self.current_image.tobytes(), width, height, bytes_per_line,
                    QImage.Format_Grayscale8
                )
            elif self.current_image.shape[2] == 3:  # RGB
                bytes_per_line = 3 * width
                qimage = QImage(
                    self.current_image.tobytes(), width, height, bytes_per_line,
                    QImage.Format_RGB888
                )
            elif self.current_image.shape[2] == 4:  # RGBA
                bytes_per_line = 4 * width
                qimage = QImage(
                    self.current_image.tobytes(), width, height, bytes_per_line,
                    QImage.Format_RGBA8888
                )
            else:
                logger.error(f"不支持具有 {self.current_image.shape[2]} 个通道的图像格式")
                return
            
            # 转换为 QPixmap
            pixmap = QPixmap.fromImage(qimage)
            
            if pixmap.isNull():
                logger.error("转换图像为 QPixmap 失败")
                return
            
            # 清除场景并添加图像
            self.scene.clear()
            self.image_item = self.scene.addPixmap(pixmap)
            
            # 重置缩放
            self.reset_zoom()
            
            # 发出信号
            self.image_loaded.emit()
            
            logger.info(f"已加载图像: 形状={self.current_image.shape}, 路径={image_path}")
            
        except Exception as e:
            logger.error(f"加载图像时出错: {e}", exc_info=True)
    
    def clear(self):
        """清除画布。"""
        self.scene.clear()
        self.image_item = None
        self.current_image = None
        self.image_path = None
        self.zoom_factor = 1.0
        logger.debug("画布已清除")
    
    def get_image(self) -> Optional[np.ndarray]:
        """
        获取当前图像。
        
        返回:
            当前图像的 numpy 数组或 None
        """
        return self.current_image.copy() if self.current_image is not None else None
    
    def zoom_in(self):
        """放大。"""
        self.zoom(self.zoom_step)
    
    def zoom_out(self):
        """缩小。"""
        self.zoom(1.0 / self.zoom_step)
    
    def zoom(self, factor: float):
        """
        应用缩放因子。
        
        参数:
            factor: 缩放倍数
        """
        new_zoom = self.zoom_factor * factor
        
        # 限制缩放级别
        if new_zoom < self.min_zoom or new_zoom > self.max_zoom:
            return
        
        self.scale(factor, factor)
        self.zoom_factor = new_zoom
        
        self.zoom_changed.emit(self.zoom_factor)
        logger.debug(f"缩放已更改: {self.zoom_factor:.2f}x")
    
    def reset_zoom(self):
        """重置缩放以使图像适应视图。"""
        if self.image_item is None:
            return
        
        # 重置变换
        self.resetTransform()
        self.zoom_factor = 1.0
        
        # 使图像适应视图
        self.fitInView(self.image_item, Qt.KeepAspectRatio)
        
        # 获取实际缩放因子
        transform = self.transform()
        self.zoom_factor = transform.m11()
        
        self.zoom_changed.emit(self.zoom_factor)
        logger.debug(f"缩放已重置为适应视图: {self.zoom_factor:.2f}x")
    
    def wheelEvent(self, event: QWheelEvent):
        """
        处理用于缩放的鼠标滚轮事件。
        
        参数:
            event: 滚轮事件
        """
        if self.image_item is None:
            return
        
        # 获取滚轮增量
        delta = event.angleDelta().y()
        
        if delta > 0:
            self.zoom_in()
        else:
            self.zoom_out()
    
    def mousePressEvent(self, event: QMouseEvent):
        """
        处理鼠标按下事件。
        
        参数:
            event: 鼠标事件
        """
        if self.image_item is None:
            super().mousePressEvent(event)
            return
        
        # 中键或 Ctrl+左键用于平移
        if (event.button() == Qt.MiddleButton or 
            (event.button() == Qt.LeftButton and 
             event.modifiers() & Qt.ControlModifier)):
            self.is_panning = True
            self.pan_start_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        
        # 转换为场景坐标
        scene_pos = self.mapToScene(event.pos())
        
        # 检查点击是否在图像上
        if self.image_item.contains(scene_pos):
            # 转换为图像坐标
            image_pos = self.image_item.mapFromScene(scene_pos)
            x, y = int(image_pos.x()), int(image_pos.y())
            
            # 发出信号
            button = event.button()
            self.mouse_clicked.emit(x, y, button)
            logger.debug(f"鼠标已点击: ({x}, {y}), 按钮={button}")
        
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event: QMouseEvent):
        """
        处理鼠标移动事件。
        
        参数:
            event: 鼠标事件
        """
        if self.image_item is None:
            super().mouseMoveEvent(event)
            return
        
        # 处理平移
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
        
        # 转换为场景坐标
        scene_pos = self.mapToScene(event.pos())
        
        # 检查鼠标是否在图像上
        if self.image_item.contains(scene_pos):
            # 转换为图像坐标
            image_pos = self.image_item.mapFromScene(scene_pos)
            x, y = int(image_pos.x()), int(image_pos.y())
            
            # 发出信号
            self.mouse_moved.emit(x, y)
        
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event: QMouseEvent):
        """
        处理鼠标释放事件。
        
        参数:
            event: 鼠标事件
        """
        if self.is_panning:
            self.is_panning = False
            self.setCursor(Qt.ArrowCursor)
            event.accept()
            return
        
        super().mouseReleaseEvent(event)
    
    def get_visible_rect(self) -> QRectF:
        """
        获取场景坐标中的可见矩形。
        
        返回:
            可见矩形
        """
        return self.mapToScene(self.viewport().rect()).boundingRect()
    
    def get_image_rect(self) -> Optional[QRectF]:
        """
        获取场景坐标中的图像矩形。
        
        返回:
            图像矩形，如果未加载图像则返回 None
        """
        if self.image_item is None:
            return None
        return self.image_item.boundingRect()
    
    def scene_to_image_coords(self, scene_pos: QPointF) -> Tuple[int, int]:
        """
        将场景坐标转换为图像坐标。
        
        参数:
            scene_pos: 场景坐标中的位置
            
        返回:
            图像坐标中的 (x, y)
        """
        if self.image_item is None:
            return (0, 0)
        
        image_pos = self.image_item.mapFromScene(scene_pos)
        return (int(image_pos.x()), int(image_pos.y()))
    
    def image_to_scene_coords(self, x: int, y: int) -> QPointF:
        """
        将图像坐标转换为场景坐标。
        
        参数:
            x: 图像中的 X 坐标
            y: 图像中的 Y 坐标
            
        返回:
            场景坐标中的位置
        """
        if self.image_item is None:
            return QPointF(0, 0)
        
        return self.image_item.mapToScene(QPointF(x, y))


class ImageCanvasWithInfo(QWidget):
    """
    带有附加信息显示的 ImageCanvas。
    
    将 ImageCanvas 与显示图像信息和鼠标坐标的标签相结合。
    """
    
    def __init__(self, parent: Optional[QWidget] = None):
        """
        初始化 ImageCanvasWithInfo。
        
        参数:
            parent: 父小部件
        """
        super().__init__(parent)
        
        # 创建布局
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # 创建画布
        self.canvas = ImageCanvas(self)
        layout.addWidget(self.canvas)
        
        # 创建信息标签
        self.info_label = QLabel("未加载图像")
        self.info_label.setStyleSheet(
            "QLabel { background-color: #3c3c3c; color: #ffffff; "
            "padding: 4px; font-family: monospace; }"
        )
        layout.addWidget(self.info_label)
        
        # 连接信号
        self.canvas.image_loaded.connect(self._update_info)
        self.canvas.mouse_moved.connect(self._update_coords)
        self.canvas.zoom_changed.connect(self._update_zoom)
    
    def load_image(self, image: np.ndarray, image_path: Optional[str] = None):
        """
        加载并显示图像。
        
        参数:
            image: numpy 数组格式的图像
            image_path: 图像文件的可选路径
        """
        self.canvas.load_image(image, image_path)
    
    def _update_info(self):
        """使用图像信息更新信息标签。"""
        if self.canvas.current_image is None:
            self.info_label.setText("未加载图像")
            return
        
        image = self.canvas.current_image
        h, w = image.shape[:2]
        channels = image.shape[2] if len(image.shape) > 2 else 1
        
        info_text = f"尺寸: {w}x{h} | 通道: {channels} | 缩放: {self.canvas.zoom_factor:.2f}x"
        
        if self.canvas.image_path:
            info_text = f"{self.canvas.image_path} | " + info_text
        
        self.info_label.setText(info_text)
    
    def _update_coords(self, x: int, y: int):
        """使用鼠标坐标更新信息标签。"""
        if self.canvas.current_image is None:
            return
        
        image = self.canvas.current_image
        h, w = image.shape[:2]
        
        # 如果坐标有效，则获取像素值
        pixel_info = ""
        if 0 <= x < w and 0 <= y < h:
            if len(image.shape) == 2:  # 灰度图
                value = image[y, x]
                pixel_info = f" | 值: {value}"
            elif image.shape[2] == 3:  # RGB
                r, g, b = image[y, x]
                pixel_info = f" | RGB: ({r}, {g}, {b})"
        
        info_text = f"位置: ({x}, {y}){pixel_info} | 缩放: {self.canvas.zoom_factor:.2f}x"
        
        self.info_label.setText(info_text)
    
    def _update_zoom(self, zoom: float):
        """使用缩放级别更新信息标签。"""
        self._update_info()
