"""
具有绘图功能的图像标注画布。

扩展了 ImageCanvas，增加了标注功能：
- 画笔和橡皮擦工具
- 多边形绘制
- 掩码叠加显示
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
    具有标注功能的图像画布。
    
    附加信号:
        annotation_changed: 标注被修改时发出
        paint_stroke_finished: 绘图笔触完成时发出
    """
    
    annotation_changed = pyqtSignal()
    paint_stroke_finished = pyqtSignal()
    
    def __init__(self, parent=None):
        """
        初始化 AnnotatableCanvas。
        
        参数:
            parent: 父小部件
        """
        super().__init__(parent)
        
        # 标注掩码
        self.annotation_mask: Optional[np.ndarray] = None
        self.mask_overlay_item: Optional[QGraphicsPixmapItem] = None
        self.mask_opacity = 0.5
        
        # 绘图状态
        self.current_tool = "select"  # 'select', 'brush', 'eraser', 'polygon'
        self.brush_size = 10
        self.is_drawing = False
        self.last_draw_point: Optional[QPoint] = None
        
        # 多边形点
        self.polygon_points: List[Tuple[int, int]] = []
        self.temp_polygon_lines = []
        
        # 画笔预览
        self.brush_preview: Optional[QGraphicsEllipseItem] = None
        
        logger.info("AnnotatableCanvas 已初始化")
    
    def load_image(self, image: np.ndarray, image_path: Optional[str] = None):
        """
        加载并显示图像。
        
        参数:
            image: RGB 格式的 numpy 数组 (HxWxC) 图像
            image_path: 图像文件的可选路径
        """
        super().load_image(image, image_path)
        
        # 初始化标注掩码
        if image is not None:
            h, w = image.shape[:2]
            self.annotation_mask = np.zeros((h, w), dtype=np.uint8)
            self._update_mask_overlay()
    
    def set_annotation_mask(self, mask: np.ndarray):
        """
        设置标注掩码。
        
        参数:
            mask: numpy 数组格式的掩码 (H, W)
        """
        if self.current_image is None:
            logger.error("未加载图像")
            return
        
        h, w = self.current_image.shape[:2]
        
        if mask.shape != (h, w):
            logger.error(f"掩码形状 {mask.shape} 与图像形状 ({h}, {w}) 不匹配")
            return
        
        self.annotation_mask = mask.copy()
        self._update_mask_overlay()
        self.annotation_changed.emit()
    
    def get_annotation_mask(self) -> Optional[np.ndarray]:
        """
        获取当前标注掩码。
        
        返回:
            标注掩码或 None
        """
        return self.annotation_mask.copy() if self.annotation_mask is not None else None
    
    def set_tool(self, tool: str):
        """
        设置当前标注工具。
        
        参数:
            tool: 工具名称 ('select', 'brush', 'eraser', 'polygon')
        """
        self.current_tool = tool
        
        # 如果切换工具，重置多边形
        if tool != 'polygon':
            self._clear_temp_polygon()
        
        # 更新光标
        if tool in ['brush', 'eraser']:
            self.setCursor(Qt.CrossCursor)
        else:
            self.setCursor(Qt.ArrowCursor)
        
        logger.debug(f"工具设置为: {tool}")
    
    def set_brush_size(self, size: int):
        """设置画笔大小。"""
        self.brush_size = size
    
    def set_mask_opacity(self, opacity: float):
        """设置掩码叠加不透明度。"""
        self.mask_opacity = opacity
        self._update_mask_overlay()
    
    def clear_annotation(self):
        """清除所有标注。"""
        if self.annotation_mask is not None:
            self.annotation_mask.fill(0)
            self._update_mask_overlay()
            self.annotation_changed.emit()
    
    def mousePressEvent(self, event):
        """处理鼠标按下事件。"""
        if self.image_item is None or self.annotation_mask is None:
            super().mousePressEvent(event)
            return
        
        scene_pos = self.mapToScene(event.pos())
        
        if not self.image_item.contains(scene_pos):
            super().mousePressEvent(event)
            return
        
        # 获取图像坐标
        image_pos = self.image_item.mapFromScene(scene_pos)
        x, y = int(image_pos.x()), int(image_pos.y())
        
        # 处理特定工具的操作
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
                # 向多边形添加点
                self.polygon_points.append((x, y))
                self._update_temp_polygon()
            elif event.button() == Qt.RightButton:
                # 完成多边形
                if len(self.polygon_points) >= 3:
                    self._complete_polygon()
                else:
                    self._clear_temp_polygon()
        else:
            super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        """处理鼠标移动事件。"""
        if self.image_item is None:
            super().mouseMoveEvent(event)
            return
        
        scene_pos = self.mapToScene(event.pos())
        
        # 更新画笔预览
        if self.current_tool in ['brush', 'eraser']:
            self._update_brush_preview(scene_pos)
        
        if not self.image_item.contains(scene_pos):
            super().mouseMoveEvent(event)
            return
        
        # 获取图像坐标
        image_pos = self.image_item.mapFromScene(scene_pos)
        x, y = int(image_pos.x()), int(image_pos.y())
        
        # 处理绘图
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
        """处理鼠标释放事件。"""
        if self.is_drawing:
            self.is_drawing = False
            self.last_draw_point = None
            self.paint_stroke_finished.emit()
        
        super().mouseReleaseEvent(event)
    
    def _draw_point(self, x: int, y: int, value: int):
        """使用画笔绘制单个点。"""
        if self.annotation_mask is None:
            return
        
        import cv2
        
        h, w = self.annotation_mask.shape
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(self.annotation_mask, (x, y), self.brush_size, value, -1)
            self._update_mask_overlay()
            self.annotation_changed.emit()
    
    def _draw_line(self, x1: int, y1: int, x2: int, y2: int, value: int):
        """使用画笔绘制线条。"""
        if self.annotation_mask is None:
            return
        
        import cv2
        
        # 使用圆形画笔绘制线条
        h, w = self.annotation_mask.shape
        
        # 沿线条插值点
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
        """更新掩码叠加显示。"""
        if self.annotation_mask is None or self.image_item is None:
            return
        
        # 创建彩色掩码叠加层
        h, w = self.annotation_mask.shape
        mask_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        
        # 在掩码非零处设置颜色（红色）
        mask_rgba[self.annotation_mask > 0] = [255, 0, 0, int(self.mask_opacity * 255)]
        
        # 转换为 QPixmap
        qimage = QImage(mask_rgba.data, w, h, w * 4, QImage.Format_RGBA8888)
        pixmap = QPixmap.fromImage(qimage)
        
        # 更新或创建叠加项
        if self.mask_overlay_item is None:
            self.mask_overlay_item = self.scene.addPixmap(pixmap)
            self.mask_overlay_item.setZValue(1)  # 在图像上方
        else:
            self.mask_overlay_item.setPixmap(pixmap)
    
    def _update_brush_preview(self, scene_pos: QPointF):
        """更新画笔预览圆圈。"""
        # 移除旧预览
        if self.brush_preview is not None:
            self.scene.removeItem(self.brush_preview)
            self.brush_preview = None
        
        # 添加新预览
        if self.image_item is not None and self.image_item.contains(scene_pos):
            pen = QPen(QColor(255, 255, 0), 2)  # 黄色边框
            brush = QBrush(QColor(255, 255, 0, 50))  # 半透明黄色
            
            radius = self.brush_size
            self.brush_preview = self.scene.addEllipse(
                scene_pos.x() - radius,
                scene_pos.y() - radius,
                radius * 2,
                radius * 2,
                pen,
                brush
            )
            self.brush_preview.setZValue(2)  # 在所有内容上方
    
    def _update_temp_polygon(self):
        """更新临时多边形线条。"""
        # 清除旧线条
        for line in self.temp_polygon_lines:
            self.scene.removeItem(line)
        self.temp_polygon_lines.clear()
        
        if len(self.polygon_points) < 2:
            return
        
        # 在点之间绘制线条
        pen = QPen(QColor(0, 255, 0), 2)  # 绿色线条
        
        for i in range(len(self.polygon_points) - 1):
            x1, y1 = self.polygon_points[i]
            x2, y2 = self.polygon_points[i + 1]
            
            # 转换为场景坐标
            p1 = self.image_to_scene_coords(x1, y1)
            p2 = self.image_to_scene_coords(x2, y2)
            
            line = self.scene.addLine(p1.x(), p1.y(), p2.x(), p2.y(), pen)
            line.setZValue(2)
            self.temp_polygon_lines.append(line)
    
    def _complete_polygon(self):
        """完成并填充多边形。"""
        if len(self.polygon_points) < 3 or self.annotation_mask is None:
            return
        
        import cv2
        
        # 将点转换为 numpy 数组
        pts = np.array(self.polygon_points, dtype=np.int32)
        
        # 在掩码中填充多边形
        cv2.fillPoly(self.annotation_mask, [pts], 255)
        
        # 更新显示
        self._update_mask_overlay()
        self.annotation_changed.emit()
        self.paint_stroke_finished.emit()
        
        # 清除多边形
        self._clear_temp_polygon()
    
    def _clear_temp_polygon(self):
        """清除临时多边形。"""
        self.polygon_points.clear()
        
        for line in self.temp_polygon_lines:
            self.scene.removeItem(line)
        self.temp_polygon_lines.clear()
