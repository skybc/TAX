"""
用于处理掩码标注的标注管理器。

此模块提供：
- 掩码创建和编辑
- 撤销/重做功能
- 掩码持久化（保存/加载）
- 导出为各种格式（COCO, YOLO, VOC）
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
    管理标注，包括掩码、边界框和元数据。
    
    提供以下功能：
    - 创建和编辑掩码
    - 撤销/重做操作
    - 保存和加载标注
    - 导出为 COCO/YOLO/VOC 格式
    
    属性:
        image_path: 当前图像路径
        image_shape: 当前图像的形状 (H, W, C)
        current_mask: 当前正在编辑的掩码
        history: 用于撤销/重做的掩码历史列表
        history_index: 历史记录中的当前位置
    """
    
    def __init__(self, max_history: int = 50):
        """
        初始化 AnnotationManager。
        
        参数:
            max_history: 要保留的最大历史状态数
        """
        self.max_history = max_history
        
        # 当前状态
        self.image_path: Optional[str] = None
        self.image_shape: Optional[Tuple[int, int]] = None
        self.current_mask: Optional[np.ndarray] = None
        
        # 用于撤销/重做的历史记录
        self.history: List[np.ndarray] = []
        self.history_index: int = -1
        
        # 标注元数据
        self.metadata: Dict = {
            'image_id': None,
            'category_id': 1,
            'category_name': 'defect',
            'created_at': None,
            'modified_at': None
        }
        
        logger.info("AnnotationManager 已初始化")
    
    def set_image(self, image_path: str, image_shape: Tuple[int, int]):
        """
        设置要标注的当前图像。
        
        参数:
            image_path: 图像文件的路径
            image_shape: 图像形状 (H, W) 或 (H, W, C)
        """
        self.image_path = image_path
        
        if len(image_shape) == 3:
            self.image_shape = image_shape[:2]
        else:
            self.image_shape = image_shape
        
        # 初始化空掩码
        self.current_mask = np.zeros(self.image_shape, dtype=np.uint8)
        
        # 重置历史记录
        self.clear_history()
        self._save_state()
        
        # 更新元数据
        self.metadata['image_id'] = Path(image_path).stem
        self.metadata['created_at'] = datetime.now().isoformat()
        self.metadata['modified_at'] = None
        
        logger.info(f"设置图像: {image_path}, 形状: {image_shape}")
    
    def get_current_mask(self) -> Optional[np.ndarray]:
        """
        获取当前掩码。
        
        返回:
            当前掩码的 numpy 数组或 None
        """
        return self.current_mask.copy() if self.current_mask is not None else None
    
    def set_mask(self, mask: np.ndarray):
        """
        设置当前掩码（替换现有掩码）。
        
        参数:
            mask: numpy 数组格式的掩码 (H, W)
        """
        if mask.shape != self.image_shape:
            logger.error(f"掩码形状 {mask.shape} 与图像形状 {self.image_shape} 不匹配")
            return
        
        self.current_mask = mask.copy()
        self._save_state()
        self._update_modified_time()
        
        logger.debug("掩码已更新")
    
    def update_mask(self, mask: np.ndarray, operation: str = 'replace'):
        """
        使用新掩码更新当前掩码。
        
        参数:
            mask: 要应用的掩码 (H, W)
            operation: 更新操作 ('replace', 'add', 'subtract', 'intersect')
        """
        if mask.shape != self.image_shape:
            logger.error(f"掩码形状 {mask.shape} 与图像形状 {self.image_shape} 不匹配")
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
            logger.error(f"未知操作: {operation}")
            return
        
        self._save_state()
        self._update_modified_time()
        
        logger.debug(f"掩码已通过操作更新: {operation}")
    
    def paint_mask(self, points: List[Tuple[int, int]], brush_size: int, 
                   value: int = 255, operation: str = 'paint'):
        """
        在给定点绘制掩码（画笔工具）。
        
        参数:
            points: (x, y) 坐标列表
            brush_size: 画笔半径（像素）
            value: 要绘制的像素值 (0-255)
            operation: 'paint' 或 'erase'
        """
        if self.current_mask is None:
            logger.error("未初始化掩码")
            return
        
        import cv2
        
        for x, y in points:
            # 确保坐标在范围内
            if 0 <= x < self.image_shape[1] and 0 <= y < self.image_shape[0]:
                if operation == 'paint':
                    cv2.circle(self.current_mask, (x, y), brush_size, value, -1)
                elif operation == 'erase':
                    cv2.circle(self.current_mask, (x, y), brush_size, 0, -1)
        
        # 不要为每个画笔笔触保存状态（状态太多）
        # 状态将在释放画笔时保存
        logger.debug(f"绘制了 {len(points)} 个点，画笔大小={brush_size}")
    
    def paint_polygon(self, points: List[Tuple[int, int]], value: int = 255):
        """
        绘制填充多边形。
        
        参数:
            points: (x, y) 顶点列表
            value: 填充像素值
        """
        if self.current_mask is None:
            logger.error("未初始化掩码")
            return
        
        if len(points) < 3:
            logger.warning("多边形至少需要 3 个点")
            return
        
        import cv2
        
        # 转换为 numpy 数组
        pts = np.array(points, dtype=np.int32)
        
        # 填充多边形
        cv2.fillPoly(self.current_mask, [pts], value)
        
        self._save_state()
        self._update_modified_time()
        
        logger.debug(f"绘制了具有 {len(points)} 个顶点的多边形")
    
    def clear_mask(self):
        """清除当前掩码。"""
        if self.current_mask is not None:
            self.current_mask.fill(0)
            self._save_state()
            self._update_modified_time()
            logger.debug("掩码已清除")
    
    def undo(self) -> bool:
        """
        撤销上次操作。
        
        返回:
            如果撤销成功则为 True，否则为 False
        """
        if self.history_index > 0:
            self.history_index -= 1
            self.current_mask = self.history[self.history_index].copy()
            logger.debug(f"撤销: history_index={self.history_index}")
            return True
        
        logger.debug("无法撤销：处于历史记录的开头")
        return False
    
    def redo(self) -> bool:
        """
        重做上次撤销的操作。
        
        返回:
            如果重做成功则为 True，否则为 False
        """
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            self.current_mask = self.history[self.history_index].copy()
            logger.debug(f"重做: history_index={self.history_index}")
            return True
        
        logger.debug("无法重做：处于历史记录的末尾")
        return False
    
    def can_undo(self) -> bool:
        """检查撤销是否可用。"""
        return self.history_index > 0
    
    def can_redo(self) -> bool:
        """检查重做是否可用。"""
        return self.history_index < len(self.history) - 1
    
    def save_mask(self, output_path: str) -> bool:
        """
        将当前掩码保存到文件。
        
        参数:
            output_path: 保存掩码文件的路径
            
        返回:
            如果成功则为 True，否则为 False
        """
        if self.current_mask is None:
            logger.error("没有要保存的掩码")
            return False
        
        try:
            from src.utils.mask_utils import save_mask
            save_mask(self.current_mask, output_path)
            logger.info(f"掩码已保存到: {output_path}")
            return True
        except Exception as e:
            logger.error(f"保存掩码失败: {e}")
            return False
    
    def load_mask(self, mask_path: str) -> bool:
        """
        从文件加载掩码。
        
        参数:
            mask_path: 掩码文件的路径
            
        返回:
            如果成功则为 True，否则为 False
        """
        try:
            from src.utils.mask_utils import load_mask
            mask = load_mask(mask_path)
            
            if mask is None:
                logger.error(f"加载掩码失败: {mask_path}")
                return False
            
            # 确保掩码与图像形状匹配
            if mask.shape != self.image_shape:
                logger.warning(f"掩码形状 {mask.shape} 与图像形状 {self.image_shape} 不匹配")
                import cv2
                mask = cv2.resize(mask, (self.image_shape[1], self.image_shape[0]), 
                                 interpolation=cv2.INTER_NEAREST)
            
            self.current_mask = mask
            self._save_state()
            logger.info(f"从 {mask_path} 加载了掩码")
            return True
            
        except Exception as e:
            logger.error(f"加载掩码失败: {e}")
            return False
    
    def export_coco_annotation(self) -> Dict:
        """
        将标注导出为 COCO 格式。
        
        返回:
            COCO 标注字典
        """
        if self.current_mask is None:
            return {}
        
        # 从掩码获取边界框
        bbox = mask_to_bbox(self.current_mask)
        
        if bbox is None:
            logger.warning("空掩码，无法导出标注")
            return {}
        
        # 获取 RLE 编码
        rle = binary_mask_to_rle(self.current_mask)
        
        # 计算面积
        area = int(np.sum(self.current_mask > 0))
        
        annotation = {
            'id': hash(self.image_path) % (10 ** 8),  # 生成唯一 ID
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
        将标注导出为 YOLO 格式（多边形格式）。
        
        参数:
            class_id: YOLO 格式的类别 ID
            
        返回:
            YOLO 标注字符串列表
        """
        if self.current_mask is None or self.image_shape is None:
            return []
        
        # 从掩码获取多边形
        polygons = mask_to_polygon(self.current_mask)
        
        if not polygons:
            logger.warning("在掩码中未找到多边形")
            return []
        
        annotations = []
        h, w = self.image_shape
        
        for polygon in polygons:
            if len(polygon) < 6:  # 至少需要 3 个点（x,y 对）
                continue
            
            # 将坐标归一化到 [0, 1]
            normalized_polygon = []
            for i in range(0, len(polygon), 2):
                x = polygon[i] / w
                y = polygon[i + 1] / h
                normalized_polygon.extend([x, y])
            
            # 格式: class_id x1 y1 x2 y2 x3 y3 ...
            annotation = f"{class_id} " + " ".join([f"{coord:.6f}" for coord in normalized_polygon])
            annotations.append(annotation)
        
        return annotations
    
    def clear_history(self):
        """清除撤销/重做历史记录。"""
        self.history.clear()
        self.history_index = -1
        logger.debug("历史记录已清除")
    
    def _save_state(self):
        """将当前掩码状态保存到历史记录。"""
        if self.current_mask is None:
            return
        
        # 移除当前索引之后的任何状态（当撤销后有新操作时）
        if self.history_index < len(self.history) - 1:
            self.history = self.history[:self.history_index + 1]
        
        # 添加当前状态
        self.history.append(self.current_mask.copy())
        self.history_index += 1
        
        # 限制历史记录大小
        if len(self.history) > self.max_history:
            self.history.pop(0)
            self.history_index -= 1
        
        logger.debug(f"状态已保存: history_index={self.history_index}, history_len={len(self.history)}")
    
    def _update_modified_time(self):
        """更新修改时间戳。"""
        self.metadata['modified_at'] = datetime.now().isoformat()
    
    def finish_paint_stroke(self):
        """
        完成一次画笔笔触并保存状态。
        
        在用户绘制后释放鼠标按钮时调用此方法。
        """
        self._save_state()
        self._update_modified_time()
        logger.debug("画笔笔触已完成，状态已保存")
