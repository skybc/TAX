"""
掩码处理实用函数。

此模块提供掩码操作、编码、解码以及不同格式之间转换的函数。
"""

from pathlib import Path
from typing import List, Tuple, Union, Optional

import cv2
import numpy as np
from pycocotools import mask as coco_mask

from src.logger import get_logger

logger = get_logger(__name__)


def load_mask(
    mask_path: Union[str, Path],
    mode: str = 'GRAY'
) -> Optional[np.ndarray]:
    """
    从文件加载掩码。
    
    参数:
        mask_path: 掩码文件路径
        mode: 颜色模式 ('GRAY', 'BGR', 'RGB')
        
    返回:
        作为 numpy 数组的掩码，灰度为 (H, W)，彩色为 (H, W, C)，
        如果加载失败则返回 None
    """
    mask_path = Path(mask_path)
    
    if not mask_path.exists():
        logger.error(f"未找到掩码文件: {mask_path}")
        return None
    
    try:
        if mode == 'GRAY':
            # 以灰度模式加载
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        elif mode == 'BGR':
            # 以彩色模式加载
            mask = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
        elif mode == 'RGB':
            # 以彩色模式加载并转换为 RGB
            mask = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
            if mask is not None:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        else:
            logger.error(f"不支持的模式: {mode}")
            return None
        
        if mask is None:
            logger.error(f"加载掩码失败: {mask_path}")
            return None
        
        logger.debug(f"已加载掩码 {mask_path}，形状为 {mask.shape}")
        return mask
    
    except Exception as e:
        logger.error(f"加载掩码 {mask_path} 时出错: {e}")
        return None


def save_mask(
    mask: np.ndarray,
    output_path: Union[str, Path]
) -> bool:
    """
    将掩码保存到文件。
    
    参数:
        mask: 作为 numpy 数组的掩码
        output_path: 保存掩码文件的路径
        
    返回:
        如果成功则为 True，否则为 False
    """
    output_path = Path(output_path)
    
    try:
        # 如果父目录不存在则创建
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 确保掩码为 uint8
        if mask.dtype != np.uint8:
            # 如果需要，归一化到 [0, 255]
            if mask.max() <= 1:
                mask = (mask * 255).astype(np.uint8)
            else:
                mask = mask.astype(np.uint8)
        
        # 使用 OpenCV 保存
        success = cv2.imwrite(str(output_path), mask)
        
        if success:
            logger.debug(f"已将掩码保存到 {output_path}")
            return True
        else:
            logger.error(f"保存掩码到 {output_path} 失败")
            return False
    
    except Exception as e:
        logger.error(f"保存掩码 {output_path} 时出错: {e}")
        return False


def binary_mask_to_rle(mask: np.ndarray) -> dict:
    """
    将二值掩码转换为 RLE (行程编码) 格式。
    
    参数:
        mask: 作为 numpy 数组的二值掩码 (H, W)
        
    返回:
        作为字典的 RLE 编码掩码
    """
    # 确保掩码为 Fortran 顺序（列优先）
    mask = np.asfortranarray(mask.astype(np.uint8))
    rle = coco_mask.encode(mask)
    
    # 将字节转换为字符串以便进行 JSON 序列化
    rle['counts'] = rle['counts'].decode('utf-8')
    
    return rle


def rle_to_binary_mask(rle: dict) -> np.ndarray:
    """
    将 RLE 编码掩码转换为二值掩码。
    
    参数:
        rle: RLE 编码掩码
        
    返回:
        作为 numpy 数组的二值掩码 (H, W)
    """
    # 如果需要，将字符串转换回字节
    if isinstance(rle['counts'], str):
        rle['counts'] = rle['counts'].encode('utf-8')
    
    mask = coco_mask.decode(rle)
    return mask


def polygon_to_mask(
    polygon: List[List[float]],
    image_shape: Tuple[int, int]
) -> np.ndarray:
    """
    将多边形转换为二值掩码。
    
    参数:
        polygon: 多边形列表，每个多边形为 [x, y] 坐标列表
        image_shape: 目标掩码形状，格式为 (高度, 宽度)
        
    返回:
        二值掩码
    """
    height, width = image_shape
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # 将多边形转换为 numpy 数组
    if isinstance(polygon[0], list):
        # 多个多边形
        for poly in polygon:
            pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], 1)
    else:
        # 单个多边形
        pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 1)
    
    return mask


def mask_to_polygon(mask: np.ndarray) -> List[List[float]]:
    """
    将二值掩码转换为多边形轮廓。
    
    参数:
        mask: 二值掩码 (H, W)
        
    返回:
        多边形列表，每个多边形为 [x, y] 坐标列表
    """
    # 查找轮廓
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    polygons = []
    for contour in contours:
        # 简化轮廓
        epsilon = 0.001 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # 转换为列表格式
        polygon = approx.reshape(-1, 2).tolist()
        if len(polygon) >= 3:  # 有效多边形至少需要 3 个点
            polygons.append(polygon)
    
    return polygons


def mask_to_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """
    从二值掩码获取边界框。
    
    参数:
        mask: 二值掩码 (H, W)
        
    返回:
        边界框，格式为 (x1, y1, x2, y2)
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    if not rows.any() or not cols.any():
        return (0, 0, 0, 0)
    
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    
    return (int(x1), int(y1), int(x2 + 1), int(y2 + 1))


def bbox_to_mask(
    bbox: Tuple[int, int, int, int],
    image_shape: Tuple[int, int]
) -> np.ndarray:
    """
    从边界框创建二值掩码。
    
    参数:
        bbox: 边界框，格式为 (x1, y1, x2, y2)
        image_shape: 目标掩码形状，格式为 (高度, 宽度)
        
    返回:
        二值掩码
    """
    height, width = image_shape
    mask = np.zeros((height, width), dtype=np.uint8)
    
    x1, y1, x2, y2 = bbox
    mask[y1:y2, x1:x2] = 1
    
    return mask


def dilate_mask(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    膨胀二值掩码。
    
    参数:
        mask: 二值掩码
        kernel_size: 膨胀核的大小
        
    返回:
        膨胀后的掩码
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
    return dilated


def erode_mask(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    腐蚀二值掩码。
    
    参数:
        mask: 二值掩码
        kernel_size: 腐蚀核的大小
        
    返回:
        腐蚀后的掩码
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
    return eroded


def open_mask(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    形态学开运算（先腐蚀后膨胀）。
    
    参数:
        mask: 二值掩码
        kernel_size: 核的大小
        
    返回:
        开运算后的掩码
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    return opened


def close_mask(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    形态学闭运算（先膨胀后腐蚀）。
    
    参数:
        mask: 二值掩码
        kernel_size: 核的大小
        
    返回:
        闭运算后的掩码
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    return closed


def remove_small_components(
    mask: np.ndarray,
    min_area: int = 100
) -> np.ndarray:
    """
    从掩码中移除小的连通分量。
    
    参数:
        mask: 二值掩码
        min_area: 要保留的分量的最小面积
        
    返回:
        过滤后的掩码
    """
    # 查找连通分量
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    
    # 创建输出掩码
    output_mask = np.zeros_like(mask)
    
    # 保留大于 min_area 的分量
    for i in range(1, num_labels):  # 跳过背景（标签 0）
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            output_mask[labels == i] = 1
    
    return output_mask


def get_largest_component(mask: np.ndarray) -> np.ndarray:
    """
    仅保留最大的连通分量。
    
    参数:
        mask: 二值掩码
        
    返回:
        仅包含最大分量的掩码
    """
    # 查找连通分量
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    
    if num_labels <= 1:
        return mask
    
    # 查找最大分量（不包括背景）
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = np.argmax(areas) + 1
    
    # 创建输出掩码
    output_mask = (labels == largest_label).astype(np.uint8)
    
    return output_mask


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """
    填充二值掩码中的孔洞。
    
    参数:
        mask: 二值掩码
        
    返回:
        填充孔洞后的掩码
    """
    # 使用从边界开始的漫水填充来查找背景
    h, w = mask.shape
    mask_copy = mask.copy()
    
    # 为漫水填充创建一个更大的掩码
    mask_floodfill = np.zeros((h + 2, w + 2), dtype=np.uint8)
    mask_floodfill[1:-1, 1:-1] = mask_copy
    
    # 从 (0, 0) 开始漫水填充
    cv2.floodFill(mask_floodfill, None, (0, 0), 1)
    
    # 反转以获取孔洞
    holes = 1 - mask_floodfill[1:-1, 1:-1]
    
    # 与原始掩码合并
    filled = mask | holes
    
    return filled


def overlay_mask_on_image(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.5
) -> np.ndarray:
    """
    以透明度将二值掩码叠加到图像上。
    
    参数:
        image: RGB 图像 (H, W, 3)
        mask: 二值掩码 (H, W)
        color: 掩码的 RGB 颜色
        alpha: 透明度 (0=透明, 1=不透明)
        
    返回:
        带有掩码叠加的图像
    """
    # 如果需要，将灰度图转换为 RGB
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # 创建彩色掩码
    colored_mask = np.zeros_like(image)
    colored_mask[mask > 0] = color
    
    # 混合图像和掩码
    overlay = cv2.addWeighted(image, 1, colored_mask, alpha, 0)
    
    return overlay


def compute_mask_area(mask: np.ndarray) -> int:
    """
    计算二值掩码的面积（像素数）。
    
    参数:
        mask: 二值掩码
        
    返回:
        正像素的数量
    """
    return int(np.sum(mask > 0))


def compute_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    计算两个掩码之间的交并比 (IoU)。
    
    参数:
        mask1: 第一个二值掩码
        mask2: 第二个二值掩码
        
    返回:
        IoU 分数 (0-1)
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    if union == 0:
        return 0.0
    
    return float(intersection / union)
