"""
分割掩码的后处理实用程序。

此模块提供：
- 形态学操作
- 连通分量分析
- 轮廓提取
- 掩码细化
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional
from scipy import ndimage

from src.logger import get_logger

logger = get_logger(__name__)


def remove_small_objects(mask: np.ndarray, 
                         min_size: int = 100) -> np.ndarray:
    """
    移除小的连通分量。
    
    参数:
        mask: 二值掩码 (HxW)
        min_size: 以像素为单位的最小对象大小
        
    返回:
        过滤后的掩码
    """
    # 标记连通分量
    labeled, num_features = ndimage.label(mask)
    
    # 获取大小
    sizes = ndimage.sum(mask, labeled, range(num_features + 1))
    
    # 创建大对象的掩码
    mask_size = sizes >= min_size
    mask_size[0] = 0  # 背景
    
    # 应用掩码
    filtered = mask_size[labeled]
    
    return filtered.astype(np.uint8) * 255


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """
    仅保留最大的连通分量。
    
    参数:
        mask: 二值掩码 (HxW)
        
    返回:
        仅包含最大分量的掩码
    """
    # 标记连通分量
    labeled, num_features = ndimage.label(mask)
    
    if num_features == 0:
        return mask
    
    # 获取大小
    sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
    
    # 保留最大的
    largest_label = np.argmax(sizes) + 1
    largest_mask = (labeled == largest_label).astype(np.uint8) * 255
    
    return largest_mask


def fill_holes(mask: np.ndarray, max_hole_size: Optional[int] = None) -> np.ndarray:
    """
    填充二值掩码中的孔洞。
    
    参数:
        mask: 二值掩码 (HxW)
        max_hole_size: 要填充的最大孔洞大小（None 表示填充所有）
        
    返回:
        填充后的掩码
    """
    # 二值化
    binary_mask = (mask > 0).astype(np.uint8)
    
    # 填充孔洞
    filled = ndimage.binary_fill_holes(binary_mask)
    
    if max_hole_size is not None:
        # 查找孔洞
        holes = filled & ~binary_mask
        
        # 标记孔洞
        labeled_holes, num_holes = ndimage.label(holes)
        
        # 获取孔洞大小
        hole_sizes = ndimage.sum(holes, labeled_holes, range(1, num_holes + 1))
        
        # 仅保留填充的小孔洞
        for i, size in enumerate(hole_sizes, 1):
            if size > max_hole_size:
                filled[labeled_holes == i] = 0
    
    return filled.astype(np.uint8) * 255


def morphological_opening(mask: np.ndarray, 
                         kernel_size: int = 5) -> np.ndarray:
    """
    应用形态学开运算（先腐蚀后膨胀）。
    
    参数:
        mask: 二值掩码 (HxW)
        kernel_size: 结构元素的大小
        
    返回:
        处理后的掩码
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return opened


def morphological_closing(mask: np.ndarray, 
                         kernel_size: int = 5) -> np.ndarray:
    """
    应用形态学闭运算（先膨胀后腐蚀）。
    
    参数:
        mask: 二值掩码 (HxW)
        kernel_size: 结构元素的大小
        
    返回:
        处理后的掩码
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return closed


def smooth_contours(mask: np.ndarray, 
                   epsilon_factor: float = 0.01) -> np.ndarray:
    """
    使用 Douglas-Peucker 算法平滑掩码轮廓。
    
    参数:
        mask: 二值掩码 (HxW)
        epsilon_factor: 近似精度因子
        
    返回:
        平滑后的掩码
    """
    # 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 平滑轮廓
    smoothed_mask = np.zeros_like(mask)
    
    for contour in contours:
        # 近似轮廓
        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # 绘制平滑后的轮廓
        cv2.fillPoly(smoothed_mask, [approx], 255)
    
    return smoothed_mask


def refine_mask(mask: np.ndarray,
               remove_small: bool = True,
               min_size: int = 100,
               fill_holes_flag: bool = True,
               smooth: bool = True,
               closing_size: int = 5) -> np.ndarray:
    """
    应用完整的掩码细化流水线。
    
    参数:
        mask: 二值掩码 (HxW)
        remove_small: 是否移除小对象
        min_size: 最小对象大小
        fill_holes_flag: 是否填充孔洞
        smooth: 是否平滑轮廓
        closing_size: 形态学闭运算的核大小
        
    返回:
        细化后的掩码
    """
    refined = mask.copy()
    
    # 形态学闭运算（填充小间隙）
    if closing_size > 0:
        refined = morphological_closing(refined, closing_size)
    
    # 填充孔洞
    if fill_holes_flag:
        refined = fill_holes(refined)
    
    # 移除小对象
    if remove_small:
        refined = remove_small_objects(refined, min_size)
    
    # 平滑轮廓
    if smooth:
        refined = smooth_contours(refined, epsilon_factor=0.005)
    
    return refined


def extract_contours(mask: np.ndarray) -> List[np.ndarray]:
    """
    从二值掩码中提取轮廓。
    
    参数:
        mask: 二值掩码 (HxW)
        
    返回:
        轮廓列表（每个都是 Nx1x2 数组）
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def get_bounding_boxes(mask: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    获取掩码中所有对象的边界框。
    
    参数:
        mask: 二值掩码 (HxW)
        
    返回:
        边界框列表 (x, y, w, h)
    """
    contours = extract_contours(mask)
    bboxes = []
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bboxes.append((x, y, w, h))
    
    return bboxes


def compute_mask_confidence(prob_map: np.ndarray, 
                           mask: np.ndarray) -> float:
    """
    基于概率图计算掩码的置信度分数。
    
    参数:
        prob_map: [0, 1] 范围内的概率图 (HxW)
        mask: 二值掩码 (HxW)
        
    返回:
        置信度分数（掩码区域内的平均概率）
    """
    if mask.sum() == 0:
        return 0.0
    
    # 获取掩码区域内的概率
    mask_binary = (mask > 0).astype(bool)
    probs = prob_map[mask_binary]
    
    return float(np.mean(probs))


def apply_crf(image: np.ndarray, 
             prob_map: np.ndarray,
             num_iterations: int = 10) -> np.ndarray:
    """
    应用条件随机场 (CRF) 进行掩码细化。
    
    注意：需要 pydensecrf 库（可选依赖项）
    
    参数:
        image: 原始图像 (HxWx3)
        prob_map: [0, 1] 范围内的概率图 (HxW)
        num_iterations: CRF 迭代次数
        
    返回:
        细化后的掩码
    """
    try:
        import pydensecrf.densecrf as dcrf
        from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral
        
        # 转换为标签
        labels = (prob_map > 0.5).astype(np.uint32)
        
        # 设置 CRF
        h, w = prob_map.shape
        d = dcrf.DenseCRF2D(w, h, 2)
        
        # 一元势能
        U = unary_from_labels(labels, 2, gt_prob=0.7)
        d.setUnaryEnergy(U)
        
        # 二元势能
        d.addPairwiseGaussian(sxy=3, compat=3)
        d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=image, compat=10)
        
        # 推理
        Q = d.inference(num_iterations)
        map_result = np.argmax(Q, axis=0).reshape(h, w)
        
        return (map_result * 255).astype(np.uint8)
        
    except ImportError:
        logger.warning("未安装 pydensecrf，跳过 CRF 细化")
        return (prob_map > 0.5).astype(np.uint8) * 255


def compute_mask_metrics(pred_mask: np.ndarray, 
                        gt_mask: np.ndarray) -> dict:
    """
    计算预测掩码和真实掩码之间的指标。
    
    参数:
        pred_mask: 预测掩码 (HxW)
        gt_mask: 真实掩码 (HxW)
        
    返回:
        包含指标的字典（IoU、Dice 等）
    """
    # 二值化
    pred_binary = (pred_mask > 0).astype(bool)
    gt_binary = (gt_mask > 0).astype(bool)
    
    # 计算交集和并集
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()
    pred_sum = pred_binary.sum()
    gt_sum = gt_binary.sum()
    
    # 计算指标
    iou = intersection / (union + 1e-6)
    dice = 2 * intersection / (pred_sum + gt_sum + 1e-6)
    
    # 像素准确率
    correct = (pred_binary == gt_binary).sum()
    total = pred_binary.size
    accuracy = correct / total
    
    # 精确率和召回率
    tp = intersection
    fp = pred_sum - intersection
    fn = gt_sum - intersection
    
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    
    return {
        'iou': float(iou),
        'dice': float(dice),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1)
    }


def create_comparison_image(image: np.ndarray,
                           pred_mask: np.ndarray,
                           gt_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    创建并排对比图像。
    
    参数:
        image: 原始图像 (HxWx3)
        pred_mask: 预测掩码 (HxW)
        gt_mask: 真实掩码 (HxW)，可选
        
    返回:
        对比图像
    """
    from src.utils.mask_utils import overlay_mask_on_image
    
    # 创建叠加层
    pred_overlay = overlay_mask_on_image(image, pred_mask, alpha=0.5, color=(0, 255, 0))
    
    if gt_mask is not None:
        gt_overlay = overlay_mask_on_image(image, gt_mask, alpha=0.5, color=(255, 0, 0))
        
        # 拼接：原始图 | 预测图 | 真实图
        comparison = np.concatenate([image, pred_overlay, gt_overlay], axis=1)
    else:
        # 拼接：原始图 | 预测图
        comparison = np.concatenate([image, pred_overlay], axis=1)
    
    return comparison
