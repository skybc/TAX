"""
图像处理实用函数。

此模块提供图像加载、预处理、调整大小、归一化和其他图像操作函数。
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from src.logger import get_logger

logger = get_logger(__name__)


def load_image(
    image_path: Union[str, Path],
    mode: str = 'RGB'
) -> np.ndarray:
    """
    从文件加载图像。
    
    参数:
        image_path: 图像文件路径
        mode: 颜色模式 ('RGB', 'BGR', 'GRAY')
        
    返回:
        作为 numpy 数组的图像，彩色为 (H, W, C)，灰度为 (H, W)
        
    抛出:
        FileNotFoundError: 如果图像文件不存在
        ValueError: 如果无法加载图像
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"未找到图像: {image_path}")
    
    try:
        if mode == 'RGB':
            # 使用 PIL 加载 RGB
            image = Image.open(image_path).convert('RGB')
            image = np.array(image)
        elif mode == 'BGR':
            # 使用 OpenCV 加载 BGR
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        elif mode == 'GRAY':
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        else:
            raise ValueError(f"不支持的模式: {mode}")
        
        if image is None:
            raise ValueError(f"加载图像失败: {image_path}")
        
        logger.debug(f"已加载图像 {image_path}，形状为 {image.shape}")
        return image
    
    except Exception as e:
        logger.error(f"加载图像 {image_path} 时出错: {e}")
        raise


def save_image(
    image: np.ndarray,
    output_path: Union[str, Path],
    quality: int = 95
) -> None:
    """
    将图像保存到文件。
    
    参数:
        image: 作为 numpy 数组的图像
        output_path: 保存图像的路径
        quality: JPEG 质量 (1-100)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 如果使用 OpenCV，将 RGB 转换为 BGR
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(str(output_path), image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    logger.debug(f"已将图像保存到 {output_path}")


def resize_image(
    image: np.ndarray,
    target_size: Tuple[int, int],
    keep_aspect_ratio: bool = True,
    interpolation: int = cv2.INTER_LINEAR
) -> np.ndarray:
    """
    将图像调整为目标大小。
    
    参数:
        image: 输入图像
        target_size: 目标大小，格式为 (宽度, 高度)
        keep_aspect_ratio: 是否保持纵横比
        interpolation: 插值方法
        
    返回:
        调整大小后的图像
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    if keep_aspect_ratio:
        # 计算纵横比
        aspect = w / h
        target_aspect = target_w / target_h
        
        if aspect > target_aspect:
            # 宽度是限制因素
            new_w = target_w
            new_h = int(target_w / aspect)
        else:
            # 高度是限制因素
            new_h = target_h
            new_w = int(target_h * aspect)
        
        resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
        
        # 填充到目标大小
        top = (target_h - new_h) // 2
        bottom = target_h - new_h - top
        left = (target_w - new_w) // 2
        right = target_w - new_w - left
        
        if len(image.shape) == 3:
            resized = cv2.copyMakeBorder(
                resized, top, bottom, left, right,
                cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )
        else:
            resized = cv2.copyMakeBorder(
                resized, top, bottom, left, right,
                cv2.BORDER_CONSTANT, value=0
            )
    else:
        resized = cv2.resize(image, target_size, interpolation=interpolation)
    
    return resized


def normalize_image(
    image: np.ndarray,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None
) -> np.ndarray:
    """
    使用均值和标准差归一化图像。
    
    参数:
        image: 输入图像 (H, W, C)，范围在 [0, 255]
        mean: 每个通道的均值（默认为 ImageNet）
        std: 每个通道的标准差（默认为 ImageNet）
        
    返回:
        归一化后的图像
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]  # ImageNet 均值
    if std is None:
        std = [0.229, 0.224, 0.225]  # ImageNet 标准差
    
    # 转换为 float 并归一化到 [0, 1]
    image = image.astype(np.float32) / 255.0
    
    # 应用均值和标准差
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    
    image = (image - mean) / std
    
    return image


def denormalize_image(
    image: np.ndarray,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None
) -> np.ndarray:
    """
    将图像反归一化回 [0, 255] 范围。
    
    参数:
        image: 归一化后的图像
        mean: 用于归一化的均值
        std: 用于归一化的标准差
        
    返回:
        [0, 255] 范围内的反归一化图像
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]
    
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    
    image = (image * std) + mean
    image = (image * 255.0).clip(0, 255).astype(np.uint8)
    
    return image


def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    """将 RGB 图像转换为 BGR。"""
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """将 BGR 图像转换为 RGB。"""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def gray_to_rgb(image: np.ndarray) -> np.ndarray:
    """将灰度图像转换为 RGB。"""
    return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)


def image_to_tensor(image: np.ndarray) -> np.ndarray:
    """
    将图像从 HWC 格式转换为 PyTorch 的 CHW 格式。
    
    参数:
        image: HWC 格式的图像
        
    返回:
        CHW 格式的图像
    """
    if len(image.shape) == 2:
        # 为灰度图添加通道维度
        image = image[np.newaxis, :, :]
    else:
        # 从 HWC 转置为 CHW
        image = np.transpose(image, (2, 0, 1))
    
    return image


def tensor_to_image(tensor: np.ndarray) -> np.ndarray:
    """
    将张量从 CHW 格式转换为 HWC 格式。
    
    参数:
        tensor: CHW 格式的张量
        
    返回:
        HWC 格式的图像
    """
    if len(tensor.shape) == 3:
        if tensor.shape[0] == 1:
            # 灰度图：移除通道维度
            image = tensor[0]
        else:
            # RGB：从 CHW 转置为 HWC
            image = np.transpose(tensor, (1, 2, 0))
    else:
        image = tensor
    
    return image


def crop_image(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int]
) -> np.ndarray:
    """
    使用边界框裁剪图像。
    
    参数:
        image: 输入图像
        bbox: 边界框，格式为 (x1, y1, x2, y2)
        
    返回:
        裁剪后的图像
    """
    x1, y1, x2, y2 = bbox
    return image[y1:y2, x1:x2]


def pad_image(
    image: np.ndarray,
    pad_size: Union[int, Tuple[int, int, int, int]],
    value: Union[int, Tuple[int, int, int]] = 0
) -> np.ndarray:
    """
    使用指定值填充图像。
    
    参数:
        image: 输入图像
        pad_size: 填充大小（单个值或 (上, 下, 左, 右)）
        value: 填充值
        
    返回:
        填充后的图像
    """
    if isinstance(pad_size, int):
        pad_size = (pad_size, pad_size, pad_size, pad_size)
    
    top, bottom, left, right = pad_size
    
    if len(image.shape) == 3:
        if not isinstance(value, tuple):
            value = (value, value, value)
    
    return cv2.copyMakeBorder(
        image, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=value
    )


def get_image_info(image: np.ndarray) -> dict:
    """
    获取图像信息。
    
    参数:
        image: 输入图像
        
    返回:
        包含图像信息的字典（形状、数据类型、最小值、最大值）
    """
    info = {
        'shape': image.shape,
        'dtype': str(image.dtype),
        'min': float(image.min()),
        'max': float(image.max()),
    }
    
    if len(image.shape) == 3:
        info['channels'] = image.shape[2]
    else:
        info['channels'] = 1
    
    return info
