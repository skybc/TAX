"""
用于分割训练的数据集加载器。

此模块提供：
- 带有数据增强功能的 SegmentationDataset
- 数据加载实用程序
- 训练/验证集划分处理
"""

import os
from pathlib import Path
from typing import Optional, Callable, List, Tuple, Dict
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.logger import get_logger
from src.utils.image_utils import load_image
from src.utils.mask_utils import load_mask

logger = get_logger(__name__)


class SegmentationDataset(Dataset):
    """
    语义分割数据集。
    
    加载图像及其对应的掩码，并应用数据增强。
    """
    
    def __init__(self,
                 image_paths: List[str],
                 mask_paths: List[str],
                 transform: Optional[Callable] = None,
                 preprocessing: Optional[Callable] = None):
        """
        初始化数据集。
        
        参数:
            image_paths: 图像文件路径列表
            mask_paths: 掩码文件路径列表
            transform: 增强变换
            preprocessing: 预处理函数
        """
        assert len(image_paths) == len(mask_paths), \
            "图像和掩码的数量必须匹配"
        
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.preprocessing = preprocessing
        
        logger.info(f"已创建包含 {len(image_paths)} 个样本的数据集")
    
    def __len__(self) -> int:
        """获取数据集大小。"""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        通过索引获取项。
        
        参数:
            idx: 样本索引
            
        返回:
            张量格式的 (image, mask) 元组
        """
        # 加载图像和掩码
        image = load_image(self.image_paths[idx])
        mask = load_mask(self.mask_paths[idx])
        
        if image is None or mask is None:
            logger.error(f"加载失败: {self.image_paths[idx]}")
            # 返回虚拟数据
            return torch.zeros(3, 256, 256), torch.zeros(1, 256, 256)
        
        # 确保掩码是 2D 的
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        
        # 将掩码归一化到 [0, 1]
        if mask.max() > 1:
            mask = mask / 255.0
        
        # 应用增强
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        # 应用预处理
        if self.preprocessing is not None:
            preprocessed = self.preprocessing(image=image, mask=mask)
            image = preprocessed['image']
            mask = preprocessed['mask']
        
        # 如果需要，为掩码添加通道维度
        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=0)
        
        # 如果还不是张量，则转换为张量
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).float()
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).float()
        
        return image, mask
    
    def get_sample_info(self, idx: int) -> Dict:
        """
        获取样本信息。
        
        参数:
            idx: 样本索引
            
        返回:
            包含样本信息的字典
        """
        return {
            'image_path': self.image_paths[idx],
            'mask_path': self.mask_paths[idx],
            'image_name': Path(self.image_paths[idx]).name,
            'mask_name': Path(self.mask_paths[idx]).name
        }


def get_training_augmentation(image_size: Tuple[int, int] = (512, 512),
                              p: float = 0.5) -> A.Compose:
    """
    获取训练数据增强流水线。
    
    参数:
        image_size: 目标图像尺寸 (高度, 宽度)
        p: 应用增强的概率
        
    返回:
        Albumentations Compose 对象
    """
    return A.Compose([
        # 几何变换
        A.Resize(*image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.2,
            rotate_limit=45,
            p=p
        ),
        
        # 颜色变换
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=1.0
            ),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1.0
            ),
            A.RandomGamma(p=1.0),
        ], p=p),
        
        # 噪声和模糊
        A.OneOf([
            A.GaussNoise(p=1.0),
            A.GaussianBlur(p=1.0),
            A.MotionBlur(p=1.0),
        ], p=p * 0.5),
        
        # 归一化并转换为张量
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ])


def get_validation_augmentation(image_size: Tuple[int, int] = (512, 512)) -> A.Compose:
    """
    获取验证数据增强流水线（无随机增强）。
    
    参数:
        image_size: 目标图像尺寸 (高度, 宽度)
        
    返回:
        Albumentations Compose 对象
    """
    return A.Compose([
        A.Resize(*image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ])


def get_preprocessing(preprocessing_fn: Optional[Callable] = None) -> A.Compose:
    """
    获取预处理流水线。
    
    参数:
        preprocessing_fn: 来自编码器的预处理函数
        
    返回:
        Albumentations Compose 对象
    """
    transforms = []
    
    if preprocessing_fn is not None:
        transforms.append(A.Lambda(image=preprocessing_fn))
    
    return A.Compose(transforms) if transforms else None


def create_dataloaders(train_image_paths: List[str],
                       train_mask_paths: List[str],
                       val_image_paths: List[str],
                       val_mask_paths: List[str],
                       batch_size: int = 8,
                       num_workers: int = 4,
                       image_size: Tuple[int, int] = (512, 512),
                       augmentation_prob: float = 0.5) -> Tuple:
    """
    创建训练和验证数据加载器。
    
    参数:
        train_image_paths: 训练图像路径
        train_mask_paths: 训练掩码路径
        val_image_paths: 验证图像路径
        val_mask_paths: 验证掩码路径
        batch_size: 批次大小
        num_workers: 数据加载的工作线程数
        image_size: 目标图像尺寸
        augmentation_prob: 应用增强的概率
        
    返回:
        (train_loader, val_loader) 元组
    """
    # 创建数据集
    train_dataset = SegmentationDataset(
        image_paths=train_image_paths,
        mask_paths=train_mask_paths,
        transform=get_training_augmentation(image_size, augmentation_prob)
    )
    
    val_dataset = SegmentationDataset(
        image_paths=val_image_paths,
        mask_paths=val_mask_paths,
        transform=get_validation_augmentation(image_size)
    )
    
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    logger.info(f"已创建数据加载器: 训练集={len(train_dataset)}, 验证集={len(val_dataset)}")
    logger.info(f"批次大小: {batch_size}, 工作线程数: {num_workers}")
    
    return train_loader, val_loader


def load_dataset_from_split_files(split_dir: str,
                                  images_dir: str,
                                  masks_dir: str) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    从划分文件（train.txt, val.txt）加载数据集。
    
    参数:
        split_dir: 包含划分文件的目录
        images_dir: 包含图像的目录
        masks_dir: 包含掩码的目录
        
    返回:
        (train_image_paths, train_mask_paths, val_image_paths, val_mask_paths)
    """
    split_dir = Path(split_dir)
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)
    
    # 加载训练划分
    train_file = split_dir / "train.txt"
    if not train_file.exists():
        logger.error(f"未找到训练划分文件: {train_file}")
        return [], [], [], []
    
    with open(train_file, 'r') as f:
        train_names = [line.strip() for line in f if line.strip()]
    
    # 加载验证划分
    val_file = split_dir / "val.txt"
    if not val_file.exists():
        logger.error(f"未找到验证划分文件: {val_file}")
        return [], [], [], []
    
    with open(val_file, 'r') as f:
        val_names = [line.strip() for line in f if line.strip()]
    
    # 构建路径
    train_image_paths = [str(images_dir / name) for name in train_names]
    train_mask_paths = [str(masks_dir / Path(name).stem) + ".png" for name in train_names]
    
    val_image_paths = [str(images_dir / name) for name in val_names]
    val_mask_paths = [str(masks_dir / Path(name).stem) + ".png" for name in val_names]
    
    # 过滤存在的文件
    train_image_paths, train_mask_paths = _filter_existing_pairs(train_image_paths, train_mask_paths)
    val_image_paths, val_mask_paths = _filter_existing_pairs(val_image_paths, val_mask_paths)
    
    logger.info(f"已从划分文件加载数据集: 训练集={len(train_image_paths)}, 验证集={len(val_image_paths)}")
    
    return train_image_paths, train_mask_paths, val_image_paths, val_mask_paths


def _filter_existing_pairs(image_paths: List[str], mask_paths: List[str]) -> Tuple[List[str], List[str]]:
    """
    过滤掉不存在的图像-掩码对。
    
    参数:
        image_paths: 图像路径列表
        mask_paths: 掩码路径列表
        
    返回:
        (filtered_image_paths, filtered_mask_paths)
    """
    filtered_images = []
    filtered_masks = []
    
    for img_path, mask_path in zip(image_paths, mask_paths):
        if Path(img_path).exists() and Path(mask_path).exists():
            filtered_images.append(img_path)
            filtered_masks.append(mask_path)
        else:
            logger.warning(f"跳过缺失的配对: {img_path} 或 {mask_path}")
    
    return filtered_images, filtered_masks


def compute_class_weights(mask_paths: List[str],
                          num_classes: int = 2) -> torch.Tensor:
    """
    为不平衡的数据集计算类别权重。
    
    参数:
        mask_paths: 掩码文件路径列表
        num_classes: 类别数量
        
    返回:
        类别权重张量
    """
    class_counts = np.zeros(num_classes)
    
    for mask_path in mask_paths:
        mask = load_mask(mask_path)
        if mask is None:
            continue
        
        # 二值化掩码
        mask_binary = (mask > 0).astype(np.uint8)
        
        # 统计像素
        class_counts[0] += np.sum(mask_binary == 0)  # 背景
        class_counts[1] += np.sum(mask_binary == 1)  # 前景
    
    # 计算权重（频率的倒数）
    total_pixels = class_counts.sum()
    class_weights = total_pixels / (num_classes * class_counts + 1e-6)
    
    # 归一化
    class_weights = class_weights / class_weights.sum() * num_classes
    
    logger.info(f"已计算类别权重: {class_weights}")
    
    return torch.from_numpy(class_weights).float()
