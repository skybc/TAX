"""
用于分割推理的预测器。

此模块提供：
- 从检查点加载模型
- 单张和批量预测
- 后处理集成
- 结果保存
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Union, Tuple
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import cv2

from src.logger import get_logger
from src.models.segmentation_models import build_model
from src.utils.image_utils import load_image, save_image, resize_image
from src.utils.mask_utils import save_mask

logger = get_logger(__name__)


class Predictor:
    """
    用于分割推理的预测器。
    
    处理：
    - 加载训练好的模型
    - 单张图像预测
    - 批量预测
    - 结果可视化和保存
    """
    
    def __init__(self,
                 checkpoint_path: str,
                 device: Optional[torch.device] = None,
                 image_size: Tuple[int, int] = (512, 512)):
        """
        初始化预测器。
        
        参数:
            checkpoint_path: 模型检查点路径
            device: 推理设备（None 表示自动检测）
            image_size: 模型的输入图像尺寸 (H, W)
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device if device is not None else \
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_size = image_size
        
        # 模型和配置
        self.model: Optional[nn.Module] = None
        self.config: Dict = {}
        
        # 统计信息
        self.num_predictions = 0
        
        logger.info(f"预测器已初始化: 设备={self.device}, 图像尺寸={image_size}")
    
    def load_model(self, architecture: str = 'unet', 
                   encoder: str = 'resnet34',
                   num_classes: int = 1) -> bool:
        """
        从检查点加载模型。
        
        参数:
            architecture: 模型架构
            encoder: 编码器主干
            num_classes: 输出类别数
            
        返回:
            如果成功则为 True，否则为 False
        """
        try:
            # 构建模型
            self.model = build_model(
                architecture=architecture,
                encoder_name=encoder,
                encoder_weights=None,  # 从检查点加载
                in_channels=3,
                num_classes=num_classes,
                activation='sigmoid'
            )
            
            # 加载检查点
            if not self.checkpoint_path.exists():
                logger.error(f"未找到检查点: {self.checkpoint_path}")
                return False
            
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            
            # 加载状态字典
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                logger.info("已从检查点加载模型")
            else:
                self.model.load_state_dict(checkpoint)
                logger.info("已直接加载模型权重")
            
            # 移动到设备并设置为评估模式
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # 存储配置
            self.config = {
                'architecture': architecture,
                'encoder': encoder,
                'num_classes': num_classes,
                'checkpoint': str(self.checkpoint_path)
            }
            
            logger.info(f"模型已加载: {architecture} 使用 {encoder}")
            return True
            
        except Exception as e:
            logger.error(f"加载模型失败: {e}", exc_info=True)
            return False
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        为推理预处理图像。
        
        参数:
            image: RGB 格式的输入图像 (HxWxC)
            
        返回:
            预处理后的张量 (1xCxHxW)
        """
        # 存储原始尺寸
        original_size = image.shape[:2]
        
        # 调整尺寸
        if image.shape[:2] != self.image_size:
            image = resize_image(image, self.image_size)
        
        # 归一化 (使用 ImageNet 统计数据)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        image = image.astype(np.float32) / 255.0
        image = (image - mean) / std
        
        # 转换为张量 (HxWxC -> CxHxW)
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        
        # 添加 batch 维度
        image = image.unsqueeze(0)
        
        return image
    
    def postprocess_mask(self, mask: torch.Tensor, 
                        original_size: Tuple[int, int],
                        threshold: float = 0.5) -> np.ndarray:
        """
        后处理预测掩码。
        
        参数:
            mask: 预测的掩码张量 (1xCxHxW)
            original_size: 原始图像尺寸 (H, W)
            threshold: 二值化阈值
            
        返回:
            uint8 格式的二值掩码 (HxW)
        """
        # 移除 batch 和 channel 维度
        mask = mask.squeeze().cpu().numpy()
        
        # 二值化
        mask = (mask > threshold).astype(np.uint8) * 255
        
        # 调整回原始尺寸
        if mask.shape != original_size:
            mask = cv2.resize(mask, (original_size[1], original_size[0]), 
                            interpolation=cv2.INTER_NEAREST)
        
        return mask
    
    def predict(self, 
                image: Union[str, np.ndarray],
                threshold: float = 0.5,
                return_prob: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        预测单张图像的掩码。
        
        参数:
            image: 图像路径或 numpy 数组 (HxWxC)
            threshold: 二值化阈值
            return_prob: 是否返回概率图
            
        返回:
            二值掩码或 (mask, prob_map) 元组
        """
        if self.model is None:
            raise RuntimeError("模型未加载。请先调用 load_model()。")
        
        # 如果是路径则加载图像
        if isinstance(image, str):
            image = load_image(image)
            if image is None:
                raise ValueError(f"加载图像失败: {image}")
        
        original_size = image.shape[:2]
        
        # 预处理
        input_tensor = self.preprocess_image(image).to(self.device)
        
        # 预测
        with torch.no_grad():
            output = self.model(input_tensor)
        
        # 后处理
        mask = self.postprocess_mask(output, original_size, threshold)
        
        self.num_predictions += 1
        
        if return_prob:
            prob_map = output.squeeze().cpu().numpy()
            if prob_map.shape != original_size:
                prob_map = cv2.resize(prob_map, (original_size[1], original_size[0]))
            return mask, prob_map
        else:
            return mask
    
    def predict_batch(self,
                     image_paths: List[str],
                     output_dir: str,
                     threshold: float = 0.5,
                     save_overlay: bool = True,
                     progress_callback: Optional[callable] = None) -> Dict:
        """
        预测多张图像的掩码。
        
        参数:
            image_paths: 图像文件路径列表
            output_dir: 保存预测结果的目录
            threshold: 二值化阈值
            save_overlay: 是否保存叠加可视化图
            progress_callback: 可选的回调函数 (current, total, image_path)
            
        返回:
            包含预测统计信息的字典
        """
        if self.model is None:
            raise RuntimeError("模型未加载。请先调用 load_model()。")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        masks_dir = output_dir / "masks"
        masks_dir.mkdir(exist_ok=True)
        
        if save_overlay:
            overlay_dir = output_dir / "overlays"
            overlay_dir.mkdir(exist_ok=True)
        
        results = {
            'total': len(image_paths),
            'successful': 0,
            'failed': 0,
            'failed_files': []
        }
        
        for i, image_path in enumerate(image_paths):
            try:
                # 进度回调
                if progress_callback is not None:
                    progress_callback(i + 1, len(image_paths), image_path)
                
                # 加载图像
                image = load_image(image_path)
                if image is None:
                    results['failed'] += 1
                    results['failed_files'].append(image_path)
                    continue
                
                # 预测
                mask, prob_map = self.predict(image, threshold, return_prob=True)
                
                # 保存掩码
                image_name = Path(image_path).stem
                mask_path = masks_dir / f"{image_name}_mask.png"
                save_mask(mask, str(mask_path))
                
                # 保存叠加图
                if save_overlay:
                    overlay = self._create_overlay(image, mask)
                    overlay_path = overlay_dir / f"{image_name}_overlay.png"
                    save_image(overlay, str(overlay_path))
                
                results['successful'] += 1
                
            except Exception as e:
                logger.error(f"预测 {image_path} 失败: {e}")
                results['failed'] += 1
                results['failed_files'].append(image_path)
        
        logger.info(f"批量预测完成: {results['successful']}/{results['total']} 成功")
        
        return results
    
    def _create_overlay(self, image: np.ndarray, mask: np.ndarray, 
                       alpha: float = 0.5, color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """
        创建叠加可视化图。
        
        参数:
            image: 原始图像 (HxWxC)
            mask: 二值掩码 (HxW)
            alpha: 透明度
            color: 叠加颜色 (R, G, B)
            
        返回:
            叠加图像 (HxWxC)
        """
        # 确保图像是 RGB 格式
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # 创建彩色掩码
        colored_mask = np.zeros_like(image)
        colored_mask[mask > 0] = color
        
        # 混合
        overlay = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)
        
        return overlay
    
    def predict_with_tta(self,
                        image: Union[str, np.ndarray],
                        threshold: float = 0.5,
                        num_augmentations: int = 4) -> np.ndarray:
        """
        使用测试时增强 (TTA) 进行预测。
        
        参数:
            image: 图像路径或 numpy 数组
            threshold: 二值化阈值
            num_augmentations: 增强变体的数量
            
        返回:
            二值掩码
        """
        if self.model is None:
            raise RuntimeError("模型未加载。请先调用 load_model()。")
        
        # 如果是路径则加载图像
        if isinstance(image, str):
            image = load_image(image)
            if image is None:
                raise ValueError(f"加载图像失败: {image}")
        
        original_size = image.shape[:2]
        predictions = []
        
        # 原始图像
        pred = self.predict(image, threshold=1.0, return_prob=True)[1]
        predictions.append(pred)
        
        # 水平翻转
        if num_augmentations >= 2:
            flipped = cv2.flip(image, 1)
            pred = self.predict(flipped, threshold=1.0, return_prob=True)[1]
            pred = cv2.flip(pred, 1)
            predictions.append(pred)
        
        # 垂直翻转
        if num_augmentations >= 3:
            flipped = cv2.flip(image, 0)
            pred = self.predict(flipped, threshold=1.0, return_prob=True)[1]
            pred = cv2.flip(pred, 0)
            predictions.append(pred)
        
        # 两种翻转
        if num_augmentations >= 4:
            flipped = cv2.flip(cv2.flip(image, 0), 1)
            pred = self.predict(flipped, threshold=1.0, return_prob=True)[1]
            pred = cv2.flip(cv2.flip(pred, 0), 1)
            predictions.append(pred)
        
        # 平均预测结果
        avg_pred = np.mean(predictions, axis=0)
        
        # 阈值处理
        mask = (avg_pred > threshold).astype(np.uint8) * 255
        
        return mask
    
    def get_model_info(self) -> Dict:
        """
        获取模型信息。
        
        返回:
            包含模型信息的字典
        """
        if self.model is None:
            return {'status': '未加载'}
        
        info = {
            'status': '已加载',
            'config': self.config,
            'device': str(self.device),
            'image_size': self.image_size,
            'num_predictions': self.num_predictions,
        }
        
        # 添加参数计数
        if hasattr(self.model, 'get_model_info'):
            info.update(self.model.get_model_info())
        
        return info
    
    def reset_stats(self):
        """重置预测统计信息。"""
        self.num_predictions = 0
        logger.debug("统计信息已重置")


def create_predictor(checkpoint_path: str,
                    architecture: str = 'unet',
                    encoder: str = 'resnet34',
                    device: Optional[str] = None,
                    image_size: Tuple[int, int] = (512, 512)) -> Predictor:
    """
    创建并初始化预测器。
    
    参数:
        checkpoint_path: 模型检查点路径
        architecture: 模型架构
        encoder: 编码器主干
        device: 设备（'cuda' 或 'cpu'，None 表示自动）
        image_size: 输入图像尺寸
        
    返回:
        初始化后的预测器
    """
    # 解析设备
    if device is None:
        device_obj = None
    else:
        device_obj = torch.device(device)
    
    # 创建预测器
    predictor = Predictor(
        checkpoint_path=checkpoint_path,
        device=device_obj,
        image_size=image_size
    )
    
    # 加载模型
    success = predictor.load_model(
        architecture=architecture,
        encoder=encoder
    )
    
    if not success:
        logger.error("创建预测器失败")
        return None
    
    return predictor
