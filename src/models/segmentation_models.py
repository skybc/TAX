"""
分割模型定义。

此模块提供：
- U-Net 架构
- DeepLabV3+ 架构
- 模型构建实用程序
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict
import segmentation_models_pytorch as smp

from src.logger import get_logger

logger = get_logger(__name__)


class UNet(nn.Module):
    """
    用于语义分割的 U-Net 架构。
    
    论文: https://arxiv.org/abs/1505.04597
    """
    
    def __init__(self,
                 encoder_name: str = "resnet34",
                 encoder_weights: Optional[str] = "imagenet",
                 in_channels: int = 3,
                 num_classes: int = 1,
                 activation: Optional[str] = None):
        """
        初始化 U-Net 模型。
        
        参数:
            encoder_name: 编码器主干名称（例如 'resnet34', 'efficientnet-b0'）
            encoder_weights: 预训练权重（'imagenet' 或 None）
            in_channels: 输入通道数
            num_classes: 输出类别数
            activation: 输出激活函数（'sigmoid', 'softmax' 或 None）
        """
        super().__init__()
        
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            activation=activation
        )
        
        self.encoder_name = encoder_name
        self.num_classes = num_classes
        
        logger.info(f"已创建 U-Net: 编码器={encoder_name}, 类别数={num_classes}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。"""
        return self.model(x)
    
    def get_model_info(self) -> Dict:
        """获取模型信息。"""
        return {
            'architecture': 'U-Net',
            'encoder': self.encoder_name,
            'num_classes': self.num_classes,
            'num_parameters': sum(p.numel() for p in self.parameters())
        }


class DeepLabV3Plus(nn.Module):
    """
    用于语义分割的 DeepLabV3+ 架构。
    
    论文: https://arxiv.org/abs/1802.02611
    """
    
    def __init__(self,
                 encoder_name: str = "resnet50",
                 encoder_weights: Optional[str] = "imagenet",
                 in_channels: int = 3,
                 num_classes: int = 1,
                 activation: Optional[str] = None):
        """
        初始化 DeepLabV3+ 模型。
        
        参数:
            encoder_name: 编码器主干名称
            encoder_weights: 预训练权重
            in_channels: 输入通道数
            num_classes: 输出类别数
            activation: 输出激活函数
        """
        super().__init__()
        
        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            activation=activation
        )
        
        self.encoder_name = encoder_name
        self.num_classes = num_classes
        
        logger.info(f"已创建 DeepLabV3+: 编码器={encoder_name}, 类别数={num_classes}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。"""
        return self.model(x)
    
    def get_model_info(self) -> Dict:
        """获取模型信息。"""
        return {
            'architecture': 'DeepLabV3+',
            'encoder': self.encoder_name,
            'num_classes': self.num_classes,
            'num_parameters': sum(p.numel() for p in self.parameters())
        }


class FPN(nn.Module):
    """
    用于语义分割的特征金字塔网络 (FPN)。
    
    论文: https://arxiv.org/abs/1612.03144
    """
    
    def __init__(self,
                 encoder_name: str = "resnet34",
                 encoder_weights: Optional[str] = "imagenet",
                 in_channels: int = 3,
                 num_classes: int = 1,
                 activation: Optional[str] = None):
        """
        初始化 FPN 模型。
        
        参数:
            encoder_name: 编码器主干名称
            encoder_weights: 预训练权重
            in_channels: 输入通道数
            num_classes: 输出类别数
            activation: 输出激活函数
        """
        super().__init__()
        
        self.model = smp.FPN(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            activation=activation
        )
        
        self.encoder_name = encoder_name
        self.num_classes = num_classes
        
        logger.info(f"已创建 FPN: 编码器={encoder_name}, 类别数={num_classes}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。"""
        return self.model(x)
    
    def get_model_info(self) -> Dict:
        """获取模型信息。"""
        return {
            'architecture': 'FPN',
            'encoder': self.encoder_name,
            'num_classes': self.num_classes,
            'num_parameters': sum(p.numel() for p in self.parameters())
        }


def build_model(architecture: str,
                encoder_name: str = "resnet34",
                encoder_weights: Optional[str] = "imagenet",
                in_channels: int = 3,
                num_classes: int = 1,
                activation: Optional[str] = None) -> nn.Module:
    """
    构建分割模型。
    
    参数:
        architecture: 模型架构 ('unet', 'deeplabv3plus', 'fpn')
        encoder_name: 编码器主干名称
        encoder_weights: 预训练权重
        in_channels: 输入通道数
        num_classes: 输出类别数
        activation: 输出激活函数
        
    返回:
        PyTorch 模型
        
    抛出:
        ValueError: 如果架构不受支持
    """
    architecture = architecture.lower()
    
    if architecture == 'unet':
        model = UNet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            num_classes=num_classes,
            activation=activation
        )
    elif architecture in ['deeplabv3plus', 'deeplabv3+']:
        model = DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            num_classes=num_classes,
            activation=activation
        )
    elif architecture == 'fpn':
        model = FPN(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            num_classes=num_classes,
            activation=activation
        )
    else:
        raise ValueError(f"不支持的架构: {architecture}")
    
    logger.info(f"已构建模型: {architecture}，使用 {encoder_name}")
    return model


def get_available_encoders() -> List[str]:
    """
    获取可用编码器主干列表。
    
    返回:
        编码器名称列表
    """
    return [
        # ResNet 系列
        'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
        # EfficientNet 系列
        'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2',
        'efficientnet-b3', 'efficientnet-b4', 'efficientnet-b5',
        # MobileNet 系列
        'mobilenet_v2',
        # DenseNet 系列
        'densenet121', 'densenet169', 'densenet201',
        # VGG 系列
        'vgg11', 'vgg13', 'vgg16', 'vgg19',
    ]


def get_model_params_count(model: nn.Module) -> Dict:
    """
    获取模型参数计数统计信息。
    
    参数:
        model: PyTorch 模型
        
    返回:
        包含参数计数的字典
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': total_params - trainable_params,
        'total_params_M': total_params / 1e6,
        'trainable_params_M': trainable_params / 1e6
    }


def freeze_encoder(model: nn.Module):
    """
    冻结编码器权重（用于微调）。
    
    参数:
        model: 分割模型
    """
    if hasattr(model, 'model') and hasattr(model.model, 'encoder'):
        for param in model.model.encoder.parameters():
            param.requires_grad = False
        logger.info("编码器权重已冻结")
    else:
        logger.warning("无法冻结编码器（无法识别模型结构）")


def unfreeze_encoder(model: nn.Module):
    """
    解冻编码器权重。
    
    参数:
        model: 分割模型
    """
    if hasattr(model, 'model') and hasattr(model.model, 'encoder'):
        for param in model.model.encoder.parameters():
            param.requires_grad = True
        logger.info("编码器权重已解冻")
    else:
        logger.warning("无法解冻编码器（无法识别模型结构）")
