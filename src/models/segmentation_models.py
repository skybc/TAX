"""
Segmentation model definitions.

This module provides:
- U-Net architecture
- DeepLabV3+ architecture
- Model builder utilities
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
    U-Net architecture for semantic segmentation.
    
    Paper: https://arxiv.org/abs/1505.04597
    """
    
    def __init__(self,
                 encoder_name: str = "resnet34",
                 encoder_weights: Optional[str] = "imagenet",
                 in_channels: int = 3,
                 num_classes: int = 1,
                 activation: Optional[str] = None):
        """
        Initialize U-Net model.
        
        Args:
            encoder_name: Encoder backbone name (e.g., 'resnet34', 'efficientnet-b0')
            encoder_weights: Pretrained weights ('imagenet' or None)
            in_channels: Number of input channels
            num_classes: Number of output classes
            activation: Output activation ('sigmoid', 'softmax', or None)
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
        
        logger.info(f"Created U-Net: encoder={encoder_name}, classes={num_classes}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)
    
    def get_model_info(self) -> Dict:
        """Get model information."""
        return {
            'architecture': 'U-Net',
            'encoder': self.encoder_name,
            'num_classes': self.num_classes,
            'num_parameters': sum(p.numel() for p in self.parameters())
        }


class DeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ architecture for semantic segmentation.
    
    Paper: https://arxiv.org/abs/1802.02611
    """
    
    def __init__(self,
                 encoder_name: str = "resnet50",
                 encoder_weights: Optional[str] = "imagenet",
                 in_channels: int = 3,
                 num_classes: int = 1,
                 activation: Optional[str] = None):
        """
        Initialize DeepLabV3+ model.
        
        Args:
            encoder_name: Encoder backbone name
            encoder_weights: Pretrained weights
            in_channels: Number of input channels
            num_classes: Number of output classes
            activation: Output activation
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
        
        logger.info(f"Created DeepLabV3+: encoder={encoder_name}, classes={num_classes}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)
    
    def get_model_info(self) -> Dict:
        """Get model information."""
        return {
            'architecture': 'DeepLabV3+',
            'encoder': self.encoder_name,
            'num_classes': self.num_classes,
            'num_parameters': sum(p.numel() for p in self.parameters())
        }


class FPN(nn.Module):
    """
    Feature Pyramid Network for semantic segmentation.
    
    Paper: https://arxiv.org/abs/1612.03144
    """
    
    def __init__(self,
                 encoder_name: str = "resnet34",
                 encoder_weights: Optional[str] = "imagenet",
                 in_channels: int = 3,
                 num_classes: int = 1,
                 activation: Optional[str] = None):
        """
        Initialize FPN model.
        
        Args:
            encoder_name: Encoder backbone name
            encoder_weights: Pretrained weights
            in_channels: Number of input channels
            num_classes: Number of output classes
            activation: Output activation
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
        
        logger.info(f"Created FPN: encoder={encoder_name}, classes={num_classes}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)
    
    def get_model_info(self) -> Dict:
        """Get model information."""
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
    Build a segmentation model.
    
    Args:
        architecture: Model architecture ('unet', 'deeplabv3plus', 'fpn')
        encoder_name: Encoder backbone name
        encoder_weights: Pretrained weights
        in_channels: Number of input channels
        num_classes: Number of output classes
        activation: Output activation
        
    Returns:
        PyTorch model
        
    Raises:
        ValueError: If architecture is not supported
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
        raise ValueError(f"Unsupported architecture: {architecture}")
    
    logger.info(f"Built model: {architecture} with {encoder_name}")
    return model


def get_available_encoders() -> List[str]:
    """
    Get list of available encoder backbones.
    
    Returns:
        List of encoder names
    """
    return [
        # ResNet family
        'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
        # EfficientNet family
        'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2',
        'efficientnet-b3', 'efficientnet-b4', 'efficientnet-b5',
        # MobileNet family
        'mobilenet_v2',
        # DenseNet family
        'densenet121', 'densenet169', 'densenet201',
        # VGG family
        'vgg11', 'vgg13', 'vgg16', 'vgg19',
    ]


def get_model_params_count(model: nn.Module) -> Dict:
    """
    Get model parameter count statistics.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
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
    Freeze encoder weights (for fine-tuning).
    
    Args:
        model: Segmentation model
    """
    if hasattr(model, 'model') and hasattr(model.model, 'encoder'):
        for param in model.model.encoder.parameters():
            param.requires_grad = False
        logger.info("Encoder weights frozen")
    else:
        logger.warning("Could not freeze encoder (model structure not recognized)")


def unfreeze_encoder(model: nn.Module):
    """
    Unfreeze encoder weights.
    
    Args:
        model: Segmentation model
    """
    if hasattr(model, 'model') and hasattr(model.model, 'encoder'):
        for param in model.model.encoder.parameters():
            param.requires_grad = True
        logger.info("Encoder weights unfrozen")
    else:
        logger.warning("Could not unfreeze encoder (model structure not recognized)")
