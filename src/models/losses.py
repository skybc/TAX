"""
Loss functions for segmentation training.

This module provides:
- Dice Loss
- Binary Cross Entropy Loss
- Focal Loss
- Combined losses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from src.logger import get_logger

logger = get_logger(__name__)


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation.
    
    Dice coefficient: 2 * |A ∩ B| / (|A| + |B|)
    Dice loss: 1 - Dice coefficient
    """
    
    def __init__(self, smooth: float = 1.0):
        """
        Initialize Dice Loss.
        
        Args:
            smooth: Smoothing factor to avoid division by zero
        """
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Dice Loss.
        
        Args:
            pred: Predicted masks (B, C, H, W)
            target: Ground truth masks (B, C, H, W)
            
        Returns:
            Dice loss value
        """
        # Flatten tensors
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        
        intersection = (pred * target).sum()
        dice = (2.0 * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Paper: https://arxiv.org/abs/1708.02002
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor in range (0,1) to balance positive/negative examples
            gamma: Exponent of the modulating factor (1 - p_t)
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Focal Loss.
        
        Args:
            pred: Predicted probabilities (B, C, H, W)
            target: Ground truth masks (B, C, H, W)
            
        Returns:
            Focal loss value
        """
        # Clip predictions to prevent log(0)
        pred = torch.clamp(pred, min=1e-7, max=1 - 1e-7)
        
        # Compute focal loss
        bce_loss = F.binary_cross_entropy(pred, target, reduction='none')
        p_t = pred * target + (1 - pred) * (1 - target)
        focal_weight = (1 - p_t) ** self.gamma
        
        if self.alpha is not None:
            alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
            focal_loss = alpha_t * focal_weight * bce_loss
        else:
            focal_loss = focal_weight * bce_loss
        
        return focal_loss.mean()


class CombinedLoss(nn.Module):
    """
    Combined loss: weighted sum of multiple losses.
    """
    
    def __init__(self,
                 dice_weight: float = 0.5,
                 bce_weight: float = 0.5,
                 focal_weight: float = 0.0):
        """
        Initialize Combined Loss.
        
        Args:
            dice_weight: Weight for Dice loss
            bce_weight: Weight for BCE loss
            focal_weight: Weight for Focal loss
        """
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCELoss()
        self.focal_loss = FocalLoss()
        
        logger.info(f"Combined loss: Dice={dice_weight}, BCE={bce_weight}, Focal={focal_weight}")
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute combined loss.
        
        Args:
            pred: Predicted masks (B, C, H, W)
            target: Ground truth masks (B, C, H, W)
            
        Returns:
            Combined loss value
        """
        loss = 0.0
        
        if self.dice_weight > 0:
            loss += self.dice_weight * self.dice_loss(pred, target)
        
        if self.bce_weight > 0:
            loss += self.bce_weight * self.bce_loss(pred, target)
        
        if self.focal_weight > 0:
            loss += self.focal_weight * self.focal_loss(pred, target)
        
        return loss


class IoULoss(nn.Module):
    """
    IoU (Intersection over Union) Loss.
    """
    
    def __init__(self, smooth: float = 1.0):
        """
        Initialize IoU Loss.
        
        Args:
            smooth: Smoothing factor
        """
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute IoU Loss.
        
        Args:
            pred: Predicted masks (B, C, H, W)
            target: Ground truth masks (B, C, H, W)
            
        Returns:
            IoU loss value
        """
        # Flatten tensors
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        
        iou = (intersection + self.smooth) / (union + self.smooth)
        
        return 1 - iou


def get_loss_function(loss_name: str, **kwargs) -> nn.Module:
    """
    Get loss function by name.
    
    Args:
        loss_name: Loss function name ('dice', 'bce', 'focal', 'combined', 'iou')
        **kwargs: Additional arguments for loss function
        
    Returns:
        Loss function module
        
    Raises:
        ValueError: If loss name is not supported
    """
    loss_name = loss_name.lower()
    
    if loss_name == 'dice':
        return DiceLoss(**kwargs)
    elif loss_name == 'bce':
        return nn.BCELoss(**kwargs)
    elif loss_name == 'focal':
        return FocalLoss(**kwargs)
    elif loss_name == 'combined':
        return CombinedLoss(**kwargs)
    elif loss_name == 'iou':
        return IoULoss(**kwargs)
    else:
        raise ValueError(f"Unsupported loss function: {loss_name}")
