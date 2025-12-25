"""
用于分割训练的损失函数。

此模块提供：
- Dice 损失
- 二元交叉熵 (BCE) 损失
- Focal 损失
- 组合损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from src.logger import get_logger

logger = get_logger(__name__)


class DiceLoss(nn.Module):
    """
    用于分割的 Dice 损失。
    
    Dice 系数: 2 * |A ∩ B| / (|A| + |B|)
    Dice 损失: 1 - Dice 系数
    """
    
    def __init__(self, smooth: float = 1.0):
        """
        初始化 Dice 损失。
        
        参数:
            smooth: 用于避免除以零的平滑因子
        """
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算 Dice 损失。
        
        参数:
            pred: 预测掩码 (B, C, H, W)
            target: 真值掩码 (B, C, H, W)
            
        返回:
            Dice 损失值
        """
        # 展平张量
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        
        intersection = (pred * target).sum()
        dice = (2.0 * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice


class FocalLoss(nn.Module):
    """
    用于解决类别不平衡问题的 Focal 损失。
    
    论文: https://arxiv.org/abs/1708.02002
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """
        初始化 Focal 损失。
        
        参数:
            alpha: 范围在 (0,1) 内的加权因子，用于平衡正/负样本
            gamma: 调制因子 (1 - p_t) 的指数
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算 Focal 损失。
        
        参数:
            pred: 预测概率 (B, C, H, W)
            target: 真值掩码 (B, C, H, W)
            
        返回:
            Focal 损失值
        """
        # 限制预测值以防止 log(0)
        pred = torch.clamp(pred, min=1e-7, max=1 - 1e-7)
        
        # 计算 focal 损失
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
    组合损失：多个损失函数的加权和。
    """
    
    def __init__(self,
                 dice_weight: float = 0.5,
                 bce_weight: float = 0.5,
                 focal_weight: float = 0.0):
        """
        初始化组合损失。
        
        参数:
            dice_weight: Dice 损失的权重
            bce_weight: BCE 损失的权重
            focal_weight: Focal 损失的权重
        """
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCELoss()
        self.focal_loss = FocalLoss()
        
        logger.info(f"组合损失: Dice={dice_weight}, BCE={bce_weight}, Focal={focal_weight}")
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算组合损失。
        
        参数:
            pred: 预测掩码 (B, C, H, W)
            target: 真值掩码 (B, C, H, W)
            
        返回:
            组合损失值
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
    IoU (交并比) 损失。
    """
    
    def __init__(self, smooth: float = 1.0):
        """
        初始化 IoU 损失。
        
        参数:
            smooth: 平滑因子
        """
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算 IoU 损失。
        
        参数:
            pred: 预测掩码 (B, C, H, W)
            target: 真值掩码 (B, C, H, W)
            
        返回:
            IoU 损失值
        """
        # 展平张量
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        
        iou = (intersection + self.smooth) / (union + self.smooth)
        
        return 1 - iou


def get_loss_function(loss_name: str, **kwargs) -> nn.Module:
    """
    通过名称获取损失函数。
    
    参数:
        loss_name: 损失函数名称 ('dice', 'bce', 'focal', 'combined', 'iou')
        **kwargs: 损失函数的附加参数
        
    返回:
        损失函数模块
        
    抛出:
        ValueError: 如果损失函数名称不受支持
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
        raise ValueError(f"不支持的损失函数: {loss_name}")
