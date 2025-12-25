"""
用于分割的评估指标。

此模块提供：
- IoU (交并比)
- Dice 系数
- 像素准确率
- 精确率/召回率/F1 分数
"""

import torch
import numpy as np
from typing import Tuple

from src.logger import get_logger

logger = get_logger(__name__)


def compute_iou(pred: torch.Tensor,
                target: torch.Tensor,
                threshold: float = 0.5,
                smooth: float = 1e-6) -> float:
    """
    计算交并比 (IoU)。
    
    参数:
        pred: 预测掩码 (B, C, H, W) 或 (B, H, W)
        target: 真值掩码 (B, C, H, W) 或 (B, H, W)
        threshold: 二值化预测的阈值
        smooth: 平滑因子
        
    返回:
        IoU 分数
    """
    # 二值化预测
    pred = (pred > threshold).float()
    
    # 展平张量
    pred = pred.view(-1)
    target = target.view(-1)
    
    # 计算交集和并集
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    
    return iou.item()


def compute_dice(pred: torch.Tensor,
                 target: torch.Tensor,
                 threshold: float = 0.5,
                 smooth: float = 1e-6) -> float:
    """
    计算 Dice 系数。
    
    参数:
        pred: 预测掩码 (B, C, H, W) 或 (B, H, W)
        target: 真值掩码 (B, C, H, W) 或 (B, H, W)
        threshold: 二值化预测的阈值
        smooth: 平滑因子
        
    返回:
        Dice 分数
    """
    # 二值化预测
    pred = (pred > threshold).float()
    
    # 展平张量
    pred = pred.view(-1)
    target = target.view(-1)
    
    # 计算 Dice
    intersection = (pred * target).sum()
    dice = (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return dice.item()


def compute_pixel_accuracy(pred: torch.Tensor,
                           target: torch.Tensor,
                           threshold: float = 0.5) -> float:
    """
    计算像素级准确率。
    
    参数:
        pred: 预测掩码 (B, C, H, W) 或 (B, H, W)
        target: 真值掩码 (B, C, H, W) 或 (B, H, W)
        threshold: 二值化预测的阈值
        
    返回:
        像素准确率
    """
    # 二值化预测
    pred = (pred > threshold).float()
    
    # 展平张量
    pred = pred.view(-1)
    target = target.view(-1)
    
    # 计算准确率
    correct = (pred == target).sum()
    total = target.numel()
    
    accuracy = correct.float() / total
    
    return accuracy.item()


def compute_precision_recall_f1(pred: torch.Tensor,
                                target: torch.Tensor,
                                threshold: float = 0.5,
                                smooth: float = 1e-6) -> Tuple[float, float, float]:
    """
    计算精确率、召回率和 F1 分数。
    
    参数:
        pred: 预测掩码 (B, C, H, W) 或 (B, H, W)
        target: 真值掩码 (B, C, H, W) 或 (B, H, W)
        threshold: 二值化预测的阈值
        smooth: 平滑因子
        
    返回:
        (精确率, 召回率, f1) 元组
    """
    # 二值化预测
    pred = (pred > threshold).float()
    
    # 展平张量
    pred = pred.view(-1)
    target = target.view(-1)
    
    # 计算 TP, FP, FN
    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    fn = ((1 - pred) * target).sum()
    
    # 计算指标
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    f1 = 2 * precision * recall / (precision + recall + smooth)
    
    return precision.item(), recall.item(), f1.item()


def compute_all_metrics(pred: torch.Tensor,
                       target: torch.Tensor,
                       threshold: float = 0.5) -> dict:
    """
    计算所有评估指标。
    
    参数:
        pred: 预测掩码
        target: 真值掩码
        threshold: 二值化预测的阈值
        
    返回:
        包含所有指标的字典
    """
    iou = compute_iou(pred, target, threshold)
    dice = compute_dice(pred, target, threshold)
    accuracy = compute_pixel_accuracy(pred, target, threshold)
    precision, recall, f1 = compute_precision_recall_f1(pred, target, threshold)
    
    return {
        'iou': iou,
        'dice': dice,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


class MetricsTracker:
    """
    用于在多个批次中累积指标的追踪器。
    """
    
    def __init__(self):
        """初始化指标追踪器。"""
        self.reset()
    
    def reset(self):
        """重置所有指标。"""
        self.metrics = {
            'iou': [],
            'dice': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
    
    def update(self, pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5):
        """
        使用新批次更新指标。
        
        参数:
            pred: 预测掩码
            target: 真值掩码
            threshold: 二值化预测的阈值
        """
        metrics = compute_all_metrics(pred, target, threshold)
        
        for key, value in metrics.items():
            self.metrics[key].append(value)
    
    def get_average(self) -> dict:
        """
        获取所有指标的平均值。
        
        返回:
            包含平均指标的字典
        """
        avg_metrics = {}
        
        for key, values in self.metrics.items():
            if values:
                avg_metrics[key] = np.mean(values)
            else:
                avg_metrics[key] = 0.0
        
        return avg_metrics
    
    def get_std(self) -> dict:
        """
        获取所有指标的标准差。
        
        返回:
            包含指标标准差的字典
        """
        std_metrics = {}
        
        for key, values in self.metrics.items():
            if values:
                std_metrics[key] = np.std(values)
            else:
                std_metrics[key] = 0.0
        
        return std_metrics
    
    def get_summary(self) -> str:
        """
        获取指标的摘要字符串。
        
        返回:
            格式化的摘要字符串
        """
        avg_metrics = self.get_average()
        
        summary = "指标摘要:\n"
        summary += f"  IoU:       {avg_metrics['iou']:.4f}\n"
        summary += f"  Dice:      {avg_metrics['dice']:.4f}\n"
        summary += f"  准确率:    {avg_metrics['accuracy']:.4f}\n"
        summary += f"  精确率:    {avg_metrics['precision']:.4f}\n"
        summary += f"  召回率:    {avg_metrics['recall']:.4f}\n"
        summary += f"  F1:        {avg_metrics['f1']:.4f}"
        
        return summary
