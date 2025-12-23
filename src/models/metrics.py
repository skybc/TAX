"""
Evaluation metrics for segmentation.

This module provides:
- IoU (Intersection over Union)
- Dice coefficient
- Pixel accuracy
- Precision/Recall/F1
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
    Compute Intersection over Union (IoU).
    
    Args:
        pred: Predicted masks (B, C, H, W) or (B, H, W)
        target: Ground truth masks (B, C, H, W) or (B, H, W)
        threshold: Threshold for binarizing predictions
        smooth: Smoothing factor
        
    Returns:
        IoU score
    """
    # Binarize predictions
    pred = (pred > threshold).float()
    
    # Flatten tensors
    pred = pred.view(-1)
    target = target.view(-1)
    
    # Compute intersection and union
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    
    return iou.item()


def compute_dice(pred: torch.Tensor,
                 target: torch.Tensor,
                 threshold: float = 0.5,
                 smooth: float = 1e-6) -> float:
    """
    Compute Dice coefficient.
    
    Args:
        pred: Predicted masks (B, C, H, W) or (B, H, W)
        target: Ground truth masks (B, C, H, W) or (B, H, W)
        threshold: Threshold for binarizing predictions
        smooth: Smoothing factor
        
    Returns:
        Dice score
    """
    # Binarize predictions
    pred = (pred > threshold).float()
    
    # Flatten tensors
    pred = pred.view(-1)
    target = target.view(-1)
    
    # Compute Dice
    intersection = (pred * target).sum()
    dice = (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return dice.item()


def compute_pixel_accuracy(pred: torch.Tensor,
                           target: torch.Tensor,
                           threshold: float = 0.5) -> float:
    """
    Compute pixel-wise accuracy.
    
    Args:
        pred: Predicted masks (B, C, H, W) or (B, H, W)
        target: Ground truth masks (B, C, H, W) or (B, H, W)
        threshold: Threshold for binarizing predictions
        
    Returns:
        Pixel accuracy
    """
    # Binarize predictions
    pred = (pred > threshold).float()
    
    # Flatten tensors
    pred = pred.view(-1)
    target = target.view(-1)
    
    # Compute accuracy
    correct = (pred == target).sum()
    total = target.numel()
    
    accuracy = correct.float() / total
    
    return accuracy.item()


def compute_precision_recall_f1(pred: torch.Tensor,
                                target: torch.Tensor,
                                threshold: float = 0.5,
                                smooth: float = 1e-6) -> Tuple[float, float, float]:
    """
    Compute precision, recall, and F1 score.
    
    Args:
        pred: Predicted masks (B, C, H, W) or (B, H, W)
        target: Ground truth masks (B, C, H, W) or (B, H, W)
        threshold: Threshold for binarizing predictions
        smooth: Smoothing factor
        
    Returns:
        (precision, recall, f1) tuple
    """
    # Binarize predictions
    pred = (pred > threshold).float()
    
    # Flatten tensors
    pred = pred.view(-1)
    target = target.view(-1)
    
    # Compute TP, FP, FN
    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    fn = ((1 - pred) * target).sum()
    
    # Compute metrics
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    f1 = 2 * precision * recall / (precision + recall + smooth)
    
    return precision.item(), recall.item(), f1.item()


def compute_all_metrics(pred: torch.Tensor,
                       target: torch.Tensor,
                       threshold: float = 0.5) -> dict:
    """
    Compute all evaluation metrics.
    
    Args:
        pred: Predicted masks
        target: Ground truth masks
        threshold: Threshold for binarizing predictions
        
    Returns:
        Dictionary with all metrics
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
    Tracker for accumulating metrics over multiple batches.
    """
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
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
        Update metrics with new batch.
        
        Args:
            pred: Predicted masks
            target: Ground truth masks
            threshold: Threshold for binarizing predictions
        """
        metrics = compute_all_metrics(pred, target, threshold)
        
        for key, value in metrics.items():
            self.metrics[key].append(value)
    
    def get_average(self) -> dict:
        """
        Get average of all metrics.
        
        Returns:
            Dictionary with average metrics
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
        Get standard deviation of all metrics.
        
        Returns:
            Dictionary with std of metrics
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
        Get summary string of metrics.
        
        Returns:
            Formatted summary string
        """
        avg_metrics = self.get_average()
        
        summary = "Metrics Summary:\n"
        summary += f"  IoU:       {avg_metrics['iou']:.4f}\n"
        summary += f"  Dice:      {avg_metrics['dice']:.4f}\n"
        summary += f"  Accuracy:  {avg_metrics['accuracy']:.4f}\n"
        summary += f"  Precision: {avg_metrics['precision']:.4f}\n"
        summary += f"  Recall:    {avg_metrics['recall']:.4f}\n"
        summary += f"  F1:        {avg_metrics['f1']:.4f}"
        
        return summary
