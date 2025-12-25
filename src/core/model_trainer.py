"""
分割模型的模型训练器。

此模块提供：
- 带有检查点保存的训练循环
- 验证
- 学习率调度
- 早停机制
- 指标跟踪
"""

import os
from pathlib import Path
from typing import Optional, Dict, Callable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import time

from src.logger import get_logger
from src.models.metrics import MetricsTracker, compute_all_metrics

logger = get_logger(__name__)


class ModelTrainer:
    """
    分割模型训练器。
    
    处理：
    - 训练循环
    - 验证
    - 检查点保存
    - 早停机制
    - 指标跟踪
    """
    
    def __init__(self,
                 model: nn.Module,
                 optimizer: Optimizer,
                 criterion: nn.Module,
                 device: torch.device,
                 checkpoint_dir: str,
                 scheduler: Optional[_LRScheduler] = None,
                 early_stopping_patience: int = 10):
        """
        初始化训练器。
        
        参数:
            model: PyTorch 模型
            optimizer: 优化器
            criterion: 损失函数
            device: 训练设备
            checkpoint_dir: 保存检查点的目录
            scheduler: 可选的学习率调度器
            early_stopping_patience: 早停机制的耐心值（epoch 数）
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.scheduler = scheduler
        self.early_stopping_patience = early_stopping_patience
        
        # 创建检查点目录
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 训练状态
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_metric = 0.0
        self.epochs_without_improvement = 0
        
        # 历史记录
        self.train_losses = []
        self.val_losses = []
        self.val_metrics_history = []
        
        logger.info(f"训练器已初始化: 设备={device}, 检查点目录={checkpoint_dir}")
    
    def train_epoch(self,
                   train_loader: DataLoader,
                   epoch: int,
                   progress_callback: Optional[Callable] = None) -> Dict:
        """
        训练一个 epoch。
        
        参数:
            train_loader: 训练数据加载器
            epoch: 当前 epoch 编号
            progress_callback: 可选的回调函数 (current, total, loss, metrics)
            
        返回:
            包含训练指标的字典
        """
        self.model.train()
        
        running_loss = 0.0
        metrics_tracker = MetricsTracker()
        
        num_batches = len(train_loader)
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            # 移动到设备
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # 计算损失
            loss = self.criterion(outputs, masks)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 跟踪指标
            running_loss += loss.item()
            
            with torch.no_grad():
                # 如果需要，应用 sigmoid
                if outputs.shape[1] == 1:  # 二值分割
                    outputs = torch.sigmoid(outputs)
                
                metrics_tracker.update(outputs, masks)
            
            # 进度回调
            if progress_callback is not None:
                avg_loss = running_loss / (batch_idx + 1)
                current_metrics = metrics_tracker.get_average()
                progress_callback(batch_idx + 1, num_batches, avg_loss, current_metrics)
        
        # 计算平均损失
        avg_loss = running_loss / num_batches
        avg_metrics = metrics_tracker.get_average()
        
        return {
            'loss': avg_loss,
            'metrics': avg_metrics
        }
    
    def validate(self,
                val_loader: DataLoader,
                progress_callback: Optional[Callable] = None) -> Dict:
        """
        验证模型。
        
        参数:
            val_loader: 验证数据加载器
            progress_callback: 可选的回调函数 (current, total, loss, metrics)
            
        返回:
            包含验证指标的字典
        """
        self.model.eval()
        
        running_loss = 0.0
        metrics_tracker = MetricsTracker()
        
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(val_loader):
                # 移动到设备
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # 前向传播
                outputs = self.model(images)
                
                # 计算损失
                loss = self.criterion(outputs, masks)
                
                # 跟踪指标
                running_loss += loss.item()
                
                # 如果需要，应用 sigmoid
                if outputs.shape[1] == 1:  # 二值分割
                    outputs = torch.sigmoid(outputs)
                
                metrics_tracker.update(outputs, masks)
                
                # 进度回调
                if progress_callback is not None:
                    avg_loss = running_loss / (batch_idx + 1)
                    current_metrics = metrics_tracker.get_average()
                    progress_callback(batch_idx + 1, num_batches, avg_loss, current_metrics)
        
        # 计算平均损失
        avg_loss = running_loss / num_batches
        avg_metrics = metrics_tracker.get_average()
        
        return {
            'loss': avg_loss,
            'metrics': avg_metrics
        }
    
    def train(self,
             train_loader: DataLoader,
             val_loader: DataLoader,
             num_epochs: int,
             epoch_callback: Optional[Callable] = None,
             batch_callback: Optional[Callable] = None) -> Dict:
        """
        训练模型多个 epoch。
        
        参数:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            num_epochs: 训练的 epoch 数
            epoch_callback: 每个 epoch 后的可选回调 (epoch, train_results, val_results)
            batch_callback: 每个 batch 后的可选回调 (current, total, loss, metrics)
            
        返回:
            包含训练历史记录的字典
        """
        logger.info(f"开始训练，共 {num_epochs} 个 epoch")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch + 1
            
            logger.info(f"Epoch {self.current_epoch}/{num_epochs}")
            
            # 训练
            train_results = self.train_epoch(train_loader, epoch, batch_callback)
            logger.info(f"训练损失: {train_results['loss']:.4f}, "
                       f"IoU: {train_results['metrics']['iou']:.4f}")
            
            # 验证
            val_results = self.validate(val_loader)
            logger.info(f"验证损失: {val_results['loss']:.4f}, "
                       f"IoU: {val_results['metrics']['iou']:.4f}")
            
            # 更新历史记录
            self.train_losses.append(train_results['loss'])
            self.val_losses.append(val_results['loss'])
            self.val_metrics_history.append(val_results['metrics'])
            
            # 学习率调度器
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_results['loss'])
                else:
                    self.scheduler.step()
                
                current_lr = self.optimizer.param_groups[0]['lr']
                logger.info(f"当前学习率: {current_lr:.6f}")
            
            # 保存检查点
            is_best = val_results['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_results['loss']
                self.best_val_metric = val_results['metrics']['iou']
                self.epochs_without_improvement = 0
                self.save_checkpoint(is_best=True)
                logger.info("✓ 已保存最佳模型")
            else:
                self.epochs_without_improvement += 1
                self.save_checkpoint(is_best=False)
            
            # 早停机制
            if self.epochs_without_improvement >= self.early_stopping_patience:
                logger.info(f"在 {self.current_epoch} 个 epoch 后触发早停")
                break
            
            # Epoch 回调
            if epoch_callback is not None:
                epoch_callback(self.current_epoch, train_results, val_results)
        
        elapsed_time = time.time() - start_time
        logger.info(f"训练完成，耗时 {elapsed_time:.2f} 秒")
        logger.info(f"最佳验证损失: {self.best_val_loss:.4f}")
        logger.info(f"最佳验证 IoU: {self.best_val_metric:.4f}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_metrics': self.val_metrics_history,
            'best_val_loss': self.best_val_loss,
            'best_val_metric': self.best_val_metric,
            'elapsed_time': elapsed_time
        }
    
    def save_checkpoint(self, is_best: bool = False):
        """
        保存模型检查点。
        
        参数:
            is_best: 这是否是迄今为止最好的模型
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_metric': self.best_val_metric,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_metrics': self.val_metrics_history
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # 保存最后的检查点
        last_path = self.checkpoint_dir / "last_checkpoint.pth"
        torch.save(checkpoint, last_path)
        
        # 保存最佳检查点
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        加载模型检查点。
        
        参数:
            checkpoint_path: 检查点文件路径
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_val_metric = checkpoint.get('best_val_metric', 0.0)
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.val_metrics_history = checkpoint.get('val_metrics', [])
        
        logger.info(f"已从 {checkpoint_path} 加载检查点")
        logger.info(f"从第 {self.current_epoch} 个 epoch 恢复训练")
    
    def get_history(self) -> Dict:
        """
        获取训练历史记录。
        
        返回:
            包含训练历史记录的字典
        """
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_metrics': self.val_metrics_history,
            'best_val_loss': self.best_val_loss,
            'best_val_metric': self.best_val_metric
        }


def create_optimizer(model: nn.Module,
                    optimizer_name: str = 'adam',
                    lr: float = 1e-4,
                    weight_decay: float = 1e-5) -> Optimizer:
    """
    创建优化器。
    
    参数:
        model: PyTorch 模型
        optimizer_name: 优化器名称 ('adam', 'sgd', 'adamw')
        lr: 学习率
        weight_decay: 权重衰减
        
    返回:
        优化器
    """
    optimizer_name = optimizer_name.lower()
    
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    elif optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"不支持的优化器: {optimizer_name}")
    
    logger.info(f"已创建优化器: {optimizer_name}, lr={lr}")
    return optimizer


def create_scheduler(optimizer: Optimizer,
                    scheduler_name: str = 'reduce_on_plateau',
                    **kwargs) -> Optional[_LRScheduler]:
    """
    创建学习率调度器。
    
    参数:
        optimizer: 优化器
        scheduler_name: 调度器名称 ('reduce_on_plateau', 'cosine', 'step', None)
        **kwargs: 额外的调度器参数
        
    返回:
        调度器或 None
    """
    if scheduler_name is None or scheduler_name.lower() == 'none':
        return None
    
    scheduler_name = scheduler_name.lower()
    
    if scheduler_name == 'reduce_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=kwargs.get('factor', 0.5),
            patience=kwargs.get('patience', 5),
            verbose=True
        )
    elif scheduler_name == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get('T_max', 50),
            eta_min=kwargs.get('eta_min', 1e-6)
        )
    elif scheduler_name == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs.get('step_size', 10),
            gamma=kwargs.get('gamma', 0.1)
        )
    else:
        raise ValueError(f"不支持的调度器: {scheduler_name}")
    
    logger.info(f"已创建调度器: {scheduler_name}")
    return scheduler
