"""
Model trainer for segmentation models.

This module provides:
- Training loop with checkpointing
- Validation
- Learning rate scheduling
- Early stopping
- Metrics tracking
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
    Trainer for segmentation models.
    
    Handles:
    - Training loop
    - Validation
    - Checkpointing
    - Early stopping
    - Metrics tracking
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
        Initialize trainer.
        
        Args:
            model: PyTorch model
            optimizer: Optimizer
            criterion: Loss function
            device: Device for training
            checkpoint_dir: Directory to save checkpoints
            scheduler: Optional learning rate scheduler
            early_stopping_patience: Patience for early stopping
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.scheduler = scheduler
        self.early_stopping_patience = early_stopping_patience
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_metric = 0.0
        self.epochs_without_improvement = 0
        
        # History
        self.train_losses = []
        self.val_losses = []
        self.val_metrics_history = []
        
        logger.info(f"Trainer initialized: device={device}, checkpoint_dir={checkpoint_dir}")
    
    def train_epoch(self,
                   train_loader: DataLoader,
                   epoch: int,
                   progress_callback: Optional[Callable] = None) -> Dict:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            progress_callback: Optional callback(current, total, loss, metrics)
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        
        running_loss = 0.0
        metrics_tracker = MetricsTracker()
        
        num_batches = len(train_loader)
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            # Move to device
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # Compute loss
            loss = self.criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            running_loss += loss.item()
            
            with torch.no_grad():
                # Apply sigmoid if needed
                if outputs.shape[1] == 1:  # Binary segmentation
                    outputs = torch.sigmoid(outputs)
                
                metrics_tracker.update(outputs, masks)
            
            # Progress callback
            if progress_callback is not None:
                avg_loss = running_loss / (batch_idx + 1)
                current_metrics = metrics_tracker.get_average()
                progress_callback(batch_idx + 1, num_batches, avg_loss, current_metrics)
        
        # Calculate average loss
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
        Validate model.
        
        Args:
            val_loader: Validation data loader
            progress_callback: Optional callback(current, total, loss, metrics)
            
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        
        running_loss = 0.0
        metrics_tracker = MetricsTracker()
        
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(val_loader):
                # Move to device
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Compute loss
                loss = self.criterion(outputs, masks)
                
                # Track metrics
                running_loss += loss.item()
                
                # Apply sigmoid if needed
                if outputs.shape[1] == 1:  # Binary segmentation
                    outputs = torch.sigmoid(outputs)
                
                metrics_tracker.update(outputs, masks)
                
                # Progress callback
                if progress_callback is not None:
                    avg_loss = running_loss / (batch_idx + 1)
                    current_metrics = metrics_tracker.get_average()
                    progress_callback(batch_idx + 1, num_batches, avg_loss, current_metrics)
        
        # Calculate average loss
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
        Train model for multiple epochs.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            epoch_callback: Optional callback after each epoch(epoch, train_results, val_results)
            batch_callback: Optional callback after each batch(current, total, loss, metrics)
            
        Returns:
            Dictionary with training history
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch + 1
            
            logger.info(f"Epoch {self.current_epoch}/{num_epochs}")
            
            # Train
            train_results = self.train_epoch(train_loader, epoch, batch_callback)
            logger.info(f"Train loss: {train_results['loss']:.4f}, "
                       f"IoU: {train_results['metrics']['iou']:.4f}")
            
            # Validate
            val_results = self.validate(val_loader)
            logger.info(f"Val loss: {val_results['loss']:.4f}, "
                       f"IoU: {val_results['metrics']['iou']:.4f}")
            
            # Update history
            self.train_losses.append(train_results['loss'])
            self.val_losses.append(val_results['loss'])
            self.val_metrics_history.append(val_results['metrics'])
            
            # Learning rate scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_results['loss'])
                else:
                    self.scheduler.step()
                
                current_lr = self.optimizer.param_groups[0]['lr']
                logger.info(f"Learning rate: {current_lr:.6f}")
            
            # Save checkpoint
            is_best = val_results['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_results['loss']
                self.best_val_metric = val_results['metrics']['iou']
                self.epochs_without_improvement = 0
                self.save_checkpoint(is_best=True)
                logger.info("✓ Best model saved")
            else:
                self.epochs_without_improvement += 1
                self.save_checkpoint(is_best=False)
            
            # Early stopping
            if self.epochs_without_improvement >= self.early_stopping_patience:
                logger.info(f"Early stopping triggered after {self.current_epoch} epochs")
                break
            
            # Epoch callback
            if epoch_callback is not None:
                epoch_callback(self.current_epoch, train_results, val_results)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Training completed in {elapsed_time:.2f} seconds")
        logger.info(f"Best val loss: {self.best_val_loss:.4f}")
        logger.info(f"Best val IoU: {self.best_val_metric:.4f}")
        
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
        Save model checkpoint.
        
        Args:
            is_best: Whether this is the best model so far
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
        
        # Save last checkpoint
        last_path = self.checkpoint_dir / "last_checkpoint.pth"
        torch.save(checkpoint, last_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
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
        
        logger.info(f"Checkpoint loaded from: {checkpoint_path}")
        logger.info(f"Resuming from epoch {self.current_epoch}")
    
    def get_history(self) -> Dict:
        """
        Get training history.
        
        Returns:
            Dictionary with training history
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
    Create optimizer.
    
    Args:
        model: PyTorch model
        optimizer_name: Optimizer name ('adam', 'sgd', 'adamw')
        lr: Learning rate
        weight_decay: Weight decay
        
    Returns:
        Optimizer
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
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    logger.info(f"Created optimizer: {optimizer_name}, lr={lr}")
    return optimizer


def create_scheduler(optimizer: Optimizer,
                    scheduler_name: str = 'reduce_on_plateau',
                    **kwargs) -> Optional[_LRScheduler]:
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: Optimizer
        scheduler_name: Scheduler name ('reduce_on_plateau', 'cosine', 'step', None)
        **kwargs: Additional scheduler parameters
        
    Returns:
        Scheduler or None
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
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")
    
    logger.info(f"Created scheduler: {scheduler_name}")
    return scheduler
