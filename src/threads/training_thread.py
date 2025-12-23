"""
Training thread for asynchronous model training.

This module provides:
- Asynchronous training in a separate thread
- Progress reporting
- Live metrics updates
"""

from typing import Dict, List
import torch
from PyQt5.QtCore import QThread, pyqtSignal

from src.logger import get_logger
from src.core.model_trainer import ModelTrainer
from src.core.segmentation_dataset import create_dataloaders, load_dataset_from_split_files
from src.models.segmentation_models import build_model
from src.models.losses import get_loss_function
from src.core.model_trainer import create_optimizer, create_scheduler

logger = get_logger(__name__)


class TrainingThread(QThread):
    """
    Thread for running model training asynchronously.
    
    Signals:
        epoch_started: Emitted when epoch starts (epoch, total_epochs)
        epoch_completed: Emitted when epoch completes (epoch, train_loss, val_loss, val_metrics)
        batch_progress: Emitted during batch processing (current, total, phase, loss, metrics)
        training_completed: Emitted when training completes (history)
        training_failed: Emitted when training fails (error_message)
    """
    
    epoch_started = pyqtSignal(int, int)  # current_epoch, total_epochs
    epoch_completed = pyqtSignal(int, float, float, dict)  # epoch, train_loss, val_loss, val_metrics
    batch_progress = pyqtSignal(int, int, str, float, dict)  # current, total, phase, loss, metrics
    training_completed = pyqtSignal(dict)  # history
    training_failed = pyqtSignal(str)  # error message
    
    def __init__(self, config: Dict):
        """
        Initialize training thread.
        
        Args:
            config: Training configuration dictionary
        """
        super().__init__()
        
        self.config = config
        self._is_running = True
        
        logger.info("Training thread initialized")
    
    def run(self):
        """Execute training."""
        try:
            # Setup device
            device = torch.device(
                self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
            )
            logger.info(f"Training device: {device}")
            
            # Load dataset
            self.batch_progress.emit(0, 100, "Loading dataset...", 0.0, {})
            
            train_img, train_mask, val_img, val_mask = load_dataset_from_split_files(
                split_dir=self.config['split_dir'],
                images_dir=self.config['images_dir'],
                masks_dir=self.config['masks_dir']
            )
            
            if not train_img or not val_img:
                self.training_failed.emit("No training/validation data found")
                return
            
            logger.info(f"Dataset loaded: train={len(train_img)}, val={len(val_img)}")
            
            # Create dataloaders
            train_loader, val_loader = create_dataloaders(
                train_image_paths=train_img,
                train_mask_paths=train_mask,
                val_image_paths=val_img,
                val_mask_paths=val_mask,
                batch_size=self.config.get('batch_size', 8),
                num_workers=self.config.get('num_workers', 4),
                image_size=(self.config.get('image_height', 512), 
                           self.config.get('image_width', 512)),
                augmentation_prob=self.config.get('augmentation_prob', 0.5)
            )
            
            # Build model
            self.batch_progress.emit(20, 100, "Building model...", 0.0, {})
            
            model = build_model(
                architecture=self.config.get('architecture', 'unet'),
                encoder_name=self.config.get('encoder', 'resnet34'),
                encoder_weights=self.config.get('encoder_weights', 'imagenet'),
                in_channels=3,
                num_classes=1,
                activation='sigmoid'
            )
            model = model.to(device)
            
            # Get loss function
            criterion = get_loss_function(
                loss_name=self.config.get('loss', 'combined'),
                dice_weight=self.config.get('dice_weight', 0.5),
                bce_weight=self.config.get('bce_weight', 0.5)
            )
            
            # Create optimizer
            optimizer = create_optimizer(
                model=model,
                optimizer_name=self.config.get('optimizer', 'adam'),
                lr=self.config.get('learning_rate', 1e-4),
                weight_decay=self.config.get('weight_decay', 1e-5)
            )
            
            # Create scheduler
            scheduler = create_scheduler(
                optimizer=optimizer,
                scheduler_name=self.config.get('scheduler', 'reduce_on_plateau'),
                patience=self.config.get('scheduler_patience', 5)
            )
            
            # Create trainer
            trainer = ModelTrainer(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                checkpoint_dir=self.config['checkpoint_dir'],
                scheduler=scheduler,
                early_stopping_patience=self.config.get('early_stopping_patience', 10)
            )
            
            # Training callbacks
            def epoch_callback(epoch, train_results, val_results):
                if not self._is_running:
                    return
                
                self.epoch_completed.emit(
                    epoch,
                    train_results['loss'],
                    val_results['loss'],
                    val_results['metrics']
                )
            
            def batch_callback(current, total, loss, metrics):
                if not self._is_running:
                    return
                
                phase = "Training" if model.training else "Validation"
                self.batch_progress.emit(current, total, phase, loss, metrics)
            
            # Train
            num_epochs = self.config.get('num_epochs', 50)
            
            history = trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=num_epochs,
                epoch_callback=epoch_callback,
                batch_callback=batch_callback
            )
            
            if self._is_running:
                self.training_completed.emit(history)
            
        except Exception as e:
            logger.error(f"Training error: {e}", exc_info=True)
            self.training_failed.emit(str(e))
    
    def stop(self):
        """Stop training."""
        self._is_running = False
        logger.info("Training stop requested")
