"""
用于异步模型训练的训练线程。

此模块提供：
- 在单独线程中进行异步训练
- 进度报告
- 实时指标更新
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
    用于异步运行模型训练的线程。
    
    信号:
        epoch_started: 在 epoch 开始时发出（当前 epoch, 总 epoch 数）
        epoch_completed: 在 epoch 完成时发出（epoch, 训练损失, 验证损失, 验证指标）
        batch_progress: 在批次处理期间发出（当前, 总计, 阶段, 损失, 指标）
        training_completed: 在训练完成时发出（历史记录）
        training_failed: 在训练失败时发出（错误消息）
    """
    
    epoch_started = pyqtSignal(int, int)  # 当前 epoch, 总 epoch 数
    epoch_completed = pyqtSignal(int, float, float, dict)  # epoch, 训练损失, 验证损失, 验证指标
    batch_progress = pyqtSignal(int, int, str, float, dict)  # 当前, 总计, 阶段, 损失, 指标
    training_completed = pyqtSignal(dict)  # 历史记录
    training_failed = pyqtSignal(str)  # 错误消息
    
    def __init__(self, config: Dict):
        """
        初始化训练线程。
        
        参数:
            config: 训练配置字典
        """
        super().__init__()
        
        self.config = config
        self._is_running = True
        
        logger.info("训练线程已初始化")
    
    def run(self):
        """执行训练。"""
        try:
            # 设置设备
            device = torch.device(
                self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
            )
            logger.info(f"训练设备: {device}")
            
            # 加载数据集
            self.batch_progress.emit(0, 100, "正在加载数据集...", 0.0, {})
            
            train_img, train_mask, val_img, val_mask = load_dataset_from_split_files(
                split_dir=self.config['split_dir'],
                images_dir=self.config['images_dir'],
                masks_dir=self.config['masks_dir']
            )
            
            if not train_img or not val_img:
                self.training_failed.emit("未找到训练/验证数据")
                return
            
            logger.info(f"数据集已加载: 训练集={len(train_img)}, 验证集={len(val_img)}")
            
            # 创建数据加载器
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
            
            # 构建模型
            self.batch_progress.emit(20, 100, "正在构建模型...", 0.0, {})
            
            model = build_model(
                architecture=self.config.get('architecture', 'unet'),
                encoder_name=self.config.get('encoder', 'resnet34'),
                encoder_weights=self.config.get('encoder_weights', 'imagenet'),
                in_channels=3,
                num_classes=1,
                activation='sigmoid'
            )
            model = model.to(device)
            
            # 获取损失函数
            criterion = get_loss_function(
                loss_name=self.config.get('loss', 'combined'),
                dice_weight=self.config.get('dice_weight', 0.5),
                bce_weight=self.config.get('bce_weight', 0.5)
            )
            
            # 创建优化器
            optimizer = create_optimizer(
                model=model,
                optimizer_name=self.config.get('optimizer', 'adam'),
                lr=self.config.get('learning_rate', 1e-4),
                weight_decay=self.config.get('weight_decay', 1e-5)
            )
            
            # 创建调度器
            scheduler = create_scheduler(
                optimizer=optimizer,
                scheduler_name=self.config.get('scheduler', 'reduce_on_plateau'),
                patience=self.config.get('scheduler_patience', 5)
            )
            
            # 创建训练器
            trainer = ModelTrainer(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                checkpoint_dir=self.config['checkpoint_dir'],
                scheduler=scheduler,
                early_stopping_patience=self.config.get('early_stopping_patience', 10)
            )
            
            # 训练回调
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
                
                phase = "训练" if model.training else "验证"
                self.batch_progress.emit(current, total, phase, loss, metrics)
            
            # 训练
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
            logger.error(f"训练错误: {e}", exc_info=True)
            self.training_failed.emit(str(e))
    
    def stop(self):
        """停止训练。"""
        self._is_running = False
        logger.info("已请求停止训练")
