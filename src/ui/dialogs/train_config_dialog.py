"""
Training configuration dialog.

This module provides:
- Model configuration UI
- Hyperparameter settings
- Training options
- Live training monitoring
"""

from pathlib import Path
from typing import Optional, Dict
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QTabWidget,
    QLabel, QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit,
    QPushButton, QCheckBox, QGroupBox, QProgressBar, QTextEdit,
    QFileDialog, QMessageBox
)
from PyQt5.QtGui import QFont
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from src.logger import get_logger
from src.threads.training_thread import TrainingThread

logger = get_logger(__name__)


class MetricsCanvas(FigureCanvasQTAgg):
    """Canvas for plotting training metrics."""
    
    def __init__(self, parent=None, width=8, height=6, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes_loss = fig.add_subplot(211)
        self.axes_iou = fig.add_subplot(212)
        super().__init__(fig)
        
        self.train_losses = []
        self.val_losses = []
        self.val_ious = []
        
        # Initial plot setup
        self.axes_loss.set_title('Loss')
        self.axes_loss.set_xlabel('Epoch')
        self.axes_loss.set_ylabel('Loss')
        self.axes_loss.grid(True, alpha=0.3)
        
        self.axes_iou.set_title('IoU Score')
        self.axes_iou.set_xlabel('Epoch')
        self.axes_iou.set_ylabel('IoU')
        self.axes_iou.grid(True, alpha=0.3)
        
        fig.tight_layout()
    
    def update_plot(self, train_loss: float, val_loss: float, val_iou: float):
        """Update plots with new data."""
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.val_ious.append(val_iou)
        
        epochs = list(range(1, len(self.train_losses) + 1))
        
        # Update loss plot
        self.axes_loss.clear()
        self.axes_loss.plot(epochs, self.train_losses, 'b-', label='Train Loss', linewidth=2)
        self.axes_loss.plot(epochs, self.val_losses, 'r-', label='Val Loss', linewidth=2)
        self.axes_loss.set_title('Loss')
        self.axes_loss.set_xlabel('Epoch')
        self.axes_loss.set_ylabel('Loss')
        self.axes_loss.legend()
        self.axes_loss.grid(True, alpha=0.3)
        
        # Update IoU plot
        self.axes_iou.clear()
        self.axes_iou.plot(epochs, self.val_ious, 'g-', label='Val IoU', linewidth=2)
        self.axes_iou.set_title('IoU Score')
        self.axes_iou.set_xlabel('Epoch')
        self.axes_iou.set_ylabel('IoU')
        self.axes_iou.legend()
        self.axes_iou.grid(True, alpha=0.3)
        
        self.draw()


class TrainConfigDialog(QDialog):
    """
    Dialog for configuring and running model training.
    """
    
    def __init__(self, config: Dict, paths_config: Dict, parent=None):
        """
        Initialize training dialog.
        
        Args:
            config: Application configuration
            paths_config: Paths configuration
            parent: Parent widget
        """
        super().__init__(parent)
        
        self.config = config
        self.paths_config = paths_config
        self.training_thread: Optional[TrainingThread] = None
        
        self.setWindowTitle("Train Segmentation Model")
        self.setMinimumSize(1000, 800)
        
        self._init_ui()
        self._load_defaults()
        
        logger.info("Training dialog opened")
    
    def _init_ui(self):
        """Initialize UI components."""
        layout = QVBoxLayout(self)
        
        # Create tabs
        tabs = QTabWidget()
        
        # Tab 1: Model Configuration
        model_tab = self._create_model_tab()
        tabs.addTab(model_tab, "Model")
        
        # Tab 2: Training Configuration
        training_tab = self._create_training_tab()
        tabs.addTab(training_tab, "Training")
        
        # Tab 3: Data Configuration
        data_tab = self._create_data_tab()
        tabs.addTab(data_tab, "Data")
        
        # Tab 4: Monitoring
        monitoring_tab = self._create_monitoring_tab()
        tabs.addTab(monitoring_tab, "Monitor")
        
        layout.addWidget(tabs)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.start_btn = QPushButton("Start Training")
        self.start_btn.setDefault(True)
        self.start_btn.clicked.connect(self._start_training)
        button_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_training)
        button_layout.addWidget(self.stop_btn)
        
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.close_btn)
        
        layout.addLayout(button_layout)
    
    def _create_model_tab(self) -> QWidget:
        """Create model configuration tab."""
        from PyQt5.QtWidgets import QWidget
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Model Architecture
        arch_group = QGroupBox("Model Architecture")
        arch_layout = QGridLayout(arch_group)
        
        arch_layout.addWidget(QLabel("Architecture:"), 0, 0)
        self.arch_combo = QComboBox()
        self.arch_combo.addItems(["U-Net", "DeepLabV3+", "FPN"])
        arch_layout.addWidget(self.arch_combo, 0, 1)
        
        arch_layout.addWidget(QLabel("Encoder:"), 1, 0)
        self.encoder_combo = QComboBox()
        self.encoder_combo.addItems([
            "resnet18", "resnet34", "resnet50", "resnet101",
            "efficientnet-b0", "efficientnet-b1", "efficientnet-b2",
            "mobilenet_v2"
        ])
        arch_layout.addWidget(self.encoder_combo, 1, 1)
        
        arch_layout.addWidget(QLabel("Pretrained:"), 2, 0)
        self.pretrained_check = QCheckBox("Use ImageNet weights")
        self.pretrained_check.setChecked(True)
        arch_layout.addWidget(self.pretrained_check, 2, 1)
        
        layout.addWidget(arch_group)
        
        # Loss Function
        loss_group = QGroupBox("Loss Function")
        loss_layout = QGridLayout(loss_group)
        
        loss_layout.addWidget(QLabel("Loss Type:"), 0, 0)
        self.loss_combo = QComboBox()
        self.loss_combo.addItems(["Combined", "Dice", "BCE", "Focal", "IoU"])
        loss_layout.addWidget(self.loss_combo, 0, 1)
        
        loss_layout.addWidget(QLabel("Dice Weight:"), 1, 0)
        self.dice_weight_spin = QDoubleSpinBox()
        self.dice_weight_spin.setRange(0.0, 1.0)
        self.dice_weight_spin.setSingleStep(0.1)
        self.dice_weight_spin.setValue(0.5)
        loss_layout.addWidget(self.dice_weight_spin, 1, 1)
        
        loss_layout.addWidget(QLabel("BCE Weight:"), 2, 0)
        self.bce_weight_spin = QDoubleSpinBox()
        self.bce_weight_spin.setRange(0.0, 1.0)
        self.bce_weight_spin.setSingleStep(0.1)
        self.bce_weight_spin.setValue(0.5)
        loss_layout.addWidget(self.bce_weight_spin, 2, 1)
        
        layout.addWidget(loss_group)
        layout.addStretch()
        
        return widget
    
    def _create_training_tab(self) -> QWidget:
        """Create training configuration tab."""
        from PyQt5.QtWidgets import QWidget
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Hyperparameters
        hyper_group = QGroupBox("Hyperparameters")
        hyper_layout = QGridLayout(hyper_group)
        
        hyper_layout.addWidget(QLabel("Epochs:"), 0, 0)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(50)
        hyper_layout.addWidget(self.epochs_spin, 0, 1)
        
        hyper_layout.addWidget(QLabel("Batch Size:"), 1, 0)
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 128)
        self.batch_size_spin.setValue(8)
        hyper_layout.addWidget(self.batch_size_spin, 1, 1)
        
        hyper_layout.addWidget(QLabel("Learning Rate:"), 2, 0)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(1e-6, 1.0)
        self.lr_spin.setDecimals(6)
        self.lr_spin.setValue(0.0001)
        self.lr_spin.setSingleStep(0.00001)
        hyper_layout.addWidget(self.lr_spin, 2, 1)
        
        hyper_layout.addWidget(QLabel("Weight Decay:"), 3, 0)
        self.weight_decay_spin = QDoubleSpinBox()
        self.weight_decay_spin.setRange(0.0, 0.1)
        self.weight_decay_spin.setDecimals(6)
        self.weight_decay_spin.setValue(0.00001)
        self.weight_decay_spin.setSingleStep(0.00001)
        hyper_layout.addWidget(self.weight_decay_spin, 3, 1)
        
        layout.addWidget(hyper_group)
        
        # Optimizer & Scheduler
        optim_group = QGroupBox("Optimization")
        optim_layout = QGridLayout(optim_group)
        
        optim_layout.addWidget(QLabel("Optimizer:"), 0, 0)
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(["Adam", "AdamW", "SGD"])
        optim_layout.addWidget(self.optimizer_combo, 0, 1)
        
        optim_layout.addWidget(QLabel("Scheduler:"), 1, 0)
        self.scheduler_combo = QComboBox()
        self.scheduler_combo.addItems(["ReduceLROnPlateau", "CosineAnnealing", "StepLR", "None"])
        optim_layout.addWidget(self.scheduler_combo, 1, 1)
        
        optim_layout.addWidget(QLabel("Early Stopping Patience:"), 2, 0)
        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(1, 100)
        self.patience_spin.setValue(10)
        optim_layout.addWidget(self.patience_spin, 2, 1)
        
        layout.addWidget(optim_group)
        layout.addStretch()
        
        return widget
    
    def _create_data_tab(self) -> QWidget:
        """Create data configuration tab."""
        from PyQt5.QtWidgets import QWidget
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Paths
        paths_group = QGroupBox("Data Paths")
        paths_layout = QGridLayout(paths_group)
        
        paths_layout.addWidget(QLabel("Split Directory:"), 0, 0)
        self.split_dir_edit = QLineEdit()
        paths_layout.addWidget(self.split_dir_edit, 0, 1)
        browse_split_btn = QPushButton("Browse...")
        browse_split_btn.clicked.connect(lambda: self._browse_dir(self.split_dir_edit))
        paths_layout.addWidget(browse_split_btn, 0, 2)
        
        paths_layout.addWidget(QLabel("Images Directory:"), 1, 0)
        self.images_dir_edit = QLineEdit()
        paths_layout.addWidget(self.images_dir_edit, 1, 1)
        browse_img_btn = QPushButton("Browse...")
        browse_img_btn.clicked.connect(lambda: self._browse_dir(self.images_dir_edit))
        paths_layout.addWidget(browse_img_btn, 1, 2)
        
        paths_layout.addWidget(QLabel("Masks Directory:"), 2, 0)
        self.masks_dir_edit = QLineEdit()
        paths_layout.addWidget(self.masks_dir_edit, 2, 1)
        browse_mask_btn = QPushButton("Browse...")
        browse_mask_btn.clicked.connect(lambda: self._browse_dir(self.masks_dir_edit))
        paths_layout.addWidget(browse_mask_btn, 2, 2)
        
        paths_layout.addWidget(QLabel("Checkpoint Directory:"), 3, 0)
        self.checkpoint_dir_edit = QLineEdit()
        paths_layout.addWidget(self.checkpoint_dir_edit, 3, 1)
        browse_ckpt_btn = QPushButton("Browse...")
        browse_ckpt_btn.clicked.connect(lambda: self._browse_dir(self.checkpoint_dir_edit))
        paths_layout.addWidget(browse_ckpt_btn, 3, 2)
        
        layout.addWidget(paths_group)
        
        # Image Settings
        img_group = QGroupBox("Image Settings")
        img_layout = QGridLayout(img_group)
        
        img_layout.addWidget(QLabel("Image Size:"), 0, 0)
        size_layout = QHBoxLayout()
        self.img_width_spin = QSpinBox()
        self.img_width_spin.setRange(128, 2048)
        self.img_width_spin.setValue(512)
        size_layout.addWidget(self.img_width_spin)
        size_layout.addWidget(QLabel("×"))
        self.img_height_spin = QSpinBox()
        self.img_height_spin.setRange(128, 2048)
        self.img_height_spin.setValue(512)
        size_layout.addWidget(self.img_height_spin)
        img_layout.addLayout(size_layout, 0, 1)
        
        img_layout.addWidget(QLabel("Augmentation Prob:"), 1, 0)
        self.aug_prob_spin = QDoubleSpinBox()
        self.aug_prob_spin.setRange(0.0, 1.0)
        self.aug_prob_spin.setSingleStep(0.1)
        self.aug_prob_spin.setValue(0.5)
        img_layout.addWidget(self.aug_prob_spin, 1, 1)
        
        layout.addWidget(img_group)
        layout.addStretch()
        
        return widget
    
    def _create_monitoring_tab(self) -> QWidget:
        """Create monitoring tab."""
        from PyQt5.QtWidgets import QWidget
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Progress
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        self.progress_bar = QProgressBar()
        progress_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("Ready to train")
        font = QFont()
        font.setPointSize(10)
        self.status_label.setFont(font)
        progress_layout.addWidget(self.status_label)
        
        layout.addWidget(progress_group)
        
        # Metrics plot
        self.metrics_canvas = MetricsCanvas(self, width=8, height=6)
        layout.addWidget(self.metrics_canvas)
        
        # Log
        log_group = QGroupBox("Training Log")
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        log_layout.addWidget(self.log_text)
        
        layout.addWidget(log_group)
        
        return widget
    
    def _load_defaults(self):
        """Load default values from config."""
        from pathlib import Path
        
        # Set default paths
        project_root = Path(self.paths_config['paths']['data_root'])
        self.split_dir_edit.setText(str(project_root / "splits"))
        self.images_dir_edit.setText(str(project_root / "processed/images"))
        self.masks_dir_edit.setText(str(project_root / "processed/masks"))
        self.checkpoint_dir_edit.setText(str(project_root / "outputs/models"))
    
    def _browse_dir(self, line_edit: QLineEdit):
        """Browse for directory."""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Directory",
            line_edit.text(),
            QFileDialog.ShowDirsOnly
        )
        
        if dir_path:
            line_edit.setText(dir_path)
    
    def _start_training(self):
        """Start training."""
        # Validate inputs
        if not self.split_dir_edit.text():
            QMessageBox.warning(self, "Error", "Please specify split directory")
            return
        
        # Prepare configuration
        train_config = {
            'architecture': self.arch_combo.currentText().lower().replace('-', '').replace('+', 'plus'),
            'encoder': self.encoder_combo.currentText(),
            'encoder_weights': 'imagenet' if self.pretrained_check.isChecked() else None,
            'loss': self.loss_combo.currentText().lower(),
            'dice_weight': self.dice_weight_spin.value(),
            'bce_weight': self.bce_weight_spin.value(),
            'num_epochs': self.epochs_spin.value(),
            'batch_size': self.batch_size_spin.value(),
            'learning_rate': self.lr_spin.value(),
            'weight_decay': self.weight_decay_spin.value(),
            'optimizer': self.optimizer_combo.currentText().lower(),
            'scheduler': self.scheduler_combo.currentText().lower(),
            'early_stopping_patience': self.patience_spin.value(),
            'split_dir': self.split_dir_edit.text(),
            'images_dir': self.images_dir_edit.text(),
            'masks_dir': self.masks_dir_edit.text(),
            'checkpoint_dir': self.checkpoint_dir_edit.text(),
            'image_width': self.img_width_spin.value(),
            'image_height': self.img_height_spin.value(),
            'augmentation_prob': self.aug_prob_spin.value(),
            'num_workers': 4,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        
        # Create training thread
        self.training_thread = TrainingThread(train_config)
        
        # Connect signals
        self.training_thread.epoch_completed.connect(self._on_epoch_completed)
        self.training_thread.batch_progress.connect(self._on_batch_progress)
        self.training_thread.training_completed.connect(self._on_training_completed)
        self.training_thread.training_failed.connect(self._on_training_failed)
        
        # Update UI
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("Training started...")
        self.log_text.append("=== Training Started ===")
        
        # Start training
        self.training_thread.start()
        
        logger.info("Training started")
    
    def _stop_training(self):
        """Stop training."""
        if self.training_thread:
            self.training_thread.stop()
            self.stop_btn.setEnabled(False)
            self.status_label.setText("Stopping...")
            self.log_text.append("Stop requested...")
    
    def _on_epoch_completed(self, epoch: int, train_loss: float, val_loss: float, val_metrics: dict):
        """Handle epoch completion."""
        self.status_label.setText(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, IoU={val_metrics['iou']:.4f}")
        self.log_text.append(f"Epoch {epoch}: Train={train_loss:.4f}, Val={val_loss:.4f}, IoU={val_metrics['iou']:.4f}")
        
        # Update plot
        self.metrics_canvas.update_plot(train_loss, val_loss, val_metrics['iou'])
    
    def _on_batch_progress(self, current: int, total: int, phase: str, loss: float, metrics: dict):
        """Handle batch progress."""
        progress = int((current / total) * 100)
        self.progress_bar.setValue(progress)
    
    def _on_training_completed(self, history: dict):
        """Handle training completion."""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText(f"✓ Training completed! Best IoU: {history['best_val_metric']:.4f}")
        self.log_text.append(f"=== Training Completed ===")
        self.log_text.append(f"Best Val Loss: {history['best_val_loss']:.4f}")
        self.log_text.append(f"Best Val IoU: {history['best_val_metric']:.4f}")
        self.log_text.append(f"Total Time: {history['elapsed_time']:.2f}s")
        
        QMessageBox.information(
            self,
            "Training Complete",
            f"Training completed successfully!\n\n"
            f"Best Val Loss: {history['best_val_loss']:.4f}\n"
            f"Best Val IoU: {history['best_val_metric']:.4f}\n"
            f"Time: {history['elapsed_time']:.2f}s"
        )
    
    def _on_training_failed(self, error_message: str):
        """Handle training failure."""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("❌ Training failed")
        self.log_text.append(f"ERROR: {error_message}")
        
        QMessageBox.critical(
            self,
            "Training Failed",
            f"Training failed with error:\n\n{error_message}"
        )
    
    def closeEvent(self, event):
        """Handle dialog close."""
        if self.training_thread and self.training_thread.isRunning():
            reply = QMessageBox.question(
                self,
                "Training in Progress",
                "Training is still running. Do you want to stop it?",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.training_thread.stop()
                self.training_thread.wait()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


# Need to import torch for device check
import torch
