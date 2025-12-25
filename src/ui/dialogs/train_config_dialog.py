"""
训练配置对话框。

此模块提供：
- 模型配置 UI
- 超参数设置
- 训练选项
- 实时训练监控
"""

from pathlib import Path
from typing import Optional, Dict
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QTabWidget,
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
    """用于绘制训练指标的画布。"""
    
    def __init__(self, parent=None, width=8, height=6, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes_loss = fig.add_subplot(211)
        self.axes_iou = fig.add_subplot(212)
        super().__init__(fig)
        
        self.train_losses = []
        self.val_losses = []
        self.val_ious = []
        
        # 初始绘图设置
        self.axes_loss.set_title('损失 (Loss)')
        self.axes_loss.set_xlabel('轮次 (Epoch)')
        self.axes_loss.set_ylabel('损失')
        self.axes_loss.grid(True, alpha=0.3)
        
        self.axes_iou.set_title('IoU 分数')
        self.axes_iou.set_xlabel('轮次 (Epoch)')
        self.axes_iou.set_ylabel('IoU')
        self.axes_iou.grid(True, alpha=0.3)
        
        fig.tight_layout()
    
    def update_plot(self, train_loss: float, val_loss: float, val_iou: float):
        """使用新数据更新图表。"""
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.val_ious.append(val_iou)
        
        epochs = list(range(1, len(self.train_losses) + 1))
        
        # 更新损失图
        self.axes_loss.clear()
        self.axes_loss.plot(epochs, self.train_losses, 'b-', label='训练损失', linewidth=2)
        self.axes_loss.plot(epochs, self.val_losses, 'r-', label='验证损失', linewidth=2)
        self.axes_loss.set_title('损失 (Loss)')
        self.axes_loss.set_xlabel('轮次 (Epoch)')
        self.axes_loss.set_ylabel('损失')
        self.axes_loss.legend()
        self.axes_loss.grid(True, alpha=0.3)
        
        # 更新 IoU 图
        self.axes_iou.clear()
        self.axes_iou.plot(epochs, self.val_ious, 'g-', label='验证 IoU', linewidth=2)
        self.axes_iou.set_title('IoU 分数')
        self.axes_iou.set_xlabel('轮次 (Epoch)')
        self.axes_iou.set_ylabel('IoU')
        self.axes_iou.legend()
        self.axes_iou.grid(True, alpha=0.3)
        
        self.draw()


class TrainConfigDialog(QDialog):
    """
    用于配置和运行模型训练的对话框。
    """
    
    def __init__(self, config: Dict, paths_config: Dict, parent=None):
        """
        初始化训练对话框。
        
        参数:
            config: 应用程序配置
            paths_config: 路径配置
            parent: 父小部件
        """
        super().__init__(parent)
        
        self.config = config
        self.paths_config = paths_config
        self.training_thread: Optional[TrainingThread] = None
        
        self.setWindowTitle("训练分割模型")
        self.setMinimumSize(1000, 800)
        
        self._init_ui()
        self._load_defaults()
        
        logger.info("训练对话框已打开")
    
    def _init_ui(self):
        """初始化 UI 组件。"""
        layout = QVBoxLayout(self)
        
        # 创建选项卡
        tabs = QTabWidget()
        
        # 选项卡 1: 模型配置
        model_tab = self._create_model_tab()
        tabs.addTab(model_tab, "模型")
        
        # 选项卡 2: 训练配置
        training_tab = self._create_training_tab()
        tabs.addTab(training_tab, "训练")
        
        # 选项卡 3: 数据配置
        data_tab = self._create_data_tab()
        tabs.addTab(data_tab, "数据")
        
        # 选项卡 4: 监控
        monitoring_tab = self._create_monitoring_tab()
        tabs.addTab(monitoring_tab, "监控")
        
        layout.addWidget(tabs)
        
        # 按钮
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.start_btn = QPushButton("开始训练")
        self.start_btn.setDefault(True)
        self.start_btn.clicked.connect(self._start_training)
        button_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_training)
        button_layout.addWidget(self.stop_btn)
        
        self.close_btn = QPushButton("关闭")
        self.close_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.close_btn)
        
        layout.addLayout(button_layout)
    
    def _create_model_tab(self) -> QWidget:
        """创建模型配置选项卡。"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 模型架构
        arch_group = QGroupBox("模型架构")
        arch_layout = QGridLayout(arch_group)
        
        arch_layout.addWidget(QLabel("架构:"), 0, 0)
        self.arch_combo = QComboBox()
        self.arch_combo.addItems(["U-Net", "DeepLabV3+", "FPN"])
        arch_layout.addWidget(self.arch_combo, 0, 1)
        
        arch_layout.addWidget(QLabel("编码器 (Encoder):"), 1, 0)
        self.encoder_combo = QComboBox()
        self.encoder_combo.addItems([
            "resnet18", "resnet34", "resnet50", "resnet101",
            "efficientnet-b0", "efficientnet-b1", "efficientnet-b2",
            "mobilenet_v2"
        ])
        arch_layout.addWidget(self.encoder_combo, 1, 1)
        
        arch_layout.addWidget(QLabel("预训练:"), 2, 0)
        self.pretrained_check = QCheckBox("使用 ImageNet 权重")
        self.pretrained_check.setChecked(True)
        arch_layout.addWidget(self.pretrained_check, 2, 1)
        
        layout.addWidget(arch_group)
        
        # 损失函数
        loss_group = QGroupBox("损失函数")
        loss_layout = QGridLayout(loss_group)
        
        loss_layout.addWidget(QLabel("损失类型:"), 0, 0)
        self.loss_combo = QComboBox()
        self.loss_combo.addItems(["Combined", "Dice", "BCE", "Focal", "IoU"])
        loss_layout.addWidget(self.loss_combo, 0, 1)
        
        loss_layout.addWidget(QLabel("Dice 权重:"), 1, 0)
        self.dice_weight_spin = QDoubleSpinBox()
        self.dice_weight_spin.setRange(0.0, 1.0)
        self.dice_weight_spin.setSingleStep(0.1)
        self.dice_weight_spin.setValue(0.5)
        loss_layout.addWidget(self.dice_weight_spin, 1, 1)
        
        loss_layout.addWidget(QLabel("BCE 权重:"), 2, 0)
        self.bce_weight_spin = QDoubleSpinBox()
        self.bce_weight_spin.setRange(0.0, 1.0)
        self.bce_weight_spin.setSingleStep(0.1)
        self.bce_weight_spin.setValue(0.5)
        loss_layout.addWidget(self.bce_weight_spin, 2, 1)
        
        layout.addWidget(loss_group)
        layout.addStretch()
        
        return widget
    
    def _create_training_tab(self) -> QWidget:
        """创建训练配置选项卡。"""

        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 超参数
        hyper_group = QGroupBox("超参数")
        hyper_layout = QGridLayout(hyper_group)
        
        hyper_layout.addWidget(QLabel("轮次 (Epochs):"), 0, 0)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(50)
        hyper_layout.addWidget(self.epochs_spin, 0, 1)
        
        hyper_layout.addWidget(QLabel("批大小 (Batch Size):"), 1, 0)
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 128)
        self.batch_size_spin.setValue(8)
        hyper_layout.addWidget(self.batch_size_spin, 1, 1)
        
        hyper_layout.addWidget(QLabel("学习率 (Learning Rate):"), 2, 0)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(1e-6, 1.0)
        self.lr_spin.setDecimals(6)
        self.lr_spin.setValue(0.0001)
        self.lr_spin.setSingleStep(0.00001)
        hyper_layout.addWidget(self.lr_spin, 2, 1)
        
        hyper_layout.addWidget(QLabel("权重衰减 (Weight Decay):"), 3, 0)
        self.weight_decay_spin = QDoubleSpinBox()
        self.weight_decay_spin.setRange(0.0, 0.1)
        self.weight_decay_spin.setDecimals(6)
        self.weight_decay_spin.setValue(0.00001)
        self.weight_decay_spin.setSingleStep(0.00001)
        hyper_layout.addWidget(self.weight_decay_spin, 3, 1)
        
        layout.addWidget(hyper_group)
        
        # 优化器与调度器
        optim_group = QGroupBox("优化设置")
        optim_layout = QGridLayout(optim_group)
        
        optim_layout.addWidget(QLabel("优化器:"), 0, 0)
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(["Adam", "AdamW", "SGD"])
        optim_layout.addWidget(self.optimizer_combo, 0, 1)
        
        optim_layout.addWidget(QLabel("调度器:"), 1, 0)
        self.scheduler_combo = QComboBox()
        self.scheduler_combo.addItems(["ReduceLROnPlateau", "CosineAnnealing", "StepLR", "None"])
        optim_layout.addWidget(self.scheduler_combo, 1, 1)
        
        optim_layout.addWidget(QLabel("早停耐心值 (Patience):"), 2, 0)
        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(1, 100)
        self.patience_spin.setValue(10)
        optim_layout.addWidget(self.patience_spin, 2, 1)
        
        layout.addWidget(optim_group)
        layout.addStretch()
        
        return widget
    
    def _create_data_tab(self) -> QWidget:
        """创建数据配置选项卡。"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 路径
        paths_group = QGroupBox("数据路径")
        paths_layout = QGridLayout(paths_group)
        
        paths_layout.addWidget(QLabel("划分目录 (Split):"), 0, 0)
        self.split_dir_edit = QLineEdit()
        paths_layout.addWidget(self.split_dir_edit, 0, 1)
        browse_split_btn = QPushButton("浏览...")
        browse_split_btn.clicked.connect(lambda: self._browse_dir(self.split_dir_edit))
        paths_layout.addWidget(browse_split_btn, 0, 2)
        
        paths_layout.addWidget(QLabel("图像目录:"), 1, 0)
        self.images_dir_edit = QLineEdit()
        paths_layout.addWidget(self.images_dir_edit, 1, 1)
        browse_img_btn = QPushButton("浏览...")
        browse_img_btn.clicked.connect(lambda: self._browse_dir(self.images_dir_edit))
        paths_layout.addWidget(browse_img_btn, 1, 2)
        
        paths_layout.addWidget(QLabel("掩码目录:"), 2, 0)
        self.masks_dir_edit = QLineEdit()
        paths_layout.addWidget(self.masks_dir_edit, 2, 1)
        browse_mask_btn = QPushButton("浏览...")
        browse_mask_btn.clicked.connect(lambda: self._browse_dir(self.masks_dir_edit))
        paths_layout.addWidget(browse_mask_btn, 2, 2)
        
        paths_layout.addWidget(QLabel("检查点目录:"), 3, 0)
        self.checkpoint_dir_edit = QLineEdit()
        paths_layout.addWidget(self.checkpoint_dir_edit, 3, 1)
        browse_ckpt_btn = QPushButton("浏览...")
        browse_ckpt_btn.clicked.connect(lambda: self._browse_dir(self.checkpoint_dir_edit))
        paths_layout.addWidget(browse_ckpt_btn, 3, 2)
        
        layout.addWidget(paths_group)
        
        # 图像设置
        img_group = QGroupBox("图像设置")
        img_layout = QGridLayout(img_group)
        
        img_layout.addWidget(QLabel("图像尺寸:"), 0, 0)
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
        
        img_layout.addWidget(QLabel("数据增强概率:"), 1, 0)
        self.aug_prob_spin = QDoubleSpinBox()
        self.aug_prob_spin.setRange(0.0, 1.0)
        self.aug_prob_spin.setSingleStep(0.1)
        self.aug_prob_spin.setValue(0.5)
        img_layout.addWidget(self.aug_prob_spin, 1, 1)
        
        layout.addWidget(img_group)
        layout.addStretch()
        
        return widget
    
    def _create_monitoring_tab(self) -> QWidget:
        """创建监控选项卡。"""

        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 进度
        progress_group = QGroupBox("进度")
        progress_layout = QVBoxLayout(progress_group)
        
        self.progress_bar = QProgressBar()
        progress_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("准备训练")
        font = QFont()
        font.setPointSize(10)
        self.status_label.setFont(font)
        progress_layout.addWidget(self.status_label)
        
        layout.addWidget(progress_group)
        
        # 指标图表
        self.metrics_canvas = MetricsCanvas(self, width=8, height=6)
        layout.addWidget(self.metrics_canvas)
        
        # 日志
        log_group = QGroupBox("训练日志")
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        log_layout.addWidget(self.log_text)
        
        layout.addWidget(log_group)
        
        return widget
    
    def _load_defaults(self):
        """从配置加载默认值。"""
        from pathlib import Path
        
        # 设置默认路径
        project_root = Path(self.paths_config['paths']['data_root'])
        self.split_dir_edit.setText(str(project_root / "splits"))
        self.images_dir_edit.setText(str(project_root / "processed/images"))
        self.masks_dir_edit.setText(str(project_root / "processed/masks"))
        self.checkpoint_dir_edit.setText(str(project_root / "outputs/models"))
    
    def _browse_dir(self, line_edit: QLineEdit):
        """浏览目录。"""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "选择目录",
            line_edit.text(),
            QFileDialog.ShowDirsOnly
        )
        
        if dir_path:
            line_edit.setText(dir_path)
    
    def _start_training(self):
        """开始训练。"""
        # 验证输入
        if not self.split_dir_edit.text():
            QMessageBox.warning(self, "错误", "请指定划分目录")
            return
        
        # 准备配置
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
        
        # 创建训练线程
        self.training_thread = TrainingThread(train_config)
        
        # 连接信号
        self.training_thread.epoch_completed.connect(self._on_epoch_completed)
        self.training_thread.batch_progress.connect(self._on_batch_progress)
        self.training_thread.training_completed.connect(self._on_training_completed)
        self.training_thread.training_failed.connect(self._on_training_failed)
        
        # 更新 UI
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("训练已开始...")
        self.log_text.append("=== 训练已开始 ===")
        
        # 开始训练
        self.training_thread.start()
        
        logger.info("训练已开始")
    
    def _stop_training(self):
        """停止训练。"""
        if self.training_thread:
            self.training_thread.stop()
            self.stop_btn.setEnabled(False)
            self.status_label.setText("正在停止...")
            self.log_text.append("已请求停止...")
    
    def _on_epoch_completed(self, epoch: int, train_loss: float, val_loss: float, val_metrics: dict):
        """处理轮次完成。"""
        self.status_label.setText(f"轮次 {epoch}: 训练损失={train_loss:.4f}, 验证损失={val_loss:.4f}, IoU={val_metrics['iou']:.4f}")
        self.log_text.append(f"轮次 {epoch}: 训练={train_loss:.4f}, 验证={val_loss:.4f}, IoU={val_metrics['iou']:.4f}")
        
        # 更新图表
        self.metrics_canvas.update_plot(train_loss, val_loss, val_metrics['iou'])
    
    def _on_batch_progress(self, current: int, total: int, phase: str, loss: float, metrics: dict):
        """处理批次进度。"""
        progress = int((current / total) * 100)
        self.progress_bar.setValue(progress)
    
    def _on_training_completed(self, history: dict):
        """处理训练完成。"""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText(f"✓ 训练已完成！最佳 IoU: {history['best_val_metric']:.4f}")
        self.log_text.append(f"=== 训练已完成 ===")
        self.log_text.append(f"最佳验证损失: {history['best_val_loss']:.4f}")
        self.log_text.append(f"最佳验证 IoU: {history['best_val_metric']:.4f}")
        self.log_text.append(f"总耗时: {history['elapsed_time']:.2f}s")
        
        QMessageBox.information(
            self,
            "训练完成",
            f"训练已成功完成！\n\n"
            f"最佳验证损失: {history['best_val_loss']:.4f}\n"
            f"最佳验证 IoU: {history['best_val_metric']:.4f}\n"
            f"耗时: {history['elapsed_time']:.2f}s"
        )
    
    def _on_training_failed(self, error_message: str):
        """处理训练失败。"""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("❌ 训练失败")
        self.log_text.append(f"错误: {error_message}")
        
        QMessageBox.critical(
            self,
            "训练失败",
            f"训练失败，错误信息：\n\n{error_message}"
        )
    
    def closeEvent(self, event):
        """处理对话框关闭。"""
        if self.training_thread and self.training_thread.isRunning():
            reply = QMessageBox.question(
                self,
                "训练正在进行中",
                "训练仍在运行。您确定要停止吗？",
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


# 需要导入 torch 进行设备检查
import torch
