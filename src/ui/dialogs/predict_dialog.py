"""
用于批量推理的预测对话框。

此模块提供：
- 模型选择
- 图像选择
- 推理配置
- 实时进度监控
"""

from pathlib import Path
from typing import List, Optional
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QTabWidget,
    QLabel, QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit,
    QPushButton, QCheckBox, QGroupBox, QProgressBar, QTextEdit,
    QFileDialog, QListWidget, QMessageBox
)

from src.logger import get_logger
from src.threads.inference_thread import InferenceThread

logger = get_logger(__name__)


class PredictDialog(QDialog):
    """
    用于配置和运行批量推理的对话框。
    """
    
    def __init__(self, config: dict, paths_config: dict, parent=None):
        """
        初始化预测对话框。
        
        参数:
            config: 应用程序配置
            paths_config: 路径配置
            parent: 父小部件
        """
        super().__init__(parent)
        
        self.config = config
        self.paths_config = paths_config
        self.inference_thread: Optional[InferenceThread] = None
        
        # 图像列表
        self.image_paths: List[str] = []
        
        self.setWindowTitle("预测分割掩码")
        self.setMinimumSize(900, 700)
        
        self._init_ui()
        self._load_defaults()
        
        logger.info("预测对话框已打开")
    
    def _init_ui(self):
        """初始化 UI 组件。"""
        layout = QVBoxLayout(self)
        
        # 创建选项卡
        tabs = QTabWidget()
        
        # 选项卡 1: 模型与数据
        model_tab = self._create_model_tab()
        tabs.addTab(model_tab, "模型与数据")
        
        # 选项卡 2: 推理设置
        settings_tab = self._create_settings_tab()
        tabs.addTab(settings_tab, "设置")
        
        # 选项卡 3: 后处理
        postprocess_tab = self._create_postprocess_tab()
        tabs.addTab(postprocess_tab, "后处理")
        
        # 选项卡 4: 监控
        monitor_tab = self._create_monitor_tab()
        tabs.addTab(monitor_tab, "监控")
        
        layout.addWidget(tabs)
        
        # 按钮
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.start_btn = QPushButton("开始预测")
        self.start_btn.setDefault(True)
        self.start_btn.clicked.connect(self._start_inference)
        button_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_inference)
        button_layout.addWidget(self.stop_btn)
        
        self.close_btn = QPushButton("关闭")
        self.close_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.close_btn)
        
        layout.addLayout(button_layout)
    
    def _create_model_tab(self):
        """创建模型与数据选项卡。"""
        from PyQt5.QtWidgets import QWidget
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 模型选择
        model_group = QGroupBox("模型")
        model_layout = QGridLayout(model_group)
        
        model_layout.addWidget(QLabel("检查点 (Checkpoint):"), 0, 0)
        self.checkpoint_edit = QLineEdit()
        model_layout.addWidget(self.checkpoint_edit, 0, 1)
        browse_model_btn = QPushButton("浏览...")
        browse_model_btn.clicked.connect(self._browse_checkpoint)
        model_layout.addWidget(browse_model_btn, 0, 2)
        
        model_layout.addWidget(QLabel("架构:"), 1, 0)
        self.arch_combo = QComboBox()
        self.arch_combo.addItems(["U-Net", "DeepLabV3+", "FPN"])
        model_layout.addWidget(self.arch_combo, 1, 1)
        
        model_layout.addWidget(QLabel("编码器 (Encoder):"), 2, 0)
        self.encoder_combo = QComboBox()
        self.encoder_combo.addItems([
            "resnet18", "resnet34", "resnet50",
            "efficientnet-b0", "efficientnet-b1",
            "mobilenet_v2"
        ])
        model_layout.addWidget(self.encoder_combo, 2, 1)
        
        layout.addWidget(model_group)
        
        # 图像选择
        image_group = QGroupBox("图像")
        image_layout = QVBoxLayout(image_group)
        
        # 输入目录
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("输入目录:"))
        self.input_dir_edit = QLineEdit()
        input_layout.addWidget(self.input_dir_edit)
        browse_input_btn = QPushButton("浏览...")
        browse_input_btn.clicked.connect(self._browse_input_dir)
        input_layout.addWidget(browse_input_btn)
        image_layout.addLayout(input_layout)
        
        # 图像列表
        self.image_list = QListWidget()
        self.image_list.setMaximumHeight(150)
        image_layout.addWidget(QLabel("已选图像:"))
        image_layout.addWidget(self.image_list)
        
        # 添加/移除按钮
        list_buttons = QHBoxLayout()
        add_images_btn = QPushButton("添加图像...")
        add_images_btn.clicked.connect(self._add_images)
        list_buttons.addWidget(add_images_btn)
        
        remove_btn = QPushButton("移除所选")
        remove_btn.clicked.connect(self._remove_selected)
        list_buttons.addWidget(remove_btn)
        
        clear_btn = QPushButton("清空全部")
        clear_btn.clicked.connect(self._clear_images)
        list_buttons.addWidget(clear_btn)
        list_buttons.addStretch()
        
        image_layout.addLayout(list_buttons)
        
        layout.addWidget(image_group)
        
        # 输出目录
        output_group = QGroupBox("输出")
        output_layout = QHBoxLayout(output_group)
        output_layout.addWidget(QLabel("输出目录:"))
        self.output_dir_edit = QLineEdit()
        output_layout.addWidget(self.output_dir_edit)
        browse_output_btn = QPushButton("浏览...")
        browse_output_btn.clicked.connect(lambda: self._browse_dir(self.output_dir_edit))
        output_layout.addWidget(browse_output_btn)
        
        layout.addWidget(output_group)
        
        return widget
    
    def _create_settings_tab(self):
        """创建推理设置选项卡。"""
        from PyQt5.QtWidgets import QWidget
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
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
        
        img_layout.addWidget(QLabel("阈值:"), 1, 0)
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 1.0)
        self.threshold_spin.setSingleStep(0.05)
        self.threshold_spin.setValue(0.5)
        img_layout.addWidget(self.threshold_spin, 1, 1)
        
        layout.addWidget(img_group)
        
        # 高级设置
        advanced_group = QGroupBox("高级设置")
        advanced_layout = QVBoxLayout(advanced_group)
        
        self.tta_check = QCheckBox("使用测试时增强 (TTA)")
        self.tta_check.setToolTip("通过平均多次增强的预测结果来提高准确性")
        advanced_layout.addWidget(self.tta_check)
        
        tta_layout = QHBoxLayout()
        tta_layout.addWidget(QLabel("TTA 增强次数:"))
        self.tta_aug_spin = QSpinBox()
        self.tta_aug_spin.setRange(2, 8)
        self.tta_aug_spin.setValue(4)
        self.tta_aug_spin.setEnabled(False)
        tta_layout.addWidget(self.tta_aug_spin)
        tta_layout.addStretch()
        advanced_layout.addLayout(tta_layout)
        
        self.tta_check.toggled.connect(self.tta_aug_spin.setEnabled)
        
        self.save_overlay_check = QCheckBox("保存叠加可视化结果")
        self.save_overlay_check.setChecked(True)
        advanced_layout.addWidget(self.save_overlay_check)
        
        layout.addWidget(advanced_group)
        layout.addStretch()
        
        return widget
    
    def _create_postprocess_tab(self):
        """创建后处理选项卡。"""
        from PyQt5.QtWidgets import QWidget
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 启用后处理
        self.postprocess_check = QCheckBox("应用后处理")
        self.postprocess_check.setChecked(True)
        layout.addWidget(self.postprocess_check)
        
        # 后处理选项
        postprocess_group = QGroupBox("后处理选项")
        postprocess_layout = QGridLayout(postprocess_group)
        
        self.remove_small_check = QCheckBox("移除小目标")
        self.remove_small_check.setChecked(True)
        postprocess_layout.addWidget(self.remove_small_check, 0, 0, 1, 2)
        
        postprocess_layout.addWidget(QLabel("最小目标尺寸:"), 1, 0)
        self.min_size_spin = QSpinBox()
        self.min_size_spin.setRange(10, 10000)
        self.min_size_spin.setValue(100)
        postprocess_layout.addWidget(self.min_size_spin, 1, 1)
        
        self.fill_holes_check = QCheckBox("填充孔洞")
        self.fill_holes_check.setChecked(True)
        postprocess_layout.addWidget(self.fill_holes_check, 2, 0, 1, 2)
        
        self.smooth_check = QCheckBox("平滑轮廓")
        self.smooth_check.setChecked(True)
        postprocess_layout.addWidget(self.smooth_check, 3, 0, 1, 2)
        
        postprocess_layout.addWidget(QLabel("闭运算核大小:"), 4, 0)
        self.closing_spin = QSpinBox()
        self.closing_spin.setRange(0, 21)
        self.closing_spin.setSingleStep(2)
        self.closing_spin.setValue(5)
        postprocess_layout.addWidget(self.closing_spin, 4, 1)
        
        # 根据复选框启用/禁用
        self.postprocess_check.toggled.connect(postprocess_group.setEnabled)
        
        layout.addWidget(postprocess_group)
        layout.addStretch()
        
        return widget
    
    def _create_monitor_tab(self):
        """创建监控选项卡。"""
        from PyQt5.QtWidgets import QWidget
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 进度
        progress_group = QGroupBox("进度")
        progress_layout = QVBoxLayout(progress_group)
        
        self.progress_bar = QProgressBar()
        progress_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("准备预测")
        progress_layout.addWidget(self.status_label)
        
        layout.addWidget(progress_group)
        
        # 日志
        log_group = QGroupBox("推理日志")
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        
        layout.addWidget(log_group)
        
        return widget
    
    def _load_defaults(self):
        """加载默认值。"""
        from pathlib import Path
        
        # 设置默认路径
        project_root = Path(self.paths_config['paths']['data_root'])
        self.output_dir_edit.setText(str(project_root / "outputs/predictions"))
        
        # 尝试查找最佳检查点
        models_dir = project_root / "outputs/models"
        if models_dir.exists():
            best_model = models_dir / "best_model.pth"
            if best_model.exists():
                self.checkpoint_edit.setText(str(best_model))
    
    def _browse_checkpoint(self):
        """浏览检查点文件。"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择模型检查点",
            self.checkpoint_edit.text(),
            "检查点文件 (*.pth *.pt);;所有文件 (*)"
        )
        
        if file_path:
            self.checkpoint_edit.setText(file_path)
    
    def _browse_input_dir(self):
        """浏览输入目录。"""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "选择输入目录",
            self.input_dir_edit.text()
        )
        
        if dir_path:
            self.input_dir_edit.setText(dir_path)
            # 从目录自动加载图像
            self._load_images_from_dir(dir_path)
    
    def _load_images_from_dir(self, dir_path: str):
        """从目录加载所有图像。"""
        from src.utils.file_utils import list_files
        
        image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        images = list_files(dir_path, extensions=image_exts, recursive=False)
        
        if images:
            self.image_paths.extend(images)
            self._update_image_list()
            self.log_text.append(f"从 {dir_path} 加载了 {len(images)} 张图像")
    
    def _add_images(self):
        """添加单个图像。"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "选择图像",
            "",
            "图像文件 (*.jpg *.jpeg *.png *.bmp *.tif *.tiff);;所有文件 (*)"
        )
        
        if file_paths:
            self.image_paths.extend(file_paths)
            self._update_image_list()
            self.log_text.append(f"添加了 {len(file_paths)} 张图像")
    
    def _remove_selected(self):
        """从列表中移除所选图像。"""
        selected_items = self.image_list.selectedItems()
        for item in selected_items:
            row = self.image_list.row(item)
            self.image_paths.pop(row)
            self.image_list.takeItem(row)
    
    def _clear_images(self):
        """清空所有图像。"""
        self.image_paths.clear()
        self.image_list.clear()
    
    def _update_image_list(self):
        """更新图像列表小部件。"""
        self.image_list.clear()
        for path in self.image_paths:
            self.image_list.addItem(Path(path).name)
    
    def _browse_dir(self, line_edit: QLineEdit):
        """浏览目录。"""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "选择目录",
            line_edit.text()
        )
        
        if dir_path:
            line_edit.setText(dir_path)
    
    def _start_inference(self):
        """开始推理。"""
        # 验证
        if not self.checkpoint_edit.text():
            QMessageBox.warning(self, "错误", "请选择模型检查点")
            return
        
        if not self.image_paths:
            QMessageBox.warning(self, "错误", "请添加要处理的图像")
            return
        
        if not self.output_dir_edit.text():
            QMessageBox.warning(self, "错误", "请指定输出目录")
            return
        
        # 准备配置
        inference_config = {
            'architecture': self.arch_combo.currentText().lower().replace('-', '').replace('+', 'plus'),
            'encoder': self.encoder_combo.currentText(),
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'image_width': self.img_width_spin.value(),
            'image_height': self.img_height_spin.value(),
            'threshold': self.threshold_spin.value(),
            'use_tta': self.tta_check.isChecked(),
            'tta_augmentations': self.tta_aug_spin.value(),
            'save_overlay': self.save_overlay_check.isChecked(),
            'apply_post_processing': self.postprocess_check.isChecked(),
            'remove_small_objects': self.remove_small_check.isChecked(),
            'min_object_size': self.min_size_spin.value(),
            'fill_holes': self.fill_holes_check.isChecked(),
            'smooth_contours': self.smooth_check.isChecked(),
            'closing_kernel_size': self.closing_spin.value()
        }
        
        # 创建推理线程
        self.inference_thread = InferenceThread(
            checkpoint_path=self.checkpoint_edit.text(),
            image_paths=self.image_paths,
            output_dir=self.output_dir_edit.text(),
            config=inference_config
        )
        
        # 连接信号
        self.inference_thread.progress_updated.connect(self._on_progress_updated)
        self.inference_thread.image_completed.connect(self._on_image_completed)
        self.inference_thread.inference_completed.connect(self._on_inference_completed)
        self.inference_thread.inference_failed.connect(self._on_inference_failed)
        
        # 更新 UI
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("推理已开始...")
        self.log_text.append("=== 推理已开始 ===")
        self.log_text.append(f"总图像数: {len(self.image_paths)}")
        
        # 开始
        self.inference_thread.start()
        
        logger.info("推理已开始")
    
    def _stop_inference(self):
        """停止推理。"""
        if self.inference_thread:
            self.inference_thread.stop()
            self.stop_btn.setEnabled(False)
            self.status_label.setText("正在停止...")
            self.log_text.append("已请求停止...")
    
    def _on_progress_updated(self, current: int, total: int, image_path: str):
        """处理进度更新。"""
        progress = int((current / total) * 100)
        self.progress_bar.setValue(progress)
        self.status_label.setText(f"正在处理: {current}/{total} - {Path(image_path).name}")
    
    def _on_image_completed(self, index: int, image_path: str, success: bool):
        """处理图像完成。"""
        status = "✓" if success else "✗"
        self.log_text.append(f"{status} {Path(image_path).name}")
    
    def _on_inference_completed(self, results: dict):
        """处理推理完成。"""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setValue(100)
        self.status_label.setText(f"✓ 推理已完成！{results['successful']}/{results['total']} 成功")
        
        self.log_text.append("=== 推理已完成 ===")
        self.log_text.append(f"成功: {results['successful']}")
        self.log_text.append(f"失败: {results['failed']}")
        
        if results['failed_files']:
            self.log_text.append("\n失败文件:")
            for path in results['failed_files']:
                self.log_text.append(f"  - {Path(path).name}")
        
        QMessageBox.information(
            self,
            "推理完成",
            f"推理已成功完成！\n\n"
            f"成功: {results['successful']}/{results['total']}\n"
            f"输出目录: {self.output_dir_edit.text()}"
        )
    
    def _on_inference_failed(self, error_message: str):
        """处理推理失败。"""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("❌ 推理失败")
        self.log_text.append(f"错误: {error_message}")
        
        QMessageBox.critical(
            self,
            "推理失败",
            f"推理失败，错误信息：\n\n{error_message}"
        )
    
    def closeEvent(self, event):
        """处理对话框关闭。"""
        if self.inference_thread and self.inference_thread.isRunning():
            reply = QMessageBox.question(
                self,
                "推理正在进行中",
                "推理仍在运行。您确定要停止吗？",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.inference_thread.stop()
                self.inference_thread.wait()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


# 需要 torch 进行设备检查
import torch
