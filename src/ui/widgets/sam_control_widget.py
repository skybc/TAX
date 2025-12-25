"""
用于管理 SAM 模型和设置的 SAM 控制小部件。

提供：
- 模型加载/卸载控制
- SAM 设置配置
- 提示模式选择
"""

from typing import Optional
from pathlib import Path
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QGroupBox, QRadioButton, QButtonGroup,
    QSpinBox, QCheckBox, QFileDialog, QLineEdit,
    QProgressBar, QTextEdit
)

from src.logger import get_logger

logger = get_logger(__name__)


class SAMControlWidget(QWidget):
    """
    用于 SAM 模型控制和配置的小部件。
    
    信号:
        model_load_requested: 请求加载模型时发出 (checkpoint_path)
        model_unload_requested: 请求卸载模型时发出
        prompt_mode_changed: 提示模式更改时发出 (mode)
        settings_changed: 设置更改时发出 (settings_dict)
    """
    
    model_load_requested = pyqtSignal(str)
    model_unload_requested = pyqtSignal()
    prompt_mode_changed = pyqtSignal(str)
    settings_changed = pyqtSignal(dict)
    
    def __init__(self, parent: Optional[QWidget] = None):
        """
        初始化 SAM 控制小部件。
        
        参数:
            parent: 父小部件
        """
        super().__init__(parent)
        
        self.checkpoint_path = None
        self.is_model_loaded = False
        
        self._init_ui()
        
        logger.info("SAMControlWidget 已初始化")
    
    def _init_ui(self):
        """初始化用户界面。"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # 模型部分
        model_group = QGroupBox("SAM 模型")
        model_layout = QVBoxLayout(model_group)
        
        # 检查点选择
        checkpoint_layout = QHBoxLayout()
        
        self.checkpoint_edit = QLineEdit()
        self.checkpoint_edit.setPlaceholderText("选择 SAM 检查点...")
        self.checkpoint_edit.setReadOnly(True)
        checkpoint_layout.addWidget(self.checkpoint_edit, 1)
        
        self.browse_checkpoint_btn = QPushButton("浏览...")
        self.browse_checkpoint_btn.clicked.connect(self._browse_checkpoint)
        checkpoint_layout.addWidget(self.browse_checkpoint_btn)
        
        model_layout.addLayout(checkpoint_layout)
        
        # 加载/卸载按钮
        button_layout = QHBoxLayout()
        
        self.load_btn = QPushButton("加载模型")
        self.load_btn.clicked.connect(self._load_model)
        self.load_btn.setEnabled(False)
        button_layout.addWidget(self.load_btn)
        
        self.unload_btn = QPushButton("卸载模型")
        self.unload_btn.clicked.connect(self._unload_model)
        self.unload_btn.setEnabled(False)
        button_layout.addWidget(self.unload_btn)
        
        model_layout.addLayout(button_layout)
        
        # 状态
        self.status_label = QLabel("状态: 未加载")
        self.status_label.setStyleSheet("QLabel { color: #888; }")
        model_layout.addWidget(self.status_label)
        
        layout.addWidget(model_group)
        
        # 提示模式部分
        prompt_group = QGroupBox("提示模式")
        prompt_layout = QVBoxLayout(prompt_group)
        
        self.prompt_mode_group = QButtonGroup(self)
        
        self.point_radio = QRadioButton("点提示")
        self.point_radio.setToolTip("点击以添加前景/背景点")
        self.point_radio.setChecked(True)
        self.point_radio.toggled.connect(lambda: self._on_prompt_mode_changed('points'))
        self.prompt_mode_group.addButton(self.point_radio)
        prompt_layout.addWidget(self.point_radio)
        
        self.box_radio = QRadioButton("框提示")
        self.box_radio.setToolTip("在对象周围绘制边界框")
        self.box_radio.toggled.connect(lambda: self._on_prompt_mode_changed('box'))
        self.prompt_mode_group.addButton(self.box_radio)
        prompt_layout.addWidget(self.box_radio)
        
        self.combined_radio = QRadioButton("组合 (点 + 框)")
        self.combined_radio.setToolTip("同时使用点和框")
        self.combined_radio.toggled.connect(lambda: self._on_prompt_mode_changed('combined'))
        self.prompt_mode_group.addButton(self.combined_radio)
        prompt_layout.addWidget(self.combined_radio)
        
        layout.addWidget(prompt_group)
        
        # 设置部分
        settings_group = QGroupBox("SAM 设置")
        settings_layout = QVBoxLayout(settings_group)
        
        # 多掩码输出
        self.multimask_check = QCheckBox("多掩码输出")
        self.multimask_check.setChecked(True)
        self.multimask_check.setToolTip("生成多个候选掩码")
        self.multimask_check.stateChanged.connect(self._on_settings_changed)
        settings_layout.addWidget(self.multimask_check)
        
        # 后处理
        self.post_process_check = QCheckBox("后处理掩码")
        self.post_process_check.setChecked(True)
        self.post_process_check.setToolTip("应用形态学操作")
        self.post_process_check.stateChanged.connect(self._on_settings_changed)
        settings_layout.addWidget(self.post_process_check)
        
        # 最小面积（用于移除小组件）
        min_area_layout = QHBoxLayout()
        min_area_layout.addWidget(QLabel("最小组件面积:"))
        
        self.min_area_spin = QSpinBox()
        self.min_area_spin.setMinimum(10)
        self.min_area_spin.setMaximum(10000)
        self.min_area_spin.setValue(100)
        self.min_area_spin.setSuffix(" px²")
        self.min_area_spin.valueChanged.connect(self._on_settings_changed)
        min_area_layout.addWidget(self.min_area_spin)
        min_area_layout.addStretch()
        
        settings_layout.addLayout(min_area_layout)
        
        layout.addWidget(settings_group)
        
        # 操作部分
        actions_group = QGroupBox("操作")
        actions_layout = QVBoxLayout(actions_group)
        
        self.run_sam_btn = QPushButton("运行 SAM")
        self.run_sam_btn.setEnabled(False)
        self.run_sam_btn.setToolTip("使用当前提示运行 SAM")
        actions_layout.addWidget(self.run_sam_btn)
        
        self.clear_prompts_btn = QPushButton("清除提示")
        self.clear_prompts_btn.setEnabled(False)
        actions_layout.addWidget(self.clear_prompts_btn)
        
        self.accept_mask_btn = QPushButton("接受掩码")
        self.accept_mask_btn.setEnabled(False)
        self.accept_mask_btn.setToolTip("接受并添加到标注中")
        actions_layout.addWidget(self.accept_mask_btn)
        
        layout.addWidget(actions_group)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # 信息文本
        self.info_text = QTextEdit()
        self.info_text.setMaximumHeight(80)
        self.info_text.setReadOnly(True)
        self.info_text.setPlaceholderText("SAM 信息和状态...")
        layout.addWidget(self.info_text)
        
        # 添加拉伸
        layout.addStretch()
    
    def _browse_checkpoint(self):
        """打开文件对话框以选择检查点。"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择 SAM 检查点",
            "",
            "PyTorch 模型 (*.pth *.pt);;所有文件 (*)"
        )
        
        if file_path:
            self.checkpoint_path = file_path
            self.checkpoint_edit.setText(file_path)
            self.load_btn.setEnabled(True)
            logger.info(f"已选择检查点: {file_path}")
    
    def _load_model(self):
        """请求加载模型。"""
        if self.checkpoint_path:
            self.model_load_requested.emit(self.checkpoint_path)
    
    def _unload_model(self):
        """请求卸载模型。"""
        self.model_unload_requested.emit()
    
    def _on_prompt_mode_changed(self, mode: str):
        """处理提示模式更改。"""
        self.prompt_mode_changed.emit(mode)
        logger.info(f"提示模式已更改为: {mode}")
    
    def _on_settings_changed(self):
        """处理设置更改。"""
        settings = self.get_settings()
        self.settings_changed.emit(settings)
    
    def set_model_loaded(self, loaded: bool):
        """
        更新模型加载状态的 UI。
        
        参数:
            loaded: 模型是否已加载
        """
        self.is_model_loaded = loaded
        
        self.load_btn.setEnabled(not loaded and self.checkpoint_path is not None)
        self.unload_btn.setEnabled(loaded)
        self.browse_checkpoint_btn.setEnabled(not loaded)
        self.run_sam_btn.setEnabled(loaded)
        self.clear_prompts_btn.setEnabled(loaded)
        
        if loaded:
            self.status_label.setText("状态: 已加载 ✓")
            self.status_label.setStyleSheet("QLabel { color: green; }")
        else:
            self.status_label.setText("状态: 未加载")
            self.status_label.setStyleSheet("QLabel { color: #888; }")
    
    def set_accept_enabled(self, enabled: bool):
        """启用/禁用接受掩码按钮。"""
        self.accept_mask_btn.setEnabled(enabled)
    
    def show_progress(self, show: bool):
        """显示/隐藏进度条。"""
        self.progress_bar.setVisible(show)
    
    def set_progress(self, value: int, message: str = ""):
        """
        更新进度条。
        
        参数:
            value: 进度值 (0-100)
            message: 进度消息
        """
        self.progress_bar.setValue(value)
        if message:
            self.add_info_message(message)
    
    def add_info_message(self, message: str):
        """向信息文本添加消息。"""
        self.info_text.append(message)
    
    def clear_info(self):
        """清除信息文本。"""
        self.info_text.clear()
    
    def get_prompt_mode(self) -> str:
        """获取当前提示模式。"""
        if self.point_radio.isChecked():
            return 'points'
        elif self.box_radio.isChecked():
            return 'box'
        else:
            return 'combined'
    
    def get_settings(self) -> dict:
        """
        获取当前 SAM 设置。
        
        返回:
            包含设置的字典
        """
        return {
            'multimask_output': self.multimask_check.isChecked(),
            'post_process': self.post_process_check.isChecked(),
            'min_area': self.min_area_spin.value()
        }
