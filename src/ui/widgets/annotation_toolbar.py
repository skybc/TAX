"""
用于选择和配置标注工具的标注工具栏。

提供：
- 工具选择（画笔、橡皮擦、多边形等）
- 工具参数（画笔大小、不透明度）
- 撤销/重做按钮
"""

from typing import Optional
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QIcon, QPixmap, QPainter, QColor
from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QToolBar,
    QAction, QActionGroup, QLabel, QSlider,
    QSpinBox, QPushButton, QButtonGroup, QToolButton
)

from src.logger import get_logger

logger = get_logger(__name__)


class AnnotationToolbar(QWidget):
    """
    标注工具和设置的工具栏。
    
    信号:
        tool_changed: 活动工具更改时发出 (tool_name)
        brush_size_changed: 画笔大小更改时发出 (size)
        opacity_changed: 掩码不透明度更改时发出 (opacity)
        undo_requested: 请求撤销时发出
        redo_requested: 请求重那时发出
        clear_requested: 请求清除时发出
        save_requested: 请求保存时发出
        load_requested: 请求加载时发出
    """
    
    tool_changed = pyqtSignal(str)
    brush_size_changed = pyqtSignal(int)
    opacity_changed = pyqtSignal(float)
    undo_requested = pyqtSignal()
    redo_requested = pyqtSignal()
    clear_requested = pyqtSignal()
    save_requested = pyqtSignal()
    load_requested = pyqtSignal()
    
    def __init__(self, parent: Optional[QWidget] = None):
        """
        初始化 AnnotationToolbar。
        
        参数:
            parent: 父小部件
        """
        super().__init__(parent)
        
        self.current_tool = "select"
        self.brush_size = 10
        self.mask_opacity = 0.5
        
        self._init_ui()
        
        logger.info("AnnotationToolbar 已初始化")
    
    def _init_ui(self):
        """初始化用户界面。"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # 工具部分
        tools_label = QLabel("<b>工具</b>")
        layout.addWidget(tools_label)
        
        # 工具按钮
        self.tool_buttons = {}
        tool_layout = QHBoxLayout()
        
        tools = [
            ("select", "选择/平移", "S"),
            ("brush", "画笔", "B"),
            ("eraser", "橡皮擦", "E"),
            ("polygon", "多边形", "P"),
        ]
        
        self.tool_button_group = QButtonGroup(self)
        self.tool_button_group.setExclusive(True)
        
        for i, (tool_id, tool_name, shortcut) in enumerate(tools):
            btn = QToolButton()
            btn.setText(tool_name[0])  # 使用首字母作为图标
            btn.setToolTip(f"{tool_name} ({shortcut})")
            btn.setCheckable(True)
            btn.setFixedSize(40, 40)
            btn.clicked.connect(lambda checked, t=tool_id: self._set_tool(t))
            
            self.tool_buttons[tool_id] = btn
            self.tool_button_group.addButton(btn, i)
            tool_layout.addWidget(btn)
        
        # 设置默认工具
        self.tool_buttons["select"].setChecked(True)
        
        tool_layout.addStretch()
        layout.addLayout(tool_layout)
        
        # 画笔大小滑块
        layout.addWidget(QLabel("<b>画笔大小</b>"))
        
        brush_size_layout = QHBoxLayout()
        
        self.brush_size_slider = QSlider(Qt.Horizontal)
        self.brush_size_slider.setMinimum(1)
        self.brush_size_slider.setMaximum(100)
        self.brush_size_slider.setValue(self.brush_size)
        self.brush_size_slider.valueChanged.connect(self._on_brush_size_changed)
        brush_size_layout.addWidget(self.brush_size_slider)
        
        self.brush_size_spin = QSpinBox()
        self.brush_size_spin.setMinimum(1)
        self.brush_size_spin.setMaximum(100)
        self.brush_size_spin.setValue(self.brush_size)
        self.brush_size_spin.setFixedWidth(60)
        self.brush_size_spin.valueChanged.connect(self._on_brush_size_changed_spin)
        brush_size_layout.addWidget(self.brush_size_spin)
        
        layout.addLayout(brush_size_layout)
        
        # 不透明度滑块
        layout.addWidget(QLabel("<b>掩码不透明度</b>"))
        
        opacity_layout = QHBoxLayout()
        
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setMinimum(0)
        self.opacity_slider.setMaximum(100)
        self.opacity_slider.setValue(int(self.mask_opacity * 100))
        self.opacity_slider.valueChanged.connect(self._on_opacity_changed)
        opacity_layout.addWidget(self.opacity_slider)
        
        self.opacity_label = QLabel(f"{int(self.mask_opacity * 100)}%")
        self.opacity_label.setFixedWidth(40)
        opacity_layout.addWidget(self.opacity_label)
        
        layout.addLayout(opacity_layout)
        
        # 撤销/重做按钮
        layout.addWidget(QLabel("<b>历史记录</b>"))
        
        history_layout = QHBoxLayout()
        
        self.undo_btn = QPushButton("撤销")
        self.undo_btn.setToolTip("撤销 (Ctrl+Z)")
        self.undo_btn.clicked.connect(self.undo_requested.emit)
        history_layout.addWidget(self.undo_btn)
        
        self.redo_btn = QPushButton("重做")
        self.redo_btn.setToolTip("重做 (Ctrl+Y)")
        self.redo_btn.clicked.connect(self.redo_requested.emit)
        history_layout.addWidget(self.redo_btn)
        
        layout.addLayout(history_layout)
        
        # 操作
        layout.addWidget(QLabel("<b>操作</b>"))
        
        actions_layout = QVBoxLayout()
        
        self.clear_btn = QPushButton("清除掩码")
        self.clear_btn.clicked.connect(self.clear_requested.emit)
        actions_layout.addWidget(self.clear_btn)
        
        self.save_btn = QPushButton("保存掩码")
        self.save_btn.clicked.connect(self.save_requested.emit)
        actions_layout.addWidget(self.save_btn)
        
        self.load_btn = QPushButton("加载掩码")
        self.load_btn.clicked.connect(self.load_requested.emit)
        actions_layout.addWidget(self.load_btn)
        
        layout.addLayout(actions_layout)
        
        # 添加拉伸以将所有内容推向顶部
        layout.addStretch()
    
    def _set_tool(self, tool: str):
        """
        设置活动工具。
        
        参数:
            tool: 工具名称 ('select', 'brush', 'eraser', 'polygon')
        """
        if tool == self.current_tool:
            return
        
        self.current_tool = tool
        self.tool_changed.emit(tool)
        
        logger.info(f"工具更改为: {tool}")
    
    def _on_brush_size_changed(self, value: int):
        """处理画笔大小滑块更改。"""
        self.brush_size = value
        self.brush_size_spin.blockSignals(True)
        self.brush_size_spin.setValue(value)
        self.brush_size_spin.blockSignals(False)
        self.brush_size_changed.emit(value)
    
    def _on_brush_size_changed_spin(self, value: int):
        """处理画笔大小微调框更改。"""
        self.brush_size = value
        self.brush_size_slider.blockSignals(True)
        self.brush_size_slider.setValue(value)
        self.brush_size_slider.blockSignals(False)
        self.brush_size_changed.emit(value)
    
    def _on_opacity_changed(self, value: int):
        """处理不透明度滑块更改。"""
        self.mask_opacity = value / 100.0
        self.opacity_label.setText(f"{value}%")
        self.opacity_changed.emit(self.mask_opacity)
    
    def get_current_tool(self) -> str:
        """获取当前选择的工具。"""
        return self.current_tool
    
    def get_brush_size(self) -> int:
        """获取当前画笔大小。"""
        return self.brush_size
    
    def get_mask_opacity(self) -> float:
        """获取当前掩码不透明度。"""
        return self.mask_opacity
    
    def set_undo_enabled(self, enabled: bool):
        """启用或禁用撤销按钮。"""
        self.undo_btn.setEnabled(enabled)
    
    def set_redo_enabled(self, enabled: bool):
        """启用或禁用重做按钮。"""
        self.redo_btn.setEnabled(enabled)
