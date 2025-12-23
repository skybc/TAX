"""
Annotation toolbar for selecting and configuring annotation tools.

Provides:
- Tool selection (brush, eraser, polygon, etc.)
- Tool parameters (brush size, opacity)
- Undo/redo buttons
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
    Toolbar for annotation tools and settings.
    
    Signals:
        tool_changed: Emitted when active tool changes (tool_name)
        brush_size_changed: Emitted when brush size changes (size)
        opacity_changed: Emitted when mask opacity changes (opacity)
        undo_requested: Emitted when undo is requested
        redo_requested: Emitted when redo is requested
        clear_requested: Emitted when clear is requested
        save_requested: Emitted when save is requested
        load_requested: Emitted when load is requested
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
        Initialize AnnotationToolbar.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        
        self.current_tool = "select"
        self.brush_size = 10
        self.mask_opacity = 0.5
        
        self._init_ui()
        
        logger.info("AnnotationToolbar initialized")
    
    def _init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Tools section
        tools_label = QLabel("<b>Tools</b>")
        layout.addWidget(tools_label)
        
        # Tool buttons
        self.tool_buttons = {}
        tool_layout = QHBoxLayout()
        
        tools = [
            ("select", "Select/Pan", "S"),
            ("brush", "Brush", "B"),
            ("eraser", "Eraser", "E"),
            ("polygon", "Polygon", "P"),
        ]
        
        self.tool_button_group = QButtonGroup(self)
        self.tool_button_group.setExclusive(True)
        
        for i, (tool_id, tool_name, shortcut) in enumerate(tools):
            btn = QToolButton()
            btn.setText(tool_name[0])  # First letter as icon
            btn.setToolTip(f"{tool_name} ({shortcut})")
            btn.setCheckable(True)
            btn.setFixedSize(40, 40)
            btn.clicked.connect(lambda checked, t=tool_id: self._set_tool(t))
            
            self.tool_buttons[tool_id] = btn
            self.tool_button_group.addButton(btn, i)
            tool_layout.addWidget(btn)
        
        # Set default tool
        self.tool_buttons["select"].setChecked(True)
        
        tool_layout.addStretch()
        layout.addLayout(tool_layout)
        
        # Brush size slider
        layout.addWidget(QLabel("<b>Brush Size</b>"))
        
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
        
        # Opacity slider
        layout.addWidget(QLabel("<b>Mask Opacity</b>"))
        
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
        
        # Undo/Redo buttons
        layout.addWidget(QLabel("<b>History</b>"))
        
        history_layout = QHBoxLayout()
        
        self.undo_btn = QPushButton("Undo")
        self.undo_btn.setToolTip("Undo (Ctrl+Z)")
        self.undo_btn.clicked.connect(self.undo_requested.emit)
        history_layout.addWidget(self.undo_btn)
        
        self.redo_btn = QPushButton("Redo")
        self.redo_btn.setToolTip("Redo (Ctrl+Y)")
        self.redo_btn.clicked.connect(self.redo_requested.emit)
        history_layout.addWidget(self.redo_btn)
        
        layout.addLayout(history_layout)
        
        # Actions
        layout.addWidget(QLabel("<b>Actions</b>"))
        
        actions_layout = QVBoxLayout()
        
        self.clear_btn = QPushButton("Clear Mask")
        self.clear_btn.clicked.connect(self.clear_requested.emit)
        actions_layout.addWidget(self.clear_btn)
        
        self.save_btn = QPushButton("Save Mask")
        self.save_btn.clicked.connect(self.save_requested.emit)
        actions_layout.addWidget(self.save_btn)
        
        self.load_btn = QPushButton("Load Mask")
        self.load_btn.clicked.connect(self.load_requested.emit)
        actions_layout.addWidget(self.load_btn)
        
        layout.addLayout(actions_layout)
        
        # Add stretch to push everything to top
        layout.addStretch()
    
    def _set_tool(self, tool: str):
        """
        Set the active tool.
        
        Args:
            tool: Tool name ('select', 'brush', 'eraser', 'polygon')
        """
        if tool == self.current_tool:
            return
        
        self.current_tool = tool
        self.tool_changed.emit(tool)
        
        logger.info(f"Tool changed to: {tool}")
    
    def _on_brush_size_changed(self, value: int):
        """Handle brush size slider change."""
        self.brush_size = value
        self.brush_size_spin.blockSignals(True)
        self.brush_size_spin.setValue(value)
        self.brush_size_spin.blockSignals(False)
        self.brush_size_changed.emit(value)
    
    def _on_brush_size_changed_spin(self, value: int):
        """Handle brush size spinbox change."""
        self.brush_size = value
        self.brush_size_slider.blockSignals(True)
        self.brush_size_slider.setValue(value)
        self.brush_size_slider.blockSignals(False)
        self.brush_size_changed.emit(value)
    
    def _on_opacity_changed(self, value: int):
        """Handle opacity slider change."""
        self.mask_opacity = value / 100.0
        self.opacity_label.setText(f"{value}%")
        self.opacity_changed.emit(self.mask_opacity)
    
    def get_current_tool(self) -> str:
        """Get the currently selected tool."""
        return self.current_tool
    
    def get_brush_size(self) -> int:
        """Get the current brush size."""
        return self.brush_size
    
    def get_mask_opacity(self) -> float:
        """Get the current mask opacity."""
        return self.mask_opacity
    
    def set_undo_enabled(self, enabled: bool):
        """Enable or disable undo button."""
        self.undo_btn.setEnabled(enabled)
    
    def set_redo_enabled(self, enabled: bool):
        """Enable or disable redo button."""
        self.redo_btn.setEnabled(enabled)
