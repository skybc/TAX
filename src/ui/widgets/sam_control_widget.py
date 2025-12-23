"""
SAM control widget for managing SAM model and settings.

Provides:
- Model loading/unloading controls
- SAM settings configuration
- Prompt mode selection
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
    Widget for SAM model control and configuration.
    
    Signals:
        model_load_requested: Emitted when model load is requested (checkpoint_path)
        model_unload_requested: Emitted when model unload is requested
        prompt_mode_changed: Emitted when prompt mode changes (mode)
        settings_changed: Emitted when settings change (settings_dict)
    """
    
    model_load_requested = pyqtSignal(str)
    model_unload_requested = pyqtSignal()
    prompt_mode_changed = pyqtSignal(str)
    settings_changed = pyqtSignal(dict)
    
    def __init__(self, parent: Optional[QWidget] = None):
        """
        Initialize SAM control widget.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        
        self.checkpoint_path = None
        self.is_model_loaded = False
        
        self._init_ui()
        
        logger.info("SAMControlWidget initialized")
    
    def _init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Model section
        model_group = QGroupBox("SAM Model")
        model_layout = QVBoxLayout(model_group)
        
        # Checkpoint selection
        checkpoint_layout = QHBoxLayout()
        
        self.checkpoint_edit = QLineEdit()
        self.checkpoint_edit.setPlaceholderText("Select SAM checkpoint...")
        self.checkpoint_edit.setReadOnly(True)
        checkpoint_layout.addWidget(self.checkpoint_edit, 1)
        
        self.browse_checkpoint_btn = QPushButton("Browse...")
        self.browse_checkpoint_btn.clicked.connect(self._browse_checkpoint)
        checkpoint_layout.addWidget(self.browse_checkpoint_btn)
        
        model_layout.addLayout(checkpoint_layout)
        
        # Load/Unload buttons
        button_layout = QHBoxLayout()
        
        self.load_btn = QPushButton("Load Model")
        self.load_btn.clicked.connect(self._load_model)
        self.load_btn.setEnabled(False)
        button_layout.addWidget(self.load_btn)
        
        self.unload_btn = QPushButton("Unload Model")
        self.unload_btn.clicked.connect(self._unload_model)
        self.unload_btn.setEnabled(False)
        button_layout.addWidget(self.unload_btn)
        
        model_layout.addLayout(button_layout)
        
        # Status
        self.status_label = QLabel("Status: Not loaded")
        self.status_label.setStyleSheet("QLabel { color: #888; }")
        model_layout.addWidget(self.status_label)
        
        layout.addWidget(model_group)
        
        # Prompt mode section
        prompt_group = QGroupBox("Prompt Mode")
        prompt_layout = QVBoxLayout(prompt_group)
        
        self.prompt_mode_group = QButtonGroup(self)
        
        self.point_radio = QRadioButton("Point Prompts")
        self.point_radio.setToolTip("Click to add foreground/background points")
        self.point_radio.setChecked(True)
        self.point_radio.toggled.connect(lambda: self._on_prompt_mode_changed('points'))
        self.prompt_mode_group.addButton(self.point_radio)
        prompt_layout.addWidget(self.point_radio)
        
        self.box_radio = QRadioButton("Box Prompt")
        self.box_radio.setToolTip("Draw a bounding box around object")
        self.box_radio.toggled.connect(lambda: self._on_prompt_mode_changed('box'))
        self.prompt_mode_group.addButton(self.box_radio)
        prompt_layout.addWidget(self.box_radio)
        
        self.combined_radio = QRadioButton("Combined (Points + Box)")
        self.combined_radio.setToolTip("Use both points and box")
        self.combined_radio.toggled.connect(lambda: self._on_prompt_mode_changed('combined'))
        self.prompt_mode_group.addButton(self.combined_radio)
        prompt_layout.addWidget(self.combined_radio)
        
        layout.addWidget(prompt_group)
        
        # Settings section
        settings_group = QGroupBox("SAM Settings")
        settings_layout = QVBoxLayout(settings_group)
        
        # Multi-mask output
        self.multimask_check = QCheckBox("Multi-mask output")
        self.multimask_check.setChecked(True)
        self.multimask_check.setToolTip("Generate multiple mask candidates")
        self.multimask_check.stateChanged.connect(self._on_settings_changed)
        settings_layout.addWidget(self.multimask_check)
        
        # Post-processing
        self.post_process_check = QCheckBox("Post-process masks")
        self.post_process_check.setChecked(True)
        self.post_process_check.setToolTip("Apply morphological operations")
        self.post_process_check.stateChanged.connect(self._on_settings_changed)
        settings_layout.addWidget(self.post_process_check)
        
        # Min area for small component removal
        min_area_layout = QHBoxLayout()
        min_area_layout.addWidget(QLabel("Min component area:"))
        
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
        
        # Actions section
        actions_group = QGroupBox("Actions")
        actions_layout = QVBoxLayout(actions_group)
        
        self.run_sam_btn = QPushButton("Run SAM")
        self.run_sam_btn.setEnabled(False)
        self.run_sam_btn.setToolTip("Run SAM with current prompts")
        actions_layout.addWidget(self.run_sam_btn)
        
        self.clear_prompts_btn = QPushButton("Clear Prompts")
        self.clear_prompts_btn.setEnabled(False)
        actions_layout.addWidget(self.clear_prompts_btn)
        
        self.accept_mask_btn = QPushButton("Accept Mask")
        self.accept_mask_btn.setEnabled(False)
        self.accept_mask_btn.setToolTip("Accept and add to annotation")
        actions_layout.addWidget(self.accept_mask_btn)
        
        layout.addWidget(actions_group)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Info text
        self.info_text = QTextEdit()
        self.info_text.setMaximumHeight(80)
        self.info_text.setReadOnly(True)
        self.info_text.setPlaceholderText("SAM information and status...")
        layout.addWidget(self.info_text)
        
        # Add stretch
        layout.addStretch()
    
    def _browse_checkpoint(self):
        """Open file dialog to select checkpoint."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select SAM Checkpoint",
            "",
            "PyTorch Models (*.pth *.pt);;All Files (*)"
        )
        
        if file_path:
            self.checkpoint_path = file_path
            self.checkpoint_edit.setText(file_path)
            self.load_btn.setEnabled(True)
            logger.info(f"Selected checkpoint: {file_path}")
    
    def _load_model(self):
        """Request model loading."""
        if self.checkpoint_path:
            self.model_load_requested.emit(self.checkpoint_path)
    
    def _unload_model(self):
        """Request model unloading."""
        self.model_unload_requested.emit()
    
    def _on_prompt_mode_changed(self, mode: str):
        """Handle prompt mode change."""
        self.prompt_mode_changed.emit(mode)
        logger.info(f"Prompt mode changed to: {mode}")
    
    def _on_settings_changed(self):
        """Handle settings change."""
        settings = self.get_settings()
        self.settings_changed.emit(settings)
    
    def set_model_loaded(self, loaded: bool):
        """
        Update UI for model loaded state.
        
        Args:
            loaded: Whether model is loaded
        """
        self.is_model_loaded = loaded
        
        self.load_btn.setEnabled(not loaded and self.checkpoint_path is not None)
        self.unload_btn.setEnabled(loaded)
        self.browse_checkpoint_btn.setEnabled(not loaded)
        self.run_sam_btn.setEnabled(loaded)
        self.clear_prompts_btn.setEnabled(loaded)
        
        if loaded:
            self.status_label.setText("Status: Loaded ✓")
            self.status_label.setStyleSheet("QLabel { color: green; }")
        else:
            self.status_label.setText("Status: Not loaded")
            self.status_label.setStyleSheet("QLabel { color: #888; }")
    
    def set_accept_enabled(self, enabled: bool):
        """Enable/disable accept mask button."""
        self.accept_mask_btn.setEnabled(enabled)
    
    def show_progress(self, show: bool):
        """Show/hide progress bar."""
        self.progress_bar.setVisible(show)
    
    def set_progress(self, value: int, message: str = ""):
        """
        Update progress bar.
        
        Args:
            value: Progress value (0-100)
            message: Progress message
        """
        self.progress_bar.setValue(value)
        if message:
            self.add_info_message(message)
    
    def add_info_message(self, message: str):
        """Add a message to info text."""
        self.info_text.append(message)
    
    def clear_info(self):
        """Clear info text."""
        self.info_text.clear()
    
    def get_prompt_mode(self) -> str:
        """Get current prompt mode."""
        if self.point_radio.isChecked():
            return 'points'
        elif self.box_radio.isChecked():
            return 'box'
        else:
            return 'combined'
    
    def get_settings(self) -> dict:
        """
        Get current SAM settings.
        
        Returns:
            Dictionary with settings
        """
        return {
            'multimask_output': self.multimask_check.isChecked(),
            'post_process': self.post_process_check.isChecked(),
            'min_area': self.min_area_spin.value()
        }
