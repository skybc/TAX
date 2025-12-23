"""
Prediction dialog for batch inference.

This module provides:
- Model selection
- Image selection
- Inference configuration
- Real-time progress monitoring
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
    Dialog for configuring and running batch inference.
    """
    
    def __init__(self, config: dict, paths_config: dict, parent=None):
        """
        Initialize prediction dialog.
        
        Args:
            config: Application configuration
            paths_config: Paths configuration
            parent: Parent widget
        """
        super().__init__(parent)
        
        self.config = config
        self.paths_config = paths_config
        self.inference_thread: Optional[InferenceThread] = None
        
        # Image list
        self.image_paths: List[str] = []
        
        self.setWindowTitle("Predict Segmentation Masks")
        self.setMinimumSize(900, 700)
        
        self._init_ui()
        self._load_defaults()
        
        logger.info("Prediction dialog opened")
    
    def _init_ui(self):
        """Initialize UI components."""
        layout = QVBoxLayout(self)
        
        # Create tabs
        tabs = QTabWidget()
        
        # Tab 1: Model & Data
        model_tab = self._create_model_tab()
        tabs.addTab(model_tab, "Model & Data")
        
        # Tab 2: Inference Settings
        settings_tab = self._create_settings_tab()
        tabs.addTab(settings_tab, "Settings")
        
        # Tab 3: Post-processing
        postprocess_tab = self._create_postprocess_tab()
        tabs.addTab(postprocess_tab, "Post-processing")
        
        # Tab 4: Monitor
        monitor_tab = self._create_monitor_tab()
        tabs.addTab(monitor_tab, "Monitor")
        
        layout.addWidget(tabs)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.start_btn = QPushButton("Start Prediction")
        self.start_btn.setDefault(True)
        self.start_btn.clicked.connect(self._start_inference)
        button_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_inference)
        button_layout.addWidget(self.stop_btn)
        
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.close_btn)
        
        layout.addLayout(button_layout)
    
    def _create_model_tab(self):
        """Create model & data tab."""
        from PyQt5.QtWidgets import QWidget
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Model selection
        model_group = QGroupBox("Model")
        model_layout = QGridLayout(model_group)
        
        model_layout.addWidget(QLabel("Checkpoint:"), 0, 0)
        self.checkpoint_edit = QLineEdit()
        model_layout.addWidget(self.checkpoint_edit, 0, 1)
        browse_model_btn = QPushButton("Browse...")
        browse_model_btn.clicked.connect(self._browse_checkpoint)
        model_layout.addWidget(browse_model_btn, 0, 2)
        
        model_layout.addWidget(QLabel("Architecture:"), 1, 0)
        self.arch_combo = QComboBox()
        self.arch_combo.addItems(["U-Net", "DeepLabV3+", "FPN"])
        model_layout.addWidget(self.arch_combo, 1, 1)
        
        model_layout.addWidget(QLabel("Encoder:"), 2, 0)
        self.encoder_combo = QComboBox()
        self.encoder_combo.addItems([
            "resnet18", "resnet34", "resnet50",
            "efficientnet-b0", "efficientnet-b1",
            "mobilenet_v2"
        ])
        model_layout.addWidget(self.encoder_combo, 2, 1)
        
        layout.addWidget(model_group)
        
        # Image selection
        image_group = QGroupBox("Images")
        image_layout = QVBoxLayout(image_group)
        
        # Input directory
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("Input Directory:"))
        self.input_dir_edit = QLineEdit()
        input_layout.addWidget(self.input_dir_edit)
        browse_input_btn = QPushButton("Browse...")
        browse_input_btn.clicked.connect(self._browse_input_dir)
        input_layout.addWidget(browse_input_btn)
        image_layout.addLayout(input_layout)
        
        # Image list
        self.image_list = QListWidget()
        self.image_list.setMaximumHeight(150)
        image_layout.addWidget(QLabel("Selected Images:"))
        image_layout.addWidget(self.image_list)
        
        # Add/Remove buttons
        list_buttons = QHBoxLayout()
        add_images_btn = QPushButton("Add Images...")
        add_images_btn.clicked.connect(self._add_images)
        list_buttons.addWidget(add_images_btn)
        
        remove_btn = QPushButton("Remove Selected")
        remove_btn.clicked.connect(self._remove_selected)
        list_buttons.addWidget(remove_btn)
        
        clear_btn = QPushButton("Clear All")
        clear_btn.clicked.connect(self._clear_images)
        list_buttons.addWidget(clear_btn)
        list_buttons.addStretch()
        
        image_layout.addLayout(list_buttons)
        
        layout.addWidget(image_group)
        
        # Output directory
        output_group = QGroupBox("Output")
        output_layout = QHBoxLayout(output_group)
        output_layout.addWidget(QLabel("Output Directory:"))
        self.output_dir_edit = QLineEdit()
        output_layout.addWidget(self.output_dir_edit)
        browse_output_btn = QPushButton("Browse...")
        browse_output_btn.clicked.connect(lambda: self._browse_dir(self.output_dir_edit))
        output_layout.addWidget(browse_output_btn)
        
        layout.addWidget(output_group)
        
        return widget
    
    def _create_settings_tab(self):
        """Create inference settings tab."""
        from PyQt5.QtWidgets import QWidget
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Image settings
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
        
        img_layout.addWidget(QLabel("Threshold:"), 1, 0)
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 1.0)
        self.threshold_spin.setSingleStep(0.05)
        self.threshold_spin.setValue(0.5)
        img_layout.addWidget(self.threshold_spin, 1, 1)
        
        layout.addWidget(img_group)
        
        # Advanced settings
        advanced_group = QGroupBox("Advanced")
        advanced_layout = QVBoxLayout(advanced_group)
        
        self.tta_check = QCheckBox("Use Test-Time Augmentation (TTA)")
        self.tta_check.setToolTip("Improve accuracy by averaging predictions from multiple augmentations")
        advanced_layout.addWidget(self.tta_check)
        
        tta_layout = QHBoxLayout()
        tta_layout.addWidget(QLabel("TTA Augmentations:"))
        self.tta_aug_spin = QSpinBox()
        self.tta_aug_spin.setRange(2, 8)
        self.tta_aug_spin.setValue(4)
        self.tta_aug_spin.setEnabled(False)
        tta_layout.addWidget(self.tta_aug_spin)
        tta_layout.addStretch()
        advanced_layout.addLayout(tta_layout)
        
        self.tta_check.toggled.connect(self.tta_aug_spin.setEnabled)
        
        self.save_overlay_check = QCheckBox("Save Overlay Visualizations")
        self.save_overlay_check.setChecked(True)
        advanced_layout.addWidget(self.save_overlay_check)
        
        layout.addWidget(advanced_group)
        layout.addStretch()
        
        return widget
    
    def _create_postprocess_tab(self):
        """Create post-processing tab."""
        from PyQt5.QtWidgets import QWidget
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Enable post-processing
        self.postprocess_check = QCheckBox("Apply Post-processing")
        self.postprocess_check.setChecked(True)
        layout.addWidget(self.postprocess_check)
        
        # Post-processing options
        postprocess_group = QGroupBox("Post-processing Options")
        postprocess_layout = QGridLayout(postprocess_group)
        
        self.remove_small_check = QCheckBox("Remove Small Objects")
        self.remove_small_check.setChecked(True)
        postprocess_layout.addWidget(self.remove_small_check, 0, 0, 1, 2)
        
        postprocess_layout.addWidget(QLabel("Min Object Size:"), 1, 0)
        self.min_size_spin = QSpinBox()
        self.min_size_spin.setRange(10, 10000)
        self.min_size_spin.setValue(100)
        postprocess_layout.addWidget(self.min_size_spin, 1, 1)
        
        self.fill_holes_check = QCheckBox("Fill Holes")
        self.fill_holes_check.setChecked(True)
        postprocess_layout.addWidget(self.fill_holes_check, 2, 0, 1, 2)
        
        self.smooth_check = QCheckBox("Smooth Contours")
        self.smooth_check.setChecked(True)
        postprocess_layout.addWidget(self.smooth_check, 3, 0, 1, 2)
        
        postprocess_layout.addWidget(QLabel("Closing Kernel Size:"), 4, 0)
        self.closing_spin = QSpinBox()
        self.closing_spin.setRange(0, 21)
        self.closing_spin.setSingleStep(2)
        self.closing_spin.setValue(5)
        postprocess_layout.addWidget(self.closing_spin, 4, 1)
        
        # Enable/disable based on checkbox
        self.postprocess_check.toggled.connect(postprocess_group.setEnabled)
        
        layout.addWidget(postprocess_group)
        layout.addStretch()
        
        return widget
    
    def _create_monitor_tab(self):
        """Create monitoring tab."""
        from PyQt5.QtWidgets import QWidget
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Progress
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        self.progress_bar = QProgressBar()
        progress_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("Ready to predict")
        progress_layout.addWidget(self.status_label)
        
        layout.addWidget(progress_group)
        
        # Log
        log_group = QGroupBox("Inference Log")
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        
        layout.addWidget(log_group)
        
        return widget
    
    def _load_defaults(self):
        """Load default values."""
        from pathlib import Path
        
        # Set default paths
        project_root = Path(self.paths_config['paths']['data_root'])
        self.output_dir_edit.setText(str(project_root / "outputs/predictions"))
        
        # Try to find best checkpoint
        models_dir = project_root / "outputs/models"
        if models_dir.exists():
            best_model = models_dir / "best_model.pth"
            if best_model.exists():
                self.checkpoint_edit.setText(str(best_model))
    
    def _browse_checkpoint(self):
        """Browse for checkpoint file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Model Checkpoint",
            self.checkpoint_edit.text(),
            "Checkpoint Files (*.pth *.pt);;All Files (*)"
        )
        
        if file_path:
            self.checkpoint_edit.setText(file_path)
    
    def _browse_input_dir(self):
        """Browse for input directory."""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Input Directory",
            self.input_dir_edit.text()
        )
        
        if dir_path:
            self.input_dir_edit.setText(dir_path)
            # Auto-load images from directory
            self._load_images_from_dir(dir_path)
    
    def _load_images_from_dir(self, dir_path: str):
        """Load all images from directory."""
        from src.utils.file_utils import list_files
        
        image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        images = list_files(dir_path, extensions=image_exts, recursive=False)
        
        if images:
            self.image_paths.extend(images)
            self._update_image_list()
            self.log_text.append(f"Loaded {len(images)} images from {dir_path}")
    
    def _add_images(self):
        """Add individual images."""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Images",
            "",
            "Image Files (*.jpg *.jpeg *.png *.bmp *.tif *.tiff);;All Files (*)"
        )
        
        if file_paths:
            self.image_paths.extend(file_paths)
            self._update_image_list()
            self.log_text.append(f"Added {len(file_paths)} images")
    
    def _remove_selected(self):
        """Remove selected images from list."""
        selected_items = self.image_list.selectedItems()
        for item in selected_items:
            row = self.image_list.row(item)
            self.image_paths.pop(row)
            self.image_list.takeItem(row)
    
    def _clear_images(self):
        """Clear all images."""
        self.image_paths.clear()
        self.image_list.clear()
    
    def _update_image_list(self):
        """Update image list widget."""
        self.image_list.clear()
        for path in self.image_paths:
            self.image_list.addItem(Path(path).name)
    
    def _browse_dir(self, line_edit: QLineEdit):
        """Browse for directory."""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Directory",
            line_edit.text()
        )
        
        if dir_path:
            line_edit.setText(dir_path)
    
    def _start_inference(self):
        """Start inference."""
        # Validate
        if not self.checkpoint_edit.text():
            QMessageBox.warning(self, "Error", "Please select a model checkpoint")
            return
        
        if not self.image_paths:
            QMessageBox.warning(self, "Error", "Please add images to process")
            return
        
        if not self.output_dir_edit.text():
            QMessageBox.warning(self, "Error", "Please specify output directory")
            return
        
        # Prepare config
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
        
        # Create inference thread
        self.inference_thread = InferenceThread(
            checkpoint_path=self.checkpoint_edit.text(),
            image_paths=self.image_paths,
            output_dir=self.output_dir_edit.text(),
            config=inference_config
        )
        
        # Connect signals
        self.inference_thread.progress_updated.connect(self._on_progress_updated)
        self.inference_thread.image_completed.connect(self._on_image_completed)
        self.inference_thread.inference_completed.connect(self._on_inference_completed)
        self.inference_thread.inference_failed.connect(self._on_inference_failed)
        
        # Update UI
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("Inference started...")
        self.log_text.append("=== Inference Started ===")
        self.log_text.append(f"Total images: {len(self.image_paths)}")
        
        # Start
        self.inference_thread.start()
        
        logger.info("Inference started")
    
    def _stop_inference(self):
        """Stop inference."""
        if self.inference_thread:
            self.inference_thread.stop()
            self.stop_btn.setEnabled(False)
            self.status_label.setText("Stopping...")
            self.log_text.append("Stop requested...")
    
    def _on_progress_updated(self, current: int, total: int, image_path: str):
        """Handle progress update."""
        progress = int((current / total) * 100)
        self.progress_bar.setValue(progress)
        self.status_label.setText(f"Processing: {current}/{total} - {Path(image_path).name}")
    
    def _on_image_completed(self, index: int, image_path: str, success: bool):
        """Handle image completion."""
        status = "✓" if success else "✗"
        self.log_text.append(f"{status} {Path(image_path).name}")
    
    def _on_inference_completed(self, results: dict):
        """Handle inference completion."""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setValue(100)
        self.status_label.setText(f"✓ Inference completed! {results['successful']}/{results['total']} successful")
        
        self.log_text.append("=== Inference Completed ===")
        self.log_text.append(f"Successful: {results['successful']}")
        self.log_text.append(f"Failed: {results['failed']}")
        
        if results['failed_files']:
            self.log_text.append("\nFailed files:")
            for path in results['failed_files']:
                self.log_text.append(f"  - {Path(path).name}")
        
        QMessageBox.information(
            self,
            "Inference Complete",
            f"Inference completed successfully!\n\n"
            f"Successful: {results['successful']}/{results['total']}\n"
            f"Output: {self.output_dir_edit.text()}"
        )
    
    def _on_inference_failed(self, error_message: str):
        """Handle inference failure."""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("❌ Inference failed")
        self.log_text.append(f"ERROR: {error_message}")
        
        QMessageBox.critical(
            self,
            "Inference Failed",
            f"Inference failed with error:\n\n{error_message}"
        )
    
    def closeEvent(self, event):
        """Handle dialog close."""
        if self.inference_thread and self.inference_thread.isRunning():
            reply = QMessageBox.question(
                self,
                "Inference in Progress",
                "Inference is still running. Do you want to stop it?",
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


# Need torch for device check
import torch
