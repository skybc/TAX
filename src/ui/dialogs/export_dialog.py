"""
Export dialog for batch annotation export.

This module provides:
- Format selection (COCO/YOLO/VOC)
- Export options configuration
- Progress tracking
- Validation and report generation
"""

from pathlib import Path
from typing import List, Dict, Optional
from PyQt5.QtCore import Qt, pyqtSignal, QThread
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QComboBox, QLineEdit, QPushButton,
    QCheckBox, QGroupBox, QProgressBar, QTextEdit,
    QFileDialog, QMessageBox, QSpinBox
)

from src.logger import get_logger
from src.utils.export_utils import COCOExporter, YOLOExporter, VOCExporter
from src.utils.dataset_validator import validate_coco_dataset, validate_yolo_dataset

logger = get_logger(__name__)


class ExportWorkerThread(QThread):
    """
    Worker thread for batch export operations.
    
    Signals:
        progress_updated: Emitted during export (current, total, message)
        export_completed: Emitted when export completes (success, message)
        export_failed: Emitted when export fails (error_message)
    """
    
    progress_updated = pyqtSignal(int, int, str)  # current, total, message
    export_completed = pyqtSignal(bool, str)  # success, message
    export_failed = pyqtSignal(str)  # error message
    
    def __init__(self, 
                 export_format: str,
                 image_paths: List[str],
                 mask_paths: List[str],
                 output_dir: str,
                 options: Dict):
        """
        Initialize export worker thread.
        
        Args:
            export_format: Export format ('coco', 'yolo', 'voc')
            image_paths: List of image file paths
            mask_paths: List of mask file paths
            output_dir: Output directory
            options: Export options dictionary
        """
        super().__init__()
        
        self.export_format = export_format
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.output_dir = output_dir
        self.options = options
        
        self._is_running = True
    
    def run(self):
        """Execute export operation."""
        try:
            total = len(self.image_paths)
            
            if self.export_format == 'coco':
                self._export_coco(total)
            elif self.export_format == 'yolo':
                self._export_yolo(total)
            elif self.export_format == 'voc':
                self._export_voc(total)
            else:
                self.export_failed.emit(f"Unknown format: {self.export_format}")
                
        except Exception as e:
            logger.error(f"Export error: {e}", exc_info=True)
            self.export_failed.emit(str(e))
    
    def _export_coco(self, total: int):
        """Export to COCO format."""
        from src.utils.export_utils import batch_export_coco
        
        self.progress_updated.emit(0, total, "Initializing COCO export...")
        
        # Get options
        dataset_name = self.options.get('dataset_name', 'Industrial Defect Dataset')
        category_names = self.options.get('category_names', ['defect'] * len(self.image_paths))
        
        # Output path
        output_path = Path(self.output_dir) / "annotations.json"
        
        # Export
        stats = batch_export_coco(
            self.image_paths,
            self.mask_paths,
            category_names,
            str(output_path),
            dataset_name
        )
        
        if not self._is_running:
            return
        
        self.progress_updated.emit(total, total, "Validating COCO format...")
        
        # Validate if requested
        if self.options.get('validate', True):
            result = validate_coco_dataset(str(output_path))
            if not result.is_valid:
                self.export_completed.emit(False, f"Export completed with validation errors:\n{result.get_report()}")
                return
        
        message = f"COCO export successful!\n\n"
        message += f"Output: {output_path}\n"
        message += f"Images: {stats['num_images']}\n"
        message += f"Annotations: {stats['num_annotations']}\n"
        message += f"Categories: {stats['num_categories']}"
        
        self.export_completed.emit(True, message)
    
    def _export_yolo(self, total: int):
        """Export to YOLO format."""
        from src.utils.export_utils import batch_export_yolo
        from src.utils.image_utils import get_image_info
        
        self.progress_updated.emit(0, total, "Initializing YOLO export...")
        
        # Get options
        class_names = self.options.get('class_names', ['defect'])
        class_ids = self.options.get('class_ids', [0] * len(self.image_paths))
        
        # Export
        count = batch_export_yolo(
            self.image_paths,
            self.mask_paths,
            class_ids,
            class_names,
            self.output_dir
        )
        
        if not self._is_running:
            return
        
        self.progress_updated.emit(total, total, "Creating data.yaml...")
        
        # Create data.yaml if requested
        if self.options.get('create_yaml', True):
            from src.utils.export_utils import YOLOExporter
            exporter = YOLOExporter(self.output_dir, class_names)
            exporter.create_data_yaml(
                train_path=self.options.get('train_path', 'images/train'),
                val_path=self.options.get('val_path', 'images/val'),
                test_path=self.options.get('test_path')
            )
        
        # Validate if requested
        if self.options.get('validate', True):
            classes_file = Path(self.output_dir) / "classes.txt"
            result = validate_yolo_dataset(self.output_dir, str(classes_file))
            if not result.is_valid:
                self.export_completed.emit(False, f"Export completed with validation errors:\n{result.get_report()}")
                return
        
        message = f"YOLO export successful!\n\n"
        message += f"Output: {self.output_dir}\n"
        message += f"Annotations: {count}\n"
        message += f"Classes: {len(class_names)}"
        
        self.export_completed.emit(True, message)
    
    def _export_voc(self, total: int):
        """Export to VOC format."""
        self.progress_updated.emit(0, total, "Initializing VOC export...")
        
        exporter = VOCExporter(self.output_dir)
        
        from src.utils.image_utils import get_image_info
        from src.utils.mask_utils import load_mask
        
        category_names = self.options.get('category_names', ['defect'] * len(self.image_paths))
        
        for i, (image_path, mask_path, category) in enumerate(zip(self.image_paths, self.mask_paths, category_names)):
            if not self._is_running:
                return
            
            self.progress_updated.emit(i, total, f"Exporting {i+1}/{total}...")
            
            # Get image info
            info = get_image_info(image_path)
            if info is None:
                continue
            
            # Load mask
            mask = load_mask(mask_path)
            if mask is None:
                continue
            
            # Export
            exporter.export_annotation(
                image_path,
                [mask],
                [category],
                info['width'],
                info['height']
            )
        
        message = f"VOC export successful!\n\n"
        message += f"Output: {self.output_dir}\n"
        message += f"Annotations: {len(self.image_paths)}"
        
        self.export_completed.emit(True, message)
    
    def stop(self):
        """Stop the export thread."""
        self._is_running = False


class ExportDialog(QDialog):
    """
    Dialog for batch annotation export.
    
    Provides interface for:
    - Format selection (COCO/YOLO/VOC)
    - Output directory selection
    - Export options configuration
    - Progress tracking
    - Validation
    """
    
    def __init__(self, 
                 image_paths: List[str],
                 mask_paths: List[str],
                 parent=None):
        """
        Initialize export dialog.
        
        Args:
            image_paths: List of image file paths
            mask_paths: List of mask file paths
            parent: Parent widget
        """
        super().__init__(parent)
        
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.worker_thread: Optional[ExportWorkerThread] = None
        
        self.setWindowTitle("Export Annotations")
        self.setMinimumWidth(600)
        self.setMinimumHeight(500)
        
        self._init_ui()
        self._connect_signals()
        
        logger.info(f"Export dialog opened: {len(image_paths)} images")
    
    def _init_ui(self):
        """Initialize UI components."""
        layout = QVBoxLayout(self)
        
        # Format selection
        format_group = QGroupBox("Export Format")
        format_layout = QVBoxLayout(format_group)
        
        self.format_combo = QComboBox()
        self.format_combo.addItems(["COCO JSON", "YOLO txt", "Pascal VOC XML"])
        format_layout.addWidget(self.format_combo)
        
        format_desc = QLabel()
        format_desc.setWordWrap(True)
        format_desc.setStyleSheet("color: gray; font-size: 10pt;")
        self.format_combo.currentIndexChanged.connect(
            lambda i: format_desc.setText(self._get_format_description(i))
        )
        format_desc.setText(self._get_format_description(0))
        format_layout.addWidget(format_desc)
        
        layout.addWidget(format_group)
        
        # Output directory
        output_group = QGroupBox("Output")
        output_layout = QGridLayout(output_group)
        
        output_layout.addWidget(QLabel("Output Directory:"), 0, 0)
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setPlaceholderText("Select output directory...")
        output_layout.addWidget(self.output_dir_edit, 0, 1)
        
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_output_dir)
        output_layout.addWidget(browse_btn, 0, 2)
        
        layout.addWidget(output_group)
        
        # Options
        options_group = QGroupBox("Options")
        options_layout = QVBoxLayout(options_group)
        
        self.dataset_name_edit = QLineEdit("Industrial Defect Dataset")
        options_layout.addWidget(QLabel("Dataset Name:"))
        options_layout.addWidget(self.dataset_name_edit)
        
        self.class_name_edit = QLineEdit("defect")
        options_layout.addWidget(QLabel("Class Name:"))
        options_layout.addWidget(self.class_name_edit)
        
        self.validate_checkbox = QCheckBox("Validate after export")
        self.validate_checkbox.setChecked(True)
        options_layout.addWidget(self.validate_checkbox)
        
        self.create_yaml_checkbox = QCheckBox("Create data.yaml (YOLO only)")
        self.create_yaml_checkbox.setChecked(True)
        options_layout.addWidget(self.create_yaml_checkbox)
        
        layout.addWidget(options_group)
        
        # Progress
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        progress_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("Ready to export")
        self.status_label.setStyleSheet("color: gray;")
        progress_layout.addWidget(self.status_label)
        
        layout.addWidget(progress_group)
        
        # Log/result
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMaximumHeight(150)
        self.result_text.setVisible(False)
        layout.addWidget(self.result_text)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.export_btn = QPushButton("Export")
        self.export_btn.setDefault(True)
        self.export_btn.clicked.connect(self._start_export)
        button_layout.addWidget(self.export_btn)
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)
        
        layout.addLayout(button_layout)
        
        # Info label
        info_label = QLabel(f"Total: {len(self.image_paths)} images, {len(self.mask_paths)} masks")
        info_label.setStyleSheet("color: gray; font-size: 9pt;")
        layout.addWidget(info_label)
    
    def _connect_signals(self):
        """Connect signals and slots."""
        pass
    
    def _get_format_description(self, index: int) -> str:
        """Get description for format."""
        descriptions = {
            0: "COCO JSON: Standard format for object detection/segmentation. "
               "Outputs single JSON file with RLE-encoded masks.",
            1: "YOLO txt: Format for YOLO training. "
               "Outputs one txt file per image with normalized polygon coordinates.",
            2: "Pascal VOC XML: Standard format with XML annotations. "
               "Outputs XML files and PNG segmentation masks."
        }
        return descriptions.get(index, "")
    
    def _browse_output_dir(self):
        """Browse for output directory."""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            "",
            QFileDialog.ShowDirsOnly
        )
        
        if dir_path:
            self.output_dir_edit.setText(dir_path)
    
    def _start_export(self):
        """Start export operation."""
        # Validate inputs
        output_dir = self.output_dir_edit.text().strip()
        if not output_dir:
            QMessageBox.warning(self, "Error", "Please select output directory")
            return
        
        if not self.image_paths or not self.mask_paths:
            QMessageBox.warning(self, "Error", "No images or masks to export")
            return
        
        if len(self.image_paths) != len(self.mask_paths):
            QMessageBox.warning(self, "Error", "Number of images and masks don't match")
            return
        
        # Get format
        format_map = {0: 'coco', 1: 'yolo', 2: 'voc'}
        export_format = format_map[self.format_combo.currentIndex()]
        
        # Prepare options
        options = {
            'dataset_name': self.dataset_name_edit.text().strip(),
            'class_names': [self.class_name_edit.text().strip()],
            'category_names': [self.class_name_edit.text().strip()] * len(self.image_paths),
            'class_ids': [0] * len(self.image_paths),
            'validate': self.validate_checkbox.isChecked(),
            'create_yaml': self.create_yaml_checkbox.isChecked(),
            'train_path': 'images/train',
            'val_path': 'images/val'
        }
        
        # Create worker thread
        self.worker_thread = ExportWorkerThread(
            export_format,
            self.image_paths,
            self.mask_paths,
            output_dir,
            options
        )
        
        # Connect signals
        self.worker_thread.progress_updated.connect(self._on_progress)
        self.worker_thread.export_completed.connect(self._on_export_completed)
        self.worker_thread.export_failed.connect(self._on_export_failed)
        
        # Update UI
        self.export_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.result_text.setVisible(False)
        self.status_label.setText("Starting export...")
        
        # Start export
        self.worker_thread.start()
        
        logger.info(f"Started export: format={export_format}, output={output_dir}")
    
    def _on_progress(self, current: int, total: int, message: str):
        """Handle progress update."""
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.status_label.setText(message)
    
    def _on_export_completed(self, success: bool, message: str):
        """Handle export completion."""
        self.export_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        if success:
            self.status_label.setText("✅ Export completed successfully!")
            self.status_label.setStyleSheet("color: green;")
            
            self.result_text.setText(message)
            self.result_text.setVisible(True)
            
            QMessageBox.information(self, "Success", message)
            self.accept()
        else:
            self.status_label.setText("⚠️ Export completed with warnings")
            self.status_label.setStyleSheet("color: orange;")
            
            self.result_text.setText(message)
            self.result_text.setVisible(True)
            
            QMessageBox.warning(self, "Validation Issues", message)
    
    def _on_export_failed(self, error_message: str):
        """Handle export failure."""
        self.export_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("❌ Export failed")
        self.status_label.setStyleSheet("color: red;")
        
        self.result_text.setText(f"Error: {error_message}")
        self.result_text.setVisible(True)
        
        QMessageBox.critical(self, "Export Failed", error_message)
    
    def closeEvent(self, event):
        """Handle dialog close."""
        if self.worker_thread and self.worker_thread.isRunning():
            reply = QMessageBox.question(
                self,
                "Export in Progress",
                "Export is still running. Do you want to cancel it?",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.worker_thread.stop()
                self.worker_thread.wait()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()
