"""
Import dialog for importing images and videos.

Provides:
- Source selection (folder, files, video)
- Import preview
- Import options configuration
"""

from pathlib import Path
from typing import Optional, List
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QLineEdit, QFileDialog, QRadioButton,
    QButtonGroup, QSpinBox, QCheckBox, QGroupBox,
    QListWidget, QProgressBar, QTextEdit
)

from src.logger import get_logger
from src.utils.file_utils import list_files

logger = get_logger(__name__)


class ImportDialog(QDialog):
    """
    Dialog for importing images and videos.
    
    Signals:
        import_completed: Emitted when import is completed (imported_paths)
    """
    
    import_completed = pyqtSignal(list)
    
    def __init__(self, parent=None):
        """
        Initialize ImportDialog.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        
        self.setWindowTitle("Import Images/Videos")
        self.setModal(True)
        self.resize(600, 500)
        
        self.import_source: Optional[str] = None
        self.import_type = "folder"  # 'folder', 'files', 'video'
        self.imported_files: List[str] = []
        
        self._init_ui()
        
        logger.info("ImportDialog initialized")
    
    def _init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        
        # Source type selection
        source_group = QGroupBox("Import Source")
        source_layout = QVBoxLayout(source_group)
        
        self.source_button_group = QButtonGroup(self)
        
        self.folder_radio = QRadioButton("Image Folder")
        self.folder_radio.setChecked(True)
        self.folder_radio.toggled.connect(lambda: self._set_import_type("folder"))
        self.source_button_group.addButton(self.folder_radio)
        source_layout.addWidget(self.folder_radio)
        
        self.files_radio = QRadioButton("Image Files")
        self.files_radio.toggled.connect(lambda: self._set_import_type("files"))
        self.source_button_group.addButton(self.files_radio)
        source_layout.addWidget(self.files_radio)
        
        self.video_radio = QRadioButton("Video File")
        self.video_radio.toggled.connect(lambda: self._set_import_type("video"))
        self.source_button_group.addButton(self.video_radio)
        source_layout.addWidget(self.video_radio)
        
        layout.addWidget(source_group)
        
        # Source selection
        source_select_layout = QHBoxLayout()
        
        self.source_edit = QLineEdit()
        self.source_edit.setPlaceholderText("Select import source...")
        self.source_edit.setReadOnly(True)
        source_select_layout.addWidget(self.source_edit, 1)
        
        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self._browse_source)
        source_select_layout.addWidget(self.browse_btn)
        
        layout.addLayout(source_select_layout)
        
        # Options for folder import
        self.folder_options = QGroupBox("Folder Options")
        folder_options_layout = QVBoxLayout(self.folder_options)
        
        self.recursive_check = QCheckBox("Include subdirectories")
        self.recursive_check.setChecked(False)
        folder_options_layout.addWidget(self.recursive_check)
        
        layout.addWidget(self.folder_options)
        
        # Options for video import
        self.video_options = QGroupBox("Video Options")
        self.video_options.setVisible(False)
        video_options_layout = QVBoxLayout(self.video_options)
        
        frame_interval_layout = QHBoxLayout()
        frame_interval_layout.addWidget(QLabel("Extract every"))
        
        self.frame_interval_spin = QSpinBox()
        self.frame_interval_spin.setMinimum(1)
        self.frame_interval_spin.setMaximum(1000)
        self.frame_interval_spin.setValue(1)
        self.frame_interval_spin.setSuffix(" frame(s)")
        frame_interval_layout.addWidget(self.frame_interval_spin)
        frame_interval_layout.addStretch()
        
        video_options_layout.addLayout(frame_interval_layout)
        
        max_frames_layout = QHBoxLayout()
        self.max_frames_check = QCheckBox("Limit to")
        max_frames_layout.addWidget(self.max_frames_check)
        
        self.max_frames_spin = QSpinBox()
        self.max_frames_spin.setMinimum(1)
        self.max_frames_spin.setMaximum(100000)
        self.max_frames_spin.setValue(1000)
        self.max_frames_spin.setSuffix(" frame(s)")
        self.max_frames_spin.setEnabled(False)
        max_frames_layout.addWidget(self.max_frames_spin)
        max_frames_layout.addStretch()
        
        self.max_frames_check.toggled.connect(self.max_frames_spin.setEnabled)
        
        video_options_layout.addLayout(max_frames_layout)
        
        layout.addWidget(self.video_options)
        
        # Preview section
        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout(preview_group)
        
        self.preview_list = QListWidget()
        self.preview_list.setMaximumHeight(150)
        preview_layout.addWidget(self.preview_list)
        
        self.preview_info = QLabel("No files to import")
        self.preview_info.setStyleSheet("QLabel { color: #888; }")
        preview_layout.addWidget(self.preview_info)
        
        self.preview_btn = QPushButton("Update Preview")
        self.preview_btn.clicked.connect(self._update_preview)
        preview_layout.addWidget(self.preview_btn)
        
        layout.addWidget(preview_group)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.import_btn = QPushButton("Import")
        self.import_btn.setEnabled(False)
        self.import_btn.clicked.connect(self._do_import)
        button_layout.addWidget(self.import_btn)
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)
        
        layout.addLayout(button_layout)
    
    def _set_import_type(self, import_type: str):
        """
        Set the import type.
        
        Args:
            import_type: Type of import ('folder', 'files', 'video')
        """
        self.import_type = import_type
        
        # Update UI visibility
        self.folder_options.setVisible(import_type == "folder")
        self.video_options.setVisible(import_type == "video")
        
        # Clear source
        self.source_edit.clear()
        self.import_source = None
        self.preview_list.clear()
        self.import_btn.setEnabled(False)
        
        logger.debug(f"Import type changed to: {import_type}")
    
    def _browse_source(self):
        """Open file/folder browser."""
        if self.import_type == "folder":
            path = QFileDialog.getExistingDirectory(
                self,
                "Select Image Folder"
            )
        elif self.import_type == "files":
            paths, _ = QFileDialog.getOpenFileNames(
                self,
                "Select Image Files",
                "",
                "Images (*.jpg *.jpeg *.png *.bmp *.tiff *.tif);;All Files (*)"
            )
            path = ";".join(paths) if paths else None
        else:  # video
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Select Video File",
                "",
                "Videos (*.mp4 *.avi *.mov *.mkv *.flv);;All Files (*)"
            )
        
        if path:
            self.import_source = path
            self.source_edit.setText(path)
            self._update_preview()
            logger.info(f"Import source selected: {path}")
    
    def _update_preview(self):
        """Update the import preview."""
        self.preview_list.clear()
        self.imported_files.clear()
        
        if not self.import_source:
            self.preview_info.setText("No files to import")
            self.import_btn.setEnabled(False)
            return
        
        try:
            if self.import_type == "folder":
                self._preview_folder()
            elif self.import_type == "files":
                self._preview_files()
            else:  # video
                self._preview_video()
            
            # Enable import button
            self.import_btn.setEnabled(len(self.imported_files) > 0)
            
        except Exception as e:
            logger.error(f"Error updating preview: {e}")
            self.preview_info.setText(f"Error: {str(e)}")
    
    def _preview_folder(self):
        """Preview folder import."""
        folder = Path(self.import_source)
        
        if not folder.exists():
            self.preview_info.setText("Folder does not exist")
            return
        
        # Find image files
        recursive = self.recursive_check.isChecked()
        supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        
        self.imported_files = list_files(
            folder,
            extensions=supported_formats,
            recursive=recursive
        )
        
        # Update preview list (show first 100)
        for path in self.imported_files[:100]:
            self.preview_list.addItem(Path(path).name)
        
        if len(self.imported_files) > 100:
            self.preview_list.addItem(f"... and {len(self.imported_files) - 100} more")
        
        self.preview_info.setText(f"Found {len(self.imported_files)} image(s)")
    
    def _preview_files(self):
        """Preview files import."""
        paths = self.import_source.split(";")
        self.imported_files = paths
        
        # Update preview list
        for path in paths:
            self.preview_list.addItem(Path(path).name)
        
        self.preview_info.setText(f"Selected {len(paths)} file(s)")
    
    def _preview_video(self):
        """Preview video import."""
        video_path = Path(self.import_source)
        
        if not video_path.exists():
            self.preview_info.setText("Video file does not exist")
            return
        
        # Get video info
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            self.preview_info.setText("Failed to open video file")
            return
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        # Calculate frames to extract
        frame_interval = self.frame_interval_spin.value()
        max_frames = self.max_frames_spin.value() if self.max_frames_check.isChecked() else None
        
        estimated_frames = total_frames // frame_interval
        if max_frames is not None:
            estimated_frames = min(estimated_frames, max_frames)
        
        # Store video path (frames will be extracted during import)
        self.imported_files = [str(video_path)]
        
        # Update preview
        self.preview_list.addItem(f"Video: {video_path.name}")
        self.preview_list.addItem(f"Total frames: {total_frames}")
        self.preview_list.addItem(f"FPS: {fps:.2f}")
        self.preview_list.addItem(f"Frames to extract: ~{estimated_frames}")
        
        self.preview_info.setText(f"Will extract ~{estimated_frames} frame(s)")
    
    def _do_import(self):
        """Execute the import."""
        if not self.imported_files:
            return
        
        logger.info(f"Starting import: {len(self.imported_files)} item(s)")
        
        # Show progress bar
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(len(self.imported_files))
        self.progress_bar.setValue(0)
        
        # Disable buttons
        self.import_btn.setEnabled(False)
        self.browse_btn.setEnabled(False)
        
        # For video, we would need to extract frames here
        # For now, just emit the file list
        
        if self.import_type == "video":
            # Note: Actual frame extraction would be done by DataManager
            logger.info("Video import requested - frames will be extracted by DataManager")
        
        # Update progress
        for i in range(len(self.imported_files)):
            self.progress_bar.setValue(i + 1)
        
        # Emit signal
        self.import_completed.emit(self.imported_files)
        
        logger.info("Import completed")
        
        # Accept dialog
        self.accept()
    
    def get_import_files(self) -> List[str]:
        """
        Get the list of files to import.
        
        Returns:
            List of file paths
        """
        return self.imported_files.copy()
    
    def get_video_options(self) -> dict:
        """
        Get video extraction options.
        
        Returns:
            Dictionary with video options
        """
        return {
            'frame_interval': self.frame_interval_spin.value(),
            'max_frames': self.max_frames_spin.value() if self.max_frames_check.isChecked() else None
        }
