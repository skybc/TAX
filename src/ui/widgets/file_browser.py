"""
File browser widget for browsing and selecting images.

Provides:
- File list view with thumbnails
- Folder navigation
- File filtering and search
- Selection management
"""

from pathlib import Path
from typing import Optional, List
from PyQt5.QtCore import Qt, QSize, pyqtSignal, QThread
from PyQt5.QtGui import QIcon, QPixmap, QImage
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, 
    QListWidgetItem, QPushButton, QLineEdit, QLabel,
    QFileDialog, QComboBox, QSplitter
)
import numpy as np

from src.logger import get_logger
from src.utils.file_utils import list_files
from src.utils.image_utils import load_image

logger = get_logger(__name__)


class ThumbnailLoader(QThread):
    """Thread for loading thumbnails asynchronously."""
    
    thumbnail_loaded = pyqtSignal(str, QPixmap)  # path, pixmap
    
    def __init__(self, image_paths: List[str], thumb_size: int = 128):
        super().__init__()
        self.image_paths = image_paths
        self.thumb_size = thumb_size
        self._is_running = True
    
    def run(self):
        """Load thumbnails for all images."""
        for path in self.image_paths:
            if not self._is_running:
                break
            
            try:
                # Load image
                image = load_image(path)
                if image is None:
                    continue
                
                # Resize to thumbnail
                h, w = image.shape[:2]
                scale = min(self.thumb_size / w, self.thumb_size / h)
                new_w, new_h = int(w * scale), int(h * scale)
                
                import cv2
                thumb = cv2.resize(image, (new_w, new_h))
                
                # Convert to QPixmap
                height, width = thumb.shape[:2]
                if len(thumb.shape) == 2:
                    qimage = QImage(
                        thumb.data, width, height, width,
                        QImage.Format_Grayscale8
                    )
                else:
                    bytes_per_line = 3 * width
                    qimage = QImage(
                        thumb.data, width, height, bytes_per_line,
                        QImage.Format_RGB888
                    )
                
                pixmap = QPixmap.fromImage(qimage)
                self.thumbnail_loaded.emit(path, pixmap)
                
            except Exception as e:
                logger.error(f"Error loading thumbnail for {path}: {e}")
    
    def stop(self):
        """Stop loading thumbnails."""
        self._is_running = False


class FileBrowser(QWidget):
    """
    File browser widget for image selection.
    
    Signals:
        file_selected: Emitted when a file is selected (file_path)
        files_selected: Emitted when multiple files are selected (file_paths)
        folder_changed: Emitted when folder changes (folder_path)
    """
    
    file_selected = pyqtSignal(str)
    files_selected = pyqtSignal(list)
    folder_changed = pyqtSignal(str)
    
    def __init__(self, parent: Optional[QWidget] = None):
        """
        Initialize FileBrowser.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        
        self.current_folder: Optional[Path] = None
        self.file_paths: List[str] = []
        self.filtered_paths: List[str] = []
        
        # Supported image formats
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        
        # Thumbnail loader
        self.thumb_loader: Optional[ThumbnailLoader] = None
        
        self._init_ui()
        
        logger.info("FileBrowser initialized")
    
    def _init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Folder selection
        folder_layout = QHBoxLayout()
        
        self.folder_label = QLabel("No folder selected")
        self.folder_label.setStyleSheet("QLabel { color: #888; }")
        folder_layout.addWidget(self.folder_label, 1)
        
        self.select_folder_btn = QPushButton("Browse...")
        self.select_folder_btn.clicked.connect(self._select_folder)
        folder_layout.addWidget(self.select_folder_btn)
        
        layout.addLayout(folder_layout)
        
        # Search and filter
        search_layout = QHBoxLayout()
        
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search files...")
        self.search_edit.textChanged.connect(self._filter_files)
        search_layout.addWidget(self.search_edit, 1)
        
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["All Images", "JPG", "PNG", "BMP", "TIFF"])
        self.filter_combo.currentTextChanged.connect(self._filter_files)
        search_layout.addWidget(self.filter_combo)
        
        layout.addLayout(search_layout)
        
        # File list
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.file_list.setIconSize(QSize(128, 128))
        self.file_list.setViewMode(QListWidget.IconMode)
        self.file_list.setResizeMode(QListWidget.Adjust)
        self.file_list.setSpacing(10)
        self.file_list.itemSelectionChanged.connect(self._on_selection_changed)
        self.file_list.itemDoubleClicked.connect(self._on_item_double_clicked)
        layout.addWidget(self.file_list)
        
        # Info label
        self.info_label = QLabel("0 files")
        self.info_label.setStyleSheet("QLabel { color: #888; font-size: 11px; }")
        layout.addWidget(self.info_label)
    
    def set_folder(self, folder_path: str):
        """
        Set the current folder and load images.
        
        Args:
            folder_path: Path to folder containing images
        """
        folder = Path(folder_path)
        
        if not folder.exists() or not folder.is_dir():
            logger.error(f"Invalid folder: {folder_path}")
            return
        
        self.current_folder = folder
        self.folder_label.setText(str(folder))
        
        # Stop previous thumbnail loader
        if self.thumb_loader is not None:
            self.thumb_loader.stop()
            self.thumb_loader.wait()
        
        # Load file list
        self._load_files()
        
        # Emit signal
        self.folder_changed.emit(str(folder))
        
        logger.info(f"Folder changed to: {folder_path}")
    
    def _select_folder(self):
        """Open folder selection dialog."""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Image Folder",
            str(self.current_folder) if self.current_folder else ""
        )
        
        if folder:
            self.set_folder(folder)
    
    def _load_files(self):
        """Load files from current folder."""
        if self.current_folder is None:
            return
        
        # Clear list
        self.file_list.clear()
        self.file_paths.clear()
        
        # Find image files
        self.file_paths = list_files(
            self.current_folder,
            extensions=self.supported_formats,
            recursive=False
        )
        
        # Sort by filename
        self.file_paths.sort()
        
        # Apply filter
        self._filter_files()
        
        logger.info(f"Loaded {len(self.file_paths)} files")
    
    def _filter_files(self):
        """Filter files based on search text and format filter."""
        if not self.file_paths:
            self.filtered_paths = []
            self._update_file_list()
            return
        
        # Get filter criteria
        search_text = self.search_edit.text().lower()
        format_filter = self.filter_combo.currentText()
        
        # Apply filters
        self.filtered_paths = []
        for path in self.file_paths:
            path_obj = Path(path)
            filename = path_obj.name.lower()
            
            # Check search text
            if search_text and search_text not in filename:
                continue
            
            # Check format filter
            if format_filter != "All Images":
                ext = path_obj.suffix.lower()
                if format_filter == "JPG" and ext not in ['.jpg', '.jpeg']:
                    continue
                elif format_filter == "PNG" and ext != '.png':
                    continue
                elif format_filter == "BMP" and ext != '.bmp':
                    continue
                elif format_filter == "TIFF" and ext not in ['.tiff', '.tif']:
                    continue
            
            self.filtered_paths.append(path)
        
        # Update UI
        self._update_file_list()
    
    def _update_file_list(self):
        """Update the file list widget."""
        self.file_list.clear()
        
        # Add items
        for path in self.filtered_paths:
            item = QListWidgetItem(Path(path).name)
            item.setData(Qt.UserRole, path)  # Store full path
            self.file_list.addItem(item)
        
        # Update info label
        self.info_label.setText(f"{len(self.filtered_paths)} files")
        
        # Start thumbnail loading
        if self.filtered_paths:
            self._load_thumbnails()
    
    def _load_thumbnails(self):
        """Load thumbnails for visible files."""
        # Stop previous loader
        if self.thumb_loader is not None:
            self.thumb_loader.stop()
            self.thumb_loader.wait()
        
        # Start new loader
        self.thumb_loader = ThumbnailLoader(self.filtered_paths, thumb_size=128)
        self.thumb_loader.thumbnail_loaded.connect(self._set_thumbnail)
        self.thumb_loader.start()
    
    def _set_thumbnail(self, path: str, pixmap: QPixmap):
        """
        Set thumbnail for a file item.
        
        Args:
            path: File path
            pixmap: Thumbnail pixmap
        """
        # Find item with this path
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            if item.data(Qt.UserRole) == path:
                item.setIcon(QIcon(pixmap))
                break
    
    def _on_selection_changed(self):
        """Handle selection change."""
        selected_items = self.file_list.selectedItems()
        
        if len(selected_items) == 0:
            return
        
        # Get selected paths
        selected_paths = [item.data(Qt.UserRole) for item in selected_items]
        
        # Emit signals
        if len(selected_paths) == 1:
            self.file_selected.emit(selected_paths[0])
        
        self.files_selected.emit(selected_paths)
    
    def _on_item_double_clicked(self, item: QListWidgetItem):
        """Handle item double click."""
        path = item.data(Qt.UserRole)
        self.file_selected.emit(path)
        logger.info(f"File double-clicked: {path}")
    
    def get_selected_files(self) -> List[str]:
        """
        Get list of selected file paths.
        
        Returns:
            List of selected file paths
        """
        selected_items = self.file_list.selectedItems()
        return [item.data(Qt.UserRole) for item in selected_items]
    
    def get_all_files(self) -> List[str]:
        """
        Get list of all file paths (after filtering).
        
        Returns:
            List of all file paths
        """
        return self.filtered_paths.copy()
    
    def refresh(self):
        """Refresh the file list."""
        if self.current_folder:
            self._load_files()
    
    def clear(self):
        """Clear the file list."""
        self.file_list.clear()
        self.file_paths.clear()
        self.filtered_paths.clear()
        self.current_folder = None
        self.folder_label.setText("No folder selected")
        self.info_label.setText("0 files")
