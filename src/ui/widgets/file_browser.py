"""
用于浏览和选择图像的文件浏览器小部件。

提供：
- 带有缩略图的文件列表视图
- 文件夹导航
- 文件过滤和搜索
- 选择管理
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
    """用于异步加载缩略图的线程。"""
    
    thumbnail_loaded = pyqtSignal(str, QPixmap)  # 路径, 像素图
    
    def __init__(self, image_paths: List[str], thumb_size: int = 128):
        super().__init__()
        self.image_paths = image_paths
        self.thumb_size = thumb_size
        self._is_running = True
    
    def run(self):
        """为所有图像加载缩略图。"""
        for path in self.image_paths:
            if not self._is_running:
                break
            
            try:
                # 加载图像
                image = load_image(path)
                if image is None:
                    continue
                
                # 缩放到缩略图大小
                h, w = image.shape[:2]
                scale = min(self.thumb_size / w, self.thumb_size / h)
                new_w, new_h = int(w * scale), int(h * scale)
                
                import cv2
                thumb = cv2.resize(image, (new_w, new_h))
                
                # 转换为 QPixmap
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
                logger.error(f"加载 {path} 的缩略图时出错: {e}")
    
    def stop(self):
        """停止加载缩略图。"""
        self._is_running = False


class FileBrowser(QWidget):
    """
    用于图像选择的文件浏览器小部件。
    
    信号:
        file_selected: 选择文件时发出 (file_path)
        files_selected: 选择多个文件时发出 (file_paths)
        folder_changed: 文件夹更改时发出 (folder_path)
    """
    
    file_selected = pyqtSignal(str)
    files_selected = pyqtSignal(list)
    folder_changed = pyqtSignal(str)
    
    def __init__(self, parent: Optional[QWidget] = None):
        """
        初始化 FileBrowser。
        
        参数:
            parent: 父小部件
        """
        super().__init__(parent)
        
        self.current_folder: Optional[Path] = None
        self.file_paths: List[str] = []
        self.filtered_paths: List[str] = []
        
        # 支持的图像格式
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        
        # 缩略图加载器
        self.thumb_loader: Optional[ThumbnailLoader] = None
        
        self._init_ui()
        
        logger.info("FileBrowser 已初始化")
    
    def _init_ui(self):
        """初始化用户界面。"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # 文件夹选择
        folder_layout = QHBoxLayout()
        
        self.folder_label = QLabel("未选择文件夹")
        self.folder_label.setStyleSheet("QLabel { color: #888; }")
        folder_layout.addWidget(self.folder_label, 1)
        
        self.select_folder_btn = QPushButton("浏览...")
        self.select_folder_btn.clicked.connect(self._select_folder)
        folder_layout.addWidget(self.select_folder_btn)
        
        layout.addLayout(folder_layout)
        
        # 搜索和过滤
        search_layout = QHBoxLayout()
        
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("搜索文件...")
        self.search_edit.textChanged.connect(self._filter_files)
        search_layout.addWidget(self.search_edit, 1)
        
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["所有图像", "JPG", "PNG", "BMP", "TIFF"])
        self.filter_combo.currentTextChanged.connect(self._filter_files)
        search_layout.addWidget(self.filter_combo)
        
        layout.addLayout(search_layout)
        
        # 文件列表
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.file_list.setIconSize(QSize(128, 128))
        self.file_list.setViewMode(QListWidget.IconMode)
        self.file_list.setResizeMode(QListWidget.Adjust)
        self.file_list.setSpacing(10)
        self.file_list.itemSelectionChanged.connect(self._on_selection_changed)
        self.file_list.itemDoubleClicked.connect(self._on_item_double_clicked)
        layout.addWidget(self.file_list)
        
        # 信息标签
        self.info_label = QLabel("0 个文件")
        self.info_label.setStyleSheet("QLabel { color: #888; font-size: 11px; }")
        layout.addWidget(self.info_label)
    
    def set_folder(self, folder_path: str):
        """
        设置当前文件夹并加载图像。
        
        参数:
            folder_path: 包含图像的文件夹路径
        """
        folder = Path(folder_path)
        
        if not folder.exists() or not folder.is_dir():
            logger.error(f"无效文件夹: {folder_path}")
            return
        
        self.current_folder = folder
        self.folder_label.setText(str(folder))
        
        # 停止之前的缩略图加载器
        if self.thumb_loader is not None:
            self.thumb_loader.stop()
            self.thumb_loader.wait()
        
        # 加载文件列表
        self._load_files()
        
        # 发出信号
        self.folder_changed.emit(str(folder))
        
        logger.info(f"文件夹已更改为: {folder_path}")
    
    def _select_folder(self):
        """打开文件夹选择对话框。"""
        folder = QFileDialog.getExistingDirectory(
            self,
            "选择图像文件夹",
            str(self.current_folder) if self.current_folder else ""
        )
        
        if folder:
            self.set_folder(folder)
    
    def _load_files(self):
        """从当前文件夹加载文件。"""
        if self.current_folder is None:
            return
        
        # 清除列表
        self.file_list.clear()
        self.file_paths.clear()
        
        # 查找图像文件
        file_paths = list_files(
            self.current_folder,
            extensions=self.supported_formats,
            recursive=False
        )
        
        # 转换 Path 对象为字符串
        self.file_paths = [str(p) for p in file_paths]
        
        # 按文件名排序
        self.file_paths.sort()
        
        # 应用过滤
        self._filter_files()
        
        logger.info(f"已加载 {len(self.file_paths)} 个文件")
    
    def _filter_files(self):
        """根据搜索文本和格式过滤器过滤文件。"""
        if not self.file_paths:
            self.filtered_paths = []
            self._update_file_list()
            return
        
        # 获取过滤标准
        search_text = self.search_edit.text().lower()
        format_filter = self.filter_combo.currentText()
        
        # 应用过滤
        self.filtered_paths = []
        for path in self.file_paths:
            path_obj = Path(path)
            filename = path_obj.name.lower()
            
            # 检查搜索文本
            if search_text and search_text not in filename:
                continue
            
            # 检查格式过滤器
            if format_filter != "所有图像":
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
        
        # 更新 UI
        self._update_file_list()
    
    def _update_file_list(self):
        """更新文件列表小部件。"""
        self.file_list.clear()
        
        # 添加项目
        for path in self.filtered_paths:
            item = QListWidgetItem(Path(path).name)
            item.setData(Qt.UserRole, path)  # 存储完整路径
            self.file_list.addItem(item)
        
        # 更新信息标签
        self.info_label.setText(f"{len(self.filtered_paths)} 个文件")
        
        # 开始加载缩略图
        if self.filtered_paths:
            self._load_thumbnails()
    
    def _load_thumbnails(self):
        """为可见文件加载缩略图。"""
        # 停止之前的加载器
        if self.thumb_loader is not None:
            self.thumb_loader.stop()
            self.thumb_loader.wait()
        
        # 开始新的加载器
        self.thumb_loader = ThumbnailLoader(self.filtered_paths, thumb_size=128)
        self.thumb_loader.thumbnail_loaded.connect(self._set_thumbnail)
        self.thumb_loader.start()
    
    def _set_thumbnail(self, path: str, pixmap: QPixmap):
        """
        为文件项目设置缩略图。
        
        参数:
            path: 文件路径
            pixmap: 缩略图像素图
        """
        # 查找具有此路径的项目
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            if item.data(Qt.UserRole) == path:
                item.setIcon(QIcon(pixmap))
                break
    
    def _on_selection_changed(self):
        """处理选择更改。"""
        selected_items = self.file_list.selectedItems()
        
        if len(selected_items) == 0:
            return
        
        # 获取选定的路径
        selected_paths = [item.data(Qt.UserRole) for item in selected_items]
        
        # 总是发出 file_selected 信号（单个选择）
        if len(selected_paths) == 1:
            self.file_selected.emit(selected_paths[0])
            logger.debug(f"文件已选择: {selected_paths[0]}")
        
        # 同时发出多选信号
        if len(selected_paths) > 0:
            self.files_selected.emit(selected_paths)
    
    def _on_item_double_clicked(self, item: QListWidgetItem):
        """处理项目双击。"""
        path = item.data(Qt.UserRole)
        self.file_selected.emit(path)
        logger.info(f"文件已双击: {path}")
    
    def get_selected_files(self) -> List[str]:
        """
        获取选定文件路径的列表。
        
        返回:
            选定文件路径的列表
        """
        selected_items = self.file_list.selectedItems()
        return [item.data(Qt.UserRole) for item in selected_items]
    
    def get_all_files(self) -> List[str]:
        """
        获取所有文件路径的列表（过滤后）。
        
        返回:
            所有文件路径的列表
        """
        return self.filtered_paths.copy()
    
    def refresh(self):
        """刷新文件列表。"""
        if self.current_folder:
            self._load_files()
    
    def clear(self):
        """清除文件列表。"""
        self.file_list.clear()
        self.file_paths.clear()
        self.filtered_paths.clear()
        self.current_folder = None
        self.folder_label.setText("未选择文件夹")
        self.info_label.setText("0 个文件")
