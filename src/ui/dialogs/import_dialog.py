"""
用于导入图像和视频的导入对话框。

提供：
- 源选择（文件夹、文件、视频）
- 导入预览
- 导入选项配置
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
    用于导入图像和视频的对话框。
    
    信号:
        import_completed: 导入完成时发出（imported_paths）
    """
    
    import_completed = pyqtSignal(list)
    
    def __init__(self, parent=None):
        """
        初始化 ImportDialog。
        
        参数:
            parent: 父小部件
        """
        super().__init__(parent)
        
        self.setWindowTitle("导入图像/视频")
        self.setModal(True)
        self.resize(600, 500)
        
        self.import_source: Optional[str] = None
        self.import_type = "folder"  # 'folder', 'files', 'video'
        self.imported_files: List[str] = []
        
        self._init_ui()
        
        logger.info("ImportDialog 已初始化")
    
    def _init_ui(self):
        """初始化用户界面。"""
        layout = QVBoxLayout(self)
        
        # 源类型选择
        source_group = QGroupBox("导入源")
        source_layout = QVBoxLayout(source_group)
        
        self.source_button_group = QButtonGroup(self)
        
        self.folder_radio = QRadioButton("图像文件夹")
        self.folder_radio.setChecked(True)
        self.folder_radio.toggled.connect(lambda: self._set_import_type("folder"))
        self.source_button_group.addButton(self.folder_radio)
        source_layout.addWidget(self.folder_radio)
        
        self.files_radio = QRadioButton("图像文件")
        self.files_radio.toggled.connect(lambda: self._set_import_type("files"))
        self.source_button_group.addButton(self.files_radio)
        source_layout.addWidget(self.files_radio)
        
        self.video_radio = QRadioButton("视频文件")
        self.video_radio.toggled.connect(lambda: self._set_import_type("video"))
        self.source_button_group.addButton(self.video_radio)
        source_layout.addWidget(self.video_radio)
        
        layout.addWidget(source_group)
        
        # 源选择
        source_select_layout = QHBoxLayout()
        
        self.source_edit = QLineEdit()
        self.source_edit.setPlaceholderText("请选择导入源...")
        self.source_edit.setReadOnly(True)
        source_select_layout.addWidget(self.source_edit, 1)
        
        self.browse_btn = QPushButton("浏览...")
        self.browse_btn.clicked.connect(self._browse_source)
        source_select_layout.addWidget(self.browse_btn)
        
        layout.addLayout(source_select_layout)
        
        # 文件夹导入选项
        self.folder_options = QGroupBox("文件夹选项")
        folder_options_layout = QVBoxLayout(self.folder_options)
        
        self.recursive_check = QCheckBox("包含子目录")
        self.recursive_check.setChecked(False)
        folder_options_layout.addWidget(self.recursive_check)
        
        layout.addWidget(self.folder_options)
        
        # 视频导入选项
        self.video_options = QGroupBox("视频选项")
        self.video_options.setVisible(False)
        video_options_layout = QVBoxLayout(self.video_options)
        
        frame_interval_layout = QHBoxLayout()
        frame_interval_layout.addWidget(QLabel("提取间隔"))
        
        self.frame_interval_spin = QSpinBox()
        self.frame_interval_spin.setMinimum(1)
        self.frame_interval_spin.setMaximum(1000)
        self.frame_interval_spin.setValue(1)
        self.frame_interval_spin.setSuffix(" 帧")
        frame_interval_layout.addWidget(self.frame_interval_spin)
        frame_interval_layout.addStretch()
        
        video_options_layout.addLayout(frame_interval_layout)
        
        max_frames_layout = QHBoxLayout()
        self.max_frames_check = QCheckBox("限制为")
        max_frames_layout.addWidget(self.max_frames_check)
        
        self.max_frames_spin = QSpinBox()
        self.max_frames_spin.setMinimum(1)
        self.max_frames_spin.setMaximum(100000)
        self.max_frames_spin.setValue(1000)
        self.max_frames_spin.setSuffix(" 帧")
        self.max_frames_spin.setEnabled(False)
        max_frames_layout.addWidget(self.max_frames_spin)
        max_frames_layout.addStretch()
        
        self.max_frames_check.toggled.connect(self.max_frames_spin.setEnabled)
        
        video_options_layout.addLayout(max_frames_layout)
        
        layout.addWidget(self.video_options)
        
        # 预览部分
        preview_group = QGroupBox("预览")
        preview_layout = QVBoxLayout(preview_group)
        
        self.preview_list = QListWidget()
        self.preview_list.setMaximumHeight(150)
        preview_layout.addWidget(self.preview_list)
        
        self.preview_info = QLabel("无待导入文件")
        self.preview_info.setStyleSheet("QLabel { color: #888; }")
        preview_layout.addWidget(self.preview_info)
        
        self.preview_btn = QPushButton("更新预览")
        self.preview_btn.clicked.connect(self._update_preview)
        preview_layout.addWidget(self.preview_btn)
        
        layout.addWidget(preview_group)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # 按钮
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.import_btn = QPushButton("导入")
        self.import_btn.setEnabled(False)
        self.import_btn.clicked.connect(self._do_import)
        button_layout.addWidget(self.import_btn)
        
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)
        
        layout.addLayout(button_layout)
    
    def _set_import_type(self, import_type: str):
        """
        设置导入类型。
        
        参数:
            import_type: 导入类型 ('folder', 'files', 'video')
        """
        self.import_type = import_type
        
        # 更新 UI 可见性
        self.folder_options.setVisible(import_type == "folder")
        self.video_options.setVisible(import_type == "video")
        
        # 清除源
        self.source_edit.clear()
        self.import_source = None
        self.preview_list.clear()
        self.import_btn.setEnabled(False)
        
        logger.debug(f"导入类型更改为: {import_type}")
    
    def _browse_source(self):
        """打开文件/文件夹浏览器。"""
        if self.import_type == "folder":
            path = QFileDialog.getExistingDirectory(
                self,
                "选择图像文件夹"
            )
        elif self.import_type == "files":
            paths, _ = QFileDialog.getOpenFileNames(
                self,
                "选择图像文件",
                "",
                "图像 (*.jpg *.jpeg *.png *.bmp *.tiff *.tif);;所有文件 (*)"
            )
            path = ";".join(paths) if paths else None
        else:  # video
            path, _ = QFileDialog.getOpenFileName(
                self,
                "选择视频文件",
                "",
                "视频 (*.mp4 *.avi *.mov *.mkv *.flv);;所有文件 (*)"
            )
        
        if path:
            self.import_source = path
            self.source_edit.setText(path)
            self._update_preview()
            logger.info(f"已选择导入源: {path}")
    
    def _update_preview(self):
        """更新导入预览。"""
        self.preview_list.clear()
        self.imported_files.clear()
        
        if not self.import_source:
            self.preview_info.setText("无待导入文件")
            self.import_btn.setEnabled(False)
            return
        
        try:
            if self.import_type == "folder":
                self._preview_folder()
            elif self.import_type == "files":
                self._preview_files()
            else:  # video
                self._preview_video()
            
            # 启用导入按钮
            self.import_btn.setEnabled(len(self.imported_files) > 0)
            
        except Exception as e:
            logger.error(f"更新预览时出错: {e}")
            self.preview_info.setText(f"错误: {str(e)}")
    
    def _preview_folder(self):
        """预览文件夹导入。"""
        folder = Path(self.import_source)
        
        if not folder.exists():
            self.preview_info.setText("文件夹不存在")
            return
        
        # 查找图像文件
        recursive = self.recursive_check.isChecked()
        supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        
        self.imported_files = list_files(
            folder,
            extensions=supported_formats,
            recursive=recursive
        )
        
        # 更新预览列表（显示前 100 个）
        for path in self.imported_files[:100]:
            self.preview_list.addItem(Path(path).name)
        
        if len(self.imported_files) > 100:
            self.preview_list.addItem(f"... 以及另外 {len(self.imported_files) - 100} 个文件")
        
        self.preview_info.setText(f"找到 {len(self.imported_files)} 张图像")
    
    def _preview_files(self):
        """预览文件导入。"""
        paths = self.import_source.split(";")
        self.imported_files = paths
        
        # 更新预览列表
        for path in paths:
            self.preview_list.addItem(Path(path).name)
        
        self.preview_info.setText(f"已选择 {len(paths)} 个文件")
    
    def _preview_video(self):
        """预览视频导入。"""
        video_path = Path(self.import_source)
        
        if not video_path.exists():
            self.preview_info.setText("视频文件不存在")
            return
        
        # 获取视频信息
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            self.preview_info.setText("无法打开视频文件")
            return
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        # 计算要提取的帧数
        frame_interval = self.frame_interval_spin.value()
        max_frames = self.max_frames_spin.value() if self.max_frames_check.isChecked() else None
        
        estimated_frames = total_frames // frame_interval
        if max_frames is not None:
            estimated_frames = min(estimated_frames, max_frames)
        
        # 存储视频路径（帧将在导入期间提取）
        self.imported_files = [str(video_path)]
        
        # 更新预览
        self.preview_list.addItem(f"视频: {video_path.name}")
        self.preview_list.addItem(f"总帧数: {total_frames}")
        self.preview_list.addItem(f"FPS: {fps:.2f}")
        self.preview_list.addItem(f"预计提取帧数: ~{estimated_frames}")
        
        self.preview_info.setText(f"将提取约 {estimated_frames} 帧")
    
    def _do_import(self):
        """执行导入。"""
        if not self.imported_files:
            return
        
        logger.info(f"开始导入: {len(self.imported_files)} 个项目")
        
        # 显示进度条
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(len(self.imported_files))
        self.progress_bar.setValue(0)
        
        # 禁用按钮
        self.import_btn.setEnabled(False)
        self.browse_btn.setEnabled(False)
        
        # 对于视频，我们需要在这里提取帧
        # 目前，仅发出文件列表
        
        if self.import_type == "video":
            # 注意：实际的帧提取将由 DataManager 完成
            logger.info("请求视频导入 - 帧将由 DataManager 提取")
        
        # 更新进度
        for i in range(len(self.imported_files)):
            self.progress_bar.setValue(i + 1)
        
        # 发出信号
        self.import_completed.emit(self.imported_files)
        
        logger.info("导入完成")
        
        # 接受对话框
        self.accept()
    
    def get_import_files(self) -> List[str]:
        """
        获取待导入的文件列表。
        
        返回:
            文件路径列表
        """
        return self.imported_files.copy()
    
    def get_video_options(self) -> dict:
        """
        获取视频提取选项。
        
        返回:
            包含视频选项的字典
        """
        return {
            'frame_interval': self.frame_interval_spin.value(),
            'max_frames': self.max_frames_spin.value() if self.max_frames_check.isChecked() else None
        }
