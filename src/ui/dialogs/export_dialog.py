"""
用于批量标注导出的导出对话框。

此模块提供：
- 格式选择 (COCO/YOLO/VOC)
- 导出选项配置
- 进度跟踪
- 验证和报告生成
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
    用于批量导出操作的工作线程。
    
    信号:
        progress_updated: 在导出期间发出（当前, 总计, 消息）
        export_completed: 导出完成时发出（成功, 消息）
        export_failed: 导出失败时发出（错误消息）
    """
    
    progress_updated = pyqtSignal(int, int, str)  # 当前, 总计, 消息
    export_completed = pyqtSignal(bool, str)  # 成功, 消息
    export_failed = pyqtSignal(str)  # 错误消息
    
    def __init__(self, 
                 export_format: str,
                 image_paths: List[str],
                 mask_paths: List[str],
                 output_dir: str,
                 options: Dict):
        """
        初始化导出工作线程。
        
        参数:
            export_format: 导出格式 ('coco', 'yolo', 'voc')
            image_paths: 图像文件路径列表
            mask_paths: 掩码文件路径列表
            output_dir: 输出目录
            options: 导出选项字典
        """
        super().__init__()
        
        self.export_format = export_format
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.output_dir = output_dir
        self.options = options
        
        self._is_running = True
    
    def run(self):
        """执行导出操作。"""
        try:
            total = len(self.image_paths)
            
            if self.export_format == 'coco':
                self._export_coco(total)
            elif self.export_format == 'yolo':
                self._export_yolo(total)
            elif self.export_format == 'voc':
                self._export_voc(total)
            else:
                self.export_failed.emit(f"未知格式: {self.export_format}")
                
        except Exception as e:
            logger.error(f"导出错误: {e}", exc_info=True)
            self.export_failed.emit(str(e))
    
    def _export_coco(self, total: int):
        """导出为 COCO 格式。"""
        from src.utils.export_utils import batch_export_coco
        
        self.progress_updated.emit(0, total, "正在初始化 COCO 导出...")
        
        # 获取选项
        dataset_name = self.options.get('dataset_name', '工业缺陷数据集')
        category_names = self.options.get('category_names', ['defect'] * len(self.image_paths))
        
        # 输出路径
        output_path = Path(self.output_dir) / "annotations.json"
        
        # 导出
        stats = batch_export_coco(
            self.image_paths,
            self.mask_paths,
            category_names,
            str(output_path),
            dataset_name
        )
        
        if not self._is_running:
            return
        
        self.progress_updated.emit(total, total, "正在验证 COCO 格式...")
        
        # 如果有要求，进行验证
        if self.options.get('validate', True):
            result = validate_coco_dataset(str(output_path))
            if not result.is_valid:
                self.export_completed.emit(False, f"导出完成，但存在验证错误：\n{result.get_report()}")
                return
        
        message = f"COCO 导出成功！\n\n"
        message += f"输出文件: {output_path}\n"
        message += f"图像数量: {stats['num_images']}\n"
        message += f"标注数量: {stats['num_annotations']}\n"
        message += f"类别数量: {stats['num_categories']}"
        
        self.export_completed.emit(True, message)
    
    def _export_yolo(self, total: int):
        """导出为 YOLO 格式。"""
        from src.utils.export_utils import batch_export_yolo
        from src.utils.image_utils import get_image_info
        
        self.progress_updated.emit(0, total, "正在初始化 YOLO 导出...")
        
        # 获取选项
        class_names = self.options.get('class_names', ['defect'])
        class_ids = self.options.get('class_ids', [0] * len(self.image_paths))
        
        # 导出
        count = batch_export_yolo(
            self.image_paths,
            self.mask_paths,
            class_ids,
            class_names,
            self.output_dir
        )
        
        if not self._is_running:
            return
        
        self.progress_updated.emit(total, total, "正在创建 data.yaml...")
        
        # 如果有要求，创建 data.yaml
        if self.options.get('create_yaml', True):
            from src.utils.export_utils import YOLOExporter
            exporter = YOLOExporter(self.output_dir, class_names)
            exporter.create_data_yaml(
                train_path=self.options.get('train_path', 'images/train'),
                val_path=self.options.get('val_path', 'images/val'),
                test_path=self.options.get('test_path')
            )
        
        # 如果有要求，进行验证
        if self.options.get('validate', True):
            classes_file = Path(self.output_dir) / "classes.txt"
            result = validate_yolo_dataset(self.output_dir, str(classes_file))
            if not result.is_valid:
                self.export_completed.emit(False, f"导出完成，但存在验证错误：\n{result.get_report()}")
                return
        
        message = f"YOLO 导出成功！\n\n"
        message += f"输出目录: {self.output_dir}\n"
        message += f"标注数量: {count}\n"
        message += f"类别数量: {len(class_names)}"
        
        self.export_completed.emit(True, message)
    
    def _export_voc(self, total: int):
        """导出为 VOC 格式。"""
        self.progress_updated.emit(0, total, "正在初始化 VOC 导出...")
        
        exporter = VOCExporter(self.output_dir)
        
        from src.utils.image_utils import get_image_info
        from src.utils.mask_utils import load_mask
        
        category_names = self.options.get('category_names', ['defect'] * len(self.image_paths))
        
        for i, (image_path, mask_path, category) in enumerate(zip(self.image_paths, self.mask_paths, category_names)):
            if not self._is_running:
                return
            
            self.progress_updated.emit(i, total, f"正在导出 {i+1}/{total}...")
            
            # 获取图像信息
            info = get_image_info(image_path)
            if info is None:
                continue
            
            # 加载掩码
            mask = load_mask(mask_path)
            if mask is None:
                continue
            
            # 导出
            exporter.export_annotation(
                image_path,
                [mask],
                [category],
                info['width'],
                info['height']
            )
        
        message = f"VOC 导出成功！\n\n"
        message += f"输出目录: {self.output_dir}\n"
        message += f"标注数量: {len(self.image_paths)}"
        
        self.export_completed.emit(True, message)
    
    def stop(self):
        """停止导出线程。"""
        self._is_running = False


class ExportDialog(QDialog):
    """
    用于批量标注导出的对话框。
    
    提供以下界面：
    - 格式选择 (COCO/YOLO/VOC)
    - 输出目录选择
    - 导出选项配置
    - 进度跟踪
    - 验证
    """
    
    def __init__(self, 
                 image_paths: List[str],
                 mask_paths: List[str],
                 parent=None):
        """
        初始化导出对话框。
        
        参数:
            image_paths: 图像文件路径列表
            mask_paths: 掩码文件路径列表
            parent: 父小部件
        """
        super().__init__(parent)
        
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.worker_thread: Optional[ExportWorkerThread] = None
        
        self.setWindowTitle("导出标注")
        self.setMinimumWidth(600)
        self.setMinimumHeight(500)
        
        self._init_ui()
        self._connect_signals()
        
        logger.info(f"导出对话框已打开: {len(image_paths)} 张图像")
    
    def _init_ui(self):
        """初始化 UI 组件。"""
        layout = QVBoxLayout(self)
        
        # 格式选择
        format_group = QGroupBox("导出格式")
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
        
        # 输出目录
        output_group = QGroupBox("输出")
        output_layout = QGridLayout(output_group)
        
        output_layout.addWidget(QLabel("输出目录:"), 0, 0)
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setPlaceholderText("请选择输出目录...")
        output_layout.addWidget(self.output_dir_edit, 0, 1)
        
        browse_btn = QPushButton("浏览...")
        browse_btn.clicked.connect(self._browse_output_dir)
        output_layout.addWidget(browse_btn, 0, 2)
        
        layout.addWidget(output_group)
        
        # 选项
        options_group = QGroupBox("选项")
        options_layout = QVBoxLayout(options_group)
        
        self.dataset_name_edit = QLineEdit("工业缺陷数据集")
        options_layout.addWidget(QLabel("数据集名称:"))
        options_layout.addWidget(self.dataset_name_edit)
        
        self.class_name_edit = QLineEdit("defect")
        options_layout.addWidget(QLabel("类别名称:"))
        options_layout.addWidget(self.class_name_edit)
        
        self.validate_checkbox = QCheckBox("导出后进行验证")
        self.validate_checkbox.setChecked(True)
        options_layout.addWidget(self.validate_checkbox)
        
        self.create_yaml_checkbox = QCheckBox("创建 data.yaml (仅限 YOLO)")
        self.create_yaml_checkbox.setChecked(True)
        options_layout.addWidget(self.create_yaml_checkbox)
        
        layout.addWidget(options_group)
        
        # 进度
        progress_group = QGroupBox("进度")
        progress_layout = QVBoxLayout(progress_group)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        progress_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("准备导出")
        self.status_label.setStyleSheet("color: gray;")
        progress_layout.addWidget(self.status_label)
        
        layout.addWidget(progress_group)
        
        # 日志/结果
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMaximumHeight(150)
        self.result_text.setVisible(False)
        layout.addWidget(self.result_text)
        
        # 按钮
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.export_btn = QPushButton("导出")
        self.export_btn.setDefault(True)
        self.export_btn.clicked.connect(self._start_export)
        button_layout.addWidget(self.export_btn)
        
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)
        
        layout.addLayout(button_layout)
        
        # 信息标签
        info_label = QLabel(f"总计: {len(self.image_paths)} 张图像, {len(self.mask_paths)} 个掩码")
        info_label.setStyleSheet("color: gray; font-size: 9pt;")
        layout.addWidget(info_label)
    
    def _connect_signals(self):
        """连接信号和槽。"""
        pass
    
    def _get_format_description(self, index: int) -> str:
        """获取格式描述。"""
        descriptions = {
            0: "COCO JSON: 用于目标检测/分割的标准格式。输出包含 RLE 编码掩码的单个 JSON 文件。",
            1: "YOLO txt: 用于 YOLO 训练的格式。每张图像输出一个包含归一化多边形坐标的 txt 文件。",
            2: "Pascal VOC XML: 带有 XML 标注的标准格式。输出 XML 文件和 PNG 分割掩码。"
        }
        return descriptions.get(index, "")
    
    def _browse_output_dir(self):
        """浏览输出目录。"""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "选择输出目录",
            "",
            QFileDialog.ShowDirsOnly
        )
        
        if dir_path:
            self.output_dir_edit.setText(dir_path)
    
    def _start_export(self):
        """开始导出操作。"""
        # 验证输入
        output_dir = self.output_dir_edit.text().strip()
        if not output_dir:
            QMessageBox.warning(self, "错误", "请选择输出目录")
            return
        
        if not self.image_paths or not self.mask_paths:
            QMessageBox.warning(self, "错误", "没有可导出的图像或掩码")
            return
        
        if len(self.image_paths) != len(self.mask_paths):
            QMessageBox.warning(self, "错误", "图像和掩码的数量不匹配")
            return
        
        # 获取格式
        format_map = {0: 'coco', 1: 'yolo', 2: 'voc'}
        export_format = format_map[self.format_combo.currentIndex()]
        
        # 准备选项
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
        
        # 创建工作线程
        self.worker_thread = ExportWorkerThread(
            export_format,
            self.image_paths,
            self.mask_paths,
            output_dir,
            options
        )
        
        # 连接信号
        self.worker_thread.progress_updated.connect(self._on_progress)
        self.worker_thread.export_completed.connect(self._on_export_completed)
        self.worker_thread.export_failed.connect(self._on_export_failed)
        
        # 更新 UI
        self.export_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.result_text.setVisible(False)
        self.status_label.setText("正在开始导出...")
        
        # 开始导出
        self.worker_thread.start()
        
        logger.info(f"已开始导出: 格式={export_format}, 输出={output_dir}")
    
    def _on_progress(self, current: int, total: int, message: str):
        """处理进度更新。"""
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.status_label.setText(message)
    
    def _on_export_completed(self, success: bool, message: str):
        """处理导出完成。"""
        self.export_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        if success:
            self.status_label.setText("✅ 导出成功完成！")
            self.status_label.setStyleSheet("color: green;")
            
            self.result_text.setText(message)
            self.result_text.setVisible(True)
            
            QMessageBox.information(self, "成功", message)
            self.accept()
        else:
            self.status_label.setText("⚠️ 导出完成，但有警告")
            self.status_label.setStyleSheet("color: orange;")
            
            self.result_text.setText(message)
            self.result_text.setVisible(True)
            
            QMessageBox.warning(self, "验证问题", message)
    
    def _on_export_failed(self, error_message: str):
        """处理导出失败。"""
        self.export_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("❌ 导出失败")
        self.status_label.setStyleSheet("color: red;")
        
        self.result_text.setText(f"错误: {error_message}")
        self.result_text.setVisible(True)
        
        QMessageBox.critical(self, "导出失败", error_message)
    
    def closeEvent(self, event):
        """处理对话框关闭。"""
        if self.worker_thread and self.worker_thread.isRunning():
            reply = QMessageBox.question(
                self,
                "导出正在进行中",
                "导出仍在运行。您确定要取消吗？",
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
