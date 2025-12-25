"""
工业缺陷分割系统的主窗口。

这是主要的 UI 组件，承载所有其他小部件并管理整个应用程序状态。
"""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QMenuBar, QMenu, QAction, QStatusBar, QLabel,
    QSplitter, QMessageBox
)

from src.logger import get_logger
from src.core.data_manager import DataManager
from src.ui.widgets.image_canvas import ImageCanvasWithInfo
from src.ui.widgets.file_browser import FileBrowser
from src.ui.dialogs.import_dialog import ImportDialog
from src.ui.dialogs.export_dialog import ExportDialog
from src.ui.dialogs.train_config_dialog import TrainConfigDialog
from src.ui.dialogs.predict_dialog import PredictDialog
from src.ui.dialogs.report_dialog import ReportDialog

logger = get_logger(__name__)


class MainWindow(QMainWindow):
    """
    主应用程序窗口。
    
    此窗口提供应用程序的主要界面，包括：
    - 带有文件、编辑、视图、工具、帮助菜单的菜单栏
    - 用于快速访问常用操作的工具栏
    - 带有图像画布和标注工具的中央小部件
    - 用于文件浏览器和属性的侧面板
    - 用于显示信息的状态栏
    """
    
    def __init__(self, config: dict, paths_config: dict, hyperparams: dict):
        """
        初始化主窗口。
        
        参数:
            config: 应用程序配置
            paths_config: 路径配置
            hyperparams: 模型超参数配置
        """
        super().__init__()
        
        self.config = config
        self.paths_config = paths_config
        self.hyperparams = hyperparams
        
        # 初始化 DataManager
        from pathlib import Path
        data_root = Path(paths_config['paths']['data_root'])
        cache_size = config['performance']['max_cache_size']
        self.data_manager = DataManager(str(data_root), cache_size_mb=cache_size)
        
        # 当前状态
        self.current_image_path: str = None
        self.current_image_index: int = -1
        
        # 窗口设置
        self.setWindowTitle(config['app']['name'])
        self.setGeometry(
            100, 100,
            config['ui']['window_width'],
            config['ui']['window_height']
        )
        
        # 初始化 UI 组件
        self._init_ui()
        self._create_menus()
        self._create_toolbar()
        self._create_statusbar()
        self._connect_signals()
        
        logger.info("主窗口已初始化")
    
    def _init_ui(self):
        """初始化用户界面。"""
        # 创建带有主布局的中央小部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        
        # 创建用于可调整大小面板的主拆分器
        self.main_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(self.main_splitter)
        
        # 左侧面板 - 文件浏览器
        self.file_browser = FileBrowser(self)
        self.main_splitter.addWidget(self.file_browser)
        
        # 中间面板 - 图像画布
        self.image_canvas = ImageCanvasWithInfo(self)
        self.main_splitter.addWidget(self.image_canvas)
        
        # 右侧面板 - 属性（目前为占位符）
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.addWidget(QLabel("属性面板\n（待实现）"))
        self.main_splitter.addWidget(right_panel)
        
        # 设置拆分器的初始大小
        self.main_splitter.setSizes([250, 900, 250])
    
    def _create_menus(self):
        """创建菜单栏和菜单。"""
        menubar = self.menuBar()
        
        # 文件菜单
        file_menu = menubar.addMenu("文件(&F)")
        
        # 文件 > 导入
        import_action = QAction("导入图像(&I)...", self)
        import_action.setShortcut("Ctrl+I")
        import_action.setStatusTip("导入图像或视频")
        import_action.triggered.connect(self._on_import)
        file_menu.addAction(import_action)
        
        # 文件 > 打开项目
        open_action = QAction("打开项目(&O)...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.setStatusTip("打开现有项目")
        file_menu.addAction(open_action)
        
        # 文件 > 保存
        save_action = QAction("保存(&S)", self)
        save_action.setShortcut("Ctrl+S")
        save_action.setStatusTip("保存当前标注")
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        # 文件 > 退出
        exit_action = QAction("退出(&X)", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.setStatusTip("退出应用程序")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 编辑菜单
        edit_menu = menubar.addMenu("编辑(&E)")
        
        undo_action = QAction("撤销(&U)", self)
        undo_action.setShortcut("Ctrl+Z")
        edit_menu.addAction(undo_action)
        
        redo_action = QAction("重做(&R)", self)
        redo_action.setShortcut("Ctrl+Y")
        edit_menu.addAction(redo_action)
        
        # 视图菜单
        view_menu = menubar.addMenu("视图(&V)")
        
        zoom_in_action = QAction("放大(&I)", self)
        zoom_in_action.setShortcut("Ctrl++")
        view_menu.addAction(zoom_in_action)
        
        zoom_out_action = QAction("缩小(&O)", self)
        zoom_out_action.setShortcut("Ctrl+-")
        view_menu.addAction(zoom_out_action)
        
        # 工具菜单
        tools_menu = menubar.addMenu("工具(&T)")
        
        train_action = QAction("训练模型(&T)...", self)
        train_action.setStatusTip("打开模型训练对话框")
        train_action.triggered.connect(self._on_train_model)
        tools_menu.addAction(train_action)
        
        predict_action = QAction("预测(&P)...", self)
        predict_action.setStatusTip("对图像运行推理")
        predict_action.triggered.connect(self._on_predict)
        tools_menu.addAction(predict_action)
        
        tools_menu.addSeparator()
        
        report_action = QAction("生成报告(&R)...", self)
        report_action.setStatusTip("生成分析报告")
        report_action.triggered.connect(self._on_generate_report)
        tools_menu.addAction(report_action)
        
        tools_menu.addSeparator()
        
        export_action = QAction("导出标注(&E)...", self)
        export_action.setStatusTip("将标注导出为 COCO/YOLO 格式")
        export_action.triggered.connect(self._on_export)
        tools_menu.addAction(export_action)
        
        # 帮助菜单
        help_menu = menubar.addMenu("帮助(&H)")
        
        about_action = QAction("关于(&A)", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
    
    def _create_toolbar(self):
        """创建带有快速操作的工具栏。"""
        toolbar = self.addToolBar("主工具栏")
        toolbar.setMovable(False)
        
        # 添加占位符操作
        # TODO: 添加带有图标的实际工具栏按钮
        pass
    
    def _create_statusbar(self):
        """创建状态栏。"""
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        
        # 向状态栏添加永久小部件
        self.status_label = QLabel("就绪")
        self.statusBar.addWidget(self.status_label)
        
        # 添加坐标标签（用于鼠标位置）
        self.coords_label = QLabel("位置: (0, 0)")
        self.statusBar.addPermanentWidget(self.coords_label)
    
    def _connect_signals(self):
        """连接信号和槽。"""
        # 文件浏览器信号
        self.file_browser.file_selected.connect(self._on_file_selected)
        self.file_browser.folder_changed.connect(self._on_folder_changed)
        
        # 图像画布信号
        self.image_canvas.canvas.mouse_moved.connect(self._on_mouse_moved)
        
        logger.debug("信号已连接")
    
    def _on_file_selected(self, file_path: str):
        """
        处理从浏览器选择的文件。
        
        参数:
            file_path: 所选文件的路径
        """
        logger.info(f"已选择文件: {file_path}")
        
        # 使用 DataManager 加载图像
        image = self.data_manager.load_image(file_path)
        
        if image is not None:
            # 在画布中显示
            self.image_canvas.load_image(image, file_path)
            self.current_image_path = file_path
            
            # 更新状态栏
            self.status_label.setText(f"已加载: {file_path}")
        else:
            QMessageBox.warning(
                self,
                "错误",
                f"加载图像失败: {file_path}"
            )
    
    def _on_folder_changed(self, folder_path: str):
        """
        处理浏览器中的文件夹更改。
        
        参数:
            folder_path: 新文件夹的路径
        """
        logger.info(f"文件夹已更改: {folder_path}")
        self.status_label.setText(f"文件夹: {folder_path}")
    
    def _on_mouse_moved(self, x: int, y: int):
        """
        处理画布上的鼠标移动。
        
        参数:
            x: X 坐标
            y: Y 坐标
        """
        self.coords_label.setText(f"位置: ({x}, {y})")
    
    def _on_import(self):
        """处理导入操作。"""
        dialog = ImportDialog(self)
        
        if dialog.exec_() == ImportDialog.Accepted:
            imported_files = dialog.get_import_files()
            
            if imported_files:
                logger.info(f"已导入 {len(imported_files)} 个文件")
                
                # 如果导入文件夹，则在文件浏览器中设置它
                if dialog.import_type == "folder":
                    self.file_browser.set_folder(imported_files[0] if len(imported_files) == 1 else dialog.import_source)
                elif dialog.import_type == "files":
                    # 将文件加载到数据管理器中
                    self.data_manager.dataset['all'] = imported_files
                    # 可选地将文件夹设置为第一个文件的父级
                    if imported_files:
                        from pathlib import Path
                        parent_folder = Path(imported_files[0]).parent
                        self.file_browser.set_folder(str(parent_folder))
                elif dialog.import_type == "video":
                    # 提取视频帧
                    video_path = imported_files[0]
                    video_options = dialog.get_video_options()
                    
                    from pathlib import Path
                    output_dir = Path(self.paths_config['paths']['raw_data']) / "video_frames"
                    
                    QMessageBox.information(
                        self,
                        "视频导入",
                        f"正在从视频中提取帧...\n这可能需要一段时间。"
                    )
                    
                    frame_paths = self.data_manager.save_video_frames(
                        video_path,
                        output_dir,
                        frame_interval=video_options['frame_interval'],
                        max_frames=video_options['max_frames']
                    )
                    
                    if frame_paths:
                        self.file_browser.set_folder(str(output_dir))
                        QMessageBox.information(
                            self,
                            "成功",
                            f"已提取 {len(frame_paths)} 帧"
                        )
                
                self.status_label.setText(f"已导入 {len(imported_files)} 个文件")
        
        logger.info("导入对话框已关闭")
    
    def _on_export(self):
        """处理导出操作。"""
        # 检查是否有任何标注数据
        if not self.data_manager.dataset.get('all'):
            QMessageBox.warning(
                self,
                "无数据",
                "未加载图像。请先导入图像。"
            )
            return
        
        # 目前使用虚拟掩码路径（在实际使用中，这些来自 AnnotationManager）
        # TODO: 从标注管理器获取实际掩码路径
        image_paths = self.data_manager.dataset.get('all', [])
        
        if not image_paths:
            QMessageBox.warning(
                self,
                "无图像",
                "没有可供导出的图像。"
            )
            return
        
        # 创建用于演示的虚拟掩码路径
        # 实际使用中: mask_paths = [annotation_manager.get_mask_path(img) for img in image_paths]
        from pathlib import Path
        masks_dir = Path(self.paths_config['paths']['masks'])
        mask_paths = []
        
        for img_path in image_paths:
            mask_name = Path(img_path).stem + ".png"
            mask_path = masks_dir / mask_name
            if mask_path.exists():
                mask_paths.append(str(mask_path))
        
        if not mask_paths:
            QMessageBox.information(
                self,
                "无标注",
                f"在 {masks_dir} 中未找到标注掩码。\n\n"
                "请在导出前先标注一些图像。"
            )
            return
        
        # 匹配图像和掩码列表
        matched_images = []
        matched_masks = []
        
        for img_path in image_paths:
            img_name = Path(img_path).stem
            for mask_path in mask_paths:
                if Path(mask_path).stem == img_name:
                    matched_images.append(img_path)
                    matched_masks.append(mask_path)
                    break
        
        if not matched_images:
            QMessageBox.warning(
                self,
                "无匹配对",
                "未找到匹配的图像-掩码对。"
            )
            return
        
        # 打开导出对话框
        dialog = ExportDialog(matched_images, matched_masks, self)
        dialog.exec_()
        
        logger.info("导出对话框已关闭")
    
    def _on_train_model(self):
        """处理训练模型操作。"""
        # 检查是否有数据
        if not self.data_manager.dataset.get('all'):
            reply = QMessageBox.question(
                self,
                "无数据",
                "未加载图像。训练需要准备好包含图像和掩码的数据。\n\n"
                "是否仍要继续？（您可以在训练对话框中配置路径）",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.No:
                return
        
        # 打开训练对话框
        dialog = TrainConfigDialog(self.config, self.paths_config, self)
        dialog.exec_()
        
        logger.info("训练对话框已关闭")
    
    def _on_predict(self):
        """处理预测操作。"""
        # 检查是否有训练好的模型
        from pathlib import Path
        models_dir = Path(self.paths_config['paths']['trained_models'])
        
        if not models_dir.exists() or not any(models_dir.glob('*.pth')):
            reply = QMessageBox.question(
                self,
                "无模型",
                "未找到训练好的模型。预测需要训练好的模型检查点。\n\n"
                "是否仍要继续？（您可以在对话框中指定检查点路径）",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.No:
                return
        
        # 打开预测对话框
        dialog = PredictDialog(self.config, self.paths_config, self)
        dialog.exec_()
        
        logger.info("预测对话框已关闭")
    
    def _on_generate_report(self):
        """处理生成报告操作。"""
        # 检查是否有要分析的数据
        from pathlib import Path
        masks_dir = Path(self.paths_config['paths']['masks'])
        
        # 检查掩码或预测
        has_masks = masks_dir.exists() and any(masks_dir.glob('*.png'))
        
        predictions_dir = Path(self.paths_config['paths']['predictions'])
        has_predictions = predictions_dir.exists() and any(
            (predictions_dir / 'masks').glob('*.png') if (predictions_dir / 'masks').exists() else []
        )
        
        if not has_masks and not has_predictions:
            reply = QMessageBox.question(
                self,
                "无数据",
                "未找到掩码文件或预测结果。\n\n"
                "生成报告需要掩码文件进行分析。\n\n"
                "是否仍要继续？（您可以在对话框中指定掩码目录）",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.No:
                return
        
        # 打开报告对话框
        dialog = ReportDialog(self.config, self.paths_config, self)
        dialog.exec_()
        
        logger.info("报告对话框已关闭")
    
    def _show_about(self):
        """显示关于对话框。"""
        QMessageBox.about(
            self,
            f"关于 {self.config['app']['name']}",
            f"{self.config['app']['name']}\n"
            f"版本: {self.config['app']['version']}\n"
            f"作者: {self.config['app']['author']}\n\n"
            f"一个完整的工业缺陷分割系统，具有 "
            f"SAM 自动标注、模型训练和推理功能。"
        )
