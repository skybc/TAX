"""
用于创建缺陷分析报告的报告生成对话框。

此模块提供：
- 报告配置 UI
- 数据源选择
- 报告格式选择 (Excel/PDF/HTML)
- 报告预览
- 报告生成进度
"""

from typing import Optional, List, Dict
from pathlib import Path

from PyQt5.QtWidgets import (
    QWidget,
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget,
    QLabel, QPushButton, QLineEdit, QFileDialog,
    QListWidget, QCheckBox, QGroupBox, QProgressBar,
    QTextEdit, QMessageBox, QComboBox, QSpinBox
)
from PyQt5.QtCore import Qt, pyqtSignal

from src.logger import get_logger
from src.ui.widgets.chart_widget import ChartWidget
from src.utils.statistics import DefectStatistics, DatasetStatistics
from src.utils.visualization import DefectVisualizer, TrainingVisualizer, DatasetVisualizer
from src.utils.report_generator import ReportManager

logger = get_logger(__name__)


class ReportDialog(QDialog):
    """
    用于生成分析报告的对话框。
    
    提供以下 UI：
    - 数据源选择
    - 报告类型和格式选择
    - 图表配置
    - 报告生成和预览
    
    信号:
        report_generated: 报告生成完成时发出 (report_paths)
    """
    
    report_generated = pyqtSignal(dict)  # report_paths
    
    def __init__(self, config: dict, paths_config: dict, parent: Optional['QWidget'] = None):
        """
        初始化 ReportDialog。
        
        参数:
            config: 应用程序配置
            paths_config: 路径配置
            parent: 父小部件
        """
        super().__init__(parent)
        
        self.config = config
        self.paths_config = paths_config
        
        # 数据
        self.mask_dir: Optional[str] = None
        self.mask_paths: List[str] = []
        self.output_dir: Optional[str] = None
        
        # 统计信息
        self.statistics: Optional[Dict] = None
        
        # 窗口设置
        self.setWindowTitle("生成报告")
        self.setMinimumSize(900, 700)
        
        # 初始化 UI
        self._init_ui()
        
        logger.info("ReportDialog 已初始化")
    
    def _init_ui(self):
        """初始化用户界面。"""
        layout = QVBoxLayout(self)
        
        # 创建选项卡
        tabs = QTabWidget()
        
        tabs.addTab(self._create_data_tab(), "数据源")
        tabs.addTab(self._create_settings_tab(), "报告设置")
        tabs.addTab(self._create_preview_tab(), "预览")
        tabs.addTab(self._create_generate_tab(), "生成")
        
        layout.addWidget(tabs)
        
        # 按钮
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.close_button = QPushButton("关闭")
        self.close_button.clicked.connect(self.close)
        button_layout.addWidget(self.close_button)
        
        layout.addLayout(button_layout)
    
    def _create_data_tab(self) -> QWidget:
        """创建数据源选项卡。"""
        from PyQt5.QtWidgets import QWidget
        
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 掩码目录选择
        dir_group = QGroupBox("掩码目录")
        dir_layout = QVBoxLayout(dir_group)
        
        dir_select_layout = QHBoxLayout()
        
        self.mask_dir_edit = QLineEdit()
        self.mask_dir_edit.setPlaceholderText("选择包含掩码文件的目录...")
        self.mask_dir_edit.setReadOnly(True)
        dir_select_layout.addWidget(self.mask_dir_edit)
        
        browse_button = QPushButton("浏览...")
        browse_button.clicked.connect(self._browse_mask_dir)
        dir_select_layout.addWidget(browse_button)
        
        dir_layout.addLayout(dir_select_layout)
        
        # 加载掩码按钮
        load_button = QPushButton("加载掩码")
        load_button.clicked.connect(self._load_masks)
        dir_layout.addWidget(load_button)
        
        layout.addWidget(dir_group)
        
        # 掩码列表
        list_group = QGroupBox("已加载掩码")
        list_layout = QVBoxLayout(list_group)
        
        self.mask_list = QListWidget()
        list_layout.addWidget(self.mask_list)
        
        # 统计按钮
        stats_button = QPushButton("计算统计信息")
        stats_button.clicked.connect(self._compute_statistics)
        list_layout.addWidget(stats_button)
        
        layout.addWidget(list_group)
        
        # 快速统计显示
        self.quick_stats_label = QLabel("未加载数据")
        self.quick_stats_label.setWordWrap(True)
        layout.addWidget(self.quick_stats_label)
        
        layout.addStretch()
        
        return widget
    
    def _create_settings_tab(self) -> QWidget:
        """创建报告设置选项卡。"""
        from PyQt5.QtWidgets import QWidget
        
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 报告格式选择
        format_group = QGroupBox("报告格式")
        format_layout = QVBoxLayout(format_group)
        
        self.excel_checkbox = QCheckBox("Excel (.xlsx)")
        self.excel_checkbox.setChecked(True)
        format_layout.addWidget(self.excel_checkbox)
        
        self.pdf_checkbox = QCheckBox("PDF (.pdf)")
        self.pdf_checkbox.setChecked(True)
        format_layout.addWidget(self.pdf_checkbox)
        
        self.html_checkbox = QCheckBox("HTML (.html)")
        self.html_checkbox.setChecked(True)
        format_layout.addWidget(self.html_checkbox)
        
        layout.addWidget(format_group)
        
        # 图表设置
        chart_group = QGroupBox("图表设置")
        chart_layout = QVBoxLayout(chart_group)
        
        self.size_dist_checkbox = QCheckBox("缺陷尺寸分布")
        self.size_dist_checkbox.setChecked(True)
        chart_layout.addWidget(self.size_dist_checkbox)
        
        self.count_dist_checkbox = QCheckBox("缺陷数量分布")
        self.count_dist_checkbox.setChecked(True)
        chart_layout.addWidget(self.count_dist_checkbox)
        
        self.coverage_checkbox = QCheckBox("覆盖率分布")
        self.coverage_checkbox.setChecked(True)
        chart_layout.addWidget(self.coverage_checkbox)
        
        self.heatmap_checkbox = QCheckBox("空间热力图")
        self.heatmap_checkbox.setChecked(False)
        chart_layout.addWidget(self.heatmap_checkbox)
        
        layout.addWidget(chart_group)
        
        # 输出目录
        output_group = QGroupBox("输出目录")
        output_layout = QHBoxLayout(output_group)
        
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setPlaceholderText("选择输出目录...")
        self.output_dir_edit.setReadOnly(True)
        output_layout.addWidget(self.output_dir_edit)
        
        output_browse_button = QPushButton("浏览...")
        output_browse_button.clicked.connect(self._browse_output_dir)
        output_layout.addWidget(output_browse_button)
        
        layout.addWidget(output_group)
        
        layout.addStretch()
        
        return widget
    
    def _create_preview_tab(self) -> QWidget:
        """创建预览选项卡。"""
        from PyQt5.QtWidgets import QWidget
        
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 预览控制
        control_layout = QHBoxLayout()
        
        control_layout.addWidget(QLabel("图表类型:"))
        
        self.preview_combo = QComboBox()
        self.preview_combo.addItems([
            "缺陷尺寸分布",
            "缺陷数量分布",
            "覆盖率分布",
            "数据集摘要"
        ])
        self.preview_combo.currentIndexChanged.connect(self._update_preview)
        control_layout.addWidget(self.preview_combo)
        
        preview_button = QPushButton("生成预览")
        preview_button.clicked.connect(self._update_preview)
        control_layout.addWidget(preview_button)
        
        control_layout.addStretch()
        
        layout.addLayout(control_layout)
        
        # 图表预览
        self.preview_chart = ChartWidget(figsize=(10, 6))
        layout.addWidget(self.preview_chart)
        
        return widget
    
    def _create_generate_tab(self) -> QWidget:
        """创建报告生成选项卡。"""
        from PyQt5.QtWidgets import QWidget
        
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 摘要
        summary_group = QGroupBox("报告摘要")
        summary_layout = QVBoxLayout(summary_group)
        
        self.summary_label = QLabel("配置数据源和设置，然后生成报告。")
        self.summary_label.setWordWrap(True)
        summary_layout.addWidget(self.summary_label)
        
        layout.addWidget(summary_group)
        
        # 生成按钮
        self.generate_button = QPushButton("生成报告")
        self.generate_button.clicked.connect(self._generate_report)
        self.generate_button.setMinimumHeight(50)
        self.generate_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 14pt;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        layout.addWidget(self.generate_button)
        
        # 进度
        progress_group = QGroupBox("生成进度")
        progress_layout = QVBoxLayout(progress_group)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("就绪")
        progress_layout.addWidget(self.status_label)
        
        layout.addWidget(progress_group)
        
        # 日志
        log_group = QGroupBox("生成日志")
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
        log_layout.addWidget(self.log_text)
        
        layout.addWidget(log_group)
        
        # 打开报告按钮
        open_group = QGroupBox("已生成报告")
        open_layout = QHBoxLayout(open_group)
        
        self.open_excel_button = QPushButton("打开 Excel")
        self.open_excel_button.clicked.connect(lambda: self._open_report('excel'))
        self.open_excel_button.setEnabled(False)
        open_layout.addWidget(self.open_excel_button)
        
        self.open_pdf_button = QPushButton("打开 PDF")
        self.open_pdf_button.clicked.connect(lambda: self._open_report('pdf'))
        self.open_pdf_button.setEnabled(False)
        open_layout.addWidget(self.open_pdf_button)
        
        self.open_html_button = QPushButton("打开 HTML")
        self.open_html_button.clicked.connect(lambda: self._open_report('html'))
        self.open_html_button.setEnabled(False)
        open_layout.addWidget(self.open_html_button)
        
        self.open_folder_button = QPushButton("打开文件夹")
        self.open_folder_button.clicked.connect(self._open_output_folder)
        self.open_folder_button.setEnabled(False)
        open_layout.addWidget(self.open_folder_button)
        
        layout.addWidget(open_group)
        
        layout.addStretch()
        
        return widget
    
    def _browse_mask_dir(self):
        """浏览掩码目录。"""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "选择掩码目录",
            self.paths_config['paths'].get('masks', '')
        )
        
        if dir_path:
            self.mask_dir = dir_path
            self.mask_dir_edit.setText(dir_path)
            logger.info(f"已选择掩码目录: {dir_path}")
    
    def _load_masks(self):
        """从目录加载掩码。"""
        if not self.mask_dir:
            QMessageBox.warning(self, "无目录", "请先选择掩码目录。")
            return
        
        # 查找掩码文件
        mask_dir = Path(self.mask_dir)
        mask_files = list(mask_dir.glob('*.png')) + list(mask_dir.glob('*.tif'))
        
        if not mask_files:
            QMessageBox.warning(self, "未找到掩码", 
                              f"在 {mask_dir} 中未找到掩码文件")
            return
        
        # 更新列表
        self.mask_paths = [str(f) for f in mask_files]
        self.mask_list.clear()
        
        for mask_path in self.mask_paths:
            self.mask_list.addItem(Path(mask_path).name)
        
        # 更新快速统计
        self.quick_stats_label.setText(f"已加载 {len(self.mask_paths)} 个掩码文件")
        
        logger.info(f"从 {mask_dir} 加载了 {len(self.mask_paths)} 个掩码")
    
    def _compute_statistics(self):
        """计算已加载掩码的统计信息。"""
        if not self.mask_paths:
            QMessageBox.warning(self, "无数据", "请先加载掩码文件。")
            return
        
        logger.info("正在计算统计信息...")
        self.status_label.setText("正在计算统计信息...")
        self.progress_bar.setValue(10)
        
        try:
            # 计算统计信息
            defect_stats = DefectStatistics()
            self.statistics = defect_stats.compute_batch_statistics(self.mask_paths)
            
            # 更新快速统计显示
            quick_stats_text = f"""
统计信息已计算：
- 总图像数: {self.statistics.get('total_images', 0)}
- 包含缺陷的图像数: {self.statistics.get('images_with_defects', 0)}
- 总缺陷数: {self.statistics.get('total_defects', 0)}
- 平均每张图像缺陷数: {self.statistics.get('mean_defects_per_image', 0):.2f}
- 平均覆盖率: {self.statistics.get('mean_coverage_ratio', 0):.4f}
            """
            
            self.quick_stats_label.setText(quick_stats_text.strip())
            
            self.progress_bar.setValue(100)
            self.status_label.setText("统计信息计算成功")
            
            QMessageBox.information(self, "成功", "统计信息计算成功！")
            
        except Exception as e:
            logger.error(f"计算统计信息失败: {e}", exc_info=True)
            QMessageBox.critical(self, "错误", f"计算统计信息失败：\n{e}")
            self.progress_bar.setValue(0)
            self.status_label.setText("计算统计信息时出错")
    
    def _browse_output_dir(self):
        """浏览输出目录。"""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "选择输出目录",
            self.paths_config['paths'].get('reports', '')
        )
        
        if dir_path:
            self.output_dir = dir_path
            self.output_dir_edit.setText(dir_path)
            logger.info(f"已选择输出目录: {dir_path}")
    
    def _update_preview(self):
        """更新图表预览。"""
        if not self.statistics:
            QMessageBox.warning(self, "无数据", "请先计算统计信息。")
            return
        
        chart_type = self.preview_combo.currentText()
        
        try:
            self.preview_chart.clear_chart()
            visualizer = DefectVisualizer()
            
            if chart_type == "缺陷尺寸分布":
                # 获取所有缺陷面积
                all_areas = []
                for img_stats in self.statistics.get('per_image_stats', []):
                    all_areas.extend(img_stats.get('defect_areas', []))
                
                if all_areas:
                    fig = visualizer.plot_defect_size_distribution(all_areas)
                    self.preview_chart.display_figure(fig)
                else:
                    self.preview_chart.get_axes().text(
                        0.5, 0.5, '无可用缺陷数据',
                        ha='center', va='center', fontsize=14
                    )
                    self.preview_chart.refresh_chart()
            
            elif chart_type == "缺陷数量分布":
                defect_counts = [stats['num_defects'] 
                               for stats in self.statistics.get('per_image_stats', [])]
                
                if defect_counts:
                    fig = visualizer.plot_defect_count_per_image(defect_counts)
                    self.preview_chart.display_figure(fig)
            
            elif chart_type == "覆盖率分布":
                coverage_ratios = [stats['coverage_ratio'] 
                                 for stats in self.statistics.get('per_image_stats', [])
                                 if stats['coverage_ratio'] > 0]
                
                if coverage_ratios:
                    fig = visualizer.plot_coverage_ratio_distribution(coverage_ratios)
                    self.preview_chart.display_figure(fig)
            
            elif chart_type == "数据集摘要":
                dataset_viz = DatasetVisualizer()
                
                # 创建摘要字典
                summary = {
                    'total_images': self.statistics.get('total_images', 0),
                    'total_masks': self.statistics.get('total_images', 0),
                    'matched_pairs': self.statistics.get('images_processed', 0),
                    'defect_statistics': self.statistics
                }
                
                fig = dataset_viz.plot_dataset_summary(summary)
                self.preview_chart.display_figure(fig)
            
            logger.info(f"预览已更新: {chart_type}")
            
        except Exception as e:
            logger.error(f"更新预览失败: {e}", exc_info=True)
            QMessageBox.warning(self, "预览错误", f"更新预览失败：\n{e}")
    
    def _generate_report(self):
        """生成报告。"""
        # 验证
        if not self.mask_paths:
            QMessageBox.warning(self, "无数据", "请先加载掩码文件。")
            return
        
        if not self.statistics:
            QMessageBox.warning(self, "无统计信息", 
                              "请先计算统计信息。")
            return
        
        if not self.output_dir:
            QMessageBox.warning(self, "无输出目录",
                              "请选择输出目录。")
            return
        
        # 获取所选格式
        formats = []
        if self.excel_checkbox.isChecked():
            formats.append('excel')
        if self.pdf_checkbox.isChecked():
            formats.append('pdf')
        if self.html_checkbox.isChecked():
            formats.append('html')
        
        if not formats:
            QMessageBox.warning(self, "未选择格式",
                              "请至少选择一种报告格式。")
            return
        
        # 生成报告
        logger.info(f"正在生成报告，格式：{formats}")
        self.log_text.append(f"正在开始生成报告...")
        self.log_text.append(f"格式：{', '.join(formats)}")
        self.progress_bar.setValue(10)
        self.status_label.setText("正在生成报告...")
        self.generate_button.setEnabled(False)
        
        try:
            # 创建报告管理器
            report_manager = ReportManager()
            
            self.progress_bar.setValue(30)
            self.log_text.append("正在计算统计信息...")
            
            # 生成完整报告
            result = report_manager.generate_complete_report(
                self.mask_paths,
                self.output_dir,
                report_formats=formats
            )
            
            self.progress_bar.setValue(100)
            self.status_label.setText("报告生成成功！")
            self.log_text.append("报告生成完成！")
            
            # 启用打开按钮
            report_paths = result.get('report_paths', {})
            
            if 'excel' in report_paths:
                self.open_excel_button.setEnabled(True)
                self.log_text.append(f"Excel: {report_paths['excel']}")
            
            if 'pdf' in report_paths:
                self.open_pdf_button.setEnabled(True)
                self.log_text.append(f"PDF: {report_paths['pdf']}")
            
            if 'html' in report_paths:
                self.open_html_button.setEnabled(True)
                self.log_text.append(f"HTML: {report_paths['html']}")
            
            self.open_folder_button.setEnabled(True)
            
            # 发出信号
            self.report_generated.emit(report_paths)
            
            QMessageBox.information(self, "成功", 
                                  f"报告生成成功！\n\n已保存至：{self.output_dir}")
            
        except Exception as e:
            logger.error(f"生成报告失败: {e}", exc_info=True)
            QMessageBox.critical(self, "错误", f"生成报告失败：\n{e}")
            self.progress_bar.setValue(0)
            self.status_label.setText("生成报告时出错")
            self.log_text.append(f"错误: {e}")
        
        finally:
            self.generate_button.setEnabled(True)
    
    def _open_report(self, format_type: str):
        """打开生成的报告。"""
        if not self.output_dir:
            return
        
        output_dir = Path(self.output_dir)
        
        if format_type == 'excel':
            report_path = output_dir / 'defect_report.xlsx'
        elif format_type == 'pdf':
            report_path = output_dir / 'defect_report.pdf'
        elif format_type == 'html':
            report_path = output_dir / 'defect_report.html'
        else:
            return
        
        if report_path.exists():
            import os
            import platform
            
            if platform.system() == 'Windows':
                os.startfile(report_path)
            elif platform.system() == 'Darwin':  # macOS
                os.system(f'open "{report_path}"')
            else:  # Linux
                os.system(f'xdg-open "{report_path}"')
            
            logger.info(f"已打开报告: {report_path}")
        else:
            QMessageBox.warning(self, "文件未找到",
                              f"未找到报告文件：\n{report_path}")
    
    def _open_output_folder(self):
        """打开输出文件夹。"""
        if not self.output_dir or not Path(self.output_dir).exists():
            return
        
        import os
        import platform
        
        if platform.system() == 'Windows':
            os.startfile(self.output_dir)
        elif platform.system() == 'Darwin':  # macOS
            os.system(f'open "{self.output_dir}"')
        else:  # Linux
            os.system(f'xdg-open "{self.output_dir}"')
        
        logger.info(f"已打开输出文件夹: {self.output_dir}")
