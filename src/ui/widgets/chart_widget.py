"""
用于 PyQt5 中集成 matplotlib 的图表显示小部件。

此模块提供：
- 用于 PyQt5 的 Matplotlib 画布
- 交互式图表显示
- 缩放和平移功能
- 图表刷新和更新
"""

from typing import Optional
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QSizePolicy
)
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from src.logger import get_logger

logger = get_logger(__name__)


class MatplotlibCanvas(FigureCanvas):
    """
    用于 PyQt5 的 Matplotlib 画布小部件。
    
    提供一个在 Qt 应用程序中显示 matplotlib 图形的画布。
    """
    
    def __init__(self, parent: Optional[QWidget] = None, 
                 figsize: tuple = (8, 6), dpi: int = 100):
        """
        初始化 MatplotlibCanvas。
        
        参数:
            parent: 父小部件
            figsize: 图形尺寸 (宽度, 高度)，单位为英寸
            dpi: 每英寸点数
        """
        # 创建图形
        self.figure = Figure(figsize=figsize, dpi=dpi)
        self.axes = self.figure.add_subplot(111)
        
        # 初始化画布
        super().__init__(self.figure)
        self.setParent(parent)
        
        # 设置尺寸策略
        FigureCanvas.setSizePolicy(
            self,
            QSizePolicy.Expanding,
            QSizePolicy.Expanding
        )
        FigureCanvas.updateGeometry(self)
        
        logger.debug("MatplotlibCanvas 已初始化")
    
    def clear(self):
        """清除画布。"""
        self.axes.clear()
        self.draw()
    
    def update_figure(self):
        """更新图形显示。"""
        self.figure.tight_layout()
        self.draw()


class ChartWidget(QWidget):
    """
    带有 matplotlib 画布和工具栏的图表显示小部件。
    
    提供：
    - 用于图表显示的 Matplotlib 画布
    - 导航工具栏（缩放、平移、保存）
    - 刷新按钮
    """
    
    def __init__(self, parent: Optional[QWidget] = None,
                 figsize: tuple = (10, 6), dpi: int = 100,
                 show_toolbar: bool = True):
        """
        初始化 ChartWidget。
        
        参数:
            parent: 父小部件
            figsize: 图形尺寸
            dpi: 图形 DPI
            show_toolbar: 是否显示导航工具栏
        """
        super().__init__(parent)
        
        self.figsize = figsize
        self.dpi = dpi
        self.show_toolbar = show_toolbar
        
        # 创建画布
        self.canvas = MatplotlibCanvas(self, figsize=figsize, dpi=dpi)
        
        # 创建布局
        self._init_ui()
        
        logger.info("ChartWidget 已初始化")
    
    def _init_ui(self):
        """初始化用户界面。"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 如果启用，添加工具栏
        if self.show_toolbar:
            toolbar = NavigationToolbar(self.canvas, self)
            layout.addWidget(toolbar)
        
        # 添加画布
        layout.addWidget(self.canvas)
        
        # 添加控制按钮
        button_layout = QHBoxLayout()
        
        self.refresh_button = QPushButton("刷新")
        self.refresh_button.clicked.connect(self.refresh_chart)
        button_layout.addWidget(self.refresh_button)
        
        self.clear_button = QPushButton("清除")
        self.clear_button.clicked.connect(self.clear_chart)
        button_layout.addWidget(self.clear_button)
        
        button_layout.addStretch()
        
        layout.addLayout(button_layout)
    
    def get_axes(self):
        """
        获取 matplotlib axes 对象。
        
        返回:
            Matplotlib axes
        """
        return self.canvas.axes
    
    def get_figure(self):
        """
        获取 matplotlib figure 对象。
        
        返回:
            Matplotlib figure
        """
        return self.canvas.figure
    
    def plot_data(self, x, y, **kwargs):
        """
        在画布上绘制数据。
        
        参数:
            x: X 轴数据
            y: Y 轴数据
            **kwargs: 附加绘图参数
        """
        self.canvas.axes.plot(x, y, **kwargs)
        self.canvas.update_figure()
    
    def bar_plot(self, x, height, **kwargs):
        """
        在画布上创建柱状图。
        
        参数:
            x: X 轴位置
            height: 柱子高度
            **kwargs: 附加柱状图参数
        """
        self.canvas.axes.bar(x, height, **kwargs)
        self.canvas.update_figure()
    
    def histogram(self, data, bins=30, **kwargs):
        """
        在画布上创建直方图。
        
        参数:
            data: 要绘制的数据
            bins: 箱子数量
            **kwargs: 附加直方图参数
        """
        self.canvas.axes.hist(data, bins=bins, **kwargs)
        self.canvas.update_figure()
    
    def scatter(self, x, y, **kwargs):
        """
        在画布上创建散点图。
        
        参数:
            x: X 轴数据
            y: Y 轴数据
            **kwargs: 附加散点图参数
        """
        self.canvas.axes.scatter(x, y, **kwargs)
        self.canvas.update_figure()
    
    def imshow(self, image, **kwargs):
        """
        在画布上显示图像。
        
        参数:
            image: 图像数组
            **kwargs: 附加 imshow 参数
        """
        self.canvas.axes.imshow(image, **kwargs)
        self.canvas.update_figure()
    
    def set_title(self, title: str):
        """
        设置图表标题。
        
        参数:
            title: 图表标题
        """
        self.canvas.axes.set_title(title, fontsize=12, fontweight='bold')
        self.canvas.update_figure()
    
    def set_labels(self, xlabel: str, ylabel: str):
        """
        设置轴标签。
        
        参数:
            xlabel: X 轴标签
            ylabel: Y 轴标签
        """
        self.canvas.axes.set_xlabel(xlabel, fontsize=10)
        self.canvas.axes.set_ylabel(ylabel, fontsize=10)
        self.canvas.update_figure()
    
    def set_legend(self, *args, **kwargs):
        """
        为图表添加图例。
        
        参数:
            *args: 图例参数
            **kwargs: 图例关键字参数
        """
        self.canvas.axes.legend(*args, **kwargs)
        self.canvas.update_figure()
    
    def grid(self, visible: bool = True, **kwargs):
        """
        切换网格显示。
        
        参数:
            visible: 是否显示网格
            **kwargs: 网格参数
        """
        self.canvas.axes.grid(visible, **kwargs)
        self.canvas.update_figure()
    
    def clear_chart(self):
        """清除图表。"""
        self.canvas.clear()
        logger.debug("图表已清除")
    
    def refresh_chart(self):
        """刷新图表显示。"""
        self.canvas.update_figure()
        logger.debug("图表已刷新")
    
    def save_chart(self, filepath: str, **kwargs):
        """
        将图表保存到文件。
        
        参数:
            filepath: 输出文件路径
            **kwargs: 保存参数（dpi, format 等）
        """
        self.canvas.figure.savefig(filepath, bbox_inches='tight', **kwargs)
        logger.info(f"图表已保存到: {filepath}")
    
    def display_figure(self, figure):
        """
        显示现有的 matplotlib 图形。
        
        参数:
            figure: Matplotlib figure 对象
        """
        # 清除当前图形
        self.canvas.figure.clear()
        
        # 从提供的图形中复制 axes
        for ax in figure.get_axes():
            # 创建具有相同位置的新 axes
            new_ax = self.canvas.figure.add_axes(ax.get_position())
            
            # 复制所有 artist（线条、补丁、文本等）
            for artist in ax.get_children():
                try:
                    new_ax.add_artist(artist)
                except:
                    pass
            
            # 复制属性
            new_ax.set_xlim(ax.get_xlim())
            new_ax.set_ylim(ax.get_ylim())
            new_ax.set_xlabel(ax.get_xlabel())
            new_ax.set_ylabel(ax.get_ylabel())
            new_ax.set_title(ax.get_title())
            
            if ax.get_legend():
                new_ax.legend()
        
        self.canvas.update_figure()
        logger.debug("外部图形已显示")


class MultiChartWidget(QWidget):
    """
    在网格中显示多个图表的小部件。
    
    提供：
    - 多个 matplotlib 画布
    - 网格布局
    - 单个图表管理
    """
    
    def __init__(self, parent: Optional[QWidget] = None,
                 rows: int = 2, cols: int = 2,
                 figsize: tuple = (12, 10), dpi: int = 100):
        """
        初始化 MultiChartWidget。
        
        参数:
            parent: 父小部件
            rows: 网格中的行数
            cols: 网格中的列数
            figsize: 整体图形尺寸
            dpi: 图形 DPI
        """
        super().__init__(parent)
        
        self.rows = rows
        self.cols = cols
        self.figsize = figsize
        self.dpi = dpi
        
        # 创建带有子图的画布
        self.canvas = MatplotlibCanvas(self, figsize=figsize, dpi=dpi)
        
        # 清除默认 axes 并创建网格
        self.canvas.figure.clear()
        self.axes_grid = []
        
        for i in range(rows * cols):
            ax = self.canvas.figure.add_subplot(rows, cols, i + 1)
            self.axes_grid.append(ax)
        
        # 创建布局
        self._init_ui()
        
        logger.info(f"MultiChartWidget 已初始化，网格大小为 {rows}x{cols}")
    
    def _init_ui(self):
        """初始化用户界面。"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 添加画布
        layout.addWidget(self.canvas)
        
        # 添加控制按钮
        button_layout = QHBoxLayout()
        
        self.refresh_button = QPushButton("全部刷新")
        self.refresh_button.clicked.connect(self.refresh_all)
        button_layout.addWidget(self.refresh_button)
        
        self.clear_button = QPushButton("全部清除")
        self.clear_button.clicked.connect(self.clear_all)
        button_layout.addWidget(self.clear_button)
        
        button_layout.addStretch()
        
        layout.addLayout(button_layout)
    
    def get_axes(self, index: int):
        """
        获取指定索引处的 axes。
        
        参数:
            index: axes 索引 (0 到 rows*cols-1)
            
        返回:
            Matplotlib axes
        """
        if 0 <= index < len(self.axes_grid):
            return self.axes_grid[index]
        else:
            logger.error(f"无效的 axes 索引: {index}")
            return None
    
    def clear_axes(self, index: int):
        """
        清除指定的 axes。
        
        参数:
            index: axes 索引
        """
        ax = self.get_axes(index)
        if ax:
            ax.clear()
            self.canvas.update_figure()
    
    def clear_all(self):
        """清除所有 axes。"""
        for ax in self.axes_grid:
            ax.clear()
        self.canvas.update_figure()
        logger.debug("所有图表已清除")
    
    def refresh_all(self):
        """刷新所有图表。"""
        self.canvas.update_figure()
        logger.debug("所有图表已刷新")
    
    def save_charts(self, filepath: str, **kwargs):
        """
        将所有图表保存到文件。
        
        参数:
            filepath: 输出文件路径
            **kwargs: 保存参数
        """
        self.canvas.figure.savefig(filepath, bbox_inches='tight', **kwargs)
        logger.info(f"图表已保存到: {filepath}")
