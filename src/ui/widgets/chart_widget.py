"""
Chart display widget for matplotlib integration with PyQt5.

This module provides:
- Matplotlib canvas for PyQt5
- Interactive chart display
- Zoom and pan functionality
- Chart refresh and update
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
    Matplotlib canvas widget for PyQt5.
    
    Provides a canvas to display matplotlib figures in Qt applications.
    """
    
    def __init__(self, parent: Optional[QWidget] = None, 
                 figsize: tuple = (8, 6), dpi: int = 100):
        """
        Initialize MatplotlibCanvas.
        
        Args:
            parent: Parent widget
            figsize: Figure size (width, height) in inches
            dpi: Dots per inch
        """
        # Create figure
        self.figure = Figure(figsize=figsize, dpi=dpi)
        self.axes = self.figure.add_subplot(111)
        
        # Initialize canvas
        super().__init__(self.figure)
        self.setParent(parent)
        
        # Set size policy
        FigureCanvas.setSizePolicy(
            self,
            QSizePolicy.Expanding,
            QSizePolicy.Expanding
        )
        FigureCanvas.updateGeometry(self)
        
        logger.debug("MatplotlibCanvas initialized")
    
    def clear(self):
        """Clear the canvas."""
        self.axes.clear()
        self.draw()
    
    def update_figure(self):
        """Update the figure display."""
        self.figure.tight_layout()
        self.draw()


class ChartWidget(QWidget):
    """
    Chart display widget with matplotlib canvas and toolbar.
    
    Provides:
    - Matplotlib canvas for chart display
    - Navigation toolbar (zoom, pan, save)
    - Refresh button
    """
    
    def __init__(self, parent: Optional[QWidget] = None,
                 figsize: tuple = (10, 6), dpi: int = 100,
                 show_toolbar: bool = True):
        """
        Initialize ChartWidget.
        
        Args:
            parent: Parent widget
            figsize: Figure size
            dpi: Figure DPI
            show_toolbar: Whether to show navigation toolbar
        """
        super().__init__(parent)
        
        self.figsize = figsize
        self.dpi = dpi
        self.show_toolbar = show_toolbar
        
        # Create canvas
        self.canvas = MatplotlibCanvas(self, figsize=figsize, dpi=dpi)
        
        # Create layout
        self._init_ui()
        
        logger.info("ChartWidget initialized")
    
    def _init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Add toolbar if enabled
        if self.show_toolbar:
            toolbar = NavigationToolbar(self.canvas, self)
            layout.addWidget(toolbar)
        
        # Add canvas
        layout.addWidget(self.canvas)
        
        # Add control buttons
        button_layout = QHBoxLayout()
        
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh_chart)
        button_layout.addWidget(self.refresh_button)
        
        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear_chart)
        button_layout.addWidget(self.clear_button)
        
        button_layout.addStretch()
        
        layout.addLayout(button_layout)
    
    def get_axes(self):
        """
        Get the matplotlib axes object.
        
        Returns:
            Matplotlib axes
        """
        return self.canvas.axes
    
    def get_figure(self):
        """
        Get the matplotlib figure object.
        
        Returns:
            Matplotlib figure
        """
        return self.canvas.figure
    
    def plot_data(self, x, y, **kwargs):
        """
        Plot data on the canvas.
        
        Args:
            x: X-axis data
            y: Y-axis data
            **kwargs: Additional plot arguments
        """
        self.canvas.axes.plot(x, y, **kwargs)
        self.canvas.update_figure()
    
    def bar_plot(self, x, height, **kwargs):
        """
        Create bar plot on the canvas.
        
        Args:
            x: X positions
            height: Bar heights
            **kwargs: Additional bar plot arguments
        """
        self.canvas.axes.bar(x, height, **kwargs)
        self.canvas.update_figure()
    
    def histogram(self, data, bins=30, **kwargs):
        """
        Create histogram on the canvas.
        
        Args:
            data: Data to plot
            bins: Number of bins
            **kwargs: Additional histogram arguments
        """
        self.canvas.axes.hist(data, bins=bins, **kwargs)
        self.canvas.update_figure()
    
    def scatter(self, x, y, **kwargs):
        """
        Create scatter plot on the canvas.
        
        Args:
            x: X-axis data
            y: Y-axis data
            **kwargs: Additional scatter arguments
        """
        self.canvas.axes.scatter(x, y, **kwargs)
        self.canvas.update_figure()
    
    def imshow(self, image, **kwargs):
        """
        Display image on the canvas.
        
        Args:
            image: Image array
            **kwargs: Additional imshow arguments
        """
        self.canvas.axes.imshow(image, **kwargs)
        self.canvas.update_figure()
    
    def set_title(self, title: str):
        """
        Set plot title.
        
        Args:
            title: Plot title
        """
        self.canvas.axes.set_title(title, fontsize=12, fontweight='bold')
        self.canvas.update_figure()
    
    def set_labels(self, xlabel: str, ylabel: str):
        """
        Set axis labels.
        
        Args:
            xlabel: X-axis label
            ylabel: Y-axis label
        """
        self.canvas.axes.set_xlabel(xlabel, fontsize=10)
        self.canvas.axes.set_ylabel(ylabel, fontsize=10)
        self.canvas.update_figure()
    
    def set_legend(self, *args, **kwargs):
        """
        Add legend to plot.
        
        Args:
            *args: Legend arguments
            **kwargs: Legend keyword arguments
        """
        self.canvas.axes.legend(*args, **kwargs)
        self.canvas.update_figure()
    
    def grid(self, visible: bool = True, **kwargs):
        """
        Toggle grid display.
        
        Args:
            visible: Whether to show grid
            **kwargs: Grid arguments
        """
        self.canvas.axes.grid(visible, **kwargs)
        self.canvas.update_figure()
    
    def clear_chart(self):
        """Clear the chart."""
        self.canvas.clear()
        logger.debug("Chart cleared")
    
    def refresh_chart(self):
        """Refresh the chart display."""
        self.canvas.update_figure()
        logger.debug("Chart refreshed")
    
    def save_chart(self, filepath: str, **kwargs):
        """
        Save chart to file.
        
        Args:
            filepath: Output file path
            **kwargs: Save arguments (dpi, format, etc.)
        """
        self.canvas.figure.savefig(filepath, bbox_inches='tight', **kwargs)
        logger.info(f"Chart saved to: {filepath}")
    
    def display_figure(self, figure):
        """
        Display an existing matplotlib figure.
        
        Args:
            figure: Matplotlib figure object
        """
        # Clear current figure
        self.canvas.figure.clear()
        
        # Copy axes from provided figure
        for ax in figure.get_axes():
            # Create new axes with same position
            new_ax = self.canvas.figure.add_axes(ax.get_position())
            
            # Copy all artists (lines, patches, text, etc.)
            for artist in ax.get_children():
                try:
                    new_ax.add_artist(artist)
                except:
                    pass
            
            # Copy properties
            new_ax.set_xlim(ax.get_xlim())
            new_ax.set_ylim(ax.get_ylim())
            new_ax.set_xlabel(ax.get_xlabel())
            new_ax.set_ylabel(ax.get_ylabel())
            new_ax.set_title(ax.get_title())
            
            if ax.get_legend():
                new_ax.legend()
        
        self.canvas.update_figure()
        logger.debug("External figure displayed")


class MultiChartWidget(QWidget):
    """
    Widget to display multiple charts in a grid.
    
    Provides:
    - Multiple matplotlib canvases
    - Grid layout
    - Individual chart management
    """
    
    def __init__(self, parent: Optional[QWidget] = None,
                 rows: int = 2, cols: int = 2,
                 figsize: tuple = (12, 10), dpi: int = 100):
        """
        Initialize MultiChartWidget.
        
        Args:
            parent: Parent widget
            rows: Number of rows in grid
            cols: Number of columns in grid
            figsize: Overall figure size
            dpi: Figure DPI
        """
        super().__init__(parent)
        
        self.rows = rows
        self.cols = cols
        self.figsize = figsize
        self.dpi = dpi
        
        # Create canvas with subplots
        self.canvas = MatplotlibCanvas(self, figsize=figsize, dpi=dpi)
        
        # Clear default axes and create grid
        self.canvas.figure.clear()
        self.axes_grid = []
        
        for i in range(rows * cols):
            ax = self.canvas.figure.add_subplot(rows, cols, i + 1)
            self.axes_grid.append(ax)
        
        # Create layout
        self._init_ui()
        
        logger.info(f"MultiChartWidget initialized with {rows}x{cols} grid")
    
    def _init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Add canvas
        layout.addWidget(self.canvas)
        
        # Add control buttons
        button_layout = QHBoxLayout()
        
        self.refresh_button = QPushButton("Refresh All")
        self.refresh_button.clicked.connect(self.refresh_all)
        button_layout.addWidget(self.refresh_button)
        
        self.clear_button = QPushButton("Clear All")
        self.clear_button.clicked.connect(self.clear_all)
        button_layout.addWidget(self.clear_button)
        
        button_layout.addStretch()
        
        layout.addLayout(button_layout)
    
    def get_axes(self, index: int):
        """
        Get axes at specific index.
        
        Args:
            index: Axes index (0 to rows*cols-1)
            
        Returns:
            Matplotlib axes
        """
        if 0 <= index < len(self.axes_grid):
            return self.axes_grid[index]
        else:
            logger.error(f"Invalid axes index: {index}")
            return None
    
    def clear_axes(self, index: int):
        """
        Clear specific axes.
        
        Args:
            index: Axes index
        """
        ax = self.get_axes(index)
        if ax:
            ax.clear()
            self.canvas.update_figure()
    
    def clear_all(self):
        """Clear all axes."""
        for ax in self.axes_grid:
            ax.clear()
        self.canvas.update_figure()
        logger.debug("All charts cleared")
    
    def refresh_all(self):
        """Refresh all charts."""
        self.canvas.update_figure()
        logger.debug("All charts refreshed")
    
    def save_charts(self, filepath: str, **kwargs):
        """
        Save all charts to file.
        
        Args:
            filepath: Output file path
            **kwargs: Save arguments
        """
        self.canvas.figure.savefig(filepath, bbox_inches='tight', **kwargs)
        logger.info(f"Charts saved to: {filepath}")
