"""
Main window for the Industrial Defect Segmentation System.

This is the primary UI component that hosts all other widgets and manages
the overall application state.
"""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QMenuBar, QMenu, QAction, QStatusBar, QLabel,
    QSplitter, QMessageBox
)

from src.logger import get_logger

logger = get_logger(__name__)


class MainWindow(QMainWindow):
    """
    Main application window.
    
    This window provides the primary interface for the application including:
    - Menu bar with File, Edit, View, Tools, Help menus
    - Toolbar for quick access to common actions
    - Central widget with image canvas and annotation tools
    - Side panels for file browser and properties
    - Status bar for displaying information
    """
    
    def __init__(self, config: dict, paths_config: dict, hyperparams: dict):
        """
        Initialize main window.
        
        Args:
            config: Application configuration
            paths_config: Paths configuration
            hyperparams: Model hyperparameters configuration
        """
        super().__init__()
        
        self.config = config
        self.paths_config = paths_config
        self.hyperparams = hyperparams
        
        # Window settings
        self.setWindowTitle(config['app']['name'])
        self.setGeometry(
            100, 100,
            config['ui']['window_width'],
            config['ui']['window_height']
        )
        
        # Initialize UI components
        self._init_ui()
        self._create_menus()
        self._create_toolbar()
        self._create_statusbar()
        
        logger.info("Main window initialized")
    
    def _init_ui(self):
        """Initialize the user interface."""
        # Create central widget with main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        
        # Create main splitter for resizable panels
        self.main_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(self.main_splitter)
        
        # Left panel - File browser (placeholder)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.addWidget(QLabel("File Browser\n(To be implemented)"))
        self.main_splitter.addWidget(left_panel)
        
        # Center panel - Image canvas (placeholder)
        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)
        center_layout.addWidget(QLabel("Image Canvas\n(To be implemented)"))
        self.main_splitter.addWidget(center_panel)
        
        # Right panel - Properties (placeholder)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.addWidget(QLabel("Properties Panel\n(To be implemented)"))
        self.main_splitter.addWidget(right_panel)
        
        # Set initial sizes for splitter
        self.main_splitter.setSizes([250, 900, 250])
    
    def _create_menus(self):
        """Create menu bar and menus."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        # File > Import
        import_action = QAction("&Import Images...", self)
        import_action.setShortcut("Ctrl+I")
        import_action.setStatusTip("Import images or videos")
        import_action.triggered.connect(self._on_import)
        file_menu.addAction(import_action)
        
        # File > Open Project
        open_action = QAction("&Open Project...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.setStatusTip("Open existing project")
        file_menu.addAction(open_action)
        
        # File > Save
        save_action = QAction("&Save", self)
        save_action.setShortcut("Ctrl+S")
        save_action.setStatusTip("Save current annotations")
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        # File > Exit
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.setStatusTip("Exit application")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Edit menu
        edit_menu = menubar.addMenu("&Edit")
        
        undo_action = QAction("&Undo", self)
        undo_action.setShortcut("Ctrl+Z")
        edit_menu.addAction(undo_action)
        
        redo_action = QAction("&Redo", self)
        redo_action.setShortcut("Ctrl+Y")
        edit_menu.addAction(redo_action)
        
        # View menu
        view_menu = menubar.addMenu("&View")
        
        zoom_in_action = QAction("Zoom &In", self)
        zoom_in_action.setShortcut("Ctrl++")
        view_menu.addAction(zoom_in_action)
        
        zoom_out_action = QAction("Zoom &Out", self)
        zoom_out_action.setShortcut("Ctrl+-")
        view_menu.addAction(zoom_out_action)
        
        # Tools menu
        tools_menu = menubar.addMenu("&Tools")
        
        train_action = QAction("&Train Model...", self)
        train_action.setStatusTip("Open model training dialog")
        tools_menu.addAction(train_action)
        
        predict_action = QAction("&Predict...", self)
        predict_action.setStatusTip("Run inference on images")
        tools_menu.addAction(predict_action)
        
        export_action = QAction("&Export Annotations...", self)
        export_action.setStatusTip("Export annotations to COCO/YOLO format")
        tools_menu.addAction(export_action)
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        
        about_action = QAction("&About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
    
    def _create_toolbar(self):
        """Create toolbar with quick actions."""
        toolbar = self.addToolBar("Main Toolbar")
        toolbar.setMovable(False)
        
        # Add placeholder actions
        # TODO: Add actual toolbar buttons with icons
        pass
    
    def _create_statusbar(self):
        """Create status bar."""
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        
        # Add permanent widgets to status bar
        self.status_label = QLabel("Ready")
        self.statusBar.addWidget(self.status_label)
        
        # Add coordinates label (for mouse position)
        self.coords_label = QLabel("Position: (0, 0)")
        self.statusBar.addPermanentWidget(self.coords_label)
    
    def _on_import(self):
        """Handle import action."""
        QMessageBox.information(
            self,
            "Import",
            "Import functionality will be implemented in Phase 2"
        )
        logger.info("Import action triggered")
    
    def _show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            f"About {self.config['app']['name']}",
            f"{self.config['app']['name']}\n"
            f"Version: {self.config['app']['version']}\n"
            f"Author: {self.config['app']['author']}\n\n"
            f"A complete industrial defect segmentation system with "
            f"SAM auto-annotation, model training, and inference."
        )
