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
from src.core.data_manager import DataManager
from src.ui.widgets.image_canvas import ImageCanvasWithInfo
from src.ui.widgets.file_browser import FileBrowser
from src.ui.dialogs.import_dialog import ImportDialog

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
        
        # Initialize DataManager
        from pathlib import Path
        data_root = Path(paths_config['paths']['data_root'])
        cache_size = config['performance']['max_cache_size']
        self.data_manager = DataManager(str(data_root), cache_size_mb=cache_size)
        
        # Current state
        self.current_image_path: str = None
        self.current_image_index: int = -1
        
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
        self._connect_signals()
        
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
        
        # Left panel - File browser
        self.file_browser = FileBrowser(self)
        self.main_splitter.addWidget(self.file_browser)
        
        # Center panel - Image canvas
        self.image_canvas = ImageCanvasWithInfo(self)
        self.main_splitter.addWidget(self.image_canvas)
        
        # Right panel - Properties (placeholder for now)
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
    
    def _connect_signals(self):
        """Connect signals and slots."""
        # File browser signals
        self.file_browser.file_selected.connect(self._on_file_selected)
        self.file_browser.folder_changed.connect(self._on_folder_changed)
        
        # Image canvas signals
        self.image_canvas.canvas.mouse_moved.connect(self._on_mouse_moved)
        
        logger.debug("Signals connected")
    
    def _on_file_selected(self, file_path: str):
        """
        Handle file selection from browser.
        
        Args:
            file_path: Path to selected file
        """
        logger.info(f"File selected: {file_path}")
        
        # Load image using DataManager
        image = self.data_manager.load_image(file_path)
        
        if image is not None:
            # Display in canvas
            self.image_canvas.load_image(image, file_path)
            self.current_image_path = file_path
            
            # Update status bar
            self.status_label.setText(f"Loaded: {file_path}")
        else:
            QMessageBox.warning(
                self,
                "Error",
                f"Failed to load image: {file_path}"
            )
    
    def _on_folder_changed(self, folder_path: str):
        """
        Handle folder change in browser.
        
        Args:
            folder_path: Path to new folder
        """
        logger.info(f"Folder changed: {folder_path}")
        self.status_label.setText(f"Folder: {folder_path}")
    
    def _on_mouse_moved(self, x: int, y: int):
        """
        Handle mouse move on canvas.
        
        Args:
            x: X coordinate
            y: Y coordinate
        """
        self.coords_label.setText(f"Position: ({x}, {y})")
    
    def _on_import(self):
        """Handle import action."""
        dialog = ImportDialog(self)
        
        if dialog.exec_() == ImportDialog.Accepted:
            imported_files = dialog.get_import_files()
            
            if imported_files:
                logger.info(f"Imported {len(imported_files)} file(s)")
                
                # If importing a folder, set it in file browser
                if dialog.import_type == "folder":
                    self.file_browser.set_folder(imported_files[0] if len(imported_files) == 1 else dialog.import_source)
                elif dialog.import_type == "files":
                    # Load files into data manager
                    self.data_manager.dataset['all'] = imported_files
                    # Optionally set folder to parent of first file
                    if imported_files:
                        from pathlib import Path
                        parent_folder = Path(imported_files[0]).parent
                        self.file_browser.set_folder(str(parent_folder))
                elif dialog.import_type == "video":
                    # Extract video frames
                    video_path = imported_files[0]
                    video_options = dialog.get_video_options()
                    
                    from pathlib import Path
                    output_dir = Path(self.paths_config['paths']['raw_data']) / "video_frames"
                    
                    QMessageBox.information(
                        self,
                        "Video Import",
                        f"Extracting frames from video...\nThis may take a while."
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
                            "Success",
                            f"Extracted {len(frame_paths)} frames"
                        )
                
                self.status_label.setText(f"Imported {len(imported_files)} file(s)")
        
        logger.info("Import dialog closed")
    
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
