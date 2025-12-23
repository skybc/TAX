"""
Main entry point for the Industrial Defect Segmentation System.

This application provides a complete workflow for defect segmentation including:
- Data management and annotation
- SAM (Segment Anything) auto-annotation
- Model training (U-Net, DeepLabV3+, YOLOv11-Seg)
- Inference and visualization
- Report generation
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from PyQt5.QtWidgets import QApplication

from src.logger import setup_logger, get_logger
from src.ui.main_window import MainWindow
from src.utils.file_utils import load_yaml, ensure_dir


def main():
    """Main application entry point."""
    # Set up logging
    logger = setup_logger(
        name="IndustrialDefectSeg",
        level="INFO",
        log_file="logs/app.log",
        console=True
    )
    logger.info("=" * 60)
    logger.info("Industrial Defect Segmentation System Starting...")
    logger.info("=" * 60)
    
    # Load configuration
    try:
        config_path = project_root / "config" / "config.yaml"
        paths_config_path = project_root / "config" / "paths.yaml"
        hyperparams_config_path = project_root / "config" / "hyperparams.yaml"
        
        config = load_yaml(config_path)
        paths_config = load_yaml(paths_config_path)
        hyperparams = load_yaml(hyperparams_config_path)
        
        logger.info("Configuration loaded successfully")
        
        # Ensure required directories exist
        for path_key, path_value in paths_config['paths'].items():
            if 'file' not in path_key.lower():
                ensure_dir(project_root / path_value)
        
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)
    
    # Create Qt application
    app = QApplication(sys.argv)
    app.setApplicationName(config['app']['name'])
    app.setApplicationVersion(config['app']['version'])
    app.setOrganizationName(config['app']['author'])
    
    # Create and show main window
    try:
        main_window = MainWindow(config, paths_config, hyperparams)
        main_window.show()
        
        logger.info("Main window created and displayed")
        logger.info("Application ready")
        
    except Exception as e:
        logger.error(f"Failed to create main window: {e}", exc_info=True)
        sys.exit(1)
    
    # Run application
    exit_code = app.exec_()
    
    logger.info("=" * 60)
    logger.info("Application closed")
    logger.info("=" * 60)
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
