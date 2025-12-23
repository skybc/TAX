"""
Logging configuration for the application.

This module provides a centralized logging configuration that can be used
throughout the application. It supports both file and console logging with
colored output.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

import colorlog


def setup_logger(
    name: str = "app",
    level: str = "INFO",
    log_file: Optional[str] = None,
    console: bool = True,
) -> logging.Logger:
    """
    Set up and configure a logger with file and console handlers.
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        console: Whether to add console handler
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    logger.handlers = []
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_formatter = colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s%(reset)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    
    # Add file handler if log_file is specified
    if log_file:
        # Create log directory if it doesn't exist
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # Add console handler if enabled
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    return logger


# Create default logger
logger = setup_logger(
    name="IndustrialDefectSeg",
    level=os.getenv("LOG_LEVEL", "INFO"),
    log_file="logs/app.log",
    console=True
)


def get_logger(name: str = None) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name (uses default if None)
        
    Returns:
        Logger instance
    """
    if name:
        return logging.getLogger(name)
    return logger
