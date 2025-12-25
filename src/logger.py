"""
应用程序的日志配置。

此模块提供了一个集中的日志配置，可在整个应用程序中使用。
它支持文件和控制台日志记录，并具有彩色输出。
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
    设置并配置带有文件和控制台处理程序的日志记录器。
    
    参数:
        name: 日志记录器名称
        level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: 日志文件路径（可选）
        console: 是否添加控制台处理程序
        
    返回:
        配置好的日志记录器实例
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # 移除现有处理程序
    logger.handlers = []
    
    # 创建格式化程序
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
    
    # 如果指定了 log_file，则添加文件处理程序
    if log_file:
        # 如果日志目录不存在，则创建它
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # 如果启用，则添加控制台处理程序
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    return logger


# 创建默认日志记录器
logger = setup_logger(
    name="IndustrialDefectSeg",
    level=os.getenv("LOG_LEVEL", "INFO"),
    log_file="logs/app.log",
    console=True
)


def get_logger(name: str = None) -> logging.Logger:
    """
    获取日志记录器实例。
    
    参数:
        name: 日志记录器名称（如果为 None，则使用默认名称）
        
    返回:
        日志记录器实例
    """
    if name:
        return logging.getLogger(name)
    return logger
