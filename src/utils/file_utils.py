"""
文件 I/O 实用函数。

此模块提供文件操作函数，包括读取、写入和管理文件路径。
"""

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from src.logger import get_logger

logger = get_logger(__name__)


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    确保目录存在，如果不存在则创建。
    
    参数:
        path: 目录路径
        
    返回:
        目录的 Path 对象
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    加载 YAML 配置文件。
    
    参数:
        file_path: YAML 文件路径
        
    返回:
        包含 YAML 内容的字典
        
    抛出:
        FileNotFoundError: 如果文件不存在
        yaml.YAMLError: 如果文件不是有效的 YAML
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"未找到 YAML 文件: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    logger.debug(f"已从 {file_path} 加载 YAML 配置")
    return config


def save_yaml(data: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """
    将字典保存到 YAML 文件。
    
    参数:
        data: 要保存的字典
        file_path: 输出 YAML 文件的路径
    """
    file_path = Path(file_path)
    ensure_dir(file_path.parent)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, default_flow_style=False, allow_unicode=True)
    
    logger.debug(f"已将 YAML 配置保存到 {file_path}")


def load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    加载 JSON 文件。
    
    参数:
        file_path: JSON 文件路径
        
    返回:
        包含 JSON 内容的字典
        
    抛出:
        FileNotFoundError: 如果文件不存在
        json.JSONDecodeError: 如果文件不是有效的 JSON
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"未找到 JSON 文件: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.debug(f"已从 {file_path} 加载 JSON")
    return data


def save_json(data: Dict[str, Any], file_path: Union[str, Path], indent: int = 2) -> None:
    """
    将字典保存到 JSON 文件。
    
    参数:
        data: 要保存的字典
        file_path: 输出 JSON 文件的路径
        indent: 用于美化打印的缩进级别
    """
    file_path = Path(file_path)
    ensure_dir(file_path.parent)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
    
    logger.debug(f"已将 JSON 保存到 {file_path}")


def list_files(
    directory: Union[str, Path],
    extensions: Optional[List[str]] = None,
    recursive: bool = False
) -> List[Path]:
    """
    列出目录中的文件，并可选择按扩展名过滤。
    
    参数:
        directory: 要搜索的目录路径
        extensions: 要过滤的文件扩展名列表（例如 ['.jpg', '.png']）
        recursive: 是否在子目录中递归搜索
        
    返回:
        匹配文件的 Path 对象列表
    """
    directory = Path(directory)
    if not directory.exists():
        logger.warning(f"未找到目录: {directory}")
        return []
    
    if recursive:
        pattern = "**/*"
    else:
        pattern = "*"
    
    files = []
    for file_path in directory.glob(pattern):
        if file_path.is_file():
            if extensions is None or file_path.suffix.lower() in extensions:
                files.append(file_path)
    
    logger.debug(f"在 {directory} 中找到 {len(files)} 个文件")
    return sorted(files)


def copy_file(src: Union[str, Path], dst: Union[str, Path]) -> None:
    """
    将文件从源复制到目的地。
    
    参数:
        src: 源文件路径
        dst: 目的文件路径
    """
    src = Path(src)
    dst = Path(dst)
    
    if not src.exists():
        raise FileNotFoundError(f"未找到源文件: {src}")
    
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)
    logger.debug(f"已将 {src} 复制到 {dst}")


def move_file(src: Union[str, Path], dst: Union[str, Path]) -> None:
    """
    将文件从源移动到目的地。
    
    参数:
        src: 源文件路径
        dst: 目的文件路径
    """
    src = Path(src)
    dst = Path(dst)
    
    if not src.exists():
        raise FileNotFoundError(f"未找到源文件: {src}")
    
    ensure_dir(dst.parent)
    shutil.move(str(src), str(dst))
    logger.debug(f"已将 {src} 移动到 {dst}")


def delete_file(file_path: Union[str, Path]) -> None:
    """
    删除文件。
    
    参数:
        file_path: 要删除的文件路径
    """
    file_path = Path(file_path)
    if file_path.exists():
        file_path.unlink()
        logger.debug(f"已删除 {file_path}")


def get_file_size(file_path: Union[str, Path]) -> int:
    """
    获取文件大小（字节）。
    
    参数:
        file_path: 文件路径
        
    返回:
        文件大小（字节）
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"未找到文件: {file_path}")
    
    return file_path.stat().st_size


def format_size(size_bytes: int) -> str:
    """
    以人类可读的格式格式化文件大小。
    
    参数:
        size_bytes: 字节大小
        
    返回:
        格式化后的字符串（例如 "1.5 MB"）
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"
