"""
File I/O utility functions.

This module provides functions for file operations including reading,
writing, and managing file paths.
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
    Ensure a directory exists, create if it doesn't.
    
    Args:
        path: Directory path
        
    Returns:
        Path object of the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    
    Args:
        file_path: Path to YAML file
        
    Returns:
        Dictionary containing YAML content
        
    Raises:
        FileNotFoundError: If file doesn't exist
        yaml.YAMLError: If file is not valid YAML
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"YAML file not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    logger.debug(f"Loaded YAML config from {file_path}")
    return config


def save_yaml(data: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """
    Save dictionary to YAML file.
    
    Args:
        data: Dictionary to save
        file_path: Path to output YAML file
    """
    file_path = Path(file_path)
    ensure_dir(file_path.parent)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, default_flow_style=False, allow_unicode=True)
    
    logger.debug(f"Saved YAML config to {file_path}")


def load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Dictionary containing JSON content
        
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.debug(f"Loaded JSON from {file_path}")
    return data


def save_json(data: Dict[str, Any], file_path: Union[str, Path], indent: int = 2) -> None:
    """
    Save dictionary to JSON file.
    
    Args:
        data: Dictionary to save
        file_path: Path to output JSON file
        indent: Indentation level for pretty printing
    """
    file_path = Path(file_path)
    ensure_dir(file_path.parent)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
    
    logger.debug(f"Saved JSON to {file_path}")


def list_files(
    directory: Union[str, Path],
    extensions: Optional[List[str]] = None,
    recursive: bool = False
) -> List[Path]:
    """
    List files in a directory with optional filtering by extension.
    
    Args:
        directory: Directory path to search
        extensions: List of file extensions to filter (e.g., ['.jpg', '.png'])
        recursive: Whether to search recursively in subdirectories
        
    Returns:
        List of Path objects for matching files
    """
    directory = Path(directory)
    if not directory.exists():
        logger.warning(f"Directory not found: {directory}")
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
    
    logger.debug(f"Found {len(files)} files in {directory}")
    return sorted(files)


def copy_file(src: Union[str, Path], dst: Union[str, Path]) -> None:
    """
    Copy file from source to destination.
    
    Args:
        src: Source file path
        dst: Destination file path
    """
    src = Path(src)
    dst = Path(dst)
    
    if not src.exists():
        raise FileNotFoundError(f"Source file not found: {src}")
    
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)
    logger.debug(f"Copied {src} to {dst}")


def move_file(src: Union[str, Path], dst: Union[str, Path]) -> None:
    """
    Move file from source to destination.
    
    Args:
        src: Source file path
        dst: Destination file path
    """
    src = Path(src)
    dst = Path(dst)
    
    if not src.exists():
        raise FileNotFoundError(f"Source file not found: {src}")
    
    ensure_dir(dst.parent)
    shutil.move(str(src), str(dst))
    logger.debug(f"Moved {src} to {dst}")


def delete_file(file_path: Union[str, Path]) -> None:
    """
    Delete a file.
    
    Args:
        file_path: Path to file to delete
    """
    file_path = Path(file_path)
    if file_path.exists():
        file_path.unlink()
        logger.debug(f"Deleted {file_path}")


def get_file_size(file_path: Union[str, Path]) -> int:
    """
    Get file size in bytes.
    
    Args:
        file_path: Path to file
        
    Returns:
        File size in bytes
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    return file_path.stat().st_size


def format_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"
