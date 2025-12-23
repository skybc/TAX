"""
Image processing utility functions.

This module provides functions for image loading, preprocessing, resizing,
normalization, and other image operations.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from src.logger import get_logger

logger = get_logger(__name__)


def load_image(
    image_path: Union[str, Path],
    mode: str = 'RGB'
) -> np.ndarray:
    """
    Load an image from file.
    
    Args:
        image_path: Path to image file
        mode: Color mode ('RGB', 'BGR', 'GRAY')
        
    Returns:
        Image as numpy array (H, W, C) for color or (H, W) for grayscale
        
    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image cannot be loaded
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    try:
        if mode == 'RGB':
            # Use PIL for RGB
            image = Image.open(image_path).convert('RGB')
            image = np.array(image)
        elif mode == 'BGR':
            # Use OpenCV for BGR
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        elif mode == 'GRAY':
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        logger.debug(f"Loaded image {image_path} with shape {image.shape}")
        return image
    
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        raise


def save_image(
    image: np.ndarray,
    output_path: Union[str, Path],
    quality: int = 95
) -> None:
    """
    Save image to file.
    
    Args:
        image: Image as numpy array
        output_path: Path to save image
        quality: JPEG quality (1-100)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert RGB to BGR if using OpenCV
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(str(output_path), image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    logger.debug(f"Saved image to {output_path}")


def resize_image(
    image: np.ndarray,
    target_size: Tuple[int, int],
    keep_aspect_ratio: bool = True,
    interpolation: int = cv2.INTER_LINEAR
) -> np.ndarray:
    """
    Resize image to target size.
    
    Args:
        image: Input image
        target_size: Target size as (width, height)
        keep_aspect_ratio: Whether to maintain aspect ratio
        interpolation: Interpolation method
        
    Returns:
        Resized image
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    if keep_aspect_ratio:
        # Calculate aspect ratio
        aspect = w / h
        target_aspect = target_w / target_h
        
        if aspect > target_aspect:
            # Width is the limiting factor
            new_w = target_w
            new_h = int(target_w / aspect)
        else:
            # Height is the limiting factor
            new_h = target_h
            new_w = int(target_h * aspect)
        
        resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
        
        # Pad to target size
        top = (target_h - new_h) // 2
        bottom = target_h - new_h - top
        left = (target_w - new_w) // 2
        right = target_w - new_w - left
        
        if len(image.shape) == 3:
            resized = cv2.copyMakeBorder(
                resized, top, bottom, left, right,
                cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )
        else:
            resized = cv2.copyMakeBorder(
                resized, top, bottom, left, right,
                cv2.BORDER_CONSTANT, value=0
            )
    else:
        resized = cv2.resize(image, target_size, interpolation=interpolation)
    
    return resized


def normalize_image(
    image: np.ndarray,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None
) -> np.ndarray:
    """
    Normalize image using mean and standard deviation.
    
    Args:
        image: Input image (H, W, C) in range [0, 255]
        mean: Mean values for each channel (defaults to ImageNet)
        std: Std values for each channel (defaults to ImageNet)
        
    Returns:
        Normalized image
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]  # ImageNet mean
    if std is None:
        std = [0.229, 0.224, 0.225]  # ImageNet std
    
    # Convert to float and normalize to [0, 1]
    image = image.astype(np.float32) / 255.0
    
    # Apply mean and std
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    
    image = (image - mean) / std
    
    return image


def denormalize_image(
    image: np.ndarray,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None
) -> np.ndarray:
    """
    Denormalize image back to [0, 255] range.
    
    Args:
        image: Normalized image
        mean: Mean values used for normalization
        std: Std values used for normalization
        
    Returns:
        Denormalized image in [0, 255] range
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]
    
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    
    image = (image * std) + mean
    image = (image * 255.0).clip(0, 255).astype(np.uint8)
    
    return image


def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    """Convert RGB image to BGR."""
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert BGR image to RGB."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def gray_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert grayscale image to RGB."""
    return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)


def image_to_tensor(image: np.ndarray) -> np.ndarray:
    """
    Convert image from HWC to CHW format for PyTorch.
    
    Args:
        image: Image in HWC format
        
    Returns:
        Image in CHW format
    """
    if len(image.shape) == 2:
        # Add channel dimension for grayscale
        image = image[np.newaxis, :, :]
    else:
        # Transpose from HWC to CHW
        image = np.transpose(image, (2, 0, 1))
    
    return image


def tensor_to_image(tensor: np.ndarray) -> np.ndarray:
    """
    Convert tensor from CHW to HWC format.
    
    Args:
        tensor: Tensor in CHW format
        
    Returns:
        Image in HWC format
    """
    if len(tensor.shape) == 3:
        if tensor.shape[0] == 1:
            # Grayscale: remove channel dimension
            image = tensor[0]
        else:
            # RGB: transpose from CHW to HWC
            image = np.transpose(tensor, (1, 2, 0))
    else:
        image = tensor
    
    return image


def crop_image(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int]
) -> np.ndarray:
    """
    Crop image using bounding box.
    
    Args:
        image: Input image
        bbox: Bounding box as (x1, y1, x2, y2)
        
    Returns:
        Cropped image
    """
    x1, y1, x2, y2 = bbox
    return image[y1:y2, x1:x2]


def pad_image(
    image: np.ndarray,
    pad_size: Union[int, Tuple[int, int, int, int]],
    value: Union[int, Tuple[int, int, int]] = 0
) -> np.ndarray:
    """
    Pad image with specified value.
    
    Args:
        image: Input image
        pad_size: Padding size (single value or (top, bottom, left, right))
        value: Padding value
        
    Returns:
        Padded image
    """
    if isinstance(pad_size, int):
        pad_size = (pad_size, pad_size, pad_size, pad_size)
    
    top, bottom, left, right = pad_size
    
    if len(image.shape) == 3:
        if not isinstance(value, tuple):
            value = (value, value, value)
    
    return cv2.copyMakeBorder(
        image, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=value
    )


def get_image_info(image: np.ndarray) -> dict:
    """
    Get image information.
    
    Args:
        image: Input image
        
    Returns:
        Dictionary with image info (shape, dtype, min, max)
    """
    info = {
        'shape': image.shape,
        'dtype': str(image.dtype),
        'min': float(image.min()),
        'max': float(image.max()),
    }
    
    if len(image.shape) == 3:
        info['channels'] = image.shape[2]
    else:
        info['channels'] = 1
    
    return info
