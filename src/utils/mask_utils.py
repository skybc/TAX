"""
Mask processing utility functions.

This module provides functions for mask manipulation, encoding, decoding,
and conversion between different formats.
"""

from typing import List, Tuple, Union

import cv2
import numpy as np
from pycocotools import mask as coco_mask

from src.logger import get_logger

logger = get_logger(__name__)


def binary_mask_to_rle(mask: np.ndarray) -> dict:
    """
    Convert binary mask to RLE (Run-Length Encoding) format.
    
    Args:
        mask: Binary mask as numpy array (H, W)
        
    Returns:
        RLE encoded mask as dictionary
    """
    # Ensure mask is in Fortran order (column-major)
    mask = np.asfortranarray(mask.astype(np.uint8))
    rle = coco_mask.encode(mask)
    
    # Convert bytes to string for JSON serialization
    rle['counts'] = rle['counts'].decode('utf-8')
    
    return rle


def rle_to_binary_mask(rle: dict) -> np.ndarray:
    """
    Convert RLE encoded mask to binary mask.
    
    Args:
        rle: RLE encoded mask
        
    Returns:
        Binary mask as numpy array (H, W)
    """
    # Convert string back to bytes if necessary
    if isinstance(rle['counts'], str):
        rle['counts'] = rle['counts'].encode('utf-8')
    
    mask = coco_mask.decode(rle)
    return mask


def polygon_to_mask(
    polygon: List[List[float]],
    image_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Convert polygon to binary mask.
    
    Args:
        polygon: List of polygons, each as list of [x, y] coordinates
        image_shape: Target mask shape as (height, width)
        
    Returns:
        Binary mask
    """
    height, width = image_shape
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Convert polygon to numpy array
    if isinstance(polygon[0], list):
        # Multiple polygons
        for poly in polygon:
            pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], 1)
    else:
        # Single polygon
        pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 1)
    
    return mask


def mask_to_polygon(mask: np.ndarray) -> List[List[float]]:
    """
    Convert binary mask to polygon contours.
    
    Args:
        mask: Binary mask (H, W)
        
    Returns:
        List of polygons, each as list of [x, y] coordinates
    """
    # Find contours
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    polygons = []
    for contour in contours:
        # Simplify contour
        epsilon = 0.001 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Convert to list format
        polygon = approx.reshape(-1, 2).tolist()
        if len(polygon) >= 3:  # Valid polygon needs at least 3 points
            polygons.append(polygon)
    
    return polygons


def mask_to_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Get bounding box from binary mask.
    
    Args:
        mask: Binary mask (H, W)
        
    Returns:
        Bounding box as (x1, y1, x2, y2)
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    if not rows.any() or not cols.any():
        return (0, 0, 0, 0)
    
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    
    return (int(x1), int(y1), int(x2 + 1), int(y2 + 1))


def bbox_to_mask(
    bbox: Tuple[int, int, int, int],
    image_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Create binary mask from bounding box.
    
    Args:
        bbox: Bounding box as (x1, y1, x2, y2)
        image_shape: Target mask shape as (height, width)
        
    Returns:
        Binary mask
    """
    height, width = image_shape
    mask = np.zeros((height, width), dtype=np.uint8)
    
    x1, y1, x2, y2 = bbox
    mask[y1:y2, x1:x2] = 1
    
    return mask


def dilate_mask(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Dilate binary mask.
    
    Args:
        mask: Binary mask
        kernel_size: Size of dilation kernel
        
    Returns:
        Dilated mask
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
    return dilated


def erode_mask(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Erode binary mask.
    
    Args:
        mask: Binary mask
        kernel_size: Size of erosion kernel
        
    Returns:
        Eroded mask
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
    return eroded


def open_mask(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Morphological opening (erosion followed by dilation).
    
    Args:
        mask: Binary mask
        kernel_size: Size of kernel
        
    Returns:
        Opened mask
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    return opened


def close_mask(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Morphological closing (dilation followed by erosion).
    
    Args:
        mask: Binary mask
        kernel_size: Size of kernel
        
    Returns:
        Closed mask
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    return closed


def remove_small_components(
    mask: np.ndarray,
    min_area: int = 100
) -> np.ndarray:
    """
    Remove small connected components from mask.
    
    Args:
        mask: Binary mask
        min_area: Minimum area for components to keep
        
    Returns:
        Filtered mask
    """
    # Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    
    # Create output mask
    output_mask = np.zeros_like(mask)
    
    # Keep components larger than min_area
    for i in range(1, num_labels):  # Skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            output_mask[labels == i] = 1
    
    return output_mask


def get_largest_component(mask: np.ndarray) -> np.ndarray:
    """
    Keep only the largest connected component.
    
    Args:
        mask: Binary mask
        
    Returns:
        Mask with only largest component
    """
    # Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    
    if num_labels <= 1:
        return mask
    
    # Find largest component (excluding background)
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = np.argmax(areas) + 1
    
    # Create output mask
    output_mask = (labels == largest_label).astype(np.uint8)
    
    return output_mask


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """
    Fill holes in binary mask.
    
    Args:
        mask: Binary mask
        
    Returns:
        Mask with holes filled
    """
    # Use flood fill from border to find background
    h, w = mask.shape
    mask_copy = mask.copy()
    
    # Create a larger mask for flood fill
    mask_floodfill = np.zeros((h + 2, w + 2), dtype=np.uint8)
    mask_floodfill[1:-1, 1:-1] = mask_copy
    
    # Flood fill from (0, 0)
    cv2.floodFill(mask_floodfill, None, (0, 0), 1)
    
    # Invert to get holes
    holes = 1 - mask_floodfill[1:-1, 1:-1]
    
    # Combine with original mask
    filled = mask | holes
    
    return filled


def overlay_mask_on_image(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.5
) -> np.ndarray:
    """
    Overlay binary mask on image with transparency.
    
    Args:
        image: RGB image (H, W, 3)
        mask: Binary mask (H, W)
        color: RGB color for mask
        alpha: Transparency (0=transparent, 1=opaque)
        
    Returns:
        Image with mask overlay
    """
    # Convert grayscale to RGB if necessary
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Create colored mask
    colored_mask = np.zeros_like(image)
    colored_mask[mask > 0] = color
    
    # Blend image and mask
    overlay = cv2.addWeighted(image, 1, colored_mask, alpha, 0)
    
    return overlay


def compute_mask_area(mask: np.ndarray) -> int:
    """
    Compute area of binary mask (number of pixels).
    
    Args:
        mask: Binary mask
        
    Returns:
        Number of positive pixels
    """
    return int(np.sum(mask > 0))


def compute_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Compute Intersection over Union (IoU) between two masks.
    
    Args:
        mask1: First binary mask
        mask2: Second binary mask
        
    Returns:
        IoU score (0-1)
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    if union == 0:
        return 0.0
    
    return float(intersection / union)
