"""
Post-processing utilities for segmentation masks.

This module provides:
- Morphological operations
- Connected component analysis
- Contour extraction
- Mask refinement
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional
from scipy import ndimage

from src.logger import get_logger

logger = get_logger(__name__)


def remove_small_objects(mask: np.ndarray, 
                         min_size: int = 100) -> np.ndarray:
    """
    Remove small connected components.
    
    Args:
        mask: Binary mask (HxW)
        min_size: Minimum object size in pixels
        
    Returns:
        Filtered mask
    """
    # Label connected components
    labeled, num_features = ndimage.label(mask)
    
    # Get sizes
    sizes = ndimage.sum(mask, labeled, range(num_features + 1))
    
    # Create mask of large objects
    mask_size = sizes >= min_size
    mask_size[0] = 0  # Background
    
    # Apply mask
    filtered = mask_size[labeled]
    
    return filtered.astype(np.uint8) * 255


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """
    Keep only the largest connected component.
    
    Args:
        mask: Binary mask (HxW)
        
    Returns:
        Mask with largest component only
    """
    # Label connected components
    labeled, num_features = ndimage.label(mask)
    
    if num_features == 0:
        return mask
    
    # Get sizes
    sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
    
    # Keep largest
    largest_label = np.argmax(sizes) + 1
    largest_mask = (labeled == largest_label).astype(np.uint8) * 255
    
    return largest_mask


def fill_holes(mask: np.ndarray, max_hole_size: Optional[int] = None) -> np.ndarray:
    """
    Fill holes in binary mask.
    
    Args:
        mask: Binary mask (HxW)
        max_hole_size: Maximum hole size to fill (None for all)
        
    Returns:
        Filled mask
    """
    # Binarize
    binary_mask = (mask > 0).astype(np.uint8)
    
    # Fill holes
    filled = ndimage.binary_fill_holes(binary_mask)
    
    if max_hole_size is not None:
        # Find holes
        holes = filled & ~binary_mask
        
        # Label holes
        labeled_holes, num_holes = ndimage.label(holes)
        
        # Get hole sizes
        hole_sizes = ndimage.sum(holes, labeled_holes, range(1, num_holes + 1))
        
        # Keep only small holes filled
        for i, size in enumerate(hole_sizes, 1):
            if size > max_hole_size:
                filled[labeled_holes == i] = 0
    
    return filled.astype(np.uint8) * 255


def morphological_opening(mask: np.ndarray, 
                         kernel_size: int = 5) -> np.ndarray:
    """
    Apply morphological opening (erosion followed by dilation).
    
    Args:
        mask: Binary mask (HxW)
        kernel_size: Size of structuring element
        
    Returns:
        Processed mask
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return opened


def morphological_closing(mask: np.ndarray, 
                         kernel_size: int = 5) -> np.ndarray:
    """
    Apply morphological closing (dilation followed by erosion).
    
    Args:
        mask: Binary mask (HxW)
        kernel_size: Size of structuring element
        
    Returns:
        Processed mask
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return closed


def smooth_contours(mask: np.ndarray, 
                   epsilon_factor: float = 0.01) -> np.ndarray:
    """
    Smooth mask contours using Douglas-Peucker algorithm.
    
    Args:
        mask: Binary mask (HxW)
        epsilon_factor: Approximation accuracy factor
        
    Returns:
        Smoothed mask
    """
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Smooth contours
    smoothed_mask = np.zeros_like(mask)
    
    for contour in contours:
        # Approximate contour
        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Draw smoothed contour
        cv2.fillPoly(smoothed_mask, [approx], 255)
    
    return smoothed_mask


def refine_mask(mask: np.ndarray,
               remove_small: bool = True,
               min_size: int = 100,
               fill_holes_flag: bool = True,
               smooth: bool = True,
               closing_size: int = 5) -> np.ndarray:
    """
    Apply complete mask refinement pipeline.
    
    Args:
        mask: Binary mask (HxW)
        remove_small: Whether to remove small objects
        min_size: Minimum object size
        fill_holes_flag: Whether to fill holes
        smooth: Whether to smooth contours
        closing_size: Kernel size for morphological closing
        
    Returns:
        Refined mask
    """
    refined = mask.copy()
    
    # Morphological closing (fill small gaps)
    if closing_size > 0:
        refined = morphological_closing(refined, closing_size)
    
    # Fill holes
    if fill_holes_flag:
        refined = fill_holes(refined)
    
    # Remove small objects
    if remove_small:
        refined = remove_small_objects(refined, min_size)
    
    # Smooth contours
    if smooth:
        refined = smooth_contours(refined, epsilon_factor=0.005)
    
    return refined


def extract_contours(mask: np.ndarray) -> List[np.ndarray]:
    """
    Extract contours from binary mask.
    
    Args:
        mask: Binary mask (HxW)
        
    Returns:
        List of contours (each is Nx1x2 array)
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def get_bounding_boxes(mask: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Get bounding boxes of all objects in mask.
    
    Args:
        mask: Binary mask (HxW)
        
    Returns:
        List of bounding boxes (x, y, w, h)
    """
    contours = extract_contours(mask)
    bboxes = []
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bboxes.append((x, y, w, h))
    
    return bboxes


def compute_mask_confidence(prob_map: np.ndarray, 
                           mask: np.ndarray) -> float:
    """
    Compute confidence score for mask based on probability map.
    
    Args:
        prob_map: Probability map (HxW) in [0, 1]
        mask: Binary mask (HxW)
        
    Returns:
        Confidence score (mean probability in mask region)
    """
    if mask.sum() == 0:
        return 0.0
    
    # Get probabilities in mask region
    mask_binary = (mask > 0).astype(bool)
    probs = prob_map[mask_binary]
    
    return float(np.mean(probs))


def apply_crf(image: np.ndarray, 
             prob_map: np.ndarray,
             num_iterations: int = 10) -> np.ndarray:
    """
    Apply Conditional Random Field (CRF) for mask refinement.
    
    Note: Requires pydensecrf library (optional dependency)
    
    Args:
        image: Original image (HxWx3)
        prob_map: Probability map (HxW) in [0, 1]
        num_iterations: Number of CRF iterations
        
    Returns:
        Refined mask
    """
    try:
        import pydensecrf.densecrf as dcrf
        from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral
        
        # Convert to labels
        labels = (prob_map > 0.5).astype(np.uint32)
        
        # Setup CRF
        h, w = prob_map.shape
        d = dcrf.DenseCRF2D(w, h, 2)
        
        # Unary potential
        U = unary_from_labels(labels, 2, gt_prob=0.7)
        d.setUnaryEnergy(U)
        
        # Pairwise potentials
        d.addPairwiseGaussian(sxy=3, compat=3)
        d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=image, compat=10)
        
        # Inference
        Q = d.inference(num_iterations)
        map_result = np.argmax(Q, axis=0).reshape(h, w)
        
        return (map_result * 255).astype(np.uint8)
        
    except ImportError:
        logger.warning("pydensecrf not installed, skipping CRF refinement")
        return (prob_map > 0.5).astype(np.uint8) * 255


def compute_mask_metrics(pred_mask: np.ndarray, 
                        gt_mask: np.ndarray) -> dict:
    """
    Compute metrics between predicted and ground truth masks.
    
    Args:
        pred_mask: Predicted mask (HxW)
        gt_mask: Ground truth mask (HxW)
        
    Returns:
        Dictionary with metrics (IoU, Dice, etc.)
    """
    # Binarize
    pred_binary = (pred_mask > 0).astype(bool)
    gt_binary = (gt_mask > 0).astype(bool)
    
    # Compute intersection and union
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()
    pred_sum = pred_binary.sum()
    gt_sum = gt_binary.sum()
    
    # Compute metrics
    iou = intersection / (union + 1e-6)
    dice = 2 * intersection / (pred_sum + gt_sum + 1e-6)
    
    # Pixel accuracy
    correct = (pred_binary == gt_binary).sum()
    total = pred_binary.size
    accuracy = correct / total
    
    # Precision and recall
    tp = intersection
    fp = pred_sum - intersection
    fn = gt_sum - intersection
    
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    
    return {
        'iou': float(iou),
        'dice': float(dice),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1)
    }


def create_comparison_image(image: np.ndarray,
                           pred_mask: np.ndarray,
                           gt_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Create side-by-side comparison image.
    
    Args:
        image: Original image (HxWx3)
        pred_mask: Predicted mask (HxW)
        gt_mask: Ground truth mask (HxW), optional
        
    Returns:
        Comparison image
    """
    from src.utils.mask_utils import overlay_mask_on_image
    
    # Create overlays
    pred_overlay = overlay_mask_on_image(image, pred_mask, alpha=0.5, color=(0, 255, 0))
    
    if gt_mask is not None:
        gt_overlay = overlay_mask_on_image(image, gt_mask, alpha=0.5, color=(255, 0, 0))
        
        # Concatenate: original | pred | gt
        comparison = np.concatenate([image, pred_overlay, gt_overlay], axis=1)
    else:
        # Concatenate: original | pred
        comparison = np.concatenate([image, pred_overlay], axis=1)
    
    return comparison
