"""
Statistical analysis utilities for defect segmentation.

This module provides:
- Defect statistics computation (count, size, location)
- Dataset analysis (distribution, class balance)
- Model performance analysis
- Comparative statistics
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import json
from collections import defaultdict, Counter

from src.logger import get_logger
from src.utils.mask_utils import load_mask, compute_mask_area

logger = get_logger(__name__)


class DefectStatistics:
    """
    Compute statistics from defect masks.
    
    Provides methods to analyze:
    - Defect counts and sizes
    - Spatial distribution
    - Defect characteristics
    """
    
    def __init__(self):
        """Initialize DefectStatistics."""
        self.stats: Dict = {}
    
    def compute_mask_statistics(self, mask: np.ndarray, 
                               image_name: Optional[str] = None) -> Dict:
        """
        Compute statistics for a single mask.
        
        Args:
            mask: Binary mask (HxW)
            image_name: Optional image name for reference
            
        Returns:
            Dictionary with statistics:
                - num_defects: Number of connected components
                - total_area: Total defect area in pixels
                - defect_areas: List of individual defect areas
                - defect_centroids: List of defect centroids (x, y)
                - defect_bboxes: List of bounding boxes (x, y, w, h)
                - coverage_ratio: Defect area / total area
                - largest_defect: Area of largest defect
                - smallest_defect: Area of smallest defect
                - mean_defect_size: Mean defect area
                - std_defect_size: Std of defect areas
        """
        from scipy import ndimage
        import cv2
        
        # Label connected components
        labeled, num_defects = ndimage.label(mask > 0)
        
        if num_defects == 0:
            return {
                'image_name': image_name,
                'num_defects': 0,
                'total_area': 0,
                'defect_areas': [],
                'defect_centroids': [],
                'defect_bboxes': [],
                'coverage_ratio': 0.0,
                'largest_defect': 0,
                'smallest_defect': 0,
                'mean_defect_size': 0.0,
                'std_defect_size': 0.0
            }
        
        # Compute properties for each defect
        defect_areas = []
        defect_centroids = []
        defect_bboxes = []
        
        for i in range(1, num_defects + 1):
            # Extract single defect
            defect_mask = (labeled == i).astype(np.uint8)
            
            # Compute area
            area = np.sum(defect_mask)
            defect_areas.append(int(area))
            
            # Compute centroid
            y_coords, x_coords = np.where(defect_mask > 0)
            if len(x_coords) > 0:
                centroid_x = float(np.mean(x_coords))
                centroid_y = float(np.mean(y_coords))
                defect_centroids.append((centroid_x, centroid_y))
            
            # Compute bounding box
            contours, _ = cv2.findContours(
                defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                x, y, w, h = cv2.boundingRect(contours[0])
                defect_bboxes.append((int(x), int(y), int(w), int(h)))
        
        # Aggregate statistics
        total_area = sum(defect_areas)
        image_area = mask.shape[0] * mask.shape[1]
        coverage_ratio = total_area / image_area if image_area > 0 else 0.0
        
        stats = {
            'image_name': image_name,
            'num_defects': num_defects,
            'total_area': total_area,
            'defect_areas': defect_areas,
            'defect_centroids': defect_centroids,
            'defect_bboxes': defect_bboxes,
            'coverage_ratio': float(coverage_ratio),
            'largest_defect': max(defect_areas) if defect_areas else 0,
            'smallest_defect': min(defect_areas) if defect_areas else 0,
            'mean_defect_size': float(np.mean(defect_areas)) if defect_areas else 0.0,
            'std_defect_size': float(np.std(defect_areas)) if defect_areas else 0.0
        }
        
        return stats
    
    def compute_batch_statistics(self, mask_paths: List[str]) -> Dict:
        """
        Compute statistics for multiple masks.
        
        Args:
            mask_paths: List of mask file paths
            
        Returns:
            Dictionary with aggregate statistics:
                - total_images: Number of images
                - images_with_defects: Number of images containing defects
                - total_defects: Total number of defects
                - total_defect_area: Total defect area across all images
                - mean_defects_per_image: Average defects per image
                - mean_coverage_ratio: Average coverage ratio
                - defect_size_distribution: Histogram of defect sizes
                - per_image_stats: List of individual image statistics
        """
        logger.info(f"Computing statistics for {len(mask_paths)} masks...")
        
        all_stats = []
        all_defect_areas = []
        total_defects = 0
        images_with_defects = 0
        total_defect_area = 0
        coverage_ratios = []
        
        for mask_path in mask_paths:
            try:
                # Load mask
                mask = load_mask(mask_path)
                if mask is None:
                    logger.warning(f"Failed to load mask: {mask_path}")
                    continue
                
                # Compute stats
                image_name = Path(mask_path).stem
                stats = self.compute_mask_statistics(mask, image_name)
                all_stats.append(stats)
                
                # Aggregate
                if stats['num_defects'] > 0:
                    images_with_defects += 1
                    total_defects += stats['num_defects']
                    total_defect_area += stats['total_area']
                    all_defect_areas.extend(stats['defect_areas'])
                    coverage_ratios.append(stats['coverage_ratio'])
                
            except Exception as e:
                logger.error(f"Error processing {mask_path}: {e}")
                continue
        
        # Compute defect size distribution (histogram)
        if all_defect_areas:
            hist, bin_edges = np.histogram(all_defect_areas, bins=20)
            size_distribution = {
                'histogram': hist.tolist(),
                'bin_edges': bin_edges.tolist()
            }
        else:
            size_distribution = {'histogram': [], 'bin_edges': []}
        
        # Aggregate statistics
        batch_stats = {
            'total_images': len(mask_paths),
            'images_processed': len(all_stats),
            'images_with_defects': images_with_defects,
            'images_without_defects': len(all_stats) - images_with_defects,
            'total_defects': total_defects,
            'total_defect_area': total_defect_area,
            'mean_defects_per_image': total_defects / len(all_stats) if all_stats else 0.0,
            'mean_coverage_ratio': np.mean(coverage_ratios) if coverage_ratios else 0.0,
            'std_coverage_ratio': np.std(coverage_ratios) if coverage_ratios else 0.0,
            'defect_size_distribution': size_distribution,
            'per_image_stats': all_stats
        }
        
        logger.info(f"Statistics computed: {total_defects} defects in {images_with_defects} images")
        
        return batch_stats
    
    def compute_spatial_distribution(self, mask_paths: List[str], 
                                    grid_size: Tuple[int, int] = (10, 10)) -> np.ndarray:
        """
        Compute spatial distribution of defects across images.
        
        Args:
            mask_paths: List of mask file paths
            grid_size: Grid size for spatial binning (rows, cols)
            
        Returns:
            Heatmap array (grid_size) showing defect frequency
        """
        heatmap = np.zeros(grid_size, dtype=np.float32)
        count = 0
        
        for mask_path in mask_paths:
            try:
                mask = load_mask(mask_path)
                if mask is None:
                    continue
                
                # Resize mask to grid size
                import cv2
                resized = cv2.resize(
                    mask.astype(np.float32), 
                    (grid_size[1], grid_size[0]),
                    interpolation=cv2.INTER_AREA
                )
                
                # Accumulate
                heatmap += (resized > 0).astype(np.float32)
                count += 1
                
            except Exception as e:
                logger.error(f"Error processing {mask_path}: {e}")
                continue
        
        # Normalize
        if count > 0:
            heatmap /= count
        
        return heatmap


class DatasetStatistics:
    """
    Compute dataset-level statistics.
    
    Analyzes dataset characteristics:
    - Class distribution
    - Train/val/test split statistics
    - Data quality metrics
    """
    
    def __init__(self):
        """Initialize DatasetStatistics."""
        pass
    
    def compute_dataset_summary(self, image_dir: str, mask_dir: str) -> Dict:
        """
        Compute summary statistics for a dataset.
        
        Args:
            image_dir: Directory containing images
            mask_dir: Directory containing masks
            
        Returns:
            Dictionary with dataset summary
        """
        from src.utils.file_utils import list_files
        
        image_dir = Path(image_dir)
        mask_dir = Path(mask_dir)
        
        # Get file lists
        image_files = list_files(image_dir, extensions=['.jpg', '.jpeg', '.png', '.bmp', '.tif'])
        mask_files = list_files(mask_dir, extensions=['.png', '.tif'])
        
        # Match images and masks
        image_names = {Path(f).stem for f in image_files}
        mask_names = {Path(f).stem for f in mask_files}
        
        matched = image_names & mask_names
        images_only = image_names - mask_names
        masks_only = mask_names - image_names
        
        # Compute mask statistics
        matched_mask_paths = [str(mask_dir / f"{name}.png") for name in matched 
                             if (mask_dir / f"{name}.png").exists()]
        
        defect_stats = DefectStatistics()
        batch_stats = defect_stats.compute_batch_statistics(matched_mask_paths)
        
        summary = {
            'total_images': len(image_files),
            'total_masks': len(mask_files),
            'matched_pairs': len(matched),
            'images_without_masks': len(images_only),
            'masks_without_images': len(masks_only),
            'defect_statistics': batch_stats
        }
        
        return summary
    
    def compute_split_statistics(self, split_file: str) -> Dict:
        """
        Compute statistics for train/val/test splits.
        
        Args:
            split_file: Path to split file (e.g., train.txt)
            
        Returns:
            Dictionary with split statistics
        """
        split_file = Path(split_file)
        
        if not split_file.exists():
            logger.warning(f"Split file not found: {split_file}")
            return {}
        
        # Read split file
        with open(split_file, 'r') as f:
            files = [line.strip() for line in f if line.strip()]
        
        stats = {
            'split_name': split_file.stem,
            'num_samples': len(files),
            'files': files
        }
        
        return stats


class ModelPerformanceAnalyzer:
    """
    Analyze model performance metrics.
    
    Computes and visualizes:
    - Training history analysis
    - Confusion matrix
    - Performance comparison
    """
    
    def __init__(self):
        """Initialize ModelPerformanceAnalyzer."""
        pass
    
    def analyze_training_history(self, history_file: str) -> Dict:
        """
        Analyze training history from log file.
        
        Args:
            history_file: Path to training history JSON file
            
        Returns:
            Dictionary with analyzed metrics
        """
        history_file = Path(history_file)
        
        if not history_file.exists():
            logger.warning(f"History file not found: {history_file}")
            return {}
        
        # Load history
        with open(history_file, 'r') as f:
            history = json.load(f)
        
        # Analyze trends
        analysis = {
            'total_epochs': len(history.get('train_loss', [])),
            'best_train_loss': min(history.get('train_loss', [float('inf')])),
            'best_val_loss': min(history.get('val_loss', [float('inf')])),
            'final_train_loss': history.get('train_loss', [])[-1] if history.get('train_loss') else None,
            'final_val_loss': history.get('val_loss', [])[-1] if history.get('val_loss') else None,
        }
        
        # Check for overfitting
        if history.get('train_loss') and history.get('val_loss'):
            train_losses = history['train_loss']
            val_losses = history['val_loss']
            
            # Compare last 5 epochs
            if len(train_losses) >= 5:
                recent_train = np.mean(train_losses[-5:])
                recent_val = np.mean(val_losses[-5:])
                
                analysis['overfitting_indicator'] = recent_val - recent_train
                analysis['is_overfitting'] = recent_val > recent_train * 1.2
        
        return analysis
    
    def compute_confusion_matrix(self, pred_masks: List[np.ndarray],
                                gt_masks: List[np.ndarray]) -> Dict:
        """
        Compute confusion matrix for predictions vs ground truth.
        
        Args:
            pred_masks: List of predicted masks
            gt_masks: List of ground truth masks
            
        Returns:
            Dictionary with confusion matrix and derived metrics
        """
        if len(pred_masks) != len(gt_masks):
            raise ValueError("Number of predictions and ground truths must match")
        
        # Accumulate confusion matrix elements
        tp_total = 0
        tn_total = 0
        fp_total = 0
        fn_total = 0
        
        for pred, gt in zip(pred_masks, gt_masks):
            # Flatten and binarize
            pred_flat = (pred.flatten() > 0).astype(np.uint8)
            gt_flat = (gt.flatten() > 0).astype(np.uint8)
            
            # Compute confusion matrix elements
            tp = np.sum((pred_flat == 1) & (gt_flat == 1))
            tn = np.sum((pred_flat == 0) & (gt_flat == 0))
            fp = np.sum((pred_flat == 1) & (gt_flat == 0))
            fn = np.sum((pred_flat == 0) & (gt_flat == 1))
            
            tp_total += tp
            tn_total += tn
            fp_total += fp
            fn_total += fn
        
        # Compute metrics
        accuracy = (tp_total + tn_total) / (tp_total + tn_total + fp_total + fn_total) if (tp_total + tn_total + fp_total + fn_total) > 0 else 0
        precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
        recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        iou = tp_total / (tp_total + fp_total + fn_total) if (tp_total + fp_total + fn_total) > 0 else 0
        
        confusion_matrix = {
            'tp': int(tp_total),
            'tn': int(tn_total),
            'fp': int(fp_total),
            'fn': int(fn_total),
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'iou': float(iou)
        }
        
        return confusion_matrix
    
    def compare_models(self, model_results: Dict[str, List[np.ndarray]],
                      gt_masks: List[np.ndarray]) -> Dict:
        """
        Compare multiple model predictions.
        
        Args:
            model_results: Dictionary mapping model names to prediction lists
            gt_masks: Ground truth masks
            
        Returns:
            Dictionary with comparison results
        """
        comparison = {}
        
        for model_name, pred_masks in model_results.items():
            logger.info(f"Evaluating {model_name}...")
            
            # Compute confusion matrix
            cm = self.compute_confusion_matrix(pred_masks, gt_masks)
            comparison[model_name] = cm
        
        # Rank models by IoU
        ranked = sorted(comparison.items(), key=lambda x: x[1]['iou'], reverse=True)
        
        comparison['ranking'] = [name for name, _ in ranked]
        comparison['best_model'] = ranked[0][0] if ranked else None
        
        return comparison


def save_statistics(stats: Dict, output_path: str):
    """
    Save statistics to JSON file.
    
    Args:
        stats: Statistics dictionary
        output_path: Output file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Statistics saved to: {output_path}")


def load_statistics(input_path: str) -> Dict:
    """
    Load statistics from JSON file.
    
    Args:
        input_path: Input file path
        
    Returns:
        Statistics dictionary
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        logger.error(f"Statistics file not found: {input_path}")
        return {}
    
    with open(input_path, 'r') as f:
        stats = json.load(f)
    
    logger.info(f"Statistics loaded from: {input_path}")
    return stats
