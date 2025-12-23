"""
Visualization utilities for defect segmentation.

This module provides:
- Chart generation (matplotlib/seaborn)
- Defect heatmaps
- Comparison visualizations
- Distribution plots
- Training curve visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import cv2

from src.logger import get_logger

logger = get_logger(__name__)

# Set matplotlib style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class DefectVisualizer:
    """
    Visualize defect statistics and masks.
    
    Provides methods to create:
    - Defect size distributions
    - Spatial heatmaps
    - Overlay visualizations
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 100):
        """
        Initialize DefectVisualizer.
        
        Args:
            figsize: Default figure size
            dpi: Figure DPI
        """
        self.figsize = figsize
        self.dpi = dpi
    
    def plot_defect_size_distribution(self, defect_areas: List[int],
                                     output_path: Optional[str] = None,
                                     title: str = "Defect Size Distribution") -> plt.Figure:
        """
        Plot histogram of defect sizes.
        
        Args:
            defect_areas: List of defect areas in pixels
            output_path: Optional path to save figure
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Plot histogram
        ax.hist(defect_areas, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        
        # Add statistics
        mean_size = np.mean(defect_areas)
        median_size = np.median(defect_areas)
        
        ax.axvline(mean_size, color='red', linestyle='--', linewidth=2, 
                  label=f'Mean: {mean_size:.1f}')
        ax.axvline(median_size, color='green', linestyle='--', linewidth=2,
                  label=f'Median: {median_size:.1f}')
        
        # Labels and title
        ax.set_xlabel('Defect Area (pixels)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            fig.savefig(output_path, bbox_inches='tight')
            logger.info(f"Saved defect size distribution to: {output_path}")
        
        return fig
    
    def plot_defect_count_per_image(self, defect_counts: List[int],
                                   output_path: Optional[str] = None,
                                   title: str = "Defects Per Image") -> plt.Figure:
        """
        Plot distribution of defect counts per image.
        
        Args:
            defect_counts: List of defect counts per image
            output_path: Optional path to save figure
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Plot bar chart
        unique_counts, frequencies = np.unique(defect_counts, return_counts=True)
        
        ax.bar(unique_counts, frequencies, alpha=0.7, color='coral', edgecolor='black')
        
        # Labels
        ax.set_xlabel('Number of Defects', fontsize=12)
        ax.set_ylabel('Number of Images', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if output_path:
            fig.savefig(output_path, bbox_inches='tight')
            logger.info(f"Saved defect count plot to: {output_path}")
        
        return fig
    
    def plot_spatial_heatmap(self, heatmap: np.ndarray,
                            output_path: Optional[str] = None,
                            title: str = "Defect Spatial Distribution") -> plt.Figure:
        """
        Plot spatial heatmap of defects.
        
        Args:
            heatmap: 2D array representing defect frequency
            output_path: Optional path to save figure
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Plot heatmap
        im = ax.imshow(heatmap, cmap='hot', interpolation='bilinear')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Defect Frequency', fontsize=12)
        
        # Labels
        ax.set_xlabel('X Position', fontsize=12)
        ax.set_ylabel('Y Position', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if output_path:
            fig.savefig(output_path, bbox_inches='tight')
            logger.info(f"Saved spatial heatmap to: {output_path}")
        
        return fig
    
    def plot_coverage_ratio_distribution(self, coverage_ratios: List[float],
                                        output_path: Optional[str] = None,
                                        title: str = "Defect Coverage Ratio") -> plt.Figure:
        """
        Plot distribution of defect coverage ratios.
        
        Args:
            coverage_ratios: List of coverage ratios (0-1)
            output_path: Optional path to save figure
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Plot histogram
        ax.hist(coverage_ratios, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        
        # Add mean line
        mean_coverage = np.mean(coverage_ratios)
        ax.axvline(mean_coverage, color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {mean_coverage:.3f}')
        
        # Labels
        ax.set_xlabel('Coverage Ratio', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            fig.savefig(output_path, bbox_inches='tight')
            logger.info(f"Saved coverage ratio plot to: {output_path}")
        
        return fig
    
    def create_comparison_grid(self, images: List[np.ndarray],
                              masks: List[np.ndarray],
                              titles: List[str],
                              output_path: Optional[str] = None,
                              grid_size: Optional[Tuple[int, int]] = None) -> plt.Figure:
        """
        Create a grid of image-mask comparisons.
        
        Args:
            images: List of images
            masks: List of masks
            titles: List of titles for each comparison
            output_path: Optional path to save figure
            grid_size: Optional grid size (rows, cols). Auto-computed if None.
            
        Returns:
            Matplotlib figure
        """
        n_samples = len(images)
        
        if grid_size is None:
            # Auto-compute grid size
            cols = min(4, n_samples)
            rows = (n_samples + cols - 1) // cols
        else:
            rows, cols = grid_size
        
        fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 6, rows * 3), dpi=self.dpi)
        
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_samples):
            row = i // cols
            col = i % cols
            
            # Original image
            ax_img = axes[row, col * 2]
            ax_img.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB) if len(images[i].shape) == 3 else images[i], cmap='gray')
            ax_img.set_title(f"{titles[i]} - Image", fontsize=10)
            ax_img.axis('off')
            
            # Mask overlay
            ax_mask = axes[row, col * 2 + 1]
            overlay = self._create_overlay(images[i], masks[i])
            ax_mask.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            ax_mask.set_title(f"{titles[i]} - Overlay", fontsize=10)
            ax_mask.axis('off')
        
        # Hide unused subplots
        for i in range(n_samples, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col * 2].axis('off')
            axes[row, col * 2 + 1].axis('off')
        
        plt.tight_layout()
        
        if output_path:
            fig.savefig(output_path, bbox_inches='tight')
            logger.info(f"Saved comparison grid to: {output_path}")
        
        return fig
    
    def _create_overlay(self, image: np.ndarray, mask: np.ndarray, 
                       alpha: float = 0.5) -> np.ndarray:
        """Create overlay of mask on image."""
        # Convert grayscale to RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Create colored mask
        colored_mask = np.zeros_like(image)
        colored_mask[mask > 0] = [0, 0, 255]  # Red for defects
        
        # Blend
        overlay = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)
        
        return overlay


class TrainingVisualizer:
    """
    Visualize model training metrics.
    
    Provides methods to plot:
    - Training/validation loss curves
    - Metric curves (IoU, Dice, etc.)
    - Learning rate schedules
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 100):
        """
        Initialize TrainingVisualizer.
        
        Args:
            figsize: Default figure size
            dpi: Figure DPI
        """
        self.figsize = figsize
        self.dpi = dpi
    
    def plot_training_curves(self, history: Dict,
                            output_path: Optional[str] = None,
                            title: str = "Training History") -> plt.Figure:
        """
        Plot training and validation curves.
        
        Args:
            history: Dictionary with 'train_loss', 'val_loss', etc.
            output_path: Optional path to save figure
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        # Determine number of subplots
        metrics = [k for k in history.keys() if not k.startswith('val_')]
        n_metrics = len(metrics)
        
        if n_metrics == 0:
            logger.warning("No metrics found in history")
            return None
        
        fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5), dpi=self.dpi)
        
        if n_metrics == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Plot train metric
            epochs = range(1, len(history[metric]) + 1)
            ax.plot(epochs, history[metric], 'b-', linewidth=2, label=f'Train {metric}')
            
            # Plot validation metric if available
            val_key = f'val_{metric}'
            if val_key in history:
                ax.plot(epochs, history[val_key], 'r-', linewidth=2, label=f'Val {metric}')
            
            # Labels
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
            ax.set_title(f'{metric.replace("_", " ").title()} vs Epoch', fontsize=13, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if output_path:
            fig.savefig(output_path, bbox_inches='tight')
            logger.info(f"Saved training curves to: {output_path}")
        
        return fig
    
    def plot_confusion_matrix(self, confusion_matrix: Dict,
                             output_path: Optional[str] = None,
                             title: str = "Confusion Matrix") -> plt.Figure:
        """
        Plot confusion matrix.
        
        Args:
            confusion_matrix: Dictionary with 'tp', 'tn', 'fp', 'fn'
            output_path: Optional path to save figure
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(8, 6), dpi=self.dpi)
        
        # Extract values
        tp = confusion_matrix.get('tp', 0)
        tn = confusion_matrix.get('tn', 0)
        fp = confusion_matrix.get('fp', 0)
        fn = confusion_matrix.get('fn', 0)
        
        # Create matrix
        cm = np.array([[tn, fp], [fn, tp]])
        
        # Plot heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'],
                   cbar_kws={'label': 'Count'})
        
        # Labels
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Actual', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Add metrics text
        accuracy = confusion_matrix.get('accuracy', 0)
        precision = confusion_matrix.get('precision', 0)
        recall = confusion_matrix.get('recall', 0)
        f1 = confusion_matrix.get('f1_score', 0)
        iou = confusion_matrix.get('iou', 0)
        
        metrics_text = (f"Accuracy: {accuracy:.4f}\n"
                       f"Precision: {precision:.4f}\n"
                       f"Recall: {recall:.4f}\n"
                       f"F1-Score: {f1:.4f}\n"
                       f"IoU: {iou:.4f}")
        
        ax.text(1.5, 0.5, metrics_text, transform=ax.transAxes,
               fontsize=11, verticalalignment='center',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if output_path:
            fig.savefig(output_path, bbox_inches='tight')
            logger.info(f"Saved confusion matrix to: {output_path}")
        
        return fig
    
    def plot_model_comparison(self, comparison_results: Dict,
                             metric: str = 'iou',
                             output_path: Optional[str] = None,
                             title: str = "Model Comparison") -> plt.Figure:
        """
        Plot model comparison bar chart.
        
        Args:
            comparison_results: Dictionary mapping model names to metrics
            metric: Metric to compare
            output_path: Optional path to save figure
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Extract model names and metric values
        models = []
        values = []
        
        for model_name, metrics in comparison_results.items():
            if model_name in ['ranking', 'best_model']:
                continue
            
            if metric in metrics:
                models.append(model_name)
                values.append(metrics[metric])
        
        # Sort by value
        sorted_pairs = sorted(zip(models, values), key=lambda x: x[1], reverse=True)
        models, values = zip(*sorted_pairs) if sorted_pairs else ([], [])
        
        # Plot bar chart
        colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
        bars = ax.barh(models, values, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, values)):
            ax.text(value, i, f' {value:.4f}', va='center', fontsize=10)
        
        # Labels
        ax.set_xlabel(metric.upper(), fontsize=12)
        ax.set_ylabel('Model', fontsize=12)
        ax.set_title(f'{title} - {metric.upper()}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if output_path:
            fig.savefig(output_path, bbox_inches='tight')
            logger.info(f"Saved model comparison to: {output_path}")
        
        return fig


class DatasetVisualizer:
    """
    Visualize dataset statistics.
    
    Provides methods to plot:
    - Class distribution
    - Data split visualization
    - Sample diversity
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 100):
        """
        Initialize DatasetVisualizer.
        
        Args:
            figsize: Default figure size
            dpi: Figure DPI
        """
        self.figsize = figsize
        self.dpi = dpi
    
    def plot_dataset_summary(self, summary: Dict,
                            output_path: Optional[str] = None,
                            title: str = "Dataset Summary") -> plt.Figure:
        """
        Plot dataset summary statistics.
        
        Args:
            summary: Dataset summary dictionary
            output_path: Optional path to save figure
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize, dpi=self.dpi)
        
        # 1. Images vs Masks
        ax1 = axes[0, 0]
        categories = ['Total\nImages', 'Total\nMasks', 'Matched\nPairs']
        counts = [
            summary.get('total_images', 0),
            summary.get('total_masks', 0),
            summary.get('matched_pairs', 0)
        ]
        ax1.bar(categories, counts, color=['skyblue', 'lightcoral', 'lightgreen'], alpha=0.8, edgecolor='black')
        ax1.set_ylabel('Count', fontsize=11)
        ax1.set_title('Dataset Composition', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 2. Defect presence
        ax2 = axes[0, 1]
        defect_stats = summary.get('defect_statistics', {})
        with_defects = defect_stats.get('images_with_defects', 0)
        without_defects = defect_stats.get('images_without_defects', 0)
        
        if with_defects + without_defects > 0:
            ax2.pie([with_defects, without_defects],
                   labels=['With Defects', 'Without Defects'],
                   autopct='%1.1f%%',
                   colors=['salmon', 'lightblue'],
                   startangle=90)
            ax2.set_title('Defect Presence', fontsize=12, fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14)
            ax2.axis('off')
        
        # 3. Defect counts
        ax3 = axes[1, 0]
        total_defects = defect_stats.get('total_defects', 0)
        mean_defects = defect_stats.get('mean_defects_per_image', 0)
        
        ax3.bar(['Total\nDefects', 'Mean per\nImage'], [total_defects, mean_defects],
               color=['coral', 'gold'], alpha=0.8, edgecolor='black')
        ax3.set_ylabel('Count', fontsize=11)
        ax3.set_title('Defect Statistics', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Coverage ratio
        ax4 = axes[1, 1]
        mean_coverage = defect_stats.get('mean_coverage_ratio', 0)
        std_coverage = defect_stats.get('std_coverage_ratio', 0)
        
        ax4.bar(['Mean Coverage'], [mean_coverage], yerr=[std_coverage],
               color='mediumseagreen', alpha=0.8, edgecolor='black', capsize=10)
        ax4.set_ylabel('Coverage Ratio', fontsize=11)
        ax4.set_title('Defect Coverage', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.set_ylim(0, min(1.0, mean_coverage + 2 * std_coverage + 0.1))
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if output_path:
            fig.savefig(output_path, bbox_inches='tight')
            logger.info(f"Saved dataset summary to: {output_path}")
        
        return fig


def close_all_figures():
    """Close all matplotlib figures."""
    plt.close('all')
    logger.debug("Closed all matplotlib figures")
