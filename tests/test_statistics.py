"""
Unit tests for statistics module.

Tests:
- DefectStatistics
- DatasetStatistics
- ModelPerformanceAnalyzer
- Statistics persistence
"""

import pytest
import numpy as np
from pathlib import Path

from src.utils.statistics import (
    DefectStatistics,
    DatasetStatistics,
    ModelPerformanceAnalyzer,
    save_statistics,
    load_statistics
)


class TestDefectStatistics:
    """Tests for DefectStatistics class."""
    
    @pytest.mark.unit
    def test_compute_mask_statistics_empty(self, empty_mask):
        """Test statistics computation for empty mask."""
        stats_calculator = DefectStatistics()
        stats = stats_calculator.compute_mask_statistics(empty_mask, "test_image.png")
        
        assert stats['image_name'] == "test_image.png"
        assert stats['num_defects'] == 0
        assert stats['total_area'] == 0
        assert len(stats['defect_areas']) == 0
        assert len(stats['defect_centroids']) == 0
        assert stats['coverage_ratio'] == 0.0
    
    @pytest.mark.unit
    def test_compute_mask_statistics_single_defect(self):
        """Test statistics computation for single defect."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[25:75, 25:75] = 255  # 50x50 defect
        
        stats_calculator = DefectStatistics()
        stats = stats_calculator.compute_mask_statistics(mask)
        
        assert stats['num_defects'] == 1
        assert stats['total_area'] == 2500  # 50x50
        assert len(stats['defect_areas']) == 1
        assert stats['defect_areas'][0] == 2500
        
        # Check coverage ratio
        expected_coverage = 2500 / (100 * 100)
        assert np.isclose(stats['coverage_ratio'], expected_coverage)
    
    @pytest.mark.unit
    def test_compute_mask_statistics_multiple_defects(self, sample_mask_with_multiple_defects):
        """Test statistics computation for multiple defects."""
        stats_calculator = DefectStatistics()
        stats = stats_calculator.compute_mask_statistics(sample_mask_with_multiple_defects)
        
        assert stats['num_defects'] == 5
        assert len(stats['defect_areas']) == 5
        assert len(stats['defect_centroids']) == 5
        assert len(stats['defect_bboxes']) == 5
        
        # Check that statistics are reasonable
        assert stats['largest_defect'] > 0
        assert stats['smallest_defect'] > 0
        assert stats['mean_defect_size'] > 0
        assert stats['largest_defect'] >= stats['smallest_defect']
    
    @pytest.mark.unit
    def test_compute_batch_statistics(self, create_test_masks):
        """Test batch statistics computation."""
        mask_paths = create_test_masks(count=10)
        
        stats_calculator = DefectStatistics()
        batch_stats = stats_calculator.compute_batch_statistics(mask_paths)
        
        assert batch_stats['total_images'] == 10
        assert batch_stats['images_with_defects'] <= 10
        assert 'total_defects' in batch_stats
        assert 'mean_defects_per_image' in batch_stats
        assert 'per_image_stats' in batch_stats
        assert len(batch_stats['per_image_stats']) == 10
    
    @pytest.mark.unit
    def test_compute_spatial_distribution(self, sample_mask_with_multiple_defects):
        """Test spatial distribution computation."""
        stats_calculator = DefectStatistics()
        heatmap = stats_calculator.compute_spatial_distribution(
            [sample_mask_with_multiple_defects],
            grid_size=20
        )
        
        assert heatmap.shape == (20, 20)
        assert heatmap.dtype == np.float32
        # Check normalization
        assert np.min(heatmap) >= 0
        assert np.max(heatmap) <= 1.0


class TestDatasetStatistics:
    """Tests for DatasetStatistics class."""
    
    @pytest.mark.unit
    def test_compute_dataset_summary(self, create_test_images, create_test_masks, temp_output_dir):
        """Test dataset summary computation."""
        image_paths = create_test_images(count=10)
        mask_paths = create_test_masks(count=10)
        
        # Create matching directory structure
        images_dir = Path(image_paths[0]).parent
        masks_dir = Path(mask_paths[0]).parent
        
        stats_calculator = DatasetStatistics()
        summary = stats_calculator.compute_dataset_summary(
            str(images_dir),
            str(masks_dir)
        )
        
        assert 'total_images' in summary
        assert 'matched_pairs' in summary
        assert 'defect_statistics' in summary
        assert summary['matched_pairs'] == 10
    
    @pytest.mark.unit
    def test_compute_split_statistics(self, create_test_masks, temp_output_dir):
        """Test split statistics computation."""
        mask_paths = create_test_masks(count=30)
        
        # Create splits
        train_split = mask_paths[:20]
        val_split = mask_paths[20:25]
        test_split = mask_paths[25:]
        
        stats_calculator = DatasetStatistics()
        split_stats = stats_calculator.compute_split_statistics({
            'train': train_split,
            'val': val_split,
            'test': test_split
        })
        
        assert 'train' in split_stats
        assert 'val' in split_stats
        assert 'test' in split_stats
        
        assert split_stats['train']['total_images'] == 20
        assert split_stats['val']['total_images'] == 5
        assert split_stats['test']['total_images'] == 5


class TestModelPerformanceAnalyzer:
    """Tests for ModelPerformanceAnalyzer class."""
    
    @pytest.mark.unit
    def test_analyze_training_history(self, mock_training_history):
        """Test training history analysis."""
        analyzer = ModelPerformanceAnalyzer()
        analysis = analyzer.analyze_training_history(mock_training_history)
        
        assert 'best_epoch' in analysis
        assert 'best_val_loss' in analysis
        assert 'final_metrics' in analysis
        assert 'is_overfitting' in analysis
        
        # Check that best epoch is reasonable
        assert 0 <= analysis['best_epoch'] < len(mock_training_history['loss'])
    
    @pytest.mark.unit
    def test_compute_confusion_matrix_perfect(self):
        """Test confusion matrix with perfect predictions."""
        predictions = [np.ones((100, 100), dtype=np.uint8) * 255]
        ground_truths = [np.ones((100, 100), dtype=np.uint8) * 255]
        
        analyzer = ModelPerformanceAnalyzer()
        metrics = analyzer.compute_confusion_matrix(predictions, ground_truths)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'iou' in metrics
        
        # Perfect predictions should have metrics close to 1.0
        assert np.isclose(metrics['accuracy'], 1.0, atol=0.01)
        assert np.isclose(metrics['iou'], 1.0, atol=0.01)
    
    @pytest.mark.unit
    def test_compute_confusion_matrix_no_overlap(self):
        """Test confusion matrix with no overlap."""
        predictions = [np.zeros((100, 100), dtype=np.uint8)]
        ground_truths = [np.ones((100, 100), dtype=np.uint8) * 255]
        
        analyzer = ModelPerformanceAnalyzer()
        metrics = analyzer.compute_confusion_matrix(predictions, ground_truths)
        
        # No overlap should result in IoU = 0
        assert np.isclose(metrics['iou'], 0.0, atol=0.01)
        assert np.isclose(metrics['recall'], 0.0, atol=0.01)
    
    @pytest.mark.unit
    def test_compare_models(self, sample_mask_batch):
        """Test model comparison."""
        ground_truths = sample_mask_batch
        
        # Create two sets of predictions
        model_results = {
            'model_a': sample_mask_batch,  # Perfect predictions
            'model_b': [np.zeros_like(m) for m in sample_mask_batch]  # Bad predictions
        }
        
        analyzer = ModelPerformanceAnalyzer()
        comparison = analyzer.compare_models(model_results, ground_truths)
        
        assert 'ranking' in comparison
        assert 'best_model' in comparison
        assert 'metrics_by_model' in comparison
        
        # Model A should rank higher than Model B
        assert comparison['best_model'] == 'model_a'


class TestStatisticsPersistence:
    """Tests for statistics save/load operations."""
    
    @pytest.mark.unit
    def test_save_load_statistics(self, mock_statistics, temp_output_dir):
        """Test saving and loading statistics."""
        stats_path = temp_output_dir / "stats.json"
        
        # Save statistics
        save_statistics(mock_statistics, str(stats_path))
        assert stats_path.exists()
        
        # Load statistics
        loaded_stats = load_statistics(str(stats_path))
        
        assert loaded_stats is not None
        assert loaded_stats['total_images'] == mock_statistics['total_images']
        assert loaded_stats['total_defects'] == mock_statistics['total_defects']
    
    @pytest.mark.unit
    def test_load_nonexistent_statistics(self):
        """Test loading non-existent statistics file."""
        loaded_stats = load_statistics("nonexistent_stats.json")
        assert loaded_stats is None
    
    @pytest.mark.unit
    def test_save_statistics_with_numpy(self, temp_output_dir):
        """Test saving statistics with numpy types."""
        stats = {
            'count': np.int64(100),
            'mean': np.float64(3.14),
            'array': np.array([1, 2, 3])
        }
        
        stats_path = temp_output_dir / "numpy_stats.json"
        
        # Should handle numpy types gracefully
        save_statistics(stats, str(stats_path))
        assert stats_path.exists()
        
        loaded_stats = load_statistics(str(stats_path))
        assert loaded_stats is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
