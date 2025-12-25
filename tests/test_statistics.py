"""
statistics 模块的单元测试。

测试内容：
- DefectStatistics
- DatasetStatistics
- ModelPerformanceAnalyzer
- 统计信息的持久化
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
    """DefectStatistics 类的测试。"""
    
    @pytest.mark.unit
    def test_compute_mask_statistics_empty(self, empty_mask):
        """测试空掩码的统计计算。"""
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
        """测试单个缺陷的统计计算。"""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[25:75, 25:75] = 255  # 50x50 缺陷
        
        stats_calculator = DefectStatistics()
        stats = stats_calculator.compute_mask_statistics(mask)
        
        assert stats['num_defects'] == 1
        assert stats['total_area'] == 2500  # 50x50
        assert len(stats['defect_areas']) == 1
        assert stats['defect_areas'][0] == 2500
        
        # 检查覆盖率
        expected_coverage = 2500 / (100 * 100)
        assert np.isclose(stats['coverage_ratio'], expected_coverage)
    
    @pytest.mark.unit
    def test_compute_mask_statistics_multiple_defects(self, sample_mask_with_multiple_defects):
        """测试多个缺陷的统计计算。"""
        stats_calculator = DefectStatistics()
        stats = stats_calculator.compute_mask_statistics(sample_mask_with_multiple_defects)
        
        assert stats['num_defects'] == 5
        assert len(stats['defect_areas']) == 5
        assert len(stats['defect_centroids']) == 5
        assert len(stats['defect_bboxes']) == 5
        
        # 检查统计信息是否合理
        assert stats['largest_defect'] > 0
        assert stats['smallest_defect'] > 0
        assert stats['mean_defect_size'] > 0
        assert stats['largest_defect'] >= stats['smallest_defect']
    
    @pytest.mark.unit
    def test_compute_batch_statistics(self, create_test_masks):
        """测试批量统计计算。"""
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
        """测试空间分布计算。"""
        stats_calculator = DefectStatistics()
        heatmap = stats_calculator.compute_spatial_distribution(
            [sample_mask_with_multiple_defects],
            grid_size=20
        )
        
        assert heatmap.shape == (20, 20)
        assert heatmap.dtype == np.float32
        # 检查归一化
        assert np.min(heatmap) >= 0
        assert np.max(heatmap) <= 1.0


class TestDatasetStatistics:
    """DatasetStatistics 类的测试。"""
    
    @pytest.mark.unit
    def test_compute_dataset_summary(self, create_test_images, create_test_masks, temp_output_dir):
        """测试数据集摘要计算。"""
        image_paths = create_test_images(count=10)
        mask_paths = create_test_masks(count=10)
        
        # 创建匹配的目录结构
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
        """测试划分统计计算。"""
        mask_paths = create_test_masks(count=30)
        
        # 创建划分
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
    """ModelPerformanceAnalyzer 类的测试。"""
    
    @pytest.mark.unit
    def test_analyze_training_history(self, mock_training_history):
        """测试训练历史分析。"""
        analyzer = ModelPerformanceAnalyzer()
        analysis = analyzer.analyze_training_history(mock_training_history)
        
        assert 'best_epoch' in analysis
        assert 'best_val_loss' in analysis
        assert 'final_metrics' in analysis
        assert 'is_overfitting' in analysis
        
        # 检查最佳 epoch 是否合理
        assert 0 <= analysis['best_epoch'] < len(mock_training_history['loss'])
    
    @pytest.mark.unit
    def test_compute_confusion_matrix_perfect(self):
        """测试完美预测下的混淆矩阵。"""
        predictions = [np.ones((100, 100), dtype=np.uint8) * 255]
        ground_truths = [np.ones((100, 100), dtype=np.uint8) * 255]
        
        analyzer = ModelPerformanceAnalyzer()
        metrics = analyzer.compute_confusion_matrix(predictions, ground_truths)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'iou' in metrics
        
        # 完美预测的指标应该接近 1.0
        assert np.isclose(metrics['accuracy'], 1.0, atol=0.01)
        assert np.isclose(metrics['iou'], 1.0, atol=0.01)
    
    @pytest.mark.unit
    def test_compute_confusion_matrix_no_overlap(self):
        """测试无重叠下的混淆矩阵。"""
        predictions = [np.zeros((100, 100), dtype=np.uint8)]
        ground_truths = [np.ones((100, 100), dtype=np.uint8) * 255]
        
        analyzer = ModelPerformanceAnalyzer()
        metrics = analyzer.compute_confusion_matrix(predictions, ground_truths)
        
        # 无重叠应导致 IoU = 0
        assert np.isclose(metrics['iou'], 0.0, atol=0.01)
        assert np.isclose(metrics['recall'], 0.0, atol=0.01)
    
    @pytest.mark.unit
    def test_compare_models(self, sample_mask_batch):
        """测试模型比较。"""
        ground_truths = sample_mask_batch
        
        # 创建两组预测结果
        model_results = {
            'model_a': sample_mask_batch,  # 完美预测
            'model_b': [np.zeros_like(m) for m in sample_mask_batch]  # 糟糕预测
        }
        
        analyzer = ModelPerformanceAnalyzer()
        comparison = analyzer.compare_models(model_results, ground_truths)
        
        assert 'ranking' in comparison
        assert 'best_model' in comparison
        assert 'metrics_by_model' in comparison
        
        # 模型 A 的排名应该高于模型 B
        assert comparison['best_model'] == 'model_a'


class TestStatisticsPersistence:
    """统计信息保存/加载操作测试。"""
    
    @pytest.mark.unit
    def test_save_load_statistics(self, mock_statistics, temp_output_dir):
        """测试保存和加载统计信息。"""
        stats_path = temp_output_dir / "stats.json"
        
        # 保存统计信息
        save_statistics(mock_statistics, str(stats_path))
        assert stats_path.exists()
        
        # 加载统计信息
        loaded_stats = load_statistics(str(stats_path))
        
        assert loaded_stats is not None
        assert loaded_stats['total_images'] == mock_statistics['total_images']
        assert loaded_stats['total_defects'] == mock_statistics['total_defects']
    
    @pytest.mark.unit
    def test_load_nonexistent_statistics(self):
        """测试加载不存在的统计信息文件。"""
        loaded_stats = load_statistics("nonexistent_stats.json")
        assert loaded_stats is None
    
    @pytest.mark.unit
    def test_save_statistics_with_numpy(self, temp_output_dir):
        """测试保存包含 numpy 类型的统计信息。"""
        stats = {
            'count': np.int64(100),
            'mean': np.float64(3.14),
            'array': np.array([1, 2, 3])
        }
        
        stats_path = temp_output_dir / "numpy_stats.json"
        
        # 应该能够优雅地处理 numpy 类型
        save_statistics(stats, str(stats_path))
        assert stats_path.exists()
        
        loaded_stats = load_statistics(str(stats_path))
        assert loaded_stats is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
