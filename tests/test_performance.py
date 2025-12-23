"""
Performance benchmarks and tests.

Tests:
- Data loading speed
- Model inference speed
- Statistics computation speed
- Report generation speed
"""

import pytest
import numpy as np
import time
from pathlib import Path

from src.core.data_manager import DataManager
from src.utils.statistics import DefectStatistics
from src.utils.visualization import DefectVisualizer
from src.models.segmentation_models import SegmentationModel


class TestDataLoadingPerformance:
    """Performance tests for data loading."""
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_image_loading_speed(self, create_test_images, temp_dir, benchmark):
        """Benchmark image loading speed."""
        image_paths = create_test_images(count=100)
        dm = DataManager(str(temp_dir))
        
        def load_images():
            for image_path in image_paths[:10]:  # Load 10 images
                dm.load_image(image_path)
        
        # Run benchmark
        if hasattr(benchmark, '__call__'):
            result = benchmark(load_images)
        else:
            # Fallback if pytest-benchmark not installed
            start = time.time()
            load_images()
            duration = time.time() - start
            print(f"Loaded 10 images in {duration:.3f}s ({duration/10:.3f}s per image)")
    
    @pytest.mark.performance
    def test_image_loading_with_cache(self, create_test_images, temp_dir):
        """Test image loading with cache."""
        image_paths = create_test_images(count=10)
        dm = DataManager(str(temp_dir), cache_size_mb=100)
        
        # First load (cold cache)
        start = time.time()
        for image_path in image_paths:
            dm.load_image(image_path)
        cold_time = time.time() - start
        
        # Second load (warm cache)
        start = time.time()
        for image_path in image_paths:
            dm.load_image(image_path)
        warm_time = time.time() - start
        
        print(f"Cold cache: {cold_time:.3f}s, Warm cache: {warm_time:.3f}s")
        # Warm cache should be faster
        assert warm_time < cold_time


class TestStatisticsPerformance:
    """Performance tests for statistics computation."""
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_batch_statistics_speed(self, create_test_masks):
        """Benchmark batch statistics computation."""
        mask_paths = create_test_masks(count=100)
        stats_calculator = DefectStatistics()
        
        start = time.time()
        statistics = stats_calculator.compute_batch_statistics(mask_paths)
        duration = time.time() - start
        
        assert statistics is not None
        print(f"Computed statistics for 100 masks in {duration:.3f}s ({duration/100*1000:.1f}ms per mask)")
        
        # Should process reasonable speed (< 1s per mask)
        assert duration < 100
    
    @pytest.mark.performance
    def test_single_mask_statistics_speed(self, sample_mask_with_multiple_defects):
        """Benchmark single mask statistics."""
        stats_calculator = DefectStatistics()
        
        # Warm up
        stats_calculator.compute_mask_statistics(sample_mask_with_multiple_defects)
        
        # Benchmark
        start = time.time()
        for _ in range(100):
            stats_calculator.compute_mask_statistics(sample_mask_with_multiple_defects)
        duration = time.time() - start
        
        print(f"Single mask statistics: {duration/100*1000:.1f}ms per mask")
        # Should be very fast (< 50ms per mask)
        assert duration < 5.0


class TestVisualizationPerformance:
    """Performance tests for visualization."""
    
    @pytest.mark.performance
    def test_chart_generation_speed(self, mock_statistics):
        """Benchmark chart generation."""
        visualizer = DefectVisualizer()
        
        # Extract data
        all_areas = []
        for img_stats in mock_statistics['per_image_stats']:
            all_areas.append(img_stats['total_area'])
        
        # Benchmark
        start = time.time()
        fig = visualizer.plot_defect_size_distribution(all_areas)
        duration = time.time() - start
        
        import matplotlib.pyplot as plt
        plt.close(fig)
        
        print(f"Chart generation: {duration*1000:.1f}ms")
        # Should be fast (< 2s)
        assert duration < 2.0
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_multiple_charts_generation(self, mock_statistics):
        """Benchmark generating multiple charts."""
        visualizer = DefectVisualizer()
        
        all_areas = []
        defect_counts = []
        coverage_ratios = []
        
        for img_stats in mock_statistics['per_image_stats']:
            all_areas.append(img_stats['total_area'])
            defect_counts.append(img_stats['num_defects'])
            coverage_ratios.append(img_stats['coverage_ratio'])
        
        start = time.time()
        
        fig1 = visualizer.plot_defect_size_distribution(all_areas)
        fig2 = visualizer.plot_defect_count_per_image(defect_counts)
        fig3 = visualizer.plot_coverage_ratio_distribution(coverage_ratios)
        
        duration = time.time() - start
        
        import matplotlib.pyplot as plt
        plt.close(fig1)
        plt.close(fig2)
        plt.close(fig3)
        
        print(f"Generated 3 charts in {duration:.3f}s ({duration/3:.3f}s per chart)")
        assert duration < 10.0


class TestModelInferencePerformance:
    """Performance tests for model inference."""
    
    @pytest.mark.performance
    def test_single_image_inference_cpu(self):
        """Benchmark single image inference on CPU."""
        import torch
        
        model = SegmentationModel(
            architecture='unet',
            encoder_name='resnet18',
            in_channels=3,
            num_classes=1
        )
        model.eval()
        model.to('cpu')
        
        x = torch.randn(1, 3, 256, 256)
        
        # Warm up
        with torch.no_grad():
            _ = model(x)
        
        # Benchmark
        start = time.time()
        with torch.no_grad():
            for _ in range(10):
                _ = model(x)
        duration = time.time() - start
        
        print(f"CPU inference: {duration/10*1000:.1f}ms per image (256x256)")
        # Should complete (no strict time limit due to CPU variance)
        assert duration < 30.0  # 10 images in < 30s
    
    @pytest.mark.performance
    @pytest.mark.requires_gpu
    def test_single_image_inference_gpu(self):
        """Benchmark single image inference on GPU."""
        import torch
        
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        model = SegmentationModel(
            architecture='unet',
            encoder_name='resnet18',
            in_channels=3,
            num_classes=1
        )
        model.eval()
        model.to('cuda')
        
        x = torch.randn(1, 3, 256, 256).cuda()
        
        # Warm up
        with torch.no_grad():
            _ = model(x)
        torch.cuda.synchronize()
        
        # Benchmark
        start = time.time()
        with torch.no_grad():
            for _ in range(100):
                _ = model(x)
        torch.cuda.synchronize()
        duration = time.time() - start
        
        print(f"GPU inference: {duration/100*1000:.1f}ms per image (256x256)")
        # GPU should be much faster
        assert duration < 5.0  # 100 images in < 5s
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_batch_inference_throughput(self):
        """Benchmark batch inference throughput."""
        import torch
        
        model = SegmentationModel(
            architecture='unet',
            encoder_name='resnet18',
            in_channels=3,
            num_classes=1
        )
        model.eval()
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        
        batch_sizes = [1, 2, 4, 8]
        
        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 3, 256, 256).to(device)
            
            # Warm up
            with torch.no_grad():
                _ = model(x)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            
            # Benchmark
            start = time.time()
            with torch.no_grad():
                for _ in range(20):
                    _ = model(x)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            
            duration = time.time() - start
            throughput = (20 * batch_size) / duration
            
            print(f"Batch size {batch_size}: {throughput:.1f} images/sec on {device}")


class TestReportGenerationPerformance:
    """Performance tests for report generation."""
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_html_report_generation_speed(self, create_test_masks, temp_output_dir):
        """Benchmark HTML report generation."""
        mask_paths = create_test_masks(count=50)
        
        from src.utils.report_generator import ReportManager
        
        report_manager = ReportManager()
        output_dir = temp_output_dir / "performance_reports"
        output_dir.mkdir(exist_ok=True)
        
        start = time.time()
        result = report_manager.generate_complete_report(
            mask_paths=mask_paths,
            output_dir=str(output_dir),
            report_formats=['html']
        )
        duration = time.time() - start
        
        assert result is not None
        print(f"HTML report generation for 50 masks: {duration:.3f}s")
        # Should be reasonably fast (< 30s for 50 images)
        assert duration < 30.0


class TestMemoryUsage:
    """Memory usage tests."""
    
    @pytest.mark.performance
    def test_image_cache_memory_limit(self, create_test_images, temp_dir):
        """Test that image cache respects memory limit."""
        image_paths = create_test_images(count=100)
        
        # Create DataManager with small cache
        dm = DataManager(str(temp_dir), cache_size_mb=10)
        
        # Load many images
        for image_path in image_paths:
            dm.load_image(image_path)
        
        # Check cache size
        cache_size_mb = dm.get_cache_size()
        print(f"Cache size after loading 100 images: {cache_size_mb:.1f}MB")
        
        # Should not exceed limit significantly (allow some overhead)
        assert cache_size_mb < 15  # 10MB limit + 5MB overhead
    
    @pytest.mark.performance
    @pytest.mark.requires_gpu
    def test_gpu_memory_usage(self):
        """Test GPU memory usage during inference."""
        import torch
        
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
        
        model = SegmentationModel(
            architecture='unet',
            encoder_name='resnet34',
            in_channels=3,
            num_classes=1
        )
        model.eval()
        model.to('cuda')
        
        x = torch.randn(4, 3, 512, 512).cuda()
        
        with torch.no_grad():
            _ = model(x)
        
        peak_memory = torch.cuda.max_memory_allocated()
        memory_used_mb = (peak_memory - initial_memory) / 1024 / 1024
        
        print(f"GPU memory used: {memory_used_mb:.1f}MB")
        
        # Should fit in reasonable GPU memory (< 2GB)
        assert memory_used_mb < 2048


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "performance"])
