"""
性能基准测试。

测试内容：
- 数据加载速度
- 模型推理速度
- 统计计算速度
- 报告生成速度
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
    """数据加载性能测试。"""
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_image_loading_speed(self, create_test_images, temp_dir, benchmark):
        """基准测试图像加载速度。"""
        image_paths = create_test_images(count=100)
        dm = DataManager(str(temp_dir))
        
        def load_images():
            for image_path in image_paths[:10]:  # 加载 10 张图像
                dm.load_image(image_path)
        
        # 运行基准测试
        if hasattr(benchmark, '__call__'):
            result = benchmark(load_images)
        else:
            # 如果未安装 pytest-benchmark，则使用备选方案
            start = time.time()
            load_images()
            duration = time.time() - start
            print(f"在 {duration:.3f}s 内加载了 10 张图像（每张图像 {duration/10:.3f}s）")
    
    @pytest.mark.performance
    def test_image_loading_with_cache(self, create_test_images, temp_dir):
        """测试带缓存的图像加载。"""
        image_paths = create_test_images(count=10)
        dm = DataManager(str(temp_dir), cache_size_mb=100)
        
        # 第一次加载（冷缓存）
        start = time.time()
        for image_path in image_paths:
            dm.load_image(image_path)
        cold_time = time.time() - start
        
        # 第二次加载（热缓存）
        start = time.time()
        for image_path in image_paths:
            dm.load_image(image_path)
        warm_time = time.time() - start
        
        print(f"冷缓存: {cold_time:.3f}s, 热缓存: {warm_time:.3f}s")
        # 热缓存应该更快
        assert warm_time < cold_time


class TestStatisticsPerformance:
    """统计计算性能测试。"""
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_batch_statistics_speed(self, create_test_masks):
        """基准测试批量统计计算速度。"""
        mask_paths = create_test_masks(count=100)
        stats_calculator = DefectStatistics()
        
        start = time.time()
        statistics = stats_calculator.compute_batch_statistics(mask_paths)
        duration = time.time() - start
        
        assert statistics is not None
        print(f"在 {duration:.3f}s 内计算了 100 个掩码的统计信息（每个掩码 {duration/100*1000:.1f}ms）")
        
        # 应该以合理的速度处理（每个掩码 < 1s）
        assert duration < 100
    
    @pytest.mark.performance
    def test_single_mask_statistics_speed(self, sample_mask_with_multiple_defects):
        """基准测试单个掩码统计。"""
        stats_calculator = DefectStatistics()
        
        # 预热
        stats_calculator.compute_mask_statistics(sample_mask_with_multiple_defects)
        
        # 基准测试
        start = time.time()
        for _ in range(100):
            stats_calculator.compute_mask_statistics(sample_mask_with_multiple_defects)
        duration = time.time() - start
        
        print(f"单个掩码统计：每个掩码 {duration/100*1000:.1f}ms")
        # 应该非常快（每个掩码 < 50ms）
        assert duration < 5.0


class TestVisualizationPerformance:
    """可视化性能测试。"""
    
    @pytest.mark.performance
    def test_chart_generation_speed(self, mock_statistics):
        """基准测试图表生成。"""
        visualizer = DefectVisualizer()
        
        # 提取数据
        all_areas = []
        for img_stats in mock_statistics['per_image_stats']:
            all_areas.append(img_stats['total_area'])
        
        # 基准测试
        start = time.time()
        fig = visualizer.plot_defect_size_distribution(all_areas)
        duration = time.time() - start
        
        import matplotlib.pyplot as plt
        plt.close(fig)
        
        print(f"图表生成：{duration*1000:.1f}ms")
        # 应该很快 (< 2s)
        assert duration < 2.0
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_multiple_charts_generation(self, mock_statistics):
        """基准测试生成多个图表。"""
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
        
        print(f"在 {duration:.3f}s 内生成了 3 个图表（每个图表 {duration/3:.3f}s）")
        assert duration < 10.0


class TestModelInferencePerformance:
    """模型推理性能测试。"""
    
    @pytest.mark.performance
    def test_single_image_inference_cpu(self):
        """基准测试 CPU 上的单张图像推理。"""
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
        
        # 预热
        with torch.no_grad():
            _ = model(x)
        
        # 基准测试
        start = time.time()
        with torch.no_grad():
            for _ in range(10):
                _ = model(x)
        duration = time.time() - start
        
        print(f"CPU 推理：每张图像 {duration/10*1000:.1f}ms (256x256)")
        # 应该完成（由于 CPU 差异，没有严格的时间限制）
        assert duration < 30.0  # 10 张图像在 30s 内
    
    @pytest.mark.performance
    @pytest.mark.requires_gpu
    def test_single_image_inference_gpu(self):
        """基准测试 GPU 上的单张图像推理。"""
        import torch
        
        if not torch.cuda.is_available():
            pytest.skip("CUDA 不可用")
        
        model = SegmentationModel(
            architecture='unet',
            encoder_name='resnet18',
            in_channels=3,
            num_classes=1
        )
        model.eval()
        model.to('cuda')
        
        x = torch.randn(1, 3, 256, 256).cuda()
        
        # 预热
        with torch.no_grad():
            _ = model(x)
        torch.cuda.synchronize()
        
        # 基准测试
        start = time.time()
        with torch.no_grad():
            for _ in range(100):
                _ = model(x)
        torch.cuda.synchronize()
        duration = time.time() - start
        
        print(f"GPU 推理：每张图像 {duration/100*1000:.1f}ms (256x256)")
        # GPU 应该快得多
        assert duration < 5.0  # 100 张图像在 5s 内
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_batch_inference_throughput(self):
        """基准测试批量推理吞吐量。"""
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
            
            # 预热
            with torch.no_grad():
                _ = model(x)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            
            # 基准测试
            start = time.time()
            with torch.no_grad():
                for _ in range(20):
                    _ = model(x)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            
            duration = time.time() - start
            throughput = (20 * batch_size) / duration
            
            print(f"Batch size {batch_size}: 在 {device} 上为 {throughput:.1f} images/sec")


class TestReportGenerationPerformance:
    """报告生成性能测试。"""
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_html_report_generation_speed(self, create_test_masks, temp_output_dir):
        """基准测试 HTML 报告生成。"""
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
        print(f"50 个掩码的 HTML 报告生成：{duration:.3f}s")
        # 应该相当快（50 张图像 < 30s）
        assert duration < 30.0


class TestMemoryUsage:
    """内存使用测试。"""
    
    @pytest.mark.performance
    def test_image_cache_memory_limit(self, create_test_images, temp_dir):
        """测试图像缓存是否遵守内存限制。"""
        image_paths = create_test_images(count=100)
        
        # 创建具有小缓存的 DataManager
        dm = DataManager(str(temp_dir), cache_size_mb=10)
        
        # 加载多张图像
        for image_path in image_paths:
            dm.load_image(image_path)
        
        # 检查缓存大小
        cache_size_mb = dm.get_cache_size()
        print(f"加载 100 张图像后的缓存大小：{cache_size_mb:.1f}MB")
        
        # 不应显著超过限制（允许一些开销）
        assert cache_size_mb < 15  # 10MB 限制 + 5MB 开销
    
    @pytest.mark.performance
    @pytest.mark.requires_gpu
    def test_gpu_memory_usage(self):
        """测试推理期间的 GPU 内存使用情况。"""
        import torch
        
        if not torch.cuda.is_available():
            pytest.skip("CUDA 不可用")
        
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
        
        print(f"使用的 GPU 内存：{memory_used_mb:.1f}MB")
        
        # 应该适合合理的 GPU 内存 (< 2GB)
        assert memory_used_mb < 2048


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "performance"])
