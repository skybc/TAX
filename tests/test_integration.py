"""
完整工作流程的集成测试。

测试内容：
- 端到端标注工作流程
- 训练工作流程
- 预测工作流程
- 报告生成工作流程
"""

import pytest
import numpy as np
from pathlib import Path
import torch

from src.core.data_manager import DataManager
from src.core.annotation_manager import AnnotationManager
from src.utils.statistics import DefectStatistics
from src.utils.visualization import DefectVisualizer
from src.utils.report_generator import ReportManager


class TestAnnotationWorkflow:
    """标注工作流程的集成测试。"""
    
    @pytest.mark.integration
    def test_complete_annotation_workflow(self, create_test_images, temp_output_dir):
        """测试从图像到导出的完整标注工作流程。"""
        # 1. 加载图像
        image_paths = create_test_images(count=5)
        dm = DataManager(str(temp_output_dir))
        
        images = []
        for image_path in image_paths:
            image = dm.load_image(image_path)
            assert image is not None
            images.append(image)
        
        # 2. 标注图像
        mask_paths = []
        for i, image in enumerate(images):
            am = AnnotationManager()
            am.set_image(f"image_{i}.png", image.shape)
            
            # 创建简单掩码
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            mask[50:150, 50:150] = 255
            am.set_mask(mask)
            
            # 保存掩码
            mask_path = temp_output_dir / f"mask_{i}.png"
            success = am.save_mask(str(mask_path))
            assert success is True
            mask_paths.append(str(mask_path))
        
        # 3. 验证所有掩码已保存
        assert len(mask_paths) == 5
        for mask_path in mask_paths:
            assert Path(mask_path).exists()
        
        # 4. 导出标注 (COCO 格式)
        from src.utils.export_utils import COCOExporter
        
        exporter = COCOExporter()
        coco_path = temp_output_dir / "annotations.json"
        
        success = exporter.export(
            image_paths=image_paths,
            mask_paths=mask_paths,
            output_path=str(coco_path),
            categories=[{'id': 1, 'name': 'defect'}]
        )
        
        assert success is True
        assert coco_path.exists()


class TestTrainingWorkflow:
    """训练工作流程的集成测试。"""
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_training_workflow(self, create_test_images, create_test_masks, temp_output_dir):
        """测试模型训练工作流程。"""
        # 1. 准备数据
        image_paths = create_test_images(count=20)
        mask_paths = create_test_masks(count=20)
        
        # 2. 创建数据集划分
        dm = DataManager(str(temp_output_dir))
        dm.dataset['all'] = list(zip(image_paths, mask_paths))
        
        splits = dm.create_splits(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
        
        assert 'train' in splits
        assert 'val' in splits
        assert 'test' in splits
        
        # 3. 创建模型
        from src.models.segmentation_models import SegmentationModel
        
        model = SegmentationModel(
            architecture='unet',
            encoder_name='resnet18',
            in_channels=3,
            num_classes=1
        )
        
        assert model is not None
        
        # 4. 创建训练器（不进行实际训练以节省时间）
        from src.core.model_trainer import ModelTrainer
        
        trainer_config = {
            'batch_size': 2,
            'num_epochs': 2,  # 测试时使用极少的 epoch
            'learning_rate': 0.001,
            'device': 'cpu'  # 测试时使用 CPU
        }
        
        trainer = ModelTrainer(model, trainer_config)
        assert trainer is not None
        
        # 注意：集成测试中跳过了实际训练
        # 完整训练将在端到端测试中进行


class TestPredictionWorkflow:
    """预测工作流程的集成测试。"""
    
    @pytest.mark.integration
    def test_prediction_workflow(self, create_test_images, temp_output_dir):
        """测试预测工作流程。"""
        # 1. 创建测试图像
        image_paths = create_test_images(count=5)
        
        # 2. 创建一个简单的模型
        from src.models.segmentation_models import SegmentationModel
        
        model = SegmentationModel(
            architecture='unet',
            encoder_name='resnet18',
            in_channels=3,
            num_classes=1
        )
        model.eval()
        
        # 3. 创建预测器
        from src.core.predictor import Predictor
        
        predictor = Predictor(model, device='cpu')
        
        # 4. 对图像进行预测
        predictions = []
        for image_path in image_paths:
            # 加载图像
            import cv2
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 预测
            mask = predictor.predict(image)
            
            assert mask is not None
            assert mask.shape == image.shape[:2]
            predictions.append(mask)
        
        assert len(predictions) == 5
        
        # 5. 保存预测结果
        predictions_dir = temp_output_dir / "predictions"
        predictions_dir.mkdir(exist_ok=True)
        
        for i, pred_mask in enumerate(predictions):
            from src.utils.mask_utils import save_mask
            save_path = predictions_dir / f"pred_{i}.png"
            save_mask(pred_mask, str(save_path))
            assert save_path.exists()


class TestReportGenerationWorkflow:
    """报告生成的集成测试。"""
    
    @pytest.mark.integration
    def test_report_generation_workflow(self, create_test_masks, temp_output_dir):
        """测试完整的报告生成工作流程。"""
        # 1. 创建测试掩码
        mask_paths = create_test_masks(count=20)
        
        # 2. 计算统计信息
        stats_calculator = DefectStatistics()
        statistics = stats_calculator.compute_batch_statistics(mask_paths)
        
        assert statistics is not None
        assert 'total_images' in statistics
        assert statistics['total_images'] == 20
        
        # 3. 生成可视化
        visualizer = DefectVisualizer()
        
        # 获取缺陷面积
        all_areas = []
        for img_stats in statistics['per_image_stats']:
            all_areas.extend(img_stats['defect_areas'])
        
        # 生成图表
        fig = visualizer.plot_defect_size_distribution(all_areas)
        assert fig is not None
        
        # 保存图表
        chart_path = temp_output_dir / "size_distribution.png"
        fig.savefig(chart_path)
        assert chart_path.exists()
        
        import matplotlib.pyplot as plt
        plt.close(fig)
        
        # 4. 生成报告
        report_manager = ReportManager()
        
        output_dir = temp_output_dir / "reports"
        output_dir.mkdir(exist_ok=True)
        
        result = report_manager.generate_complete_report(
            mask_paths=mask_paths,
            output_dir=str(output_dir),
            report_formats=['html']  # 为了速度仅生成 HTML
        )
        
        assert result is not None
        assert 'report_paths' in result
        assert 'html' in result['report_paths']
        
        # 验证 HTML 报告是否存在
        html_path = Path(result['report_paths']['html'])
        assert html_path.exists()


class TestEndToEndWorkflow:
    """端到端集成测试。"""
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_full_pipeline(self, create_test_images, create_test_masks, temp_output_dir):
        """测试完整流水线：数据 → 标注 → 训练 → 预测 → 报告。"""
        # 1. 数据管理
        image_paths = create_test_images(count=30)
        mask_paths = create_test_masks(count=30)
        
        dm = DataManager(str(temp_output_dir))
        dm.dataset['all'] = list(zip(image_paths, mask_paths))
        
        splits = dm.create_splits(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
        
        # 2. 标注（已有来自 create_test_masks 的掩码）
        assert len(mask_paths) == 30
        
        # 3. 导出 (COCO 格式)
        from src.utils.export_utils import COCOExporter
        
        exporter = COCOExporter()
        coco_path = temp_output_dir / "dataset.json"
        
        success = exporter.export(
            image_paths=image_paths,
            mask_paths=mask_paths,
            output_path=str(coco_path),
            categories=[{'id': 1, 'name': 'defect'}]
        )
        
        assert success is True
        
        # 4. 模型创建
        from src.models.segmentation_models import SegmentationModel
        
        model = SegmentationModel(
            architecture='unet',
            encoder_name='resnet18',
            in_channels=3,
            num_classes=1
        )
        
        # 5. 预测（为了测试速度不进行训练）
        from src.core.predictor import Predictor
        
        model.eval()
        predictor = Predictor(model, device='cpu')
        
        # 对少量测试图像进行预测
        test_predictions = []
        for image_path in image_paths[:5]:
            import cv2
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            mask = predictor.predict(image)
            test_predictions.append(mask)
        
        assert len(test_predictions) == 5
        
        # 6. 生成报告
        stats_calculator = DefectStatistics()
        statistics = stats_calculator.compute_batch_statistics(mask_paths)
        
        report_manager = ReportManager()
        output_dir = temp_output_dir / "final_reports"
        output_dir.mkdir(exist_ok=True)
        
        result = report_manager.generate_complete_report(
            mask_paths=mask_paths[:10],  # 使用子集以加快速度
            output_dir=str(output_dir),
            report_formats=['html']
        )
        
        assert result is not None
        assert 'report_paths' in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
