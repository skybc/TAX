"""
Integration tests for complete workflows.

Tests:
- End-to-end annotation workflow
- Training workflow
- Prediction workflow
- Report generation workflow
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
    """Integration tests for annotation workflow."""
    
    @pytest.mark.integration
    def test_complete_annotation_workflow(self, create_test_images, temp_output_dir):
        """Test complete annotation workflow from image to export."""
        # 1. Load images
        image_paths = create_test_images(count=5)
        dm = DataManager(str(temp_output_dir))
        
        images = []
        for image_path in image_paths:
            image = dm.load_image(image_path)
            assert image is not None
            images.append(image)
        
        # 2. Annotate images
        mask_paths = []
        for i, image in enumerate(images):
            am = AnnotationManager()
            am.set_image(f"image_{i}.png", image.shape)
            
            # Create simple mask
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            mask[50:150, 50:150] = 255
            am.set_mask(mask)
            
            # Save mask
            mask_path = temp_output_dir / f"mask_{i}.png"
            success = am.save_mask(str(mask_path))
            assert success is True
            mask_paths.append(str(mask_path))
        
        # 3. Verify all masks saved
        assert len(mask_paths) == 5
        for mask_path in mask_paths:
            assert Path(mask_path).exists()
        
        # 4. Export annotations (COCO format)
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
    """Integration tests for training workflow."""
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_training_workflow(self, create_test_images, create_test_masks, temp_output_dir):
        """Test model training workflow."""
        # 1. Prepare data
        image_paths = create_test_images(count=20)
        mask_paths = create_test_masks(count=20)
        
        # 2. Create dataset splits
        dm = DataManager(str(temp_output_dir))
        dm.dataset['all'] = list(zip(image_paths, mask_paths))
        
        splits = dm.create_splits(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
        
        assert 'train' in splits
        assert 'val' in splits
        assert 'test' in splits
        
        # 3. Create model
        from src.models.segmentation_models import SegmentationModel
        
        model = SegmentationModel(
            architecture='unet',
            encoder_name='resnet18',
            in_channels=3,
            num_classes=1
        )
        
        assert model is not None
        
        # 4. Create trainer (without actual training to save time)
        from src.core.model_trainer import ModelTrainer
        
        trainer_config = {
            'batch_size': 2,
            'num_epochs': 2,  # Very few epochs for test
            'learning_rate': 0.001,
            'device': 'cpu'  # Use CPU for test
        }
        
        trainer = ModelTrainer(model, trainer_config)
        assert trainer is not None
        
        # Note: Actual training is skipped in integration test
        # Full training would be tested in end-to-end tests


class TestPredictionWorkflow:
    """Integration tests for prediction workflow."""
    
    @pytest.mark.integration
    def test_prediction_workflow(self, create_test_images, temp_output_dir):
        """Test prediction workflow."""
        # 1. Create test images
        image_paths = create_test_images(count=5)
        
        # 2. Create a simple model
        from src.models.segmentation_models import SegmentationModel
        
        model = SegmentationModel(
            architecture='unet',
            encoder_name='resnet18',
            in_channels=3,
            num_classes=1
        )
        model.eval()
        
        # 3. Create predictor
        from src.core.predictor import Predictor
        
        predictor = Predictor(model, device='cpu')
        
        # 4. Predict on images
        predictions = []
        for image_path in image_paths:
            # Load image
            import cv2
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Predict
            mask = predictor.predict(image)
            
            assert mask is not None
            assert mask.shape == image.shape[:2]
            predictions.append(mask)
        
        assert len(predictions) == 5
        
        # 5. Save predictions
        predictions_dir = temp_output_dir / "predictions"
        predictions_dir.mkdir(exist_ok=True)
        
        for i, pred_mask in enumerate(predictions):
            from src.utils.mask_utils import save_mask
            save_path = predictions_dir / f"pred_{i}.png"
            save_mask(pred_mask, str(save_path))
            assert save_path.exists()


class TestReportGenerationWorkflow:
    """Integration tests for report generation."""
    
    @pytest.mark.integration
    def test_report_generation_workflow(self, create_test_masks, temp_output_dir):
        """Test complete report generation workflow."""
        # 1. Create test masks
        mask_paths = create_test_masks(count=20)
        
        # 2. Compute statistics
        stats_calculator = DefectStatistics()
        statistics = stats_calculator.compute_batch_statistics(mask_paths)
        
        assert statistics is not None
        assert 'total_images' in statistics
        assert statistics['total_images'] == 20
        
        # 3. Generate visualizations
        visualizer = DefectVisualizer()
        
        # Get defect areas
        all_areas = []
        for img_stats in statistics['per_image_stats']:
            all_areas.extend(img_stats['defect_areas'])
        
        # Generate chart
        fig = visualizer.plot_defect_size_distribution(all_areas)
        assert fig is not None
        
        # Save chart
        chart_path = temp_output_dir / "size_distribution.png"
        fig.savefig(chart_path)
        assert chart_path.exists()
        
        import matplotlib.pyplot as plt
        plt.close(fig)
        
        # 4. Generate report
        report_manager = ReportManager()
        
        output_dir = temp_output_dir / "reports"
        output_dir.mkdir(exist_ok=True)
        
        result = report_manager.generate_complete_report(
            mask_paths=mask_paths,
            output_dir=str(output_dir),
            report_formats=['html']  # Only HTML for speed
        )
        
        assert result is not None
        assert 'report_paths' in result
        assert 'html' in result['report_paths']
        
        # Verify HTML report exists
        html_path = Path(result['report_paths']['html'])
        assert html_path.exists()


class TestEndToEndWorkflow:
    """End-to-end integration tests."""
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_full_pipeline(self, create_test_images, create_test_masks, temp_output_dir):
        """Test full pipeline: data → annotation → training → prediction → report."""
        # 1. Data Management
        image_paths = create_test_images(count=30)
        mask_paths = create_test_masks(count=30)
        
        dm = DataManager(str(temp_output_dir))
        dm.dataset['all'] = list(zip(image_paths, mask_paths))
        
        splits = dm.create_splits(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
        
        # 2. Annotation (already have masks from create_test_masks)
        assert len(mask_paths) == 30
        
        # 3. Export (COCO format)
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
        
        # 4. Model Creation
        from src.models.segmentation_models import SegmentationModel
        
        model = SegmentationModel(
            architecture='unet',
            encoder_name='resnet18',
            in_channels=3,
            num_classes=1
        )
        
        # 5. Prediction (without training for test speed)
        from src.core.predictor import Predictor
        
        model.eval()
        predictor = Predictor(model, device='cpu')
        
        # Predict on a few test images
        test_predictions = []
        for image_path in image_paths[:5]:
            import cv2
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            mask = predictor.predict(image)
            test_predictions.append(mask)
        
        assert len(test_predictions) == 5
        
        # 6. Generate Report
        stats_calculator = DefectStatistics()
        statistics = stats_calculator.compute_batch_statistics(mask_paths)
        
        report_manager = ReportManager()
        output_dir = temp_output_dir / "final_reports"
        output_dir.mkdir(exist_ok=True)
        
        result = report_manager.generate_complete_report(
            mask_paths=mask_paths[:10],  # Use subset for speed
            output_dir=str(output_dir),
            report_formats=['html']
        )
        
        assert result is not None
        assert 'report_paths' in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
