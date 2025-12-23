"""
Unit tests for model definitions.

Tests:
- Model architecture creation
- Forward pass
- Output shapes
"""

import pytest
import torch
import numpy as np

from src.models.segmentation_models import SegmentationModel


class TestSegmentationModelCreation:
    """Tests for segmentation model creation."""
    
    @pytest.mark.unit
    def test_create_unet(self):
        """Test U-Net model creation."""
        model = SegmentationModel(
            architecture='unet',
            encoder_name='resnet34',
            in_channels=3,
            num_classes=1
        )
        
        assert model is not None
        assert hasattr(model, 'forward')
    
    @pytest.mark.unit
    def test_create_deeplabv3plus(self):
        """Test DeepLabV3+ model creation."""
        model = SegmentationModel(
            architecture='deeplabv3plus',
            encoder_name='resnet50',
            in_channels=3,
            num_classes=1
        )
        
        assert model is not None
        assert hasattr(model, 'forward')
    
    @pytest.mark.unit
    def test_create_fpn(self):
        """Test FPN model creation."""
        model = SegmentationModel(
            architecture='fpn',
            encoder_name='resnet18',
            in_channels=3,
            num_classes=1
        )
        
        assert model is not None
        assert hasattr(model, 'forward')
    
    @pytest.mark.unit
    def test_invalid_architecture(self):
        """Test error handling for invalid architecture."""
        with pytest.raises((ValueError, KeyError)):
            SegmentationModel(
                architecture='invalid_arch',
                encoder_name='resnet34',
                in_channels=3,
                num_classes=1
            )


class TestModelForwardPass:
    """Tests for model forward pass."""
    
    @pytest.mark.unit
    def test_unet_forward(self):
        """Test U-Net forward pass."""
        model = SegmentationModel(
            architecture='unet',
            encoder_name='resnet18',  # Smaller for faster test
            in_channels=3,
            num_classes=1
        )
        
        # Create dummy input
        x = torch.randn(2, 3, 256, 256)  # Batch=2, C=3, H=256, W=256
        
        # Forward pass
        output = model(x)
        
        # Check output shape
        assert output.shape == (2, 1, 256, 256)
    
    @pytest.mark.unit
    def test_model_training_mode(self):
        """Test model in training mode."""
        model = SegmentationModel(
            architecture='unet',
            encoder_name='resnet18',
            in_channels=3,
            num_classes=1
        )
        
        model.train()
        assert model.training is True
        
        x = torch.randn(1, 3, 128, 128)
        output = model(x)
        
        assert output.requires_grad is True
    
    @pytest.mark.unit
    def test_model_eval_mode(self):
        """Test model in evaluation mode."""
        model = SegmentationModel(
            architecture='unet',
            encoder_name='resnet18',
            in_channels=3,
            num_classes=1
        )
        
        model.eval()
        assert model.training is False
        
        with torch.no_grad():
            x = torch.randn(1, 3, 128, 128)
            output = model(x)
        
        assert output.requires_grad is False


class TestModelDeviceHandling:
    """Tests for model device management."""
    
    @pytest.mark.unit
    def test_model_cpu(self):
        """Test model on CPU."""
        model = SegmentationModel(
            architecture='unet',
            encoder_name='resnet18',
            in_channels=3,
            num_classes=1
        )
        
        model = model.to('cpu')
        x = torch.randn(1, 3, 128, 128)
        
        output = model(x)
        assert output.device.type == 'cpu'
    
    @pytest.mark.unit
    @pytest.mark.requires_gpu
    def test_model_cuda(self):
        """Test model on CUDA."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        model = SegmentationModel(
            architecture='unet',
            encoder_name='resnet18',
            in_channels=3,
            num_classes=1
        )
        
        model = model.to('cuda')
        x = torch.randn(1, 3, 128, 128).cuda()
        
        output = model(x)
        assert output.device.type == 'cuda'


class TestModelParameters:
    """Tests for model parameters."""
    
    @pytest.mark.unit
    def test_model_has_parameters(self):
        """Test that model has learnable parameters."""
        model = SegmentationModel(
            architecture='unet',
            encoder_name='resnet18',
            in_channels=3,
            num_classes=1
        )
        
        params = list(model.parameters())
        assert len(params) > 0
        
        # Check parameters require gradients
        trainable_params = [p for p in params if p.requires_grad]
        assert len(trainable_params) > 0
    
    @pytest.mark.unit
    def test_model_parameter_count(self):
        """Test model parameter count."""
        model = SegmentationModel(
            architecture='unet',
            encoder_name='resnet18',
            in_channels=3,
            num_classes=1
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        assert total_params > 0
        assert trainable_params > 0
        assert trainable_params == total_params  # All params should be trainable


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
