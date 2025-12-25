"""
模型定义的单元测试。

测试内容：
- 模型架构创建
- 前向传播
- 输出形状
"""

import pytest
import torch
import numpy as np

from src.models.segmentation_models import SegmentationModel


class TestSegmentationModelCreation:
    """分割模型创建测试。"""
    
    @pytest.mark.unit
    def test_create_unet(self):
        """测试 U-Net 模型创建。"""
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
        """测试 DeepLabV3+ 模型创建。"""
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
        """测试 FPN 模型创建。"""
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
        """测试无效架构的错误处理。"""
        with pytest.raises((ValueError, KeyError)):
            SegmentationModel(
                architecture='invalid_arch',
                encoder_name='resnet34',
                in_channels=3,
                num_classes=1
            )


class TestModelForwardPass:
    """模型前向传播测试。"""
    
    @pytest.mark.unit
    def test_unet_forward(self):
        """测试 U-Net 前向传播。"""
        model = SegmentationModel(
            architecture='unet',
            encoder_name='resnet18',  # 使用较小的模型以加快测试速度
            in_channels=3,
            num_classes=1
        )
        
        # 创建虚拟输入
        x = torch.randn(2, 3, 256, 256)  # Batch=2, C=3, H=256, W=256
        
        # 前向传播
        output = model(x)
        
        # 检查输出形状
        assert output.shape == (2, 1, 256, 256)
    
    @pytest.mark.unit
    def test_model_training_mode(self):
        """测试训练模式下的模型。"""
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
        """测试评估模式下的模型。"""
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
    """模型设备管理测试。"""
    
    @pytest.mark.unit
    def test_model_cpu(self):
        """测试 CPU 上的模型。"""
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
        """测试 CUDA 上的模型。"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA 不可用")
        
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
    """模型参数测试。"""
    
    @pytest.mark.unit
    def test_model_has_parameters(self):
        """测试模型是否具有可学习参数。"""
        model = SegmentationModel(
            architecture='unet',
            encoder_name='resnet18',
            in_channels=3,
            num_classes=1
        )
        
        params = list(model.parameters())
        assert len(params) > 0
        
        # 检查参数是否需要梯度
        trainable_params = [p for p in params if p.requires_grad]
        assert len(trainable_params) > 0
    
    @pytest.mark.unit
    def test_model_parameter_count(self):
        """测试模型参数计数。"""
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
        assert trainable_params == total_params  # 所有参数都应该是可训练的


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
