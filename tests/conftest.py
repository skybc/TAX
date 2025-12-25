"""
Pytest 配置和固件。

此模块提供：
- 通用测试固件
- 模拟数据生成器
- 测试实用程序
- Pytest 钩子
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil
import yaml

# 测试数据目录
TEST_DATA_DIR = Path(__file__).parent / "test_data"
TEMP_DIR = Path(tempfile.gettempdir()) / "industrial_defect_seg_tests"


@pytest.fixture(scope="session")
def test_data_dir():
    """获取测试数据目录。"""
    TEST_DATA_DIR.mkdir(exist_ok=True)
    return TEST_DATA_DIR


@pytest.fixture(scope="session")
def temp_dir():
    """为测试输出创建临时目录。"""
    TEMP_DIR.mkdir(exist_ok=True)
    yield TEMP_DIR
    # 测试完成后清理
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)


@pytest.fixture
def temp_output_dir(temp_dir):
    """为每个测试创建临时输出目录。"""
    output_dir = temp_dir / f"output_{pytest.current_test_name}"
    output_dir.mkdir(exist_ok=True)
    yield output_dir
    # 每个测试后清理
    if output_dir.exists():
        shutil.rmtree(output_dir)


@pytest.fixture
def sample_image():
    """生成示例 RGB 图像 (256x256x3)。"""
    image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    return image


@pytest.fixture
def sample_grayscale_image():
    """生成示例灰度图像 (256x256)。"""
    image = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
    return image


@pytest.fixture
def sample_mask():
    """生成示例二值掩码 (256x256)。"""
    mask = np.zeros((256, 256), dtype=np.uint8)
    # 添加一些缺陷
    mask[50:100, 50:100] = 255  # 正方形缺陷
    mask[150:180, 150:200] = 255  # 矩形缺陷
    return mask


@pytest.fixture
def sample_mask_with_multiple_defects():
    """生成具有多个独立缺陷的掩码。"""
    mask = np.zeros((256, 256), dtype=np.uint8)
    # 添加 5 个独立的缺陷
    mask[20:40, 20:40] = 255
    mask[60:90, 100:130] = 255
    mask[120:150, 50:80] = 255
    mask[180:210, 180:210] = 255
    mask[200:230, 80:110] = 255
    return mask


@pytest.fixture
def empty_mask():
    """生成空掩码（全零）。"""
    return np.zeros((256, 256), dtype=np.uint8)


@pytest.fixture
def sample_image_batch():
    """生成一批示例图像。"""
    return [np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8) for _ in range(5)]


@pytest.fixture
def sample_mask_batch():
    """生成一批示例掩码。"""
    masks = []
    for i in range(5):
        mask = np.zeros((256, 256), dtype=np.uint8)
        # 每个掩码具有不同的缺陷
        y_start = 50 + i * 30
        x_start = 50 + i * 20
        mask[y_start:y_start+40, x_start:x_start+40] = 255
        masks.append(mask)
    return masks


@pytest.fixture
def sample_config():
    """生成示例配置字典。"""
    return {
        'app': {
            'name': 'Test App',
            'version': '1.0.0-test'
        },
        'device': {
            'type': 'cpu'
        },
        'performance': {
            'max_cache_size': 512,
            'num_workers': 2
        }
    }


@pytest.fixture
def sample_paths_config(temp_dir):
    """生成示例路径配置。"""
    return {
        'paths': {
            'data_root': str(temp_dir / 'data'),
            'raw_data': str(temp_dir / 'data' / 'raw'),
            'processed_data': str(temp_dir / 'data' / 'processed'),
            'images': str(temp_dir / 'data' / 'processed' / 'images'),
            'masks': str(temp_dir / 'data' / 'processed' / 'masks'),
            'outputs': str(temp_dir / 'outputs'),
            'predictions': str(temp_dir / 'outputs' / 'predictions'),
            'reports': str(temp_dir / 'outputs' / 'reports')
        }
    }


@pytest.fixture
def create_test_images(temp_dir):
    """创建测试图像文件的工厂固件。"""
    def _create_images(count=5, size=(256, 256)):
        """创建测试图像文件并返回其路径。"""
        import cv2
        images_dir = temp_dir / 'test_images'
        images_dir.mkdir(exist_ok=True)
        
        image_paths = []
        for i in range(count):
            image = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
            image_path = images_dir / f'image_{i:03d}.png'
            cv2.imwrite(str(image_path), image)
            image_paths.append(str(image_path))
        
        return image_paths
    
    return _create_images


@pytest.fixture
def create_test_masks(temp_dir):
    """创建测试掩码文件的工厂固件。"""
    def _create_masks(count=5, size=(256, 256)):
        """创建测试掩码文件并返回其路径。"""
        import cv2
        masks_dir = temp_dir / 'test_masks'
        masks_dir.mkdir(exist_ok=True)
        
        mask_paths = []
        for i in range(count):
            mask = np.zeros(size, dtype=np.uint8)
            # 添加一些缺陷
            y_start = 50 + (i * 30) % 150
            x_start = 50 + (i * 40) % 150
            mask[y_start:y_start+40, x_start:x_start+40] = 255
            
            mask_path = masks_dir / f'mask_{i:03d}.png'
            cv2.imwrite(str(mask_path), mask)
            mask_paths.append(str(mask_path))
        
        return mask_paths
    
    return _create_masks


@pytest.fixture
def mock_statistics():
    """生成用于测试的模拟统计数据。"""
    return {
        'total_images': 100,
        'images_with_defects': 85,
        'images_without_defects': 15,
        'total_defects': 423,
        'mean_defects_per_image': 4.23,
        'std_defects_per_image': 2.15,
        'total_defect_area': 125430,
        'mean_defect_area': 296.5,
        'std_defect_area': 145.2,
        'mean_coverage_ratio': 0.0234,
        'defect_size_distribution': {
            'bins': [0, 100, 200, 300, 400, 500],
            'counts': [50, 120, 150, 80, 23]
        },
        'per_image_stats': [
            {
                'image_name': f'image_{i:03d}.png',
                'num_defects': np.random.randint(1, 10),
                'total_area': np.random.randint(500, 2000),
                'coverage_ratio': np.random.uniform(0.01, 0.05)
            }
            for i in range(100)
        ]
    }


@pytest.fixture
def mock_training_history():
    """生成模拟训练历史记录。"""
    epochs = 50
    return {
        'loss': np.linspace(0.5, 0.1, epochs).tolist(),
        'val_loss': np.linspace(0.55, 0.15, epochs).tolist(),
        'iou': np.linspace(0.6, 0.9, epochs).tolist(),
        'val_iou': np.linspace(0.55, 0.85, epochs).tolist(),
        'dice': np.linspace(0.65, 0.92, epochs).tolist(),
        'val_dice': np.linspace(0.6, 0.88, epochs).tolist()
    }


# Pytest 钩子

def pytest_configure(config):
    """配置 pytest。"""
    # 注册自定义标记
    config.addinivalue_line("markers", "unit: 单元测试")
    config.addinivalue_line("markers", "integration: 集成测试")
    config.addinivalue_line("markers", "performance: 性能测试")
    config.addinivalue_line("markers", "slow: 慢速测试")
    config.addinivalue_line("markers", "requires_gpu: 需要 GPU 的测试")
    config.addinivalue_line("markers", "requires_sam: 需要 SAM 权重的测试")
    config.addinivalue_line("markers", "ui: UI 测试")


def pytest_collection_modifyitems(config, items):
    """在收集后修改测试项。"""
    # 如果 CUDA 不可用，则跳过 GPU 测试
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False
    
    skip_gpu = pytest.mark.skip(reason="CUDA 不可用")
    
    for item in items:
        if "requires_gpu" in item.keywords and not has_cuda:
            item.add_marker(skip_gpu)
        
        # 标记慢速测试
        if "slow" in item.keywords:
            item.add_marker(pytest.mark.slow)


@pytest.fixture(scope="session", autouse=True)
def setup_logging():
    """为测试设置日志。"""
    import logging
    logging.basicConfig(
        level=logging.WARNING,  # 测试期间仅显示警告和错误
        format='%(levelname)s - %(name)s - %(message)s'
    )
