"""
Pytest configuration and fixtures.

This module provides:
- Common test fixtures
- Mock data generators
- Test utilities
- Pytest hooks
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil
import yaml

# Test data directory
TEST_DATA_DIR = Path(__file__).parent / "test_data"
TEMP_DIR = Path(tempfile.gettempdir()) / "industrial_defect_seg_tests"


@pytest.fixture(scope="session")
def test_data_dir():
    """Get test data directory."""
    TEST_DATA_DIR.mkdir(exist_ok=True)
    return TEST_DATA_DIR


@pytest.fixture(scope="session")
def temp_dir():
    """Create temporary directory for test outputs."""
    TEMP_DIR.mkdir(exist_ok=True)
    yield TEMP_DIR
    # Cleanup after all tests
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)


@pytest.fixture
def temp_output_dir(temp_dir):
    """Create temporary output directory for each test."""
    output_dir = temp_dir / f"output_{pytest.current_test_name}"
    output_dir.mkdir(exist_ok=True)
    yield output_dir
    # Cleanup after each test
    if output_dir.exists():
        shutil.rmtree(output_dir)


@pytest.fixture
def sample_image():
    """Generate a sample RGB image (256x256x3)."""
    image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    return image


@pytest.fixture
def sample_grayscale_image():
    """Generate a sample grayscale image (256x256)."""
    image = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
    return image


@pytest.fixture
def sample_mask():
    """Generate a sample binary mask (256x256)."""
    mask = np.zeros((256, 256), dtype=np.uint8)
    # Add some defects
    mask[50:100, 50:100] = 255  # Square defect
    mask[150:180, 150:200] = 255  # Rectangle defect
    return mask


@pytest.fixture
def sample_mask_with_multiple_defects():
    """Generate a mask with multiple separate defects."""
    mask = np.zeros((256, 256), dtype=np.uint8)
    # Add 5 separate defects
    mask[20:40, 20:40] = 255
    mask[60:90, 100:130] = 255
    mask[120:150, 50:80] = 255
    mask[180:210, 180:210] = 255
    mask[200:230, 80:110] = 255
    return mask


@pytest.fixture
def empty_mask():
    """Generate an empty mask (all zeros)."""
    return np.zeros((256, 256), dtype=np.uint8)


@pytest.fixture
def sample_image_batch():
    """Generate a batch of sample images."""
    return [np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8) for _ in range(5)]


@pytest.fixture
def sample_mask_batch():
    """Generate a batch of sample masks."""
    masks = []
    for i in range(5):
        mask = np.zeros((256, 256), dtype=np.uint8)
        # Different defect for each mask
        y_start = 50 + i * 30
        x_start = 50 + i * 20
        mask[y_start:y_start+40, x_start:x_start+40] = 255
        masks.append(mask)
    return masks


@pytest.fixture
def sample_config():
    """Generate sample configuration dictionary."""
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
    """Generate sample paths configuration."""
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
    """Factory fixture to create test image files."""
    def _create_images(count=5, size=(256, 256)):
        """Create test image files and return their paths."""
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
    """Factory fixture to create test mask files."""
    def _create_masks(count=5, size=(256, 256)):
        """Create test mask files and return their paths."""
        import cv2
        masks_dir = temp_dir / 'test_masks'
        masks_dir.mkdir(exist_ok=True)
        
        mask_paths = []
        for i in range(count):
            mask = np.zeros(size, dtype=np.uint8)
            # Add some defects
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
    """Generate mock statistics for testing."""
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
    """Generate mock training history."""
    epochs = 50
    return {
        'loss': np.linspace(0.5, 0.1, epochs).tolist(),
        'val_loss': np.linspace(0.55, 0.15, epochs).tolist(),
        'iou': np.linspace(0.6, 0.9, epochs).tolist(),
        'val_iou': np.linspace(0.55, 0.85, epochs).tolist(),
        'dice': np.linspace(0.65, 0.92, epochs).tolist(),
        'val_dice': np.linspace(0.6, 0.88, epochs).tolist()
    }


# Pytest hooks

def pytest_configure(config):
    """Configure pytest."""
    # Register custom markers
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "slow: Slow tests")
    config.addinivalue_line("markers", "requires_gpu: Tests requiring GPU")
    config.addinivalue_line("markers", "requires_sam: Tests requiring SAM weights")
    config.addinivalue_line("markers", "ui: UI tests")


def pytest_collection_modifyitems(config, items):
    """Modify test items after collection."""
    # Skip GPU tests if CUDA not available
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False
    
    skip_gpu = pytest.mark.skip(reason="CUDA not available")
    
    for item in items:
        if "requires_gpu" in item.keywords and not has_cuda:
            item.add_marker(skip_gpu)
        
        # Mark slow tests
        if "slow" in item.keywords:
            item.add_marker(pytest.mark.slow)


@pytest.fixture(scope="session", autouse=True)
def setup_logging():
    """Setup logging for tests."""
    import logging
    logging.basicConfig(
        level=logging.WARNING,  # Only show warnings and errors during tests
        format='%(levelname)s - %(name)s - %(message)s'
    )
