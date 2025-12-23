"""
Unit tests for DataManager.

Tests:
- Image loading and caching
- Video frame extraction
- Dataset organization
- Split creation
"""

import pytest
import numpy as np
from pathlib import Path

from src.core.data_manager import DataManager


class TestDataManagerInit:
    """Tests for DataManager initialization."""
    
    @pytest.mark.unit
    def test_init_default(self, temp_dir):
        """Test DataManager initialization with defaults."""
        dm = DataManager(str(temp_dir))
        
        assert dm.data_root == str(temp_dir)
        assert dm.cache_size_mb == 1024  # Default
        assert isinstance(dm.dataset, dict)
        assert 'all' in dm.dataset
    
    @pytest.mark.unit
    def test_init_custom_cache_size(self, temp_dir):
        """Test DataManager with custom cache size."""
        dm = DataManager(str(temp_dir), cache_size_mb=512)
        assert dm.cache_size_mb == 512


class TestImageLoading:
    """Tests for image loading operations."""
    
    @pytest.mark.unit
    def test_load_image_success(self, create_test_images):
        """Test successful image loading."""
        image_paths = create_test_images(count=1)
        image_path = image_paths[0]
        
        dm = DataManager(str(Path(image_path).parent.parent))
        image = dm.load_image(image_path)
        
        assert image is not None
        assert isinstance(image, np.ndarray)
        assert len(image.shape) == 3  # HxWxC
        assert image.shape[2] == 3  # RGB
    
    @pytest.mark.unit
    def test_load_image_nonexistent(self, temp_dir):
        """Test loading non-existent image."""
        dm = DataManager(str(temp_dir))
        image = dm.load_image("nonexistent_image.png")
        
        assert image is None
    
    @pytest.mark.unit
    def test_load_image_caching(self, create_test_images, temp_dir):
        """Test image caching mechanism."""
        image_paths = create_test_images(count=1)
        image_path = image_paths[0]
        
        dm = DataManager(str(temp_dir))
        
        # Load image first time
        image1 = dm.load_image(image_path)
        cache_size_1 = dm.get_cache_size()
        
        # Load same image again
        image2 = dm.load_image(image_path)
        cache_size_2 = dm.get_cache_size()
        
        # Should return same image from cache
        assert np.array_equal(image1, image2)
        # Cache size should not increase
        assert cache_size_1 == cache_size_2
    
    @pytest.mark.unit
    def test_load_batch_images(self, create_test_images, temp_dir):
        """Test loading batch of images."""
        image_paths = create_test_images(count=5)
        
        dm = DataManager(str(temp_dir))
        images = dm.load_batch(image_paths)
        
        assert len(images) == 5
        for image in images:
            assert image is not None
            assert isinstance(image, np.ndarray)


class TestVideoProcessing:
    """Tests for video frame extraction."""
    
    @pytest.mark.unit
    @pytest.mark.slow
    def test_load_video_frames(self, temp_dir):
        """Test video frame extraction."""
        # Create a simple test video
        import cv2
        video_path = temp_dir / "test_video.mp4"
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_path), fourcc, 30.0, (640, 480))
        
        # Write 30 frames
        for i in range(30):
            frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
            out.write(frame)
        
        out.release()
        
        # Extract frames
        dm = DataManager(str(temp_dir))
        frames = dm.load_video(str(video_path), frame_interval=5)
        
        assert len(frames) > 0
        assert len(frames) <= 6  # 30 frames / 5 interval = 6 frames
    
    @pytest.mark.unit
    def test_save_video_frames(self, temp_dir):
        """Test saving video frames to disk."""
        # Create dummy frames
        import cv2
        video_path = temp_dir / "test_video2.mp4"
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_path), fourcc, 30.0, (320, 240))
        
        for i in range(15):
            frame = np.random.randint(0, 256, (240, 320, 3), dtype=np.uint8)
            out.write(frame)
        
        out.release()
        
        # Save frames
        dm = DataManager(str(temp_dir))
        output_dir = temp_dir / "video_frames"
        frame_paths = dm.save_video_frames(str(video_path), output_dir, frame_interval=3)
        
        assert len(frame_paths) > 0
        assert output_dir.exists()
        for frame_path in frame_paths:
            assert Path(frame_path).exists()


class TestDatasetOrganization:
    """Tests for dataset organization."""
    
    @pytest.mark.unit
    def test_add_to_dataset(self, create_test_images, temp_dir):
        """Test adding images to dataset."""
        image_paths = create_test_images(count=10)
        
        dm = DataManager(str(temp_dir))
        dm.dataset['all'] = image_paths
        
        assert len(dm.dataset['all']) == 10
    
    @pytest.mark.unit
    def test_create_splits(self, create_test_images, temp_dir):
        """Test creating train/val/test splits."""
        image_paths = create_test_images(count=100)
        
        dm = DataManager(str(temp_dir))
        dm.dataset['all'] = image_paths
        
        # Create splits
        splits = dm.create_splits(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
        
        assert 'train' in splits
        assert 'val' in splits
        assert 'test' in splits
        
        # Check split sizes
        total = len(splits['train']) + len(splits['val']) + len(splits['test'])
        assert total == 100
        
        # Check approximate ratios
        assert 65 <= len(splits['train']) <= 75  # ~70%
        assert 10 <= len(splits['val']) <= 20    # ~15%
        assert 10 <= len(splits['test']) <= 20   # ~15%
    
    @pytest.mark.unit
    def test_save_load_splits(self, create_test_images, temp_dir):
        """Test saving and loading split lists."""
        image_paths = create_test_images(count=30)
        
        dm = DataManager(str(temp_dir))
        dm.dataset['all'] = image_paths
        
        # Create and save splits
        splits = dm.create_splits(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
        
        splits_dir = temp_dir / "splits"
        splits_dir.mkdir(exist_ok=True)
        
        dm.save_splits(splits, str(splits_dir))
        
        # Check files exist
        assert (splits_dir / "train.txt").exists()
        assert (splits_dir / "val.txt").exists()
        assert (splits_dir / "test.txt").exists()
        
        # Load splits
        loaded_splits = dm.load_splits(str(splits_dir))
        
        assert len(loaded_splits['train']) == len(splits['train'])
        assert len(loaded_splits['val']) == len(splits['val'])
        assert len(loaded_splits['test']) == len(splits['test'])


class TestCacheManagement:
    """Tests for cache management."""
    
    @pytest.mark.unit
    def test_clear_cache(self, create_test_images, temp_dir):
        """Test clearing image cache."""
        image_paths = create_test_images(count=5)
        
        dm = DataManager(str(temp_dir), cache_size_mb=10)
        
        # Load images to fill cache
        for image_path in image_paths:
            dm.load_image(image_path)
        
        cache_size_before = dm.get_cache_size()
        assert cache_size_before > 0
        
        # Clear cache
        dm.clear_cache()
        cache_size_after = dm.get_cache_size()
        
        assert cache_size_after == 0
    
    @pytest.mark.unit
    def test_cache_eviction(self, create_test_images, temp_dir):
        """Test LRU cache eviction."""
        # Create many images to exceed cache
        image_paths = create_test_images(count=50)
        
        dm = DataManager(str(temp_dir), cache_size_mb=5)  # Small cache
        
        # Load all images
        for image_path in image_paths:
            dm.load_image(image_path)
        
        # Cache should not exceed limit (approximately)
        cache_size_mb = dm.get_cache_size()
        assert cache_size_mb <= 10  # Allow some overhead


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
