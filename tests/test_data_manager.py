"""
DataManager 的单元测试。

测试内容：
- 图像加载和缓存
- 视频帧提取
- 数据集组织
- 划分创建
"""

import pytest
import numpy as np
from pathlib import Path

from src.core.data_manager import DataManager


class TestDataManagerInit:
    """DataManager 初始化测试。"""
    
    @pytest.mark.unit
    def test_init_default(self, temp_dir):
        """测试使用默认值初始化 DataManager。"""
        dm = DataManager(str(temp_dir))
        
        assert dm.data_root == str(temp_dir)
        assert dm.cache_size_mb == 1024  # 默认值
        assert isinstance(dm.dataset, dict)
        assert 'all' in dm.dataset
    
    @pytest.mark.unit
    def test_init_custom_cache_size(self, temp_dir):
        """测试使用自定义缓存大小初始化 DataManager。"""
        dm = DataManager(str(temp_dir), cache_size_mb=512)
        assert dm.cache_size_mb == 512


class TestImageLoading:
    """图像加载操作测试。"""
    
    @pytest.mark.unit
    def test_load_image_success(self, create_test_images):
        """测试成功加载图像。"""
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
        """测试加载不存在的图像。"""
        dm = DataManager(str(temp_dir))
        image = dm.load_image("nonexistent_image.png")
        
        assert image is None
    
    @pytest.mark.unit
    def test_load_image_caching(self, create_test_images, temp_dir):
        """测试图像缓存机制。"""
        image_paths = create_test_images(count=1)
        image_path = image_paths[0]
        
        dm = DataManager(str(temp_dir))
        
        # 第一次加载图像
        image1 = dm.load_image(image_path)
        cache_size_1 = dm.get_cache_size()
        
        # 再次加载同一张图像
        image2 = dm.load_image(image_path)
        cache_size_2 = dm.get_cache_size()
        
        # 应该从缓存返回相同的图像
        assert np.array_equal(image1, image2)
        # 缓存大小不应增加
        assert cache_size_1 == cache_size_2
    
    @pytest.mark.unit
    def test_load_batch_images(self, create_test_images, temp_dir):
        """测试批量加载图像。"""
        image_paths = create_test_images(count=5)
        
        dm = DataManager(str(temp_dir))
        images = dm.load_batch(image_paths)
        
        assert len(images) == 5
        for image in images:
            assert image is not None
            assert isinstance(image, np.ndarray)


class TestVideoProcessing:
    """视频帧提取测试。"""
    
    @pytest.mark.unit
    @pytest.mark.slow
    def test_load_video_frames(self, temp_dir):
        """测试视频帧提取。"""
        # 创建一个简单的测试视频
        import cv2
        video_path = temp_dir / "test_video.mp4"
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_path), fourcc, 30.0, (640, 480))
        
        # 写入 30 帧
        for i in range(30):
            frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
            out.write(frame)
        
        out.release()
        
        # 提取帧
        dm = DataManager(str(temp_dir))
        frames = dm.load_video(str(video_path), frame_interval=5)
        
        assert len(frames) > 0
        assert len(frames) <= 6  # 30 帧 / 5 间隔 = 6 帧
    
    @pytest.mark.unit
    def test_save_video_frames(self, temp_dir):
        """测试将视频帧保存到磁盘。"""
        # 创建虚拟帧
        import cv2
        video_path = temp_dir / "test_video2.mp4"
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_path), fourcc, 30.0, (320, 240))
        
        for i in range(15):
            frame = np.random.randint(0, 256, (240, 320, 3), dtype=np.uint8)
            out.write(frame)
        
        out.release()
        
        # 保存帧
        dm = DataManager(str(temp_dir))
        output_dir = temp_dir / "video_frames"
        frame_paths = dm.save_video_frames(str(video_path), output_dir, frame_interval=3)
        
        assert len(frame_paths) > 0
        assert output_dir.exists()
        for frame_path in frame_paths:
            assert Path(frame_path).exists()


class TestDatasetOrganization:
    """数据集组织测试。"""
    
    @pytest.mark.unit
    def test_add_to_dataset(self, create_test_images, temp_dir):
        """测试向数据集添加图像。"""
        image_paths = create_test_images(count=10)
        
        dm = DataManager(str(temp_dir))
        dm.dataset['all'] = image_paths
        
        assert len(dm.dataset['all']) == 10
    
    @pytest.mark.unit
    def test_create_splits(self, create_test_images, temp_dir):
        """测试创建训练/验证/测试划分。"""
        image_paths = create_test_images(count=100)
        
        dm = DataManager(str(temp_dir))
        dm.dataset['all'] = image_paths
        
        # 创建划分
        splits = dm.create_splits(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
        
        assert 'train' in splits
        assert 'val' in splits
        assert 'test' in splits
        
        # 检查划分大小
        total = len(splits['train']) + len(splits['val']) + len(splits['test'])
        assert total == 100
        
        # 检查大致比例
        assert 65 <= len(splits['train']) <= 75  # ~70%
        assert 10 <= len(splits['val']) <= 20    # ~15%
        assert 10 <= len(splits['test']) <= 20   # ~15%
    
    @pytest.mark.unit
    def test_save_load_splits(self, create_test_images, temp_dir):
        """测试保存和加载划分列表。"""
        image_paths = create_test_images(count=30)
        
        dm = DataManager(str(temp_dir))
        dm.dataset['all'] = image_paths
        
        # 创建并保存划分
        splits = dm.create_splits(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
        
        splits_dir = temp_dir / "splits"
        splits_dir.mkdir(exist_ok=True)
        
        dm.save_splits(splits, str(splits_dir))
        
        # 检查文件是否存在
        assert (splits_dir / "train.txt").exists()
        assert (splits_dir / "val.txt").exists()
        assert (splits_dir / "test.txt").exists()
        
        # 加载划分
        loaded_splits = dm.load_splits(str(splits_dir))
        
        assert len(loaded_splits['train']) == len(splits['train'])
        assert len(loaded_splits['val']) == len(splits['val'])
        assert len(loaded_splits['test']) == len(splits['test'])


class TestCacheManagement:
    """缓存管理测试。"""
    
    @pytest.mark.unit
    def test_clear_cache(self, create_test_images, temp_dir):
        """测试清除图像缓存。"""
        image_paths = create_test_images(count=5)
        
        dm = DataManager(str(temp_dir), cache_size_mb=10)
        
        # 加载图像以填充缓存
        for image_path in image_paths:
            dm.load_image(image_path)
        
        cache_size_before = dm.get_cache_size()
        assert cache_size_before > 0
        
        # 清除缓存
        dm.clear_cache()
        cache_size_after = dm.get_cache_size()
        
        assert cache_size_after == 0
    
    @pytest.mark.unit
    def test_cache_eviction(self, create_test_images, temp_dir):
        """测试 LRU 缓存淘汰。"""
        # 创建多张图像以超过缓存限制
        image_paths = create_test_images(count=50)
        
        dm = DataManager(str(temp_dir), cache_size_mb=5)  # 小缓存
        
        # 加载所有图像
        for image_path in image_paths:
            dm.load_image(image_path)
        
        # 缓存不应超过限制（大致）
        cache_size_mb = dm.get_cache_size()
        assert cache_size_mb <= 10  # 允许一些开销


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
