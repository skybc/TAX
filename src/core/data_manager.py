"""
工业缺陷分割系统的数据管理模块。

此模块负责：
- 图像和视频加载
- 数据集组织和缓存
- 批量数据管理
- 训练/验证/测试集划分管理
"""

import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
import numpy as np
import cv2
from PIL import Image

from src.logger import get_logger
from src.utils.file_utils import list_files, ensure_dir, save_json, load_json
from src.utils.image_utils import load_image, get_image_info

logger = get_logger(__name__)


class DataManager:
    """
    管理图像数据加载、缓存和数据集组织。
    
    此类提供以下功能：
    - 从各种来源（文件夹、文件、视频）加载图像
    - 缓存加载的图像以提高性能
    - 使用训练/验证/测试划分组织数据集
    - 管理数据集元数据
    
    属性:
        data_root: 所有数据的根目录
        cache_size_mb: 最大缓存大小（MB）
        image_cache: 用于加载图像的 LRU 缓存
        dataset: 当前数据集信息
    """
    
    def __init__(self, data_root: str, cache_size_mb: int = 2048):
        """
        初始化 DataManager。
        
        参数:
            data_root: 数据存储的根目录
            cache_size_mb: 最大缓存大小（MB）（默认：2048）
        """
        self.data_root = Path(data_root)
        self.cache_size_mb = cache_size_mb
        self.cache_size_bytes = cache_size_mb * 1024 * 1024
        
        # 图像缓存: {path: (image_array, size_bytes)}
        self.image_cache: Dict[str, Tuple[np.ndarray, int]] = {}
        self.cache_used_bytes = 0
        
        # 数据集结构
        self.dataset: Dict[str, List[str]] = {
            'all': [],      # 所有图像路径
            'train': [],    # 训练集
            'val': [],      # 验证集
            'test': []      # 测试集
        }
        
        # 支持的图像格式
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # 支持的视频格式
        self.supported_video_formats = {'.mp4', '.avi', '.mov', '.mkv', '.flv'}
        
        logger.info(f"DataManager 已初始化，缓存大小: {cache_size_mb} MB")
    
    def load_image(self, image_path: Union[str, Path], use_cache: bool = True) -> Optional[np.ndarray]:
        """
        从文件加载图像，可选是否使用缓存。
        
        参数:
            image_path: 图像文件路径
            use_cache: 是否使用缓存（默认：True）
            
        返回:
            numpy 数组格式的图像 (HxWxC)，如果失败则返回 None
        """
        image_path = str(image_path)
        
        # 首先检查缓存
        if use_cache and image_path in self.image_cache:
            logger.debug(f"从缓存加载图像: {image_path}")
            return self.image_cache[image_path][0].copy()
        
        # 从文件加载图像
        try:
            image = load_image(image_path)
            
            if image is None:
                logger.error(f"加载图像失败: {image_path}")
                return None
            
            # 如果启用，则添加到缓存
            if use_cache:
                self._add_to_cache(image_path, image)
            
            logger.debug(f"已加载图像: {image_path}, 形状: {image.shape}")
            return image
            
        except Exception as e:
            logger.error(f"加载图像 {image_path} 时出错: {e}")
            return None
    
    def load_images_from_folder(self, folder_path: Union[str, Path], 
                                recursive: bool = False) -> List[str]:
        """
        从文件夹加载所有图像。
        
        参数:
            folder_path: 包含图像的文件夹路径
            recursive: 是否在子目录中递归搜索
            
        返回:
            图像文件路径列表
        """
        folder_path = Path(folder_path)
        
        if not folder_path.exists():
            logger.error(f"文件夹不存在: {folder_path}")
            return []
        
        # 查找所有图像文件
        image_files = list_files(
            folder_path,
            extensions=list(self.supported_formats),
            recursive=recursive
        )
        
        logger.info(f"在 {folder_path} 中找到 {len(image_files)} 张图像")
        return image_files
    
    def load_video(self, video_path: Union[str, Path], 
                   frame_interval: int = 1,
                   max_frames: Optional[int] = None) -> List[np.ndarray]:
        """
        从视频文件加载帧。
        
        参数:
            video_path: 视频文件路径
            frame_interval: 每隔 N 帧提取一次（默认：1，提取所有帧）
            max_frames: 最大提取帧数（默认：None，提取所有）
            
        返回:
            numpy 数组格式的帧图像列表
        """
        video_path = Path(video_path)
        
        if not video_path.exists():
            logger.error(f"视频文件不存在: {video_path}")
            return []
        
        if video_path.suffix.lower() not in self.supported_video_formats:
            logger.error(f"不支持的视频格式: {video_path.suffix}")
            return []
        
        frames = []
        
        try:
            cap = cv2.VideoCapture(str(video_path))
            
            if not cap.isOpened():
                logger.error(f"无法打开视频: {video_path}")
                return []
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"视频信息 - 总帧数: {total_frames}, FPS: {fps:.2f}")
            
            frame_count = 0
            extracted_count = 0
            
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # 按指定间隔提取帧
                if frame_count % frame_interval == 0:
                    # 将 BGR 转换为 RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
                    extracted_count += 1
                    
                    # 检查最大帧数限制
                    if max_frames is not None and extracted_count >= max_frames:
                        break
                
                frame_count += 1
            
            cap.release()
            
            logger.info(f"从视频中提取了 {len(frames)} 帧")
            
        except Exception as e:
            logger.error(f"加载视频 {video_path} 时出错: {e}")
        
        return frames
    
    def save_video_frames(self, video_path: Union[str, Path],
                         output_dir: Union[str, Path],
                         frame_interval: int = 1,
                         max_frames: Optional[int] = None,
                         prefix: str = "frame") -> List[str]:
        """
        从视频中提取帧并保存到文件夹。
        
        参数:
            video_path: 视频文件路径
            output_dir: 保存帧的目录
            frame_interval: 每隔 N 帧提取一次
            max_frames: 最大提取帧数
            prefix: 保存帧的文件名前缀
            
        返回:
            保存的帧文件路径列表
        """
        output_dir = Path(output_dir)
        ensure_dir(output_dir)
        
        frames = self.load_video(video_path, frame_interval, max_frames)
        saved_paths = []
        
        for i, frame in enumerate(frames):
            frame_path = output_dir / f"{prefix}_{i:06d}.jpg"
            
            try:
                # 为 OpenCV 将 RGB 转换为 BGR
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(frame_path), frame_bgr)
                saved_paths.append(str(frame_path))
            except Exception as e:
                logger.error(f"保存第 {i} 帧失败: {e}")
        
        logger.info(f"已将 {len(saved_paths)} 帧保存到 {output_dir}")
        return saved_paths
    
    def create_dataset(self, image_paths: List[str],
                      train_ratio: float = 0.7,
                      val_ratio: float = 0.15,
                      test_ratio: float = 0.15,
                      shuffle: bool = True,
                      random_seed: int = 42) -> Dict[str, List[str]]:
        """
        将图像路径划分为训练/验证/测试集。
        
        参数:
            image_paths: 图像文件路径列表
            train_ratio: 训练集比例（默认：0.7）
            val_ratio: 验证集比例（默认：0.15）
            test_ratio: 测试集比例（默认：0.15）
            shuffle: 是否在划分前打乱
            random_seed: 用于重现的随机种子
            
        返回:
            包含 'train'、'val'、'test' 键及其对应图像路径的字典
        """
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            logger.error("训练/验证/测试比例之和必须为 1.0")
            return {}
        
        # 如果有要求，则打乱
        if shuffle:
            np.random.seed(random_seed)
            image_paths = image_paths.copy()
            np.random.shuffle(image_paths)
        
        n_total = len(image_paths)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        self.dataset['all'] = image_paths
        self.dataset['train'] = image_paths[:n_train]
        self.dataset['val'] = image_paths[n_train:n_train + n_val]
        self.dataset['test'] = image_paths[n_train + n_val:]
        
        logger.info(f"数据集已创建 - 训练集: {len(self.dataset['train'])}, "
                   f"验证集: {len(self.dataset['val'])}, 测试集: {len(self.dataset['test'])}")
        
        return self.dataset
    
    def save_dataset_split(self, output_dir: Union[str, Path]):
        """
        将训练/验证/测试划分保存到文本文件。
        
        参数:
            output_dir: 保存划分文件的目录
        """
        output_dir = Path(output_dir)
        ensure_dir(output_dir)
        
        for split_name in ['train', 'val', 'test']:
            split_file = output_dir / f"{split_name}.txt"
            
            with open(split_file, 'w') as f:
                for path in self.dataset[split_name]:
                    f.write(f"{path}\n")
            
            logger.info(f"已将 {split_name} 划分保存到 {split_file}")
    
    def load_dataset_split(self, split_dir: Union[str, Path]) -> Dict[str, List[str]]:
        """
        从文本文件加载训练/验证/测试划分。
        
        参数:
            split_dir: 包含划分文件的目录
            
        返回:
            包含 'train'、'val'、'test' 键的字典
        """
        split_dir = Path(split_dir)
        
        for split_name in ['train', 'val', 'test']:
            split_file = split_dir / f"{split_name}.txt"
            
            if split_file.exists():
                with open(split_file, 'r') as f:
                    self.dataset[split_name] = [line.strip() for line in f if line.strip()]
                
                logger.info(f"已加载 {split_name} 划分: {len(self.dataset[split_name])} 张图像")
            else:
                logger.warning(f"未找到划分文件: {split_file}")
        
        # 更新 'all' 列表
        self.dataset['all'] = (self.dataset['train'] + 
                               self.dataset['val'] + 
                               self.dataset['test'])
        
        return self.dataset
    
    def get_dataset_info(self) -> Dict[str, any]:
        """
        获取有关当前数据集的信息。
        
        返回:
            包含数据集统计信息的字典
        """
        info = {
            'total_images': len(self.dataset['all']),
            'train_images': len(self.dataset['train']),
            'val_images': len(self.dataset['val']),
            'test_images': len(self.dataset['test']),
            'cache_size_mb': self.cache_size_mb,
            'cache_used_mb': self.cache_used_bytes / (1024 * 1024),
            'cached_images': len(self.image_cache)
        }
        
        return info
    
    def clear_cache(self):
        """清除图像缓存。"""
        self.image_cache.clear()
        self.cache_used_bytes = 0
        logger.info("图像缓存已清除")
    
    def _add_to_cache(self, image_path: str, image: np.ndarray):
        """
        使用 LRU 淘汰策略将图像添加到缓存。
        
        参数:
            image_path: 图像路径
            image: 图像数组
        """
        # 计算图像大小（字节）
        image_size = image.nbytes
        
        # 如果缓存已满，则淘汰图像
        while (self.cache_used_bytes + image_size > self.cache_size_bytes 
               and len(self.image_cache) > 0):
            # 移除最旧的条目（字典中的第一个）
            oldest_path = next(iter(self.image_cache))
            oldest_size = self.image_cache[oldest_path][1]
            del self.image_cache[oldest_path]
            self.cache_used_bytes -= oldest_size
            logger.debug(f"已从缓存中淘汰: {oldest_path}")
        
        # 将新图像添加到缓存
        self.image_cache[image_path] = (image.copy(), image_size)
        self.cache_used_bytes += image_size
        
        logger.debug(f"已添加到缓存: {image_path}, "
                    f"缓存使用情况: {self.cache_used_bytes / (1024*1024):.1f} MB")
    
    def preload_images(self, image_paths: List[str], max_images: Optional[int] = None):
        """
        预加载图像到缓存。
        
        参数:
            image_paths: 要预加载的图像路径列表
            max_images: 最大预加载图像数（默认：None，全部）
        """
        if max_images is not None:
            image_paths = image_paths[:max_images]
        
        loaded_count = 0
        for path in image_paths:
            if self.load_image(path, use_cache=True) is not None:
                loaded_count += 1
        
        logger.info(f"已预加载 {loaded_count} 张图像到缓存")
    
    def get_image_by_index(self, index: int, split: str = 'all') -> Optional[np.ndarray]:
        """
        通过特定划分中的索引获取图像。
        
        参数:
            index: 图像索引
            split: 数据集划分 ('all', 'train', 'val', 'test')
            
        返回:
            图像数组，如果索引超出范围则返回 None
        """
        if split not in self.dataset:
            logger.error(f"无效的划分: {split}")
            return None
        
        if index < 0 or index >= len(self.dataset[split]):
            logger.error(f"索引 {index} 超出划分 {split} 的范围")
            return None
        
        image_path = self.dataset[split][index]
        return self.load_image(image_path)
    
    def get_batch(self, indices: List[int], split: str = 'all') -> List[np.ndarray]:
        """
        通过索引获取一批图像。
        
        参数:
            indices: 图像索引列表
            split: 数据集划分 ('all', 'train', 'val', 'test')
            
        返回:
            图像数组列表
        """
        images = []
        for idx in indices:
            image = self.get_image_by_index(idx, split)
            if image is not None:
                images.append(image)
        
        return images
