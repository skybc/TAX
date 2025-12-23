"""
Data management module for the Industrial Defect Segmentation System.

This module handles:
- Image and video loading
- Dataset organization and caching
- Batch data management
- Train/val/test split management
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
    Manages image data loading, caching, and dataset organization.
    
    This class provides functionality for:
    - Loading images from various sources (folders, files, videos)
    - Caching loaded images for performance
    - Organizing datasets with train/val/test splits
    - Managing dataset metadata
    
    Attributes:
        data_root: Root directory for all data
        cache_size_mb: Maximum cache size in MB
        image_cache: LRU cache for loaded images
        dataset: Current dataset information
    """
    
    def __init__(self, data_root: str, cache_size_mb: int = 2048):
        """
        Initialize DataManager.
        
        Args:
            data_root: Root directory for data storage
            cache_size_mb: Maximum cache size in MB (default: 2048)
        """
        self.data_root = Path(data_root)
        self.cache_size_mb = cache_size_mb
        self.cache_size_bytes = cache_size_mb * 1024 * 1024
        
        # Image cache: {path: (image_array, size_bytes)}
        self.image_cache: Dict[str, Tuple[np.ndarray, int]] = {}
        self.cache_used_bytes = 0
        
        # Dataset structure
        self.dataset: Dict[str, List[str]] = {
            'all': [],      # All image paths
            'train': [],    # Training set
            'val': [],      # Validation set
            'test': []      # Test set
        }
        
        # Supported image formats
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # Supported video formats
        self.supported_video_formats = {'.mp4', '.avi', '.mov', '.mkv', '.flv'}
        
        logger.info(f"DataManager initialized with cache size: {cache_size_mb} MB")
    
    def load_image(self, image_path: Union[str, Path], use_cache: bool = True) -> Optional[np.ndarray]:
        """
        Load an image from file with optional caching.
        
        Args:
            image_path: Path to image file
            use_cache: Whether to use cache (default: True)
            
        Returns:
            Image as numpy array (HxWxC) or None if failed
        """
        image_path = str(image_path)
        
        # Check cache first
        if use_cache and image_path in self.image_cache:
            logger.debug(f"Loading image from cache: {image_path}")
            return self.image_cache[image_path][0].copy()
        
        # Load image from file
        try:
            image = load_image(image_path)
            
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return None
            
            # Add to cache if enabled
            if use_cache:
                self._add_to_cache(image_path, image)
            
            logger.debug(f"Loaded image: {image_path}, shape: {image.shape}")
            return image
            
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return None
    
    def load_images_from_folder(self, folder_path: Union[str, Path], 
                                recursive: bool = False) -> List[str]:
        """
        Load all images from a folder.
        
        Args:
            folder_path: Path to folder containing images
            recursive: Whether to search recursively in subdirectories
            
        Returns:
            List of image file paths
        """
        folder_path = Path(folder_path)
        
        if not folder_path.exists():
            logger.error(f"Folder does not exist: {folder_path}")
            return []
        
        # Find all image files
        image_files = list_files(
            folder_path,
            extensions=list(self.supported_formats),
            recursive=recursive
        )
        
        logger.info(f"Found {len(image_files)} images in {folder_path}")
        return image_files
    
    def load_video(self, video_path: Union[str, Path], 
                   frame_interval: int = 1,
                   max_frames: Optional[int] = None) -> List[np.ndarray]:
        """
        Load frames from a video file.
        
        Args:
            video_path: Path to video file
            frame_interval: Extract every N-th frame (default: 1, all frames)
            max_frames: Maximum number of frames to extract (default: None, all)
            
        Returns:
            List of frame images as numpy arrays
        """
        video_path = Path(video_path)
        
        if not video_path.exists():
            logger.error(f"Video file does not exist: {video_path}")
            return []
        
        if video_path.suffix.lower() not in self.supported_video_formats:
            logger.error(f"Unsupported video format: {video_path.suffix}")
            return []
        
        frames = []
        
        try:
            cap = cv2.VideoCapture(str(video_path))
            
            if not cap.isOpened():
                logger.error(f"Failed to open video: {video_path}")
                return []
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"Video info - Total frames: {total_frames}, FPS: {fps:.2f}")
            
            frame_count = 0
            extracted_count = 0
            
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Extract frame at specified interval
                if frame_count % frame_interval == 0:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
                    extracted_count += 1
                    
                    # Check max frames limit
                    if max_frames is not None and extracted_count >= max_frames:
                        break
                
                frame_count += 1
            
            cap.release()
            
            logger.info(f"Extracted {len(frames)} frames from video")
            
        except Exception as e:
            logger.error(f"Error loading video {video_path}: {e}")
        
        return frames
    
    def save_video_frames(self, video_path: Union[str, Path],
                         output_dir: Union[str, Path],
                         frame_interval: int = 1,
                         max_frames: Optional[int] = None,
                         prefix: str = "frame") -> List[str]:
        """
        Extract and save frames from video to folder.
        
        Args:
            video_path: Path to video file
            output_dir: Directory to save frames
            frame_interval: Extract every N-th frame
            max_frames: Maximum number of frames to extract
            prefix: Filename prefix for saved frames
            
        Returns:
            List of saved frame file paths
        """
        output_dir = Path(output_dir)
        ensure_dir(output_dir)
        
        frames = self.load_video(video_path, frame_interval, max_frames)
        saved_paths = []
        
        for i, frame in enumerate(frames):
            frame_path = output_dir / f"{prefix}_{i:06d}.jpg"
            
            try:
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(frame_path), frame_bgr)
                saved_paths.append(str(frame_path))
            except Exception as e:
                logger.error(f"Failed to save frame {i}: {e}")
        
        logger.info(f"Saved {len(saved_paths)} frames to {output_dir}")
        return saved_paths
    
    def create_dataset(self, image_paths: List[str],
                      train_ratio: float = 0.7,
                      val_ratio: float = 0.15,
                      test_ratio: float = 0.15,
                      shuffle: bool = True,
                      random_seed: int = 42) -> Dict[str, List[str]]:
        """
        Split image paths into train/val/test sets.
        
        Args:
            image_paths: List of image file paths
            train_ratio: Ratio of training set (default: 0.7)
            val_ratio: Ratio of validation set (default: 0.15)
            test_ratio: Ratio of test set (default: 0.15)
            shuffle: Whether to shuffle before splitting
            random_seed: Random seed for reproducibility
            
        Returns:
            Dictionary with 'train', 'val', 'test' keys containing image paths
        """
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            logger.error("Train/val/test ratios must sum to 1.0")
            return {}
        
        # Shuffle if requested
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
        
        logger.info(f"Dataset created - Train: {len(self.dataset['train'])}, "
                   f"Val: {len(self.dataset['val'])}, Test: {len(self.dataset['test'])}")
        
        return self.dataset
    
    def save_dataset_split(self, output_dir: Union[str, Path]):
        """
        Save train/val/test split to text files.
        
        Args:
            output_dir: Directory to save split files
        """
        output_dir = Path(output_dir)
        ensure_dir(output_dir)
        
        for split_name in ['train', 'val', 'test']:
            split_file = output_dir / f"{split_name}.txt"
            
            with open(split_file, 'w') as f:
                for path in self.dataset[split_name]:
                    f.write(f"{path}\n")
            
            logger.info(f"Saved {split_name} split to {split_file}")
    
    def load_dataset_split(self, split_dir: Union[str, Path]) -> Dict[str, List[str]]:
        """
        Load train/val/test split from text files.
        
        Args:
            split_dir: Directory containing split files
            
        Returns:
            Dictionary with 'train', 'val', 'test' keys
        """
        split_dir = Path(split_dir)
        
        for split_name in ['train', 'val', 'test']:
            split_file = split_dir / f"{split_name}.txt"
            
            if split_file.exists():
                with open(split_file, 'r') as f:
                    self.dataset[split_name] = [line.strip() for line in f if line.strip()]
                
                logger.info(f"Loaded {split_name} split: {len(self.dataset[split_name])} images")
            else:
                logger.warning(f"Split file not found: {split_file}")
        
        # Update 'all' list
        self.dataset['all'] = (self.dataset['train'] + 
                               self.dataset['val'] + 
                               self.dataset['test'])
        
        return self.dataset
    
    def get_dataset_info(self) -> Dict[str, any]:
        """
        Get information about the current dataset.
        
        Returns:
            Dictionary containing dataset statistics
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
        """Clear the image cache."""
        self.image_cache.clear()
        self.cache_used_bytes = 0
        logger.info("Image cache cleared")
    
    def _add_to_cache(self, image_path: str, image: np.ndarray):
        """
        Add image to cache with LRU eviction.
        
        Args:
            image_path: Path to image
            image: Image array
        """
        # Calculate image size in bytes
        image_size = image.nbytes
        
        # Evict images if cache is full
        while (self.cache_used_bytes + image_size > self.cache_size_bytes 
               and len(self.image_cache) > 0):
            # Remove oldest entry (first in dict)
            oldest_path = next(iter(self.image_cache))
            oldest_size = self.image_cache[oldest_path][1]
            del self.image_cache[oldest_path]
            self.cache_used_bytes -= oldest_size
            logger.debug(f"Evicted from cache: {oldest_path}")
        
        # Add new image to cache
        self.image_cache[image_path] = (image.copy(), image_size)
        self.cache_used_bytes += image_size
        
        logger.debug(f"Added to cache: {image_path}, "
                    f"cache usage: {self.cache_used_bytes / (1024*1024):.1f} MB")
    
    def preload_images(self, image_paths: List[str], max_images: Optional[int] = None):
        """
        Preload images into cache.
        
        Args:
            image_paths: List of image paths to preload
            max_images: Maximum number of images to preload (default: None, all)
        """
        if max_images is not None:
            image_paths = image_paths[:max_images]
        
        loaded_count = 0
        for path in image_paths:
            if self.load_image(path, use_cache=True) is not None:
                loaded_count += 1
        
        logger.info(f"Preloaded {loaded_count} images into cache")
    
    def get_image_by_index(self, index: int, split: str = 'all') -> Optional[np.ndarray]:
        """
        Get image by index from a specific split.
        
        Args:
            index: Image index
            split: Dataset split ('all', 'train', 'val', 'test')
            
        Returns:
            Image array or None if index out of range
        """
        if split not in self.dataset:
            logger.error(f"Invalid split: {split}")
            return None
        
        if index < 0 or index >= len(self.dataset[split]):
            logger.error(f"Index {index} out of range for split {split}")
            return None
        
        image_path = self.dataset[split][index]
        return self.load_image(image_path)
    
    def get_batch(self, indices: List[int], split: str = 'all') -> List[np.ndarray]:
        """
        Get a batch of images by indices.
        
        Args:
            indices: List of image indices
            split: Dataset split ('all', 'train', 'val', 'test')
            
        Returns:
            List of image arrays
        """
        images = []
        for idx in indices:
            image = self.get_image_by_index(idx, split)
            if image is not None:
                images.append(image)
        
        return images
