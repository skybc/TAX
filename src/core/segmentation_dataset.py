"""
Dataset loader for segmentation training.

This module provides:
- SegmentationDataset with augmentation
- Data loading utilities
- Train/val split handling
"""

import os
from pathlib import Path
from typing import Optional, Callable, List, Tuple, Dict
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.logger import get_logger
from src.utils.image_utils import load_image
from src.utils.mask_utils import load_mask

logger = get_logger(__name__)


class SegmentationDataset(Dataset):
    """
    Dataset for semantic segmentation.
    
    Loads images and corresponding masks, applies augmentations.
    """
    
    def __init__(self,
                 image_paths: List[str],
                 mask_paths: List[str],
                 transform: Optional[Callable] = None,
                 preprocessing: Optional[Callable] = None):
        """
        Initialize dataset.
        
        Args:
            image_paths: List of image file paths
            mask_paths: List of mask file paths
            transform: Augmentation transform
            preprocessing: Preprocessing function
        """
        assert len(image_paths) == len(mask_paths), \
            "Number of images and masks must match"
        
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.preprocessing = preprocessing
        
        logger.info(f"Created dataset with {len(image_paths)} samples")
    
    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get item by index.
        
        Args:
            idx: Sample index
            
        Returns:
            (image, mask) tuple as tensors
        """
        # Load image and mask
        image = load_image(self.image_paths[idx])
        mask = load_mask(self.mask_paths[idx])
        
        if image is None or mask is None:
            logger.error(f"Failed to load: {self.image_paths[idx]}")
            # Return dummy data
            return torch.zeros(3, 256, 256), torch.zeros(1, 256, 256)
        
        # Ensure mask is 2D
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        
        # Normalize mask to [0, 1]
        if mask.max() > 1:
            mask = mask / 255.0
        
        # Apply augmentation
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        # Apply preprocessing
        if self.preprocessing is not None:
            preprocessed = self.preprocessing(image=image, mask=mask)
            image = preprocessed['image']
            mask = preprocessed['mask']
        
        # Add channel dimension to mask if needed
        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=0)
        
        # Convert to tensors if not already
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).float()
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).float()
        
        return image, mask
    
    def get_sample_info(self, idx: int) -> Dict:
        """
        Get information about a sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with sample info
        """
        return {
            'image_path': self.image_paths[idx],
            'mask_path': self.mask_paths[idx],
            'image_name': Path(self.image_paths[idx]).name,
            'mask_name': Path(self.mask_paths[idx]).name
        }


def get_training_augmentation(image_size: Tuple[int, int] = (512, 512),
                              p: float = 0.5) -> A.Compose:
    """
    Get training augmentation pipeline.
    
    Args:
        image_size: Target image size (height, width)
        p: Probability of applying augmentation
        
    Returns:
        Albumentations Compose object
    """
    return A.Compose([
        # Geometric transforms
        A.Resize(*image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.2,
            rotate_limit=45,
            p=p
        ),
        
        # Color transforms
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=1.0
            ),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1.0
            ),
            A.RandomGamma(p=1.0),
        ], p=p),
        
        # Noise and blur
        A.OneOf([
            A.GaussNoise(p=1.0),
            A.GaussianBlur(p=1.0),
            A.MotionBlur(p=1.0),
        ], p=p * 0.5),
        
        # Normalize and convert to tensor
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ])


def get_validation_augmentation(image_size: Tuple[int, int] = (512, 512)) -> A.Compose:
    """
    Get validation augmentation pipeline (no random augmentation).
    
    Args:
        image_size: Target image size (height, width)
        
    Returns:
        Albumentations Compose object
    """
    return A.Compose([
        A.Resize(*image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ])


def get_preprocessing(preprocessing_fn: Optional[Callable] = None) -> A.Compose:
    """
    Get preprocessing pipeline.
    
    Args:
        preprocessing_fn: Preprocessing function from encoder
        
    Returns:
        Albumentations Compose object
    """
    transforms = []
    
    if preprocessing_fn is not None:
        transforms.append(A.Lambda(image=preprocessing_fn))
    
    return A.Compose(transforms) if transforms else None


def create_dataloaders(train_image_paths: List[str],
                       train_mask_paths: List[str],
                       val_image_paths: List[str],
                       val_mask_paths: List[str],
                       batch_size: int = 8,
                       num_workers: int = 4,
                       image_size: Tuple[int, int] = (512, 512),
                       augmentation_prob: float = 0.5) -> Tuple:
    """
    Create train and validation dataloaders.
    
    Args:
        train_image_paths: Training image paths
        train_mask_paths: Training mask paths
        val_image_paths: Validation image paths
        val_mask_paths: Validation mask paths
        batch_size: Batch size
        num_workers: Number of workers for data loading
        image_size: Target image size
        augmentation_prob: Probability of applying augmentation
        
    Returns:
        (train_loader, val_loader) tuple
    """
    # Create datasets
    train_dataset = SegmentationDataset(
        image_paths=train_image_paths,
        mask_paths=train_mask_paths,
        transform=get_training_augmentation(image_size, augmentation_prob)
    )
    
    val_dataset = SegmentationDataset(
        image_paths=val_image_paths,
        mask_paths=val_mask_paths,
        transform=get_validation_augmentation(image_size)
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    logger.info(f"Created dataloaders: train={len(train_dataset)}, val={len(val_dataset)}")
    logger.info(f"Batch size: {batch_size}, num_workers: {num_workers}")
    
    return train_loader, val_loader


def load_dataset_from_split_files(split_dir: str,
                                  images_dir: str,
                                  masks_dir: str) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Load dataset from split files (train.txt, val.txt).
    
    Args:
        split_dir: Directory containing split files
        images_dir: Directory containing images
        masks_dir: Directory containing masks
        
    Returns:
        (train_image_paths, train_mask_paths, val_image_paths, val_mask_paths)
    """
    split_dir = Path(split_dir)
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)
    
    # Load train split
    train_file = split_dir / "train.txt"
    if not train_file.exists():
        logger.error(f"Train split file not found: {train_file}")
        return [], [], [], []
    
    with open(train_file, 'r') as f:
        train_names = [line.strip() for line in f if line.strip()]
    
    # Load val split
    val_file = split_dir / "val.txt"
    if not val_file.exists():
        logger.error(f"Val split file not found: {val_file}")
        return [], [], [], []
    
    with open(val_file, 'r') as f:
        val_names = [line.strip() for line in f if line.strip()]
    
    # Build paths
    train_image_paths = [str(images_dir / name) for name in train_names]
    train_mask_paths = [str(masks_dir / Path(name).stem) + ".png" for name in train_names]
    
    val_image_paths = [str(images_dir / name) for name in val_names]
    val_mask_paths = [str(masks_dir / Path(name).stem) + ".png" for name in val_names]
    
    # Filter existing files
    train_image_paths, train_mask_paths = _filter_existing_pairs(train_image_paths, train_mask_paths)
    val_image_paths, val_mask_paths = _filter_existing_pairs(val_image_paths, val_mask_paths)
    
    logger.info(f"Loaded dataset from splits: train={len(train_image_paths)}, val={len(val_image_paths)}")
    
    return train_image_paths, train_mask_paths, val_image_paths, val_mask_paths


def _filter_existing_pairs(image_paths: List[str], mask_paths: List[str]) -> Tuple[List[str], List[str]]:
    """
    Filter out non-existing image-mask pairs.
    
    Args:
        image_paths: List of image paths
        mask_paths: List of mask paths
        
    Returns:
        (filtered_image_paths, filtered_mask_paths)
    """
    filtered_images = []
    filtered_masks = []
    
    for img_path, mask_path in zip(image_paths, mask_paths):
        if Path(img_path).exists() and Path(mask_path).exists():
            filtered_images.append(img_path)
            filtered_masks.append(mask_path)
        else:
            logger.warning(f"Skipping missing pair: {img_path} or {mask_path}")
    
    return filtered_images, filtered_masks


def compute_class_weights(mask_paths: List[str],
                          num_classes: int = 2) -> torch.Tensor:
    """
    Compute class weights for imbalanced datasets.
    
    Args:
        mask_paths: List of mask file paths
        num_classes: Number of classes
        
    Returns:
        Class weights tensor
    """
    class_counts = np.zeros(num_classes)
    
    for mask_path in mask_paths:
        mask = load_mask(mask_path)
        if mask is None:
            continue
        
        # Binarize mask
        mask_binary = (mask > 0).astype(np.uint8)
        
        # Count pixels
        class_counts[0] += np.sum(mask_binary == 0)  # Background
        class_counts[1] += np.sum(mask_binary == 1)  # Foreground
    
    # Compute weights (inverse frequency)
    total_pixels = class_counts.sum()
    class_weights = total_pixels / (num_classes * class_counts + 1e-6)
    
    # Normalize
    class_weights = class_weights / class_weights.sum() * num_classes
    
    logger.info(f"Class weights computed: {class_weights}")
    
    return torch.from_numpy(class_weights).float()
