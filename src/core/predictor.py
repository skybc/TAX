"""
Predictor for segmentation inference.

This module provides:
- Model loading from checkpoint
- Single and batch prediction
- Post-processing integration
- Result saving
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Union, Tuple
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import cv2

from src.logger import get_logger
from src.models.segmentation_models import build_model
from src.utils.image_utils import load_image, save_image, resize_image
from src.utils.mask_utils import save_mask

logger = get_logger(__name__)


class Predictor:
    """
    Predictor for segmentation inference.
    
    Handles:
    - Loading trained models
    - Single image prediction
    - Batch prediction
    - Result visualization and saving
    """
    
    def __init__(self,
                 checkpoint_path: str,
                 device: Optional[torch.device] = None,
                 image_size: Tuple[int, int] = (512, 512)):
        """
        Initialize predictor.
        
        Args:
            checkpoint_path: Path to model checkpoint
            device: Device for inference (None for auto-detect)
            image_size: Input image size for model (H, W)
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device if device is not None else \
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_size = image_size
        
        # Model and config
        self.model: Optional[nn.Module] = None
        self.config: Dict = {}
        
        # Statistics
        self.num_predictions = 0
        
        logger.info(f"Predictor initialized: device={self.device}, image_size={image_size}")
    
    def load_model(self, architecture: str = 'unet', 
                   encoder: str = 'resnet34',
                   num_classes: int = 1) -> bool:
        """
        Load model from checkpoint.
        
        Args:
            architecture: Model architecture
            encoder: Encoder backbone
            num_classes: Number of output classes
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Build model
            self.model = build_model(
                architecture=architecture,
                encoder_name=encoder,
                encoder_weights=None,  # Load from checkpoint
                in_channels=3,
                num_classes=num_classes,
                activation='sigmoid'
            )
            
            # Load checkpoint
            if not self.checkpoint_path.exists():
                logger.error(f"Checkpoint not found: {self.checkpoint_path}")
                return False
            
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            
            # Load state dict
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                logger.info("Loaded model from checkpoint")
            else:
                self.model.load_state_dict(checkpoint)
                logger.info("Loaded model weights directly")
            
            # Move to device and set eval mode
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Store config
            self.config = {
                'architecture': architecture,
                'encoder': encoder,
                'num_classes': num_classes,
                'checkpoint': str(self.checkpoint_path)
            }
            
            logger.info(f"Model loaded: {architecture} with {encoder}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            return False
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for inference.
        
        Args:
            image: Input image (HxWxC) in RGB
            
        Returns:
            Preprocessed tensor (1xCxHxW)
        """
        # Store original size
        original_size = image.shape[:2]
        
        # Resize
        if image.shape[:2] != self.image_size:
            image = resize_image(image, self.image_size)
        
        # Normalize (ImageNet stats)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        image = image.astype(np.float32) / 255.0
        image = (image - mean) / std
        
        # Convert to tensor (HxWxC -> CxHxW)
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        
        # Add batch dimension
        image = image.unsqueeze(0)
        
        return image
    
    def postprocess_mask(self, mask: torch.Tensor, 
                        original_size: Tuple[int, int],
                        threshold: float = 0.5) -> np.ndarray:
        """
        Postprocess prediction mask.
        
        Args:
            mask: Predicted mask tensor (1xCxHxW)
            original_size: Original image size (H, W)
            threshold: Threshold for binarization
            
        Returns:
            Binary mask (HxW) in uint8
        """
        # Remove batch and channel dimensions
        mask = mask.squeeze().cpu().numpy()
        
        # Binarize
        mask = (mask > threshold).astype(np.uint8) * 255
        
        # Resize back to original size
        if mask.shape != original_size:
            mask = cv2.resize(mask, (original_size[1], original_size[0]), 
                            interpolation=cv2.INTER_NEAREST)
        
        return mask
    
    def predict(self, 
                image: Union[str, np.ndarray],
                threshold: float = 0.5,
                return_prob: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict mask for a single image.
        
        Args:
            image: Image path or numpy array (HxWxC)
            threshold: Threshold for binarization
            return_prob: Whether to return probability map
            
        Returns:
            Binary mask or (mask, prob_map) tuple
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Load image if path
        if isinstance(image, str):
            image = load_image(image)
            if image is None:
                raise ValueError(f"Failed to load image: {image}")
        
        original_size = image.shape[:2]
        
        # Preprocess
        input_tensor = self.preprocess_image(image).to(self.device)
        
        # Predict
        with torch.no_grad():
            output = self.model(input_tensor)
        
        # Postprocess
        mask = self.postprocess_mask(output, original_size, threshold)
        
        self.num_predictions += 1
        
        if return_prob:
            prob_map = output.squeeze().cpu().numpy()
            if prob_map.shape != original_size:
                prob_map = cv2.resize(prob_map, (original_size[1], original_size[0]))
            return mask, prob_map
        else:
            return mask
    
    def predict_batch(self,
                     image_paths: List[str],
                     output_dir: str,
                     threshold: float = 0.5,
                     save_overlay: bool = True,
                     progress_callback: Optional[callable] = None) -> Dict:
        """
        Predict masks for multiple images.
        
        Args:
            image_paths: List of image file paths
            output_dir: Directory to save predictions
            threshold: Threshold for binarization
            save_overlay: Whether to save overlay visualization
            progress_callback: Optional callback(current, total, image_path)
            
        Returns:
            Dictionary with prediction statistics
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        masks_dir = output_dir / "masks"
        masks_dir.mkdir(exist_ok=True)
        
        if save_overlay:
            overlay_dir = output_dir / "overlays"
            overlay_dir.mkdir(exist_ok=True)
        
        results = {
            'total': len(image_paths),
            'successful': 0,
            'failed': 0,
            'failed_files': []
        }
        
        for i, image_path in enumerate(image_paths):
            try:
                # Progress callback
                if progress_callback is not None:
                    progress_callback(i + 1, len(image_paths), image_path)
                
                # Load image
                image = load_image(image_path)
                if image is None:
                    results['failed'] += 1
                    results['failed_files'].append(image_path)
                    continue
                
                # Predict
                mask, prob_map = self.predict(image, threshold, return_prob=True)
                
                # Save mask
                image_name = Path(image_path).stem
                mask_path = masks_dir / f"{image_name}_mask.png"
                save_mask(mask, str(mask_path))
                
                # Save overlay
                if save_overlay:
                    overlay = self._create_overlay(image, mask)
                    overlay_path = overlay_dir / f"{image_name}_overlay.png"
                    save_image(overlay, str(overlay_path))
                
                results['successful'] += 1
                
            except Exception as e:
                logger.error(f"Failed to predict {image_path}: {e}")
                results['failed'] += 1
                results['failed_files'].append(image_path)
        
        logger.info(f"Batch prediction completed: {results['successful']}/{results['total']} successful")
        
        return results
    
    def _create_overlay(self, image: np.ndarray, mask: np.ndarray, 
                       alpha: float = 0.5, color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """
        Create overlay visualization.
        
        Args:
            image: Original image (HxWxC)
            mask: Binary mask (HxW)
            alpha: Transparency
            color: Overlay color (R, G, B)
            
        Returns:
            Overlay image (HxWxC)
        """
        # Ensure image is RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Create colored mask
        colored_mask = np.zeros_like(image)
        colored_mask[mask > 0] = color
        
        # Blend
        overlay = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)
        
        return overlay
    
    def predict_with_tta(self,
                        image: Union[str, np.ndarray],
                        threshold: float = 0.5,
                        num_augmentations: int = 4) -> np.ndarray:
        """
        Predict with Test-Time Augmentation (TTA).
        
        Args:
            image: Image path or numpy array
            threshold: Threshold for binarization
            num_augmentations: Number of augmentation variations
            
        Returns:
            Binary mask
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Load image if path
        if isinstance(image, str):
            image = load_image(image)
            if image is None:
                raise ValueError(f"Failed to load image: {image}")
        
        original_size = image.shape[:2]
        predictions = []
        
        # Original
        pred = self.predict(image, threshold=1.0, return_prob=True)[1]
        predictions.append(pred)
        
        # Horizontal flip
        if num_augmentations >= 2:
            flipped = cv2.flip(image, 1)
            pred = self.predict(flipped, threshold=1.0, return_prob=True)[1]
            pred = cv2.flip(pred, 1)
            predictions.append(pred)
        
        # Vertical flip
        if num_augmentations >= 3:
            flipped = cv2.flip(image, 0)
            pred = self.predict(flipped, threshold=1.0, return_prob=True)[1]
            pred = cv2.flip(pred, 0)
            predictions.append(pred)
        
        # Both flips
        if num_augmentations >= 4:
            flipped = cv2.flip(cv2.flip(image, 0), 1)
            pred = self.predict(flipped, threshold=1.0, return_prob=True)[1]
            pred = cv2.flip(cv2.flip(pred, 0), 1)
            predictions.append(pred)
        
        # Average predictions
        avg_pred = np.mean(predictions, axis=0)
        
        # Threshold
        mask = (avg_pred > threshold).astype(np.uint8) * 255
        
        return mask
    
    def get_model_info(self) -> Dict:
        """
        Get model information.
        
        Returns:
            Dictionary with model info
        """
        if self.model is None:
            return {'status': 'not_loaded'}
        
        info = {
            'status': 'loaded',
            'config': self.config,
            'device': str(self.device),
            'image_size': self.image_size,
            'num_predictions': self.num_predictions,
        }
        
        # Add parameter count
        if hasattr(self.model, 'get_model_info'):
            info.update(self.model.get_model_info())
        
        return info
    
    def reset_stats(self):
        """Reset prediction statistics."""
        self.num_predictions = 0
        logger.debug("Statistics reset")


def create_predictor(checkpoint_path: str,
                    architecture: str = 'unet',
                    encoder: str = 'resnet34',
                    device: Optional[str] = None,
                    image_size: Tuple[int, int] = (512, 512)) -> Predictor:
    """
    Create and initialize a predictor.
    
    Args:
        checkpoint_path: Path to model checkpoint
        architecture: Model architecture
        encoder: Encoder backbone
        device: Device ('cuda' or 'cpu', None for auto)
        image_size: Input image size
        
    Returns:
        Initialized Predictor
    """
    # Parse device
    if device is None:
        device_obj = None
    else:
        device_obj = torch.device(device)
    
    # Create predictor
    predictor = Predictor(
        checkpoint_path=checkpoint_path,
        device=device_obj,
        image_size=image_size
    )
    
    # Load model
    success = predictor.load_model(
        architecture=architecture,
        encoder=encoder
    )
    
    if not success:
        logger.error("Failed to create predictor")
        return None
    
    return predictor
