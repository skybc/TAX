"""
SAM (Segment Anything Model) handler for auto-annotation.

This module provides:
- SAM model loading and initialization
- Image encoding
- Prompt-based mask prediction (points, boxes)
- Mask post-processing
"""

from pathlib import Path
from typing import Optional, List, Tuple, Dict
import numpy as np
import torch
from torch import nn

from src.logger import get_logger

logger = get_logger(__name__)


class SAMHandler:
    """
    Handler for Segment Anything Model (SAM).
    
    Provides functionality for:
    - Loading SAM model
    - Encoding images
    - Predicting masks from prompts (points, boxes)
    - Managing model state
    
    Attributes:
        model_type: SAM model type ('vit_h', 'vit_l', 'vit_b')
        device: Computation device ('cuda' or 'cpu')
        sam_model: SAM model instance
        image_encoder: Image encoder module
        prompt_encoder: Prompt encoder module
        mask_decoder: Mask decoder module
    """
    
    def __init__(self, 
                 model_type: str = 'vit_h',
                 checkpoint_path: Optional[str] = None,
                 device: Optional[str] = None):
        """
        Initialize SAM handler.
        
        Args:
            model_type: Model type ('vit_h', 'vit_l', 'vit_b')
            checkpoint_path: Path to model checkpoint
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Model components
        self.sam_model = None
        self.predictor = None
        
        # Encoded image features
        self.encoded_image = None
        self.current_image_shape = None
        
        logger.info(f"SAMHandler initialized - Model: {model_type}, Device: {self.device}")
    
    def load_model(self, checkpoint_path: Optional[str] = None) -> bool:
        """
        Load SAM model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            True if successful, False otherwise
        """
        if checkpoint_path is not None:
            self.checkpoint_path = checkpoint_path
        
        if self.checkpoint_path is None:
            logger.error("No checkpoint path provided")
            return False
        
        checkpoint_file = Path(self.checkpoint_path)
        if not checkpoint_file.exists():
            logger.error(f"Checkpoint file not found: {self.checkpoint_path}")
            return False
        
        try:
            # Import SAM (segment-anything package)
            try:
                from segment_anything import sam_model_registry, SamPredictor
            except ImportError:
                logger.error("segment-anything package not installed. Install with: pip install git+https://github.com/facebookresearch/segment-anything.git")
                return False
            
            # Load model
            logger.info(f"Loading SAM model from {self.checkpoint_path}...")
            self.sam_model = sam_model_registry[self.model_type](checkpoint=str(checkpoint_file))
            self.sam_model.to(self.device)
            self.sam_model.eval()
            
            # Create predictor
            self.predictor = SamPredictor(self.sam_model)
            
            logger.info(f"SAM model loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load SAM model: {e}", exc_info=True)
            return False
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.sam_model is not None and self.predictor is not None
    
    def encode_image(self, image: np.ndarray) -> bool:
        """
        Encode image for SAM prediction.
        
        This should be called once per image before making predictions.
        
        Args:
            image: Image as numpy array (HxWx3) in RGB format
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_loaded():
            logger.error("SAM model not loaded")
            return False
        
        try:
            # Set image in predictor (this will encode it)
            self.predictor.set_image(image)
            self.current_image_shape = image.shape[:2]
            
            logger.info(f"Image encoded successfully: {image.shape}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to encode image: {e}", exc_info=True)
            return False
    
    def predict_mask_from_points(self,
                                 points: List[Tuple[int, int]],
                                 labels: List[int],
                                 multimask_output: bool = True) -> Optional[Dict]:
        """
        Predict mask from point prompts.
        
        Args:
            points: List of (x, y) coordinates
            labels: List of labels (1 for foreground, 0 for background)
            multimask_output: Whether to return multiple masks
            
        Returns:
            Dictionary with 'masks', 'scores', 'logits' or None if failed
        """
        if not self.is_loaded():
            logger.error("SAM model not loaded")
            return None
        
        if self.current_image_shape is None:
            logger.error("No image encoded. Call encode_image() first")
            return None
        
        try:
            # Convert to numpy arrays
            point_coords = np.array(points, dtype=np.float32)
            point_labels = np.array(labels, dtype=np.int32)
            
            # Predict
            masks, scores, logits = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=multimask_output
            )
            
            logger.info(f"Predicted {len(masks)} mask(s) from {len(points)} point(s)")
            
            return {
                'masks': masks,
                'scores': scores,
                'logits': logits
            }
            
        except Exception as e:
            logger.error(f"Failed to predict mask from points: {e}", exc_info=True)
            return None
    
    def predict_mask_from_box(self,
                             box: Tuple[int, int, int, int],
                             multimask_output: bool = False) -> Optional[Dict]:
        """
        Predict mask from bounding box prompt.
        
        Args:
            box: Bounding box as (x1, y1, x2, y2)
            multimask_output: Whether to return multiple masks
            
        Returns:
            Dictionary with 'masks', 'scores', 'logits' or None if failed
        """
        if not self.is_loaded():
            logger.error("SAM model not loaded")
            return None
        
        if self.current_image_shape is None:
            logger.error("No image encoded. Call encode_image() first")
            return None
        
        try:
            # Convert to numpy array
            box_array = np.array(box, dtype=np.float32)
            
            # Predict
            masks, scores, logits = self.predictor.predict(
                box=box_array,
                multimask_output=multimask_output
            )
            
            logger.info(f"Predicted {len(masks)} mask(s) from bounding box")
            
            return {
                'masks': masks,
                'scores': scores,
                'logits': logits
            }
            
        except Exception as e:
            logger.error(f"Failed to predict mask from box: {e}", exc_info=True)
            return None
    
    def predict_mask_from_combined(self,
                                   points: Optional[List[Tuple[int, int]]] = None,
                                   labels: Optional[List[int]] = None,
                                   box: Optional[Tuple[int, int, int, int]] = None,
                                   multimask_output: bool = True) -> Optional[Dict]:
        """
        Predict mask from combined prompts (points and box).
        
        Args:
            points: List of (x, y) coordinates
            labels: List of labels (1 for foreground, 0 for background)
            box: Bounding box as (x1, y1, x2, y2)
            multimask_output: Whether to return multiple masks
            
        Returns:
            Dictionary with 'masks', 'scores', 'logits' or None if failed
        """
        if not self.is_loaded():
            logger.error("SAM model not loaded")
            return None
        
        if self.current_image_shape is None:
            logger.error("No image encoded. Call encode_image() first")
            return None
        
        try:
            # Prepare arguments
            point_coords = np.array(points, dtype=np.float32) if points else None
            point_labels = np.array(labels, dtype=np.int32) if labels else None
            box_array = np.array(box, dtype=np.float32) if box else None
            
            # Predict
            masks, scores, logits = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box_array,
                multimask_output=multimask_output
            )
            
            logger.info(f"Predicted {len(masks)} mask(s) from combined prompts")
            
            return {
                'masks': masks,
                'scores': scores,
                'logits': logits
            }
            
        except Exception as e:
            logger.error(f"Failed to predict mask from combined prompts: {e}", exc_info=True)
            return None
    
    def get_best_mask(self, prediction: Dict) -> Optional[np.ndarray]:
        """
        Get the best mask from prediction results.
        
        Args:
            prediction: Prediction dictionary with 'masks' and 'scores'
            
        Returns:
            Best mask as binary numpy array (HxW) or None
        """
        if prediction is None or 'masks' not in prediction or 'scores' not in prediction:
            return None
        
        masks = prediction['masks']
        scores = prediction['scores']
        
        if len(masks) == 0:
            return None
        
        # Get mask with highest score
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]
        
        # Convert to binary mask (0 or 255)
        binary_mask = (best_mask > 0.5).astype(np.uint8) * 255
        
        return binary_mask
    
    def post_process_mask(self, 
                         mask: np.ndarray,
                         remove_small: bool = True,
                         min_area: int = 100,
                         fill_holes: bool = True) -> np.ndarray:
        """
        Post-process mask to improve quality.
        
        Args:
            mask: Binary mask (HxW)
            remove_small: Whether to remove small components
            min_area: Minimum area for components
            fill_holes: Whether to fill holes
            
        Returns:
            Processed mask
        """
        from src.utils.mask_utils import (
            remove_small_components, fill_holes as fill_mask_holes
        )
        
        processed_mask = mask.copy()
        
        # Remove small components
        if remove_small:
            processed_mask = remove_small_components(processed_mask, min_area)
        
        # Fill holes
        if fill_holes:
            processed_mask = fill_mask_holes(processed_mask)
        
        return processed_mask
    
    def unload_model(self):
        """Unload model to free memory."""
        if self.sam_model is not None:
            del self.sam_model
            self.sam_model = None
        
        if self.predictor is not None:
            del self.predictor
            self.predictor = None
        
        self.encoded_image = None
        self.current_image_shape = None
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("SAM model unloaded")
    
    def get_model_info(self) -> Dict:
        """
        Get model information.
        
        Returns:
            Dictionary with model info
        """
        return {
            'model_type': self.model_type,
            'device': str(self.device),
            'checkpoint_path': self.checkpoint_path,
            'is_loaded': self.is_loaded(),
            'current_image_shape': self.current_image_shape
        }
