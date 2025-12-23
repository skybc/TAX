# Industrial Defect Segmentation System - API Reference

**Version**: 1.0.0  
**Last Updated**: December 23, 2025

---

## Table of Contents

1. [Overview](#overview)
2. [Core Modules](#core-modules)
3. [Model Modules](#model-modules)
4. [Utility Modules](#utility-modules)
5. [Thread Modules](#thread-modules)
6. [Usage Examples](#usage-examples)

---

## Overview

This document provides comprehensive API documentation for all public classes and functions in the Industrial Defect Segmentation System.

### Module Organization

```
src/
├── core/           # Business logic modules
├── models/         # Model architectures and training
├── utils/          # Utility functions
├── threads/        # Asynchronous operations
└── ui/             # User interface components
```

### Import Convention

```python
# Core modules
from src.core.data_manager import DataManager
from src.core.annotation_manager import AnnotationManager
from src.core.sam_handler import SAMHandler
from src.core.model_trainer import ModelTrainer
from src.core.predictor import Predictor

# Model modules
from src.models.segmentation_models import SegmentationModel
from src.models.losses import DiceLoss, FocalLoss, CombinedLoss
from src.models.metrics import compute_iou, compute_dice, PixelAccuracy

# Utility modules
from src.utils.mask_utils import (
    binary_mask_to_rle, mask_to_polygon, mask_to_bbox
)
from src.utils.export_utils import export_to_coco, export_to_yolo
from src.utils.statistics import DefectStatistics
from src.utils.report_generator import ReportGenerator

# Thread modules
from src.threads.sam_inference_thread import SAMInferenceThread
from src.threads.training_thread import TrainingThread
from src.threads.inference_thread import InferenceThread
```

---

## Core Modules

### DataManager

**Module**: `src.core.data_manager`

Centralized data loading and management with LRU caching.

#### Class: DataManager

```python
class DataManager:
    """
    Manages data loading, caching, and dataset organization.
    
    Attributes:
        data_root (str): Root directory for data
        dataset (Dict): Dataset splits (train/val/test)
        cache_info (Dict): Cache statistics
    """
```

**Constructor**:
```python
def __init__(self, data_root: str, cache_size_mb: int = 1024):
    """
    Initialize DataManager.
    
    Args:
        data_root: Root directory containing data
        cache_size_mb: Maximum cache size in MB (default: 1024)
    
    Example:
        >>> dm = DataManager("data", cache_size_mb=2048)
    """
```

**Methods**:

##### load_image
```python
def load_image(self, image_path: str) -> Optional[np.ndarray]:
    """
    Load image from path with caching.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Image as RGB numpy array (H, W, 3) or None if failed
        
    Example:
        >>> image = dm.load_image("data/raw/image_001.jpg")
        >>> print(image.shape)  # (1024, 1024, 3)
    """
```

##### load_batch_images
```python
def load_batch_images(self, image_paths: List[str]) -> List[np.ndarray]:
    """
    Load multiple images as batch.
    
    Args:
        image_paths: List of image file paths
        
    Returns:
        List of images as numpy arrays
        
    Example:
        >>> images = dm.load_batch_images(image_paths[:10])
        >>> len(images)  # 10
    """
```

##### load_video
```python
def load_video(self, video_path: str, frame_interval: int = 10, 
               max_frames: Optional[int] = None) -> List[np.ndarray]:
    """
    Extract frames from video.
    
    Args:
        video_path: Path to video file
        frame_interval: Extract every N frames (default: 10)
        max_frames: Maximum frames to extract (default: None)
        
    Returns:
        List of frames as numpy arrays
        
    Example:
        >>> frames = dm.load_video("video.mp4", frame_interval=30)
        >>> print(f"Extracted {len(frames)} frames")
    """
```

##### create_splits
```python
def create_splits(self, image_paths: List[str], 
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15,
                 random_state: int = 42) -> Dict[str, List[str]]:
    """
    Split dataset into train/val/test sets.
    
    Args:
        image_paths: List of all image paths
        train_ratio: Training set ratio (default: 0.7)
        val_ratio: Validation set ratio (default: 0.15)
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with keys 'train', 'val', 'test'
        
    Example:
        >>> splits = dm.create_splits(all_images)
        >>> print(f"Train: {len(splits['train'])}")
        >>> print(f"Val: {len(splits['val'])}")
        >>> print(f"Test: {len(splits['test'])}")
    """
```

##### get_cache_info
```python
def get_cache_info(self) -> Dict[str, Any]:
    """
    Get cache statistics.
    
    Returns:
        Dictionary with cache hits, misses, size, etc.
        
    Example:
        >>> info = dm.get_cache_info()
        >>> print(f"Hit rate: {info['hit_rate']:.2%}")
    """
```

---

### AnnotationManager

**Module**: `src.core.annotation_manager`

Manages mask annotations with undo/redo support and export capabilities.

#### Class: AnnotationManager

```python
class AnnotationManager:
    """
    Manages annotation state, history, and export.
    
    Attributes:
        image_path (str): Current image path
        image_shape (Tuple): Current image shape (H, W)
        current_mask (np.ndarray): Current mask
        history (List): Undo/redo history
        metadata (Dict): Annotation metadata
    """
```

**Constructor**:
```python
def __init__(self, max_history: int = 50):
    """
    Initialize AnnotationManager.
    
    Args:
        max_history: Maximum undo/redo states (default: 50)
    
    Example:
        >>> am = AnnotationManager(max_history=100)
    """
```

**Methods**:

##### set_image
```python
def set_image(self, image_path: str, image_shape: Tuple[int, int]):
    """
    Set the image to annotate.
    
    Args:
        image_path: Path to image file
        image_shape: Image shape (H, W) or (H, W, C)
        
    Example:
        >>> am.set_image("image.jpg", (1024, 1024))
    """
```

##### set_mask
```python
def set_mask(self, mask: np.ndarray):
    """
    Set current mask (replaces existing).
    
    Args:
        mask: Binary mask (H, W)
        
    Example:
        >>> sam_mask = sam_handler.predict_mask(...)
        >>> am.set_mask(sam_mask)
    """
```

##### update_mask
```python
def update_mask(self, mask: np.ndarray, operation: str = 'replace'):
    """
    Update mask with operation.
    
    Args:
        mask: Mask to apply
        operation: One of 'replace', 'add', 'subtract', 'intersect'
        
    Example:
        >>> # Add new region to existing mask
        >>> am.update_mask(new_region, operation='add')
    """
```

##### paint_mask
```python
def paint_mask(self, points: List[Tuple[int, int]], brush_size: int, 
               value: int = 255, operation: str = 'paint'):
    """
    Paint mask at points with brush.
    
    Args:
        points: List of (x, y) coordinates
        brush_size: Brush radius in pixels
        value: Pixel value (0-255)
        operation: 'paint' or 'erase'
        
    Example:
        >>> # Paint a stroke
        >>> am.paint_mask([(100, 100), (101, 101), (102, 102)], 
        ...               brush_size=5)
    """
```

##### undo / redo
```python
def undo(self) -> bool:
    """Undo last operation. Returns True if successful."""
    
def redo(self) -> bool:
    """Redo undone operation. Returns True if successful."""
    
def can_undo(self) -> bool:
    """Check if undo is available."""
    
def can_redo(self) -> bool:
    """Check if redo is available."""
```

##### export_coco_annotation
```python
def export_coco_annotation(self) -> Dict:
    """
    Export annotation to COCO format.
    
    Returns:
        COCO annotation dictionary with keys:
        - id, image_id, category_id, bbox, area, segmentation, iscrowd
        
    Example:
        >>> annotation = am.export_coco_annotation()
        >>> print(annotation['bbox'])  # [x, y, width, height]
    """
```

##### export_yolo_annotation
```python
def export_yolo_annotation(self, class_id: int = 0) -> List[str]:
    """
    Export annotation to YOLO format.
    
    Args:
        class_id: Class ID for YOLO format
        
    Returns:
        List of YOLO annotation strings (normalized coordinates)
        
    Example:
        >>> yolo_lines = am.export_yolo_annotation(class_id=0)
        >>> for line in yolo_lines:
        ...     print(line)  # "0 0.123 0.456 0.234 0.567 ..."
    """
```

---

### SAMHandler

**Module**: `src.core.sam_handler`

Interface to Segment Anything Model for interactive segmentation.

#### Class: SAMHandler

```python
class SAMHandler:
    """
    Handler for Segment Anything Model (SAM).
    
    Attributes:
        model_type (str): SAM model type ('vit_h', 'vit_l', 'vit_b')
        device (str): Computation device ('cuda' or 'cpu')
        model: SAM model instance
        predictor: SAM predictor instance
    """
```

**Constructor**:
```python
def __init__(self, model_type: str = 'vit_h', 
             checkpoint_path: Optional[str] = None,
             device: str = 'cuda'):
    """
    Initialize SAM handler.
    
    Args:
        model_type: Model type ('vit_h', 'vit_l', 'vit_b')
        checkpoint_path: Path to model checkpoint
        device: Device for computation ('cuda' or 'cpu')
        
    Example:
        >>> sam = SAMHandler(model_type='vit_h', device='cuda')
    """
```

**Methods**:

##### load_model
```python
def load_model(self) -> bool:
    """
    Load SAM model (lazy loading).
    
    Returns:
        True if successful, False otherwise
        
    Example:
        >>> if sam.load_model():
        ...     print("Model loaded successfully")
    """
```

##### encode_image
```python
def encode_image(self, image: np.ndarray) -> bool:
    """
    Encode image for SAM prediction (required before prediction).
    
    Args:
        image: RGB image (H, W, 3)
        
    Returns:
        True if successful, False otherwise
        
    Example:
        >>> image = cv2.imread("image.jpg")
        >>> image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        >>> sam.encode_image(image)
    """
```

##### predict_mask_from_points
```python
def predict_mask_from_points(self, points: List[Tuple[int, int]], 
                            labels: List[int],
                            multimask_output: bool = True) -> Optional[Dict]:
    """
    Predict mask from point prompts.
    
    Args:
        points: List of (x, y) coordinates
        labels: List of labels (1=foreground, 0=background)
        multimask_output: Whether to output multiple masks
        
    Returns:
        Dictionary with keys:
        - masks: np.ndarray (N, H, W) - N masks
        - scores: np.ndarray (N,) - Quality scores
        - logits: np.ndarray (N, H, W) - Raw logits
        
    Example:
        >>> # Click on defect center
        >>> prediction = sam.predict_mask_from_points(
        ...     points=[(512, 512)],
        ...     labels=[1]
        ... )
        >>> best_mask = prediction['masks'][0]
    """
```

##### predict_mask_from_box
```python
def predict_mask_from_box(self, box: Tuple[int, int, int, int],
                         multimask_output: bool = False) -> Optional[Dict]:
    """
    Predict mask from bounding box prompt.
    
    Args:
        box: Bounding box (x_min, y_min, x_max, y_max)
        multimask_output: Whether to output multiple masks
        
    Returns:
        Dictionary with masks, scores, logits
        
    Example:
        >>> # Draw bounding box around defect
        >>> prediction = sam.predict_mask_from_box(
        ...     box=(100, 100, 300, 300)
        ... )
    """
```

##### predict_mask_from_combined
```python
def predict_mask_from_combined(self, 
                              points: Optional[List[Tuple[int, int]]] = None,
                              labels: Optional[List[int]] = None,
                              box: Optional[Tuple[int, int, int, int]] = None,
                              multimask_output: bool = True) -> Optional[Dict]:
    """
    Predict mask from combined prompts (points + box).
    
    Args:
        points: List of point coordinates (optional)
        labels: List of point labels (optional)
        box: Bounding box (optional)
        multimask_output: Whether to output multiple masks
        
    Returns:
        Dictionary with masks, scores, logits
        
    Example:
        >>> # Combine box and refinement points
        >>> prediction = sam.predict_mask_from_combined(
        ...     points=[(200, 200)],
        ...     labels=[1],
        ...     box=(100, 100, 300, 300)
        ... )
    """
```

##### get_best_mask
```python
def get_best_mask(self, prediction: Dict, 
                 threshold: float = 0.0) -> Optional[np.ndarray]:
    """
    Select best mask from multi-mask output.
    
    Args:
        prediction: Prediction dictionary from predict_mask_*
        threshold: Minimum quality score threshold
        
    Returns:
        Best mask as binary array (H, W)
        
    Example:
        >>> prediction = sam.predict_mask_from_points(...)
        >>> best_mask = sam.get_best_mask(prediction)
    """
```

##### post_process_mask
```python
def post_process_mask(self, mask: np.ndarray, 
                     min_area: int = 100,
                     remove_small_holes: bool = True) -> np.ndarray:
    """
    Post-process mask (remove small regions, fill holes).
    
    Args:
        mask: Binary mask
        min_area: Minimum component area to keep
        remove_small_holes: Whether to fill small holes
        
    Returns:
        Processed mask
        
    Example:
        >>> clean_mask = sam.post_process_mask(raw_mask, min_area=500)
    """
```

---

### ModelTrainer

**Module**: `src.core.model_trainer`

Orchestrates model training with callbacks and checkpointing.

#### Class: ModelTrainer

```python
class ModelTrainer:
    """
    Handles model training, validation, and checkpointing.
    
    Attributes:
        model: Segmentation model
        config (Dict): Training configuration
        optimizer: Optimizer instance
        loss_fn: Loss function
        device (str): Computation device
    """
```

**Constructor**:
```python
def __init__(self, model, config: Dict):
    """
    Initialize trainer.
    
    Args:
        model: Segmentation model (nn.Module)
        config: Training configuration dict with keys:
            - epochs: Number of epochs
            - lr: Learning rate
            - device: 'cuda' or 'cpu'
            - loss: Loss function name
            - optimizer: Optimizer name
            - checkpoint_dir: Directory for checkpoints
            
    Example:
        >>> model = SegmentationModel('unet', 'resnet34')
        >>> trainer = ModelTrainer(model, {
        ...     'epochs': 50,
        ...     'lr': 1e-4,
        ...     'device': 'cuda',
        ...     'loss': 'dice',
        ...     'checkpoint_dir': 'outputs/models'
        ... })
    """
```

**Methods**:

##### train
```python
def train(self, train_loader, val_loader, 
         callbacks: Optional[List] = None):
    """
    Main training loop.
    
    Args:
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        callbacks: List of callback functions (optional)
        
    Example:
        >>> trainer.train(train_loader, val_loader)
        Epoch 1/50 - Loss: 0.234, Val IoU: 0.823
        ...
    """
```

##### train_epoch
```python
def train_epoch(self, data_loader) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Args:
        data_loader: Training DataLoader
        
    Returns:
        Dictionary with metrics:
        - loss: Average training loss
        - iou: Average IoU
        - dice: Average Dice coefficient
        
    Example:
        >>> metrics = trainer.train_epoch(train_loader)
        >>> print(f"Loss: {metrics['loss']:.4f}")
    """
```

##### validate
```python
def validate(self, data_loader) -> Dict[str, float]:
    """
    Validate model.
    
    Args:
        data_loader: Validation DataLoader
        
    Returns:
        Dictionary with validation metrics
        
    Example:
        >>> val_metrics = trainer.validate(val_loader)
        >>> print(f"Val IoU: {val_metrics['iou']:.4f}")
    """
```

##### save_checkpoint
```python
def save_checkpoint(self, path: str, is_best: bool = False, 
                   **extra_state):
    """
    Save model checkpoint.
    
    Args:
        path: Save path
        is_best: Whether this is the best model
        **extra_state: Additional state to save
        
    Example:
        >>> trainer.save_checkpoint(
        ...     'outputs/models/checkpoint_epoch10.pth',
        ...     is_best=True,
        ...     epoch=10,
        ...     optimizer_state=optimizer.state_dict()
        ... )
    """
```

##### load_checkpoint
```python
def load_checkpoint(self, path: str) -> Dict:
    """
    Load model checkpoint.
    
    Args:
        path: Checkpoint path
        
    Returns:
        Checkpoint dictionary
        
    Example:
        >>> checkpoint = trainer.load_checkpoint('model_best.pth')
        >>> print(f"Loaded epoch {checkpoint['epoch']}")
    """
```

---

### Predictor

**Module**: `src.core.predictor`

Inference engine for trained models.

#### Class: Predictor

```python
class Predictor:
    """
    Handles batch inference with trained models.
    
    Attributes:
        model: Trained segmentation model
        device (str): Computation device
        config (Dict): Prediction configuration
    """
```

**Constructor**:
```python
def __init__(self, model_path: str, device: str = 'cuda', 
             config: Optional[Dict] = None):
    """
    Initialize predictor.
    
    Args:
        model_path: Path to model checkpoint
        device: Computation device
        config: Prediction configuration (optional)
        
    Example:
        >>> predictor = Predictor(
        ...     'outputs/models/model_best.pth',
        ...     device='cuda'
        ... )
    """
```

**Methods**:

##### predict
```python
def predict(self, image: np.ndarray, 
           threshold: float = 0.5,
           post_process: bool = True) -> np.ndarray:
    """
    Predict mask for single image.
    
    Args:
        image: Input image (H, W, 3)
        threshold: Probability threshold (0.0-1.0)
        post_process: Whether to apply post-processing
        
    Returns:
        Binary mask (H, W)
        
    Example:
        >>> mask = predictor.predict(image, threshold=0.6)
    """
```

##### predict_batch
```python
def predict_batch(self, images: List[np.ndarray], 
                 batch_size: int = 8,
                 threshold: float = 0.5,
                 tta: bool = False) -> List[np.ndarray]:
    """
    Predict masks for batch of images.
    
    Args:
        images: List of images
        batch_size: Batch size for inference
        threshold: Probability threshold
        tta: Whether to use test-time augmentation
        
    Returns:
        List of binary masks
        
    Example:
        >>> masks = predictor.predict_batch(
        ...     images,
        ...     batch_size=16,
        ...     tta=True
        ... )
    """
```

##### predict_with_tta
```python
def predict_with_tta(self, image: np.ndarray, 
                    threshold: float = 0.5) -> np.ndarray:
    """
    Predict with test-time augmentation.
    
    Args:
        image: Input image
        threshold: Probability threshold
        
    Returns:
        Binary mask
        
    Example:
        >>> # More robust prediction with TTA
        >>> mask = predictor.predict_with_tta(image)
    """
```

---

## Model Modules

### SegmentationModel

**Module**: `src.models.segmentation_models`

Wrapper for segmentation model architectures.

#### Class: SegmentationModel

```python
class SegmentationModel(nn.Module):
    """
    Segmentation model wrapper.
    
    Supports: U-Net, DeepLabV3+, FPN, PSPNet
    """
```

**Constructor**:
```python
def __init__(self, architecture: str, encoder_name: str,
             in_channels: int = 3, num_classes: int = 1,
             encoder_weights: str = 'imagenet'):
    """
    Initialize segmentation model.
    
    Args:
        architecture: Architecture name ('unet', 'deeplabv3plus', 'fpn')
        encoder_name: Encoder backbone ('resnet34', 'resnet50', etc.)
        in_channels: Input channels (default: 3 for RGB)
        num_classes: Number of output classes (default: 1)
        encoder_weights: Pretrained weights ('imagenet' or None)
        
    Example:
        >>> model = SegmentationModel(
        ...     architecture='unet',
        ...     encoder_name='resnet34',
        ...     encoder_weights='imagenet'
        ... )
    """
```

**Static Methods**:

```python
@staticmethod
def get_available_architectures() -> List[str]:
    """Get list of available architectures."""
    
@staticmethod
def get_available_encoders(architecture: str) -> List[str]:
    """Get list of available encoders for architecture."""
    
# Example usage:
>>> SegmentationModel.get_available_architectures()
['unet', 'deeplabv3plus', 'fpn', 'pspnet']

>>> SegmentationModel.get_available_encoders('unet')
['resnet18', 'resnet34', 'resnet50', 'efficientnet-b0', ...]
```

---

### Loss Functions

**Module**: `src.models.losses`

#### DiceLoss

```python
class DiceLoss(nn.Module):
    """
    Dice loss for segmentation.
    
    Formula: 1 - (2 * TP) / (2 * TP + FP + FN)
    """
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Dice loss.
        
        Args:
            pred: Predictions (B, C, H, W)
            target: Ground truth (B, C, H, W)
            
        Returns:
            Loss value
        """
```

#### FocalLoss

```python
class FocalLoss(nn.Module):
    """
    Focal loss for handling class imbalance.
    
    Formula: -α * (1 - p)^γ * log(p)
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor (default: 0.25)
            gamma: Focusing parameter (default: 2.0)
        """
```

#### CombinedLoss

```python
class CombinedLoss(nn.Module):
    """
    Combination of multiple losses.
    
    Example: 0.5 * DiceLoss + 0.5 * FocalLoss
    """
    
    def __init__(self, losses: List[nn.Module], weights: List[float]):
        """
        Initialize combined loss.
        
        Args:
            losses: List of loss modules
            weights: List of weights for each loss
            
        Example:
            >>> loss = CombinedLoss(
            ...     losses=[DiceLoss(), FocalLoss()],
            ...     weights=[0.5, 0.5]
            ... )
        """
```

---

### Metrics

**Module**: `src.models.metrics`

#### IoU (Intersection over Union)

```python
def compute_iou(pred: torch.Tensor, target: torch.Tensor, 
               threshold: float = 0.5) -> float:
    """
    Compute IoU metric.
    
    Args:
        pred: Predictions (B, C, H, W) or (B, H, W)
        target: Ground truth (B, C, H, W) or (B, H, W)
        threshold: Binarization threshold
        
    Returns:
        IoU score (0.0-1.0)
        
    Example:
        >>> iou = compute_iou(pred_masks, true_masks)
        >>> print(f"IoU: {iou:.4f}")
    """
```

#### Dice Coefficient

```python
def compute_dice(pred: torch.Tensor, target: torch.Tensor,
                threshold: float = 0.5) -> float:
    """
    Compute Dice coefficient.
    
    Formula: 2 * TP / (2 * TP + FP + FN)
    
    Args:
        pred: Predictions
        target: Ground truth
        threshold: Binarization threshold
        
    Returns:
        Dice score (0.0-1.0)
    """
```

#### Pixel Accuracy

```python
class PixelAccuracy:
    """
    Pixel-wise accuracy metric.
    """
    
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """
        Compute pixel accuracy.
        
        Returns:
            Accuracy (0.0-1.0)
        """
```

---

## Utility Modules

### Mask Utilities

**Module**: `src.utils.mask_utils`

#### Binary Mask to RLE

```python
def binary_mask_to_rle(mask: np.ndarray) -> Dict[str, Any]:
    """
    Convert binary mask to RLE (Run-Length Encoding).
    
    Args:
        mask: Binary mask (H, W)
        
    Returns:
        RLE dictionary with 'counts' and 'size'
        
    Example:
        >>> rle = binary_mask_to_rle(mask)
        >>> print(rle)  # {'counts': [...], 'size': [H, W]}
    """
```

#### RLE to Binary Mask

```python
def rle_to_binary_mask(rle: Dict[str, Any]) -> np.ndarray:
    """
    Convert RLE to binary mask.
    
    Args:
        rle: RLE dictionary
        
    Returns:
        Binary mask (H, W)
        
    Example:
        >>> mask = rle_to_binary_mask(rle)
    """
```

#### Mask to Polygon

```python
def mask_to_polygon(mask: np.ndarray, 
                   epsilon: float = 1.0) -> List[List[int]]:
    """
    Convert mask to polygon coordinates.
    
    Args:
        mask: Binary mask
        epsilon: Approximation accuracy (lower = more points)
        
    Returns:
        List of polygons, each as [x1, y1, x2, y2, ...]
        
    Example:
        >>> polygons = mask_to_polygon(mask)
        >>> for poly in polygons:
        ...     print(f"Polygon with {len(poly)//2} vertices")
    """
```

#### Mask to Bounding Box

```python
def mask_to_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Extract bounding box from mask.
    
    Args:
        mask: Binary mask
        
    Returns:
        Bounding box (x, y, width, height) or None if empty
        
    Example:
        >>> bbox = mask_to_bbox(mask)
        >>> x, y, w, h = bbox
    """
```

#### Compute Mask IoU

```python
def compute_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Compute IoU between two masks.
    
    Args:
        mask1: First binary mask
        mask2: Second binary mask
        
    Returns:
        IoU score (0.0-1.0)
        
    Example:
        >>> iou = compute_mask_iou(pred_mask, gt_mask)
        >>> print(f"IoU: {iou:.2%}")
    """
```

---

### Export Utilities

**Module**: `src.utils.export_utils`

#### Export to COCO

```python
def export_to_coco(image_paths: List[str],
                  mask_paths: List[str],
                  output_path: str,
                  category_name: str = 'defect',
                  category_id: int = 1) -> bool:
    """
    Export annotations to COCO format.
    
    Args:
        image_paths: List of image file paths
        mask_paths: List of corresponding mask paths
        output_path: Output JSON file path
        category_name: Category name
        category_id: Category ID
        
    Returns:
        True if successful
        
    Example:
        >>> export_to_coco(
        ...     image_paths=train_images,
        ...     mask_paths=train_masks,
        ...     output_path='annotations.json',
        ...     category_name='scratch'
        ... )
    """
```

#### Export to YOLO

```python
def export_to_yolo(image_paths: List[str],
                  mask_paths: List[str],
                  output_dir: str,
                  class_id: int = 0) -> bool:
    """
    Export annotations to YOLO format.
    
    Args:
        image_paths: List of image file paths
        mask_paths: List of corresponding mask paths
        output_dir: Output directory for txt files
        class_id: Class ID (default: 0)
        
    Returns:
        True if successful
        
    Example:
        >>> export_to_yolo(
        ...     image_paths=all_images,
        ...     mask_paths=all_masks,
        ...     output_dir='yolo_annotations',
        ...     class_id=0
        ... )
    """
```

#### Export to VOC

```python
def export_to_voc(image_paths: List[str],
                 mask_paths: List[str],
                 output_dir: str,
                 class_names: List[str] = ['background', 'defect']) -> bool:
    """
    Export annotations to PASCAL VOC format.
    
    Args:
        image_paths: List of image file paths
        mask_paths: List of corresponding mask paths
        output_dir: Output directory
        class_names: List of class names
        
    Returns:
        True if successful
        
    Example:
        >>> export_to_voc(
        ...     image_paths=images,
        ...     mask_paths=masks,
        ...     output_dir='VOCdevkit/VOC2012',
        ...     class_names=['background', 'scratch', 'dent']
        ... )
    """
```

---

### Statistics

**Module**: `src.utils.statistics`

#### DefectStatistics

```python
class DefectStatistics:
    """
    Compute defect statistics from masks.
    """
    
    def compute_mask_statistics(self, mask: np.ndarray) -> Dict[str, Any]:
        """
        Compute statistics for single mask.
        
        Args:
            mask: Binary mask
            
        Returns:
            Dictionary with:
            - defect_count: Number of defects
            - total_area: Total defect area
            - average_area: Average defect area
            - areas: List of individual defect areas
            - bboxes: List of bounding boxes
            
        Example:
            >>> stats_calc = DefectStatistics()
            >>> stats = stats_calc.compute_mask_statistics(mask)
            >>> print(f"Found {stats['defect_count']} defects")
        """
    
    def compute_batch_statistics(self, mask_paths: List[str]) -> Dict[str, Any]:
        """
        Compute statistics for batch of masks.
        
        Args:
            mask_paths: List of mask file paths
            
        Returns:
            Aggregated statistics dictionary
            
        Example:
            >>> batch_stats = stats_calc.compute_batch_statistics(all_masks)
            >>> print(f"Total defects: {batch_stats['total_defects']}")
        """
```

---

### Report Generator

**Module**: `src.utils.report_generator`

#### ReportGenerator

```python
class ReportGenerator:
    """
    Generate analysis reports in multiple formats.
    """
    
    def __init__(self, output_dir: str):
        """
        Initialize report generator.
        
        Args:
            output_dir: Directory for output reports
        """
    
    def generate_excel_report(self, data: Dict, 
                             output_filename: str) -> str:
        """
        Generate Excel report.
        
        Args:
            data: Report data dictionary
            output_filename: Output filename
            
        Returns:
            Path to generated report
            
        Example:
            >>> generator = ReportGenerator('outputs/reports')
            >>> report_path = generator.generate_excel_report(
            ...     data=analysis_results,
            ...     output_filename='defect_report.xlsx'
            ... )
        """
    
    def generate_pdf_report(self, data: Dict, 
                           output_filename: str) -> str:
        """
        Generate PDF report.
        
        Args:
            data: Report data dictionary
            output_filename: Output filename
            
        Returns:
            Path to generated report
        """
    
    def generate_html_report(self, data: Dict, 
                            output_filename: str) -> str:
        """
        Generate HTML report.
        
        Args:
            data: Report data dictionary
            output_filename: Output filename
            
        Returns:
            Path to generated report
        """
```

---

## Thread Modules

### SAMInferenceThread

**Module**: `src.threads.sam_inference_thread`

```python
class SAMInferenceThread(QThread):
    """
    Thread for asynchronous SAM inference.
    
    Signals:
        progress_updated(int, str): Progress percentage and message
        inference_completed(dict): Inference result
        inference_failed(str): Error message
    """
    
    def __init__(self, sam_handler: SAMHandler, image: np.ndarray,
                 prompt_type: str, prompt_data: Dict):
        """
        Initialize SAM inference thread.
        
        Args:
            sam_handler: SAMHandler instance
            image: Image to process
            prompt_type: 'points', 'box', or 'combined'
            prompt_data: Prompt configuration
            
        Example:
            >>> thread = SAMInferenceThread(
            ...     sam_handler=sam,
            ...     image=image,
            ...     prompt_type='points',
            ...     prompt_data={'points': [(512, 512)], 'labels': [1]}
            ... )
            >>> thread.inference_completed.connect(on_complete)
            >>> thread.start()
        """
```

### TrainingThread

**Module**: `src.threads.training_thread`

```python
class TrainingThread(QThread):
    """
    Thread for asynchronous model training.
    
    Signals:
        epoch_completed(int, dict): Epoch number and metrics
        training_completed(dict): Final results
        training_failed(str): Error message
    """
    
    def __init__(self, trainer: ModelTrainer, 
                 train_loader, val_loader):
        """
        Initialize training thread.
        
        Example:
            >>> thread = TrainingThread(trainer, train_loader, val_loader)
            >>> thread.epoch_completed.connect(update_progress)
            >>> thread.start()
        """
```

### InferenceThread

**Module**: `src.threads.inference_thread`

```python
class InferenceThread(QThread):
    """
    Thread for asynchronous batch inference.
    
    Signals:
        image_processed(int, np.ndarray): Image index and mask
        batch_completed(list): List of all masks
        inference_failed(str): Error message
    """
    
    def __init__(self, predictor: Predictor, 
                 image_paths: List[str], config: Dict):
        """
        Initialize inference thread.
        
        Example:
            >>> thread = InferenceThread(predictor, image_paths, config)
            >>> thread.image_processed.connect(save_result)
            >>> thread.start()
        """
```

---

## Usage Examples

### Complete Workflow Example

```python
"""
Complete defect segmentation workflow.
"""

from pathlib import Path
import numpy as np
import cv2

# 1. Data Management
from src.core.data_manager import DataManager

dm = DataManager("data", cache_size_mb=2048)
all_images = list(Path("data/raw").glob("*.jpg"))
splits = dm.create_splits([str(p) for p in all_images])

# 2. SAM Auto-Annotation
from src.core.sam_handler import SAMHandler
from src.core.annotation_manager import AnnotationManager

sam = SAMHandler(model_type='vit_h', device='cuda')
am = AnnotationManager(max_history=50)

for image_path in splits['train'][:10]:
    # Load image
    image = dm.load_image(image_path)
    
    # SAM prediction (with point prompt)
    sam.encode_image(image)
    prediction = sam.predict_mask_from_points(
        points=[(512, 512)],  # Click on defect
        labels=[1]
    )
    mask = sam.get_best_mask(prediction)
    
    # Save annotation
    am.set_image(image_path, image.shape[:2])
    am.set_mask(mask)
    am.save_mask(f"data/processed/masks/{Path(image_path).stem}.png")

# 3. Export to COCO
from src.utils.export_utils import export_to_coco

export_to_coco(
    image_paths=splits['train'],
    mask_paths=[f"data/processed/masks/{Path(p).stem}.png" 
                for p in splits['train']],
    output_path="data/processed/annotations/train.json",
    category_name="defect"
)

# 4. Model Training
from src.models.segmentation_models import SegmentationModel
from src.core.model_trainer import ModelTrainer
from torch.utils.data import DataLoader

model = SegmentationModel(
    architecture='unet',
    encoder_name='resnet34',
    encoder_weights='imagenet'
)

trainer = ModelTrainer(model, config={
    'epochs': 50,
    'lr': 1e-4,
    'device': 'cuda',
    'loss': 'dice',
    'checkpoint_dir': 'outputs/models'
})

# Assuming you have created DataLoaders
trainer.train(train_loader, val_loader)

# 5. Inference
from src.core.predictor import Predictor

predictor = Predictor(
    model_path='outputs/models/model_best.pth',
    device='cuda'
)

test_results = []
for image_path in splits['test']:
    image = dm.load_image(image_path)
    mask = predictor.predict(image, threshold=0.5)
    test_results.append(mask)

# 6. Generate Report
from src.utils.statistics import DefectStatistics
from src.utils.report_generator import ReportGenerator

stats_calc = DefectStatistics()
stats = stats_calc.compute_batch_statistics(
    [f"outputs/predictions/masks/{Path(p).stem}.png" 
     for p in splits['test']]
)

generator = ReportGenerator('outputs/reports')
report_path = generator.generate_excel_report(
    data=stats,
    output_filename='defect_analysis.xlsx'
)

print(f"Analysis complete! Report saved to: {report_path}")
```

---

**Document Version**: 1.0.0  
**Last Updated**: December 23, 2025  
**Maintainer**: Industrial AI Team
