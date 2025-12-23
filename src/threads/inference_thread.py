"""
Inference thread for asynchronous prediction.

This module provides:
- Asynchronous batch inference
- Progress reporting
- Result aggregation
"""

from typing import List, Dict, Optional
from pathlib import Path
from PyQt5.QtCore import QThread, pyqtSignal

from src.logger import get_logger
from src.core.predictor import Predictor

logger = get_logger(__name__)


class InferenceThread(QThread):
    """
    Thread for running inference asynchronously.
    
    Signals:
        progress_updated: Emitted during processing (current, total, image_path)
        image_completed: Emitted when one image is processed (index, image_path, success)
        inference_completed: Emitted when all images are processed (results)
        inference_failed: Emitted when inference fails (error_message)
    """
    
    progress_updated = pyqtSignal(int, int, str)  # current, total, image_path
    image_completed = pyqtSignal(int, str, bool)  # index, image_path, success
    inference_completed = pyqtSignal(dict)  # results dict
    inference_failed = pyqtSignal(str)  # error message
    
    def __init__(self,
                 checkpoint_path: str,
                 image_paths: List[str],
                 output_dir: str,
                 config: Dict):
        """
        Initialize inference thread.
        
        Args:
            checkpoint_path: Path to model checkpoint
            image_paths: List of image file paths
            output_dir: Directory to save predictions
            config: Inference configuration
        """
        super().__init__()
        
        self.checkpoint_path = checkpoint_path
        self.image_paths = image_paths
        self.output_dir = output_dir
        self.config = config
        
        self._is_running = True
        
        logger.info(f"Inference thread initialized: {len(image_paths)} images")
    
    def run(self):
        """Execute inference."""
        try:
            # Create predictor
            from src.core.predictor import create_predictor
            
            predictor = create_predictor(
                checkpoint_path=self.checkpoint_path,
                architecture=self.config.get('architecture', 'unet'),
                encoder=self.config.get('encoder', 'resnet34'),
                device=self.config.get('device'),
                image_size=(self.config.get('image_height', 512),
                           self.config.get('image_width', 512))
            )
            
            if predictor is None:
                self.inference_failed.emit("Failed to create predictor")
                return
            
            # Create output directory
            output_dir = Path(self.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            masks_dir = output_dir / "masks"
            masks_dir.mkdir(exist_ok=True)
            
            save_overlay = self.config.get('save_overlay', True)
            if save_overlay:
                overlay_dir = output_dir / "overlays"
                overlay_dir.mkdir(exist_ok=True)
            
            # Apply post-processing
            apply_post_processing = self.config.get('apply_post_processing', True)
            
            # Results
            results = {
                'total': len(self.image_paths),
                'successful': 0,
                'failed': 0,
                'failed_files': []
            }
            
            # Process each image
            for i, image_path in enumerate(self.image_paths):
                if not self._is_running:
                    logger.info("Inference stopped by user")
                    break
                
                try:
                    # Update progress
                    self.progress_updated.emit(i + 1, len(self.image_paths), image_path)
                    
                    # Load and predict
                    from src.utils.image_utils import load_image, save_image
                    from src.utils.mask_utils import save_mask
                    
                    image = load_image(image_path)
                    if image is None:
                        results['failed'] += 1
                        results['failed_files'].append(image_path)
                        self.image_completed.emit(i, image_path, False)
                        continue
                    
                    # Predict
                    use_tta = self.config.get('use_tta', False)
                    threshold = self.config.get('threshold', 0.5)
                    
                    if use_tta:
                        mask = predictor.predict_with_tta(
                            image,
                            threshold=threshold,
                            num_augmentations=self.config.get('tta_augmentations', 4)
                        )
                    else:
                        mask = predictor.predict(image, threshold=threshold)
                    
                    # Post-processing
                    if apply_post_processing:
                        from src.utils.post_processing import refine_mask
                        
                        mask = refine_mask(
                            mask,
                            remove_small=self.config.get('remove_small_objects', True),
                            min_size=self.config.get('min_object_size', 100),
                            fill_holes_flag=self.config.get('fill_holes', True),
                            smooth=self.config.get('smooth_contours', True),
                            closing_size=self.config.get('closing_kernel_size', 5)
                        )
                    
                    # Save mask
                    image_name = Path(image_path).stem
                    mask_path = masks_dir / f"{image_name}_mask.png"
                    save_mask(mask, str(mask_path))
                    
                    # Save overlay
                    if save_overlay:
                        overlay = predictor._create_overlay(
                            image, mask,
                            alpha=self.config.get('overlay_alpha', 0.5),
                            color=tuple(self.config.get('overlay_color', [0, 255, 0]))
                        )
                        overlay_path = overlay_dir / f"{image_name}_overlay.png"
                        save_image(overlay, str(overlay_path))
                    
                    results['successful'] += 1
                    self.image_completed.emit(i, image_path, True)
                    
                except Exception as e:
                    logger.error(f"Failed to process {image_path}: {e}")
                    results['failed'] += 1
                    results['failed_files'].append(image_path)
                    self.image_completed.emit(i, image_path, False)
            
            # Complete
            if self._is_running:
                logger.info(f"Inference completed: {results['successful']}/{results['total']} successful")
                self.inference_completed.emit(results)
            
        except Exception as e:
            logger.error(f"Inference error: {e}", exc_info=True)
            self.inference_failed.emit(str(e))
    
    def stop(self):
        """Stop inference."""
        self._is_running = False
        logger.info("Inference stop requested")
