"""
SAM inference thread for asynchronous mask prediction.

This module provides:
- Asynchronous SAM inference in a separate thread
- Progress reporting
- Result signaling
"""

from typing import Optional, List, Tuple, Dict
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from src.logger import get_logger
from src.core.sam_handler import SAMHandler

logger = get_logger(__name__)


class SAMInferenceThread(QThread):
    """
    Thread for running SAM inference asynchronously.
    
    Signals:
        progress_updated: Emitted during processing (progress_percent, message)
        inference_completed: Emitted when inference is complete (mask_result)
        inference_failed: Emitted when inference fails (error_message)
    """
    
    progress_updated = pyqtSignal(int, str)  # progress, message
    inference_completed = pyqtSignal(dict)  # result dict
    inference_failed = pyqtSignal(str)  # error message
    
    def __init__(self, 
                 sam_handler: SAMHandler,
                 image: np.ndarray,
                 prompt_type: str,
                 prompt_data: Dict):
        """
        Initialize SAM inference thread.
        
        Args:
            sam_handler: SAMHandler instance
            image: Image to process (HxWx3)
            prompt_type: Type of prompt ('points', 'box', 'combined')
            prompt_data: Prompt data dictionary
        """
        super().__init__()
        
        self.sam_handler = sam_handler
        self.image = image
        self.prompt_type = prompt_type
        self.prompt_data = prompt_data
        
        self._is_running = True
    
    def run(self):
        """Execute SAM inference."""
        try:
            # Step 1: Encode image
            self.progress_updated.emit(10, "Encoding image...")
            
            if not self.sam_handler.encode_image(self.image):
                self.inference_failed.emit("Failed to encode image")
                return
            
            if not self._is_running:
                return
            
            # Step 2: Predict mask
            self.progress_updated.emit(50, "Predicting mask...")
            
            prediction = None
            
            if self.prompt_type == 'points':
                points = self.prompt_data.get('points', [])
                labels = self.prompt_data.get('labels', [])
                multimask = self.prompt_data.get('multimask_output', True)
                
                prediction = self.sam_handler.predict_mask_from_points(
                    points, labels, multimask
                )
                
            elif self.prompt_type == 'box':
                box = self.prompt_data.get('box')
                multimask = self.prompt_data.get('multimask_output', False)
                
                prediction = self.sam_handler.predict_mask_from_box(
                    box, multimask
                )
                
            elif self.prompt_type == 'combined':
                points = self.prompt_data.get('points')
                labels = self.prompt_data.get('labels')
                box = self.prompt_data.get('box')
                multimask = self.prompt_data.get('multimask_output', True)
                
                prediction = self.sam_handler.predict_mask_from_combined(
                    points, labels, box, multimask
                )
            
            else:
                self.inference_failed.emit(f"Unknown prompt type: {self.prompt_type}")
                return
            
            if not self._is_running:
                return
            
            if prediction is None:
                self.inference_failed.emit("SAM prediction failed")
                return
            
            # Step 3: Get best mask
            self.progress_updated.emit(80, "Processing result...")
            
            best_mask = self.sam_handler.get_best_mask(prediction)
            
            if best_mask is None:
                self.inference_failed.emit("No valid mask generated")
                return
            
            # Step 4: Post-process (optional)
            post_process = self.prompt_data.get('post_process', True)
            if post_process:
                self.progress_updated.emit(90, "Post-processing mask...")
                best_mask = self.sam_handler.post_process_mask(best_mask)
            
            if not self._is_running:
                return
            
            # Step 5: Complete
            self.progress_updated.emit(100, "Complete!")
            
            result = {
                'mask': best_mask,
                'all_masks': prediction['masks'],
                'scores': prediction['scores'],
                'prompt_type': self.prompt_type,
                'prompt_data': self.prompt_data
            }
            
            self.inference_completed.emit(result)
            
        except Exception as e:
            logger.error(f"SAM inference error: {e}", exc_info=True)
            self.inference_failed.emit(str(e))
    
    def stop(self):
        """Stop the inference thread."""
        self._is_running = False


class SAMBatchInferenceThread(QThread):
    """
    Thread for batch SAM inference on multiple images.
    
    Signals:
        progress_updated: Emitted during processing (current, total, message)
        image_completed: Emitted when one image is processed (index, mask)
        batch_completed: Emitted when all images are processed (results_list)
        inference_failed: Emitted when inference fails (error_message)
    """
    
    progress_updated = pyqtSignal(int, int, str)  # current, total, message
    image_completed = pyqtSignal(int, np.ndarray)  # index, mask
    batch_completed = pyqtSignal(list)  # list of masks
    inference_failed = pyqtSignal(str)  # error message
    
    def __init__(self,
                 sam_handler: SAMHandler,
                 images: List[np.ndarray],
                 prompt_type: str,
                 prompt_data_list: List[Dict]):
        """
        Initialize batch SAM inference thread.
        
        Args:
            sam_handler: SAMHandler instance
            images: List of images to process
            prompt_type: Type of prompt ('points', 'box', 'combined')
            prompt_data_list: List of prompt data dictionaries (one per image)
        """
        super().__init__()
        
        self.sam_handler = sam_handler
        self.images = images
        self.prompt_type = prompt_type
        self.prompt_data_list = prompt_data_list
        
        self._is_running = True
    
    def run(self):
        """Execute batch SAM inference."""
        try:
            results = []
            total = len(self.images)
            
            for i, (image, prompt_data) in enumerate(zip(self.images, self.prompt_data_list)):
                if not self._is_running:
                    break
                
                # Update progress
                self.progress_updated.emit(i, total, f"Processing image {i+1}/{total}")
                
                # Encode image
                if not self.sam_handler.encode_image(image):
                    logger.error(f"Failed to encode image {i}")
                    results.append(None)
                    continue
                
                # Predict mask
                prediction = None
                
                if self.prompt_type == 'points':
                    points = prompt_data.get('points', [])
                    labels = prompt_data.get('labels', [])
                    multimask = prompt_data.get('multimask_output', True)
                    prediction = self.sam_handler.predict_mask_from_points(
                        points, labels, multimask
                    )
                    
                elif self.prompt_type == 'box':
                    box = prompt_data.get('box')
                    multimask = prompt_data.get('multimask_output', False)
                    prediction = self.sam_handler.predict_mask_from_box(
                        box, multimask
                    )
                
                if prediction is None:
                    logger.error(f"Failed to predict mask for image {i}")
                    results.append(None)
                    continue
                
                # Get best mask
                best_mask = self.sam_handler.get_best_mask(prediction)
                
                if best_mask is not None:
                    # Post-process if requested
                    if prompt_data.get('post_process', True):
                        best_mask = self.sam_handler.post_process_mask(best_mask)
                    
                    results.append(best_mask)
                    self.image_completed.emit(i, best_mask)
                else:
                    results.append(None)
            
            # Complete
            if self._is_running:
                self.progress_updated.emit(total, total, "Batch processing complete!")
                self.batch_completed.emit(results)
                
        except Exception as e:
            logger.error(f"Batch SAM inference error: {e}", exc_info=True)
            self.inference_failed.emit(str(e))
    
    def stop(self):
        """Stop the batch inference thread."""
        self._is_running = False
