"""
用于异步掩码预测的 SAM 推理线程。

此模块提供：
- 在单独线程中进行异步 SAM 推理
- 进度报告
- 结果信号
"""

from typing import Optional, List, Tuple, Dict
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from src.logger import get_logger
from src.core.sam_handler import SAMHandler

logger = get_logger(__name__)


class SAMInferenceThread(QThread):
    """
    用于异步运行 SAM 推理的线程。
    
    信号:
        progress_updated: 在处理期间发出（进度百分比, 消息）
        inference_completed: 推理完成时发出（掩码结果）
        inference_failed: 推理失败时发出（错误消息）
    """
    
    progress_updated = pyqtSignal(int, str)  # 进度, 消息
    inference_completed = pyqtSignal(dict)  # 结果字典
    inference_failed = pyqtSignal(str)  # 错误消息
    
    def __init__(self, 
                 sam_handler: SAMHandler,
                 image: np.ndarray,
                 prompt_type: str,
                 prompt_data: Dict):
        """
        初始化 SAM 推理线程。
        
        参数:
            sam_handler: SAMHandler 实例
            image: 要处理的图像 (HxWx3)
            prompt_type: 提示类型 ('points', 'box', 'combined')
            prompt_data: 提示数据字典
        """
        super().__init__()
        
        self.sam_handler = sam_handler
        self.image = image
        self.prompt_type = prompt_type
        self.prompt_data = prompt_data
        
        self._is_running = True
    
    def run(self):
        """执行 SAM 推理。"""
        try:
            # 步骤 1: 编码图像
            self.progress_updated.emit(10, "正在编码图像...")
            
            if not self.sam_handler.encode_image(self.image):
                self.inference_failed.emit("编码图像失败")
                return
            
            if not self._is_running:
                return
            
            # 步骤 2: 预测掩码
            self.progress_updated.emit(50, "正在预测掩码...")
            
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
                self.inference_failed.emit(f"未知提示类型: {self.prompt_type}")
                return
            
            if not self._is_running:
                return
            
            if prediction is None:
                self.inference_failed.emit("SAM 预测失败")
                return
            
            # 步骤 3: 获取最佳掩码
            self.progress_updated.emit(80, "正在处理结果...")
            
            best_mask = self.sam_handler.get_best_mask(prediction)
            
            if best_mask is None:
                self.inference_failed.emit("未生成有效掩码")
                return
            
            # 步骤 4: 后处理（可选）
            post_process = self.prompt_data.get('post_process', True)
            if post_process:
                self.progress_updated.emit(90, "正在对掩码进行后处理...")
                best_mask = self.sam_handler.post_process_mask(best_mask)
            
            if not self._is_running:
                return
            
            # 步骤 5: 完成
            self.progress_updated.emit(100, "完成！")
            
            result = {
                'mask': best_mask,
                'all_masks': prediction['masks'],
                'scores': prediction['scores'],
                'prompt_type': self.prompt_type,
                'prompt_data': self.prompt_data
            }
            
            self.inference_completed.emit(result)
            
        except Exception as e:
            logger.error(f"SAM 推理错误: {e}", exc_info=True)
            self.inference_failed.emit(str(e))
    
    def stop(self):
        """停止推理线程。"""
        self._is_running = False


class SAMBatchInferenceThread(QThread):
    """
    用于对多张图像进行批量 SAM 推理的线程。
    
    信号:
        progress_updated: 在处理期间发出（当前, 总计, 消息）
        image_completed: 处理完一张图像时发出（索引, 掩码）
        batch_completed: 处理完所有图像时发出（结果列表）
        inference_failed: 推理失败时发出（错误消息）
    """
    
    progress_updated = pyqtSignal(int, int, str)  # 当前, 总计, 消息
    image_completed = pyqtSignal(int, np.ndarray)  # 索引, 掩码
    batch_completed = pyqtSignal(list)  # 掩码列表
    inference_failed = pyqtSignal(str)  # 错误消息
    
    def __init__(self,
                 sam_handler: SAMHandler,
                 images: List[np.ndarray],
                 prompt_type: str,
                 prompt_data_list: List[Dict]):
        """
        初始化批量 SAM 推理线程。
        
        参数:
            sam_handler: SAMHandler 实例
            images: 要处理的图像列表
            prompt_type: 提示类型 ('points', 'box', 'combined')
            prompt_data_list: 提示数据字典列表（每张图像一个）
        """
        super().__init__()
        
        self.sam_handler = sam_handler
        self.images = images
        self.prompt_type = prompt_type
        self.prompt_data_list = prompt_data_list
        
        self._is_running = True
    
    def run(self):
        """执行批量 SAM 推理。"""
        try:
            results = []
            total = len(self.images)
            
            for i, (image, prompt_data) in enumerate(zip(self.images, self.prompt_data_list)):
                if not self._is_running:
                    break
                
                # 更新进度
                self.progress_updated.emit(i, total, f"正在处理图像 {i+1}/{total}")
                
                # 编码图像
                if not self.sam_handler.encode_image(image):
                    logger.error(f"编码图像 {i} 失败")
                    results.append(None)
                    continue
                
                # 预测掩码
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
                    logger.error(f"预测图像 {i} 的掩码失败")
                    results.append(None)
                    continue
                
                # 获取最佳掩码
                best_mask = self.sam_handler.get_best_mask(prediction)
                
                if best_mask is not None:
                    # 如果有要求，进行后处理
                    if prompt_data.get('post_process', True):
                        best_mask = self.sam_handler.post_process_mask(best_mask)
                    
                    results.append(best_mask)
                    self.image_completed.emit(i, best_mask)
                else:
                    results.append(None)
            
            # 完成
            if self._is_running:
                self.progress_updated.emit(total, total, "批量处理完成！")
                self.batch_completed.emit(results)
                
        except Exception as e:
            logger.error(f"批量 SAM 推理错误: {e}", exc_info=True)
            self.inference_failed.emit(str(e))
    
    def stop(self):
        """停止批量推理线程。"""
        self._is_running = False
