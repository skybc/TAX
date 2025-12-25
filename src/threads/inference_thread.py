"""
用于异步预测的推理线程。

此模块提供：
- 异步批量推理
- 进度报告
- 结果汇总
"""

from typing import List, Dict, Optional
from pathlib import Path
from PyQt5.QtCore import QThread, pyqtSignal

from src.logger import get_logger
from src.core.predictor import Predictor

logger = get_logger(__name__)


class InferenceThread(QThread):
    """
    用于异步运行推理的线程。
    
    信号:
        progress_updated: 在处理期间发出（当前, 总计, 图像路径）
        image_completed: 处理完一张图像时发出（索引, 图像路径, 是否成功）
        inference_completed: 处理完所有图像时发出（结果）
        inference_failed: 推理失败时发出（错误消息）
    """
    
    progress_updated = pyqtSignal(int, int, str)  # 当前, 总计, 图像路径
    image_completed = pyqtSignal(int, str, bool)  # 索引, 图像路径, 是否成功
    inference_completed = pyqtSignal(dict)  # 结果字典
    inference_failed = pyqtSignal(str)  # 错误消息
    
    def __init__(self,
                 checkpoint_path: str,
                 image_paths: List[str],
                 output_dir: str,
                 config: Dict):
        """
        初始化推理线程。
        
        参数:
            checkpoint_path: 模型检查点路径
            image_paths: 图像文件路径列表
            output_dir: 保存预测结果的目录
            config: 推理配置
        """
        super().__init__()
        
        self.checkpoint_path = checkpoint_path
        self.image_paths = image_paths
        self.output_dir = output_dir
        self.config = config
        
        self._is_running = True
        
        logger.info(f"推理线程已初始化: {len(image_paths)} 张图像")
    
    def run(self):
        """执行推理。"""
        try:
            # 创建预测器
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
                self.inference_failed.emit("创建预测器失败")
                return
            
            # 创建输出目录
            output_dir = Path(self.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            masks_dir = output_dir / "masks"
            masks_dir.mkdir(exist_ok=True)
            
            save_overlay = self.config.get('save_overlay', True)
            if save_overlay:
                overlay_dir = output_dir / "overlays"
                overlay_dir.mkdir(exist_ok=True)
            
            # 应用后处理
            apply_post_processing = self.config.get('apply_post_processing', True)
            
            # 结果
            results = {
                'total': len(self.image_paths),
                'successful': 0,
                'failed': 0,
                'failed_files': []
            }
            
            # 处理每张图像
            for i, image_path in enumerate(self.image_paths):
                if not self._is_running:
                    logger.info("推理被用户停止")
                    break
                
                try:
                    # 更新进度
                    self.progress_updated.emit(i + 1, len(self.image_paths), image_path)
                    
                    # 加载并预测
                    from src.utils.image_utils import load_image, save_image
                    from src.utils.mask_utils import save_mask
                    
                    image = load_image(image_path)
                    if image is None:
                        results['failed'] += 1
                        results['failed_files'].append(image_path)
                        self.image_completed.emit(i, image_path, False)
                        continue
                    
                    # 预测
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
                    
                    # 后处理
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
                    
                    # 保存掩码
                    image_name = Path(image_path).stem
                    mask_path = masks_dir / f"{image_name}_mask.png"
                    save_mask(mask, str(mask_path))
                    
                    # 保存叠加图
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
                    logger.error(f"处理 {image_path} 失败: {e}")
                    results['failed'] += 1
                    results['failed_files'].append(image_path)
                    self.image_completed.emit(i, image_path, False)
            
            # 完成
            if self._is_running:
                logger.info(f"推理完成: {results['successful']}/{results['total']} 成功")
                self.inference_completed.emit(results)
            
        except Exception as e:
            logger.error(f"推理错误: {e}", exc_info=True)
            self.inference_failed.emit(str(e))
    
    def stop(self):
        """停止推理。"""
        self._is_running = False
        logger.info("已请求停止推理")
