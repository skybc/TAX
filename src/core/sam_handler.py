"""
用于自动标注的 SAM (Segment Anything Model) 处理程序。

此模块提供：
- SAM 模型的加载和初始化
- 图像编码
- 基于提示的掩码预测（点、框）
- 掩码后处理
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
    Segment Anything Model (SAM) 处理程序。
    
    提供以下功能：
    - 加载 SAM 模型
    - 编码图像
    - 根据提示（点、框）预测掩码
    - 管理模型状态
    
    属性:
        model_type: SAM 模型类型 ('vit_h', 'vit_l', 'vit_b')
        device: 计算设备 ('cuda' 或 'cpu')
        sam_model: SAM 模型实例
        image_encoder: 图像编码器模块
        prompt_encoder: 提示编码器模块
        mask_decoder: 掩码解码器模块
    """
    
    def __init__(self, 
                 model_type: str = 'vit_h',
                 checkpoint_path: Optional[str] = None,
                 device: Optional[str] = None):
        """
        初始化 SAM 处理程序。
        
        参数:
            model_type: 模型类型 ('vit_h', 'vit_l', 'vit_b')
            checkpoint_path: 模型检查点路径
            device: 要使用的设备（'cuda'、'cpu' 或 None 表示自动选择）
        """
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        
        # 设置设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # 模型组件
        self.sam_model = None
        self.predictor = None
        
        # 编码后的图像特征
        self.encoded_image = None
        self.current_image_shape = None
        
        logger.info(f"SAMHandler 已初始化 - 模型: {model_type}, 设备: {self.device}")
    
    def load_model(self, checkpoint_path: Optional[str] = None) -> bool:
        """
        从检查点加载 SAM 模型。
        
        参数:
            checkpoint_path: 检查点文件路径
            
        返回:
            如果成功则为 True，否则为 False
        """
        if checkpoint_path is not None:
            self.checkpoint_path = checkpoint_path
        
        if self.checkpoint_path is None:
            logger.error("未提供检查点路径")
            return False
        
        checkpoint_file = Path(self.checkpoint_path)
        if not checkpoint_file.exists():
            logger.error(f"未找到检查点文件: {self.checkpoint_path}")
            return False
        
        try:
            # 导入 SAM (segment-anything 包)
            try:
                from segment_anything import sam_model_registry, SamPredictor
            except ImportError:
                logger.error("未安装 segment-anything 包。请使用以下命令安装: pip install git+https://github.com/facebookresearch/segment-anything.git")
                return False
            
            # 加载模型
            logger.info(f"正在从 {self.checkpoint_path} 加载 SAM 模型...")
            self.sam_model = sam_model_registry[self.model_type](checkpoint=str(checkpoint_file))
            self.sam_model.to(self.device)
            self.sam_model.eval()
            
            # 创建预测器
            self.predictor = SamPredictor(self.sam_model)
            
            logger.info(f"SAM 模型已成功加载到 {self.device}")
            return True
            
        except Exception as e:
            logger.error(f"加载 SAM 模型失败: {e}", exc_info=True)
            return False
    
    def is_loaded(self) -> bool:
        """检查模型是否已加载。"""
        return self.sam_model is not None and self.predictor is not None
    
    def encode_image(self, image: np.ndarray) -> bool:
        """
        为 SAM 预测编码图像。
        
        在进行预测之前，每张图像应调用一次此方法。
        
        参数:
            image: RGB 格式的 numpy 数组 (HxWx3) 图像
            
        返回:
            如果成功则为 True，否则为 False
        """
        if not self.is_loaded():
            logger.error("SAM 模型未加载")
            return False
        
        try:
            # 在预测器中设置图像（这将对其进行编码）
            self.predictor.set_image(image)
            self.current_image_shape = image.shape[:2]
            
            logger.info(f"图像编码成功: {image.shape}")
            return True
            
        except Exception as e:
            logger.error(f"编码图像失败: {e}", exc_info=True)
            return False
    
    def predict_mask_from_points(self,
                                 points: List[Tuple[int, int]],
                                 labels: List[int],
                                 multimask_output: bool = True) -> Optional[Dict]:
        """
        根据点提示预测掩码。
        
        参数:
            points: (x, y) 坐标列表
            labels: 标签列表（1 表示前景，0 表示背景）
            multimask_output: 是否返回多个掩码
            
        返回:
            包含 'masks'、'scores'、'logits' 的字典，如果失败则返回 None
        """
        if not self.is_loaded():
            logger.error("SAM 模型未加载")
            return None
        
        if self.current_image_shape is None:
            logger.error("未编码图像。请先调用 encode_image()")
            return None
        
        try:
            # 转换为 numpy 数组
            point_coords = np.array(points, dtype=np.float32)
            point_labels = np.array(labels, dtype=np.int32)
            
            # 预测
            masks, scores, logits = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=multimask_output
            )
            
            logger.info(f"根据 {len(points)} 个点预测了 {len(masks)} 个掩码")
            
            return {
                'masks': masks,
                'scores': scores,
                'logits': logits
            }
            
        except Exception as e:
            logger.error(f"根据点预测掩码失败: {e}", exc_info=True)
            return None
    
    def predict_mask_from_box(self,
                             box: Tuple[int, int, int, int],
                             multimask_output: bool = False) -> Optional[Dict]:
        """
        根据边界框提示预测掩码。
        
        参数:
            box: 边界框，格式为 (x1, y1, x2, y2)
            multimask_output: 是否返回多个掩码
            
        返回:
            包含 'masks'、'scores'、'logits' 的字典，如果失败则返回 None
        """
        if not self.is_loaded():
            logger.error("SAM 模型未加载")
            return None
        
        if self.current_image_shape is None:
            logger.error("未编码图像。请先调用 encode_image()")
            return None
        
        try:
            # 转换为 numpy 数组
            box_array = np.array(box, dtype=np.float32)
            
            # 预测
            masks, scores, logits = self.predictor.predict(
                box=box_array,
                multimask_output=multimask_output
            )
            
            logger.info(f"根据边界框预测了 {len(masks)} 个掩码")
            
            return {
                'masks': masks,
                'scores': scores,
                'logits': logits
            }
            
        except Exception as e:
            logger.error(f"根据框预测掩码失败: {e}", exc_info=True)
            return None
    
    def predict_mask_from_combined(self,
                                   points: Optional[List[Tuple[int, int]]] = None,
                                   labels: Optional[List[int]] = None,
                                   box: Optional[Tuple[int, int, int, int]] = None,
                                   multimask_output: bool = True) -> Optional[Dict]:
        """
        根据组合提示（点和框）预测掩码。
        
        参数:
            points: (x, y) 坐标列表
            labels: 标签列表（1 表示前景，0 表示背景）
            box: 边界框，格式为 (x1, y1, x2, y2)
            multimask_output: 是否返回多个掩码
            
        返回:
            包含 'masks'、'scores'、'logits' 的字典，如果失败则返回 None
        """
        if not self.is_loaded():
            logger.error("SAM 模型未加载")
            return None
        
        if self.current_image_shape is None:
            logger.error("未编码图像。请先调用 encode_image()")
            return None
        
        try:
            # 准备参数
            point_coords = np.array(points, dtype=np.float32) if points else None
            point_labels = np.array(labels, dtype=np.int32) if labels else None
            box_array = np.array(box, dtype=np.float32) if box else None
            
            # 预测
            masks, scores, logits = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box_array,
                multimask_output=multimask_output
            )
            
            logger.info(f"根据组合提示预测了 {len(masks)} 个掩码")
            
            return {
                'masks': masks,
                'scores': scores,
                'logits': logits
            }
            
        except Exception as e:
            logger.error(f"根据组合提示预测掩码失败: {e}", exc_info=True)
            return None
    
    def get_best_mask(self, prediction: Dict) -> Optional[np.ndarray]:
        """
        从预测结果中获取最佳掩码。
        
        参数:
            prediction: 包含 'masks' 和 'scores' 的预测字典
            
        返回:
            二值 numpy 数组格式的最佳掩码 (HxW)，或 None
        """
        if prediction is None or 'masks' not in prediction or 'scores' not in prediction:
            return None
        
        masks = prediction['masks']
        scores = prediction['scores']
        
        if len(masks) == 0:
            return None
        
        # 获取得分最高的掩码
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]
        
        # 转换为二值掩码 (0 或 255)
        binary_mask = (best_mask > 0.5).astype(np.uint8) * 255
        
        return binary_mask
    
    def post_process_mask(self, 
                         mask: np.ndarray,
                         remove_small: bool = True,
                         min_area: int = 100,
                         fill_holes: bool = True) -> np.ndarray:
        """
        对掩码进行后处理以提高质量。
        
        参数:
            mask: 二值掩码 (HxW)
            remove_small: 是否移除小组件
            min_area: 组件的最小面积
            fill_holes: 是否填充孔洞
            
        返回:
            处理后的掩码
        """
        from src.utils.mask_utils import (
            remove_small_components, fill_holes as fill_mask_holes
        )
        
        processed_mask = mask.copy()
        
        # 移除小组件
        if remove_small:
            processed_mask = remove_small_components(processed_mask, min_area)
        
        # 填充孔洞
        if fill_holes:
            processed_mask = fill_mask_holes(processed_mask)
        
        return processed_mask
    
    def unload_model(self):
        """卸载模型以释放内存。"""
        if self.sam_model is not None:
            del self.sam_model
            self.sam_model = None
        
        if self.predictor is not None:
            del self.predictor
            self.predictor = None
        
        self.encoded_image = None
        self.current_image_shape = None
        
        # 如果可用，清除 CUDA 缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("SAM 模型已卸载")
    
    def get_model_info(self) -> Dict:
        """
        获取模型信息。
        
        返回:
            包含模型信息的字典
        """
        return {
            'model_type': self.model_type,
            'device': str(self.device),
            'checkpoint_path': self.checkpoint_path,
            'is_loaded': self.is_loaded(),
            'current_image_shape': self.current_image_shape
        }
