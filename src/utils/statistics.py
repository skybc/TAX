"""
缺陷分割的统计分析工具。

此模块提供：
- 缺陷统计计算（数量、大小、位置）
- 数据集分析（分布、类别平衡）
- 模型性能分析
- 比较统计
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import json
from collections import defaultdict, Counter

from src.logger import get_logger
from src.utils.mask_utils import load_mask, compute_mask_area

logger = get_logger(__name__)


class DefectStatistics:
    """
    从缺陷掩码计算统计信息。
    
    提供分析以下内容的方法：
    - 缺陷数量和大小
    - 空间分布
    - 缺陷特征
    """
    
    def __init__(self):
        """初始化 DefectStatistics。"""
        self.stats: Dict = {}
    
    def compute_mask_statistics(self, mask: np.ndarray, 
                               image_name: Optional[str] = None) -> Dict:
        """
        计算单个掩码的统计信息。
        
        参数:
            mask: 二值掩码 (HxW)
            image_name: 用于引用的可选图像名称
            
        返回:
            包含统计信息的字典：
                - num_defects: 连通分量数量
                - total_area: 总缺陷面积（像素）
                - defect_areas: 单个缺陷面积列表
                - defect_centroids: 缺陷质心列表 (x, y)
                - defect_bboxes: 边界框列表 (x, y, w, h)
                - coverage_ratio: 缺陷面积 / 总面积
                - largest_defect: 最大缺陷的面积
                - smallest_defect: 最小缺陷的面积
                - mean_defect_size: 平均缺陷面积
                - std_defect_size: 缺陷面积的标准差
        """
        from scipy import ndimage
        import cv2
        
        # 标记连通分量
        labeled, num_defects = ndimage.label(mask > 0)
        
        if num_defects == 0:
            return {
                'image_name': image_name,
                'num_defects': 0,
                'total_area': 0,
                'defect_areas': [],
                'defect_centroids': [],
                'defect_bboxes': [],
                'coverage_ratio': 0.0,
                'largest_defect': 0,
                'smallest_defect': 0,
                'mean_defect_size': 0.0,
                'std_defect_size': 0.0
            }
        
        # 计算每个缺陷的属性
        defect_areas = []
        defect_centroids = []
        defect_bboxes = []
        
        for i in range(1, num_defects + 1):
            # 提取单个缺陷
            defect_mask = (labeled == i).astype(np.uint8)
            
            # 计算面积
            area = np.sum(defect_mask)
            defect_areas.append(int(area))
            
            # 计算质心
            y_coords, x_coords = np.where(defect_mask > 0)
            if len(x_coords) > 0:
                centroid_x = float(np.mean(x_coords))
                centroid_y = float(np.mean(y_coords))
                defect_centroids.append((centroid_x, centroid_y))
            
            # 计算边界框
            contours, _ = cv2.findContours(
                defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                x, y, w, h = cv2.boundingRect(contours[0])
                defect_bboxes.append((int(x), int(y), int(w), int(h)))
        
        # 汇总统计
        total_area = sum(defect_areas)
        image_area = mask.shape[0] * mask.shape[1]
        coverage_ratio = total_area / image_area if image_area > 0 else 0.0
        
        stats = {
            'image_name': image_name,
            'num_defects': num_defects,
            'total_area': total_area,
            'defect_areas': defect_areas,
            'defect_centroids': defect_centroids,
            'defect_bboxes': defect_bboxes,
            'coverage_ratio': float(coverage_ratio),
            'largest_defect': max(defect_areas) if defect_areas else 0,
            'smallest_defect': min(defect_areas) if defect_areas else 0,
            'mean_defect_size': float(np.mean(defect_areas)) if defect_areas else 0.0,
            'std_defect_size': float(np.std(defect_areas)) if defect_areas else 0.0
        }
        
        return stats
    
    def compute_batch_statistics(self, mask_paths: List[str]) -> Dict:
        """
        计算多个掩码的统计信息。
        
        参数:
            mask_paths: 掩码文件路径列表
            
        返回:
            包含汇总统计信息的字典：
                - total_images: 图像总数
                - images_with_defects: 包含缺陷的图像数量
                - total_defects: 缺陷总数
                - total_defect_area: 所有图像的总缺陷面积
                - mean_defects_per_image: 每张图像的平均缺陷数
                - mean_coverage_ratio: 平均覆盖率
                - defect_size_distribution: 缺陷大小分布直方图
                - per_image_stats: 单个图像统计信息列表
        """
        logger.info(f"正在计算 {len(mask_paths)} 个掩码的统计信息...")
        
        all_stats = []
        all_defect_areas = []
        total_defects = 0
        images_with_defects = 0
        total_defect_area = 0
        coverage_ratios = []
        
        for mask_path in mask_paths:
            try:
                # 加载掩码
                mask = load_mask(mask_path)
                if mask is None:
                    logger.warning(f"加载掩码失败: {mask_path}")
                    continue
                
                # 计算统计信息
                image_name = Path(mask_path).stem
                stats = self.compute_mask_statistics(mask, image_name)
                all_stats.append(stats)
                
                # 汇总
                if stats['num_defects'] > 0:
                    images_with_defects += 1
                    total_defects += stats['num_defects']
                    total_defect_area += stats['total_area']
                    all_defect_areas.extend(stats['defect_areas'])
                    coverage_ratios.append(stats['coverage_ratio'])
                
            except Exception as e:
                logger.error(f"处理 {mask_path} 时出错: {e}")
                continue
        
        # 计算缺陷大小分布（直方图）
        if all_defect_areas:
            hist, bin_edges = np.histogram(all_defect_areas, bins=20)
            size_distribution = {
                'histogram': hist.tolist(),
                'bin_edges': bin_edges.tolist()
            }
        else:
            size_distribution = {'histogram': [], 'bin_edges': []}
        
        # 汇总统计
        batch_stats = {
            'total_images': len(mask_paths),
            'images_processed': len(all_stats),
            'images_with_defects': images_with_defects,
            'images_without_defects': len(all_stats) - images_with_defects,
            'total_defects': total_defects,
            'total_defect_area': total_defect_area,
            'mean_defects_per_image': total_defects / len(all_stats) if all_stats else 0.0,
            'mean_coverage_ratio': np.mean(coverage_ratios) if coverage_ratios else 0.0,
            'std_coverage_ratio': np.std(coverage_ratios) if coverage_ratios else 0.0,
            'defect_size_distribution': size_distribution,
            'per_image_stats': all_stats
        }
        
        logger.info(f"统计计算完成: {images_with_defects} 张图像中共有 {total_defects} 个缺陷")
        
        return batch_stats
    
    def compute_spatial_distribution(self, mask_paths: List[str], 
                                    grid_size: Tuple[int, int] = (10, 10)) -> np.ndarray:
        """
        计算缺陷在图像中的空间分布。
        
        参数:
            mask_paths: 掩码文件路径列表
            grid_size: 空间分箱的网格大小 (rows, cols)
            
        返回:
            显示缺陷频率的热图数组 (grid_size)
        """
        heatmap = np.zeros(grid_size, dtype=np.float32)
        count = 0
        
        for mask_path in mask_paths:
            try:
                mask = load_mask(mask_path)
                if mask is None:
                    continue
                
                # 将掩码调整为网格大小
                import cv2
                resized = cv2.resize(
                    mask.astype(np.float32), 
                    (grid_size[1], grid_size[0]),
                    interpolation=cv2.INTER_AREA
                )
                
                # 累加
                heatmap += (resized > 0).astype(np.float32)
                count += 1
                
            except Exception as e:
                logger.error(f"处理 {mask_path} 时出错: {e}")
                continue
        
        # 归一化
        if count > 0:
            heatmap /= count
        
        return heatmap


class DatasetStatistics:
    """
    计算数据集级别的统计信息。
    
    分析数据集特征：
    - 类别分布
    - 训练/验证/测试拆分统计
    - 数据质量指标
    """
    
    def __init__(self):
        """初始化 DatasetStatistics。"""
        pass
    
    def compute_dataset_summary(self, image_dir: str, mask_dir: str) -> Dict:
        """
        计算数据集的摘要统计信息。
        
        参数:
            image_dir: 包含图像的目录
            mask_dir: 包含掩码的目录
            
        返回:
            包含数据集摘要的字典
        """
        from src.utils.file_utils import list_files
        
        image_dir = Path(image_dir)
        mask_dir = Path(mask_dir)
        
        # 获取文件列表
        image_files = list_files(image_dir, extensions=['.jpg', '.jpeg', '.png', '.bmp', '.tif'])
        mask_files = list_files(mask_dir, extensions=['.png', '.tif'])
        
        # 匹配图像和掩码
        image_names = {Path(f).stem for f in image_files}
        mask_names = {Path(f).stem for f in mask_files}
        
        matched = image_names & mask_names
        images_only = image_names - mask_names
        masks_only = mask_names - image_names
        
        # 计算掩码统计信息
        matched_mask_paths = [str(mask_dir / f"{name}.png") for name in matched 
                             if (mask_dir / f"{name}.png").exists()]
        
        defect_stats = DefectStatistics()
        batch_stats = defect_stats.compute_batch_statistics(matched_mask_paths)
        
        summary = {
            'total_images': len(image_files),
            'total_masks': len(mask_files),
            'matched_pairs': len(matched),
            'images_without_masks': len(images_only),
            'masks_without_images': len(masks_only),
            'defect_statistics': batch_stats
        }
        
        return summary
    
    def compute_split_statistics(self, split_file: str) -> Dict:
        """
        计算训练/验证/测试拆分的统计信息。
        
        参数:
            split_file: 拆分文件的路径（例如 train.txt）
            
        返回:
            包含拆分统计信息的字典
        """
        split_file = Path(split_file)
        
        if not split_file.exists():
            logger.warning(f"未找到拆分文件: {split_file}")
            return {}
        
        # 读取拆分文件
        with open(split_file, 'r') as f:
            files = [line.strip() for line in f if line.strip()]
        
        stats = {
            'split_name': split_file.stem,
            'num_samples': len(files),
            'files': files
        }
        
        return stats


class ModelPerformanceAnalyzer:
    """
    分析模型性能指标。
    
    计算并可视化：
    - 训练历史分析
    - 混淆矩阵
    - 性能比较
    """
    
    def __init__(self):
        """初始化 ModelPerformanceAnalyzer。"""
        pass
    
    def analyze_training_history(self, history_file: str) -> Dict:
        """
        从日志文件分析训练历史。
        
        参数:
            history_file: 训练历史 JSON 文件的路径
            
        返回:
            包含分析指标的字典
        """
        history_file = Path(history_file)
        
        if not history_file.exists():
            logger.warning(f"未找到历史文件: {history_file}")
            return {}
        
        # 加载历史记录
        with open(history_file, 'r') as f:
            history = json.load(f)
        
        # 分析趋势
        analysis = {
            'total_epochs': len(history.get('train_loss', [])),
            'best_train_loss': min(history.get('train_loss', [float('inf')])),
            'best_val_loss': min(history.get('val_loss', [float('inf')])),
            'final_train_loss': history.get('train_loss', [])[-1] if history.get('train_loss') else None,
            'final_val_loss': history.get('val_loss', [])[-1] if history.get('val_loss') else None,
        }
        
        # 检查过拟合
        if history.get('train_loss') and history.get('val_loss'):
            train_losses = history['train_loss']
            val_losses = history['val_loss']
            
            # 比较最后 5 个 epoch
            if len(train_losses) >= 5:
                recent_train = np.mean(train_losses[-5:])
                recent_val = np.mean(val_losses[-5:])
                
                analysis['overfitting_indicator'] = recent_val - recent_train
                analysis['is_overfitting'] = recent_val > recent_train * 1.2
        
        return analysis
    
    def compute_confusion_matrix(self, pred_masks: List[np.ndarray],
                                gt_masks: List[np.ndarray]) -> Dict:
        """
        计算预测值与真实值的混淆矩阵。
        
        参数:
            pred_masks: 预测掩码列表
            gt_masks: 真实掩码列表
            
        返回:
            包含混淆矩阵和派生指标的字典
        """
        if len(pred_masks) != len(gt_masks):
            raise ValueError("预测数量和真实值数量必须匹配")
        
        # 累加混淆矩阵元素
        tp_total = 0
        tn_total = 0
        fp_total = 0
        fn_total = 0
        
        for pred, gt in zip(pred_masks, gt_masks):
            # 展平并二值化
            pred_flat = (pred.flatten() > 0).astype(np.uint8)
            gt_flat = (gt.flatten() > 0).astype(np.uint8)
            
            # 计算混淆矩阵元素
            tp = np.sum((pred_flat == 1) & (gt_flat == 1))
            tn = np.sum((pred_flat == 0) & (gt_flat == 0))
            fp = np.sum((pred_flat == 1) & (gt_flat == 0))
            fn = np.sum((pred_flat == 0) & (gt_flat == 1))
            
            tp_total += tp
            tn_total += tn
            fp_total += fp
            fn_total += fn
        
        # 计算指标
        accuracy = (tp_total + tn_total) / (tp_total + tn_total + fp_total + fn_total) if (tp_total + tn_total + fp_total + fn_total) > 0 else 0
        precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
        recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        iou = tp_total / (tp_total + fp_total + fn_total) if (tp_total + fp_total + fn_total) > 0 else 0
        
        confusion_matrix = {
            'tp': int(tp_total),
            'tn': int(tn_total),
            'fp': int(fp_total),
            'fn': int(fn_total),
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'iou': float(iou)
        }
        
        return confusion_matrix
    
    def compare_models(self, model_results: Dict[str, List[np.ndarray]],
                      gt_masks: List[np.ndarray]) -> Dict:
        """
        比较多个模型的预测结果。
        
        参数:
            model_results: 将模型名称映射到预测列表的字典
            gt_masks: 真实掩码
            
        返回:
            包含比较结果的字典
        """
        comparison = {}
        
        for model_name, pred_masks in model_results.items():
            logger.info(f"正在评估 {model_name}...")
            
            # 计算混淆矩阵
            cm = self.compute_confusion_matrix(pred_masks, gt_masks)
            comparison[model_name] = cm
        
        # 按 IoU 对模型进行排名
        ranked = sorted(comparison.items(), key=lambda x: x[1]['iou'], reverse=True)
        
        comparison['ranking'] = [name for name, _ in ranked]
        comparison['best_model'] = ranked[0][0] if ranked else None
        
        return comparison


def save_statistics(stats: Dict, output_path: str):
    """
    将统计信息保存到 JSON 文件。
    
    参数:
        stats: 统计信息字典
        output_path: 输出文件路径
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"统计信息已保存到: {output_path}")


def load_statistics(input_path: str) -> Dict:
    """
    从 JSON 文件加载统计信息。
    
    参数:
        input_path: 输入文件路径
        
    返回:
        统计信息字典
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        logger.error(f"未找到统计文件: {input_path}")
        return {}
    
    with open(input_path, 'r') as f:
        stats = json.load(f)
    
    logger.info(f"统计信息已从 {input_path} 加载")
    return stats
