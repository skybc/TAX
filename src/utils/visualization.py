"""
缺陷分割的可视化工具。

此模块提供：
- 图表生成 (matplotlib/seaborn)
- 缺陷热图
- 比较可视化
- 分布图
- 训练曲线可视化
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import cv2

from src.logger import get_logger

logger = get_logger(__name__)

# 设置 matplotlib 样式
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class DefectVisualizer:
    """
    可视化缺陷统计信息和掩码。
    
    提供创建以下内容的方法：
    - 缺陷大小分布
    - 空间热图
    - 叠加可视化
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 100):
        """
        初始化 DefectVisualizer。
        
        参数:
            figsize: 默认图形大小
            dpi: 图形 DPI
        """
        self.figsize = figsize
        self.dpi = dpi
    
    def plot_defect_size_distribution(self, defect_areas: List[int],
                                     output_path: Optional[str] = None,
                                     title: str = "缺陷大小分布") -> plt.Figure:
        """
        绘制缺陷大小的直方图。
        
        参数:
            defect_areas: 缺陷面积列表（像素）
            output_path: 可选的图形保存路径
            title: 图表标题
            
        返回:
            Matplotlib 图形对象
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # 绘制直方图
        ax.hist(defect_areas, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        
        # 添加统计信息
        mean_size = np.mean(defect_areas)
        median_size = np.median(defect_areas)
        
        ax.axvline(mean_size, color='red', linestyle='--', linewidth=2, 
                  label=f'平均值: {mean_size:.1f}')
        ax.axvline(median_size, color='green', linestyle='--', linewidth=2,
                  label=f'中位数: {median_size:.1f}')
        
        # 标签和标题
        ax.set_xlabel('缺陷面积 (像素)', fontsize=12)
        ax.set_ylabel('频率', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            fig.savefig(output_path, bbox_inches='tight')
            logger.info(f"缺陷大小分布图已保存到: {output_path}")
        
        return fig
    
    def plot_defect_count_per_image(self, defect_counts: List[int],
                                   output_path: Optional[str] = None,
                                   title: str = "每张图像的缺陷数量") -> plt.Figure:
        """
        绘制每张图像缺陷数量的分布图。
        
        参数:
            defect_counts: 每张图像的缺陷数量列表
            output_path: 可选的图形保存路径
            title: 图表标题
            
        返回:
            Matplotlib 图形对象
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # 绘制柱状图
        unique_counts, frequencies = np.unique(defect_counts, return_counts=True)
        
        ax.bar(unique_counts, frequencies, alpha=0.7, color='coral', edgecolor='black')
        
        # 标签
        ax.set_xlabel('缺陷数量', fontsize=12)
        ax.set_ylabel('图像数量', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if output_path:
            fig.savefig(output_path, bbox_inches='tight')
            logger.info(f"缺陷数量图已保存到: {output_path}")
        
        return fig
    
    def plot_spatial_heatmap(self, heatmap: np.ndarray,
                            output_path: Optional[str] = None,
                            title: str = "缺陷空间分布") -> plt.Figure:
        """
        绘制缺陷的空间热图。
        
        参数:
            heatmap: 表示缺陷频率的二维数组
            output_path: 可选的图形保存路径
            title: 图表标题
            
        返回:
            Matplotlib 图形对象
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # 绘制热图
        im = ax.imshow(heatmap, cmap='hot', interpolation='bilinear')
        
        # 颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('缺陷频率', fontsize=12)
        
        # 标签
        ax.set_xlabel('X 位置', fontsize=12)
        ax.set_ylabel('Y 位置', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if output_path:
            fig.savefig(output_path, bbox_inches='tight')
            logger.info(f"空间热图已保存到: {output_path}")
        
        return fig
    
    def plot_coverage_ratio_distribution(self, coverage_ratios: List[float],
                                        output_path: Optional[str] = None,
                                        title: str = "缺陷覆盖率") -> plt.Figure:
        """
        绘制缺陷覆盖率的分布图。
        
        参数:
            coverage_ratios: 覆盖率列表 (0-1)
            output_path: 可选的图形保存路径
            title: 图表标题
            
        返回:
            Matplotlib 图形对象
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # 绘制直方图
        ax.hist(coverage_ratios, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        
        # 添加平均值线
        mean_coverage = np.mean(coverage_ratios)
        ax.axvline(mean_coverage, color='red', linestyle='--', linewidth=2,
                  label=f'平均值: {mean_coverage:.3f}')
        
        # 标签
        ax.set_xlabel('覆盖率', fontsize=12)
        ax.set_ylabel('频率', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            fig.savefig(output_path, bbox_inches='tight')
            logger.info(f"覆盖率分布图已保存到: {output_path}")
        
        return fig
    
    def create_comparison_grid(self, images: List[np.ndarray],
                              masks: List[np.ndarray],
                              titles: List[str],
                              output_path: Optional[str] = None,
                              grid_size: Optional[Tuple[int, int]] = None) -> plt.Figure:
        """
        创建图像-掩码对比网格。
        
        参数:
            images: 图像列表
            masks: 掩码列表
            titles: 每个对比的标题列表
            output_path: 可选的图形保存路径
            grid_size: 可选的网格大小 (rows, cols)。如果为 None 则自动计算。
            
        返回:
            Matplotlib 图形对象
        """
        n_samples = len(images)
        
        if grid_size is None:
            # 自动计算网格大小
            cols = min(4, n_samples)
            rows = (n_samples + cols - 1) // cols
        else:
            rows, cols = grid_size
        
        fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 6, rows * 3), dpi=self.dpi)
        
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_samples):
            row = i // cols
            col = i % cols
            
            # 原始图像
            ax_img = axes[row, col * 2]
            ax_img.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB) if len(images[i].shape) == 3 else images[i], cmap='gray')
            ax_img.set_title(f"{titles[i]} - 图像", fontsize=10)
            ax_img.axis('off')
            
            # 掩码叠加
            ax_mask = axes[row, col * 2 + 1]
            overlay = self._create_overlay(images[i], masks[i])
            ax_mask.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            ax_mask.set_title(f"{titles[i]} - 叠加", fontsize=10)
            ax_mask.axis('off')
        
        # 隐藏未使用的子图
        for i in range(n_samples, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col * 2].axis('off')
            axes[row, col * 2 + 1].axis('off')
        
        plt.tight_layout()
        
        if output_path:
            fig.savefig(output_path, bbox_inches='tight')
            logger.info(f"对比网格图已保存到: {output_path}")
        
        return fig
    
    def _create_overlay(self, image: np.ndarray, mask: np.ndarray, 
                       alpha: float = 0.5) -> np.ndarray:
        """在图像上创建掩码叠加层。"""
        # 将灰度图转换为 RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # 创建彩色掩码
        colored_mask = np.zeros_like(image)
        colored_mask[mask > 0] = [0, 0, 255]  # 缺陷显示为红色
        
        # 混合
        overlay = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)
        
        return overlay


class TrainingVisualizer:
    """
    可视化模型训练指标。
    
    提供绘制以下内容的方法：
    - 训练/验证损失曲线
    - 指标曲线 (IoU, Dice 等)
    - 学习率计划
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 100):
        """
        初始化 TrainingVisualizer。
        
        参数:
            figsize: 默认图形大小
            dpi: 图形 DPI
        """
        self.figsize = figsize
        self.dpi = dpi
    
    def plot_training_curves(self, history: Dict,
                            output_path: Optional[str] = None,
                            title: str = "训练历史") -> plt.Figure:
        """
        绘制训练和验证曲线。
        
        参数:
            history: 包含 'train_loss', 'val_loss' 等的字典
            output_path: 可选的图形保存路径
            title: 图表标题
            
        返回:
            Matplotlib 图形对象
        """
        # 确定子图数量
        metrics = [k for k in history.keys() if not k.startswith('val_')]
        n_metrics = len(metrics)
        
        if n_metrics == 0:
            logger.warning("历史记录中未找到指标")
            return None
        
        fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5), dpi=self.dpi)
        
        if n_metrics == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # 绘制训练指标
            epochs = range(1, len(history[metric]) + 1)
            ax.plot(epochs, history[metric], 'b-', linewidth=2, label=f'训练 {metric}')
            
            # 如果可用，绘制验证指标
            val_key = f'val_{metric}'
            if val_key in history:
                ax.plot(epochs, history[val_key], 'r-', linewidth=2, label=f'验证 {val_key}')
            
            # 标签
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
            ax.set_title(f'{metric.replace("_", " ").title()} 随 Epoch 变化图', fontsize=13, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if output_path:
            fig.savefig(output_path, bbox_inches='tight')
            logger.info(f"训练曲线图已保存到: {output_path}")
        
        return fig
    
    def plot_confusion_matrix(self, confusion_matrix: Dict,
                             output_path: Optional[str] = None,
                             title: str = "混淆矩阵") -> plt.Figure:
        """
        绘制混淆矩阵。
        
        参数:
            confusion_matrix: 包含 'tp', 'tn', 'fp', 'fn' 的字典
            output_path: 可选的图形保存路径
            title: 图表标题
            
        返回:
            Matplotlib 图形对象
        """
        fig, ax = plt.subplots(figsize=(8, 6), dpi=self.dpi)
        
        # 提取值
        tp = confusion_matrix.get('tp', 0)
        tn = confusion_matrix.get('tn', 0)
        fp = confusion_matrix.get('fp', 0)
        fn = confusion_matrix.get('fn', 0)
        
        # 创建矩阵
        cm = np.array([[tn, fp], [fn, tp]])
        
        # 绘制热图
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['负样本', '正样本'],
                   yticklabels=['负样本', '正样本'],
                   cbar_kws={'label': '数量'})
        
        # 标签
        ax.set_xlabel('预测值', fontsize=12)
        ax.set_ylabel('真实值', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # 添加指标文本
        accuracy = confusion_matrix.get('accuracy', 0)
        precision = confusion_matrix.get('precision', 0)
        recall = confusion_matrix.get('recall', 0)
        f1 = confusion_matrix.get('f1_score', 0)
        iou = confusion_matrix.get('iou', 0)
        
        metrics_text = (f"准确率 (Accuracy): {accuracy:.4f}\n"
                       f"精确率 (Precision): {precision:.4f}\n"
                       f"召回率 (Recall): {recall:.4f}\n"
                       f"F1 分数: {f1:.4f}\n"
                       f"IoU: {iou:.4f}")
        
        ax.text(1.5, 0.5, metrics_text, transform=ax.transAxes,
               fontsize=11, verticalalignment='center',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if output_path:
            fig.savefig(output_path, bbox_inches='tight')
            logger.info(f"混淆矩阵图已保存到: {output_path}")
        
        return fig
    
    def plot_model_comparison(self, comparison_results: Dict,
                             metric: str = 'iou',
                             output_path: Optional[str] = None,
                             title: str = "模型对比") -> plt.Figure:
        """
        绘制模型对比柱状图。
        
        参数:
            comparison_results: 将模型名称映射到指标的字典
            metric: 用于对比的指标
            output_path: 可选的图形保存路径
            title: 图表标题
            
        返回:
            Matplotlib 图形对象
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # 提取模型名称和指标值
        models = []
        values = []
        
        for model_name, metrics in comparison_results.items():
            if model_name in ['ranking', 'best_model']:
                continue
            
            if metric in metrics:
                models.append(model_name)
                values.append(metrics[metric])
        
        # 按值排序
        sorted_pairs = sorted(zip(models, values), key=lambda x: x[1], reverse=True)
        models, values = zip(*sorted_pairs) if sorted_pairs else ([], [])
        
        # 绘制柱状图
        colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
        bars = ax.barh(models, values, color=colors, alpha=0.8, edgecolor='black')
        
        # 添加数值标签
        for i, (bar, value) in enumerate(zip(bars, values)):
            ax.text(value, i, f' {value:.4f}', va='center', fontsize=10)
        
        # 标签
        ax.set_xlabel(metric.upper(), fontsize=12)
        ax.set_ylabel('模型', fontsize=12)
        ax.set_title(f'{title} - {metric.upper()}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if output_path:
            fig.savefig(output_path, bbox_inches='tight')
            logger.info(f"模型对比图已保存到: {output_path}")
        
        return fig


class DatasetVisualizer:
    """
    可视化数据集统计信息。
    
    提供绘制以下内容的方法：
    - 类别分布
    - 数据拆分可视化
    - 样本多样性
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 100):
        """
        初始化 DatasetVisualizer。
        
        参数:
            figsize: 默认图形大小
            dpi: 图形 DPI
        """
        self.figsize = figsize
        self.dpi = dpi
    
    def plot_dataset_summary(self, summary: Dict,
                            output_path: Optional[str] = None,
                            title: str = "数据集摘要") -> plt.Figure:
        """
        绘制数据集摘要统计图。
        
        参数:
            summary: 数据集摘要字典
            output_path: 可选的图形保存路径
            title: 图表标题
            
        返回:
            Matplotlib 图形对象
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize, dpi=self.dpi)
        
        # 1. 图像 vs 掩码
        ax1 = axes[0, 0]
        categories = ['总图像数', '总掩码数', '匹配对数']
        counts = [
            summary.get('total_images', 0),
            summary.get('total_masks', 0),
            summary.get('matched_pairs', 0)
        ]
        ax1.bar(categories, counts, color=['skyblue', 'lightcoral', 'lightgreen'], alpha=0.8, edgecolor='black')
        ax1.set_ylabel('数量', fontsize=11)
        ax1.set_title('数据集构成', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 2. 缺陷存在情况
        ax2 = axes[0, 1]
        defect_stats = summary.get('defect_statistics', {})
        with_defects = defect_stats.get('images_with_defects', 0)
        without_defects = defect_stats.get('images_without_defects', 0)
        
        if with_defects + without_defects > 0:
            ax2.pie([with_defects, without_defects],
                   labels=['有缺陷', '无缺陷'],
                   autopct='%1.1f%%',
                   colors=['salmon', 'lightblue'],
                   startangle=90)
            ax2.set_title('缺陷检出率', fontsize=12, fontweight='bold')
        else:
            ax2.text(0.5, 0.5, '无数据', ha='center', va='center', fontsize=14)
            ax2.axis('off')
        
        # 3. 缺陷统计
        ax3 = axes[1, 0]
        total_defects = defect_stats.get('total_defects', 0)
        mean_defects = defect_stats.get('mean_defects_per_image', 0)
        
        ax3.bar(['总缺陷数', '平均每图\n缺陷数'], [total_defects, mean_defects],
               color=['coral', 'gold'], alpha=0.8, edgecolor='black')
        ax3.set_ylabel('数量', fontsize=11)
        ax3.set_title('缺陷统计', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. 覆盖率
        ax4 = axes[1, 1]
        mean_coverage = defect_stats.get('mean_coverage_ratio', 0)
        std_coverage = defect_stats.get('std_coverage_ratio', 0)
        
        ax4.bar(['平均覆盖率'], [mean_coverage], yerr=[std_coverage],
               color='mediumseagreen', alpha=0.8, edgecolor='black', capsize=10)
        ax4.set_ylabel('覆盖率', fontsize=11)
        ax4.set_title('缺陷覆盖情况', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.set_ylim(0, min(1.0, mean_coverage + 2 * std_coverage + 0.1))
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if output_path:
            fig.savefig(output_path, bbox_inches='tight')
            logger.info(f"数据集摘要图已保存到: {output_path}")
        
        return fig


def close_all_figures():
    """关闭所有 matplotlib 图形。"""
    plt.close('all')
    logger.debug("已关闭所有 matplotlib 图形")
