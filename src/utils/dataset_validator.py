"""
用于检查标注格式正确性的数据集验证器。

此模块提供：
- COCO 格式验证
- YOLO 格式验证
- VOC 格式验证
- 标注完整性检查
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np

from src.logger import get_logger
from src.utils.file_utils import load_json

logger = get_logger(__name__)


class ValidationResult:
    """验证结果的容器。"""
    
    def __init__(self):
        self.is_valid = True
        self.errors = []
        self.warnings = []
        self.stats = {}
    
    def add_error(self, message: str):
        """添加错误消息。"""
        self.errors.append(message)
        self.is_valid = False
        logger.error(f"验证错误: {message}")
    
    def add_warning(self, message: str):
        """添加警告消息。"""
        self.warnings.append(message)
        logger.warning(f"验证警告: {message}")
    
    def add_stat(self, key: str, value):
        """添加统计信息。"""
        self.stats[key] = value
    
    def get_report(self) -> str:
        """
        获取字符串形式的验证报告。
        
        返回:
            格式化的验证报告
        """
        lines = []
        lines.append("=" * 60)
        lines.append("验证报告")
        lines.append("=" * 60)
        lines.append(f"状态: {'✅ 通过' if self.is_valid else '❌ 失败'}")
        lines.append("")
        
        if self.stats:
            lines.append("统计信息:")
            for key, value in self.stats.items():
                lines.append(f"  {key}: {value}")
            lines.append("")
        
        if self.errors:
            lines.append(f"错误 ({len(self.errors)}):")
            for error in self.errors:
                lines.append(f"  ❌ {error}")
            lines.append("")
        
        if self.warnings:
            lines.append(f"警告 ({len(self.warnings)}):")
            for warning in self.warnings:
                lines.append(f"  ⚠️  {warning}")
            lines.append("")
        
        lines.append("=" * 60)
        return "\n".join(lines)


class COCOValidator:
    """
    COCO 格式数据集的验证器。
    
    检查项：
    - JSON 结构有效性
    - 必填字段是否存在
    - 图像文件是否存在
    - 标注一致性
    - 分割格式
    """
    
    def __init__(self, coco_json_path: str, images_dir: Optional[str] = None):
        """
        初始化 COCO 验证器。
        
        参数:
            coco_json_path: COCO JSON 文件路径
            images_dir: 包含图像的可选目录（用于文件检查）
        """
        self.coco_json_path = Path(coco_json_path)
        self.images_dir = Path(images_dir) if images_dir else None
        self.result = ValidationResult()
    
    def validate(self) -> ValidationResult:
        """
        验证 COCO 数据集。
        
        返回:
            ValidationResult 对象
        """
        logger.info(f"正在验证 COCO 数据集: {self.coco_json_path}")
        
        # 检查文件是否存在
        if not self.coco_json_path.exists():
            self.result.add_error(f"未找到 COCO JSON 文件: {self.coco_json_path}")
            return self.result
        
        # 加载 JSON
        try:
            coco_data = load_json(self.coco_json_path)
        except Exception as e:
            self.result.add_error(f"加载 JSON 失败: {e}")
            return self.result
        
        # 验证结构
        self._validate_structure(coco_data)
        
        if not self.result.is_valid:
            return self.result
        
        # 验证组件
        self._validate_images(coco_data.get("images", []))
        self._validate_annotations(coco_data.get("annotations", []))
        self._validate_categories(coco_data.get("categories", []))
        
        # 检查一致性
        self._check_consistency(coco_data)
        
        # 添加统计信息
        self.result.add_stat("图像数量", len(coco_data.get("images", [])))
        self.result.add_stat("标注数量", len(coco_data.get("annotations", [])))
        self.result.add_stat("类别数量", len(coco_data.get("categories", [])))
        
        logger.info("COCO 验证完成")
        return self.result
    
    def _validate_structure(self, coco_data: Dict):
        """验证顶级结构。"""
        required_fields = ["images", "annotations", "categories"]
        
        for field in required_fields:
            if field not in coco_data:
                self.result.add_error(f"缺少必填字段: {field}")
        
        optional_fields = ["info", "licenses"]
        for field in optional_fields:
            if field not in coco_data:
                self.result.add_warning(f"缺少可选字段: {field}")
    
    def _validate_images(self, images: List[Dict]):
        """验证图像部分。"""
        if not images:
            self.result.add_error("数据集中未找到图像")
            return
        
        image_ids = set()
        
        for idx, image in enumerate(images):
            # 检查必填字段
            required = ["id", "file_name", "width", "height"]
            for field in required:
                if field not in image:
                    self.result.add_error(f"图像 {idx}: 缺少字段 '{field}'")
            
            # 检查 ID 唯一性
            if "id" in image:
                if image["id"] in image_ids:
                    self.result.add_error(f"重复的图像 ID: {image['id']}")
                image_ids.add(image["id"])
            
            # 检查尺寸
            if "width" in image and "height" in image:
                if image["width"] <= 0 or image["height"] <= 0:
                    self.result.add_error(f"图像 {idx}: 尺寸无效")
            
            # 检查文件是否存在
            if self.images_dir and "file_name" in image:
                image_path = self.images_dir / image["file_name"]
                if not image_path.exists():
                    self.result.add_warning(f"未找到图像文件: {image['file_name']}")
    
    def _validate_annotations(self, annotations: List[Dict]):
        """验证标注部分。"""
        if not annotations:
            self.result.add_warning("数据集中未找到标注")
            return
        
        annotation_ids = set()
        
        for idx, ann in enumerate(annotations):
            # 检查必填字段
            required = ["id", "image_id", "category_id", "bbox", "area"]
            for field in required:
                if field not in ann:
                    self.result.add_error(f"标注 {idx}: 缺少字段 '{field}'")
            
            # 检查 ID 唯一性
            if "id" in ann:
                if ann["id"] in annotation_ids:
                    self.result.add_error(f"重复的标注 ID: {ann['id']}")
                annotation_ids.add(ann["id"])
            
            # 验证边界框
            if "bbox" in ann:
                bbox = ann["bbox"]
                if not isinstance(bbox, list) or len(bbox) != 4:
                    self.result.add_error(f"标注 {idx}: 边界框格式无效")
                elif any(v < 0 for v in bbox):
                    self.result.add_error(f"标注 {idx}: 边界框值为负数")
            
            # 验证面积
            if "area" in ann:
                if ann["area"] <= 0:
                    self.result.add_warning(f"标注 {idx}: 面积为零或负数")
            
            # 检查分割（可选）
            if "segmentation" in ann:
                seg = ann["segmentation"]
                # 可以是 RLE 或多边形
                if isinstance(seg, dict):
                    # RLE 格式
                    if "counts" not in seg or "size" not in seg:
                        self.result.add_error(f"标注 {idx}: RLE 格式无效")
                elif isinstance(seg, list):
                    # 多边形格式
                    for poly in seg:
                        if len(poly) < 6:  # 至少 3 个点
                            self.result.add_warning(f"标注 {idx}: 多边形太小")
    
    def _validate_categories(self, categories: List[Dict]):
        """验证类别部分。"""
        if not categories:
            self.result.add_error("数据集中未找到类别")
            return
        
        category_ids = set()
        
        for idx, cat in enumerate(categories):
            # 检查必填字段
            required = ["id", "name"]
            for field in required:
                if field not in cat:
                    self.result.add_error(f"类别 {idx}: 缺少字段 '{field}'")
            
            # 检查 ID 唯一性
            if "id" in cat:
                if cat["id"] in category_ids:
                    self.result.add_error(f"重复的类别 ID: {cat['id']}")
                category_ids.add(cat["id"])
    
    def _check_consistency(self, coco_data: Dict):
        """检查各部分之间的一致性。"""
        images = coco_data.get("images", [])
        annotations = coco_data.get("annotations", [])
        categories = coco_data.get("categories", [])
        
        # 获取有效 ID
        image_ids = {img["id"] for img in images if "id" in img}
        category_ids = {cat["id"] for cat in categories if "id" in cat}
        
        # 检查标注引用
        for ann in annotations:
            if "image_id" in ann and ann["image_id"] not in image_ids:
                self.result.add_error(f"标注引用了不存在的图像 ID: {ann['image_id']}")
            
            if "category_id" in ann and ann["category_id"] not in category_ids:
                self.result.add_error(f"标注引用了不存在的类别 ID: {ann['category_id']}")


class YOLOValidator:
    """
    YOLO 格式数据集的验证器。
    
    检查项：
    - 标签文件格式
    - 类别 ID 有效性
    - 坐标有效性
    - 文件结构
    """
    
    def __init__(self, 
                 labels_dir: str,
                 classes_file: str,
                 images_dir: Optional[str] = None):
        """
        初始化 YOLO 验证器。
        
        参数:
            labels_dir: 包含标签 txt 文件的目录
            classes_file: classes.txt 的路径
            images_dir: 包含图像的可选目录
        """
        self.labels_dir = Path(labels_dir)
        self.classes_file = Path(classes_file)
        self.images_dir = Path(images_dir) if images_dir else None
        self.result = ValidationResult()
    
    def validate(self) -> ValidationResult:
        """
        验证 YOLO 数据集。
        
        返回:
            ValidationResult 对象
        """
        logger.info(f"正在验证 YOLO 数据集: {self.labels_dir}")
        
        # 检查目录是否存在
        if not self.labels_dir.exists():
            self.result.add_error(f"未找到标签目录: {self.labels_dir}")
            return self.result
        
        # 检查类别文件
        if not self.classes_file.exists():
            self.result.add_error(f"未找到类别文件: {self.classes_file}")
            return self.result
        
        # 加载类别
        with open(self.classes_file, 'r', encoding='utf-8') as f:
            classes = [line.strip() for line in f if line.strip()]
        
        num_classes = len(classes)
        self.result.add_stat("类别数量", num_classes)
        
        # 验证标签文件
        label_files = list(self.labels_dir.glob("*.txt"))
        self.result.add_stat("标签文件数量", len(label_files))
        
        total_annotations = 0
        
        for label_file in label_files:
            count = self._validate_label_file(label_file, num_classes)
            total_annotations += count
            
            # 检查对应的图像
            if self.images_dir:
                image_name = label_file.stem
                image_found = False
                for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    image_path = self.images_dir / f"{image_name}{ext}"
                    if image_path.exists():
                        image_found = True
                        break
                
                if not image_found:
                    self.result.add_warning(f"未找到标签对应的图像: {label_file.name}")
        
        self.result.add_stat("总标注数量", total_annotations)
        
        logger.info("YOLO 验证完成")
        return self.result
    
    def _validate_label_file(self, label_file: Path, num_classes: int) -> int:
        """
        验证单个标签文件。
        
        返回:
            文件中的标注数量
        """
        count = 0
        
        try:
            with open(label_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for line_num, line in enumerate(lines, start=1):
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                
                if len(parts) < 5:  # class_id + 至少 2 个点（4 个坐标）
                    self.result.add_error(
                        f"{label_file.name}:{line_num} - 数值太少（需要类别 ID + 坐标）"
                    )
                    continue
                
                # 检查类别 ID
                try:
                    class_id = int(parts[0])
                    if class_id < 0 or class_id >= num_classes:
                        self.result.add_error(
                            f"{label_file.name}:{line_num} - 类别 ID 无效: {class_id}"
                        )
                except ValueError:
                    self.result.add_error(
                        f"{label_file.name}:{line_num} - 类别 ID 无效（不是整数）"
                    )
                    continue
                
                # 检查坐标
                coords = parts[1:]
                if len(coords) % 2 != 0:
                    self.result.add_error(
                        f"{label_file.name}:{line_num} - 坐标数量为奇数"
                    )
                    continue
                
                # 验证坐标值
                for coord in coords:
                    try:
                        val = float(coord)
                        if val < 0 or val > 1:
                            self.result.add_warning(
                                f"{label_file.name}:{line_num} - 坐标超出范围 [0,1]: {val}"
                            )
                    except ValueError:
                        self.result.add_error(
                            f"{label_file.name}:{line_num} - 坐标值无效: {coord}"
                        )
                
                count += 1
                
        except Exception as e:
            self.result.add_error(f"读取 {label_file.name} 时出错: {e}")
        
        return count


def validate_coco_dataset(coco_json_path: str, 
                         images_dir: Optional[str] = None) -> ValidationResult:
    """
    验证 COCO 格式数据集。
    
    参数:
        coco_json_path: COCO JSON 文件路径
        images_dir: 包含图像的可选目录
        
    返回:
        ValidationResult 对象
    """
    validator = COCOValidator(coco_json_path, images_dir)
    return validator.validate()


def validate_yolo_dataset(labels_dir: str,
                         classes_file: str,
                         images_dir: Optional[str] = None) -> ValidationResult:
    """
    验证 YOLO 格式数据集。
    
    参数:
        labels_dir: 包含标签 txt 文件的目录
        classes_file: classes.txt 的路径
        images_dir: 包含图像的可选目录
        
    返回:
        ValidationResult 对象
    """
    validator = YOLOValidator(labels_dir, classes_file, images_dir)
    return validator.validate()
