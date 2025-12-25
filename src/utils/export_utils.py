"""
标注数据的导出实用程序。

此模块提供：
- COCO 格式数据集导出
- YOLO 格式数据集导出
- VOC/Pascal 格式数据集导出
- 数据集转换实用程序
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import numpy as np

from src.logger import get_logger
from src.utils.mask_utils import binary_mask_to_rle, mask_to_bbox, mask_to_polygon
from src.utils.file_utils import ensure_dir, save_json

logger = get_logger(__name__)


class COCOExporter:
    """
    将标注导出为 COCO JSON 格式。
    
    COCO 格式结构：
    {
        "info": {...},
        "licenses": [...],
        "images": [...],
        "annotations": [...],
        "categories": [...]
    }
    """
    
    def __init__(self, 
                 dataset_name: str = "工业缺陷数据集",
                 description: str = "用于工业缺陷分割的数据集",
                 version: str = "1.0"):
        """
        初始化 COCO 导出器。
        
        参数:
            dataset_name: 数据集名称
            description: 数据集描述
            version: 数据集版本
        """
        self.dataset_name = dataset_name
        self.description = description
        self.version = version
        
        # 初始化 COCO 结构
        self.coco_data = {
            "info": {
                "description": description,
                "version": version,
                "year": datetime.now().year,
                "contributor": "工业 AI 团队",
                "date_created": datetime.now().isoformat()
            },
            "licenses": [
                {
                    "id": 1,
                    "name": "自定义许可证",
                    "url": ""
                }
            ],
            "images": [],
            "annotations": [],
            "categories": []
        }
        
        self.image_id_counter = 1
        self.annotation_id_counter = 1
        self.category_dict = {}  # 类别名称 -> 类别 ID
    
    def add_category(self, category_name: str, supercategory: str = "defect") -> int:
        """
        向数据集添加类别。
        
        参数:
            category_name: 类别名称
            supercategory: 父类别名称
            
        返回:
            类别 ID
        """
        if category_name in self.category_dict:
            return self.category_dict[category_name]
        
        category_id = len(self.category_dict) + 1
        self.category_dict[category_name] = category_id
        
        self.coco_data["categories"].append({
            "id": category_id,
            "name": category_name,
            "supercategory": supercategory
        })
        
        logger.debug(f"已添加类别: {category_name} (id={category_id})")
        return category_id
    
    def add_image(self, 
                  image_path: str,
                  width: int,
                  height: int,
                  image_id: Optional[int] = None) -> int:
        """
        向数据集添加图像。
        
        参数:
            image_path: 图像文件路径
            width: 图像宽度
            height: 图像高度
            image_id: 可选的自定义图像 ID
            
        返回:
            图像 ID
        """
        if image_id is None:
            image_id = self.image_id_counter
            self.image_id_counter += 1
        
        image_filename = Path(image_path).name
        
        self.coco_data["images"].append({
            "id": image_id,
            "file_name": image_filename,
            "width": width,
            "height": height,
            "license": 1,
            "date_captured": datetime.now().isoformat()
        })
        
        logger.debug(f"已添加图像: {image_filename} (id={image_id})")
        return image_id
    
    def add_annotation(self,
                       image_id: int,
                       category_id: int,
                       mask: np.ndarray,
                       annotation_id: Optional[int] = None) -> int:
        """
        向数据集添加标注。
        
        参数:
            image_id: 图像 ID
            category_id: 类别 ID
            mask: 二值掩码 (H, W)
            annotation_id: 可选的自定义标注 ID
            
        返回:
            标注 ID
        """
        if annotation_id is None:
            annotation_id = self.annotation_id_counter
            self.annotation_id_counter += 1
        
        # 获取边界框
        bbox = mask_to_bbox(mask)
        if bbox is None:
            logger.warning(f"标注 {annotation_id} 的掩码为空，正在跳过")
            return annotation_id
        
        # 获取 RLE 分割
        rle = binary_mask_to_rle(mask)
        
        # 计算面积
        area = int(np.sum(mask > 0))
        
        annotation = {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": bbox,  # [x, y, width, height]
            "area": area,
            "segmentation": rle,
            "iscrowd": 0
        }
        
        self.coco_data["annotations"].append(annotation)
        logger.debug(f"已添加标注: id={annotation_id}, image_id={image_id}, area={area}")
        return annotation_id
    
    def save(self, output_path: str):
        """
        将 COCO 数据集保存到 JSON 文件。
        
        参数:
            output_path: 保存 JSON 文件的路径
        """
        ensure_dir(Path(output_path).parent)
        save_json(self.coco_data, output_path, indent=2)
        logger.info(f"已将 COCO 数据集保存到: {output_path}")
        logger.info(f"  图像数: {len(self.coco_data['images'])}")
        logger.info(f"  标注数: {len(self.coco_data['annotations'])}")
        logger.info(f"  类别数: {len(self.coco_data['categories'])}")
    
    def get_statistics(self) -> Dict:
        """
        获取数据集统计信息。
        
        返回:
            包含统计信息的字典
        """
        stats = {
            "num_images": len(self.coco_data["images"]),
            "num_annotations": len(self.coco_data["annotations"]),
            "num_categories": len(self.coco_data["categories"]),
            "category_names": [cat["name"] for cat in self.coco_data["categories"]]
        }
        
        # 计算每个类别的标注数
        category_counts = {}
        for ann in self.coco_data["annotations"]:
            cat_id = ann["category_id"]
            category_counts[cat_id] = category_counts.get(cat_id, 0) + 1
        
        stats["annotations_per_category"] = category_counts
        
        return stats


class YOLOExporter:
    """
    将标注导出为 YOLO 格式。
    
    YOLO 格式：
    - 每张图像一个 txt 文件
    - 每行：class_id x1 y1 x2 y2 x3 y3 ...（归一化的多边形坐标）
    - 包含类别名称的 classes.txt 文件
    """
    
    def __init__(self, output_dir: str, class_names: List[str]):
        """
        初始化 YOLO 导出器。
        
        参数:
            output_dir: YOLO 文件的输出目录
            class_names: 按顺序排列的类别名称列表
        """
        self.output_dir = Path(output_dir)
        self.class_names = class_names
        
        # 创建输出目录
        ensure_dir(self.output_dir)
        
        # 保存 classes.txt
        classes_file = self.output_dir / "classes.txt"
        with open(classes_file, 'w', encoding='utf-8') as f:
            for class_name in class_names:
                f.write(f"{class_name}\n")
        
        logger.info(f"已创建 YOLO 导出器: {output_dir}")
        logger.info(f"类别: {class_names}")
    
    def export_annotation(self,
                         image_path: str,
                         masks: List[np.ndarray],
                         class_ids: List[int],
                         image_width: int,
                         image_height: int):
        """
        将一张图像的标注导出为 YOLO 格式。
        
        参数:
            image_path: 图像文件路径
            masks: 二值掩码列表
            class_ids: 类别 ID 列表（从 0 开始）
            image_width: 图像宽度
            image_height: 图像高度
        """
        image_name = Path(image_path).stem
        output_file = self.output_dir / f"{image_name}.txt"
        
        lines = []
        
        for mask, class_id in zip(masks, class_ids):
            # 从掩码获取多边形
            polygons = mask_to_polygon(mask)
            
            if not polygons:
                logger.warning(f"在 {image_name} 的掩码中未找到多边形")
                continue
            
            for polygon in polygons:
                if len(polygon) < 6:  # 至少需要 3 个点
                    continue
                
                # 将坐标归一化到 [0, 1]
                normalized_polygon = []
                for i in range(0, len(polygon), 2):
                    x = polygon[i] / image_width
                    y = polygon[i + 1] / image_height
                    normalized_polygon.extend([x, y])
                
                # 格式: class_id x1 y1 x2 y2 x3 y3 ...
                line = f"{class_id} " + " ".join([f"{coord:.6f}" for coord in normalized_polygon])
                lines.append(line)
        
        # 写入文件
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
        
        logger.debug(f"已导出 YOLO 标注: {output_file} ({len(lines)} 个对象)")
    
    def create_data_yaml(self, 
                        train_path: str,
                        val_path: str,
                        test_path: Optional[str] = None):
        """
        为 YOLO 训练创建 data.yaml。
        
        参数:
            train_path: 训练图像目录路径
            val_path: 验证图像目录路径
            test_path: 可选的测试图像目录路径
        """
        data_yaml = {
            'path': str(self.output_dir.absolute()),
            'train': train_path,
            'val': val_path,
            'names': {i: name for i, name in enumerate(self.class_names)}
        }
        
        if test_path:
            data_yaml['test'] = test_path
        
        yaml_path = self.output_dir / "data.yaml"
        
        # 手动写入 YAML（简单格式）
        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write(f"path: {data_yaml['path']}\n")
            f.write(f"train: {data_yaml['train']}\n")
            f.write(f"val: {data_yaml['val']}\n")
            if test_path:
                f.write(f"test: {data_yaml['test']}\n")
            f.write("\nnames:\n")
            for idx, name in data_yaml['names'].items():
                f.write(f"  {idx}: {name}\n")
        
        logger.info(f"已创建 data.yaml: {yaml_path}")


class VOCExporter:
    """
    将标注导出为 Pascal VOC XML 格式。
    
    VOC 格式：
    - 每张图像一个 XML 文件
    - 包含图像信息和边界框
    - 分割掩码单独存储为 PNG
    """
    
    def __init__(self, output_dir: str):
        """
        初始化 VOC 导出器。
        
        参数:
            output_dir: VOC 文件的输出目录
        """
        self.output_dir = Path(output_dir)
        self.annotations_dir = self.output_dir / "Annotations"
        self.segmentation_dir = self.output_dir / "SegmentationClass"
        
        # 创建目录
        ensure_dir(self.annotations_dir)
        ensure_dir(self.segmentation_dir)
        
        logger.info(f"已创建 VOC 导出器: {output_dir}")
    
    def export_annotation(self,
                         image_path: str,
                         masks: List[np.ndarray],
                         class_names: List[str],
                         image_width: int,
                         image_height: int):
        """
        将一张图像的标注导出为 VOC 格式。
        
        参数:
            image_path: 图像文件路径
            masks: 二值掩码列表
            class_names: 每个掩码的类别名称列表
            image_width: 图像宽度
            image_height: 图像高度
        """
        image_name = Path(image_path).stem
        
        # 创建 XML 标注
        annotation = ET.Element("annotation")
        
        # 添加文件夹
        folder = ET.SubElement(annotation, "folder")
        folder.text = "VOC2012"
        
        # 添加文件名
        filename = ET.SubElement(annotation, "filename")
        filename.text = Path(image_path).name
        
        # 添加来源
        source = ET.SubElement(annotation, "source")
        database = ET.SubElement(source, "database")
        database.text = "工业缺陷数据集"
        
        # 添加尺寸
        size = ET.SubElement(annotation, "size")
        width_elem = ET.SubElement(size, "width")
        width_elem.text = str(image_width)
        height_elem = ET.SubElement(size, "height")
        height_elem.text = str(image_height)
        depth_elem = ET.SubElement(size, "depth")
        depth_elem.text = "3"
        
        # 添加分割标志
        segmented = ET.SubElement(annotation, "segmented")
        segmented.text = "1"
        
        # 添加对象
        for mask, class_name in zip(masks, class_names):
            bbox = mask_to_bbox(mask)
            if bbox is None:
                continue
            
            x, y, w, h = bbox
            xmin, ymin = int(x), int(y)
            xmax, ymax = int(x + w), int(y + h)
            
            obj = ET.SubElement(annotation, "object")
            
            name_elem = ET.SubElement(obj, "name")
            name_elem.text = class_name
            
            pose = ET.SubElement(obj, "pose")
            pose.text = "未指定"
            
            truncated = ET.SubElement(obj, "truncated")
            truncated.text = "0"
            
            difficult = ET.SubElement(obj, "difficult")
            difficult.text = "0"
            
            bndbox = ET.SubElement(obj, "bndbox")
            xmin_elem = ET.SubElement(bndbox, "xmin")
            xmin_elem.text = str(xmin)
            ymin_elem = ET.SubElement(bndbox, "ymin")
            ymin_elem.text = str(ymin)
            xmax_elem = ET.SubElement(bndbox, "xmax")
            xmax_elem.text = str(xmax)
            ymax_elem = ET.SubElement(bndbox, "ymax")
            ymax_elem.text = str(ymax)
        
        # 写入 XML 文件
        tree = ET.ElementTree(annotation)
        xml_path = self.annotations_dir / f"{image_name}.xml"
        tree.write(xml_path, encoding='utf-8', xml_declaration=True)
        
        # 保存分割掩码
        if masks:
            # 将所有掩码合并为一张图像（每个类别具有不同的像素值）
            combined_mask = np.zeros((image_height, image_width), dtype=np.uint8)
            for i, mask in enumerate(masks, start=1):
                combined_mask[mask > 0] = i
            
            from src.utils.mask_utils import save_mask
            mask_path = self.segmentation_dir / f"{image_name}.png"
            save_mask(combined_mask, str(mask_path))
        
        logger.debug(f"已导出 VOC 标注: {xml_path}")


def batch_export_coco(image_paths: List[str],
                     mask_paths: List[str],
                     category_names: List[str],
                     output_path: str,
                     dataset_name: str = "工业缺陷数据集") -> Dict:
    """
    批量将标注导出为 COCO 格式。
    
    参数:
        image_paths: 图像文件路径列表
        mask_paths: 掩码文件路径列表（与图像顺序相同）
        category_names: 每个掩码的类别名称列表
        output_path: 输出 JSON 文件路径
        dataset_name: 数据集名称
        
    返回:
        导出统计信息
    """
    exporter = COCOExporter(dataset_name=dataset_name)
    
    # 添加类别
    unique_categories = list(set(category_names))
    for cat_name in unique_categories:
        exporter.add_category(cat_name)
    
    # 处理每张图像
    from src.utils.image_utils import get_image_info
    from src.utils.mask_utils import load_mask
    
    for image_path, mask_path, category_name in zip(image_paths, mask_paths, category_names):
        # 获取图像信息
        info = get_image_info(image_path)
        if info is None:
            logger.warning(f"获取信息失败: {image_path}")
            continue
        
        width, height = info['width'], info['height']
        
        # 添加图像
        image_id = exporter.add_image(image_path, width, height)
        
        # 加载并添加掩码
        mask = load_mask(mask_path)
        if mask is None:
            logger.warning(f"加载掩码失败: {mask_path}")
            continue
        
        category_id = exporter.category_dict[category_name]
        exporter.add_annotation(image_id, category_id, mask)
    
    # 保存
    exporter.save(output_path)
    
    return exporter.get_statistics()


def batch_export_yolo(image_paths: List[str],
                     mask_paths: List[str],
                     class_ids: List[int],
                     class_names: List[str],
                     output_dir: str) -> int:
    """
    批量将标注导出为 YOLO 格式。
    
    参数:
        image_paths: 图像文件路径列表
        mask_paths: 掩码文件路径列表
        class_ids: 每个掩码的类别 ID 列表
        class_names: 所有类别名称列表
        output_dir: 输出目录
        
    返回:
        导出的标注数量
    """
    exporter = YOLOExporter(output_dir, class_names)
    
    from src.utils.image_utils import get_image_info
    from src.utils.mask_utils import load_mask
    
    count = 0
    
    for image_path, mask_path, class_id in zip(image_paths, mask_paths, class_ids):
        # 获取图像信息
        info = get_image_info(image_path)
        if info is None:
            continue
        
        width, height = info['width'], info['height']
        
        # 加载掩码
        mask = load_mask(mask_path)
        if mask is None:
            continue
        
        # 导出
        exporter.export_annotation(
            image_path,
            [mask],
            [class_id],
            width,
            height
        )
        count += 1
    
    logger.info(f"已导出 {count} 个 YOLO 标注")
    return count
