"""
AnnotationManager 的单元测试。

测试内容：
- 掩码创建和编辑
- 撤销/重做功能
- 掩码持久化
- 导出操作
"""

import pytest
import numpy as np
from pathlib import Path

from src.core.annotation_manager import AnnotationManager


class TestAnnotationManagerInit:
    """AnnotationManager 初始化的测试。"""
    
    @pytest.mark.unit
    def test_init_default(self):
        """测试 AnnotationManager 初始化。"""
        am = AnnotationManager(max_history=50)
        
        assert am.max_history == 50
        assert am.image_path is None
        assert am.current_mask is None
        assert len(am.history) == 0
    
    @pytest.mark.unit
    def test_set_image(self, sample_image):
        """测试设置标注图像。"""
        am = AnnotationManager()
        am.set_image("test_image.png", sample_image.shape)
        
        assert am.image_path == "test_image.png"
        assert am.image_shape == sample_image.shape[:2]
        assert am.current_mask is not None
        assert am.current_mask.shape == sample_image.shape[:2]
        assert np.all(am.current_mask == 0)  # 初始为空


class TestMaskOperations:
    """掩码操作的测试。"""
    
    @pytest.mark.unit
    def test_get_current_mask(self, sample_image):
        """测试获取当前掩码。"""
        am = AnnotationManager()
        am.set_image("test.png", sample_image.shape)
        
        mask = am.get_current_mask()
        
        assert mask is not None
        assert mask.shape == sample_image.shape[:2]
        # 应该是副本，而不是引用
        mask[0, 0] = 255
        assert am.current_mask[0, 0] == 0
    
    @pytest.mark.unit
    def test_set_mask(self, sample_image, sample_mask):
        """测试设置掩码。"""
        am = AnnotationManager()
        am.set_image("test.png", sample_image.shape)
        
        am.set_mask(sample_mask)
        
        current_mask = am.get_current_mask()
        assert np.array_equal(current_mask, sample_mask)
    
    @pytest.mark.unit
    def test_update_mask_replace(self, sample_image, sample_mask):
        """测试使用替换操作更新掩码。"""
        am = AnnotationManager()
        am.set_image("test.png", sample_image.shape)
        
        am.update_mask(sample_mask, operation='replace')
        
        current_mask = am.get_current_mask()
        assert np.array_equal(current_mask, sample_mask)
    
    @pytest.mark.unit
    def test_update_mask_add(self, sample_image):
        """测试使用添加操作更新掩码。"""
        am = AnnotationManager()
        am.set_image("test.png", sample_image.shape)
        
        # 创建两个不重叠的掩码
        mask1 = np.zeros(sample_image.shape[:2], dtype=np.uint8)
        mask1[50:100, 50:100] = 255
        
        mask2 = np.zeros(sample_image.shape[:2], dtype=np.uint8)
        mask2[150:200, 150:200] = 255
        
        am.update_mask(mask1, operation='replace')
        am.update_mask(mask2, operation='add')
        
        current_mask = am.get_current_mask()
        # 两个区域都应该被填充
        assert np.all(current_mask[50:100, 50:100] == 255)
        assert np.all(current_mask[150:200, 150:200] == 255)
    
    @pytest.mark.unit
    def test_update_mask_subtract(self, sample_image, sample_mask):
        """测试使用减法操作更新掩码。"""
        am = AnnotationManager()
        am.set_image("test.png", sample_image.shape)
        
        # 设置初始掩码
        am.update_mask(sample_mask, operation='replace')
        
        # 减去一部分
        subtract_mask = np.zeros(sample_image.shape[:2], dtype=np.uint8)
        subtract_mask[60:90, 60:90] = 255
        
        am.update_mask(subtract_mask, operation='subtract')
        
        current_mask = am.get_current_mask()
        # 减去的区域应该为空
        assert np.all(current_mask[60:90, 60:90] == 0)
    
    @pytest.mark.unit
    def test_clear_mask(self, sample_image, sample_mask):
        """测试清除掩码。"""
        am = AnnotationManager()
        am.set_image("test.png", sample_image.shape)
        am.set_mask(sample_mask)
        
        am.clear_mask()
        
        current_mask = am.get_current_mask()
        assert np.all(current_mask == 0)


class TestPaintingOperations:
    """绘制操作的测试。"""
    
    @pytest.mark.unit
    def test_paint_mask(self, sample_image):
        """测试使用画笔绘制掩码。"""
        am = AnnotationManager()
        am.set_image("test.png", sample_image.shape)
        
        # 绘制一些点
        points = [(100, 100), (101, 100), (102, 100)]
        am.paint_mask(points, brush_size=5, value=255, operation='paint')
        
        current_mask = am.get_current_mask()
        # 检查点是否已绘制
        assert current_mask[100, 100] == 255
    
    @pytest.mark.unit
    def test_erase_mask(self, sample_image, sample_mask):
        """测试擦除掩码。"""
        am = AnnotationManager()
        am.set_image("test.png", sample_image.shape)
        am.set_mask(sample_mask)
        
        # 擦除一些点
        points = [(60, 60), (61, 60), (62, 60)]
        am.paint_mask(points, brush_size=10, operation='erase')
        
        current_mask = am.get_current_mask()
        # 擦除的区域应该被清除
        assert current_mask[60, 60] == 0
    
    @pytest.mark.unit
    def test_paint_polygon(self, sample_image):
        """测试绘制填充多边形。"""
        am = AnnotationManager()
        am.set_image("test.png", sample_image.shape)
        
        # 绘制三角形
        points = [(100, 100), (150, 100), (125, 150)]
        am.paint_polygon(points, value=255)
        
        current_mask = am.get_current_mask()
        # 检查多边形是否已填充
        assert np.sum(current_mask > 0) > 0
    
    @pytest.mark.unit
    def test_finish_paint_stroke(self, sample_image):
        """测试完成绘制笔触会保存状态。"""
        am = AnnotationManager()
        am.set_image("test.png", sample_image.shape)
        
        history_len_before = len(am.history)
        
        # 绘制并完成笔触
        points = [(100, 100)]
        am.paint_mask(points, brush_size=5, value=255)
        am.finish_paint_stroke()
        
        history_len_after = len(am.history)
        assert history_len_after == history_len_before + 1


class TestUndoRedo:
    """撤销/重做功能的测试。"""
    
    @pytest.mark.unit
    def test_undo(self, sample_image, sample_mask):
        """测试撤销操作。"""
        am = AnnotationManager()
        am.set_image("test.png", sample_image.shape)
        
        # 初始状态（为空）
        initial_mask = am.get_current_mask().copy()
        
        # 进行更改
        am.set_mask(sample_mask)
        
        # 撤销
        success = am.undo()
        
        assert success is True
        current_mask = am.get_current_mask()
        assert np.array_equal(current_mask, initial_mask)
    
    @pytest.mark.unit
    def test_undo_limit(self, sample_image):
        """测试在历史记录开头撤销。"""
        am = AnnotationManager()
        am.set_image("test.png", sample_image.shape)
        
        # 尝试在开始时撤销
        success = am.undo()
        
        # 不应该能够撤销初始状态
        assert success is False
    
    @pytest.mark.unit
    def test_redo(self, sample_image, sample_mask):
        """测试重做操作。"""
        am = AnnotationManager()
        am.set_image("test.png", sample_image.shape)
        
        # 进行更改
        am.set_mask(sample_mask)
        changed_mask = am.get_current_mask().copy()
        
        # 撤销
        am.undo()
        
        # 重做
        success = am.redo()
        
        assert success is True
        current_mask = am.get_current_mask()
        assert np.array_equal(current_mask, changed_mask)
    
    @pytest.mark.unit
    def test_redo_limit(self, sample_image):
        """测试在历史记录末尾重做。"""
        am = AnnotationManager()
        am.set_image("test.png", sample_image.shape)
        
        # 尝试在没有撤销的情况下重做
        success = am.redo()
        
        assert success is False
    
    @pytest.mark.unit
    def test_can_undo_redo(self, sample_image, sample_mask):
        """测试撤销/重做可用性检查。"""
        am = AnnotationManager()
        am.set_image("test.png", sample_image.shape)
        
        # 初始不能撤销（只有一个状态）
        assert am.can_undo() is False
        assert am.can_redo() is False
        
        # 进行更改
        am.set_mask(sample_mask)
        
        # 现在可以撤销
        assert am.can_undo() is True
        assert am.can_redo() is False
        
        # 撤销
        am.undo()
        
        # 现在可以重做
        assert am.can_undo() is False
        assert am.can_redo() is True


class TestMaskPersistence:
    """掩码保存/加载操作的测试。"""
    
    @pytest.mark.unit
    def test_save_mask(self, sample_image, sample_mask, temp_output_dir):
        """测试保存掩码。"""
        am = AnnotationManager()
        am.set_image("test.png", sample_image.shape)
        am.set_mask(sample_mask)
        
        mask_path = temp_output_dir / "saved_mask.png"
        success = am.save_mask(str(mask_path))
        
        assert success is True
        assert mask_path.exists()
    
    @pytest.mark.unit
    def test_load_mask(self, sample_image, sample_mask, temp_output_dir):
        """测试加载掩码。"""
        # 首先保存一个掩码
        from src.utils.mask_utils import save_mask
        mask_path = temp_output_dir / "test_mask.png"
        save_mask(sample_mask, str(mask_path))
        
        # 加载掩码
        am = AnnotationManager()
        am.set_image("test.png", sample_image.shape)
        success = am.load_mask(str(mask_path))
        
        assert success is True
        current_mask = am.get_current_mask()
        assert np.array_equal(current_mask, sample_mask)
    
    @pytest.mark.unit
    def test_load_nonexistent_mask(self, sample_image):
        """测试加载不存在的掩码。"""
        am = AnnotationManager()
        am.set_image("test.png", sample_image.shape)
        success = am.load_mask("nonexistent_mask.png")
        
        assert success is False


class TestAnnotationExport:
    """标注导出操作的测试。"""
    
    @pytest.mark.unit
    def test_export_coco_annotation(self, sample_image, sample_mask):
        """测试导出为 COCO 格式。"""
        am = AnnotationManager()
        am.set_image("test_image.png", sample_image.shape)
        am.set_mask(sample_mask)
        
        coco_ann = am.export_coco_annotation()
        
        assert 'id' in coco_ann
        assert 'image_id' in coco_ann
        assert 'category_id' in coco_ann
        assert 'bbox' in coco_ann
        assert 'area' in coco_ann
        assert 'segmentation' in coco_ann
        
        # 检查边界框格式 [x, y, width, height]
        assert len(coco_ann['bbox']) == 4
    
    @pytest.mark.unit
    def test_export_yolo_annotation(self, sample_image, sample_mask):
        """测试导出为 YOLO 格式。"""
        am = AnnotationManager()
        am.set_image("test_image.png", sample_image.shape)
        am.set_mask(sample_mask)
        
        yolo_anns = am.export_yolo_annotation(class_id=0)
        
        assert isinstance(yolo_anns, list)
        assert len(yolo_anns) > 0
        
        # 检查 YOLO 格式 (class_id x1 y1 x2 y2 ...)
        for ann in yolo_anns:
            parts = ann.split()
            assert len(parts) >= 7  # class_id + 至少 3 个点 (6 个坐标)
            assert parts[0] == '0'  # class_id
    
    @pytest.mark.unit
    def test_export_empty_mask(self, sample_image, empty_mask):
        """测试导出空掩码。"""
        am = AnnotationManager()
        am.set_image("test_image.png", sample_image.shape)
        am.set_mask(empty_mask)
        
        coco_ann = am.export_coco_annotation()
        yolo_anns = am.export_yolo_annotation()
        
        # 空掩码应返回空/最小标注
        assert coco_ann == {} or coco_ann.get('area', 0) == 0
        assert yolo_anns == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
