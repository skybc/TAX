"""
mask_utils 模块的单元测试。

测试内容：
- RLE 编码/解码
- 多边形转换
- 边界框操作
- 形态学操作
- 掩码指标
"""

import pytest
import numpy as np
import cv2

from src.utils.mask_utils import (
    binary_mask_to_rle, rle_to_binary_mask,
    polygon_to_mask, mask_to_polygon,
    mask_to_bbox, bbox_to_mask,
    dilate_mask, erode_mask, open_mask, close_mask,
    remove_small_components, get_largest_component,
    fill_holes, overlay_mask_on_image,
    compute_mask_area, compute_mask_iou,
    save_mask, load_mask
)


class TestRLEOperations:
    """RLE 编码/解码测试。"""
    
    @pytest.mark.unit
    def test_rle_encode_decode_empty_mask(self, empty_mask):
        """测试空掩码的 RLE 编码/解码。"""
        rle = binary_mask_to_rle(empty_mask)
        decoded = rle_to_binary_mask(rle)
        
        assert decoded.shape == empty_mask.shape
        assert np.array_equal(decoded, empty_mask)
    
    @pytest.mark.unit
    def test_rle_encode_decode_simple_mask(self, sample_mask):
        """测试带有缺陷的掩码的 RLE 编码/解码。"""
        rle = binary_mask_to_rle(sample_mask)
        decoded = rle_to_binary_mask(rle)
        
        assert decoded.shape == sample_mask.shape
        assert np.array_equal(decoded, sample_mask)
    
    @pytest.mark.unit
    def test_rle_encode_decode_full_mask(self):
        """测试完全填充掩码的 RLE 编码/解码。"""
        mask = np.ones((100, 100), dtype=np.uint8) * 255
        rle = binary_mask_to_rle(mask)
        decoded = rle_to_binary_mask(rle)
        
        assert decoded.shape == mask.shape
        assert np.array_equal(decoded, mask)


class TestPolygonOperations:
    """多边形转换测试。"""
    
    @pytest.mark.unit
    def test_polygon_to_mask_simple(self):
        """测试多边形到掩码的转换。"""
        polygon = [(50, 50), (150, 50), (150, 150), (50, 150)]
        mask = polygon_to_mask(polygon, (200, 200))
        
        assert mask.shape == (200, 200)
        assert mask.dtype == np.uint8
        # 检查多边形区域是否已填充
        assert np.sum(mask > 0) > 0
        # 大致面积检查 (100x100 正方形)
        assert 9000 < np.sum(mask > 0) < 11000
    
    @pytest.mark.unit
    def test_mask_to_polygon(self, sample_mask):
        """测试掩码到多边形的转换。"""
        polygons = mask_to_polygon(sample_mask)
        
        assert isinstance(polygons, list)
        assert len(polygons) > 0  # 应该至少有一个多边形
        
        # 检查多边形格式
        for polygon in polygons:
            assert len(polygon) >= 6  # 至少 3 个点 (x, y 对)
            assert len(polygon) % 2 == 0  # 偶数 (x, y 对)
    
    @pytest.mark.unit
    def test_polygon_to_mask_empty(self):
        """测试无点的多边形到掩码的转换。"""
        polygon = []
        mask = polygon_to_mask(polygon, (100, 100))
        
        assert mask.shape == (100, 100)
        assert np.sum(mask) == 0  # 空掩码


class TestBoundingBoxOperations:
    """边界框操作测试。"""
    
    @pytest.mark.unit
    def test_mask_to_bbox(self, sample_mask):
        """测试掩码到边界框的转换。"""
        bbox = mask_to_bbox(sample_mask)
        
        assert bbox is not None
        assert len(bbox) == 4
        x, y, w, h = bbox
        
        # 检查合理的值
        assert 0 <= x < sample_mask.shape[1]
        assert 0 <= y < sample_mask.shape[0]
        assert w > 0
        assert h > 0
        assert x + w <= sample_mask.shape[1]
        assert y + h <= sample_mask.shape[0]
    
    @pytest.mark.unit
    def test_mask_to_bbox_empty(self, empty_mask):
        """测试空掩码到边界框的转换。"""
        bbox = mask_to_bbox(empty_mask)
        assert bbox is None
    
    @pytest.mark.unit
    def test_bbox_to_mask(self):
        """测试边界框到掩码的转换。"""
        bbox = (50, 50, 100, 100)  # x, y, w, h
        mask = bbox_to_mask(bbox, (200, 200))
        
        assert mask.shape == (200, 200)
        assert mask.dtype == np.uint8
        
        # 检查边界框区域是否已填充
        expected_area = 100 * 100
        actual_area = np.sum(mask > 0)
        assert actual_area == expected_area
        
        # 检查边界框位置
        assert np.all(mask[50:150, 50:150] == 255)
        assert np.all(mask[0:50, :] == 0)
        assert np.all(mask[:, 0:50] == 0)


class TestMorphologicalOperations:
    """形态学操作测试。"""
    
    @pytest.mark.unit
    def test_dilate_mask(self, sample_mask):
        """测试掩码膨胀。"""
        dilated = dilate_mask(sample_mask, kernel_size=5)
        
        assert dilated.shape == sample_mask.shape
        assert dilated.dtype == sample_mask.dtype
        # 膨胀应该增加掩码面积
        assert np.sum(dilated > 0) > np.sum(sample_mask > 0)
    
    @pytest.mark.unit
    def test_erode_mask(self, sample_mask):
        """测试掩码腐蚀。"""
        eroded = erode_mask(sample_mask, kernel_size=5)
        
        assert eroded.shape == sample_mask.shape
        assert eroded.dtype == sample_mask.dtype
        # 腐蚀应该减少掩码面积
        assert np.sum(eroded > 0) < np.sum(sample_mask > 0)
    
    @pytest.mark.unit
    def test_open_mask(self, sample_mask):
        """测试形态学开运算（先腐蚀后膨胀）。"""
        opened = open_mask(sample_mask, kernel_size=3)
        
        assert opened.shape == sample_mask.shape
        assert opened.dtype == sample_mask.dtype
        # 开运算移除细小噪声
        assert np.sum(opened > 0) <= np.sum(sample_mask > 0)
    
    @pytest.mark.unit
    def test_close_mask(self, sample_mask):
        """测试形态学闭运算（先膨胀后腐蚀）。"""
        closed = close_mask(sample_mask, kernel_size=3)
        
        assert closed.shape == sample_mask.shape
        assert closed.dtype == sample_mask.dtype
        # 闭运算填充细小孔洞
        assert np.sum(closed > 0) >= np.sum(sample_mask > 0)


class TestComponentOperations:
    """连通分量操作测试。"""
    
    @pytest.mark.unit
    def test_remove_small_components(self, sample_mask_with_multiple_defects):
        """测试移除细小分量。"""
        # 移除小于 500 像素的分量
        filtered = remove_small_components(sample_mask_with_multiple_defects, min_size=500)
        
        assert filtered.shape == sample_mask_with_multiple_defects.shape
        assert filtered.dtype == sample_mask_with_multiple_defects.dtype
        # 像素应该减少（移除细小分量）
        assert np.sum(filtered > 0) < np.sum(sample_mask_with_multiple_defects > 0)
    
    @pytest.mark.unit
    def test_get_largest_component(self, sample_mask_with_multiple_defects):
        """测试提取最大分量。"""
        largest = get_largest_component(sample_mask_with_multiple_defects)
        
        assert largest.shape == sample_mask_with_multiple_defects.shape
        assert largest.dtype == sample_mask_with_multiple_defects.dtype
        # 应该只有一个分量
        from scipy import ndimage
        labeled, num = ndimage.label(largest > 0)
        assert num == 1
    
    @pytest.mark.unit
    def test_get_largest_component_empty(self, empty_mask):
        """测试空掩码的最大分量。"""
        largest = get_largest_component(empty_mask)
        
        assert largest.shape == empty_mask.shape
        assert np.sum(largest) == 0


class TestMaskMetrics:
    """掩码指标计算测试。"""
    
    @pytest.mark.unit
    def test_compute_mask_area(self, sample_mask):
        """测试掩码面积计算。"""
        area = compute_mask_area(sample_mask)
        
        assert isinstance(area, int)
        assert area > 0
        # 应该等于非零像素的数量
        assert area == np.sum(sample_mask > 0)
    
    @pytest.mark.unit
    def test_compute_mask_area_empty(self, empty_mask):
        """测试空掩码的面积。"""
        area = compute_mask_area(empty_mask)
        assert area == 0
    
    @pytest.mark.unit
    def test_compute_mask_iou_perfect(self, sample_mask):
        """测试相同掩码的 IoU。"""
        iou = compute_mask_iou(sample_mask, sample_mask)
        assert np.isclose(iou, 1.0)
    
    @pytest.mark.unit
    def test_compute_mask_iou_no_overlap(self, sample_mask, empty_mask):
        """测试无重叠的 IoU。"""
        iou = compute_mask_iou(sample_mask, empty_mask)
        assert np.isclose(iou, 0.0)
    
    @pytest.mark.unit
    def test_compute_mask_iou_partial(self):
        """测试部分重叠的 IoU。"""
        mask1 = np.zeros((100, 100), dtype=np.uint8)
        mask1[25:75, 25:75] = 255  # 50x50 正方形
        
        mask2 = np.zeros((100, 100), dtype=np.uint8)
        mask2[50:100, 50:100] = 255  # 50x50 正方形，偏移
        
        iou = compute_mask_iou(mask1, mask2)
        
        # 交集: 25x25 = 625
        # 并集: 50x50 + 50x50 - 25x25 = 4375
        # IoU = 625/4375 ≈ 0.143
        expected_iou = 625 / 4375
        assert np.isclose(iou, expected_iou, atol=0.01)


class TestMaskIO:
    """掩码保存/加载操作测试。"""
    
    @pytest.mark.unit
    def test_save_load_mask(self, sample_mask, temp_output_dir):
        """测试保存和加载掩码。"""
        mask_path = temp_output_dir / "test_mask.png"
        
        # 保存掩码
        save_mask(sample_mask, str(mask_path))
        assert mask_path.exists()
        
        # 加载掩码
        loaded_mask = load_mask(str(mask_path))
        
        assert loaded_mask is not None
        assert loaded_mask.shape == sample_mask.shape
        assert np.array_equal(loaded_mask, sample_mask)
    
    @pytest.mark.unit
    def test_load_nonexistent_mask(self):
        """测试加载不存在的掩码。"""
        loaded_mask = load_mask("nonexistent_mask.png")
        assert loaded_mask is None
    
    @pytest.mark.unit
    def test_save_mask_creates_directory(self, sample_mask, temp_output_dir):
        """测试 save_mask 在需要时创建目录。"""
        mask_path = temp_output_dir / "subdir" / "nested" / "mask.png"
        
        save_mask(sample_mask, str(mask_path))
        assert mask_path.exists()


class TestOverlayOperations:
    """掩码叠加可视化测试。"""
    
    @pytest.mark.unit
    def test_overlay_mask_on_image(self, sample_image, sample_mask):
        """测试在图像上叠加掩码。"""
        overlay = overlay_mask_on_image(sample_image, sample_mask, alpha=0.5)
        
        assert overlay.shape == sample_image.shape
        assert overlay.dtype == sample_image.dtype
        
        # 检查掩码激活处的叠加层是否与原始图像不同
        mask_pixels = sample_mask > 0
        assert not np.array_equal(overlay[mask_pixels], sample_image[mask_pixels])
    
    @pytest.mark.unit
    def test_overlay_mask_custom_color(self, sample_image, sample_mask):
        """测试使用自定义颜色的叠加。"""
        color = (0, 255, 0)  # 绿色
        overlay = overlay_mask_on_image(sample_image, sample_mask, color=color, alpha=0.7)
        
        assert overlay.shape == sample_image.shape
        # 检查掩码激活处的绿色通道是否增强
        mask_pixels = sample_mask > 0
        assert np.mean(overlay[mask_pixels, 1]) > np.mean(sample_image[mask_pixels, 1])
    
    @pytest.mark.unit
    def test_overlay_empty_mask(self, sample_image, empty_mask):
        """测试叠加空掩码。"""
        overlay = overlay_mask_on_image(sample_image, empty_mask)
        
        # 应该与原始图像相同
        assert np.array_equal(overlay, sample_image)


class TestFillHoles:
    """孔洞填充操作测试。"""
    
    @pytest.mark.unit
    def test_fill_holes(self):
        """测试填充掩码中的孔洞。"""
        # 创建带有孔洞的掩码
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 20:80] = 255  # 外正方形
        mask[40:60, 40:60] = 0    # 内孔洞
        
        filled = fill_holes(mask)
        
        assert filled.shape == mask.shape
        # 孔洞应该被填充
        assert np.all(filled[40:60, 40:60] == 255)
        # 外部边界应该保持不变
        assert np.all(filled[20:80, 20:80] == 255)
    
    @pytest.mark.unit
    def test_fill_holes_no_holes(self, sample_mask):
        """测试对没有孔洞的掩码使用 fill_holes。"""
        filled = fill_holes(sample_mask)
        
        # 应该与原始掩码相似（由于算法原因可能略有不同）
        assert filled.shape == sample_mask.shape
        assert np.sum(filled > 0) >= np.sum(sample_mask > 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
