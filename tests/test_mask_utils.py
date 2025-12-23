"""
Unit tests for mask_utils module.

Tests:
- RLE encoding/decoding
- Polygon conversions
- Bounding box operations
- Morphological operations
- Mask metrics
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
    """Tests for RLE encoding/decoding."""
    
    @pytest.mark.unit
    def test_rle_encode_decode_empty_mask(self, empty_mask):
        """Test RLE encoding/decoding of empty mask."""
        rle = binary_mask_to_rle(empty_mask)
        decoded = rle_to_binary_mask(rle)
        
        assert decoded.shape == empty_mask.shape
        assert np.array_equal(decoded, empty_mask)
    
    @pytest.mark.unit
    def test_rle_encode_decode_simple_mask(self, sample_mask):
        """Test RLE encoding/decoding of mask with defects."""
        rle = binary_mask_to_rle(sample_mask)
        decoded = rle_to_binary_mask(rle)
        
        assert decoded.shape == sample_mask.shape
        assert np.array_equal(decoded, sample_mask)
    
    @pytest.mark.unit
    def test_rle_encode_decode_full_mask(self):
        """Test RLE encoding/decoding of completely filled mask."""
        mask = np.ones((100, 100), dtype=np.uint8) * 255
        rle = binary_mask_to_rle(mask)
        decoded = rle_to_binary_mask(rle)
        
        assert decoded.shape == mask.shape
        assert np.array_equal(decoded, mask)


class TestPolygonOperations:
    """Tests for polygon conversions."""
    
    @pytest.mark.unit
    def test_polygon_to_mask_simple(self):
        """Test polygon to mask conversion."""
        polygon = [(50, 50), (150, 50), (150, 150), (50, 150)]
        mask = polygon_to_mask(polygon, (200, 200))
        
        assert mask.shape == (200, 200)
        assert mask.dtype == np.uint8
        # Check that polygon area is filled
        assert np.sum(mask > 0) > 0
        # Approximate area check (100x100 square)
        assert 9000 < np.sum(mask > 0) < 11000
    
    @pytest.mark.unit
    def test_mask_to_polygon(self, sample_mask):
        """Test mask to polygon conversion."""
        polygons = mask_to_polygon(sample_mask)
        
        assert isinstance(polygons, list)
        assert len(polygons) > 0  # Should have at least one polygon
        
        # Check polygon format
        for polygon in polygons:
            assert len(polygon) >= 6  # At least 3 points (x, y pairs)
            assert len(polygon) % 2 == 0  # Even number (x, y pairs)
    
    @pytest.mark.unit
    def test_polygon_to_mask_empty(self):
        """Test polygon to mask with no points."""
        polygon = []
        mask = polygon_to_mask(polygon, (100, 100))
        
        assert mask.shape == (100, 100)
        assert np.sum(mask) == 0  # Empty mask


class TestBoundingBoxOperations:
    """Tests for bounding box operations."""
    
    @pytest.mark.unit
    def test_mask_to_bbox(self, sample_mask):
        """Test mask to bounding box conversion."""
        bbox = mask_to_bbox(sample_mask)
        
        assert bbox is not None
        assert len(bbox) == 4
        x, y, w, h = bbox
        
        # Check reasonable values
        assert 0 <= x < sample_mask.shape[1]
        assert 0 <= y < sample_mask.shape[0]
        assert w > 0
        assert h > 0
        assert x + w <= sample_mask.shape[1]
        assert y + h <= sample_mask.shape[0]
    
    @pytest.mark.unit
    def test_mask_to_bbox_empty(self, empty_mask):
        """Test mask to bbox with empty mask."""
        bbox = mask_to_bbox(empty_mask)
        assert bbox is None
    
    @pytest.mark.unit
    def test_bbox_to_mask(self):
        """Test bounding box to mask conversion."""
        bbox = (50, 50, 100, 100)  # x, y, w, h
        mask = bbox_to_mask(bbox, (200, 200))
        
        assert mask.shape == (200, 200)
        assert mask.dtype == np.uint8
        
        # Check that bbox area is filled
        expected_area = 100 * 100
        actual_area = np.sum(mask > 0)
        assert actual_area == expected_area
        
        # Check bbox location
        assert np.all(mask[50:150, 50:150] == 255)
        assert np.all(mask[0:50, :] == 0)
        assert np.all(mask[:, 0:50] == 0)


class TestMorphologicalOperations:
    """Tests for morphological operations."""
    
    @pytest.mark.unit
    def test_dilate_mask(self, sample_mask):
        """Test mask dilation."""
        dilated = dilate_mask(sample_mask, kernel_size=5)
        
        assert dilated.shape == sample_mask.shape
        assert dilated.dtype == sample_mask.dtype
        # Dilation should increase mask area
        assert np.sum(dilated > 0) > np.sum(sample_mask > 0)
    
    @pytest.mark.unit
    def test_erode_mask(self, sample_mask):
        """Test mask erosion."""
        eroded = erode_mask(sample_mask, kernel_size=5)
        
        assert eroded.shape == sample_mask.shape
        assert eroded.dtype == sample_mask.dtype
        # Erosion should decrease mask area
        assert np.sum(eroded > 0) < np.sum(sample_mask > 0)
    
    @pytest.mark.unit
    def test_open_mask(self, sample_mask):
        """Test morphological opening (erosion then dilation)."""
        opened = open_mask(sample_mask, kernel_size=3)
        
        assert opened.shape == sample_mask.shape
        assert opened.dtype == sample_mask.dtype
        # Opening removes small noise
        assert np.sum(opened > 0) <= np.sum(sample_mask > 0)
    
    @pytest.mark.unit
    def test_close_mask(self, sample_mask):
        """Test morphological closing (dilation then erosion)."""
        closed = close_mask(sample_mask, kernel_size=3)
        
        assert closed.shape == sample_mask.shape
        assert closed.dtype == sample_mask.dtype
        # Closing fills small holes
        assert np.sum(closed > 0) >= np.sum(sample_mask > 0)


class TestComponentOperations:
    """Tests for connected component operations."""
    
    @pytest.mark.unit
    def test_remove_small_components(self, sample_mask_with_multiple_defects):
        """Test removal of small components."""
        # Remove components smaller than 500 pixels
        filtered = remove_small_components(sample_mask_with_multiple_defects, min_size=500)
        
        assert filtered.shape == sample_mask_with_multiple_defects.shape
        assert filtered.dtype == sample_mask_with_multiple_defects.dtype
        # Should have fewer pixels (small components removed)
        assert np.sum(filtered > 0) < np.sum(sample_mask_with_multiple_defects > 0)
    
    @pytest.mark.unit
    def test_get_largest_component(self, sample_mask_with_multiple_defects):
        """Test extraction of largest component."""
        largest = get_largest_component(sample_mask_with_multiple_defects)
        
        assert largest.shape == sample_mask_with_multiple_defects.shape
        assert largest.dtype == sample_mask_with_multiple_defects.dtype
        # Should have only one component
        from scipy import ndimage
        labeled, num = ndimage.label(largest > 0)
        assert num == 1
    
    @pytest.mark.unit
    def test_get_largest_component_empty(self, empty_mask):
        """Test largest component with empty mask."""
        largest = get_largest_component(empty_mask)
        
        assert largest.shape == empty_mask.shape
        assert np.sum(largest) == 0


class TestMaskMetrics:
    """Tests for mask metric calculations."""
    
    @pytest.mark.unit
    def test_compute_mask_area(self, sample_mask):
        """Test mask area computation."""
        area = compute_mask_area(sample_mask)
        
        assert isinstance(area, int)
        assert area > 0
        # Should equal number of non-zero pixels
        assert area == np.sum(sample_mask > 0)
    
    @pytest.mark.unit
    def test_compute_mask_area_empty(self, empty_mask):
        """Test mask area with empty mask."""
        area = compute_mask_area(empty_mask)
        assert area == 0
    
    @pytest.mark.unit
    def test_compute_mask_iou_perfect(self, sample_mask):
        """Test IoU with identical masks."""
        iou = compute_mask_iou(sample_mask, sample_mask)
        assert np.isclose(iou, 1.0)
    
    @pytest.mark.unit
    def test_compute_mask_iou_no_overlap(self, sample_mask, empty_mask):
        """Test IoU with no overlap."""
        iou = compute_mask_iou(sample_mask, empty_mask)
        assert np.isclose(iou, 0.0)
    
    @pytest.mark.unit
    def test_compute_mask_iou_partial(self):
        """Test IoU with partial overlap."""
        mask1 = np.zeros((100, 100), dtype=np.uint8)
        mask1[25:75, 25:75] = 255  # 50x50 square
        
        mask2 = np.zeros((100, 100), dtype=np.uint8)
        mask2[50:100, 50:100] = 255  # 50x50 square, offset
        
        iou = compute_mask_iou(mask1, mask2)
        
        # Intersection: 25x25 = 625
        # Union: 50x50 + 50x50 - 25x25 = 4375
        # IoU = 625/4375 ≈ 0.143
        expected_iou = 625 / 4375
        assert np.isclose(iou, expected_iou, atol=0.01)


class TestMaskIO:
    """Tests for mask save/load operations."""
    
    @pytest.mark.unit
    def test_save_load_mask(self, sample_mask, temp_output_dir):
        """Test saving and loading mask."""
        mask_path = temp_output_dir / "test_mask.png"
        
        # Save mask
        save_mask(sample_mask, str(mask_path))
        assert mask_path.exists()
        
        # Load mask
        loaded_mask = load_mask(str(mask_path))
        
        assert loaded_mask is not None
        assert loaded_mask.shape == sample_mask.shape
        assert np.array_equal(loaded_mask, sample_mask)
    
    @pytest.mark.unit
    def test_load_nonexistent_mask(self):
        """Test loading non-existent mask."""
        loaded_mask = load_mask("nonexistent_mask.png")
        assert loaded_mask is None
    
    @pytest.mark.unit
    def test_save_mask_creates_directory(self, sample_mask, temp_output_dir):
        """Test that save_mask creates directories if needed."""
        mask_path = temp_output_dir / "subdir" / "nested" / "mask.png"
        
        save_mask(sample_mask, str(mask_path))
        assert mask_path.exists()


class TestOverlayOperations:
    """Tests for mask overlay visualization."""
    
    @pytest.mark.unit
    def test_overlay_mask_on_image(self, sample_image, sample_mask):
        """Test overlaying mask on image."""
        overlay = overlay_mask_on_image(sample_image, sample_mask, alpha=0.5)
        
        assert overlay.shape == sample_image.shape
        assert overlay.dtype == sample_image.dtype
        
        # Check that overlay is different from original where mask is active
        mask_pixels = sample_mask > 0
        assert not np.array_equal(overlay[mask_pixels], sample_image[mask_pixels])
    
    @pytest.mark.unit
    def test_overlay_mask_custom_color(self, sample_image, sample_mask):
        """Test overlay with custom color."""
        color = (0, 255, 0)  # Green
        overlay = overlay_mask_on_image(sample_image, sample_mask, color=color, alpha=0.7)
        
        assert overlay.shape == sample_image.shape
        # Check that green channel is enhanced where mask is active
        mask_pixels = sample_mask > 0
        assert np.mean(overlay[mask_pixels, 1]) > np.mean(sample_image[mask_pixels, 1])
    
    @pytest.mark.unit
    def test_overlay_empty_mask(self, sample_image, empty_mask):
        """Test overlay with empty mask."""
        overlay = overlay_mask_on_image(sample_image, empty_mask)
        
        # Should be identical to original image
        assert np.array_equal(overlay, sample_image)


class TestFillHoles:
    """Tests for hole filling operations."""
    
    @pytest.mark.unit
    def test_fill_holes(self):
        """Test filling holes in mask."""
        # Create mask with hole
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 20:80] = 255  # Outer square
        mask[40:60, 40:60] = 0    # Inner hole
        
        filled = fill_holes(mask)
        
        assert filled.shape == mask.shape
        # Hole should be filled
        assert np.all(filled[40:60, 40:60] == 255)
        # Outer boundary should be unchanged
        assert np.all(filled[20:80, 20:80] == 255)
    
    @pytest.mark.unit
    def test_fill_holes_no_holes(self, sample_mask):
        """Test fill_holes with mask that has no holes."""
        filled = fill_holes(sample_mask)
        
        # Should be similar to original (may differ slightly due to algorithm)
        assert filled.shape == sample_mask.shape
        assert np.sum(filled > 0) >= np.sum(sample_mask > 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
