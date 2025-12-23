"""
Unit tests for AnnotationManager.

Tests:
- Mask creation and editing
- Undo/redo functionality
- Mask persistence
- Export operations
"""

import pytest
import numpy as np
from pathlib import Path

from src.core.annotation_manager import AnnotationManager


class TestAnnotationManagerInit:
    """Tests for AnnotationManager initialization."""
    
    @pytest.mark.unit
    def test_init_default(self):
        """Test AnnotationManager initialization."""
        am = AnnotationManager(max_history=50)
        
        assert am.max_history == 50
        assert am.image_path is None
        assert am.current_mask is None
        assert len(am.history) == 0
    
    @pytest.mark.unit
    def test_set_image(self, sample_image):
        """Test setting image for annotation."""
        am = AnnotationManager()
        am.set_image("test_image.png", sample_image.shape)
        
        assert am.image_path == "test_image.png"
        assert am.image_shape == sample_image.shape[:2]
        assert am.current_mask is not None
        assert am.current_mask.shape == sample_image.shape[:2]
        assert np.all(am.current_mask == 0)  # Initially empty


class TestMaskOperations:
    """Tests for mask operations."""
    
    @pytest.mark.unit
    def test_get_current_mask(self, sample_image):
        """Test getting current mask."""
        am = AnnotationManager()
        am.set_image("test.png", sample_image.shape)
        
        mask = am.get_current_mask()
        
        assert mask is not None
        assert mask.shape == sample_image.shape[:2]
        # Should be a copy, not reference
        mask[0, 0] = 255
        assert am.current_mask[0, 0] == 0
    
    @pytest.mark.unit
    def test_set_mask(self, sample_image, sample_mask):
        """Test setting mask."""
        am = AnnotationManager()
        am.set_image("test.png", sample_image.shape)
        
        am.set_mask(sample_mask)
        
        current_mask = am.get_current_mask()
        assert np.array_equal(current_mask, sample_mask)
    
    @pytest.mark.unit
    def test_update_mask_replace(self, sample_image, sample_mask):
        """Test updating mask with replace operation."""
        am = AnnotationManager()
        am.set_image("test.png", sample_image.shape)
        
        am.update_mask(sample_mask, operation='replace')
        
        current_mask = am.get_current_mask()
        assert np.array_equal(current_mask, sample_mask)
    
    @pytest.mark.unit
    def test_update_mask_add(self, sample_image):
        """Test updating mask with add operation."""
        am = AnnotationManager()
        am.set_image("test.png", sample_image.shape)
        
        # Create two non-overlapping masks
        mask1 = np.zeros(sample_image.shape[:2], dtype=np.uint8)
        mask1[50:100, 50:100] = 255
        
        mask2 = np.zeros(sample_image.shape[:2], dtype=np.uint8)
        mask2[150:200, 150:200] = 255
        
        am.update_mask(mask1, operation='replace')
        am.update_mask(mask2, operation='add')
        
        current_mask = am.get_current_mask()
        # Both regions should be filled
        assert np.all(current_mask[50:100, 50:100] == 255)
        assert np.all(current_mask[150:200, 150:200] == 255)
    
    @pytest.mark.unit
    def test_update_mask_subtract(self, sample_image, sample_mask):
        """Test updating mask with subtract operation."""
        am = AnnotationManager()
        am.set_image("test.png", sample_image.shape)
        
        # Set initial mask
        am.update_mask(sample_mask, operation='replace')
        
        # Subtract a portion
        subtract_mask = np.zeros(sample_image.shape[:2], dtype=np.uint8)
        subtract_mask[60:90, 60:90] = 255
        
        am.update_mask(subtract_mask, operation='subtract')
        
        current_mask = am.get_current_mask()
        # Subtracted region should be empty
        assert np.all(current_mask[60:90, 60:90] == 0)
    
    @pytest.mark.unit
    def test_clear_mask(self, sample_image, sample_mask):
        """Test clearing mask."""
        am = AnnotationManager()
        am.set_image("test.png", sample_image.shape)
        am.set_mask(sample_mask)
        
        am.clear_mask()
        
        current_mask = am.get_current_mask()
        assert np.all(current_mask == 0)


class TestPaintingOperations:
    """Tests for painting operations."""
    
    @pytest.mark.unit
    def test_paint_mask(self, sample_image):
        """Test painting mask with brush."""
        am = AnnotationManager()
        am.set_image("test.png", sample_image.shape)
        
        # Paint some points
        points = [(100, 100), (101, 100), (102, 100)]
        am.paint_mask(points, brush_size=5, value=255, operation='paint')
        
        current_mask = am.get_current_mask()
        # Check that points were painted
        assert current_mask[100, 100] == 255
    
    @pytest.mark.unit
    def test_erase_mask(self, sample_image, sample_mask):
        """Test erasing mask."""
        am = AnnotationManager()
        am.set_image("test.png", sample_image.shape)
        am.set_mask(sample_mask)
        
        # Erase some points
        points = [(60, 60), (61, 60), (62, 60)]
        am.paint_mask(points, brush_size=10, operation='erase')
        
        current_mask = am.get_current_mask()
        # Erased region should be cleared
        assert current_mask[60, 60] == 0
    
    @pytest.mark.unit
    def test_paint_polygon(self, sample_image):
        """Test painting filled polygon."""
        am = AnnotationManager()
        am.set_image("test.png", sample_image.shape)
        
        # Draw triangle
        points = [(100, 100), (150, 100), (125, 150)]
        am.paint_polygon(points, value=255)
        
        current_mask = am.get_current_mask()
        # Check that polygon is filled
        assert np.sum(current_mask > 0) > 0
    
    @pytest.mark.unit
    def test_finish_paint_stroke(self, sample_image):
        """Test finishing paint stroke saves state."""
        am = AnnotationManager()
        am.set_image("test.png", sample_image.shape)
        
        history_len_before = len(am.history)
        
        # Paint and finish stroke
        points = [(100, 100)]
        am.paint_mask(points, brush_size=5, value=255)
        am.finish_paint_stroke()
        
        history_len_after = len(am.history)
        assert history_len_after == history_len_before + 1


class TestUndoRedo:
    """Tests for undo/redo functionality."""
    
    @pytest.mark.unit
    def test_undo(self, sample_image, sample_mask):
        """Test undo operation."""
        am = AnnotationManager()
        am.set_image("test.png", sample_image.shape)
        
        # Initial state (empty)
        initial_mask = am.get_current_mask().copy()
        
        # Make a change
        am.set_mask(sample_mask)
        
        # Undo
        success = am.undo()
        
        assert success is True
        current_mask = am.get_current_mask()
        assert np.array_equal(current_mask, initial_mask)
    
    @pytest.mark.unit
    def test_undo_limit(self, sample_image):
        """Test undo at beginning of history."""
        am = AnnotationManager()
        am.set_image("test.png", sample_image.shape)
        
        # Try to undo at start
        success = am.undo()
        
        # Should not be able to undo initial state
        assert success is False
    
    @pytest.mark.unit
    def test_redo(self, sample_image, sample_mask):
        """Test redo operation."""
        am = AnnotationManager()
        am.set_image("test.png", sample_image.shape)
        
        # Make a change
        am.set_mask(sample_mask)
        changed_mask = am.get_current_mask().copy()
        
        # Undo
        am.undo()
        
        # Redo
        success = am.redo()
        
        assert success is True
        current_mask = am.get_current_mask()
        assert np.array_equal(current_mask, changed_mask)
    
    @pytest.mark.unit
    def test_redo_limit(self, sample_image):
        """Test redo at end of history."""
        am = AnnotationManager()
        am.set_image("test.png", sample_image.shape)
        
        # Try to redo without undo
        success = am.redo()
        
        assert success is False
    
    @pytest.mark.unit
    def test_can_undo_redo(self, sample_image, sample_mask):
        """Test undo/redo availability checks."""
        am = AnnotationManager()
        am.set_image("test.png", sample_image.shape)
        
        # Initially can't undo (only one state)
        assert am.can_undo() is False
        assert am.can_redo() is False
        
        # Make a change
        am.set_mask(sample_mask)
        
        # Now can undo
        assert am.can_undo() is True
        assert am.can_redo() is False
        
        # Undo
        am.undo()
        
        # Now can redo
        assert am.can_undo() is False
        assert am.can_redo() is True


class TestMaskPersistence:
    """Tests for mask save/load operations."""
    
    @pytest.mark.unit
    def test_save_mask(self, sample_image, sample_mask, temp_output_dir):
        """Test saving mask."""
        am = AnnotationManager()
        am.set_image("test.png", sample_image.shape)
        am.set_mask(sample_mask)
        
        mask_path = temp_output_dir / "saved_mask.png"
        success = am.save_mask(str(mask_path))
        
        assert success is True
        assert mask_path.exists()
    
    @pytest.mark.unit
    def test_load_mask(self, sample_image, sample_mask, temp_output_dir):
        """Test loading mask."""
        # First save a mask
        from src.utils.mask_utils import save_mask
        mask_path = temp_output_dir / "test_mask.png"
        save_mask(sample_mask, str(mask_path))
        
        # Load mask
        am = AnnotationManager()
        am.set_image("test.png", sample_image.shape)
        success = am.load_mask(str(mask_path))
        
        assert success is True
        current_mask = am.get_current_mask()
        assert np.array_equal(current_mask, sample_mask)
    
    @pytest.mark.unit
    def test_load_nonexistent_mask(self, sample_image):
        """Test loading non-existent mask."""
        am = AnnotationManager()
        am.set_image("test.png", sample_image.shape)
        success = am.load_mask("nonexistent_mask.png")
        
        assert success is False


class TestAnnotationExport:
    """Tests for annotation export operations."""
    
    @pytest.mark.unit
    def test_export_coco_annotation(self, sample_image, sample_mask):
        """Test exporting to COCO format."""
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
        
        # Check bbox format [x, y, width, height]
        assert len(coco_ann['bbox']) == 4
    
    @pytest.mark.unit
    def test_export_yolo_annotation(self, sample_image, sample_mask):
        """Test exporting to YOLO format."""
        am = AnnotationManager()
        am.set_image("test_image.png", sample_image.shape)
        am.set_mask(sample_mask)
        
        yolo_anns = am.export_yolo_annotation(class_id=0)
        
        assert isinstance(yolo_anns, list)
        assert len(yolo_anns) > 0
        
        # Check YOLO format (class_id x1 y1 x2 y2 ...)
        for ann in yolo_anns:
            parts = ann.split()
            assert len(parts) >= 7  # class_id + at least 3 points (6 coords)
            assert parts[0] == '0'  # class_id
    
    @pytest.mark.unit
    def test_export_empty_mask(self, sample_image, empty_mask):
        """Test exporting empty mask."""
        am = AnnotationManager()
        am.set_image("test_image.png", sample_image.shape)
        am.set_mask(empty_mask)
        
        coco_ann = am.export_coco_annotation()
        yolo_anns = am.export_yolo_annotation()
        
        # Empty mask should return empty/minimal annotations
        assert coco_ann == {} or coco_ann.get('area', 0) == 0
        assert yolo_anns == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
