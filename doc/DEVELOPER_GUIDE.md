# Industrial Defect Segmentation System - Developer Guide

**Version**: 1.0.0  
**Last Updated**: December 23, 2025  
**Target Audience**: Developers, Contributors, Maintainers

---

## Table of Contents

1. [Introduction](#introduction)
2. [Development Setup](#development-setup)
3. [Architecture Overview](#architecture-overview)
4. [Module Reference](#module-reference)
5. [Development Workflow](#development-workflow)
6. [Testing](#testing)
7. [Contributing Guidelines](#contributing-guidelines)
8. [Code Style](#code-style)
9. [Advanced Topics](#advanced-topics)
10. [Troubleshooting](#troubleshooting)

---

## Introduction

### Purpose

This guide provides comprehensive documentation for developers working on the Industrial Defect Segmentation System. Whether you're adding features, fixing bugs, or understanding the codebase, this guide will help you navigate the project effectively.

### Prerequisites

- **Programming**: Strong Python 3.8+ knowledge
- **Frameworks**: PyQt5, PyTorch, OpenCV experience
- **Tools**: Git, pytest, Docker (optional)
- **Concepts**: Familiarity with:
  - Image segmentation and computer vision
  - Deep learning (CNNs, training pipelines)
  - GUI event-driven programming
  - Software architecture patterns (MVC, Observer)

### Quick Links

- **User Manual**: [USER_MANUAL.md](USER_MANUAL.md)
- **API Reference**: [API_REFERENCE.md](API_REFERENCE.md)
- **Testing Guide**: [testing-guide.md](testing-guide.md)
- **Architecture Design**: [architecture-design.md](architecture-design.md)
- **GitHub Repository**: [github.com/your-org/industrial-defect-seg](https://github.com/your-org/industrial-defect-seg)

---

## Development Setup

### 1. Clone Repository

```bash
git clone https://github.com/your-org/industrial-defect-seg.git
cd industrial-defect-seg
```

### 2. Create Virtual Environment

```bash
# Using venv
python -m venv venv

# Activate
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install package in editable mode
pip install -e .

# Install pre-commit hooks
pre-commit install
```

### 4. Configure IDE

#### VS Code (Recommended)

**Install Extensions**:
- Python (Microsoft)
- Pylance
- Python Test Explorer
- GitLens

**.vscode/settings.json**:
```json
{
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "python.linting.flake8Enabled": true,
  "python.formatting.provider": "black",
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["tests"],
  "editor.formatOnSave": true,
  "files.exclude": {
    "**/__pycache__": true,
    "**/*.pyc": true,
    "**/.pytest_cache": true
  }
}
```

#### PyCharm

1. Open project directory
2. Configure Python interpreter (venv)
3. Mark `src` as Sources Root
4. Mark `tests` as Test Sources Root
5. Enable pytest as test runner

### 5. Verify Setup

```bash
# Run tests
pytest tests/ -v

# Check code style
flake8 src/
black --check src/

# Run application
python src/main.py
```

---

## Architecture Overview

### System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Presentation Layer                  │
│  ┌───────────┐  ┌──────────┐  ┌──────────────────┐   │
│  │ MainWindow│  │ Dialogs  │  │ Widget Components│   │
│  └─────┬─────┘  └────┬─────┘  └────────┬─────────┘   │
│        │             │                  │              │
│        └─────────────┴──────────────────┘              │
│                      │                                 │
│                Signals/Slots                           │
│                      │                                 │
├──────────────────────┼─────────────────────────────────┤
│               Business Logic Layer                      │
│  ┌───────────┬──────┴──────┬─────────────┬──────────┐│
│  │DataManager│AnnotationMgr│ SAMHandler  │ Trainer  ││
│  └─────┬─────┴──────┬──────┴──────┬──────┴────┬─────┘│
│        │            │             │            │      │
│        └────────────┴─────────────┴────────────┘      │
│                      │                                 │
├──────────────────────┼─────────────────────────────────┤
│                 Service Layer                          │
│  ┌──────────┬───────┴────────┬──────────┬──────────┐ │
│  │ Models   │ Utils          │ Threads  │ Export   │ │
│  └──────────┴────────────────┴──────────┴──────────┘ │
└─────────────────────────────────────────────────────────┘
```

### Directory Structure

```
industrial-defect-seg/
├── src/                      # Source code
│   ├── ui/                   # User interface
│   │   ├── main_window.py    # Main application window
│   │   ├── dialogs/          # Dialog windows
│   │   └── widgets/          # Reusable UI components
│   ├── core/                 # Business logic
│   │   ├── data_manager.py   # Data loading and management
│   │   ├── annotation_manager.py  # Annotation handling
│   │   ├── sam_handler.py    # SAM integration
│   │   ├── model_trainer.py  # Model training logic
│   │   └── predictor.py      # Inference engine
│   ├── models/               # Model architectures
│   │   ├── segmentation_models.py  # U-Net, DeepLabV3+, FPN
│   │   ├── losses.py         # Loss functions
│   │   └── metrics.py        # Evaluation metrics
│   ├── utils/                # Utility functions
│   │   ├── file_utils.py     # File I/O
│   │   ├── image_utils.py    # Image processing
│   │   ├── mask_utils.py     # Mask operations
│   │   ├── export_utils.py   # Dataset export
│   │   ├── statistics.py     # Statistical analysis
│   │   ├── visualization.py  # Chart generation
│   │   └── report_generator.py  # Report creation
│   ├── threads/              # Asynchronous operations
│   │   ├── sam_inference_thread.py  # SAM inference
│   │   ├── training_thread.py      # Model training
│   │   └── inference_thread.py     # Batch prediction
│   ├── main.py               # Application entry point
│   └── logger.py             # Logging configuration
├── tests/                    # Test suite
│   ├── conftest.py           # Test fixtures
│   ├── test_mask_utils.py    # Utils tests
│   ├── test_data_manager.py  # Core tests
│   └── test_integration.py   # Integration tests
├── config/                   # Configuration files
│   ├── config.yaml           # App configuration
│   ├── paths.yaml            # Path configuration
│   └── hyperparams.yaml      # Training hyperparameters
├── data/                     # Data directories
│   ├── raw/                  # Raw images
│   ├── processed/            # Processed data
│   │   ├── images/
│   │   ├── masks/
│   │   └── annotations/
│   └── outputs/              # Generated outputs
│       ├── models/           # Trained models
│       ├── predictions/      # Predictions
│       └── reports/          # Reports
├── models/                   # Model weights
│   └── checkpoints/          # Pretrained weights (SAM)
├── doc/                      # Documentation
│   ├── USER_MANUAL.md        # User manual
│   ├── DEVELOPER_GUIDE.md    # This file
│   ├── API_REFERENCE.md      # API documentation
│   └── testing-guide.md      # Testing guide
├── requirements.txt          # Production dependencies
├── requirements-dev.txt      # Development dependencies
├── setup.py                  # Package setup
├── pytest.ini                # pytest configuration
├── .gitignore                # Git ignore rules
└── README.md                 # Project overview
```

### Design Patterns

#### 1. Model-View-Controller (MVC)

**Model** (Business Logic):
- `DataManager`: Data persistence and loading
- `AnnotationManager`: Annotation state management
- `ModelTrainer`: Training orchestration

**View** (UI):
- `MainWindow`: Main application view
- Dialogs: `ImportDialog`, `TrainConfigDialog`, etc.
- Widgets: `ImageCanvas`, `FileBrowser`, etc.

**Controller** (Signal Handlers):
- Signal/slot connections in `MainWindow`
- Event handlers in widgets

#### 2. Observer Pattern (Qt Signals/Slots)

**Example**:
```python
# Emitter (Subject)
class SAMInferenceThread(QThread):
    progress_updated = pyqtSignal(int, str)  # Signal
    
    def run(self):
        self.progress_updated.emit(50, "Processing...")  # Emit

# Observer (Subscriber)
class MainWindow(QMainWindow):
    def __init__(self):
        self.sam_thread = SAMInferenceThread()
        self.sam_thread.progress_updated.connect(self._on_progress)  # Connect
    
    def _on_progress(self, progress, message):
        self.progress_bar.setValue(progress)  # React
```

#### 3. Factory Pattern (Model Creation)

```python
def create_segmentation_model(architecture, encoder_name, **kwargs):
    """Factory function for creating segmentation models."""
    if architecture == 'unet':
        return smp.Unet(encoder_name=encoder_name, **kwargs)
    elif architecture == 'deeplabv3plus':
        return smp.DeepLabV3Plus(encoder_name=encoder_name, **kwargs)
    elif architecture == 'fpn':
        return smp.FPN(encoder_name=encoder_name, **kwargs)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
```

#### 4. Strategy Pattern (Loss Functions)

```python
class LossStrategy:
    """Base class for loss strategies."""
    def compute(self, pred, target):
        raise NotImplementedError

class DiceLoss(LossStrategy):
    def compute(self, pred, target):
        return 1 - dice_coefficient(pred, target)

class FocalLoss(LossStrategy):
    def compute(self, pred, target):
        return focal_loss(pred, target, alpha=0.25, gamma=2.0)
```

---

## Module Reference

### Core Modules

#### src/core/data_manager.py

**Purpose**: Centralized data loading and management

**Key Classes**:
```python
class DataManager:
    def __init__(self, data_root: str, cache_size_mb: int = 1024):
        """Initialize data manager with caching."""
        
    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """Load image with LRU caching."""
        
    def load_video(self, video_path: str, frame_interval: int = 10) -> List[np.ndarray]:
        """Extract frames from video."""
        
    def create_splits(self, image_paths: List[str], 
                     train_ratio: float = 0.7) -> Dict[str, List[str]]:
        """Create train/val/test splits."""
```

**Design Decisions**:
- LRU cache for image loading (configurable size)
- Lazy loading to minimize memory usage
- Automatic format detection (RGB/grayscale)

**Usage Example**:
```python
dm = DataManager(data_root="data", cache_size_mb=2048)

# Load single image
image = dm.load_image("data/raw/image_001.jpg")

# Load batch
images = dm.load_batch_images(image_paths)

# Create splits
splits = dm.create_splits(image_paths, train_ratio=0.7)
```

#### src/core/annotation_manager.py

**Purpose**: Manage annotation state and operations

**Key Classes**:
```python
class AnnotationManager:
    def __init__(self, max_history: int = 50):
        """Initialize with undo/redo history."""
        
    def set_mask(self, mask: np.ndarray):
        """Replace current mask."""
        
    def update_mask(self, mask: np.ndarray, operation: str = 'replace'):
        """Update mask with operation (replace/add/subtract)."""
        
    def undo(self) -> bool:
        """Undo last operation."""
        
    def redo(self) -> bool:
        """Redo undone operation."""
        
    def export_coco_annotation(self) -> Dict:
        """Export annotation to COCO format."""
```

**Design Decisions**:
- History-based undo/redo (circular buffer)
- Mask operations (replace/add/subtract/intersect)
- Multiple export formats (COCO/YOLO/VOC)

**Usage Example**:
```python
am = AnnotationManager(max_history=50)
am.set_image("image_001.jpg", (1024, 1024))

# Paint mask
am.paint_mask(points=[(100, 100), (101, 100)], brush_size=10)

# Undo
am.undo()

# Export
coco_annotation = am.export_coco_annotation()
```

#### src/core/sam_handler.py

**Purpose**: Interface to Segment Anything Model

**Key Classes**:
```python
class SAMHandler:
    def __init__(self, model_type: str = 'vit_h', device: str = 'cuda'):
        """Initialize SAM model."""
        
    def encode_image(self, image: np.ndarray) -> bool:
        """Encode image (required before prediction)."""
        
    def predict_mask_from_points(self, points: List[Tuple[int, int]], 
                                 labels: List[int]) -> Dict:
        """Predict mask from point prompts."""
        
    def predict_mask_from_box(self, box: Tuple[int, int, int, int]) -> Dict:
        """Predict mask from bounding box."""
        
    def get_best_mask(self, prediction: Dict) -> np.ndarray:
        """Select best mask from multi-mask output."""
```

**Design Decisions**:
- Lazy model loading (load on first use)
- Image encoding cached per image
- Support for multiple prompt types

**Usage Example**:
```python
sam_handler = SAMHandler(model_type='vit_h', device='cuda')

# Encode image (once per image)
sam_handler.encode_image(image)

# Predict from point
prediction = sam_handler.predict_mask_from_points(
    points=[(512, 512)],
    labels=[1]  # 1=foreground, 0=background
)

# Get best mask
mask = sam_handler.get_best_mask(prediction)
```

#### src/core/model_trainer.py

**Purpose**: Orchestrate model training

**Key Classes**:
```python
class ModelTrainer:
    def __init__(self, model, config: Dict):
        """Initialize trainer with model and config."""
        
    def train(self, train_loader, val_loader):
        """Main training loop."""
        
    def train_epoch(self, data_loader) -> Dict:
        """Train for one epoch."""
        
    def validate(self, data_loader) -> Dict:
        """Validate model."""
        
    def save_checkpoint(self, path: str, is_best: bool = False):
        """Save model checkpoint."""
```

**Design Decisions**:
- Configurable loss functions and optimizers
- Automatic best model tracking
- Checkpoint saving (best + last)
- Real-time metric logging via signals

**Usage Example**:
```python
model = SegmentationModel(architecture='unet', encoder_name='resnet34')
trainer = ModelTrainer(model, config={
    'epochs': 50,
    'lr': 1e-4,
    'device': 'cuda'
})

# Train
trainer.train(train_loader, val_loader)

# Model saved to config['checkpoint_dir']
```

### UI Modules

#### src/ui/main_window.py

**Purpose**: Main application window and coordinator

**Key Components**:
- Menu bar (File, Edit, View, Tools, Help)
- Toolbar (Quick actions)
- Central splitter (File browser | Canvas | Properties)
- Status bar (Info and coordinates)

**Signal Coordination**:
```python
class MainWindow(QMainWindow):
    def __init__(self, config, paths_config, hyperparams):
        # Initialize UI
        self._init_ui()
        
        # Connect signals
        self.file_browser.file_selected.connect(self._on_file_selected)
        self.image_canvas.mouse_clicked.connect(self._on_canvas_clicked)
        
    def _on_file_selected(self, file_path: str):
        """Handle file selection from browser."""
        image = self.data_manager.load_image(file_path)
        self.image_canvas.load_image(image, file_path)
```

**Best Practices**:
- Keep `MainWindow` as thin coordinator
- Delegate logic to core modules
- Use signals for cross-component communication

#### src/ui/widgets/image_canvas.py

**Purpose**: Image display with zoom, pan, and interaction

**Key Classes**:
```python
class ImageCanvas(QGraphicsView):
    # Signals
    image_loaded = pyqtSignal()
    mouse_moved = pyqtSignal(int, int)
    mouse_clicked = pyqtSignal(int, int, int)
    zoom_changed = pyqtSignal(float)
    
    def load_image(self, image: np.ndarray, image_path: str = None):
        """Load and display image."""
        
    def zoom(self, factor: float):
        """Apply zoom factor."""
        
    def scene_to_image_coords(self, scene_pos: QPointF) -> Tuple[int, int]:
        """Convert scene coordinates to image coordinates."""
```

**Design Decisions**:
- Based on `QGraphicsView` for efficient rendering
- Coordinate transformation between scene/image spaces
- Emit signals for external interaction handling

### Model Modules

#### src/models/segmentation_models.py

**Purpose**: Segmentation model architectures

**Key Functions**:
```python
class SegmentationModel(nn.Module):
    def __init__(self, architecture: str, encoder_name: str, 
                 in_channels: int = 3, num_classes: int = 1):
        """Initialize segmentation model."""
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        
    @staticmethod
    def get_available_architectures() -> List[str]:
        """Get list of available architectures."""
        
    @staticmethod
    def get_available_encoders(architecture: str) -> List[str]:
        """Get list of available encoders for architecture."""
```

**Design Decisions**:
- Wrapper around `segmentation_models_pytorch` library
- Consistent interface across architectures
- Static methods for discoverability

#### src/models/losses.py

**Purpose**: Loss functions for training

**Implemented Losses**:
```python
class DiceLoss(nn.Module):
    """Dice loss for segmentation."""
    
class FocalLoss(nn.Module):
    """Focal loss for hard examples."""
    
class CombinedLoss(nn.Module):
    """Combination of multiple losses."""
    
def get_loss_function(name: str, **kwargs) -> nn.Module:
    """Factory function for loss functions."""
```

### Utility Modules

#### src/utils/mask_utils.py

**Purpose**: Mask processing operations

**Key Functions**:
```python
def binary_mask_to_rle(mask: np.ndarray) -> Dict:
    """Convert binary mask to RLE format."""
    
def mask_to_polygon(mask: np.ndarray) -> List[List[int]]:
    """Convert mask to polygon coordinates."""
    
def mask_to_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Extract bounding box from mask."""
    
def compute_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute IoU between two masks."""
```

#### src/utils/statistics.py

**Purpose**: Statistical analysis

**Key Classes**:
```python
class DefectStatistics:
    def compute_mask_statistics(self, mask: np.ndarray) -> Dict:
        """Compute statistics for single mask."""
        
    def compute_batch_statistics(self, mask_paths: List[str]) -> Dict:
        """Compute statistics for batch of masks."""
        
class ModelPerformanceAnalyzer:
    def analyze_training_history(self, history: Dict) -> Dict:
        """Analyze training metrics."""
        
    def compute_confusion_matrix(self, predictions: List, ground_truths: List) -> Dict:
        """Compute confusion matrix metrics."""
```

---

## Development Workflow

### Git Workflow

#### Branch Strategy

```
main (production-ready)
  ├── develop (integration branch)
  │   ├── feature/new-annotation-tool
  │   ├── feature/yolov11-integration
  │   ├── bugfix/mask-export-issue
  │   └── hotfix/sam-memory-leak
  └── release/v1.1.0
```

**Branch Types**:
- `main`: Production-ready code
- `develop`: Integration branch for features
- `feature/*`: New features
- `bugfix/*`: Bug fixes
- `hotfix/*`: Urgent production fixes
- `release/*`: Release preparation

#### Workflow Steps

1. **Create Feature Branch**:
   ```bash
   git checkout develop
   git pull origin develop
   git checkout -b feature/new-feature-name
   ```

2. **Develop and Commit**:
   ```bash
   # Make changes
   git add .
   git commit -m "feat: Add new annotation tool"
   ```

3. **Push and Create PR**:
   ```bash
   git push origin feature/new-feature-name
   # Create Pull Request on GitHub
   ```

4. **Code Review and Merge**:
   ```bash
   # After approval
   git checkout develop
   git merge feature/new-feature-name
   git push origin develop
   ```

### Commit Message Convention

Follow [Conventional Commits](https://www.conventionalcommits.org/):

**Format**: `<type>(<scope>): <description>`

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting)
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

**Examples**:
```bash
feat(sam): Add multi-prompt SAM integration
fix(export): Fix COCO RLE encoding issue
docs(api): Update API reference for predictor
test(integration): Add end-to-end annotation workflow test
refactor(ui): Simplify canvas coordinate transformation
```

### Development Cycle

#### 1. Setup Development Environment

```bash
# Create feature branch
git checkout -b feature/my-feature

# Install dev dependencies
pip install -r requirements-dev.txt
```

#### 2. Implement Feature

```python
# src/new_module.py
"""
New module description.

This module provides...
"""

from typing import Optional
import numpy as np

def new_function(param1: int, param2: str) -> Optional[np.ndarray]:
    """
    Function description.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When param1 is negative
    """
    if param1 < 0:
        raise ValueError("param1 must be non-negative")
    
    # Implementation
    result = np.zeros((param1, param1))
    return result
```

#### 3. Write Tests

```python
# tests/test_new_module.py
"""Tests for new_module."""

import pytest
import numpy as np
from src.new_module import new_function

class TestNewFunction:
    """Tests for new_function."""
    
    def test_basic_functionality(self):
        """Test basic behavior."""
        result = new_function(5, "test")
        assert result.shape == (5, 5)
    
    def test_negative_param_raises_error(self):
        """Test error handling."""
        with pytest.raises(ValueError, match="must be non-negative"):
            new_function(-1, "test")
```

#### 4. Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_new_module.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

#### 5. Check Code Quality

```bash
# Format code
black src/ tests/

# Check style
flake8 src/ tests/

# Type checking
mypy src/

# Sort imports
isort src/ tests/
```

#### 6. Update Documentation

```markdown
# In relevant .md file
## New Feature

Description of new feature...

### Usage

\`\`\`python
from src.new_module import new_function
result = new_function(10, "example")
\`\`\`
```

#### 7. Commit and Push

```bash
git add .
git commit -m "feat: Add new feature with tests and docs"
git push origin feature/my-feature
```

#### 8. Create Pull Request

**PR Template**:
```markdown
## Description
Brief description of changes

## Changes Made
- [ ] Added new_function to new_module.py
- [ ] Added unit tests (test_new_module.py)
- [ ] Updated documentation

## Testing
- [x] Unit tests pass
- [x] Integration tests pass
- [x] Manual testing performed

## Checklist
- [x] Code follows project style guide
- [x] Tests added/updated
- [x] Documentation updated
- [x] No breaking changes (or documented)
```

---

## Testing

### Test Organization

```
tests/
├── conftest.py              # Shared fixtures
├── test_mask_utils.py       # Utils tests (29 tests)
├── test_statistics.py       # Statistics tests (12 tests)
├── test_data_manager.py     # Data manager tests (13 tests)
├── test_annotation_manager.py  # Annotation tests (25 tests)
├── test_models.py           # Model tests (11 tests)
├── test_integration.py      # Integration tests (5 workflows)
└── test_performance.py      # Performance benchmarks (12 tests)
```

### Running Tests

```bash
# All tests
pytest

# Specific test file
pytest tests/test_mask_utils.py

# Specific test class
pytest tests/test_mask_utils.py::TestRLEOperations

# Specific test function
pytest tests/test_mask_utils.py::TestRLEOperations::test_rle_encode_decode

# With coverage
pytest --cov=src --cov-report=html

# Fast tests only (skip slow)
pytest -m "not slow"

# Integration tests only
pytest -m integration

# Performance tests
pytest -m performance
```

### Writing Tests

#### Unit Test Example

```python
import pytest
import numpy as np
from src.utils.mask_utils import compute_mask_iou

class TestMaskIoU:
    """Tests for IoU computation."""
    
    def test_iou_identical_masks(self):
        """Test IoU for identical masks."""
        mask = np.ones((100, 100), dtype=np.uint8)
        iou = compute_mask_iou(mask, mask)
        assert iou == 1.0
    
    def test_iou_no_overlap(self):
        """Test IoU for non-overlapping masks."""
        mask1 = np.zeros((100, 100), dtype=np.uint8)
        mask1[0:50, :] = 1
        
        mask2 = np.zeros((100, 100), dtype=np.uint8)
        mask2[50:100, :] = 1
        
        iou = compute_mask_iou(mask1, mask2)
        assert iou == 0.0
    
    @pytest.mark.parametrize("overlap_ratio,expected_iou", [
        (0.25, 0.143),
        (0.50, 0.333),
        (0.75, 0.600),
    ])
    def test_iou_various_overlaps(self, overlap_ratio, expected_iou):
        """Test IoU with various overlap ratios."""
        # Create masks with specific overlap
        mask1 = np.zeros((100, 100), dtype=np.uint8)
        mask1[0:50, :] = 1
        
        overlap_size = int(50 * overlap_ratio)
        mask2 = np.zeros((100, 100), dtype=np.uint8)
        mask2[50-overlap_size:100-overlap_size, :] = 1
        
        iou = compute_mask_iou(mask1, mask2)
        assert abs(iou - expected_iou) < 0.01
```

#### Integration Test Example

```python
@pytest.mark.integration
def test_complete_annotation_workflow(create_test_images, temp_output_dir):
    """Test complete annotation workflow."""
    # 1. Data Management
    image_paths = create_test_images(count=5)
    dm = DataManager(str(temp_output_dir))
    
    # 2. Annotation
    am = AnnotationManager()
    for image_path in image_paths:
        image = dm.load_image(image_path)
        am.set_image(image_path, image.shape[:2])
        
        # Create mock mask
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask[100:200, 100:200] = 255
        am.set_mask(mask)
        
        # Save
        mask_dir = temp_output_dir / "masks"
        mask_dir.mkdir(exist_ok=True)
        am.save_mask(str(mask_dir / f"{Path(image_path).stem}.png"))
    
    # 3. Export
    from src.utils.export_utils import export_to_coco
    
    mask_paths = list((temp_output_dir / "masks").glob("*.png"))
    output_path = temp_output_dir / "annotations.json"
    
    result = export_to_coco(
        image_paths=image_paths,
        mask_paths=[str(p) for p in mask_paths],
        output_path=str(output_path),
        category_name="defect"
    )
    
    # Verify
    assert result is True
    assert output_path.exists()
    
    # Load and verify structure
    import json
    with open(output_path) as f:
        data = json.load(f)
    
    assert 'images' in data
    assert 'annotations' in data
    assert len(data['images']) == 5
```

### Test Fixtures

**Common Fixtures** (in `conftest.py`):

```python
@pytest.fixture
def sample_image():
    """Generate sample RGB image."""
    return np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)

@pytest.fixture
def sample_mask():
    """Generate sample binary mask."""
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[50:100, 50:100] = 255
    return mask

@pytest.fixture
def create_test_images(temp_dir):
    """Factory fixture for creating test images."""
    def _create(count=5, size=(256, 256)):
        image_paths = []
        for i in range(count):
            image = np.random.randint(0, 256, size + (3,), dtype=np.uint8)
            path = temp_dir / f"image_{i:03d}.png"
            cv2.imwrite(str(path), image)
            image_paths.append(str(path))
        return image_paths
    return _create
```

### Coverage Goals

**Target Coverage**: 80% overall, 90% for critical modules

**Check Coverage**:
```bash
pytest --cov=src --cov-report=term-missing

# Example output:
# Name                            Stmts   Miss  Cover   Missing
# -------------------------------------------------------------
# src/utils/mask_utils.py           245     18    93%   156-162
# src/core/data_manager.py           187     24    87%   89-95
# src/models/segmentation_models.py  156     28    82%   45-52
```

**Improve Coverage**:
1. Identify uncovered lines in report
2. Write tests for those specific cases
3. Focus on error paths and edge cases

---

## Contributing Guidelines

### Contribution Process

1. **Find or Create Issue**:
   - Check existing issues on GitHub
   - Create new issue if not exists
   - Discuss approach before implementation

2. **Fork and Branch**:
   ```bash
   # Fork repository on GitHub
   git clone https://github.com/your-username/industrial-defect-seg.git
   git checkout -b feature/your-feature
   ```

3. **Implement Changes**:
   - Follow code style guide
   - Add tests for new functionality
   - Update documentation

4. **Submit Pull Request**:
   - Provide clear description
   - Reference related issues
   - Ensure all checks pass

### Code Review Criteria

**Required**:
- ✅ All tests pass
- ✅ Code coverage maintained (>80%)
- ✅ No lint errors
- ✅ Documentation updated
- ✅ Commit messages follow convention

**Evaluation**:
- Code clarity and readability
- Appropriate error handling
- Performance considerations
- Backward compatibility

### Documentation Standards

**Code Documentation**:
- All modules have module-level docstrings
- All public functions have docstrings (Google style)
- Complex logic has inline comments

**Example**:
```python
def compute_statistics(masks: List[np.ndarray], **kwargs) -> Dict[str, Any]:
    """
    Compute statistical metrics for mask batch.
    
    This function analyzes a collection of masks and computes
    comprehensive statistics including defect counts, areas,
    and spatial distributions.
    
    Args:
        masks: List of binary masks (H x W arrays)
        **kwargs: Additional options
            min_area: Minimum defect area to consider (default: 100)
            max_area: Maximum defect area to consider (default: None)
            compute_spatial: Whether to compute spatial distribution (default: True)
    
    Returns:
        Dictionary containing:
            - total_defects (int): Total number of defects
            - average_area (float): Average defect area in pixels
            - defect_sizes (List[int]): Individual defect sizes
            - spatial_distribution (np.ndarray, optional): 2D heatmap
    
    Raises:
        ValueError: If masks list is empty
        TypeError: If masks are not numpy arrays
    
    Example:
        >>> masks = [load_mask(p) for p in mask_paths]
        >>> stats = compute_statistics(masks, min_area=50)
        >>> print(f"Found {stats['total_defects']} defects")
    
    Note:
        This function assumes binary masks (0 and 255 values).
        Masks with other values will be thresholded at 128.
    """
    if not masks:
        raise ValueError("Masks list cannot be empty")
    
    # Implementation...
```

### Communication

**Channels**:
- **GitHub Issues**: Bug reports, feature requests
- **GitHub Discussions**: General questions, ideas
- **Pull Requests**: Code changes and reviews
- **Email**: support@your-org.com (for private matters)

**Response Times**:
- Bug reports: 24-48 hours
- Feature requests: 1 week
- Pull requests: 2-3 days for initial review

---

## Code Style

### Python Style Guide

**Follow**: [PEP 8](https://pep8.org/) with project-specific preferences

**Tools**:
- **Formatter**: `black` (line length 100)
- **Linter**: `flake8` + `pylint`
- **Import Sorter**: `isort`
- **Type Checker**: `mypy`

### Formatting with Black

```bash
# Format files
black src/ tests/

# Check formatting
black --check src/ tests/

# Configuration (pyproject.toml)
[tool.black]
line-length = 100
target-version = ['py38', 'py39', 'py310']
```

### Linting with Flake8

```bash
# Check style
flake8 src/ tests/

# Configuration (.flake8)
[flake8]
max-line-length = 100
ignore = E203, W503
exclude = .git,__pycache__,venv
```

### Type Hints

```python
from typing import Optional, List, Dict, Tuple, Union

def process_images(
    image_paths: List[str],
    output_dir: str,
    resize_to: Optional[Tuple[int, int]] = None,
    **kwargs
) -> Dict[str, Union[int, float]]:
    """Process images with optional resizing."""
    results: Dict[str, Union[int, float]] = {
        'processed_count': 0,
        'average_time': 0.0
    }
    # Implementation...
    return results
```

### Naming Conventions

**Variables and Functions**: `snake_case`
```python
image_path = "data/image.jpg"
def load_image_from_path(path: str) -> np.ndarray:
    pass
```

**Classes**: `PascalCase`
```python
class DataManager:
    pass

class SAMInferenceThread(QThread):
    pass
```

**Constants**: `UPPER_SNAKE_CASE`
```python
MAX_IMAGE_SIZE = 2048
DEFAULT_BATCH_SIZE = 4
```

**Private Members**: `_leading_underscore`
```python
class MyClass:
    def __init__(self):
        self._private_attribute = None
    
    def _private_method(self):
        pass
```

### File Organization

```python
"""
Module docstring describing purpose.

Longer description if needed...
"""

# Standard library imports
import os
import sys
from pathlib import Path

# Third-party imports
import numpy as np
import torch
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# Local imports
from src.logger import get_logger
from src.utils.file_utils import ensure_dir

# Module-level constants
LOGGER = get_logger(__name__)
MAX_CACHE_SIZE = 1024

# Module-level functions
def helper_function():
    pass

# Classes
class MainClass:
    pass
```

---

## Advanced Topics

### Custom Model Integration

To add a new segmentation architecture:

**1. Extend Model Module** (`src/models/segmentation_models.py`):

```python
def create_segmentation_model(architecture: str, ...):
    # Existing architectures...
    
    elif architecture == 'pspnet':
        # Add PSPNet
        import segmentation_models_pytorch as smp
        model = smp.PSPNet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes
        )
        return SegmentationModelWrapper(model)
    
    # Register in available architectures
@staticmethod
def get_available_architectures():
    return ['unet', 'deeplabv3plus', 'fpn', 'pspnet']
```

**2. Update UI** (`src/ui/dialogs/train_config_dialog.py`):

```python
# Add to architecture dropdown
self.architecture_combo.addItem("PSPNet", "pspnet")
```

**3. Add Tests** (`tests/test_models.py`):

```python
def test_create_pspnet():
    """Test PSPNet creation."""
    model = SegmentationModel(
        architecture='pspnet',
        encoder_name='resnet50'
    )
    assert model is not None
```

**4. Update Documentation** (USER_MANUAL.md, API_REFERENCE.md)

### Custom Loss Function

**1. Implement Loss** (`src/models/losses.py`):

```python
class TverskyLoss(nn.Module):
    """
    Tversky loss for imbalanced segmentation.
    
    Args:
        alpha: Weight for false positives
        beta: Weight for false negatives
    """
    
    def __init__(self, alpha: float = 0.7, beta: float = 0.3):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        
        # Flatten
        pred = pred.view(-1)
        target = target.view(-1)
        
        # True positives, false positives, false negatives
        tp = (pred * target).sum()
        fp = (pred * (1 - target)).sum()
        fn = ((1 - pred) * target).sum()
        
        tversky = tp / (tp + self.alpha * fp + self.beta * fn + 1e-7)
        
        return 1 - tversky

# Register in factory
def get_loss_function(name: str, **kwargs):
    losses = {
        'dice': DiceLoss,
        'focal': FocalLoss,
        'tversky': TverskyLoss,  # Add here
    }
    return losses[name](**kwargs)
```

**2. Add to Training Config**:

```python
# In ModelTrainer
loss_fn = get_loss_function('tversky', alpha=0.7, beta=0.3)
```

### Plugin System (Future)

**Architecture**:
```python
# src/plugins/base.py
class PluginBase:
    """Base class for plugins."""
    
    def __init__(self, main_window):
        self.main_window = main_window
    
    def register(self):
        """Register plugin (add menu items, shortcuts)."""
        raise NotImplementedError
    
    def unregister(self):
        """Unregister plugin."""
        raise NotImplementedError

# src/plugins/my_plugin.py
class MyPlugin(PluginBase):
    def register(self):
        # Add menu item
        action = QAction("My Plugin", self.main_window)
        action.triggered.connect(self.run)
        self.main_window.plugins_menu.addAction(action)
    
    def run(self):
        # Plugin logic
        pass
```

### Performance Optimization

**Profiling**:
```python
# Use cProfile
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Code to profile
result = expensive_function()

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumtime')
stats.print_stats(20)  # Top 20 functions
```

**Memory Profiling**:
```python
from memory_profiler import profile

@profile
def memory_intensive_function():
    large_array = np.zeros((10000, 10000))
    # ...
```

**GPU Profiling**:
```python
import torch

# Start profiler
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
) as prof:
    # Code to profile
    model(input_tensor)

# Print results
print(prof.key_averages().table(sort_by="cuda_time_total"))
```

---

## Troubleshooting

### Development Issues

#### Issue: Import Errors

```bash
# Symptom: ModuleNotFoundError: No module named 'src'

# Solution: Install package in editable mode
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

#### Issue: Qt Platform Plugin Error

```bash
# Symptom: Could not find the Qt platform plugin "windows"

# Solution: Reinstall PyQt5
pip uninstall PyQt5 PyQt5-Qt5 PyQt5-sip
pip install PyQt5
```

#### Issue: CUDA Out of Memory

```python
# Solution 1: Reduce batch size
config['batch_size'] = 2

# Solution 2: Clear cache
torch.cuda.empty_cache()

# Solution 3: Use gradient checkpointing
model.enable_checkpointing()
```

### Debugging Tips

**Enable Debug Logging**:
```python
# In main.py or config
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Qt Debug Mode**:
```bash
# Set environment variable
export QT_DEBUG_PLUGINS=1
python src/main.py
```

**PyTorch Anomaly Detection**:
```python
torch.autograd.set_detect_anomaly(True)
```

**Python Debugger**:
```python
import pdb; pdb.set_trace()  # Breakpoint
```

---

## Resources

### Documentation
- PyQt5: https://doc.qt.io/qt-5/
- PyTorch: https://pytorch.org/docs/
- Segmentation Models PyTorch: https://smp.readthedocs.io/
- OpenCV: https://docs.opencv.org/

### Tools
- Black: https://black.readthedocs.io/
- Flake8: https://flake8.pycqa.org/
- pytest: https://docs.pytest.org/
- mypy: https://mypy.readthedocs.io/

### Community
- GitHub Issues: [Report bugs](https://github.com/your-org/industrial-defect-seg/issues)
- Discussions: [Ask questions](https://github.com/your-org/industrial-defect-seg/discussions)
- Stack Overflow: Tag `industrial-defect-seg`

---

**Document Version**: 1.0.0  
**Last Updated**: December 23, 2025  
**Maintainer**: Industrial AI Team

Happy coding! 🚀
