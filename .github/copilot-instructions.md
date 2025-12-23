# AI Agent Instructions - Industrial Defect Segmentation System

## Project Overview
A complete PyQt5-based industrial defect segmentation system combining SAM-powered auto-annotation, model training (U-Net/DeepLabV3+/YOLOv11-Seg), and batch inference with reporting.

---

## Architecture Essentials

### Core Layering Pattern
```
UI Layer (src/ui/)
    ↓ signals/slots
Core Business Logic (src/core/)
    ↓ imports
Utilities & Models (src/models/, src/utils/)
```

**Critical boundary**: UI code must never perform heavy operations—always use QThread subclasses (`src/threads/`) for:
- SAM inference (`SAMInferenceThread`)
- Model training (`TrainingThread`)  
- Batch predictions (`InferenceThread`)

### Key Components & Responsibilities
| Module | Purpose | Key Classes |
|--------|---------|------------|
| `core/data_manager.py` | Image/video loading, dataset organization | `DataManager` |
| `core/sam_handler.py` | SAM model lifecycle, encoding, prompt handling | `SAMHandler` |
| `core/annotation_manager.py` | Mask I/O, history (undo/redo), export (COCO/YOLO) | `AnnotationManager` |
| `core/model_trainer.py` | Model building, training loop, checkpoints | `ModelTrainer` |
| `core/predictor.py` | Inference on trained models, post-processing | `Predictor` |
| `ui/main_window.py` | Main layout, signal routing | `MainWindow` |
| `ui/widgets/image_canvas.py` | Drawing, display, mouse handling | `ImageCanvas` |

---

## Critical Workflows

### Auto-Annotation Flow (SAM Integration)
1. **User provides prompt** → `ImageCanvas` captures clicks/bounding box
2. **GUI signals SAMHandler** → encode current image (blocking, once per image)
3. **Worker thread calls SAMHandler.predict_mask()** → emits progress
4. **Result displayed as semi-transparent overlay** on `ImageCanvas`
5. **User can refine** with brush tools or accept

**Pattern**: Never call SAM methods directly from UI thread. Always emit signals to trigger `SAMInferenceThread`.

### Model Training Flow
1. **User configures** in `TrainConfigDialog` → batch_size, epochs, lr, etc.
2. **TrainingThread spawned** → calls `ModelTrainer.train()` 
3. **Trainer emits signals** for loss/metrics updates
4. **UI updates live** via `MatplotlibCanvas` integration
5. **Checkpoints saved** to `data/outputs/models/`

**Pattern**: Training runs in QThread; communicate via signals (progress, loss, epoch metrics).

### Data Export Pattern
```python
# From AnnotationManager
masks_list, bboxes_list → COCO JSON (via pycocotools)
                        → YOLO txt files
                        → PNG masks in data/processed/masks/
```

---

## Code Conventions & Patterns

### Configuration Management
- **Single source of truth**: `config/config.yaml` (app settings, model defaults, paths)
- **Pattern**: Load once at startup via `load_config()` → pass to modules as arguments
- **Example**: `ModelTrainer(config['training'])` not accessing config globally

### Device Handling (GPU/CPU)
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Store in config, pass to all modules needing computation
model = model.to(device)
```
**Critical**: Always respect config-driven device selection; never hardcode 'cuda'.

### Error Handling in Threads
```python
class WorkerThread(QThread):
    error_signal = pyqtSignal(str)  # Always emit errors
    def run(self):
        try:
            # work
        except Exception as e:
            self.error_signal.emit(str(e))
```

### Naming Conventions
- **Classes**: PascalCase (`DataManager`, `SAMHandler`)
- **Functions**: snake_case (`load_image`, `predict_mask`)
- **Constants**: UPPER_SNAKE_CASE (`MAX_IMAGE_SIZE`, `DEFAULT_BATCH_SIZE`)
- **Private members**: `_leading_underscore`

### Docstring Style (Google format)
```python
def predict_batch(self, images: List[np.ndarray], **kwargs) -> List[np.ndarray]:
    """Batch prediction for multiple images.
    
    Args:
        images: List of HxWx3 numpy arrays
        **kwargs: Additional model-specific parameters
        
    Returns:
        List of segmentation masks (HxWx1) in same order
        
    Raises:
        ValueError: If images list is empty
    """
```

---

## Testing Patterns

### Unit Tests (pytest)
- Place in `tests/test_*.py` mirroring `src/` structure
- Test core logic independent of UI: `DataManager`, `SAMHandler`, `ModelTrainer`
- **Do NOT** test UI/QThread directly (use mocking)
- Example: `tests/test_annotation_manager.py` tests mask save/load, undo/redo logic

### Run Tests
```bash
pytest tests/                        # All
pytest tests/test_data_manager.py    # Specific file
pytest --cov=src                     # Coverage report
```

---

## Development Checklist

When adding new features:

1. **Architecture alignment**: Does it fit core/ui separation?
2. **Threading**: Will it block UI? Use QThread if yes
3. **Config**: New hyperparameters? Add to `config.yaml`
4. **Signals/slots**: Document in file header what signals are emitted
5. **Error handling**: Try/except → emit error signals, never silent fails
6. **Testing**: Unit tests for core logic before UI integration
7. **Logging**: Use `logger.info/warning/error` (configured in `src/logger.py`)

---

## Key File References

- **Entry point**: [src/main.py](../src/main.py)
- **UI definition**: [src/ui/main_window.py](../src/ui/main_window.py)
- **Canvas (drawing)**: [src/ui/widgets/image_canvas.py](../src/ui/widgets/image_canvas.py)
- **Business logic hub**: [src/core/annotation_manager.py](../src/core/annotation_manager.py)
- **Config schema**: [config/config.yaml](../config/config.yaml)
- **Data paths**: [config/paths.yaml](../config/paths.yaml)

---

## Common Debugging Scenarios

### "GUI freezes during SAM inference"
- Symptom: Application unresponsive
- Check: Is SAM call on main thread? Move to `SAMInferenceThread`
- Reference: [src/threads/sam_inference_thread.py](../src/threads/sam_inference_thread.py)

### "Model training losses not updating in UI"
- Symptom: Loss graph empty or stuck
- Check: `TrainingThread` emits `loss_updated` signal?
- Check: UI slot connected? `trainer_thread.loss_signal.connect(self.update_loss_graph)`

### "CUDA out of memory during training"
- Reduce batch_size in `config/hyperparams.yaml`
- Enable gradient checkpointing: `model.enable_checkpointing()` in `ModelTrainer.build_model()`

### "Mask export produces invalid COCO JSON"
- Verify mask encoding in `utils/mask_utils.py` → RLE format
- Test: `pycocotools.mask.encode()` handles your mask shape correctly

---

## Performance Targets

| Operation | Target | Notes |
|-----------|--------|-------|
| Image load | <500ms | Use PIL/OpenCV caching |
| SAM inference (ViT-H) | <1s | RTX 3070; accepts ~1s latency |
| U-Net batch (32 images) | <200ms | Training throughput |
| Mask export (1000 images) | <5s | COCO JSON generation |

---

## Dependencies to Know
- **PyQt5 5.15.x**: UI, threading, drawing canvas
- **PyTorch 2.1+**: SAM, U-Net, model training
- **Albumentations 1.3+**: Data augmentation (use in training pipeline)
- **pycocotools 2.0.6+**: COCO dataset format (critical for export)
- **OpenCV 4.8+**: Image I/O and preprocessing
