# Industrial Defect Segmentation System - User Manual

**Version**: 1.0.0  
**Last Updated**: December 23, 2025  
**Author**: Industrial AI Team

---

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Installation](#installation)
4. [Quick Start Guide](#quick-start-guide)
5. [User Interface Overview](#user-interface-overview)
6. [Core Workflows](#core-workflows)
7. [Feature Reference](#feature-reference)
8. [Troubleshooting](#troubleshooting)
9. [FAQ](#faq)
10. [Support](#support)

---

## Introduction

### What is Industrial Defect Segmentation System?

The Industrial Defect Segmentation System is a comprehensive PyQt5-based application for **defect detection and segmentation** in industrial quality control. It combines state-of-the-art AI models with an intuitive user interface to streamline the entire defect analysis workflow.

### Key Features

- 🖼️ **Image Management** - Import and organize images from folders, files, or videos
- 🖌️ **Smart Annotation** - Manual tools + SAM-powered auto-annotation
- 🤖 **Model Training** - Train U-Net, DeepLabV3+, or FPN models with your data
- 🔍 **Batch Inference** - Predict defects on multiple images efficiently
- 📊 **Visualization** - Generate comprehensive reports in Excel, PDF, or HTML
- 📦 **Export** - Export annotations in COCO, YOLO, or VOC formats

### Who Should Use This?

- Quality control engineers
- Manufacturing process analysts
- Machine vision researchers
- Industrial automation specialists
- Anyone working with defect detection in production lines

---

## Getting Started

### System Requirements

**Minimum Requirements**:
- OS: Windows 10/11, macOS 10.15+, or Linux (Ubuntu 20.04+)
- CPU: Intel Core i5 or equivalent
- RAM: 8 GB
- Storage: 10 GB free space
- Python: 3.8, 3.9, or 3.10

**Recommended Requirements**:
- CPU: Intel Core i7 or AMD Ryzen 7
- RAM: 16 GB or more
- GPU: NVIDIA GPU with 6+ GB VRAM (for faster SAM inference and training)
- Storage: 50 GB SSD

### Prerequisites

- Python 3.8+ installed
- CUDA Toolkit 11.8+ (for GPU support)
- Basic understanding of image processing concepts

---

## Installation

### Method 1: Install from PyPI (Recommended)

```bash
# Install the package
pip install industrial-defect-seg

# Verify installation
industrial-defect-seg --version
```

### Method 2: Install from Source

```bash
# Clone the repository
git clone https://github.com/your-org/industrial-defect-seg.git
cd industrial-defect-seg

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Method 3: Docker (Isolated Environment)

```bash
# Pull the Docker image
docker pull your-org/industrial-defect-seg:latest

# Run the container
docker run -it --rm \
  --gpus all \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $(pwd)/data:/app/data \
  your-org/industrial-defect-seg:latest
```

### Download SAM Weights (Required for Auto-Annotation)

```bash
# Download SAM ViT-H weights (~2.4 GB)
python scripts/download_sam_weights.py

# Or manually download from:
# https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
# Save to: models/checkpoints/sam_vit_h.pth
```

### Verify Installation

```bash
# Run the application
python src/main.py

# Or if installed via pip:
industrial-defect-seg
```

---

## Quick Start Guide

### 5-Minute Tutorial

#### Step 1: Launch the Application

```bash
python src/main.py
```

#### Step 2: Import Images

1. Click **File → Import Images...**
2. Select import source:
   - **Folder**: Import all images from a folder
   - **Files**: Select specific image files
   - **Video**: Extract frames from a video
3. Click **Import**

#### Step 3: Annotate Defects

**Option A: Manual Annotation**
1. Select an image from the file browser (left panel)
2. Click the **Brush** tool in the toolbar
3. Paint over defect areas
4. Use **Eraser** to correct mistakes
5. Save: **Ctrl+S**

**Option B: SAM Auto-Annotation**
1. Select an image
2. Open **Tools → SAM Auto-Annotation**
3. Choose prompt type:
   - **Points**: Click on defect (✅ fastest)
   - **Box**: Draw bounding box around defect
   - **Combined**: Use both points and box
4. Click **Run SAM**
5. Review and refine the generated mask
6. Click **Accept** to save

#### Step 4: Export Annotations

1. Click **Tools → Export Annotations...**
2. Select export format:
   - **COCO**: For Detectron2, MMDetection
   - **YOLO**: For YOLOv8, YOLOv11
   - **VOC**: For traditional frameworks
3. Choose output directory
4. Click **Export**

#### Step 5: Train a Model (Optional)

1. Click **Tools → Train Model...**
2. Configure training:
   - **Architecture**: U-Net (recommended for beginners)
   - **Encoder**: ResNet34
   - **Epochs**: 50
   - **Batch Size**: 4 (adjust based on GPU memory)
3. Click **Start Training**
4. Monitor training progress in real-time

#### Step 6: Run Predictions

1. Click **Tools → Predict...**
2. Select trained model checkpoint
3. Choose input images or folder
4. Configure post-processing (optional)
5. Click **Start Prediction**
6. View results with overlay visualization

#### Step 7: Generate Report

1. Click **Tools → Generate Report...**
2. Select mask directory
3. Choose report formats (Excel, PDF, HTML)
4. Click **Generate Report**
5. Open generated report

**Congratulations!** 🎉 You've completed the full workflow.

---

## User Interface Overview

### Main Window Layout

```
┌─────────────────────────────────────────────────────────────┐
│  File  Edit  View  Tools  Help                 [_][□][X]   │
├──────────┬────────────────────────────────┬─────────────────┤
│          │                                │                 │
│  File    │                                │   Properties    │
│  Browser │       Image Canvas             │   Panel         │
│          │                                │                 │
│  [Tree]  │   [Zoom/Pan/Annotate Area]    │   [Image Info]  │
│          │                                │   [Mask Info]   │
│  Images/ │                                │   [Settings]    │
│  ├─raw/  │                                │                 │
│  ├─masks/│                                │                 │
│  └─...   │                                │                 │
│          │                                │                 │
├──────────┴────────────────────────────────┴─────────────────┤
│  Status: Ready  |  Position: (0, 0)  |  Zoom: 100%        │
└─────────────────────────────────────────────────────────────┘
```

### Menu Bar

#### File Menu
- **Import Images...** (Ctrl+I): Import images/videos
- **Open Project...** (Ctrl+O): Open existing project
- **Save** (Ctrl+S): Save current annotations
- **Exit** (Ctrl+Q): Close application

#### Edit Menu
- **Undo** (Ctrl+Z): Undo last annotation
- **Redo** (Ctrl+Y): Redo undone annotation
- **Clear Mask**: Clear all annotations for current image

#### View Menu
- **Zoom In** (Ctrl++): Increase zoom level
- **Zoom Out** (Ctrl+-): Decrease zoom level
- **Fit to Window**: Reset zoom to fit image
- **Toggle Mask Overlay**: Show/hide mask overlay

#### Tools Menu
- **Train Model...**: Open training configuration
- **Predict...**: Run inference on images
- **Export Annotations...**: Export to COCO/YOLO/VOC
- **Generate Report...**: Create analysis reports
- **SAM Auto-Annotation**: Open SAM control panel

#### Help Menu
- **Documentation**: Open user manual
- **About**: Show version and credits

### Toolbar

| Icon | Tool | Shortcut | Description |
|------|------|----------|-------------|
| 🖱️ | Select | V | Selection mode (pan/zoom) |
| 🖌️ | Brush | B | Paint defect masks |
| 🧹 | Eraser | E | Erase mask areas |
| 📐 | Polygon | P | Draw polygon masks |
| 🤖 | SAM | S | SAM auto-annotation |
| 🔍 | Zoom In | Ctrl++ | Increase zoom |
| 🔍 | Zoom Out | Ctrl+- | Decrease zoom |

### Panels

#### Left Panel: File Browser
- Tree view of project directories
- Thumbnail previews
- Quick navigation between images
- Search and filter functionality

#### Center Panel: Image Canvas
- Main workspace for viewing and annotating
- Zoom and pan controls
- Real-time mask overlay
- Coordinate display

#### Right Panel: Properties
- Image information (size, format, path)
- Mask statistics (defect count, area, coverage)
- Annotation tools settings (brush size, opacity)
- Quick actions (clear, undo, redo)

---

## Core Workflows

### Workflow 1: Annotation Project

**Goal**: Create annotated dataset for defect detection

**Steps**:
1. **Import Images**
   - File → Import Images
   - Select folder with raw images
   - Images appear in file browser

2. **Annotate Each Image**
   - Select image from browser
   - Use manual tools or SAM
   - Save each annotation (Ctrl+S)

3. **Quality Check**
   - Review all annotations
   - Fix any mistakes
   - Ensure consistent labeling

4. **Export Dataset**
   - Tools → Export Annotations
   - Choose COCO format
   - Save to `data/processed/annotations/`

**Tips**:
- Use SAM for quick rough annotations, then refine manually
- Save frequently to avoid data loss
- Use consistent labeling criteria across all images

### Workflow 2: Model Training

**Goal**: Train a custom segmentation model

**Steps**:
1. **Prepare Data**
   - Ensure annotations are exported (COCO format)
   - Verify dataset structure:
     ```
     data/processed/
     ├── images/
     ├── masks/
     └── annotations/
         └── instances.json
     ```

2. **Configure Training**
   - Tools → Train Model
   - Select architecture (U-Net recommended)
   - Set hyperparameters:
     - Epochs: 50-100
     - Batch size: 2-8 (based on GPU memory)
     - Learning rate: 1e-4 (default)
     - Encoder: ResNet34 (good balance)

3. **Monitor Training**
   - Watch loss curves in real-time
   - Check IoU/Dice metrics
   - Training saves checkpoints automatically

4. **Evaluate Results**
   - Review training metrics
   - Check best checkpoint (lowest val_loss)
   - Checkpoints saved to `data/outputs/models/`

**Tips**:
- Start with U-Net + ResNet34 (fastest, good results)
- Use data augmentation (enabled by default)
- Monitor GPU memory usage
- Save best model checkpoint for inference

### Workflow 3: Batch Prediction

**Goal**: Predict defects on new images

**Steps**:
1. **Prepare Images**
   - Place new images in a folder
   - Or import via File → Import Images

2. **Load Model**
   - Tools → Predict
   - Select trained checkpoint (.pth file)
   - Model loads automatically

3. **Configure Prediction**
   - Select input images/folder
   - Enable post-processing (recommended):
     - Remove small components
     - Fill holes
     - Smooth boundaries
   - Set confidence threshold (0.5 default)

4. **Run Inference**
   - Click Start Prediction
   - Progress bar shows completion
   - Results saved to `data/outputs/predictions/`

5. **Review Results**
   - View overlay visualizations
   - Check prediction quality
   - Export results if needed

**Tips**:
- Use batch processing for multiple images
- Enable TTA (Test-Time Augmentation) for better accuracy
- Adjust post-processing based on defect types
- Save predictions for report generation

### Workflow 4: Report Generation

**Goal**: Create analysis reports for stakeholders

**Steps**:
1. **Prepare Data**
   - Ensure masks are available (annotations or predictions)
   - Place masks in `data/processed/masks/`

2. **Configure Report**
   - Tools → Generate Report
   - Select mask directory
   - Choose report formats:
     - Excel: Detailed statistics
     - PDF: Professional presentation
     - HTML: Interactive web report

3. **Customize Content**
   - Select charts to include:
     - Defect size distribution
     - Defect count per image
     - Coverage ratio distribution
     - Spatial heatmap
   - Add custom notes/comments

4. **Generate Report**
   - Click Generate Report
   - Wait for processing (few seconds)
   - Reports saved to `data/outputs/reports/`

5. **Share Results**
   - Open generated reports
   - Review statistics and charts
   - Share with team/management

**Tips**:
- Generate all formats for different audiences
- Include spatial heatmap for location analysis
- Add context notes for clarity
- Schedule regular report generation

---

## Feature Reference

### 1. Import Images

**Location**: File → Import Images

**Supported Formats**: JPG, PNG, BMP, TIFF

**Import Options**:

#### Folder Import
- Select directory containing images
- Recursively finds all supported images
- Preserves folder structure

#### File Import
- Select specific image files
- Ctrl+Click for multiple selection
- Shift+Click for range selection

#### Video Import
- Supported formats: MP4, AVI, MOV
- Configuration options:
  - **Frame Interval**: Extract every Nth frame (default: 10)
  - **Max Frames**: Limit total frames extracted (default: 1000)
  - **Start Time**: Begin extraction at specific timestamp
  - **End Time**: Stop extraction at specific timestamp

**Example: Video Import**
```
Video: defect_inspection.mp4 (1000 frames, 30 fps)
Settings:
- Frame interval: 10
- Max frames: 100
Result: Extracts 100 frames (1 every 10 frames)
Saved to: data/raw/video_frames/
```

### 2. Manual Annotation Tools

#### Brush Tool

**Shortcut**: B  
**Description**: Paint defect areas

**Settings**:
- **Brush Size**: 5-100 pixels (adjust with slider or `[` `]` keys)
- **Opacity**: 0-100% (controls mask transparency)
- **Color**: RGB selection (for different defect classes)

**Usage**:
1. Select Brush tool
2. Click and drag to paint
3. Hold Shift for straight lines
4. Release to see result

**Tips**:
- Use larger brush for rough areas
- Use smaller brush for details
- Paint multiple strokes before saving

#### Eraser Tool

**Shortcut**: E  
**Description**: Remove mask areas

**Settings**:
- **Eraser Size**: 5-100 pixels
- **Opacity**: 0-100%

**Usage**:
1. Select Eraser tool
2. Click and drag to erase
3. Use for corrections and refinements

**Tips**:
- Same controls as Brush tool
- Use after SAM auto-annotation for refinement

#### Polygon Tool

**Shortcut**: P  
**Description**: Draw polygonal masks

**Usage**:
1. Select Polygon tool
2. Click to place vertices
3. Double-click or press Enter to close polygon
4. Polygon fills automatically

**Tips**:
- Best for regular defect shapes
- Faster than brush for geometric defects
- Press Esc to cancel current polygon

### 3. SAM Auto-Annotation

**Location**: Tools → SAM Auto-Annotation

**Description**: Use Segment Anything Model (SAM) for automatic mask generation

**Requirements**: SAM weights must be downloaded (see Installation)

#### Prompt Types

**1. Point Prompts** (Fastest, ~0.6-1.1s)
- Click on defect area
- Green marker = positive point (include)
- Red marker = negative point (exclude)
- Add multiple points for refinement

**Example**:
```
1. Click center of defect → SAM generates mask
2. If mask too large, click background (right-click for negative point)
3. If mask too small, add more positive points
```

**2. Box Prompts** (Medium, ~0.8-1.5s)
- Draw bounding box around defect
- SAM segments within box
- More precise than points for isolated defects

**3. Combined Prompts** (Most Accurate, ~1.0-2.0s)
- Use both points and box
- Box defines rough area
- Points refine segmentation

#### SAM Workflow

```
1. Select image
2. Open SAM panel
3. Choose prompt type
4. Provide prompt (click/draw)
5. Click "Run SAM"
6. Wait for inference (~1 second)
7. Review mask
8. Refine if needed (add more prompts)
9. Click "Accept" to save
```

#### SAM Settings

- **Multi-mask Output**: Generate 3 masks, select best
- **Post-processing**: Smooth, fill holes, remove small areas
- **Confidence Threshold**: Filter low-confidence regions (0-1)

**Tips**:
- Start with single point in defect center
- Use box for irregular defects
- Enable post-processing for cleaner masks
- SAM works best with clear contrast

### 4. Model Training

**Location**: Tools → Train Model

**Description**: Train custom segmentation models

#### Architecture Selection

**U-Net** (Recommended for beginners)
- Fast training and inference
- Good for most defect types
- Encoder options: ResNet18/34/50, EfficientNet

**DeepLabV3+**
- Better for complex defects
- Higher accuracy, slower inference
- Encoder options: ResNet50/101

**FPN (Feature Pyramid Network)**
- Multi-scale defect detection
- Good for varying defect sizes
- Encoder options: ResNet18/34/50

#### Training Configuration

**Basic Settings**:
- **Epochs**: 50-100 (more for complex datasets)
- **Batch Size**: 2-8 (reduce if GPU memory error)
- **Learning Rate**: 1e-4 (default, adjust if diverging)
- **Device**: Auto-detect GPU (fallback to CPU)

**Advanced Settings**:
- **Optimizer**: Adam (default), SGD, AdamW
- **Loss Function**: 
  - Dice Loss (default, handles class imbalance)
  - BCE Loss (binary cross-entropy)
  - Focal Loss (for hard examples)
  - Combined Loss
- **Augmentation**: Enable data augmentation (recommended)
  - Random flip, rotate, scale
  - Color jitter, brightness, contrast
  - Gaussian noise, blur

**Data Split**:
- Training: 70% (model learns from these)
- Validation: 15% (monitors overfitting)
- Test: 15% (final evaluation)

#### Training Monitoring

Real-time metrics displayed:
- **Loss**: Should decrease over time
- **IoU (Intersection over Union)**: 0-1, higher is better
- **Dice Score**: 0-1, higher is better
- **Epoch Progress**: Current/Total

**Example Training Output**:
```
Epoch 10/50
Loss: 0.234 | Val Loss: 0.256
IoU: 0.852 | Val IoU: 0.839
Dice: 0.918 | Val Dice: 0.911
Time: 45s
```

**Stopping Criteria**:
- Training completes all epochs
- Manual stop via "Stop Training" button
- Early stopping if validation loss increases (optional)

#### Checkpoints

- **Best Model**: Saved when validation loss improves
- **Last Model**: Saved at end of training
- **Location**: `data/outputs/models/`
- **Naming**: `{architecture}_{encoder}_{timestamp}.pth`

**Example**: `unet_resnet34_20251223_143022.pth`

### 5. Batch Prediction

**Location**: Tools → Predict

**Description**: Run trained model on new images

#### Configuration

**Input Options**:
- **Single Image**: Select one image file
- **Image Folder**: Process all images in folder
- **Image List**: Select multiple specific files

**Model Selection**:
- Browse to trained checkpoint (.pth file)
- Recent models shown in dropdown
- Model architecture detected automatically

#### Post-Processing Options

**Basic Operations**:
- **Threshold**: Confidence threshold (0-1, default 0.5)
- **Remove Small**: Minimum defect size in pixels
- **Fill Holes**: Fill internal holes in masks
- **Smooth Boundaries**: Morphological smoothing

**Advanced Operations**:
- **Dilation**: Expand mask boundaries (kernel size)
- **Erosion**: Shrink mask boundaries
- **Opening**: Remove noise (erosion + dilation)
- **Closing**: Fill gaps (dilation + erosion)

#### Test-Time Augmentation (TTA)

**Description**: Predict on multiple augmented versions, merge results

**Augmentations**:
- Horizontal flip
- Vertical flip
- 90° rotation
- 180° rotation
- 270° rotation

**Effect**: Improves accuracy by ~2-5%, increases time by 5x

**Usage**: Enable for final production predictions

#### Output Options

**Formats**:
- **Masks**: PNG binary masks (0/255)
- **Overlay**: Original image + colored mask overlay
- **JSON**: Prediction metadata (confidence, bbox, area)

**Directory Structure**:
```
data/outputs/predictions/
├── masks/           # Binary masks
├── overlays/        # Visualizations
└── metadata.json    # Statistics
```

### 6. Export Annotations

**Location**: Tools → Export Annotations

**Description**: Export annotations in standard formats

#### Format Options

**COCO Format**
- **File**: `instances.json`
- **Structure**:
  ```json
  {
    "images": [...],
    "annotations": [...],
    "categories": [...]
  }
  ```
- **Use Case**: Detectron2, MMDetection, PyTorch frameworks
- **Contains**: Bounding boxes, segmentation masks (RLE), areas

**YOLO Format**
- **Files**: One .txt per image
- **Structure**: `class_id x1 y1 x2 y2 ... (polygon points, normalized)`
- **Use Case**: YOLOv8, YOLOv11 segmentation models
- **Contains**: Polygon coordinates (normalized 0-1)

**VOC/Pascal Format**
- **Files**: XML files + PNG masks
- **Structure**: 
  ```xml
  <annotation>
    <object>
      <name>defect</name>
      <bndbox>...</bndbox>
    </object>
  </annotation>
  ```
- **Use Case**: Traditional CV frameworks
- **Contains**: Bounding boxes, class labels

#### Export Settings

**Options**:
- **Include Images**: Copy images to export directory
- **Validate**: Run validator after export
- **Split Dataset**: Create train/val/test splits
- **Class Mapping**: Map defect types to class IDs

**Example Export**:
```
Export Configuration:
- Format: COCO
- Images: 500
- Masks: 500
- Output: data/exports/coco_20251223/

Result:
✅ Exported 500 annotations
✅ Validation passed
✅ Dataset ready for training
```

### 7. Report Generation

**Location**: Tools → Generate Report

**Description**: Generate comprehensive analysis reports

#### Report Formats

**Excel Report** (.xlsx)
- **Sheets**:
  - Summary: Overall statistics
  - Per-Image: Individual image analysis
  - Charts: Embedded visualizations
- **Data**: Defect count, area, coverage, bounding boxes
- **Use Case**: Detailed analysis, data export

**PDF Report** (.pdf)
- **Sections**:
  - Executive Summary
  - Statistical Analysis
  - Visualizations
  - Detailed Tables
- **Professional**: Print-ready format
- **Use Case**: Management presentations, documentation

**HTML Report** (.html)
- **Interactive**: Clickable charts
- **Responsive**: Mobile-friendly
- **Embeds**: Images and statistics
- **Use Case**: Web sharing, dashboards

#### Content Selection

**Statistics**:
- Total defect count
- Average defect size
- Defect size distribution
- Coverage ratio per image
- Spatial distribution

**Charts**:
- Defect size histogram
- Defect count per image (bar chart)
- Coverage ratio distribution
- Spatial heatmap (2D grid)
- Before/after comparison

**Customization**:
- Add custom notes/comments
- Select specific images to include
- Adjust chart styles and colors
- Set report title and metadata

#### Example Report Contents

```
=== Defect Analysis Report ===
Date: 2025-12-23
Images Analyzed: 100

Summary Statistics:
- Total Defects: 245
- Average Defects per Image: 2.45
- Total Defect Area: 458,920 pixels
- Average Coverage: 3.2%
- Largest Defect: 8,542 pixels
- Smallest Defect: 125 pixels

Top 10 Images by Defect Count:
1. image_042.jpg - 8 defects
2. image_087.jpg - 7 defects
...

[Charts follow]
```

---

## Troubleshooting

### Common Issues

#### Issue 1: Application Won't Start

**Symptoms**: Application crashes on launch, or window doesn't appear

**Solutions**:
1. **Check Python version**:
   ```bash
   python --version  # Should be 3.8, 3.9, or 3.10
   ```

2. **Verify dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Check logs**:
   ```bash
   cat logs/app.log  # Look for error messages
   ```

4. **Try debug mode**:
   ```bash
   python src/main.py --debug
   ```

#### Issue 2: SAM Inference Very Slow

**Symptoms**: SAM takes >5 seconds per image

**Solutions**:
1. **Check GPU availability**:
   ```python
   import torch
   print(torch.cuda.is_available())  # Should be True
   ```

2. **Reduce image size**:
   - Resize large images before annotation
   - SAM works well on 512-1024px images

3. **Use lighter SAM model**:
   - Switch from ViT-H to ViT-B (faster, slightly less accurate)
   - Edit `config/config.yaml`:
     ```yaml
     sam:
       model_type: "vit_b"  # Instead of "vit_h"
     ```

4. **Close other GPU applications**:
   - Free up GPU memory
   - Check with `nvidia-smi`

#### Issue 3: Out of Memory During Training

**Symptoms**: "CUDA out of memory" error

**Solutions**:
1. **Reduce batch size**:
   - Decrease from 8 → 4 → 2 → 1
   - Smaller batches use less GPU memory

2. **Use smaller model**:
   - Switch encoder: ResNet50 → ResNet34 → ResNet18
   - Reduces model parameters

3. **Reduce image size**:
   - Edit `config/hyperparams.yaml`:
     ```yaml
     input_size: [256, 256]  # Instead of [512, 512]
     ```

4. **Enable gradient checkpointing**:
   - Trade computation for memory
   - Edit training config

5. **Use CPU**:
   - Slower but works with limited GPU
   - Edit `config/config.yaml`:
     ```yaml
     device:
       type: "cpu"
     ```

#### Issue 4: Masks Not Saving

**Symptoms**: Annotations disappear after closing application

**Solutions**:
1. **Ensure proper save**:
   - Always press Ctrl+S or File → Save
   - Check status bar for "Saved" confirmation

2. **Check permissions**:
   ```bash
   # On Linux/macOS:
   chmod -R u+w data/processed/masks/
   ```

3. **Verify output directory**:
   - Check `config/paths.yaml`:
     ```yaml
     paths:
       masks: "data/processed/masks"  # Must exist
     ```
   - Create if missing:
     ```bash
     mkdir -p data/processed/masks
     ```

#### Issue 5: Export Validation Fails

**Symptoms**: Export completes but shows validation errors

**Solutions**:
1. **Check error messages**:
   - Read validation report in export dialog
   - Common issues: Missing images, invalid masks

2. **Verify image-mask pairs**:
   - Each image needs corresponding mask
   - File names must match (except extension)

3. **Check mask format**:
   - Masks should be binary (0 and 255 only)
   - No grayscale values

4. **Re-export with validation disabled**:
   - Skip validation temporarily
   - Fix issues manually

#### Issue 6: Model Not Predicting Anything

**Symptoms**: Predictions are all black (no defects detected)

**Solutions**:
1. **Check confidence threshold**:
   - Lower threshold (0.5 → 0.3 → 0.1)
   - Some defects may have lower confidence

2. **Verify model checkpoint**:
   - Ensure using correct .pth file
   - Check training was successful

3. **Check input preprocessing**:
   - Images should match training format
   - Same normalization and size

4. **Review training metrics**:
   - If training IoU was low (<0.5), model may not have learned
   - Retrain with more data or different hyperparameters

### Error Messages

#### "SAM weights not found"

**Solution**: Download SAM weights
```bash
python scripts/download_sam_weights.py
```

#### "Invalid annotation format"

**Solution**: Re-export annotations with proper format validation enabled

#### "Insufficient GPU memory"

**Solution**: Reduce batch size or use smaller model (see Issue 3)

#### "Dataset split failed"

**Solution**: Ensure at least 10 images for splitting

### Performance Tips

1. **For Faster Annotation**:
   - Use SAM for initial masks, refine manually
   - Enable auto-save (every 5 minutes)
   - Use keyboard shortcuts (B, E, P, S)

2. **For Better Training**:
   - Use at least 100 annotated images
   - Enable data augmentation
   - Monitor validation metrics
   - Save checkpoints frequently

3. **For Accurate Predictions**:
   - Use TTA for final predictions
   - Enable post-processing
   - Adjust threshold based on defect type
   - Review predictions manually

---

## FAQ

### General

**Q: What types of defects can this system detect?**

A: The system is designed for visual surface defects in manufacturing:
- Scratches, dents, cracks
- Contamination, stains, corrosion
- Missing components, misalignments
- Color defects, texture anomalies
- Any visible abnormality that can be segmented

**Q: How many images do I need for training?**

A: Recommended:
- Minimum: 50-100 annotated images
- Good: 500-1000 images
- Excellent: 2000+ images
- More images = better model performance

**Q: Can I use this for real-time defect detection?**

A: Partially:
- **Batch processing**: Yes, excellent for offline analysis
- **Real-time**: Possible with GPU, ~10-30 FPS depending on model
- For production lines, consider deploying trained model separately

### Technical

**Q: Which model architecture should I choose?**

A:
- **U-Net + ResNet34**: Best starting point (fast + accurate)
- **DeepLabV3+ + ResNet50**: When accuracy is critical
- **FPN**: When defects vary greatly in size

**Q: How do I improve model accuracy?**

A:
1. Add more training data
2. Improve annotation quality
3. Use data augmentation
4. Try different architectures
5. Tune hyperparameters
6. Use ensemble predictions

**Q: Can I train on CPU?**

A: Yes, but:
- Training will be 10-50x slower
- Recommended only for small datasets (<100 images)
- Use GPU for production training

**Q: What's the difference between COCO, YOLO, and VOC formats?**

A:
- **COCO**: Single JSON file, polygon/RLE masks, most versatile
- **YOLO**: Text files, polygon points, optimized for YOLO models
- **VOC**: XML + PNG masks, traditional format, good compatibility

### Usage

**Q: Can I annotate multiple defect types?**

A: Currently:
- Single class (defect vs background) is supported
- Multi-class support planned for future version
- Workaround: Use separate projects for different defect types

**Q: How do I transfer annotations between projects?**

A:
1. Export annotations (COCO format)
2. Copy images and annotations to new project
3. Import in new project
4. Verify with validation

**Q: Can I pause and resume training?**

A:
- **Pause**: Click "Stop Training" button
- **Resume**: Not currently supported in GUI
- **Workaround**: Use command-line training with `--resume` flag

**Q: How do I share trained models?**

A:
1. Locate model checkpoint: `data/outputs/models/*.pth`
2. Copy .pth file
3. Share with team
4. Recipients load via Tools → Predict

---

## Support

### Getting Help

**Documentation**:
- User Manual (this document): [USER_MANUAL.md](USER_MANUAL.md)
- Developer Guide: [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md)
- API Documentation: [API_REFERENCE.md](API_REFERENCE.md)
- Testing Guide: [testing-guide.md](testing-guide.md)

**Community**:
- GitHub Issues: [github.com/your-org/industrial-defect-seg/issues](https://github.com/your-org/industrial-defect-seg/issues)
- Discussion Forum: [github.com/your-org/industrial-defect-seg/discussions](https://github.com/your-org/industrial-defect-seg/discussions)
- Stack Overflow: Tag `industrial-defect-seg`

**Commercial Support**:
- Email: support@your-org.com
- Response time: 24-48 hours
- Include: Version, OS, error logs, screenshots

### Reporting Bugs

**Before Reporting**:
1. Check existing issues on GitHub
2. Update to latest version
3. Review troubleshooting section

**Bug Report Template**:
```markdown
**Environment**:
- OS: Windows 10 / macOS 12 / Ubuntu 20.04
- Python version: 3.9.7
- Package version: 1.0.0
- GPU: NVIDIA RTX 3070 / None

**Description**:
Clear description of the bug

**Steps to Reproduce**:
1. Step 1
2. Step 2
3. ...

**Expected Behavior**:
What should happen

**Actual Behavior**:
What actually happens

**Logs**:
Attach logs/app.log

**Screenshots**:
If applicable
```

### Feature Requests

Submit via GitHub Issues with label `enhancement`:
```markdown
**Feature Name**: Multi-class defect annotation

**Problem**: Currently only supports single defect class

**Proposed Solution**: Add class dropdown in annotation toolbar

**Use Case**: Distinguish between different defect types

**Priority**: High / Medium / Low
```

---

## Appendix

### Keyboard Shortcuts Reference

| Shortcut | Action |
|----------|--------|
| Ctrl+I | Import images |
| Ctrl+O | Open project |
| Ctrl+S | Save annotations |
| Ctrl+Q | Quit application |
| Ctrl+Z | Undo |
| Ctrl+Y | Redo |
| Ctrl++ | Zoom in |
| Ctrl+- | Zoom out |
| Ctrl+0 | Fit to window |
| B | Brush tool |
| E | Eraser tool |
| P | Polygon tool |
| S | SAM tool |
| V | Select tool |
| [ | Decrease brush size |
| ] | Increase brush size |
| Space | Pan mode (hold) |
| Delete | Clear current mask |
| F11 | Fullscreen |

### File Formats Supported

**Input Images**:
- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tif, .tiff)
- WebP (.webp)

**Input Videos**:
- MP4 (.mp4)
- AVI (.avi)
- MOV (.mov)
- MKV (.mkv)

**Export Formats**:
- COCO JSON (.json)
- YOLO Text (.txt)
- VOC XML (.xml)
- PNG Masks (.png)

**Model Formats**:
- PyTorch (.pth, .pt)
- ONNX (.onnx) - export only

**Report Formats**:
- Excel (.xlsx)
- PDF (.pdf)
- HTML (.html)

### Configuration Files

**config/config.yaml**: Application settings
```yaml
app:
  name: "Industrial Defect Segmentation System"
  version: "1.0.0"

ui:
  window_width: 1600
  window_height: 1000
  mask_opacity: 0.5

sam:
  model_type: "vit_h"
  checkpoint: "sam_vit_h.pth"

logging:
  level: "INFO"
  file: "logs/app.log"
```

**config/paths.yaml**: Directory paths
```yaml
paths:
  data_root: "data"
  raw_data: "data/raw"
  masks: "data/processed/masks"
  models: "data/outputs/models"
```

**config/hyperparams.yaml**: Training hyperparameters
```yaml
training:
  epochs: 50
  batch_size: 4
  learning_rate: 0.0001
  optimizer: "adam"
  
augmentation:
  enabled: true
  horizontal_flip: true
  vertical_flip: true
  rotation: 15
```

### Glossary

- **Annotation**: Manual or automatic labeling of defect regions
- **Batch Size**: Number of images processed simultaneously during training
- **COCO**: Common Objects in Context, a popular annotation format
- **Defect**: Abnormality or imperfection in an inspected item
- **Dice Score**: Similarity metric for segmentation (0-1, higher is better)
- **Encoder**: Feature extraction network (e.g., ResNet, EfficientNet)
- **Epoch**: One complete pass through the training dataset
- **IoU**: Intersection over Union, segmentation accuracy metric
- **Mask**: Binary image indicating defect regions (white) and background (black)
- **Post-processing**: Refinement operations after prediction
- **RLE**: Run-Length Encoding, compact mask storage format
- **SAM**: Segment Anything Model, foundation model for segmentation
- **Segmentation**: Pixel-level classification of image regions
- **TTA**: Test-Time Augmentation, prediction averaging technique
- **YOLO**: You Only Look Once, efficient object detection framework

---

**Document Version**: 1.0.0  
**Last Updated**: December 23, 2025  
**Copyright**: © 2025 Industrial AI Team  
**License**: MIT License

For latest updates, visit: [github.com/your-org/industrial-defect-seg](https://github.com/your-org/industrial-defect-seg)
