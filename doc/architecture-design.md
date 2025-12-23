# 工业缺陷分割系统 - 架构设计文档

## 1. 项目概述

**项目名称**：工业缺陷分割系统  
**开发语言**：Python 3.9+  
**主要框架**：PyQt5/PyQt6 + PyTorch  
**目标**：构建完整的 标注→训练→推理→可视化 全流程系统

---

## 2. 技术栈选择

### 2.1 前端与 GUI
- **PyQt5** (5.15.x) / PyQt6 (6.x)
  - 理由：跨平台、功能完整、图形绘制性能好
  - 核心模块：QGraphicsView、QGraphicsScene、QThread

### 2.2 深度学习框架
- **PyTorch** (2.0+)
  - 理由：SAM 官方支持、部署灵活、自动求导机制完善
- **torchvision** (0.15+)
  - 理由：预训练模型库、数据增强工具

### 2.3 分割模型库
- **Segment Anything (SAM)** - facebook-ai/segment-anything
  - 用途：自动标注
  - 版本：latest
- **Segmentation Models** (segmentation-models-pytorch)
  - 包含：U-Net、DeepLabV3+、FPN 等
- **YOLOv11** (ultralytics)
  - 用途：目标检测+分割 (YOLOv11-Seg)

### 2.4 数据处理与增强
- **OpenCV** (4.8+)
  - 用途：图片/视频读取、预处理
- **NumPy** (1.24+)
  - 用途：数组操作、mask 处理
- **Albumentations** (1.3+)
  - 用途：高效的数据增强管道
- **Pillow** (10.0+)
  - 用途：图片 I/O 和处理

### 2.5 数据格式与导出
- **COCO JSON** - pycocotools (2.0.6+)
- **YOLO TXT** - 自定义实现
- **openpyxl** (3.10+)
  - 用途：Excel 报表生成
- **reportlab** (4.0+)
  - 用途：PDF 报告生成

### 2.6 可视化与绘图
- **Matplotlib** (3.7+)
  - 用途：统计图表、集成到 PyQt
- **PyQtGraph** (0.13+)
  - 用途：高性能实时绘图
- **plotly** (5.17+) [可选]
  - 用途：交互式报告

### 2.7 日志与监控
- **logging** (Python 标准库)
- **tensorboard** (2.14+) [可选]
  - 用途：训练过程可视化

### 2.8 配置管理
- **PyYAML** (6.0+)
  - 用途：配置文件管理
- **python-dotenv** (1.0+)
  - 用途：环境变量管理

---

## 3. 项目目录结构

```
industrial-defect-segmentation/
├── README.md
├── requirements.txt
├── setup.py
├── config/
│   ├── config.yaml                 # 全局配置
│   ├── paths.yaml                  # 路径配置
│   └── hyperparams.yaml            # 超参数配置
├── src/
│   ├── __init__.py
│   ├── main.py                     # 应用入口
│   ├── ui/                         # PyQt 前端
│   │   ├── __init__.py
│   │   ├── main_window.py          # 主窗口
│   │   ├── dialogs/                # 对话框
│   │   │   ├── import_dialog.py
│   │   │   ├── train_config_dialog.py
│   │   │   └── export_dialog.py
│   │   ├── widgets/                # 自定义 Widget
│   │   │   ├── image_canvas.py     # 图片编辑画布
│   │   │   ├── annotation_toolbar.py
│   │   │   ├── file_browser.py
│   │   │   └── log_viewer.py
│   │   └── styles/
│   │       └── stylesheet.qss      # UI 样式表
│   ├── core/                       # 核心业务逻辑
│   │   ├── __init__.py
│   │   ├── data_manager.py         # 数据管理
│   │   ├── annotation_manager.py   # 标注管理
│   │   ├── sam_handler.py          # SAM 处理器
│   │   ├── model_trainer.py        # 模型训练
│   │   ├── predictor.py            # 预测推理
│   │   └── visualization.py        # 可视化
│   ├── models/                     # 模型定义
│   │   ├── __init__.py
│   │   ├── unet.py
│   │   ├── deeplabv3.py
│   │   └── yolov11_seg.py          # YOLOv11 分割模型
│   ├── utils/                      # 工具函数
│   │   ├── __init__.py
│   │   ├── mask_utils.py           # mask 处理
│   │   ├── bbox_utils.py           # bbox 处理
│   │   ├── file_utils.py           # 文件操作
│   │   ├── image_utils.py          # 图片处理
│   │   ├── metrics.py              # 评估指标
│   │   ├── augmentation.py         # 数据增强
│   │   └── export_utils.py         # 导出格式处理
│   ├── threads/                    # 多线程处理
│   │   ├── __init__.py
│   │   ├── training_thread.py
│   │   ├── inference_thread.py
│   │   └── sam_inference_thread.py
│   └── logger.py                   # 日志配置
├── data/
│   ├── raw/                        # 原始数据
│   ├── processed/                  # 处理后数据
│   │   ├── images/
│   │   ├── masks/
│   │   └── annotations/
│   ├── splits/                     # 数据分割
│   │   ├── train.txt
│   │   ├── val.txt
│   │   └── test.txt
│   └── outputs/                    # 输出结果
│       ├── predictions/
│       ├── reports/
│       └── models/
├── models/
│   └── checkpoints/                # 模型权重存储
├── tests/
│   ├── __init__.py
│   ├── test_data_manager.py
│   ├── test_annotation.py
│   ├── test_models.py
│   └── test_utils.py
├── scripts/
│   ├── prepare_dataset.py          # 数据集准备脚本
│   ├── train.py                    # 训练脚本
│   ├── evaluate.py                 # 评估脚本
│   └── export_onnx.py              # 模型导出脚本
└── docs/
    ├── architecture.md             # 架构设计
    ├── user_guide.md               # 用户指南
    ├── api_reference.md            # API 参考
    └── troubleshooting.md          # 故障排除
```

---

## 4. 核心模块设计

### 4.1 数据管理模块 (`src/core/data_manager.py`)

**职责**：
- 图片/视频导入和管理
- 文件夹遍历和批量加载
- 数据集组织和分割

**关键类**：
```python
class DataManager:
    - load_image(path: str) -> np.ndarray
    - load_video(path: str) -> Generator[np.ndarray]
    - get_file_list(folder: str) -> List[str]
    - save_dataset_structure(root: str) -> None
    - get_image_by_index(idx: int) -> Tuple[np.ndarray, str]
```

### 4.2 SAM 自动标注模块 (`src/core/sam_handler.py`)

**职责**：
- 加载 SAM 模型
- 处理用户提示（点、框、文本）
- 异步推理管理

**关键类**：
```python
class SAMHandler:
    - load_model(model_type: str) -> None
    - encode_image(image: np.ndarray) -> None
    - predict_mask(prompts: Dict) -> np.ndarray
    - set_image_embeddings(image: np.ndarray) -> None
```

### 4.3 标注管理模块 (`src/core/annotation_manager.py`)

**职责**：
- 保存/加载标注数据
- 管理撤销/重做历史
- 导出多种格式

**关键类**：
```python
class AnnotationManager:
    - save_mask(mask: np.ndarray, path: str) -> None
    - load_mask(path: str) -> np.ndarray
    - export_coco_json(annotations: List, output_path: str) -> None
    - export_yolo_txt(masks: List, output_dir: str) -> None
    - undo() / redo()
```

### 4.4 模型训练模块 (`src/core/model_trainer.py`)

**职责**：
- 模型初始化和训练循环
- 参数优化和学习率调度
- 模型权重保存

**关键类**：
```python
class ModelTrainer:
    - build_model(model_name: str, **kwargs) -> nn.Module
    - train_epoch(train_loader: DataLoader) -> Dict[str, float]
    - validate(val_loader: DataLoader) -> Dict[str, float]
    - save_checkpoint(path: str) -> None
    - load_checkpoint(path: str) -> None
```

### 4.5 预测模块 (`src/core/predictor.py`)

**职责**：
- 加载训练好的模型
- 单图和批量预测
- 后处理结果

**关键类**：
```python
class Predictor:
    - load_model(model_path: str) -> None
    - predict(image: np.ndarray) -> np.ndarray
    - predict_batch(images: List[np.ndarray]) -> List[np.ndarray]
    - post_process(mask: np.ndarray) -> np.ndarray
```

### 4.6 可视化模块 (`src/core/visualization.py`)

**职责**：
- 统计分析
- 图表生成
- 报告导出

**关键类**：
```python
class Visualizer:
    - plot_defect_distribution() -> plt.Figure
    - plot_training_history() -> plt.Figure
    - generate_report(output_path: str) -> None
    - export_excel_report() -> None
    - export_pdf_report() -> None
```

---

## 5. 前端 UI 架构

### 5.1 主窗口布局
```
┌─────────────────────────────────────┐
│         菜单栏（File, Edit...）      │
├─────────────────────────────────────┤
│  │ 工具栏（导入、标注、训练...）     │
├──┼─────────────────┬─────────────────┤
│  │                 │                 │
│文│   图片编辑      │   属性面板      │
│件│   画布          │   统计信息      │
│浏│ (QGraphicsView) │   日志输出      │
│览│                 │                 │
│  │                 │                 │
├──┼─────────────────┴─────────────────┤
│             状态栏                   │
└─────────────────────────────────────┘
```

### 5.2 核心 Widgets
- `ImageCanvas` - 图片编辑和交互
- `FileBrowser` - 文件浏览
- `AnnotationToolbar` - 标注工具条
- `LogViewer` - 日志查看器

---

## 6. 数据流

```
导入 → 浏览 → SAM 自动标注 → 人工修正 → 保存
          ↓
       标注管理 → 导出 COCO/YOLO
                    ↓
                  训练数据集
                    ↓
                  模型训练
                    ↓
                  模型权重
                    ↓
                  预测推理
                    ↓
                可视化 + 报告
```

---

## 7. 配置文件示例

### config/config.yaml
```yaml
app:
  name: "Industrial Defect Segmentation"
  version: "1.0.0"
  
ui:
  theme: "light"
  default_canvas_size: [1024, 768]
  
models:
  default_segmentation_model: "unet"
  sam_model_type: "vit_h"
  device: "cuda"
  
training:
  batch_size: 16
  learning_rate: 0.001
  epochs: 50
  optimizer: "adam"
```

---

## 8. 开发顺序

1. 项目结构和基础框架搭建
2. 数据管理和文件浏览
3. 图片编辑画布和基础标注工具
4. SAM 集成
5. 标注数据导出
6. 模型训练管道
7. 预测推理
8. 可视化和报告
9. 测试和优化
10. 部署和打包
