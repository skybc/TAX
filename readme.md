# 工业缺陷分割系统

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyQt5 5.15+](https://img.shields.io/badge/PyQt5-5.15+-green.svg)](https://pypi.org/project/PyQt5/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

一套完整的工业缺陷分割系统，实现 **标注 → 模型训练 → 预测 → 可视化** 全流程。集成 SAM（Segment Anything）自动标注、多种深度学习模型训练、批量推理和报告生成功能。

## 🎯 核心特性

- **🤖 SAM 自动标注**：基于 Facebook AI 的 SAM 模型，支持点击、框选、文字等多种提示方式进行自动分割
- **✏️ 半自动标注**：SAM 自动生成 → 人工修正的闭环流程，支持笔刷、橡皮、多边形工具
- **🧠 多模型支持**：U-Net、DeepLabV3+、YOLOv11-Seg，灵活选择最适合的模型架构
- **⚡ 异步训练推理**：QThread 异步处理，保证 GUI 响应流畅，实时显示训练曲线
- **📊 批量推理与可视化**：支持单张/批量/视频逐帧预测，生成 Excel/PDF 报告和统计分析
- **💾 多格式导出**：支持 COCO JSON、YOLO txt、PNG mask 等多种标准格式
- **🎨 完整的 UI 工具**：基于 PyQt5 的专业级图形界面，支持缩放、平移、图层管理

## 🚀 快速开始

### 1. 环境配置

```bash
# 克隆项目
git clone <repo-url>
cd industrial-defect-segmentation

# 创建虚拟环境（推荐 Python 3.10）
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 下载模型权重

```bash
# 创建模型目录
mkdir -p models/checkpoints

# 下载 SAM 模型（选择一个）
# ViT-H（推荐，准确率最高）
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O models/checkpoints/sam_vit_h.pth

# 或 ViT-B（速度快，显存占用小）  
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -O models/checkpoints/sam_vit_b.pth
```

### 3. 创建数据目录

```bash
mkdir -p data/{raw,processed/{images,masks,annotations},splits,outputs/{predictions,reports,models}}
```

### 4. 运行应用

```bash
# 启动 PyQt5 应用
python src/main.py

# 或指定自定义配置
python src/main.py --config config/custom_config.yaml
```

## 📁 项目结构

```
industrial-defect-segmentation/
├── README.md                           # 本文件
├── requirements.txt                    # 依赖列表
├── setup.py                           # 项目配置
│
├── config/                            # 配置文件目录
│   ├── config.yaml                    # 应用全局配置
│   ├── paths.yaml                     # 路径配置
│   └── hyperparams.yaml               # 模型超参数
│
├── src/                               # 源代码目录
│   ├── main.py                        # 应用入口点
│   │
│   ├── ui/                           # PyQt5 前端
│   │   ├── main_window.py            # 主窗口
│   │   ├── dialogs/                  # 对话框
│   │   │   ├── import_dialog.py      # 导入图片对话框
│   │   │   ├── train_config_dialog.py # 训练配置对话框
│   │   │   └── export_dialog.py      # 导出对话框
│   │   ├── widgets/                  # 自定义 Widget
│   │   │   ├── image_canvas.py       # 图片编辑画布（核心）
│   │   │   ├── annotation_toolbar.py # 标注工具条
│   │   │   ├── file_browser.py       # 文件浏览器
│   │   │   └── log_viewer.py         # 日志查看器
│   │   └── styles/
│   │       └── stylesheet.qss        # UI 样式表
│   │
│   ├── core/                         # 核心业务逻辑（不依赖UI）
│   │   ├── data_manager.py           # 数据管理（图片/视频加载）
│   │   ├── annotation_manager.py     # 标注管理（mask保存/加载/undo-redo）
│   │   ├── sam_handler.py            # SAM处理器（模型生命周期）
│   │   ├── model_trainer.py          # 模型训练（训练循环）
│   │   ├── predictor.py              # 预测推理（模型推理）
│   │   └── visualization.py          # 可视化（统计图表生成）
│   │
│   ├── models/                       # 模型定义
│   │   ├── unet.py                   # U-Net 实现
│   │   ├── deeplabv3.py              # DeepLabV3+ 实现
│   │   └── yolov11_seg.py            # YOLOv11 分割模型
│   │
│   ├── utils/                        # 工具函数模块
│   │   ├── mask_utils.py             # Mask 处理（二值化、RLE编码）
│   │   ├── bbox_utils.py             # 边界框处理
│   │   ├── file_utils.py             # 文件操作
│   │   ├── image_utils.py            # 图片处理（缩放、归一化）
│   │   ├── metrics.py                # 评估指标（IoU、Dice）
│   │   ├── augmentation.py           # 数据增强管道
│   │   └── export_utils.py           # 格式转换（COCO/YOLO）
│   │
│   ├── threads/                      # 异步处理线程
│   │   ├── training_thread.py        # 训练线程
│   │   ├── inference_thread.py       # 推理线程
│   │   └── sam_inference_thread.py   # SAM推理线程
│   │
│   └── logger.py                     # 日志配置
│
├── data/                             # 数据目录
│   ├── raw/                          # 原始数据存储
│   ├── processed/                    # 处理后的标注数据
│   │   ├── images/                   # 图片副本
│   │   ├── masks/                    # Mask PNG 文件
│   │   └── annotations/              # 标注元数据
│   ├── splits/                       # 数据分割文件
│   │   ├── train.txt                 # 训练集列表
│   │   ├── val.txt                   # 验证集列表
│   │   └── test.txt                  # 测试集列表
│   └── outputs/                      # 输出结果目录
│       ├── predictions/              # 模型预测结果
│       ├── reports/                  # 生成的报告
│       └── models/                   # 保存的模型权重
│
├── models/                           # 预训练模型权重目录
│   └── checkpoints/                  # 模型检查点存储
│       ├── sam_vit_h.pth            # SAM ViT-H 权重
│       └── best_model.pth           # 最优训练模型
│
├── tests/                            # 测试目录
│   ├── test_data_manager.py         # 数据管理单元测试
│   ├── test_annotation.py           # 标注功能测试
│   ├── test_models.py               # 模型单元测试
│   └── test_utils.py                # 工具函数测试
│
├── scripts/                          # 独立脚本目录
│   ├── prepare_dataset.py           # 数据集准备脚本
│   ├── train.py                     # 命令行训练脚本
│   ├── evaluate.py                  # 模型评估脚本
│   └── export_onnx.py               # ONNX 模型导出
│
├── doc/                             # 文档目录
│   ├── readme.md                    # 系统概览
│   ├── architecture-design.md       # 架构设计文档
│   ├── quick-start-guide.md         # 快速开发指南
│   ├── tech-stack-dependencies.md   # 技术栈选择
│   └── implementation-timeline.md   # 实现时间表
│
├── .github/                         # GitHub 配置
│   └── copilot-instructions.md      # AI 代理开发指南
│
└── .vscode/                         # VS Code 配置（可选）
    ├── settings.json
    └── launch.json
```

## 🛠️ 核心工作流

### 工作流程：标注 → 训练 → 推理 → 报告

```
导入图片/视频
    ↓
浏览和预处理
    ↓
SAM 自动标注 (点击/框选)
    ↓
人工修正 (笔刷/橡皮/多边形)
    ↓
保存标注数据 (PNG mask + JSON 元数据)
    ↓
导出为标准格式 (COCO/YOLO)
    ↓
划分训练/验证集
    ↓
配置和训练模型 (选择架构/超参数)
    ↓
选择最优权重进行推理
    ↓
批量预测和可视化
    ↓
生成统计报告 (Excel/PDF)
```

## 📚 文档导航

| 文档 | 内容 |
|------|------|
| [快速开发指南](doc/quick-start-guide.md) | 环境配置、常用命令、调试技巧、常见问题 |
| [架构设计文档](doc/architecture-design.md) | 系统设计、模块职责、数据流、配置示例 |
| [技术栈详解](doc/tech-stack-dependencies.md) | 依赖选择理由、版本兼容性、安装脚本 |
| [实现时间表](doc/implementation-timeline.md) | 开发阶段、任务清单、里程碑规划 |
| [AI 开发指南](.github/copilot-instructions.md) | 代码约定、关键流程、调试场景 |

## 🔧 常用命令

### 开发和测试

```bash
# 运行应用
python src/main.py

# 运行所有测试
pytest tests/

# 运行特定测试
pytest tests/test_data_manager.py -v

# 代码格式化
black src/ scripts/ tests/

# 代码检查
flake8 src/ --max-line-length=100
mypy src/
```

### 模型训练

```bash
# 使用默认配置训练
python scripts/train.py --data_dir data/processed/

# 自定义参数训练
python scripts/train.py \
    --model unet \
    --batch_size 32 \
    --epochs 200 \
    --learning_rate 0.0001

# 继续之前的训练
python scripts/train.py --resume models/checkpoints/latest.pth
```

### 模型推理

```bash
# 单张图片预测
python scripts/evaluate.py \
    --model models/checkpoints/best_model.pth \
    --image_path test_image.jpg

# 批量预测
python scripts/evaluate.py \
    --model models/checkpoints/best_model.pth \
    --image_dir test_images/ \
    --output_dir results/
```

### 数据集准备

```bash
# 分割训练/验证集
python scripts/prepare_dataset.py \
    --data_dir data/processed/ \
    --train_ratio 0.8 \
    --val_ratio 0.1

# 导出为 COCO 格式
python scripts/prepare_dataset.py \
    --format coco \
    --output_dir data/coco_format/
```

## 🎓 开发指南

### 新手入门

1. **阅读快速开始**：[快速开发指南](doc/quick-start-guide.md) 第 1-3 章
2. **了解架构**：[架构设计文档](doc/architecture-design.md) 第 4-5 章  
3. **查看代码示例**：`src/core/data_manager.py` 和 `src/ui/main_window.py`
4. **运行测试**：`pytest tests/` 验证环境正确性

### 代码规范

- **命名**：类用 PascalCase，函数用 snake_case，常量用 UPPER_CASE
- **文档**：使用 Google 风格 docstring
- **导入**：标准库 → 第三方 → 本地模块
- **线程**：所有耗时操作必须在 `QThread` 中运行，通过 signal/slot 通信

### 关键约定

| 方面 | 规则 |
|------|------|
| **配置** | 在 `config/config.yaml` 中定义，启动时加载，传递给模块 |
| **设备** | 在配置中指定（`cuda`/`cpu`），模块从配置读取 |
| **错误处理** | 线程中捕获异常，通过 signal 发射到 UI |
| **日志** | 使用 `src/logger.py` 中配置的 logger，避免 print() |
| **单元测试** | 测试 core 模块，不测试 UI（使用 mock） |

## 🚨 常见问题

### GUI 冻结

**问题**：运行 SAM 推理或模型训练时 GUI 无响应  
**解决**：确保在 `SAMInferenceThread` 或 `TrainingThread` 中运行，不在主线程调用

### 显存不足

**问题**：训练时 CUDA out of memory  
**解决**：
```python
# config.yaml 中修改
training:
  batch_size: 8  # 从 16 改小
  
# 或在代码中启用梯度检查点
model.enable_checkpointing()
```

### 模型加载失败

**问题**：`RuntimeError: Error(s) in loading state_dict`  
**解决**：检查模型架构和权重文件兼容性，或使用 `strict=False`
```python
model.load_state_dict(torch.load('model.pth'), strict=False)
```

更多问题请参考 [快速开发指南 - 常见问题解决](doc/quick-start-guide.md#7-常见问题解决)

## 📊 性能指标

| 操作 | 目标 | 环境 |
|------|------|------|
| 图片加载 | <500ms | OpenCV 缓存 |
| SAM 推理（ViT-H） | ~800ms | RTX 3070 |
| SAM 推理（ViT-B） | ~200ms | RTX 3070 |
| U-Net 训练（32张） | <200ms/batch | RTX 3070 |
| 批量导出（1000张） | <5s | COCO JSON |

## 🔌 依赖版本

**核心依赖**（完整列表见 `requirements.txt`）：

```
PyQt5==5.15.9              # GUI 框架
torch==2.1.0               # 深度学习框架
torchvision==0.16.0        # 预训练模型
segment-anything           # SAM 自动分割
segmentation-models-pytorch # U-Net/DeepLabV3+
ultralytics==8.0.195       # YOLOv11
opencv-python==4.8.1.78    # 图像处理
numpy==1.24.3              # 数值计算
albumentations==1.3.1      # 数据增强
pycocotools==2.0.6         # COCO 格式
pytest==7.4.2              # 测试框架
```

## 💻 系统要求

| 组件 | 最低要求 | 推荐配置 |
|------|---------|---------|
| **Python** | 3.9 | 3.10+ |
| **RAM** | 8GB | 16GB+ |
| **GPU** | - | RTX 3070 或更高 |
| **CUDA** | - | 11.8+ |
| **驱动** | - | 515+ |

## 📝 许可证

- **项目代码**：MIT 许可
- **SAM 模型**：CC-BY-NC 2.0（学术/研究用途）- 商业使用需获得许可

## 🤝 贡献指南

1. 创建特性分支：`git checkout -b feature/your-feature`
2. 提交前运行测试：`pytest tests/`
3. 格式化代码：`black src/`
4. 提交时遵循规范：`feat: add xxx`、`fix: resolve xxx`

详见 [快速开发指南 - 贡献指南](doc/quick-start-guide.md#8-贡献指南)

## 📞 获取帮助

- 📖 查看 [文档目录](doc/)
- 🐛 报告 Bug：GitHub Issues
- 💬 讨论功能：GitHub Discussions
- 📧 联系开发者：[邮箱]

## 🎯 项目路线图

- [x] 项目框架和配置
- [x] 数据管理模块
- [x] 基础标注工具
- [x] SAM 集成
- [ ] 模型训练模块
- [ ] 预测推理模块
- [ ] 可视化和报告
- [ ] 全系统集成和发布

详见 [实现时间表](doc/implementation-timeline.md)

---

**最后更新**：2025-12-23  
**版本**：1.0.0-dev  
**维护者**：Industrial AI Team

