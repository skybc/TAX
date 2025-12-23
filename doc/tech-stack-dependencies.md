# 技术栈详细选择与依赖配置

## 1. Python 环境

### 1.1 Python 版本

**选择**：Python 3.9 - 3.11  
**原因**：
- 3.9：稳定性强，广泛支持
- 3.10/3.11：新特性支持、性能提升

**不选择 3.8**：部分库已停止支持  
**不选择 3.12**：部分库尚未完全兼容

---

## 2. GUI 框架选择

### 2.1 PyQt5 vs PyQt6 vs PySide6

| 特性 | PyQt5 | PyQt6 | PySide6 |
|------|-------|-------|---------|
| 发布时间 | 2016 | 2021 | 2021 |
| 社区规模 | 大 | 中 | 大 |
| 许可 | GPL/商 | GPL/商 | LGPL |
| 文档质量 | 优 | 中 | 优 |
| 性能 | 优 | 优 | 优 |
| 第三方库支持 | 多 | 少 | 中 |

**最终选择**：**PyQt5**
- 原因：社区规模大、文档完整、与现有项目兼容

**版本**：5.15.x (最后的 5.x 分支)

---

## 3. 深度学习框架

### 3.1 PyTorch vs TensorFlow

| 维度 | PyTorch | TensorFlow |
|------|---------|-----------|
| 动态图 | ✓ | ✓ (2.x) |
| SAM 支持 | ✓ 官方 | ✗ |
| 模型库 | 丰富 | 丰富 |
| 易用性 | 高 | 中 |
| 部署 | TorchScript/ONNX | SavedModel/Lite |
| 性能 | 优 | 优 |

**最终选择**：**PyTorch 2.0+**
- 原因：SAM 官方基于 PyTorch，更好的 Python 一等支持

**关键库**：
```
torch==2.1.0
torchvision==0.16.0
```

---

## 4. 分割模型库

### 4.1 核心模型库

**segmentation_models_pytorch** (smp)
```
pip install segmentation-models-pytorch
```
- U-Net、DeepLabV3+、FPN、PAN 等
- 与 torchvision backbone 整合
- 官方 ImageNet 预训练权重

**SAM (Segment Anything)**
```
pip install git+https://github.com/facebookresearch/segment-anything.git
```
- 官方实现
- 支持多种 backbone (ViT-H/L/B)

**YOLOv11**
```
pip install ultralytics
```
- 包含分割（YOLOv11-Seg）
- 易用的 API

---

## 5. 数据处理与增强

### 5.1 核心库

**OpenCV**
```
pip install opencv-python==4.8.1.78
```
- 图片/视频读写
- 预处理（缩放、色彩转换）
- 形态学操作

**NumPy**
```
pip install numpy==1.24.3
```
- 数组操作
- Mask 处理

**Pillow**
```
pip install Pillow==10.0.0
```
- 图片 I/O
- 轻量级处理

**Albumentations**
```
pip install albumentations==1.3.1
```
- 高效的图像增强
- 支持 bbox、mask、keypoint
- 比 torchvision 更快

**scikit-image**
```
pip install scikit-image==0.21.0
```
- 高级图像处理
- 形态学操作、连通域标记

---

## 6. 数据格式与导出

### 6.1 COCO 格式

**pycocotools**
```
pip install pycocotools-windows  # Windows
pip install pycocotools          # Linux/Mac
```
- COCO JSON 数据集格式
- 评估指标计算

### 6.2 Excel/PDF 导出

**openpyxl**
```
pip install openpyxl==3.1.2
```
- Excel (.xlsx) 读写

**reportlab**
```
pip install reportlab==4.0.7
```
- PDF 生成
- 表格、图表、文本

**pandas**
```
pip install pandas==2.0.3
```
- 数据操作和表格

---

## 7. 可视化

### 7.1 Matplotlib (PyQt 集成)

```
pip install matplotlib==3.7.2
```

```python
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        fig = Figure(figsize=(5, 4), dpi=100)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)
```

### 7.2 PyQtGraph (可选，更快)

```
pip install pyqtgraph==0.13.3
```

- 实时绘图优化
- 用于训练曲线监控

### 7.3 Plotly (可选，交互式)

```
pip install plotly==5.17.0
```

- 交互式报告
- HTML 导出

---

## 8. 配置与日志

**PyYAML**
```
pip install PyYAML==6.0.1
```

**python-dotenv**
```
pip install python-dotenv==1.0.0
```

**loguru** (可选)
```
pip install loguru==0.7.0
```
- 更强大的日志库
- 彩色输出、自动 rotation

---

## 9. 测试框架

**pytest**
```
pip install pytest==7.4.2
pip install pytest-cov==4.1.0  # 代码覆盖率
```

**pytest-qt** (PyQt 测试)
```
pip install pytest-qt==4.2.0
```

**unittest** (Python 内置)
- 不需要额外安装

---

## 10. 开发工具

**Black** (代码格式化)
```
pip install black==23.9.1
```

**Pylint** (代码检查)
```
pip install pylint==2.17.5
```

**mypy** (类型检查)
```
pip install mypy==1.5.1
```

**flake8** (风格检查)
```
pip install flake8==6.1.0
```

---

## 11. 性能优化

**tqdm** (进度条)
```
pip install tqdm==4.66.1
```

**scikit-learn** (指标计算)
```
pip install scikit-learn==1.3.2
```

**optuna** (超参数优化，可选)
```
pip install optuna==3.14.0
```

---

## 12. 完整 requirements.txt

```
# Core GUI
PyQt5==5.15.9
PyQt5-sip==12.13.0

# Deep Learning
torch==2.1.0
torchvision==0.16.0
segment-anything @ git+https://github.com/facebookresearch/segment-anything.git
segmentation-models-pytorch==0.3.3
ultralytics==8.0.195

# Data Processing
opencv-python==4.8.1.78
numpy==1.24.3
Pillow==10.0.0
albumentations==1.3.1
scikit-image==0.21.0
pandas==2.0.3

# Visualization
matplotlib==3.7.2
pyqtgraph==0.13.3
plotly==5.17.0

# Data Formats
pycocotools==2.0.6
openpyxl==3.1.2
reportlab==4.0.7
PyYAML==6.0.1

# Utilities
scikit-learn==1.3.2
tqdm==4.66.1
python-dotenv==1.0.0

# Development
pytest==7.4.2
pytest-cov==4.1.0
pytest-qt==4.2.0
black==23.9.1
pylint==2.17.5
mypy==1.5.1
flake8==6.1.0

# Optional
loguru==0.7.0
tensorboard==2.14.0
```

---

## 13. 环境配置脚本

### 13.1 conda 环境创建

```bash
# 创建环境
conda create -n defect-seg python=3.10

# 激活环境
conda activate defect-seg

# 安装 PyTorch（GPU）
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 或 CPU only
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# 安装其他依赖
pip install -r requirements.txt
```

### 13.2 venv 环境创建

```bash
# 创建虚拟环境
python -m venv venv

# 激活
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

---

## 14. GPU 支持

### 14.1 CUDA 版本兼容性

| PyTorch | CUDA | cuDNN |
|---------|------|-------|
| 2.1.0 | 11.8, 12.1 | 8.7+ |
| 2.0.x | 11.7, 11.8 | 8.7+ |

### 14.2 检查 GPU

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_properties(0))
```

---

## 15. 性能基准

### 15.1 模型推理时间（相对基准）

| 模型 | 输入大小 | GPU (RTX 3070) | CPU (i7) |
|------|---------|---|----------|
| SAM ViT-H | 1024x1024 | ~800ms | ~5s |
| SAM ViT-B | 1024x1024 | ~200ms | ~1.5s |
| U-Net | 512x512 | ~20ms | ~100ms |
| DeepLabV3+ | 512x512 | ~30ms | ~150ms |

### 15.2 训练时间（估计）

| 数据集 | 模型 | GPU | 时间 |
|-------|------|-----|------|
| 1000 张 | U-Net | RTX 3070 | 30 分钟 |
| 5000 张 | DeepLabV3+ | RTX 3070 | 3-4 小时 |
| 10000 张 | YOLOv11-Seg | RTX 3070 | 8-12 小时 |

---

## 16. 第三方许可证

| 库 | 许可证 | 商业使用 |
|----|--------|---------|
| PyQt5 | GPL v3 | ✓ (商业许可) |
| PyTorch | BSD | ✓ |
| OpenCV | Apache 2.0 | ✓ |
| SAM | CC-BY-NC 2.0 | ✗ (学术/研究) |
| NumPy | BSD | ✓ |
| Matplotlib | PSF | ✓ |

**重要**：SAM 使用 CC-BY-NC 许可，商业使用需获得许可。

---

## 17. 常见问题解决

### 17.1 PyTorch 安装问题

```bash
# 清除缓存
pip cache purge

# 使用清华镜像
pip install -i https://pypi.tsinghua.edu.cn/simple torch

# 验证安装
python -c "import torch; print(torch.__version__)"
```

### 17.2 OpenCV 显示问题

```bash
# 如果 cv2.imshow() 无法显示
pip uninstall opencv-python
pip install opencv-contrib-python
```

### 17.3 PyQt5 导入错误

```bash
# Windows 可能需要
pip install --upgrade PyQt5
```

---

## 18. Docker 部署（可选）

```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3.10 python3-pip

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ /app/src/
WORKDIR /app

CMD ["python", "src/main.py"]
```

---

## 19. 打包与分发

### 19.1 PyInstaller

```bash
pip install pyinstaller

pyinstaller \
    --onefile \
    --windowed \
    --add-data="config:config" \
    --add-data="src/ui/styles:src/ui/styles" \
    src/main.py
```

### 19.2 发布到 PyPI

```bash
pip install build twine
python -m build
twine upload dist/*
```

---

## 20. CI/CD 配置（可选）

### 20.1 GitHub Actions

```yaml
# .github/workflows/tests.yml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      - run: pip install -r requirements.txt
      - run: pytest
```

---

## 总结

**推荐配置**：
- Python 3.10
- PyQt5 5.15.x
- PyTorch 2.1.0 + CUDA 11.8
- segmentation_models_pytorch
- SAM (latest)
- albumentations + OpenCV
- pytest 用于测试

**开发流程**：
1. 克隆仓库
2. 创建虚拟环境
3. `pip install -r requirements.txt`
4. 下载 SAM 模型权重
5. 运行 `python src/main.py`
