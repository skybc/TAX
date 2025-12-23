# 快速开发指南

## 1. 项目初始化

### 1.1 克隆和设置

```bash
# 克隆项目
git clone <repo-url>
cd industrial-defect-segmentation

# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 下载 SAM 模型权重
mkdir -p models/checkpoints
cd models/checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
cd ../../

# 创建数据目录
mkdir -p data/{raw,processed/{images,masks,annotations}}
```

### 1.2 验证安装

```bash
# 检查 PyTorch
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"

# 检查 PyQt5
python -c "from PyQt5.QtWidgets import QApplication; print('PyQt5 OK')"

# 检查 SAM
python -c "from segment_anything import SamPredictor; print('SAM OK')"

# 运行测试
pytest tests/
```

---

## 2. 代码风格和规范

### 2.1 代码格式化

```bash
# 使用 Black 格式化代码
black src/ scripts/ tests/

# 检查风格问题
flake8 src/ --max-line-length=100

# 类型检查
mypy src/
```

### 2.2 代码注释规范

```python
# Google 风格 docstring

def train_model(model: nn.Module, 
                train_loader: DataLoader,
                epochs: int = 100) -> Dict[str, float]:
    """训练分割模型。
    
    Args:
        model: PyTorch 分割模型实例
        train_loader: 训练数据加载器
        epochs: 训练轮数，默认 100
        
    Returns:
        包含训练指标的字典 {"loss": ..., "iou": ...}
        
    Raises:
        ValueError: 如果模型不是 nn.Module
        RuntimeError: 如果 GPU 显存不足
        
    Example:
        >>> model = UNet()
        >>> history = train_model(model, train_loader, epochs=50)
        >>> print(history['loss'][-1])
    """
```

### 2.3 命名规范

```python
# 类名：PascalCase
class DataManager:
    pass

# 函数名：snake_case
def load_image_data():
    pass

# 常量：UPPER_CASE
MAX_IMAGE_SIZE = 1024
DEFAULT_BATCH_SIZE = 16

# 私有成员：_leading_underscore
def _internal_method():
    pass
```

---

## 3. 项目文件约定

### 3.1 模块导入顺序

```python
# 1. 标准库
import os
import sys
from typing import Dict, List

# 2. 第三方库
import numpy as np
import torch
from PyQt5.QtWidgets import QWidget

# 3. 本地模块
from src.core.data_manager import DataManager
from src.utils.mask_utils import apply_threshold
```

### 3.2 文件组织

```
src/
├── core/          # 业务逻辑（不依赖 UI）
├── ui/            # UI 相关（可依赖 core）
├── models/        # 模型定义
├── utils/         # 工具函数
├── threads/       # 多线程类
└── logger.py      # 日志配置
```

---

## 4. 常用开发命令

### 4.1 运行应用

```bash
# 运行主应用
python src/main.py

# 调试模式
python -m pdb src/main.py

# 使用特定配置
python src/main.py --config config/custom_config.yaml
```

### 4.2 训练模型

```bash
# 使用默认配置训练
python scripts/train.py --data_dir data/processed/

# 自定义参数
python scripts/train.py \
    --config config/training_config.yaml \
    --batch_size 32 \
    --epochs 200 \
    --lr 0.0001

# 继续之前的训练
python scripts/train.py --resume models/checkpoints/latest.pth
```

### 4.3 模型推理

```bash
# 单张图片预测
python scripts/evaluate.py \
    --model models/checkpoints/best_model.pth \
    --image_path test_image.jpg \
    --output_dir results/

# 批量预测
python scripts/evaluate.py \
    --model models/checkpoints/best_model.pth \
    --image_dir test_images/ \
    --output_dir results/
```

### 4.4 数据集准备

```bash
# 分割训练/验证集
python scripts/prepare_dataset.py \
    --data_dir data/processed/ \
    --train_ratio 0.8 \
    --val_ratio 0.1

# 导出为 COCO 格式
python scripts/prepare_dataset.py \
    --data_dir data/processed/ \
    --format coco \
    --output_dir data/coco_format/
```

---

## 5. 调试技巧

### 5.1 使用 Python Debugger

```python
import pdb

def problematic_function():
    x = 10
    pdb.set_trace()  # 在此处暂停
    y = x * 2
    return y

# 常用命令：
# n (next)     - 下一行
# s (step)     - 进入函数
# c (continue) - 继续执行
# p <var>      - 打印变量
# l (list)     - 显示代码
# w (where)    - 显示调用栈
# q (quit)     - 退出调试
```

### 5.2 使用日志调试

```python
from src.logger import logger

logger.info("程序启动")
logger.warning("警告信息")
logger.error("错误信息")
logger.debug("调试信息")

# 在配置中设置日志级别
# logging:
#   level: DEBUG  # INFO, WARNING, ERROR, CRITICAL
```

### 5.3 内存和性能分析

```bash
# 内存分析
pip install memory-profiler
python -m memory_profiler src/main.py

# 性能分析
pip install line_profiler
kernprof -l -v src/main.py
```

---

## 6. 测试编写

### 6.1 单元测试

```python
# tests/test_data_manager.py
import pytest
from src.core.data_manager import DataManager

class TestDataManager:
    
    @pytest.fixture
    def manager(self):
        """为每个测试创建一个 DataManager 实例"""
        return DataManager(data_root="tests/fixtures")
    
    def test_load_image(self, manager):
        """测试图片加载"""
        image = manager.load_image("tests/fixtures/test_image.jpg")
        assert image is not None
        assert image.shape == (1024, 1024, 3)
        
    def test_invalid_path(self, manager):
        """测试无效路径"""
        with pytest.raises(FileNotFoundError):
            manager.load_image("non_existent.jpg")
```

### 6.2 运行测试

```bash
# 运行所有测试
pytest

# 运行特定测试文件
pytest tests/test_data_manager.py

# 运行特定测试函数
pytest tests/test_data_manager.py::TestDataManager::test_load_image

# 显示打印输出
pytest -s

# 生成覆盖率报告
pytest --cov=src tests/

# 生成 HTML 覆盖率报告
pytest --cov=src --cov-report=html tests/
```

---

## 7. 常见问题解决

### 7.1 导入错误

**问题**：`ModuleNotFoundError: No module named 'src'`

**解决**：
```bash
# 在项目根目录运行
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 或在代码中
import sys
sys.path.insert(0, '/path/to/project')
```

### 7.2 GPU 显存不足

**问题**：`RuntimeError: CUDA out of memory`

**解决**：
```python
# 减少 batch size
config['batch_size'] = 8  # from 16

# 启用梯度检查点
model.enable_checkpointing()

# 清理缓存
import torch
torch.cuda.empty_cache()
```

### 7.3 PyQt5 显示问题

**问题**：窗口无法显示或闪退

**解决**：
```bash
# 检查依赖
pip install --upgrade PyQt5 PyQt5-sip

# 检查显示服务器（Linux）
echo $DISPLAY  # 应该是 :0 或类似

# 强制使用特定平台
export QT_QPA_PLATFORM=offscreen  # Headless
```

### 7.4 模型加载失败

**问题**：`RuntimeError: Error(s) in loading state_dict`

**解决**：
```python
# 检查模型和权重兼容性
model = UNet()
try:
    model.load_state_dict(torch.load('model.pth'))
except RuntimeError as e:
    # 使用 strict=False 忽略不匹配的键
    state_dict = torch.load('model.pth')
    model.load_state_dict(state_dict, strict=False)
```

---

## 8. 贡献指南

### 8.1 提交代码前

```bash
# 1. 更新分支
git pull origin main

# 2. 创建特性分支
git checkout -b feature/my-feature

# 3. 做出更改并测试
pytest

# 4. 格式化代码
black src/

# 5. 运行代码检查
flake8 src/
mypy src/

# 6. 提交
git add .
git commit -m "feat: add new segmentation model"

# 7. 推送
git push origin feature/my-feature
```

### 8.2 Commit 消息规范

```
<type>(<scope>): <subject>

<body>

<footer>

# type: feat, fix, docs, style, refactor, test, chore
# scope: core, ui, models, utils, etc.
# subject: 简洁描述，使用命令式

# Example:
feat(core): add SAM model support for automatic annotation
- Implement SAMHandler class with point and bbox prompts
- Add async inference thread to avoid GUI blocking
- Update ImageCanvas to display SAM results

# Closes: #123
```

---

## 9. 文档编写

### 9.1 添加功能文档

```markdown
# Feature Name

## 概述
功能的简短描述。

## 使用方法
如何使用该功能的代码示例。

## API Reference
详细的 API 文档。

## 性能
性能相关的信息。

## 相关链接
- [链接](url)
```

### 9.2 更新 README

- 添加新功能到功能列表
- 更新依赖版本
- 添加使用示例

---

## 10. 发布检查清单

发布新版本前：

- [ ] 所有测试通过 (`pytest`)
- [ ] 代码风格检查通过 (`flake8`, `black`)
- [ ] 类型检查通过 (`mypy`)
- [ ] 文档已更新
- [ ] CHANGELOG 已更新
- [ ] 版本号已更新 (`__version__`)
- [ ] 标签已创建 (`git tag v1.0.0`)
- [ ] 发布到 PyPI (可选)

---

## 11. 快速参考

### 11.1 关键文件位置

```
config/
├── config.yaml          # 应用配置
├── paths.yaml           # 路径配置
└── hyperparams.yaml     # 超参数

src/
├── main.py             # 入口点
├── core/
│   ├── data_manager.py
│   ├── annotation_manager.py
│   ├── sam_handler.py
│   ├── model_trainer.py
│   ├── predictor.py
│   └── visualization.py
└── ui/
    ├── main_window.py
    └── widgets/
        ├── image_canvas.py
        ├── file_browser.py
        └── log_viewer.py

scripts/
├── train.py            # 训练脚本
├── evaluate.py         # 评估脚本
└── prepare_dataset.py  # 数据准备

tests/
└── test_*.py           # 测试文件
```

### 11.2 常用快捷键（PyQt5）

```
Ctrl+O        - 打开图片
Ctrl+S        - 保存
Ctrl+Z        - 撤销
Ctrl+Y        - 重做
Ctrl+Q        - 退出
```

---

## 12. 资源

- [PyQt5 官方文档](https://doc.qt.io/qt-5/)
- [PyTorch 官方文档](https://pytorch.org/docs/)
- [SAM 项目](https://github.com/facebookresearch/segment-anything)
- [Segmentation Models PyTorch](https://github.com/qubvel/segmentation_models.pytorch)
- [Albumentations](https://albumentations.ai/)

---

祝您开发顺利！有问题请查看文档或提出 Issue。
