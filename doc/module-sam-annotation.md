# SAM 自动标注模块技术文档

## 1. 模块概述

**模块名称**：SAM 自动标注模块  
**文件位置**：`src/core/sam_handler.py` + `src/threads/sam_inference_thread.py`  
**职责**：集成 Segment Anything 模型，提供交互式自动标注功能

---

## 2. 依赖与安装

```bash
pip install segment-anything torch torchvision opencv-python
```

**SAM 模型权重下载**：
```
ViT-H: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth (~2.5GB)
ViT-L: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth (~1.2GB)
ViT-B: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth (~375MB)
MobileSAM: 更轻量级版本
```

---

## 3. 核心类设计

### 3.1 `SAMHandler` 主类

```python
from segment_anything import SamPredictor, sam_model_registry

class SAMHandler:
    """SAM 处理器主类"""
    
    def __init__(self, model_type: str = "vit_h", device: str = "cuda"):
        """
        Args:
            model_type: "vit_h" | "vit_l" | "vit_b" | "mobile_sam"
            device: "cuda" | "cpu"
        """
        self.model_type = model_type
        self.device = device
        self.predictor = None
        self.model = None
        self.image_embedding = None
        self.current_image = None
        
    # ============ 模型初始化 ============
    def load_model(self, checkpoint_path: str = None) -> None:
        """加载 SAM 模型到显存"""
        
    def unload_model(self) -> None:
        """卸载模型释放显存"""
        
    # ============ 图片处理 ============
    def set_image(self, image: np.ndarray) -> None:
        """
        设置要处理的图片并生成 embedding
        Args:
            image: np.ndarray, 形状 (H, W, 3), RGB 格式
        """
        
    def encode_image(self) -> np.ndarray:
        """获取当前图片的 embedding"""
        
    # ============ 推理预测 ============
    def predict_with_points(self, 
                            points: np.ndarray,
                            labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        使用点提示进行预测
        Args:
            points: shape (N, 2), 坐标 [[x1, y1], [x2, y2], ...]
            labels: shape (N,), 1=前景点, 0=背景点
        Returns:
            masks: shape (N, H, W)
            scores: shape (N,) 置信度
            logits: shape (N, H, W) 原始输出
        """
        
    def predict_with_bbox(self, 
                          bbox: Tuple[int, int, int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        使用边界框提示进行预测
        Args:
            bbox: (x_min, y_min, x_max, y_max)
        Returns:
            同上
        """
        
    def predict_with_mask(self, 
                         mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        使用前一帧 mask 作为提示进行预测（用于视频续标）
        """
        
    def predict_combined(self,
                        points: np.ndarray = None,
                        labels: np.ndarray = None,
                        bbox: Tuple = None,
                        mask: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        结合多种提示进行预测
        """
```

### 3.2 `SAMInferenceThread` 线程类

```python
from PyQt5.QtCore import QThread, pyqtSignal

class SAMInferenceThread(QThread):
    """异步 SAM 推理线程，避免 GUI 阻塞"""
    
    # 信号定义
    inference_finished = pyqtSignal(np.ndarray)  # 推理完成，返回 mask
    inference_error = pyqtSignal(str)            # 推理出错
    progress_updated = pyqtSignal(int)           # 进度更新
    
    def __init__(self, sam_handler: SAMHandler):
        super().__init__()
        self.sam_handler = sam_handler
        self.prompts = {}
        self.stop_flag = False
        
    def set_prompts(self, prompts: Dict) -> None:
        """设置推理提示"""
        
    def run(self) -> None:
        """线程主体，执行推理"""
        
    def stop(self) -> None:
        """停止推理线程"""
```

---

## 4. 提示类型和交互方式

### 4.1 点提示 (Point Prompt)

```python
# 前景点
points = np.array([[100, 150], [200, 250]])
labels = np.array([1, 1])  # 1=前景

# 前景+背景点
points = np.array([[100, 150], [200, 250], [300, 300]])
labels = np.array([1, 1, 0])  # 0=背景
```

### 4.2 边界框提示 (Bbox Prompt)

```python
bbox = (100, 150, 300, 400)  # (x_min, y_min, x_max, y_max)
```

### 4.3 多模态组合提示

```python
prompts = {
    "points": np.array([[100, 150], [200, 250]]),
    "labels": np.array([1, 0]),
    "bbox": (50, 100, 400, 500),
    "mask": None
}
```

---

## 5. 工作流

```
用户点击图片 / 绘制框
        ↓
收集提示信息
        ↓
启动 SAMInferenceThread
        ↓
SAMHandler.set_image() 编码图片
        ↓
SAMHandler.predict_*() 推理
        ↓
返回 mask
        ↓
UI 展示 mask
        ↓
用户人工修正 (可选)
        ↓
保存
```

---

## 6. 关键实现细节

### 6.1 模型初始化和显存管理

```python
def load_model(self, checkpoint_path: str = None) -> None:
    """
    1. 下载或使用本地权重
    2. 注册模型
    3. 加载到设备（GPU/CPU）
    4. 设置推理模式
    """
    sam = sam_model_registry[self.model_type](
        checkpoint=checkpoint_path or self._get_default_checkpoint()
    )
    sam.to(device=self.device)
    self.predictor = SamPredictor(sam)
    self.model = sam
```

### 6.2 图片编码优化

```python
def set_image(self, image: np.ndarray) -> None:
    """
    关键优化：
    - 预处理（归一化、大小调整）
    - 一次编码，多次推理
    - 缓存 embedding 避免重复计算
    """
    # 检查输入格式
    assert image.ndim == 3 and image.shape[2] == 3
    assert image.dtype == np.uint8
    
    # 设置图片并编码
    self.predictor.set_image(image)
    self.current_image = image
    self.image_embedding = self.predictor.get_image_embedding()
```

### 6.3 多实例处理

```python
def predict_with_points(self, points, labels):
    """
    SAM 输出多个 mask（对应不同 IoU 阈值）
    需要选择最优的 mask
    """
    masks, scores, logits = self.predictor.predict(
        point_coords=points,
        point_labels=labels,
        multimask_output=True
    )
    
    # 选择置信度最高的 mask
    best_mask = masks[scores.argmax()]
    return best_mask
```

### 6.4 后处理

```python
def post_process_mask(self, mask: np.ndarray) -> np.ndarray:
    """
    1. 二值化处理
    2. 腐蚀膨胀去噪
    3. 连通域分析
    """
    # 二值化
    binary_mask = (mask > 0.5).astype(np.uint8) * 255
    
    # 形态学操作
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    
    return binary_mask
```

---

## 7. 性能优化

| 优化方法 | 效果 |
|---------|------|
| 使用 MobileSAM | 推理时间减少 90% |
| 模型量化 (int8) | 显存使用减少 4 倍 |
| 批量推理 | 吞吐量提升 3-5 倍 |
| 异步线程 | GUI 响应时间 < 100ms |

---

## 8. 错误处理

```python
class SAMError(Exception):
    """SAM 特定错误"""
    pass

# 错误类型
- ModelLoadError: 模型加载失败
- InferenceError: 推理失败
- GPUOOMError: 显存不足
- InvalidPromptError: 提示信息无效
```

---

## 9. 配置参数

```yaml
# config/config.yaml
sam:
  model_type: "vit_h"  # vit_h | vit_l | vit_b | mobile_sam
  device: "cuda"       # cuda | cpu
  checkpoint_dir: "models/checkpoints/"
  max_batch_size: 4
  use_fp16: true       # 混合精度推理
  
  inference:
    multimask_output: true
    return_logits: true
    confidence_threshold: 0.5
```

---

## 10. 单元测试用例

```python
def test_model_loading():
    """测试模型加载"""
    
def test_point_prompt():
    """测试点提示推理"""
    
def test_bbox_prompt():
    """测试边界框推理"""
    
def test_combined_prompts():
    """测试组合提示"""
    
def test_inference_thread():
    """测试异步推理线程"""
    
def test_memory_management():
    """测试显存管理和模型卸载"""
```

---

## 11. 与 UI 的集成

```python
# main_window.py
class MainWindow(QMainWindow):
    def setup_sam(self):
        self.sam_handler = SAMHandler(model_type="vit_h")
        self.sam_thread = SAMInferenceThread(self.sam_handler)
        self.sam_thread.inference_finished.connect(self.on_mask_generated)
        
    def on_image_canvas_mouse_click(self, event):
        """收集点提示"""
        point = [event.pos().x(), event.pos().y()]
        # 发送到 SAM 推理线程
        self.sam_thread.set_prompts({"points": [...], "labels": [...]})
        self.sam_thread.start()
        
    def on_mask_generated(self, mask: np.ndarray):
        """处理推理结果"""
        self.image_canvas.overlay_mask(mask)
```

---

## 12. 扩展支持

- **多种 backbone**：ViT-H/L/B、MobileSAM、FastSAM
- **实时推理**：支持流式处理
- **模型蒸馏**：训练轻量级 SAM 替代品
- **ONNX 导出**：部署到边缘设备
