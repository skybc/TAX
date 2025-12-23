# 数据管理模块技术文档

## 1. 模块概述

**模块名称**：数据管理模块  
**文件位置**：`src/core/data_manager.py`  
**职责**：负责图片/视频的加载、缓存、批量管理和数据集组织

---

## 2. 核心类设计

### 2.1 `DataManager` 类

```python
class DataManager:
    """数据管理主类"""
    
    def __init__(self, data_root: str = "data/processed"):
        self.data_root = data_root
        self.image_cache = {}  # 图片缓存
        self.current_dataset = None
        self.file_list = []
        self.current_index = 0
        
    # ============ 数据加载 ============
    def load_image(self, path: str) -> np.ndarray:
        """加载单张图片"""
        
    def load_video(self, path: str) -> Generator[np.ndarray]:
        """逐帧加载视频"""
        
    def load_folder(self, folder_path: str) -> List[str]:
        """加载文件夹下所有支持的图片"""
        
    # ============ 缓存管理 ============
    def cache_image(self, path: str, image: np.ndarray) -> None:
        """将图片缓存到内存"""
        
    def get_cached_image(self, path: str) -> Optional[np.ndarray]:
        """获取缓存的图片"""
        
    def clear_cache(self) -> None:
        """清空缓存"""
        
    # ============ 数据集管理 ============
    def create_dataset_structure(self, root: str) -> None:
        """创建标准数据集结构"""
        
    def save_dataset_config(self, config: Dict) -> None:
        """保存数据集配置"""
        
    def load_dataset_config(self) -> Dict:
        """加载数据集配置"""
        
    # ============ 批量操作 ============
    def get_file_list(self) -> List[str]:
        """获取当前文件列表"""
        
    def get_image_by_index(self, idx: int) -> Tuple[np.ndarray, str]:
        """按索引获取图片和路径"""
        
    def get_next_image(self) -> Tuple[np.ndarray, str]:
        """获取下一张图片"""
        
    def get_prev_image(self) -> Tuple[np.ndarray, str]:
        """获取上一张图片"""
```

### 2.2 支持的文件格式

**图片格式**：
- `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.webp`

**视频格式**：
- `.mp4`, `.avi`, `.mov`, `.mkv`

---

## 3. 工作流

```
用户选择文件/文件夹
        ↓
DataManager.load_folder() / load_image()
        ↓
生成文件列表
        ↓
缓存管理（可选）
        ↓
UI 展示（QGraphicsView）
```

---

## 4. 关键实现细节

### 4.1 内存管理

- **缓存策略**：LRU（最近最少使用）
- **缓存大小限制**：可配置（默认 500MB）
- **图片预加载**：支持预加载下一张

### 4.2 视频处理

```python
def load_video(self, path: str) -> Generator[np.ndarray]:
    """
    逐帧读取视频，避免一次加载整个视频到内存
    """
    cap = cv2.VideoCapture(path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cap.release()
```

### 4.3 图片预处理

```python
def load_image(self, path: str) -> np.ndarray:
    """
    1. 检查缓存
    2. 读取图片（处理透明通道）
    3. 颜色空间转换 (BGR → RGB)
    4. 数据类型转换 (uint8)
    """
```

---

## 5. API 接口

### 5.1 初始化
```python
dm = DataManager(data_root="data/processed")
```

### 5.2 加载文件夹
```python
file_list = dm.load_folder("raw_images/")
# 返回：["image1.jpg", "image2.png", ...]
```

### 5.3 按索引访问
```python
image, path = dm.get_image_by_index(0)
next_image, next_path = dm.get_next_image()
```

### 5.4 创建数据集结构
```python
dm.create_dataset_structure("data/processed/")
# 创建：images/, masks/, annotations/ 目录
```

---

## 6. 与其他模块的交互

```
┌─────────────────┐
│  UI 层          │ (QGraphicsView, FileBrowser)
└────────┬────────┘
         │
┌────────▼─────────┐
│ DataManager      │ ◄─── 与文件系统交互
└────────┬─────────┘
         │
┌────────▼──────────────┐
│ AnnotationManager     │ ◄─── 读取/保存标注
│ ModelTrainer         │ ◄─── 提供训练数据
│ Predictor           │ ◄─── 提供预测数据
└──────────────────────┘
```

---

## 7. 错误处理

- 文件不存在：抛出 `FileNotFoundError`
- 不支持的格式：抛出 `ValueError`
- 内存不足：自动清理缓存
- 视频损坏：记录日志并跳过

---

## 8. 性能指标

| 操作 | 预期耗时 |
|------|---------|
| 加载单张图片 (1920x1080) | < 50ms |
| 加载 1000 张图片列表 | < 500ms |
| 视频逐帧读取 (30fps) | 实时 |
| 缓存查询 | < 1ms |

---

## 9. 配置参数

```yaml
# config/paths.yaml
data_manager:
  cache_size_mb: 500
  preload_next: true
  supported_formats:
    image: [jpg, jpeg, png, bmp, tiff, webp]
    video: [mp4, avi, mov, mkv]
```

---

## 10. 单元测试用例

```python
def test_load_image():
    """测试加载单张图片"""
    
def test_load_folder():
    """测试加载文件夹"""
    
def test_image_cache():
    """测试缓存机制"""
    
def test_video_loading():
    """测试视频逐帧读取"""
    
def test_memory_management():
    """测试内存管理和缓存清理"""
```
