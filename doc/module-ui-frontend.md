# 前端 UI 架构技术文档

## 1. 模块概述

**模块名称**：PyQt5 前端 UI  
**文件位置**：`src/ui/`  
**职责**：提供用户交互界面，集成所有核心功能模块

---

## 2. 主窗口架构

### 2.1 `MainWindow` 类

```python
from PyQt5.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QVBoxLayout
from PyQt5.QtCore import pyqtSignal

class MainWindow(QMainWindow):
    """应用主窗口"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("工业缺陷分割系统 v1.0")
        self.setGeometry(100, 100, 1600, 1000)
        
        # 初始化各个模块
        self.setup_ui()
        self.setup_connections()
        self.setup_manager_instances()
        
    # ============ UI 构建 ============
    def setup_ui(self) -> None:
        """构建 UI 布局"""
        
    def create_menu_bar(self) -> None:
        """创建菜单栏"""
        
    def create_toolbar(self) -> None:
        """创建工具栏"""
        
    def create_central_widget(self) -> None:
        """创建中心 Widget"""
        
    def create_status_bar(self) -> None:
        """创建状态栏"""
        
    # ============ 信号连接 ============
    def setup_connections(self) -> None:
        """连接所有信号槽"""
        
    # ============ 业务逻辑 ============
    def setup_manager_instances(self) -> None:
        """初始化数据管理、模型训练等实例"""
```

### 2.2 主窗口布局结构

```
┌─────────────────────────────────────┐
│         菜单栏                      │
│ File(打开/保存/导出) Edit(撤销/重做) │
│ View(视图) Tools(工具) Help(帮助)    │
├─────────────────────────────────────┤
│  工具栏 (导入/保存/标注/训练/预测)   │
├──────────────────┬──────────────────┤
│                  │                  │
│  图片编辑画布    │   右侧面板       │
│ (QGraphicsView)  │                  │
│  ┌────────────┐ │  ┌──────────────┐│
│  │            │ │  │ 属性面板     ││
│  │   图片     │ │  ├──────────────┤│
│  │            │ │  │ 工作区信息   ││
│  │            │ │  ├──────────────┤│
│  │            │ │  │ 标注统计     ││
│  │            │ │  └──────────────┘│
│  │            │ │  ┌──────────────┐│
│  │            │ │  │ 日志输出     ││
│  │            │ │  │ QTextEdit    ││
│  │            │ │  │              ││
│  └────────────┘ │  └──────────────┘│
│                  │                  │
├──────────────────┴──────────────────┤
│  状态栏 (进度/消息/坐标)            │
└─────────────────────────────────────┘
```

---

## 3. 核心 Widgets

### 3.1 图片编辑画布 - `ImageCanvas`

**文件**：`src/ui/widgets/image_canvas.py`

```python
from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene
from PyQt5.QtCore import pyqtSignal, QMouseEvent

class ImageCanvas(QGraphicsView):
    """图片编辑和交互的主画布"""
    
    # 信号
    point_clicked = pyqtSignal(tuple)      # 点击坐标
    bbox_drawn = pyqtSignal(tuple)         # 绘制的框
    mask_drawn = pyqtSignal(object)        # 绘制的 mask
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        
        # 状态
        self.current_image = None
        self.current_mask = None
        self.editing_mode = "view"  # view|brush|eraser|polygon|bbox
        self.brush_size = 5
        self.brush_color = (255, 0, 0)
        
    # ============ 图片加载 ============
    def load_image(self, image: np.ndarray) -> None:
        """加载图片"""
        
    def overlay_mask(self, mask: np.ndarray, alpha: float = 0.5) -> None:
        """在图片上叠加 mask"""
        
    # ============ 编辑工具 ============
    def set_editing_mode(self, mode: str) -> None:
        """设置编辑模式"""
        
    def set_brush_size(self, size: int) -> None:
        """设置笔刷大小"""
        
    def set_brush_color(self, color: Tuple) -> None:
        """设置笔刷颜色"""
        
    # ============ 鼠标事件 ============
    def mousePressEvent(self, event: QMouseEvent) -> None:
        """鼠标按下"""
        
    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """鼠标移动（笔刷绘制）"""
        
    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """鼠标释放"""
        
    def wheelEvent(self, event) -> None:
        """鼠标滚轮（缩放）"""
        
    # ============ 工具方法 ============
    def zoom_in(self) -> None:
        """放大"""
        
    def zoom_out(self) -> None:
        """缩小"""
        
    def fit_in_view(self) -> None:
        """适应视图"""
        
    def save_drawing(self, path: str) -> None:
        """保存绘制结果"""
```

**关键实现**：
- 使用 `QGraphicsView + QGraphicsScene` 高效渲染
- 支持缩放、平移、拖拽
- 实时鼠标跟踪用于笔刷效果
- Mask 叠加显示

### 3.2 文件浏览器 - `FileBrowser`

**文件**：`src/ui/widgets/file_browser.py`

```python
from PyQt5.QtWidgets import QWidget, QListWidget, QListWidgetItem

class FileBrowser(QWidget):
    """文件和文件夹浏览器"""
    
    # 信号
    file_selected = pyqtSignal(str)        # 文件被选择
    file_double_clicked = pyqtSignal(str)  # 文件被双击
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.file_list_widget = QListWidget()
        self.current_folder = None
        self.file_list = []
        
    def load_folder(self, folder_path: str) -> None:
        """加载文件夹"""
        
    def get_selected_file(self) -> str:
        """获取选中的文件"""
        
    def refresh(self) -> None:
        """刷新文件列表"""
```

### 3.3 标注工具栏 - `AnnotationToolbar`

**文件**：`src/ui/widgets/annotation_toolbar.py`

```python
from PyQt5.QtWidgets import QToolBar, QPushButton, QSpinBox

class AnnotationToolbar(QToolBar):
    """标注工具栏"""
    
    # 信号
    mode_changed = pyqtSignal(str)         # 模式改变
    brush_size_changed = pyqtSignal(int)   # 笔刷大小改变
    undo_requested = pyqtSignal()
    redo_requested = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__("Annotation Tools", parent)
        self.setup_buttons()
        self.setup_spinboxes()
        
    def setup_buttons(self) -> None:
        """创建工具按钮"""
        # 笔刷、橡皮、多边形、框选、SAM、撤销、重做
        
    def setup_spinboxes(self) -> None:
        """创建参数调整框"""
        # 笔刷大小、透明度
```

### 3.4 日志查看器 - `LogViewer`

**文件**：`src/ui/widgets/log_viewer.py`

```python
from PyQt5.QtWidgets import QTextEdit

class LogViewer(QTextEdit):
    """日志查看器"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        
    def log(self, message: str, level: str = "info") -> None:
        """输出日志"""
        # 支持 info/warning/error 不同颜色
```

---

## 4. 对话框 - `src/ui/dialogs/`

### 4.1 导入对话框 - `ImportDialog`

```python
class ImportDialog(QDialog):
    """文件导入对话框"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
    def get_import_params(self) -> Dict:
        """获取导入参数"""
        return {
            "source": "folder|video|images",
            "path": "...",
            "format": "jpg|png|..."
        }
```

### 4.2 训练配置对话框 - `TrainConfigDialog`

```python
class TrainConfigDialog(QDialog):
    """模型训练配置对话框"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self) -> None:
        """创建训练参数配置界面"""
        # 模型选择、学习率、batch size、epoch、...
        
    def get_config(self) -> Dict:
        """获取配置"""
```

### 4.3 导出对话框 - `ExportDialog`

```python
class ExportDialog(QDialog):
    """标注数据导出对话框"""
    
    def get_export_params(self) -> Dict:
        """获取导出参数"""
        return {
            "format": "coco|yolo|voc|...",
            "output_dir": "...",
            "include_masks": True,
            "include_bboxes": True
        }
```

---

## 5. 菜单栏结构

### 5.1 File 菜单

```
File
├── Open Image (Ctrl+O)
├── Open Folder (Ctrl+Shift+O)
├── Open Video
├── ─────────────
├── Import Dataset
├── ─────────────
├── Save Annotation (Ctrl+S)
├── Save All As...
├── ─────────────
├── Export...
│   ├── Export COCO JSON
│   ├── Export YOLO TXT
│   └── Export to Segmentation Models
├── ─────────────
├── Recent Files
└── Exit (Ctrl+Q)
```

### 5.2 Edit 菜单

```
Edit
├── Undo (Ctrl+Z)
├── Redo (Ctrl+Y)
├── ─────────────
├── Clear Current Mask
├── Clear All Masks
├── ─────────────
└── Preferences (Ctrl+,)
```

### 5.3 View 菜单

```
View
├── Zoom In (Ctrl++)
├── Zoom Out (Ctrl+-)
├── Fit in View (Ctrl+0)
├── ─────────────
├── Show Mask (Toggle)
├── Show Bbox (Toggle)
├── Show Polygon (Toggle)
├── ─────────────
└── Toggle Right Panel (F2)
```

### 5.4 Tools 菜单

```
Tools
├── Annotation
│   ├── Enable SAM (Ctrl+A)
│   ├── Enable Brush (B)
│   ├── Enable Eraser (E)
│   ├── Enable Polygon (P)
│   └── Enable Bbox (C)
├── ─────────────
├── Train Model... (Ctrl+T)
├── Predict... (Ctrl+P)
├── Evaluate Model
├── ─────────────
└── Batch Process...
```

### 5.5 Help 菜单

```
Help
├── User Guide (F1)
├── API Reference (F2)
├── ─────────────
├── Check for Updates
└── About
```

---

## 6. 快捷键

| 快捷键 | 功能 |
|-------|------|
| Ctrl+O | 打开图片 |
| Ctrl+Shift+O | 打开文件夹 |
| Ctrl+S | 保存标注 |
| Ctrl+Z | 撤销 |
| Ctrl+Y | 重做 |
| Ctrl++ | 放大 |
| Ctrl+- | 缩小 |
| Ctrl+0 | 适应视图 |
| B | 笔刷工具 |
| E | 橡皮工具 |
| P | 多边形工具 |
| C | 框选工具 |
| Ctrl+A | SAM 自动标注 |
| Ctrl+T | 训练模型 |
| Ctrl+P | 预测 |
| F1 | 用户指南 |
| F2 | 切换右侧面板 |

---

## 7. 状态栏信息

```
┌──────────────────────────────────┬────────┬──────────────┐
│ 当前操作/消息                     │ 坐标   │ GPU: 0% 内存 │
│ 例：已加载 100 张图片             │ 512,678│ CPU: 45%     │
├──────────────────────────────────┴────────┴──────────────┤
│ 进度条 (训练/预测时显示)                    │ 时间 |
└────────────────────────────────────────────────────────────┘
```

---

## 8. 样式设计 - `src/ui/styles/stylesheet.qss`

```css
/* 主窗口 */
QMainWindow {
    background-color: #f5f5f5;
}

/* 按钮 */
QPushButton {
    background-color: #0066cc;
    color: white;
    border: none;
    border-radius: 4px;
    padding: 6px 12px;
    font-weight: bold;
}

QPushButton:hover {
    background-color: #0052a3;
}

QPushButton:pressed {
    background-color: #003d7a;
}

/* 工具栏 */
QToolBar {
    background-color: #ffffff;
    border-bottom: 1px solid #e0e0e0;
    spacing: 6px;
}

/* 文本编辑 */
QTextEdit, QPlainTextEdit {
    border: 1px solid #cccccc;
    border-radius: 4px;
    padding: 4px;
    font-family: 'Courier New', monospace;
}
```

---

## 9. 多语言支持

**文件**：`src/ui/i18n/`

```
i18n/
├── en_US.ts (英文)
├── zh_CN.ts (中文)
└── ja_JP.ts (日文)
```

---

## 10. 配置与本地存储

```python
# 保存用户偏好
class UserPreferences:
    """用户偏好设置"""
    
    - last_opened_folder
    - recent_files_list
    - window_geometry
    - theme_color
    - brush_size_history
    - default_model
```

---

## 11. 响应式设计

- 自适应窗口大小
- 可调整的分割线（Splitter）
- 自动调整画布大小
- 移动端适配（可选）

---

## 12. 错误提示和警告

```python
# 使用 QMessageBox
QMessageBox.warning(self, "警告", "请先加载图片")
QMessageBox.critical(self, "错误", "模型加载失败：...")
QMessageBox.information(self, "信息", "操作完成")
```

---

## 13. 性能优化

- 使用 QThread 异步操作
- 图片缓存管理
- 渲染优化（减少重绘）
- 内存监控和自动清理

---

## 14. 可访问性 (Accessibility)

- 支持键盘导航
- 高对比度主题
- 屏幕阅读器兼容
- 字体大小可调

---

## 15. 测试

```python
# UI 测试框架：pytest-qt
def test_image_loading(qtbot):
    """测试图片加载"""
    
def test_mask_overlay(qtbot):
    """测试 mask 叠加"""
    
def test_toolbar_functions(qtbot):
    """测试工具栏功能"""
```
