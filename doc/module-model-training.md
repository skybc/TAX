# 模型训练模块技术文档

## 1. 模块概述

**模块名称**：模型训练模块  
**文件位置**：`src/core/model_trainer.py` + `src/threads/training_thread.py` + `scripts/train.py`  
**职责**：支持多种分割模型的训练管道、超参数优化、进度监控

---

## 2. 支持的模型

### 2.1 U-Net 系列

- **经典 U-Net**：编码器-解码器架构
- **U-Net++**：嵌套跳接连接
- **3+ U-Net**：全尺度聚合

```python
# 使用 segmentation_models_pytorch
import segmentation_models_pytorch as smp

model = smp.Unet(
    encoder_name="resnet50",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
    activation="sigmoid"
)
```

### 2.2 DeepLabV3+

```python
model = smp.DeepLabV3Plus(
    encoder_name="resnet50",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
    activation="sigmoid"
)
```

### 2.3 YOLOv11-Seg（检测+分割）

```python
from ultralytics import YOLO

model = YOLO("yolov11-seg.pt")  # 预训练权重
```

### 2.4 其他模型

- FPN (Feature Pyramid Network)
- PSPNet
- LinkNet
- PAN (Pyramid Attention Network)

---

## 3. 核心类设计

### 3.1 `ModelTrainer` 主类

```python
class ModelTrainer:
    """模型训练主类"""
    
    def __init__(self, config: Dict):
        """
        Args:
            config: 训练配置字典
        """
        self.config = config
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.train_history = {
            "loss": [],
            "iou": [],
            "val_loss": [],
            "val_iou": []
        }
        self.best_metric = float('inf')
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
    # ============ 模型构建 ============
    def build_model(self, model_name: str, **kwargs) -> nn.Module:
        """
        构建模型
        Args:
            model_name: "unet" | "deeplabv3+" | "yolov11-seg" | ...
        """
        
    def build_model_from_checkpoint(self, checkpoint_path: str) -> None:
        """从检查点加载模型"""
        
    # ============ 优化器和调度器 ============
    def setup_optimizer(self, 
                       optimizer_name: str = "adam",
                       learning_rate: float = 0.001,
                       weight_decay: float = 1e-5) -> None:
        """设置优化器"""
        
    def setup_scheduler(self, scheduler_name: str = "cosine") -> None:
        """设置学习率调度器"""
        
    # ============ 训练循环 ============
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        训练一个 epoch
        Returns:
            {"loss": ..., "iou": ..., "dice": ...}
        """
        
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """验证"""
        
    def train(self, 
              train_loader: DataLoader,
              val_loader: DataLoader,
              epochs: int,
              callback = None) -> Dict:
        """
        完整训练流程
        Args:
            callback: 每个 epoch 完成后的回调函数
        """
        
    # ============ 模型保存与加载 ============
    def save_checkpoint(self, save_path: str, is_best: bool = False) -> None:
        """保存检查点"""
        
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """加载检查点"""
        
    def save_model(self, save_path: str, format: str = "pth") -> None:
        """
        保存模型权重
        Args:
            format: "pth" | "onnx" | "torchscript"
        """
        
    # ============ 工具方法 ============
    def get_training_history(self) -> Dict:
        """获取训练历史"""
        
    def reset_training_state(self) -> None:
        """重置训练状态"""
```

### 3.2 `TrainingThread` 线程类

```python
from PyQt5.QtCore import QThread, pyqtSignal

class TrainingThread(QThread):
    """异步训练线程"""
    
    # 信号
    epoch_finished = pyqtSignal(dict)      # epoch 完成
    training_finished = pyqtSignal(dict)   # 训练完成
    training_error = pyqtSignal(str)       # 错误
    progress_updated = pyqtSignal(int)     # 进度
    
    def __init__(self, trainer: ModelTrainer):
        super().__init__()
        self.trainer = trainer
        self.stop_flag = False
        
    def run(self) -> None:
        """执行训练"""
        
    def stop(self) -> None:
        """停止训练"""
```

---

## 4. 数据加载和预处理

### 4.1 数据集类

```python
from torch.utils.data import Dataset

class SegmentationDataset(Dataset):
    """分割数据集"""
    
    def __init__(self, 
                 image_dir: str,
                 mask_dir: str,
                 transform = None,
                 augmentation = None):
        """
        Args:
            image_dir: 图片目录
            mask_dir: mask 目录
            transform: 数据预处理
            augmentation: 数据增强
        """
        self.image_files = sorted(os.listdir(image_dir))
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.augmentation = augmentation
        
    def __len__(self):
        return len(self.image_files)
        
    def __getitem__(self, idx):
        image = cv2.imread(os.path.join(self.image_dir, self.image_files[idx]))
        mask = cv2.imread(os.path.join(self.mask_dir, self.image_files[idx]), 0)
        
        if self.augmentation:
            augmented = self.augmentation(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']
            
        if self.transform:
            image = self.transform(image)
            
        mask = torch.from_numpy(mask / 255.0).float()
        return image, mask
```

### 4.2 数据增强

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

augmentation = A.Compose([
    A.Rotate(limit=45, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.GaussNoise(p=0.2),
    A.RandomBrightnessContrast(p=0.3),
    A.Resize(height=256, width=256),
    A.Normalize(mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
], bbox_params=A.BboxParams(format='pascal_voc'))
```

---

## 5. 损失函数

### 5.1 支持的损失函数

```python
# Dice Loss
class DiceLoss(nn.Module):
    def forward(self, pred, target):
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        return 1 - (2 * intersection + 1e-6) / (union + 1e-6)

# IoU Loss
class IoULoss(nn.Module):
    def forward(self, pred, target):
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        return 1 - (intersection + 1e-6) / (union + 1e-6)

# Combined Loss
class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()
        
    def forward(self, pred, target):
        return 0.5 * self.bce(pred, target) + 0.5 * self.dice(pred, target)
```

---

## 6. 评估指标

### 6.1 实现的指标

```python
def dice_coefficient(pred: np.ndarray, target: np.ndarray) -> float:
    """Dice 系数"""
    intersection = (pred * target).sum()
    return 2 * intersection / (pred.sum() + target.sum())

def iou_score(pred: np.ndarray, target: np.ndarray) -> float:
    """IoU (Intersection over Union)"""
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return intersection / union

def mean_iou(predictions: List, targets: List) -> float:
    """平均 IoU"""
    return np.mean([iou_score(p, t) for p, t in zip(predictions, targets)])

def hausdorff_distance(pred: np.ndarray, target: np.ndarray) -> float:
    """Hausdorff 距离"""
    # 轮廓提取
    contour_pred = cv2.findContours(pred, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_target = cv2.findContours(target, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return cv2.matchShapes(contour_pred, contour_target, cv2.CONTOURS_MATCH_I3, 0)
```

---

## 7. 训练工作流

```
加载配置
    ↓
构建数据加载器 (train/val split)
    ↓
初始化模型
    ↓
设置优化器和调度器
    ↓
启动训练线程
    ├─ Epoch 1
    ├─ Epoch 2
    │   ├─ Forward pass
    │   ├─ 计算损失
    │   ├─ Backward pass
    │   └─ 更新权重
    │
    └─ ...Epoch N
        ├─ 验证
        ├─ 保存最优模型
        └─ 返回训练历史
```

---

## 8. 超参数配置

### 8.1 配置文件示例

```yaml
# config/training_config.yaml
training:
  model_name: "unet"
  encoder: "resnet50"
  
  batch_size: 16
  num_workers: 4
  
  optimizer: "adam"
  learning_rate: 0.001
  weight_decay: 1e-5
  momentum: 0.9
  
  scheduler: "cosine"
  epochs: 100
  warmup_epochs: 10
  
  loss_function: "combined"  # bce | dice | iou | combined
  
  validation_split: 0.2
  early_stopping_patience: 20
  
  augmentation:
    enabled: true
    rotation: 45
    flip_h: true
    flip_v: true
    brightness: 0.3
    
  mixed_precision: true  # AMP 混合精度
```

---

## 9. 训练脚本使用

### 9.1 命令行使用

```bash
# 基础训练
python scripts/train.py --config config/training_config.yaml --data_dir data/processed/

# 指定 GPU
python scripts/train.py --config config/training_config.yaml --device cuda:0

# 继续训练
python scripts/train.py --config config/training_config.yaml --resume models/checkpoints/latest.pth

# 参数覆盖
python scripts/train.py --config config/training_config.yaml --batch_size 32 --epochs 200
```

### 9.2 Python API 使用

```python
from src.core.model_trainer import ModelTrainer
from torch.utils.data import DataLoader

# 配置
config = {
    "model_name": "unet",
    "encoder": "resnet50",
    "batch_size": 16,
    "epochs": 100,
    "learning_rate": 0.001,
}

# 初始化训练器
trainer = ModelTrainer(config)
trainer.build_model("unet", encoder="resnet50")
trainer.setup_optimizer("adam", learning_rate=0.001)
trainer.setup_scheduler("cosine")

# 准备数据
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# 训练
history = trainer.train(train_loader, val_loader, epochs=100)

# 保存
trainer.save_model("models/checkpoints/final_model.pth")
```

---

## 10. 监控和可视化

### 10.1 实时监控

```python
# PyQt 集成
class TrainingMonitor(QWidget):
    def setup_monitoring(self):
        self.figure, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 4))
        self.canvas = FigureCanvas(self.figure)
        
    def update_metrics(self, epoch: dict):
        """更新图表"""
        # 绘制损失曲线
        # 绘制 IoU 曲线
```

### 10.2 TensorBoard 支持

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs/experiment1")
writer.add_scalar("Loss/train", loss, epoch)
writer.add_scalar("IoU/val", iou, epoch)
```

---

## 11. 性能优化

| 优化方法 | 效果 |
|---------|------|
| 混合精度 (AMP) | 内存减少 50%，速度提升 20% |
| 分布式训练 (DDP) | N 卡线性加速 |
| 梯度累积 | 有效 batch size 提升 |
| 模型量化 | 推理速度提升 4 倍 |
| TorchScript | 部署时推理速度提升 |

---

## 12. 单元测试用例

```python
def test_model_building():
    """测试模型构建"""
    
def test_data_loading():
    """测试数据加载"""
    
def test_forward_pass():
    """测试前向传播"""
    
def test_training_step():
    """测试训练步骤"""
    
def test_checkpoint_save_load():
    """测试检查点保存和加载"""
    
def test_metrics_calculation():
    """测试指标计算"""
```

---

## 13. 错误处理

- 无效模型名称
- 数据加载失败
- GPU 显存不足
- 数据分割错误
- 权重加载失败

---

## 14. 扩展功能

- **分布式训练**：支持多 GPU/多节点
- **自动超参数调优**：使用 Optuna/Ray Tune
- **模型蒸馏**：将大模型知识迁移到小模型
- **迁移学习**：使用预训练权重
- **早停机制**：防止过拟合
