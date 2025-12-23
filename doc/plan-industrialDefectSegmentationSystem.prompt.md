# Plan: 工业缺陷分割系统详细实现步骤

该项目需要构建一套完整的工业缺陷分割系统，包括数据管理、SAM 自动标注、人工修正、模型训练和预测推理等七个核心模块。建议按照分层渐进式开发方法，从基础的数据管理开始，逐步集成 SAM、训练管道和可视化功能。

## Steps

1. **创建项目架构和基础框架** — 建立 PyQt 应用主窗口、模块目录结构、配置文件（paths、hyperparams）、依赖环境配置

2. **实现数据管理模块** — 开发图片/视频导入（QFileDialog）、QGraphicsView 图片展示、缩放滚动、文件列表管理（QListWidget）、标准数据存储结构

3. **实现人工标注基础模块** — 添加笔刷/橡皮工具、多边形绘制、撤销/重做（QStack）、Mask 保存为 PNG 和 numpy array

4. **集成 SAM 自动标注** — 加载 SAM 模型、实现交互点/框提示输入、异步推理线程管理、mask 结果叠加显示

5. **实现标注数据管理和导出** — 支持 YOLO/COCO 格式导出、批量保存整个任务、建立训练数据集组织结构

6. **实现模型训练模块** — GUI 参数配置、数据增强（旋转、翻转、亮度）、多模型支持（U-Net、YOLOv11-Seg、DeepLabV3+）、异步训练与日志显示

7. **实现预测推理模块** — 单图/批量预测、mask 和 bbox 显示、后处理（腐蚀膨胀去噪）、人工微调功能

8. **实现可视化与报告模块** — 缺陷统计分析、Matplotlib/PyQtGraph 图表、Excel/PDF 报告导出

## Further Considerations

1. **技术栈选择确认** — Python 版本、PyQt5/PyQt6、PyTorch/TensorFlow for SAM、数据增强库（albumentations/torchvision）、报表库（openpyxl/reportlab）是否满足需求？

2. **部署和性能** — 是否需要支持 GPU 加速、模型量化、实时推理？单机应用还是需要后端服务？
