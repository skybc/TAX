"""
工业缺陷分割系统的主入口点。

此应用程序提供完整的缺陷分割工作流程，包括：
- 数据管理和标注
- SAM (Segment Anything) 自动标注
- 模型训练 (U-Net, DeepLabV3+, YOLOv11-Seg)
- 推理和可视化
- 报告生成
"""

import sys
from pathlib import Path

# 将项目根目录添加到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from PyQt5.QtWidgets import QApplication

from src.logger import setup_logger, get_logger
from src.ui.main_window import MainWindow
from src.utils.file_utils import load_yaml, ensure_dir


def main():
    """应用程序主入口点。"""
    # 设置日志
    logger = setup_logger(
        name="IndustrialDefectSeg",
        level="INFO",
        log_file="logs/app.log",
        console=True
    )
    logger.info("=" * 60)
    logger.info("工业缺陷分割系统正在启动...")
    logger.info("=" * 60)
    
    # 加载配置
    try:
        config_path = project_root / "config" / "config.yaml"
        paths_config_path = project_root / "config" / "paths.yaml"
        hyperparams_config_path = project_root / "config" / "hyperparams.yaml"
        
        config = load_yaml(config_path)
        paths_config = load_yaml(paths_config_path)
        hyperparams = load_yaml(hyperparams_config_path)
        
        logger.info("配置加载成功")
        
        # 确保所需的目录存在
        for path_key, path_value in paths_config['paths'].items():
            # 跳过文件路径（键中包含 'file' 或具有文件扩展名）
            is_file_key = 'file' in path_key.lower()
            is_file_path = any(path_value.lower().endswith(ext) for ext in ['.pth', '.pt', '.txt', '.json', '.yaml', '.yml'])
            
            if not is_file_key and not is_file_path:
                ensure_dir(project_root / path_value)
        
    except Exception as e:
        logger.error(f"加载配置失败: {e}")
        sys.exit(1)
    
    # 创建 Qt 应用程序
    app = QApplication(sys.argv)
    app.setApplicationName(config['app']['name'])
    app.setApplicationVersion(config['app']['version'])
    app.setOrganizationName(config['app']['author'])
    
    # 创建并显示主窗口
    try:
        main_window = MainWindow(config, paths_config, hyperparams)
        main_window.show()
        
        logger.info("主窗口已创建并显示")
        logger.info("应用程序就绪")
        
    except Exception as e:
        logger.error(f"创建主窗口失败: {e}", exc_info=True)
        sys.exit(1)
    
    # 运行应用程序
    exit_code = app.exec_()
    
    logger.info("=" * 60)
    logger.info("应用程序已关闭")
    logger.info("=" * 60)
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
