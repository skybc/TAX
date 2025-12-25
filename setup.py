"""工业缺陷分割系统的安装脚本。"""

from setuptools import setup, find_packages
from pathlib import Path

# 读取 README
readme_file = Path(__file__).parent / "readme.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# 从包中读取版本
version = "1.0.0"

setup(
    name="industrial-defect-seg",
    version=version,
    author="工业 AI 团队",
    author_email="support@industrial-ai.com",
    description="一个完整的工业缺陷分割系统，集成了 SAM、模型训练和批量推理功能",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/industrial-defect-seg",
    project_urls={
        "Bug Tracker": "https://github.com/your-org/industrial-defect-seg/issues",
        "Documentation": "https://github.com/your-org/industrial-defect-seg/tree/main/doc",
        "Source Code": "https://github.com/your-org/industrial-defect-seg",
    },
    packages=find_packages(exclude=["tests", "tests.*", "doc", "doc.*"]),
    include_package_data=True,
    package_data={
        "": ["config/*.yaml", "config/*.yml"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Manufacturing",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Environment :: X11 Applications :: Qt",
        "Framework :: Pytest",
    ],
    keywords="defect-detection, segmentation, SAM, industrial-inspection, computer-vision, deep-learning, pytorch, pyqt5",
    python_requires=">=3.8",
    install_requires=[
        # GUI 框架
        "PyQt5>=5.15.9",
        
        # 深度学习
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "segmentation-models-pytorch>=0.3.3",
        
        # 计算机视觉
        "opencv-python>=4.8.1",
        "Pillow>=10.0.0",
        "scikit-image>=0.21.0",
        
        # 科学计算
        "numpy>=1.24.3,<2.0.0",
        "scipy>=1.11.0",
        
        # 数据处理
        "pandas>=2.0.3",
        "albumentations>=1.3.1",
        "pycocotools>=2.0.6",
        
        # 可视化
        "matplotlib>=3.7.2",
        "seaborn>=0.12.2",
        
        # 实用程序
        "PyYAML>=6.0.1",
        "tqdm>=4.66.1",
        "colorlog>=6.7.0",
        "openpyxl>=3.1.2",  # Excel 报告生成
        "reportlab>=4.0.4",  # PDF 报告生成
        "lxml>=4.9.3",  # VOC 格式导出
    ],
    extras_require={
        "dev": [
            # 测试
            "pytest>=7.4.2",
            "pytest-cov>=4.1.0",
            "pytest-qt>=4.2.0",
            "pytest-xdist>=3.3.1",
            "pytest-timeout>=2.1.0",
            
            # 代码质量
            "black>=23.7.0",
            "flake8>=6.1.0",
            "mypy>=1.5.1",
            "isort>=5.12.0",
            "pylint>=2.17.5",
            
            # Pre-commit Hooks
            "pre-commit>=3.3.3",
            
            # 文档
            "sphinx>=7.1.2",
            "sphinx-rtd-theme>=1.3.0",
            "pdoc3>=0.10.0",
        ],
        "sam": [
            # SAM (Segment Anything) 依赖项
            "segment-anything>=1.0",
        ],
        "gpu": [
            # GPU 特定依赖项
            "torch>=2.1.0+cu118",
            "torchvision>=0.16.0+cu118",
        ],
        "all": [
            # 所有可选依赖项
            "segment-anything>=1.0",
            "pytest>=7.4.2",
            "black>=23.7.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "defect-seg=src.main:main",
            "defect-seg-train=src.scripts.train:main",
            "defect-seg-predict=src.scripts.predict:main",
        ],
    },
    zip_safe=False,
)
