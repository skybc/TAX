"""Setup script for Industrial Defect Segmentation System."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "readme.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read version from package
version = "1.0.0"

setup(
    name="industrial-defect-seg",
    version=version,
    author="Industrial AI Team",
    author_email="support@industrial-ai.com",
    description="A complete industrial defect segmentation system with SAM integration, model training, and batch inference",
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
        # GUI Framework
        "PyQt5>=5.15.9",
        
        # Deep Learning
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "segmentation-models-pytorch>=0.3.3",
        
        # Computer Vision
        "opencv-python>=4.8.1",
        "Pillow>=10.0.0",
        "scikit-image>=0.21.0",
        
        # Scientific Computing
        "numpy>=1.24.3,<2.0.0",
        "scipy>=1.11.0",
        
        # Data Processing
        "pandas>=2.0.3",
        "albumentations>=1.3.1",
        "pycocotools>=2.0.6",
        
        # Visualization
        "matplotlib>=3.7.2",
        "seaborn>=0.12.2",
        
        # Utilities
        "PyYAML>=6.0.1",
        "tqdm>=4.66.1",
        "colorlog>=6.7.0",
        "openpyxl>=3.1.2",  # Excel report generation
        "reportlab>=4.0.4",  # PDF report generation
        "lxml>=4.9.3",  # VOC format export
    ],
    extras_require={
        "dev": [
            # Testing
            "pytest>=7.4.2",
            "pytest-cov>=4.1.0",
            "pytest-qt>=4.2.0",
            "pytest-xdist>=3.3.1",
            "pytest-timeout>=2.1.0",
            
            # Code Quality
            "black>=23.7.0",
            "flake8>=6.1.0",
            "mypy>=1.5.1",
            "isort>=5.12.0",
            "pylint>=2.17.5",
            
            # Pre-commit Hooks
            "pre-commit>=3.3.3",
            
            # Documentation
            "sphinx>=7.1.2",
            "sphinx-rtd-theme>=1.3.0",
            "pdoc3>=0.10.0",
        ],
        "sam": [
            # SAM (Segment Anything) dependencies
            "segment-anything>=1.0",
        ],
        "gpu": [
            # GPU-specific dependencies
            "torch>=2.1.0+cu118",
            "torchvision>=0.16.0+cu118",
        ],
        "all": [
            # All optional dependencies
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
