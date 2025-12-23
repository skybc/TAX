"""Setup script for Industrial Defect Segmentation System."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "readme.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="industrial-defect-segmentation",
    version="1.0.0-dev",
    author="Industrial AI Team",
    description="A complete PyQt5-based industrial defect segmentation system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/skybc/TAX",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        "PyQt5>=5.15.9",
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "opencv-python>=4.8.1",
        "numpy>=1.24.3",
        "pandas>=2.0.3",
        "matplotlib>=3.7.2",
        "albumentations>=1.3.1",
        "pycocotools>=2.0.6",
        "PyYAML>=6.0.1",
        "tqdm>=4.66.1",
        "colorlog>=6.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.2",
            "pytest-cov>=4.1.0",
            "pytest-qt>=4.2.0",
            "black>=23.7.0",
            "flake8>=6.1.0",
            "mypy>=1.5.1",
            "isort>=5.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "defect-seg=src.main:main",
        ],
    },
)
