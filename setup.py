"""Setup script for Stitch2Stitch"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="stitch2stitch",
    version="0.1.0",
    description="Advanced panoramic image stitching application",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Stitch2Stitch Contributors",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "opencv-python>=4.8.0",
        "opencv-contrib-python>=4.8.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "Pillow>=10.0.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "scikit-image>=0.21.0",
        "scikit-learn>=1.3.0",
        "PyQt6>=6.5.0",
        "PyQt6-Qt6>=6.5.0",
        "matplotlib>=3.7.0",
        "numba>=0.57.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
        "tifffile>=2023.4.0",
        "psutil>=5.9.0",
    ],
    entry_points={
        "console_scripts": [
            "stitch2stitch=src.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Multimedia :: Graphics",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
)

