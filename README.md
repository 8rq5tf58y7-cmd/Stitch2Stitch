# Stitch2Stitch - Advanced Panoramic Image Stitching Application

A modern, high-performance image stitching application that surpasses AutoPano Giga and PTGui with cutting-edge AI/ML algorithms, optimized for large-scale panoramas (1400+ images) and exceptional image quality.

## Key Features

### 🚀 Advanced Algorithms

**Feature Detectors:**
- **LP-SIFT (Local-Peak SIFT)**: Fast, efficient feature detection optimized for large datasets
- **SIFT**: Classic high-accuracy detector
- **ORB**: Fast binary feature detector
- **AKAZE**: Noise-resistant detector with non-linear scale space

**Feature Matchers:**
- **FLANN**: Fast approximate nearest neighbor matching
- **LoFTR**: Deep learning transformer-based matcher (2021)
- **SuperGlue**: Graph neural network-based matcher (2020)
- **DISK**: Learned feature matching

**Blending Methods:**
- **Multiband Blending**: High-quality seamless blending
- **Semantic Blending**: Foreground-aware blending (SemanticStitch, 2024)
- **PixelStitch Blending**: Structure-preserving pixel-wise warps (2025)
- **Feather/Linear**: Fast blending options

### 🎯 Image Quality Focus
- Automatic blur detection and sharpness scoring
- Intelligent image selection (uses only best non-blurry images)
- Color and exposure consistency correction
- High-resolution output preservation

### 📐 Flat Image Specialization
- Optimized for flat images without lens distortion
- No focal distance calculations needed
- Handles overlapping images with precision
- Support for transparency masks and irregular shapes

### 🔍 Visualization & Preview
- 2D grid alignment mode (visualize without merging)
- Real-time preview of stitching results
- Interactive adjustment tools
- Overlap visualization

### ⚡ Performance & Scale
- GPU acceleration support (CUDA/OpenCL)
- Memory-efficient processing for 1400+ images
- Parallel processing pipelines
- Chunked processing for large panoramas
- Optimized data structures (SandFall-inspired compression)

### 🎨 Modern GUI
- Intuitive drag-and-drop interface
- **Real-time progress bars** with percentage tracking
- **Stop/Cancel button** for long-running operations
- **Status messages** showing current operation
- Advanced parameter controls
- Batch processing support
- Export to multiple formats
- **Cross-platform** (Windows & macOS compatible)

## Installation

### Quick Install (Self-Contained)

**Windows:** Double-click `install.bat` or run `install_windows.ps1`  
**macOS:** Run `./install_macos.sh` in Terminal

The installer will automatically:
- Install Python if needed
- Create virtual environment
- Install all dependencies
- Create launcher scripts

**📖 See [README_INSTALL.md](README_INSTALL.md) for quick start or [INSTALL.md](INSTALL.md) for detailed instructions.**

### Manual Installation

**Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Standalone Executable (No Installation)

Build a single executable with everything included:
```bash
python build_standalone.py
```
Result: `dist/Stitch2Stitch.exe` (Windows) or `dist/Stitch2Stitch.app` (macOS)

## Quick Start

### GUI Mode (Recommended)
```bash
python src/main.py
```

### Command Line Mode
```bash
python src/main.py --cli --input /path/to/images --output panorama.tif
```

**📖 See [USAGE_INSTRUCTIONS.md](USAGE_INSTRUCTIONS.md) for complete usage guide.**

## Architecture

```
tbm/
├── src/
│   ├── core/              # Core stitching algorithms
│   ├── ml/                # ML models and AI components
│   ├── quality/           # Image quality assessment
│   ├── gui/               # GUI components
│   ├── utils/             # Utilities and helpers
│   └── main.py            # Entry point
├── models/                # Pre-trained ML models
├── tests/                 # Unit tests
├── docs/                  # Documentation
└── requirements.txt       # Python dependencies
```

## Usage Examples

### Basic Stitching
1. Launch the application
2. Drag and drop images into the workspace
3. Select algorithms (or use defaults)
4. Click "Stitch Images"
5. Review preview and adjust if needed
6. Export final panorama

### Grid Visualization Mode
1. Load images
2. Click "Create Grid Alignment"
3. Images are arranged in 2D grid based on overlap
4. Review layout
5. Export grid or proceed to stitching

### Algorithm Selection
Choose from multiple state-of-the-art algorithms:
- **Feature Detector**: LP-SIFT (default), SIFT, ORB, AKAZE
- **Feature Matcher**: FLANN (default), LoFTR, SuperGlue, DISK
- **Blending**: Multiband (default), Semantic, PixelStitch, Feather, Linear

See [ALGORITHMS.md](docs/ALGORITHMS.md) for detailed algorithm guide.

### Advanced Options
- Enable GPU acceleration for deep learning matchers
- Adjust quality thresholds for blur detection
- Configure memory limits for large datasets
- Select from multiple blending algorithms
- Set output resolution and format

## Technical Details

### Algorithms Implemented
- **Feature Detection**: LP-SIFT, ORB, AKAZE
- **Matching**: FLANN-based matcher with RANSAC
- **Alignment**: DSFN-inspired depth-aware alignment
- **Blending**: Multi-band blending, GAN-based blending
- **Quality**: Laplacian variance, gradient magnitude, ML-based scoring

### Performance Optimizations
- Chunked image loading
- Lazy evaluation of transforms
- GPU-accelerated feature detection
- Parallel matching pipelines
- Memory-mapped file I/O for large outputs

## Requirements

- Python 3.9+
- CUDA-capable GPU (optional, for acceleration)
- 16GB+ RAM recommended for large panoramas
- OpenCV 4.8+
- PyTorch 2.0+ (for ML features)
- PyQt6 (for GUI)

## License

MIT License

## Contributing

Contributions welcome! Please see CONTRIBUTING.md for guidelines.
