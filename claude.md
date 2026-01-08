# Stitch2Stitch

Advanced panoramic image stitching application designed to handle large-scale panoramas (1400+ images) with AI/ML algorithms.

## Project Overview

- **Language**: Python 3.9+
- **GUI Framework**: PyQt6
- **ML Framework**: PyTorch
- **Computer Vision**: OpenCV

## Architecture

```
src/
├── core/           # Core stitching algorithms
│   ├── stitcher.py         # Main stitching orchestration
│   ├── blender.py          # Image blending (multiband, feather, etc.)
│   ├── alignment.py        # Image alignment/registration
│   ├── post_processing.py  # Post-processing effects
│   └── semantic_blender.py # Foreground-aware blending
├── ml/             # Machine learning components
│   ├── feature_detector.py # SIFT, ORB, AKAZE detection
│   ├── matcher.py          # Feature matching (FLANN, etc.)
│   ├── advanced_matchers.py # LoFTR, SuperGlue matchers
│   └── superpoint.py       # SuperPoint feature detection
├── gui/            # User interface
│   └── main_window.py      # Main PyQt6 application window
├── quality/        # Image quality assessment
│   └── assessor.py         # Blur detection, sharpness scoring
├── utils/          # Utilities
│   ├── memory_manager.py   # Memory optimization for large images
│   ├── logger.py           # Logging configuration
│   └── platform_utils.py   # Cross-platform helpers
├── external/       # External pipeline integrations
│   ├── pipelines.py        # COLMAP/HLOC pipeline wrappers
│   ├── wsl_colmap_bridge.py # WSL GPU acceleration for Windows
│   └── colmap_cache_manager.py # Caching for COLMAP results
└── main.py         # Application entry point
```

## Key Commands

```bash
# Run GUI application
python src/main.py

# Run in CLI mode
python src/main.py --cli --input /path/to/images --output panorama.tif

# Build standalone executable
python build_standalone.py
```

## Development Notes

### Running Tests
```bash
python test_run.py
```

### Platform Considerations
- **Windows**: Uses WSL bridge for GPU-accelerated COLMAP
- **macOS/Linux**: Native COLMAP support

### Key Algorithms
- **Feature Detectors**: LP-SIFT (default), SIFT, ORB, AKAZE
- **Feature Matchers**: FLANN (default), LoFTR, SuperGlue
- **Blending**: Multiband (default), Feather, AutoStitch, Linear

### External Dependencies
- COLMAP: Structure-from-Motion pipeline (optional, for advanced stitching)
- HLOC: Hierarchical localization for feature matching (optional)

## Code Style

- Follow PEP 8
- Use type hints where practical
- Keep functions focused and testable
- Document complex algorithms with inline comments
- Use logging instead of print statements

## Common Tasks

### Adding a New Blending Method
1. Add method to `src/core/blender.py` in the `ImageBlender` class
2. Register in `BLEND_METHODS` dict
3. Add UI option in `src/gui/main_window.py`

### Adding a New Feature Detector
1. Implement in `src/ml/feature_detector.py`
2. Add to detector selection in `src/gui/main_window.py`

### Working with COLMAP Integration
- Native pipeline: `src/external/pipelines.py`
- WSL bridge (Windows GPU): `src/external/wsl_colmap_bridge.py`
- Cache management: `src/external/colmap_cache_manager.py`
