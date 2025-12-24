# Architecture Overview

## System Design

Stitch2Stitch is designed with a modular architecture that separates concerns and allows for easy extension and optimization.

## Core Components

### 1. Image Stitcher (`src/core/stitcher.py`)
Main orchestrator that coordinates the stitching pipeline:
- Image loading and quality assessment
- Feature detection
- Feature matching
- Image alignment
- Image blending

### 2. Feature Detection (`src/ml/feature_detector.py`)
Implements modern feature detection algorithms:
- **LP-SIFT**: Local-Peak SIFT for fast, efficient feature detection
- **ORB**: Alternative detector for different use cases
- **AKAZE**: Another alternative detector

### 3. Feature Matching (`src/ml/matcher.py`)
Advanced matching with:
- FLANN-based matcher for speed
- Lowe's ratio test for quality filtering
- Confidence scoring

### 4. Image Quality Assessment (`src/quality/assessor.py`)
Multi-metric quality assessment:
- Laplacian variance (blur detection)
- Gradient magnitude
- Contrast scoring
- Noise estimation

### 5. Image Alignment (`src/core/alignment.py`)
Handles geometric alignment:
- Homography estimation using RANSAC
- Reference image selection
- Transform composition
- Support for transparency masks

### 6. Image Blending (`src/core/blender.py`)
Multiple blending algorithms:
- Multi-band blending (default)
- Feather blending
- Linear blending

### 7. Memory Management (`src/utils/memory_manager.py`)
Optimized for large panoramas:
- Memory usage monitoring
- Chunk size recommendations
- Image memory estimation

### 8. Grid Visualization (`src/core/grid_visualizer.py`)
2D grid alignment without merging:
- Spatial arrangement
- Overlap visualization
- Export capabilities

## Data Flow

```
Input Images
    ↓
Quality Assessment → Filter blurry images
    ↓
Feature Detection → Extract keypoints and descriptors
    ↓
Feature Matching → Find correspondences
    ↓
Alignment → Calculate transforms
    ↓
Blending → Create final panorama
    ↓
Output
```

## Performance Optimizations

1. **Chunked Processing**: Large datasets processed in chunks
2. **Lazy Evaluation**: Transforms calculated on-demand
3. **GPU Acceleration**: Optional GPU support for compute-intensive operations
4. **Memory Mapping**: Large outputs use memory-mapped files
5. **Parallel Processing**: Feature detection and matching parallelized

## Extension Points

- **Custom Detectors**: Implement new feature detectors
- **Custom Blenders**: Add new blending algorithms
- **Custom Quality Metrics**: Extend quality assessment
- **Custom Alignment**: Implement alternative alignment strategies

