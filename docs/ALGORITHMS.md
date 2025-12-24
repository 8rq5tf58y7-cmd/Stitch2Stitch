# Algorithm Options Guide

Stitch2Stitch supports multiple state-of-the-art algorithms for each stage of the stitching pipeline. This guide explains when to use each algorithm.

## Feature Detectors

### LP-SIFT (Local-Peak SIFT) - **Recommended**
- **Best for**: General purpose, large datasets, speed-critical applications
- **Speed**: Fast (optimized SIFT variant)
- **Accuracy**: High
- **Use when**: You need a good balance of speed and accuracy
- **Based on**: "Local-Peak Scale-Invariant Feature Transform" (2024)

### SIFT (Scale-Invariant Feature Transform)
- **Best for**: High accuracy requirements, challenging lighting conditions
- **Speed**: Moderate
- **Accuracy**: Very High
- **Use when**: Image quality is more important than speed
- **Note**: Classic algorithm, well-tested and reliable

### ORB (Oriented FAST and Rotated BRIEF)
- **Best for**: Real-time applications, mobile devices
- **Speed**: Very Fast
- **Accuracy**: Good
- **Use when**: Speed is critical and some accuracy can be sacrificed
- **Note**: Binary descriptors, very efficient

### AKAZE (Accelerated-KAZE)
- **Best for**: Images with noise, medical/scientific imaging
- **Speed**: Moderate
- **Accuracy**: High
- **Use when**: Working with noisy images or need non-linear scale space
- **Note**: Handles noise better than SIFT

## Feature Matchers

### FLANN (Fast Library for Approximate Nearest Neighbors) - **Recommended**
- **Best for**: General purpose, large feature sets
- **Speed**: Fast
- **Accuracy**: High
- **Use when**: Standard matching is sufficient
- **Note**: Efficient approximate matching

### LoFTR (Loosely-Fine Matching with Transformers) - **Deep Learning**
- **Best for**: Challenging scenes, low-texture areas, large parallax
- **Speed**: Moderate (GPU recommended)
- **Accuracy**: Very High
- **Use when**: Traditional methods fail, need dense matching
- **Requirements**: PyTorch, GPU recommended
- **Based on**: "LoFTR: Detector-Free Local Feature Matching with Transformers" (2021)

### SuperGlue - **Deep Learning**
- **Best for**: High accuracy requirements, complex scenes
- **Speed**: Moderate (GPU recommended)
- **Accuracy**: Very High
- **Use when**: Need maximum matching accuracy
- **Requirements**: PyTorch, GPU recommended
- **Based on**: "SuperGlue: Learning Feature Matching with Graph Neural Networks" (2020)

### DISK (Differentiable Inlier Scoring for Keypoints) - **Deep Learning**
- **Best for**: Learned features, modern learned descriptors
- **Speed**: Fast (with learned features)
- **Accuracy**: High
- **Use when**: Using learned feature detectors
- **Requirements**: PyTorch
- **Based on**: "DISK: Learning local features with non-linear descriptors" (2020)

## Blending Methods

### Multiband Blending - **Recommended**
- **Best for**: High-quality panoramas, seamless transitions
- **Speed**: Moderate
- **Quality**: Excellent
- **Use when**: Quality is priority
- **Note**: Uses Laplacian pyramid for smooth blending

### Feather Blending
- **Best for**: Fast processing, good quality
- **Speed**: Fast
- **Quality**: Good
- **Use when**: Need faster processing with good results
- **Note**: Simple distance-based blending

### Linear Blending
- **Best for**: Maximum speed
- **Speed**: Very Fast
- **Quality**: Acceptable
- **Use when**: Speed is critical
- **Note**: Basic averaging, fastest method

### Semantic Blending - **Foreground-Aware**
- **Best for**: Scenes with foreground objects, preserving important subjects
- **Speed**: Moderate (slower with semantic model)
- **Quality**: Excellent (preserves foreground integrity)
- **Use when**: Need to preserve foreground objects (people, vehicles, etc.)
- **Requirements**: Semantic segmentation model (optional, falls back to saliency)
- **Based on**: "SemanticStitch: Foreground-Aware Seam Carving for Image Stitching" (2024)

### PixelStitch Blending - **Structure-Preserving**
- **Best for**: Large parallax scenes, structure preservation
- **Speed**: Moderate
- **Quality**: Excellent (preserves structure)
- **Use when**: Need to preserve geometric structure, handle parallax
- **Based on**: "PixelStitch: Structure-Preserving Pixel-Wise Bidirectional Warps" (2025)

## Recommended Combinations

### High Quality (Slow)
- Detector: SIFT
- Matcher: SuperGlue
- Blender: Semantic or PixelStitch

### Balanced (Recommended)
- Detector: LP-SIFT
- Matcher: FLANN
- Blender: Multiband

### Fast Processing
- Detector: ORB
- Matcher: FLANN
- Blender: Feather

### Challenging Scenes (Large Parallax)
- Detector: LP-SIFT or SIFT
- Matcher: LoFTR
- Blender: PixelStitch

### Foreground Preservation
- Detector: LP-SIFT
- Matcher: FLANN or SuperGlue
- Blender: Semantic

## GPU Acceleration

Deep learning matchers (LoFTR, SuperGlue) benefit significantly from GPU acceleration:
- **CPU**: 5-10x slower
- **GPU**: Near real-time for small batches

Enable GPU acceleration in settings if you have CUDA-capable GPU.

## Algorithm Selection Tips

1. **Start with defaults**: LP-SIFT + FLANN + Multiband works well for most cases
2. **If matching fails**: Try LoFTR or SuperGlue (requires GPU for best performance)
3. **If speed is critical**: Use ORB + FLANN + Feather
4. **If quality is critical**: Use SIFT + SuperGlue + Semantic/PixelStitch
5. **For large parallax**: Use LoFTR + PixelStitch
6. **For foreground objects**: Use Semantic blending

## Performance Comparison

Approximate relative speeds (normalized to FLANN = 1.0):

| Matcher | CPU Speed | GPU Speed |
|---------|-----------|-----------|
| FLANN   | 1.0       | 1.0       |
| LoFTR   | 0.1       | 5.0       |
| SuperGlue| 0.1      | 4.0       |
| DISK    | 0.5       | 2.0       |

*Note: Actual speeds depend on image size, number of features, and hardware*

