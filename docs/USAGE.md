# Usage Guide

## GUI Mode

Launch the application:
```bash
python src/main.py
```

### Basic Workflow

1. **Add Images**: Click "Add Images..." and select your images
2. **Configure Settings**: Adjust quality threshold, enable GPU if available
3. **Preview Grid** (Optional): Click "Create Grid Alignment" to see how images arrange
4. **Stitch**: Click "Stitch Images" to create the panorama
5. **Save**: Click "Save Result" to export

### Settings Explained

- **Quality Threshold**: Minimum quality score (0.0-1.0) for image inclusion
  - Higher values = stricter quality requirements
  - Recommended: 0.6-0.8
  
- **GPU Acceleration**: Enable if you have CUDA-capable GPU
  - Significantly faster for large datasets and deep learning matchers
  - Requires CUDA and PyTorch with CUDA support
  - Essential for LoFTR and SuperGlue matchers

- **Feature Detector**: Algorithm for detecting image features
  - **LP-SIFT (Recommended)**: Fast and accurate, best balance
  - **SIFT**: Highest accuracy, slower
  - **ORB**: Fastest, good for real-time
  - **AKAZE**: Best for noisy images

- **Feature Matcher**: Algorithm for matching features between images
  - **FLANN (Recommended)**: Fast and reliable
  - **LoFTR**: Deep learning, best for challenging scenes (needs GPU)
  - **SuperGlue**: Deep learning, highest accuracy (needs GPU)
  - **DISK**: Learned features, good balance

- **Blending Method**: Algorithm for blending aligned images
  - **Multiband (Recommended)**: Best quality, seamless transitions
  - **Feather**: Good quality, faster
  - **Linear**: Fastest, basic blending
  - **Semantic**: Preserves foreground objects (people, vehicles, etc.)
  - **PixelStitch**: Structure-preserving, best for large parallax

- **Max Images**: Limit number of images processed (0 = unlimited)

See [ALGORITHMS.md](ALGORITHMS.md) for detailed algorithm guide and recommendations.

## Command-Line Mode

### Basic Stitching
```bash
python src/main.py --cli --input /path/to/images --output panorama.tif
```

### With Options
```bash
python src/main.py --cli \
    --input /path/to/images \
    --output panorama.tif \
    --quality-threshold 0.7 \
    --gpu \
    --detector lp_sift \
    --matcher flann \
    --blender multiband
```

### Using Advanced Algorithms
```bash
# Use deep learning matcher (requires GPU)
python src/main.py --cli \
    --input /path/to/images \
    --output panorama.tif \
    --matcher loftr \
    --blender semantic \
    --gpu

# Use structure-preserving blending
python src/main.py --cli \
    --input /path/to/images \
    --output panorama.tif \
    --blender pixelstitch
```

### Grid Alignment Only
```bash
python src/main.py --cli \
    --input /path/to/images \
    --output grid.tif \
    --grid-only
```

## Tips for Best Results

### Image Preparation
- Use images with good overlap (30-50% recommended)
- Ensure images are sharp and in focus
- Avoid extreme exposure differences
- Images should be flat (no lens distortion)

### Large Panoramas (1400+ images)
- Enable GPU acceleration if available
- Use quality threshold to filter blurry images
- Process in batches if memory is limited
- Use TIFF format for output (better for large files)

### Transparency Masks
- Supported formats: PNG with alpha channel
- Images with transparency are handled automatically
- Irregular shapes are preserved

### Performance
- GPU acceleration: 5-10x faster for feature detection
- Memory: ~100MB per image (rough estimate)
- Processing time: Varies with image count and size
  - 100 images: ~5-10 minutes (CPU)
  - 1000 images: ~1-2 hours (CPU), ~10-20 minutes (GPU)

## Troubleshooting

### "No images passed quality assessment"
- Lower the quality threshold
- Check that images are not too blurry
- Ensure images are readable

### "No feature matches found"
- Images may not have enough overlap
- Try different feature detector
- Check image quality

### Out of Memory
- Reduce max_images setting
- Process in smaller batches
- Close other applications
- Use lower resolution images

### GPU Not Working
- Ensure CUDA is installed
- Check PyTorch CUDA support: `python -c "import torch; print(torch.cuda.is_available())"`
- Install CUDA-enabled PyTorch if needed

