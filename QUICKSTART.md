# Quick Start Guide

Get started with Stitch2Stitch in minutes!

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd tbm

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## First Stitch

### Using GUI (Recommended)

1. Launch the application:
   ```bash
   python src/main.py
   ```

2. Click "Add Images..." and select your overlapping images

3. Set output path by clicking "Browse..." in the Output section

4. Click "Stitch Images"

5. Wait for processing to complete

6. Click "Save Result" to export your panorama

### Using Command Line

```bash
python src/main.py --cli \
    --input /path/to/your/images \
    --output panorama.tif \
    --quality-threshold 0.7
```

## Example: Grid Alignment Preview

Before stitching, preview how images will be arranged:

1. Add images
2. Set output path
3. Click "Create Grid Alignment"
4. Review the grid layout
5. If satisfied, click "Stitch Images"

## Key Features to Try

- **Quality Filtering**: Adjust quality threshold to automatically filter blurry images
- **GPU Acceleration**: Enable in settings if you have CUDA GPU (much faster!)
- **Transparency Support**: Works with PNG images that have alpha channels
- **Large Panoramas**: Handles 1400+ images efficiently

## Next Steps

- Read [USAGE.md](docs/USAGE.md) for detailed usage instructions
- Check [ARCHITECTURE.md](docs/ARCHITECTURE.md) to understand the system
- See [README.md](README.md) for full feature list

## Getting Help

- Check the Logs tab in GUI for detailed processing information
- Review error messages for troubleshooting
- Open an issue on GitHub for bugs or questions

