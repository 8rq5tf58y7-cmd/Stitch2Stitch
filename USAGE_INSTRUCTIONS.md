# Stitch2Stitch - Complete Usage Instructions

## Table of Contents
1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [GUI Usage](#gui-usage)
4. [Command Line Usage](#command-line-usage)
5. [Cross-Platform Notes](#cross-platform-notes)
6. [Troubleshooting](#troubleshooting)

## Installation

### Windows

1. **Install Python 3.9 or later**
   - Download from [python.org](https://www.python.org/downloads/)
   - During installation, check "Add Python to PATH"

2. **Open Command Prompt or PowerShell**
   - Press `Win + R`, type `cmd`, press Enter
   - Or search for "PowerShell" in Start menu

3. **Navigate to project directory**
   ```cmd
   cd path\to\Stitch2Stitch\tbm
   ```

4. **Create virtual environment**
   ```cmd
   python -m venv venv
   ```

5. **Activate virtual environment**
   ```cmd
   venv\Scripts\activate
   ```

6. **Install dependencies**
   ```cmd
   pip install -r requirements.txt
   ```

### macOS

1. **Install Python 3.9 or later**
   - Python may already be installed (check with `python3 --version`)
   - If not, install via Homebrew: `brew install python3`
   - Or download from [python.org](https://www.python.org/downloads/)

2. **Open Terminal**
   - Press `Cmd + Space`, type "Terminal", press Enter
   - Or find Terminal in Applications > Utilities

3. **Navigate to project directory**
   ```bash
   cd path/to/Stitch2Stitch/tbm
   ```

4. **Create virtual environment**
   ```bash
   python3 -m venv venv
   ```

5. **Activate virtual environment**
   ```bash
   source venv/bin/activate
   ```

6. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### GUI Mode (Recommended for Beginners)

**Windows:**
```cmd
venv\Scripts\activate
python src\main.py
```

**macOS:**
```bash
source venv/bin/activate
python src/main.py
```

### Command Line Mode

**Windows:**
```cmd
venv\Scripts\activate
python src\main.py --cli --input "C:\path\to\images" --output "C:\path\to\output.tif"
```

**macOS:**
```bash
source venv/bin/activate
python src/main.py --cli --input /path/to/images --output /path/to/output.tif
```

## GUI Usage

### Step-by-Step Guide

1. **Launch the Application**
   - Run `python src/main.py` from the project directory
   - The GUI window will open

2. **Add Images**
   - Click "Add Images..." button
   - Select one or more image files (JPG, PNG, TIFF supported)
   - Images appear in the list
   - You can add more images or remove selected ones

3. **Configure Settings** (Optional)
   - **Quality Threshold**: Adjust slider (0.0-1.0)
     - Higher = stricter quality requirements
     - Recommended: 0.6-0.8
   - **GPU Acceleration**: Check if you have CUDA GPU
   - **Feature Detector**: Choose algorithm (LP-SIFT recommended)
   - **Feature Matcher**: Choose algorithm (FLANN recommended)
   - **Blending Method**: Choose algorithm (Multiband recommended)

4. **Set Output Path**
   - Click "Browse..." in Output section
   - Choose where to save the result
   - Select file format (TIFF recommended for large files)

5. **Preview Grid** (Optional)
   - Click "Create Grid Alignment"
   - See how images will be arranged
   - Progress bar shows status
   - Can stop with "Stop" button if needed

6. **Stitch Images**
   - Click "Stitch Images" button
   - Watch progress bar and status messages
   - Can stop with "Stop" button if needed
   - Process may take several minutes for large panoramas

7. **Review and Save**
   - Preview appears in Preview tab
   - Check Logs tab for detailed information
   - Click "Save Result" to export

### Progress Tracking

- **Progress Bar**: Shows percentage complete (0-100%)
- **Status Label**: Shows current operation
- **Logs Tab**: Detailed progress information
- **Stop Button**: Cancel current operation (with confirmation)

### Understanding Progress Stages

1. **0-10%**: Initializing, loading images
2. **10-30%**: Loading and assessing image quality
3. **30-50%**: Detecting features in images
4. **50-70%**: Matching features between images
5. **70-85%**: Aligning images
6. **85-100%**: Blending final panorama

## Command Line Usage

### Basic Syntax

```bash
python src/main.py --cli [OPTIONS]
```

### Required Arguments

- `--input PATH`: Directory containing input images
- `--output PATH`: Output file path for panorama

### Optional Arguments

- `--quality-threshold FLOAT`: Quality threshold (0.0-1.0, default: 0.7)
- `--gpu`: Enable GPU acceleration
- `--detector NAME`: Feature detector (lp_sift, sift, orb, akaze)
- `--matcher NAME`: Feature matcher (flann, loftr, superglue, disk)
- `--blender NAME`: Blending method (multiband, feather, linear, semantic, pixelstitch)
- `--grid-only`: Create grid alignment only, don't stitch

### Examples

**Basic stitching:**
```bash
python src/main.py --cli --input images/ --output panorama.tif
```

**High quality with GPU:**
```bash
python src/main.py --cli \
    --input images/ \
    --output panorama.tif \
    --quality-threshold 0.8 \
    --detector sift \
    --matcher superglue \
    --blender semantic \
    --gpu
```

**Fast processing:**
```bash
python src/main.py --cli \
    --input images/ \
    --output panorama.tif \
    --detector orb \
    --blender feather
```

**Grid alignment only:**
```bash
python src/main.py --cli \
    --input images/ \
    --output grid.tif \
    --grid-only
```

## Cross-Platform Notes

### Path Differences

**Windows:**
- Uses backslashes: `C:\Users\Name\Pictures`
- Drive letters: `C:`, `D:`, etc.
- Case-insensitive file names

**macOS:**
- Uses forward slashes: `/Users/name/Pictures`
- No drive letters
- Case-sensitive file names

**The application handles these automatically**, but be aware when typing paths manually.

### File Permissions

**macOS/Linux:**
- May need to make scripts executable: `chmod +x script.py`
- Log files stored in `~/Library/Application Support/Stitch2Stitch/logs/` (macOS)

**Windows:**
- Log files stored in `%LOCALAPPDATA%\Stitch2Stitch\logs\`

### GPU Support

**Windows:**
- Requires NVIDIA GPU with CUDA
- Install CUDA Toolkit from NVIDIA website
- Install PyTorch with CUDA: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`

**macOS:**
- GPU acceleration limited (Metal support may be available)
- CPU processing recommended for most cases
- M1/M2/M3 chips: Check PyTorch MPS support

### Performance Tips

**Windows:**
- Use NTFS file system for best performance
- Close other applications to free memory
- Use SSD for faster I/O

**macOS:**
- Use APFS file system
- Close other applications
- Consider using Activity Monitor to check memory usage

## Troubleshooting

### Common Issues

**"No module named 'PyQt6'"**
- Solution: Make sure virtual environment is activated
- Reinstall: `pip install PyQt6`

**"Could not load image"**
- Check file path is correct
- Verify image format is supported (JPG, PNG, TIFF)
- Check file permissions

**"Out of memory"**
- Reduce number of images (use --max-images)
- Lower image resolution
- Close other applications
- Use CPU instead of GPU

**"No feature matches found"**
- Images may not have enough overlap (need 30-50%)
- Try different feature detector (SIFT instead of ORB)
- Check image quality (may be too blurry)

**Progress bar stuck**
- Check Logs tab for error messages
- Try stopping and restarting
- Check if images are accessible

**GUI won't open**
- Check Python version: `python --version` (need 3.9+)
- Check PyQt6 installation: `pip list | grep PyQt6`
- Try command line mode instead

### Getting Help

1. Check Logs tab in GUI for error messages
2. Review log file:
   - Windows: `%LOCALAPPDATA%\Stitch2Stitch\logs\stitch2stitch.log`
   - macOS: `~/Library/Application Support/Stitch2Stitch/logs/stitch2stitch.log`
3. Run with verbose logging:
   ```bash
   python src/main.py --cli --input images/ --output out.tif 2>&1 | tee debug.log
   ```

### Performance Optimization

**For Large Panoramas (1000+ images):**
- Use GPU acceleration if available
- Process in batches
- Use faster algorithms (ORB + FLANN + Feather)
- Increase system memory if possible
- Use SSD storage

**For Best Quality:**
- Use SIFT detector
- Use SuperGlue matcher (with GPU)
- Use Semantic or PixelStitch blending
- Higher quality threshold (0.8)

**For Speed:**
- Use ORB detector
- Use FLANN matcher
- Use Feather blending
- Lower quality threshold (0.6)

## Advanced Usage

### Batch Processing

Process multiple directories:

**Windows (PowerShell):**
```powershell
Get-ChildItem -Directory | ForEach-Object {
    python src\main.py --cli --input $_.FullName --output "$($_.Name).tif"
}
```

**macOS/Linux:**
```bash
for dir in */; do
    python src/main.py --cli --input "$dir" --output "${dir%/}.tif"
done
```

### Scripting

Create a batch script:

**Windows (`stitch.bat`):**
```batch
@echo off
cd /d "%~dp0"
call venv\Scripts\activate
python src\main.py --cli --input %1 --output %2
```

**macOS/Linux (`stitch.sh`):**
```bash
#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
python src/main.py --cli --input "$1" --output "$2"
```

Make executable: `chmod +x stitch.sh`

## Tips and Best Practices

1. **Image Preparation**
   - Ensure 30-50% overlap between images
   - Use sharp, in-focus images
   - Avoid extreme exposure differences
   - Keep images flat (no lens distortion)

2. **Workflow**
   - Start with grid alignment to verify coverage
   - Use lower quality threshold initially to test
   - Adjust settings based on results
   - Save frequently

3. **File Management**
   - Use TIFF format for large panoramas
   - Keep original images as backup
   - Organize images in folders
   - Use descriptive output names

4. **Performance**
   - Process during off-peak hours for large jobs
   - Monitor system resources
   - Use appropriate algorithms for your needs
   - Consider processing in batches

