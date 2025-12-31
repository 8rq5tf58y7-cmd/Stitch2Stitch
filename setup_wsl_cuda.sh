#!/bin/bash
# Stitch2Stitch WSL CUDA Setup Script
# Run this in WSL: bash /mnt/c/Users/ryanf/OneDrive\ -\ University\ of\ Maryland/Desktop/Stitch2Stitch/Stitch2Stitch-1/setup_wsl_cuda.sh

echo "=== Stitch2Stitch WSL CUDA Setup ==="

# Check for NVIDIA GPU
echo "Checking GPU..."
if nvidia-smi > /dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    echo "ERROR: NVIDIA GPU not detected in WSL"
    exit 1
fi

# Install pycolmap-cuda12
echo ""
echo "Installing pycolmap-cuda12..."
sudo python3 -m pip install pycolmap-cuda12 --break-system-packages

# Verify installation
echo ""
echo "Verifying installation..."
python3 -c "import pycolmap; print('pycolmap version:', pycolmap.__version__); print('CUDA available:', pycolmap.has_cuda)"

if python3 -c "import pycolmap; exit(0 if pycolmap.has_cuda else 1)"; then
    echo ""
    echo "=== SUCCESS! pycolmap-cuda12 installed with GPU support ==="
else
    echo ""
    echo "=== WARNING: pycolmap installed but CUDA not detected ==="
fi









