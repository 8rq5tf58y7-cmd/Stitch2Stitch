#!/bin/bash
# Stitch2Stitch Launcher for macOS
# This script activates the virtual environment and launches the application

cd "$(dirname "$0")"

# Check if virtual environment exists
if [ ! -f "venv/bin/activate" ]; then
    echo "Virtual environment not found!"
    echo "Please run ./install_macos.sh first."
    exit 1
fi

# Activate virtual environment and run
source venv/bin/activate
python src/main.py "$@"

