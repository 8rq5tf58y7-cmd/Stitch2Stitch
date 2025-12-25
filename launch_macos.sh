#!/bin/bash
# Stitch2Stitch Launcher for macOS
# This script activates the virtual environment and launches the application

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_DIR="$HOME/Applications/Stitch2Stitch"

# Always run from the script directory (where src/ is located)
cd "$SCRIPT_DIR"

# Verify source files exist
if [ ! -f "$SCRIPT_DIR/src/main.py" ]; then
    echo "Error: Source files not found!"
    echo "Expected: $SCRIPT_DIR/src/main.py"
    echo ""
    echo "Please make sure you're running this script from the Stitch2Stitch project directory."
    exit 1
fi

# Try to find virtual environment
VENV_PATH=""

# First, check current directory (for development)
if [ -f "$SCRIPT_DIR/venv/bin/activate" ]; then
    VENV_PATH="$SCRIPT_DIR/venv/bin/activate"
# Then check installation directory
elif [ -f "$INSTALL_DIR/venv/bin/activate" ]; then
    VENV_PATH="$INSTALL_DIR/venv/bin/activate"
else
    echo "Virtual environment not found!"
    echo ""
    echo "Please run the installer first:"
    echo "  ./install_macos.sh"
    echo ""
    echo "Or if you've already installed, the venv should be at:"
    echo "  $INSTALL_DIR/venv"
    echo ""
    echo "Current directory: $SCRIPT_DIR"
    exit 1
fi

# Activate virtual environment and run
source "$VENV_PATH"
python src/main.py "$@"

