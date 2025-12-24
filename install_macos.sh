#!/bin/bash
# Stitch2Stitch macOS Installer
# Shell script to install Python and all dependencies

set -e  # Exit on error

INSTALL_DIR="$HOME/Applications/Stitch2Stitch"
PYTHON_VERSION="3.11"

echo "========================================"
echo "Stitch2Stitch macOS Installer"
echo "========================================"
echo ""

# Check for Homebrew
check_homebrew() {
    if ! command -v brew &> /dev/null; then
        echo "Homebrew not found. Installing Homebrew..."
        echo "This will require your password."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        
        # Add Homebrew to PATH for Apple Silicon Macs
        if [[ $(uname -m) == "arm64" ]]; then
            echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
            eval "$(/opt/homebrew/bin/brew shellenv)"
        fi
    else
        echo "Homebrew found ✓" | grep -q . && echo "Homebrew found ✓"
    fi
}

# Check for Python
check_python() {
    echo "Checking for Python installation..."
    
    # Check for python3 in PATH
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION_INSTALLED=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
        echo "Found Python $PYTHON_VERSION_INSTALLED"
        
        # Check if version is 3.9 or higher
        if (( $(echo "$PYTHON_VERSION_INSTALLED >= 3.9" | bc -l) )); then
            PYTHON_CMD="python3"
            return 0
        fi
    fi
    
    return 1
}

# Install Python via Homebrew
install_python() {
    echo "Python 3.9+ not found. Installing Python via Homebrew..."
    echo "This may take a few minutes..."
    
    brew install python@${PYTHON_VERSION}
    
    # Add to PATH
    if [[ $(uname -m) == "arm64" ]]; then
        export PATH="/opt/homebrew/opt/python@${PYTHON_VERSION}/bin:$PATH"
    else
        export PATH="/usr/local/opt/python@${PYTHON_VERSION}/bin:$PATH"
    fi
    
    PYTHON_CMD="python3"
    echo "Python installed successfully!"
}

# Main installation
main() {
    # Check/install Homebrew
    check_homebrew
    
    # Check/install Python
    if ! check_python; then
        install_python
    fi
    
    # Verify Python
    echo "Verifying Python installation..."
    $PYTHON_CMD --version
    
    # Create installation directory
    echo "Creating installation directory..."
    mkdir -p "$INSTALL_DIR"
    cd "$INSTALL_DIR"
    
    # Copy application files
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    if [ "$SCRIPT_DIR" != "$INSTALL_DIR" ]; then
        echo "Copying application files..."
        cp -r "$SCRIPT_DIR"/* "$INSTALL_DIR/" 2>/dev/null || true
    fi
    
    # Create virtual environment
    echo "Creating virtual environment..."
    $PYTHON_CMD -m venv venv
    
    # Activate virtual environment
    echo "Activating virtual environment..."
    source venv/bin/activate
    
    # Upgrade pip
    echo "Upgrading pip..."
    pip install --upgrade pip --quiet
    
    # Install requirements
    echo "Installing dependencies..."
    echo "This may take several minutes..."
    
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    else
        echo "requirements.txt not found. Installing default packages..."
        pip install opencv-python opencv-contrib-python numpy scipy Pillow PyQt6 matplotlib tqdm pyyaml tifffile psutil
    fi
    
    # Create launcher script
    echo "Creating launcher..."
    cat > Stitch2Stitch.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
python src/main.py "$@"
EOF
    
    chmod +x Stitch2Stitch.sh
    
    # Create macOS app bundle launcher
    create_app_bundle
    
    echo ""
    echo "========================================"
    echo "Installation Complete!"
    echo "========================================"
    echo ""
    echo "Installation directory: $INSTALL_DIR"
    echo ""
    echo "To run Stitch2Stitch:"
    echo "  1. Double-click Stitch2Stitch.app in Applications"
    echo "  2. Or run: $INSTALL_DIR/Stitch2Stitch.sh"
    echo ""
}

# Create macOS .app bundle
create_app_bundle() {
    echo "Creating macOS application bundle..."
    
    APP_DIR="$HOME/Applications/Stitch2Stitch.app"
    mkdir -p "$APP_DIR/Contents/MacOS"
    mkdir -p "$APP_DIR/Contents/Resources"
    
    # Create Info.plist
    cat > "$APP_DIR/Contents/Info.plist" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>Stitch2Stitch</string>
    <key>CFBundleIdentifier</key>
    <string>com.stitch2stitch.app</string>
    <key>CFBundleName</key>
    <string>Stitch2Stitch</string>
    <key>CFBundleVersion</key>
    <string>1.0</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>LSMinimumSystemVersion</key>
    <string>10.13</string>
</dict>
</plist>
EOF
    
    # Create executable
    cat > "$APP_DIR/Contents/MacOS/Stitch2Stitch" << EOF
#!/bin/bash
cd "$INSTALL_DIR"
source venv/bin/activate
python src/main.py "\$@"
EOF
    
    chmod +x "$APP_DIR/Contents/MacOS/Stitch2Stitch"
    
    echo "Application bundle created at: $APP_DIR"
}

# Run main function
main

