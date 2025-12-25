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
        PYTHON_VERSION_INSTALLED=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
        echo "Found Python $PYTHON_VERSION_INSTALLED"
        
        # Check if version is 3.9 or higher (using awk instead of bc)
        MAJOR=$(echo "$PYTHON_VERSION_INSTALLED" | cut -d'.' -f1)
        MINOR=$(echo "$PYTHON_VERSION_INSTALLED" | cut -d'.' -f2)
        
        if [ "$MAJOR" -gt 3 ] || ([ "$MAJOR" -eq 3 ] && [ "$MINOR" -ge 9 ]); then
            PYTHON_CMD="python3"
            # Verify Python actually works
            if ! $PYTHON_CMD --version &> /dev/null; then
                echo "Warning: python3 found but not working properly"
                return 1
            fi
            return 0
        else
            echo "Python version $PYTHON_VERSION_INSTALLED is too old. Need 3.9 or higher."
        fi
    fi
    
    return 1
}

# Install Python via Homebrew
install_python() {
    echo "Python 3.9+ not found. Installing Python via Homebrew..."
    echo "This may take a few minutes..."
    
    brew install python@${PYTHON_VERSION}
    
    # Add to PATH and verify
    if [[ $(uname -m) == "arm64" ]]; then
        export PATH="/opt/homebrew/opt/python@${PYTHON_VERSION}/bin:$PATH"
        PYTHON_CMD="/opt/homebrew/opt/python@${PYTHON_VERSION}/bin/python3"
    else
        export PATH="/usr/local/opt/python@${PYTHON_VERSION}/bin:$PATH"
        PYTHON_CMD="/usr/local/opt/python@${PYTHON_VERSION}/bin/python3"
    fi
    
    # Verify Python is accessible
    if [ ! -f "$PYTHON_CMD" ]; then
        # Fallback to python3 in PATH
    PYTHON_CMD="python3"
    fi
    
    if ! $PYTHON_CMD --version &> /dev/null; then
        echo "Error: Python installation failed or Python is not accessible" >&2
        exit 1
    fi
    
    echo "Python installed successfully!"
    $PYTHON_CMD --version
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
    if ! $PYTHON_CMD --version; then
        echo "Error: Python verification failed" >&2
        exit 1
    fi
    
    # Verify Python executable path
    PYTHON_FULL_PATH=$(which $PYTHON_CMD || command -v $PYTHON_CMD)
    if [ -z "$PYTHON_FULL_PATH" ]; then
        echo "Error: Could not find Python executable: $PYTHON_CMD" >&2
        exit 1
    fi
    echo "Using Python: $PYTHON_FULL_PATH"
    
    # Create installation directory
    echo "Creating installation directory..."
    mkdir -p "$INSTALL_DIR"
    if [ ! -d "$INSTALL_DIR" ]; then
        echo "Error: Could not create installation directory: $INSTALL_DIR" >&2
        exit 1
    fi
    cd "$INSTALL_DIR" || {
        echo "Error: Could not change to installation directory: $INSTALL_DIR" >&2
        exit 1
    }
    
    # Copy application files
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    if [ "$SCRIPT_DIR" != "$INSTALL_DIR" ]; then
        echo "Copying application files..."
        # Copy src directory
        if [ -d "$SCRIPT_DIR/src" ]; then
            cp -r "$SCRIPT_DIR/src" "$INSTALL_DIR/" || {
                echo "Warning: Failed to copy src directory" >&2
            }
        fi
        # Copy other necessary files
        for file in requirements.txt README.md LICENSE setup.py; do
            if [ -f "$SCRIPT_DIR/$file" ]; then
                cp "$SCRIPT_DIR/$file" "$INSTALL_DIR/" || {
                    echo "Warning: Failed to copy $file" >&2
                }
            fi
        done
        echo "Application files copied successfully."
    fi
    
    # Check if venv module is available
    echo "Checking for venv module..."
    if ! $PYTHON_CMD -m venv --help &> /dev/null; then
        echo "venv module not found. Attempting to ensure pip and venv are available..."
        # Try to ensure pip is installed (which often includes venv)
        $PYTHON_CMD -m ensurepip --upgrade --default-pip 2>/dev/null || true
        
        # Check again for venv
        if ! $PYTHON_CMD -m venv --help &> /dev/null; then
            echo "venv module still not found. Installing virtualenv as fallback..."
            $PYTHON_CMD -m pip install --user virtualenv --quiet 2>/dev/null || true
            if command -v virtualenv &> /dev/null; then
                VENV_CMD="virtualenv"
            else
                # Try to use virtualenv from user install
                VENV_CMD="$PYTHON_CMD -m virtualenv"
            fi
        else
            VENV_CMD="$PYTHON_CMD -m venv"
        fi
    else
        VENV_CMD="$PYTHON_CMD -m venv"
    fi
    
    # Create virtual environment
    echo "Creating virtual environment..."
    if ! $VENV_CMD venv; then
        echo "Error: Failed to create virtual environment" >&2
        echo "Trying alternative method..." >&2
        # Try installing venv package if missing
        $PYTHON_CMD -m pip install --user venv --quiet || true
        if ! $PYTHON_CMD -m venv venv; then
            echo "Error: Could not create virtual environment. Please ensure Python is properly installed." >&2
            exit 1
        fi
    fi
    
    # Verify venv was created
    if [ ! -f "venv/bin/activate" ]; then
        echo "Error: Virtual environment was not created successfully" >&2
        exit 1
    fi
    
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
    cat > "$APP_DIR/Contents/MacOS/Stitch2Stitch" << 'APPEOF'
#!/bin/bash
# Stitch2Stitch macOS App Launcher

INSTALL_DIR="$HOME/Applications/Stitch2Stitch"

# Try to find source files and virtual environment
SRC_DIR=""
VENV_PATH=""
WORK_DIR=""

# First check installation directory (where files should be copied)
if [ -f "$INSTALL_DIR/src/main.py" ]; then
    SRC_DIR="$INSTALL_DIR/src"
    WORK_DIR="$INSTALL_DIR"
    if [ -f "$INSTALL_DIR/venv/bin/activate" ]; then
        VENV_PATH="$INSTALL_DIR/venv/bin/activate"
    fi
fi

# If not found in installation directory, try to find project directory
if [ -z "$SRC_DIR" ]; then
    # Try common locations for the project
    POSSIBLE_DIRS=(
        "$HOME/Downloads/Stitch2Stitch/Stitch2Stitch"
        "$HOME/Documents/Stitch2Stitch"
        "$HOME/Stitch2Stitch"
        "$(dirname "$0")/../../.."
    )
    
    for dir in "${POSSIBLE_DIRS[@]}"; do
        if [ -f "$dir/src/main.py" ]; then
            SRC_DIR="$dir/src"
            WORK_DIR="$dir"
            if [ -f "$dir/venv/bin/activate" ]; then
                VENV_PATH="$dir/venv/bin/activate"
            elif [ -f "$INSTALL_DIR/venv/bin/activate" ]; then
                VENV_PATH="$INSTALL_DIR/venv/bin/activate"
            fi
            break
        fi
    done
fi

# Verify we found source files
if [ -z "$SRC_DIR" ] || [ ! -f "$SRC_DIR/main.py" ]; then
    osascript -e 'display dialog "Source files not found! Please make sure Stitch2Stitch is properly installed. Run ./install_macos.sh to install." buttons {"OK"} default button "OK" with icon stop'
    exit 1
fi

# Verify we found virtual environment
if [ -z "$VENV_PATH" ]; then
    osascript -e 'display dialog "Virtual environment not found! Please run ./install_macos.sh first." buttons {"OK"} default button "OK" with icon stop'
    exit 1
fi

# Change to working directory
cd "$WORK_DIR" || {
    osascript -e 'display dialog "Could not change to working directory!" buttons {"OK"} default button "OK" with icon stop'
    exit 1
}

# Activate virtual environment and run
source "$VENV_PATH"
python "$SRC_DIR/main.py" "$@"
APPEOF
    
    chmod +x "$APP_DIR/Contents/MacOS/Stitch2Stitch"
    
    echo "Application bundle created at: $APP_DIR"
}

# Run main function
main

