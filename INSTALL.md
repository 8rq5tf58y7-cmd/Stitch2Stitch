# Stitch2Stitch - Self-Contained Installation Guide

This guide explains how to install Stitch2Stitch as a self-contained application that includes Python and all dependencies.

## Quick Installation

### Windows

**Option 1: Automatic Installer (Recommended)**
1. Right-click `install_windows.ps1`
2. Select "Run with PowerShell"
3. Follow the prompts
4. The installer will:
   - Check for Python, install if needed
   - Create virtual environment
   - Install all dependencies
   - Create desktop shortcut

**Option 2: Manual Installation**
```powershell
# Run PowerShell as Administrator (recommended)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\install_windows.ps1
```

**Option 3: Standalone Executable**
```powershell
# Build standalone .exe (includes everything)
python build_standalone.py
# Executable will be in dist/Stitch2Stitch.exe
```

### macOS

**Option 1: Automatic Installer (Recommended)**
```bash
# Make script executable
chmod +x install_macos.sh

# Run installer
./install_macos.sh
```

The installer will:
- Check for Homebrew, install if needed
- Check for Python 3.9+, install if needed
- Create virtual environment
- Install all dependencies
- Create .app bundle in Applications

**Option 2: Standalone Application**
```bash
# Build standalone .app bundle
python3 build_standalone.py
# Application will be in dist/Stitch2Stitch.app
```

## Installation Methods Explained

### Method 1: Installer Scripts (Recommended for Development)

**Windows (`install_windows.ps1`):**
- Automatically downloads and installs Python if not found
- Creates isolated virtual environment
- Installs all dependencies
- Creates launcher script and desktop shortcut
- Installation directory: `%LOCALAPPDATA%\Stitch2Stitch`

**macOS (`install_macos.sh`):**
- Uses Homebrew to install Python if needed
- Creates isolated virtual environment
- Installs all dependencies
- Creates .app bundle in Applications folder
- Installation directory: `~/Applications/Stitch2Stitch`

**Advantages:**
- Easy to update
- Can use latest Python versions
- Smaller installation size
- Easy to uninstall (just delete folder)

**Disadvantages:**
- Requires internet connection
- Takes time to install dependencies

### Method 2: Standalone Executable (Recommended for Distribution)

**Windows:**
- Creates single `.exe` file with everything bundled
- No Python installation needed
- No virtual environment needed
- Just double-click to run

**macOS:**
- Creates `.app` bundle with everything bundled
- No Python installation needed
- No virtual environment needed
- Just double-click to run

**Advantages:**
- Completely self-contained
- No installation needed
- Can distribute single file
- Works on systems without Python

**Disadvantages:**
- Larger file size (200-500 MB)
- Longer build time
- Requires PyInstaller

## Building Standalone Executables

### Prerequisites

1. **Python 3.9+** (for building only)
2. **PyInstaller**: `pip install pyinstaller`

### Build Process

**Windows:**
```powershell
# Install PyInstaller
pip install pyinstaller

# Build executable
python build_standalone.py

# Or use spec file directly
pyinstaller pyinstaller.spec
```

**macOS:**
```bash
# Install PyInstaller
pip3 install pyinstaller

# Build application
python3 build_standalone.py

# Or use spec file directly
pyinstaller pyinstaller.spec
```

### Build Output

**Windows:**
- Executable: `dist/Stitch2Stitch.exe`
- Size: ~200-400 MB
- Can be distributed as single file

**macOS:**
- Application: `dist/Stitch2Stitch.app`
- Size: ~300-500 MB
- Can be distributed as .app bundle or .dmg

## Running After Installation

### Using Installer Scripts

**Windows:**
```cmd
# Method 1: Double-click Stitch2Stitch.bat
# Method 2: Run from command line
cd %LOCALAPPDATA%\Stitch2Stitch
Stitch2Stitch.bat
```

**macOS:**
```bash
# Method 1: Double-click Stitch2Stitch.app in Applications
# Method 2: Run from command line
~/Applications/Stitch2Stitch/Stitch2Stitch.sh
```

### Using Standalone Executables

**Windows:**
- Double-click `Stitch2Stitch.exe`
- Or run from command line: `Stitch2Stitch.exe --cli --input images/ --output out.tif`

**macOS:**
- Double-click `Stitch2Stitch.app`
- Or run from command line: `open Stitch2Stitch.app --args --cli --input images/ --output out.tif`

## Creating Distribution Packages

### Windows: Create Installer (.msi)

Using Inno Setup or NSIS:

```inno
[Setup]
AppName=Stitch2Stitch
AppVersion=1.0
DefaultDirName={localappdata}\Stitch2Stitch
DefaultGroupName=Stitch2Stitch
OutputBaseFilename=Stitch2Stitch-Setup

[Files]
Source: "dist\Stitch2Stitch.exe"; DestDir: "{app}"

[Icons]
Name: "{group}\Stitch2Stitch"; Filename: "{app}\Stitch2Stitch.exe"
```

### macOS: Create DMG

```bash
# Create DMG from .app bundle
hdiutil create -volname "Stitch2Stitch" -srcfolder dist/Stitch2Stitch.app -ov -format UDZO Stitch2Stitch.dmg
```

## Troubleshooting Installation

### Windows Issues

**"Execution Policy" Error:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Python Not Found After Installation:**
- Restart terminal/command prompt
- Check PATH environment variable
- Manually add Python to PATH if needed

**Permission Denied:**
- Run PowerShell as Administrator
- Or install to user directory (default)

### macOS Issues

**Homebrew Installation Fails:**
- Check internet connection
- May need to run: `xcode-select --install`
- For Apple Silicon: Ensure Rosetta 2 is installed

**Python Not Found:**
- Check PATH: `echo $PATH`
- Add Homebrew Python to PATH in `~/.zshrc` or `~/.bash_profile`

**App Won't Open:**
- Right-click app → Open (first time only)
- Or: `xattr -cr Stitch2Stitch.app`

### Build Issues

**PyInstaller Errors:**
- Update PyInstaller: `pip install --upgrade pyinstaller`
- Check hidden imports in spec file
- Try building with `--debug` flag

**Missing Dependencies:**
- Ensure all packages in requirements.txt are installed
- Check hiddenimports in spec file
- Add missing modules to hiddenimports

## Uninstallation

### Installer Script Installation

**Windows:**
```cmd
# Delete installation directory
rmdir /s %LOCALAPPDATA%\Stitch2Stitch
# Delete desktop shortcut manually
```

**macOS:**
```bash
# Delete installation directory
rm -rf ~/Applications/Stitch2Stitch
# Delete app bundle
rm -rf ~/Applications/Stitch2Stitch.app
```

### Standalone Executable

**Windows:**
- Delete the `.exe` file

**macOS:**
- Delete the `.app` bundle

## Advanced: Custom Installation

### Custom Python Version

**Windows:**
```powershell
.\install_windows.ps1 -PythonVersion "3.12.0"
```

**macOS:**
```bash
PYTHON_VERSION="3.12" ./install_macos.sh
```

### Custom Installation Directory

**Windows:**
```powershell
.\install_windows.ps1 -InstallDir "D:\Stitch2Stitch"
```

**macOS:**
```bash
INSTALL_DIR="/Applications/Stitch2Stitch" ./install_macos.sh
```

## Distribution Checklist

When distributing Stitch2Stitch:

- [ ] Test on clean system (no Python installed)
- [ ] Test on both Windows and macOS
- [ ] Include README.md and USAGE_INSTRUCTIONS.md
- [ ] Create installer package (.msi/.dmg)
- [ ] Sign executables (optional but recommended)
- [ ] Test all features
- [ ] Include license file
- [ ] Create uninstaller (optional)

## Support

For installation issues:
1. Check log files in installation directory
2. Review error messages carefully
3. Ensure system meets requirements
4. Try manual installation steps
5. Check GitHub issues for known problems

