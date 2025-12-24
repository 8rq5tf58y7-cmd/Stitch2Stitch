# Quick Installation Guide

## Windows Users

### Easiest Method: Double-Click Installer

1. **Double-click `install.bat`**
   - This will automatically install Python and all dependencies
   - Follow the prompts
   - A desktop shortcut will be created

2. **Run Stitch2Stitch**
   - Double-click the desktop shortcut
   - Or run `Stitch2Stitch.bat` from the installation folder

### Alternative: PowerShell Installer

1. **Right-click `install_windows.ps1`**
2. **Select "Run with PowerShell"**
3. Follow the prompts

## macOS Users

### Easiest Method: Terminal Installer

1. **Open Terminal** (Applications > Utilities > Terminal)

2. **Navigate to the Stitch2Stitch folder:**
   ```bash
   cd /path/to/Stitch2Stitch/tbm
   ```

3. **Make installer executable:**
   ```bash
   chmod +x install_macos.sh
   ```

4. **Run installer:**
   ```bash
   ./install_macos.sh
   ```

5. **Run Stitch2Stitch:**
   - Double-click `Stitch2Stitch.app` in Applications
   - Or run: `~/Applications/Stitch2Stitch/Stitch2Stitch.sh`

## What Gets Installed

- **Python 3.9+** (if not already installed)
- **All required libraries** (OpenCV, PyQt6, NumPy, etc.)
- **Virtual environment** (isolated from system Python)
- **Application launcher**

## Standalone Executable (No Installation Needed)

Want a single file with everything included?

**Windows:**
```cmd
python build_standalone.py
```
Result: `dist/Stitch2Stitch.exe` - Just double-click to run!

**macOS:**
```bash
python3 build_standalone.py
```
Result: `dist/Stitch2Stitch.app` - Just double-click to run!

## Troubleshooting

**Windows: "Execution Policy" error?**
- Open PowerShell as Administrator
- Run: `Set-ExecutionPolicy RemoteSigned`

**macOS: "Permission denied"?**
- Run: `chmod +x install_macos.sh`

**Python not found?**
- The installer will download and install Python automatically
- On Windows, make sure to check "Add Python to PATH" if installing manually

**Need help?**
- See [INSTALL.md](INSTALL.md) for detailed instructions
- See [USAGE_INSTRUCTIONS.md](USAGE_INSTRUCTIONS.md) for usage guide

