# Building Standalone Executables

This guide explains how to create self-contained executables that include Python and all dependencies.

## Prerequisites

1. **Python 3.9+** installed on your system
2. **PyInstaller**: `pip install pyinstaller`
3. **All dependencies**: `pip install -r requirements.txt`

## Quick Build

### Windows

```cmd
python build_standalone.py
```

Output: `dist/Stitch2Stitch.exe` (200-400 MB)

### macOS

```bash
python3 build_standalone.py
```

Output: `dist/Stitch2Stitch.app` (300-500 MB)

## Manual Build with PyInstaller

### Using the Spec File

```bash
pyinstaller pyinstaller.spec
```

### Custom Build Options

**Windows (single file, no console):**
```cmd
pyinstaller --onefile --windowed --name Stitch2Stitch src/main.py
```

**macOS (single file, no console):**
```bash
pyinstaller --onefile --windowed --name Stitch2Stitch src/main.py
```

**Linux:**
```bash
pyinstaller --onefile --name Stitch2Stitch src/main.py
```

## Build Options Explained

- `--onefile`: Creates single executable file
- `--windowed`: No console window (GUI only)
- `--console`: Show console window (for debugging)
- `--name`: Name of output executable
- `--icon`: Path to icon file (.ico for Windows, .icns for macOS)

## Troubleshooting Build Issues

### Missing Modules

If you get "ModuleNotFoundError" when running the executable:

1. Add to `hiddenimports` in `pyinstaller.spec`:
```python
hiddenimports=[
    'missing.module.name',
]
```

2. Rebuild:
```bash
pyinstaller --clean pyinstaller.spec
```

### Large File Size

The executable includes Python interpreter and all libraries. To reduce size:

1. Exclude unnecessary modules in spec file
2. Use `--exclude-module` flag
3. Consider using `--onedir` instead of `--onefile` (creates folder with multiple files)

### macOS Code Signing

To sign the macOS app (required for distribution):

```bash
codesign --deep --force --verify --verbose --sign "Developer ID Application: Your Name" dist/Stitch2Stitch.app
```

### Windows Antivirus False Positives

Some antivirus software may flag PyInstaller executables. Solutions:

1. Sign the executable with a code signing certificate
2. Submit to antivirus vendors for whitelisting
3. Use `--key` option to encrypt Python bytecode

## Creating Distribution Packages

### Windows: Inno Setup Installer

1. Download Inno Setup: https://jrsoftware.org/isinfo.php
2. Create installer script (see INSTALL.md)
3. Build installer

### macOS: DMG Creation

```bash
# Create DMG
hdiutil create -volname "Stitch2Stitch" \
    -srcfolder dist/Stitch2Stitch.app \
    -ov -format UDZO \
    Stitch2Stitch.dmg
```

### macOS: PKG Installer

```bash
# Create .pkg installer
pkgbuild --root dist/Stitch2Stitch.app \
    --identifier com.stitch2stitch.app \
    --version 1.0 \
    --install-location /Applications \
    Stitch2Stitch.pkg
```

## Testing the Build

### Windows

1. Copy `Stitch2Stitch.exe` to a clean Windows machine (no Python)
2. Double-click to run
3. Test GUI and CLI modes
4. Check all features work

### macOS

1. Copy `Stitch2Stitch.app` to a clean macOS machine (no Python)
2. Right-click → Open (first time)
3. Test GUI and CLI modes
4. Check all features work

## Build Scripts

The `build_standalone.py` script automates the build process:

```bash
python build_standalone.py
```

It will:
1. Check for PyInstaller
2. Install if needed
3. Build executable
4. Report output location

## Advanced: Custom Build Configuration

Edit `pyinstaller.spec` to customize:

- **Data files**: Add to `datas` list
- **Hidden imports**: Add to `hiddenimports` list
- **Excluded modules**: Add to `excludes` list
- **Icon**: Set `icon` path
- **Console mode**: Set `console=True/False`

## Distribution Checklist

- [ ] Test on clean system (no Python)
- [ ] Test all features
- [ ] Check file size (optimize if needed)
- [ ] Sign executables (recommended)
- [ ] Create installer package
- [ ] Test installer on clean system
- [ ] Create documentation
- [ ] Version number correct
- [ ] License included

## File Sizes

Typical sizes:
- **Windows .exe**: 200-400 MB
- **macOS .app**: 300-500 MB
- **Linux executable**: 250-450 MB

Size depends on:
- Included libraries
- Python version
- Compression settings
- UPX compression (if enabled)

## Performance

Standalone executables may be slightly slower than installed versions due to:
- Extraction of bundled files on startup
- No shared libraries
- Larger memory footprint

This is normal and acceptable for most use cases.

