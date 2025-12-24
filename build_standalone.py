#!/usr/bin/env python3
"""
Build standalone executables for Stitch2Stitch
Creates self-contained executables using PyInstaller
"""

import sys
import subprocess
import platform
import shutil
from pathlib import Path

def check_pyinstaller():
    """Check if PyInstaller is installed"""
    try:
        import PyInstaller
        return True
    except ImportError:
        return False

def install_pyinstaller():
    """Install PyInstaller"""
    print("Installing PyInstaller...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])

def build_executable():
    """Build standalone executable"""
    system = platform.system()
    
    print(f"Building standalone executable for {system}...")
    
    # PyInstaller spec file content
    spec_content = """# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['src/main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('src', 'src'),
        ('docs', 'docs'),
    ],
    hiddenimports=[
        'PyQt6.QtCore',
        'PyQt6.QtGui',
        'PyQt6.QtWidgets',
        'cv2',
        'numpy',
        'scipy',
        'PIL',
        'tifffile',
        'psutil',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='Stitch2Stitch',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Set to True for debugging
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Add icon path here if you have one
)
"""
    
    # Write spec file
    spec_file = Path("Stitch2Stitch.spec")
    spec_file.write_text(spec_content)
    
    # Build command
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--clean",
        "--noconfirm",
        str(spec_file)
    ]
    
    if system == "Windows":
        cmd.extend(["--onefile", "--windowed"])
    elif system == "Darwin":  # macOS
        cmd.extend(["--onefile", "--windowed"])
    else:  # Linux
        cmd.extend(["--onefile"])
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.check_call(cmd)
    
    print("\n" + "="*50)
    print("Build Complete!")
    print("="*50)
    
    if system == "Windows":
        exe_path = Path("dist/Stitch2Stitch.exe")
        print(f"\nExecutable created at: {exe_path.absolute()}")
    elif system == "Darwin":
        app_path = Path("dist/Stitch2Stitch.app")
        print(f"\nApplication bundle created at: {app_path.absolute()}")
    else:
        exe_path = Path("dist/Stitch2Stitch")
        print(f"\nExecutable created at: {exe_path.absolute()}")

def main():
    """Main function"""
    print("="*50)
    print("Stitch2Stitch Standalone Builder")
    print("="*50)
    print()
    
    # Check Python version
    if sys.version_info < (3, 9):
        print("Error: Python 3.9 or higher is required")
        sys.exit(1)
    
    # Check/install PyInstaller
    if not check_pyinstaller():
        print("PyInstaller not found.")
        response = input("Install PyInstaller? (y/n): ")
        if response.lower() == 'y':
            install_pyinstaller()
        else:
            print("PyInstaller is required. Exiting.")
            sys.exit(1)
    
    # Build executable
    try:
        build_executable()
    except subprocess.CalledProcessError as e:
        print(f"\nError during build: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nBuild cancelled by user")
        sys.exit(1)

if __name__ == "__main__":
    main()

