"""
Cross-platform utilities for Windows and macOS compatibility
"""

import sys
import platform
from pathlib import Path
from typing import Union


def get_home_directory() -> Path:
    """Get user home directory in a cross-platform way"""
    return Path.home()


def normalize_path(path: Union[str, Path]) -> Path:
    """Normalize path for cross-platform compatibility"""
    return Path(path).resolve()


def get_app_data_directory() -> Path:
    """Get application data directory"""
    system = platform.system()
    
    if system == "Windows":
        # Windows: Use AppData\Local
        app_data = Path.home() / "AppData" / "Local" / "Stitch2Stitch"
    elif system == "Darwin":  # macOS
        # macOS: Use ~/Library/Application Support
        app_data = Path.home() / "Library" / "Application Support" / "Stitch2Stitch"
    else:  # Linux and others
        # Linux: Use ~/.local/share
        app_data = Path.home() / ".local" / "share" / "Stitch2Stitch"
    
    app_data.mkdir(parents=True, exist_ok=True)
    return app_data


def get_logs_directory() -> Path:
    """Get logs directory"""
    logs_dir = get_app_data_directory() / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir


def is_windows() -> bool:
    """Check if running on Windows"""
    return platform.system() == "Windows"


def is_macos() -> bool:
    """Check if running on macOS"""
    return platform.system() == "Darwin"


def is_linux() -> bool:
    """Check if running on Linux"""
    return platform.system() == "Linux"


def get_platform_name() -> str:
    """Get platform name"""
    return platform.system()


def fix_path_separators(path: str) -> str:
    """Fix path separators for current platform"""
    if is_windows():
        return path.replace('/', '\\')
    else:
        return path.replace('\\', '/')


def ensure_executable_permissions(file_path: Path):
    """Ensure file has executable permissions (Unix-like systems)"""
    if not is_windows():
        import os
        os.chmod(file_path, 0o755)

