"""
Automatic dependency installer for Stitch2Stitch

Checks for required packages and installs them if missing.
"""

import subprocess
import sys
import importlib
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Core dependencies with their package names and import names
CORE_DEPENDENCIES = [
    # (pip package name, import name, min_version or None)
    ("opencv-python", "cv2", "4.5.0"),
    ("opencv-contrib-python", "cv2", "4.5.0"),  # For SIFT, SURF, etc.
    ("numpy", "numpy", "1.20.0"),
    ("scipy", "scipy", "1.7.0"),
    ("Pillow", "PIL", "8.0.0"),
    ("PySide6", "PySide6", "6.0.0"),
    ("tqdm", "tqdm", None),
    ("scikit-image", "skimage", None),
    ("psutil", "psutil", None),
]

# Optional dependencies for advanced features
OPTIONAL_DEPENDENCIES = [
    # Deep Learning
    ("torch", "torch", "1.9.0"),
    ("torchvision", "torchvision", "0.10.0"),

    # Additional features
    ("scikit-learn", "sklearn", None),
    ("matplotlib", "matplotlib", None),
    ("tifffile", "tifffile", None),
    ("pyyaml", "yaml", None),
    ("numba", "numba", None),

    # Image processing
    ("imagehash", "imagehash", "4.3.0"),  # For duplicate detection

    # AI Super Resolution (optional - large download)
    # ("realesrgan", "realesrgan", None),  # Uncomment to auto-install
    # ("basicsr", "basicsr", None),
]

# External tools (not pip-installable)
EXTERNAL_TOOLS = {
    "colmap": {
        "check_cmd": "colmap --help",
        "install_url": "https://colmap.github.io/install.html",
        "description": "COLMAP - 3D reconstruction from images"
    },
    "meshroom": {
        "check_cmd": "meshroom_batch --help",
        "install_url": "https://alicevision.org/#meshroom",
        "description": "Meshroom/AliceVision - Photogrammetry pipeline"
    }
}


def check_package_installed(import_name: str) -> bool:
    """Check if a package is importable."""
    if import_name is None:
        return True
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        return False


def get_package_version(import_name: str) -> Optional[str]:
    """Get the version of an installed package."""
    if import_name is None:
        return None
    try:
        module = importlib.import_module(import_name)
        return getattr(module, '__version__', None)
    except ImportError:
        return None


def install_package(package_name: str, upgrade: bool = False) -> bool:
    """Install a package using pip.

    Tries multiple strategies to handle different environments:
    1. Standard install (venvs, Windows)
    2. User install (externally-managed Linux like WSL Ubuntu)
    3. Break system packages (last resort for WSL)
    """
    try:
        # Build base command
        base_cmd = [sys.executable, "-m", "pip", "install"]
        if upgrade:
            base_cmd.append("--upgrade")

        # Try different install strategies
        install_strategies = [
            base_cmd + [package_name],  # Standard
            base_cmd + ["--user", package_name],  # User install for externally-managed
            base_cmd + ["--break-system-packages", package_name],  # Last resort
        ]

        logger.info(f"Installing {package_name}...")

        for cmd in install_strategies:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            if result.returncode == 0:
                logger.info(f"Successfully installed {package_name}")
                return True

            # Check if it's an externally-managed error - try next strategy
            if "externally-managed-environment" in result.stderr:
                logger.debug(f"Trying alternative install strategy for {package_name}...")
                continue

            # Other error - log and try next
            logger.debug(f"Install attempt failed: {result.stderr[:200]}")

        # All strategies failed
        logger.error(f"Failed to install {package_name} after trying all strategies")
        return False

    except subprocess.TimeoutExpired:
        logger.error(f"Timeout installing {package_name}")
        return False
    except Exception as e:
        logger.error(f"Error installing {package_name}: {e}")
        return False


def check_and_install_dependencies(
    install_optional: bool = True,
    interactive: bool = True
) -> Dict[str, bool]:
    """
    Check for required dependencies and install missing ones.
    
    Args:
        install_optional: Also check/install optional dependencies
        interactive: Show prompts (for GUI mode)
        
    Returns:
        Dict mapping package name to installation status
    """
    results = {}
    missing_core = []
    missing_optional = []
    
    # Check core dependencies
    logger.info("Checking core dependencies...")
    for pip_name, import_name, min_version in CORE_DEPENDENCIES:
        if check_package_installed(import_name):
            results[pip_name] = True
            version = get_package_version(import_name)
            logger.debug(f"  [OK] {pip_name} ({version or 'installed'})")
        else:
            results[pip_name] = False
            missing_core.append(pip_name)
            logger.warning(f"  [MISSING] {pip_name} (missing)")
    
    # Check optional dependencies
    if install_optional:
        logger.info("Checking optional dependencies...")
        for pip_name, import_name, min_version in OPTIONAL_DEPENDENCIES:
            if check_package_installed(import_name):
                results[pip_name] = True
                version = get_package_version(import_name)
                logger.debug(f"  [OK] {pip_name} ({version or 'installed'})")
            else:
                results[pip_name] = False
                missing_optional.append(pip_name)
                logger.debug(f"  [-] {pip_name} (optional, not installed)")
    
    # Install missing core dependencies
    if missing_core:
        logger.info(f"Installing {len(missing_core)} missing core dependencies...")
        for package in missing_core:
            success = install_package(package)
            results[package] = success
            if not success:
                logger.error(f"Failed to install required package: {package}")
    
    # Install missing optional dependencies (silently)
    if install_optional and missing_optional:
        logger.info(f"Installing {len(missing_optional)} optional dependencies...")
        for package in missing_optional:
            success = install_package(package)
            results[package] = success
            if not success:
                logger.debug(f"Optional package not installed: {package}")
    
    return results


def check_opencv_features() -> Dict[str, bool]:
    """Check for specific OpenCV features/algorithms."""
    features = {}
    
    try:
        import cv2
        
        # Check for SIFT (should be in opencv-contrib-python)
        try:
            sift = cv2.SIFT_create()
            features['SIFT'] = True
        except:
            features['SIFT'] = False
        
        # Check for USAC/MAGSAC methods
        try:
            _ = cv2.USAC_MAGSAC
            features['MAGSAC++'] = True
        except AttributeError:
            features['MAGSAC++'] = False
        
        try:
            _ = cv2.USAC_ACCURATE
            features['USAC++'] = True
        except AttributeError:
            features['USAC++'] = False
        
        # Check for LSD line detector
        try:
            lsd = cv2.createLineSegmentDetector()
            features['LSD'] = True
        except:
            features['LSD'] = False
        
        features['version'] = cv2.__version__
        
    except ImportError:
        features['cv2'] = False
    
    return features


def check_gpu_available() -> Dict[str, bool]:
    """Check for GPU acceleration availability."""
    gpu_status = {
        'cuda_available': False,
        'cuda_version': None,
        'cudnn_available': False,
        'torch_cuda': False,
        'opencv_cuda': False
    }
    
    # Check PyTorch CUDA
    try:
        import torch
        gpu_status['torch_cuda'] = torch.cuda.is_available()
        if gpu_status['torch_cuda']:
            gpu_status['cuda_available'] = True
            gpu_status['cuda_version'] = torch.version.cuda
    except ImportError:
        pass
    
    # Check OpenCV CUDA
    try:
        import cv2
        gpu_status['opencv_cuda'] = cv2.cuda.getCudaEnabledDeviceCount() > 0
    except:
        pass
    
    return gpu_status


def get_install_report() -> str:
    """Generate a human-readable installation report."""
    lines = ["=" * 50, "Stitch2Stitch Dependency Report", "=" * 50, ""]
    
    # Check dependencies
    lines.append("Core Dependencies:")
    for pip_name, import_name, min_version in CORE_DEPENDENCIES:
        if check_package_installed(import_name):
            version = get_package_version(import_name)
            lines.append(f"  [OK] {pip_name}: {version or 'installed'}")
        else:
            lines.append(f"  [MISSING] {pip_name}: MISSING")
    
    lines.append("")
    lines.append("Optional Dependencies:")
    for pip_name, import_name, min_version in OPTIONAL_DEPENDENCIES:
        if check_package_installed(import_name):
            version = get_package_version(import_name)
            lines.append(f"  [OK] {pip_name}: {version or 'installed'}")
        else:
            lines.append(f"  [-] {pip_name}: not installed")
    
    # OpenCV features
    lines.append("")
    lines.append("OpenCV Features:")
    cv_features = check_opencv_features()
    for feature, available in cv_features.items():
        if feature == 'version':
            lines.append(f"  Version: {available}")
        else:
            status = "[OK]" if available else "[X]"
            lines.append(f"  {status} {feature}")
    
    # GPU status
    lines.append("")
    lines.append("GPU Acceleration:")
    gpu = check_gpu_available()
    lines.append(f"  CUDA available: {'Yes' if gpu['cuda_available'] else 'No'}")
    if gpu['cuda_version']:
        lines.append(f"  CUDA version: {gpu['cuda_version']}")
    lines.append(f"  PyTorch CUDA: {'Yes' if gpu['torch_cuda'] else 'No'}")
    lines.append(f"  OpenCV CUDA: {'Yes' if gpu['opencv_cuda'] else 'No'}")
    
    lines.append("")
    lines.append("=" * 50)
    
    return "\n".join(lines)


def ensure_dependencies(silent: bool = False) -> bool:
    """
    Ensure all dependencies are installed.
    Called at application startup.

    Returns:
        True if all core dependencies are available
    """
    if not silent:
        logger.info("Checking dependencies...")

    # Quick check of core dependencies first
    all_core_ok = True
    for pip_name, import_name, _ in CORE_DEPENDENCIES:
        if import_name and not check_package_installed(import_name):
            all_core_ok = False
            break

    # Also check key optional dependencies that should auto-install
    key_optional = ["imagehash"]  # These will be auto-installed if missing
    missing_optional = []
    for pip_name, import_name, _ in OPTIONAL_DEPENDENCIES:
        if pip_name in key_optional and import_name and not check_package_installed(import_name):
            missing_optional.append(pip_name)

    if all_core_ok and not missing_optional:
        if not silent:
            logger.info("All core dependencies satisfied")
        return True

    # Install missing dependencies
    if not silent:
        if not all_core_ok:
            logger.info("Installing missing core dependencies...")
        if missing_optional:
            logger.info(f"Installing optional dependencies: {missing_optional}")

    # Install core dependencies if needed
    if not all_core_ok:
        results = check_and_install_dependencies(install_optional=False, interactive=False)

    # Install missing key optional dependencies
    for pip_name in missing_optional:
        if not silent:
            logger.info(f"Installing {pip_name}...")
        install_package(pip_name)

    # Verify core dependencies
    for pip_name, import_name, _ in CORE_DEPENDENCIES:
        if import_name and not check_package_installed(import_name):
            logger.error(f"Critical dependency missing: {pip_name}")
            return False

    return True


if __name__ == "__main__":
    # Run as script to check/install dependencies
    logging.basicConfig(level=logging.INFO)
    print(get_install_report())
    
    print("\nInstalling missing dependencies...")
    results = check_and_install_dependencies()
    
    print("\nFinal status:")
    for package, status in results.items():
        print(f"  {package}: {'OK' if status else 'FAILED'}")






