"""
Main GUI window for Stitch2Stitch
Modern, intuitive interface for panoramic image stitching
"""

import sys
from pathlib import Path
from typing import List, Optional
import logging
import numpy as np
import cv2

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QFileDialog, QLabel, QProgressBar, QTextEdit, QTabWidget,
    QGroupBox, QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox,
    QSlider, QMessageBox, QSplitter, QListWidget, QListWidgetItem,
    QScrollArea, QLineEdit
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QPixmap, QImage, QFont

from core.stitcher import ImageStitcher
from external.wsl_colmap_bridge import compute_cache_key
from utils.logger import setup_logger

logger = setup_logger(__name__)


class StitchingThread(QThread):
    """Thread for running stitching operations"""
    
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    
    def __init__(self, stitcher: ImageStitcher, image_paths: List[Path], grid_only: bool = False):
        super().__init__()
        self.stitcher = stitcher
        self.image_paths = image_paths
        self.grid_only = grid_only
        self._cancelled = False
    
    def cancel(self):
        """Cancel the stitching operation"""
        self._cancelled = True
        self.stitcher.cancel()
    
    def run(self):
        try:
            # Set up progress callback
            def progress_callback(percentage: int, message: str = ""):
                self.progress.emit(percentage)
                if message:
                    self.status.emit(message)
            
            # Set up cancel flag
            def cancel_flag():
                return self._cancelled
            
            # Update stitcher with callbacks
            self.stitcher.progress_callback = progress_callback
            self.stitcher.cancel_flag = cancel_flag
            
            if self.grid_only:
                self.status.emit("Creating grid alignment...")
                self.progress.emit(10)
                grid = self.stitcher.create_grid_alignment(self.image_paths)
                self.progress.emit(100)
                self.finished.emit(grid)
            else:
                self.status.emit("Starting stitching process...")
                panorama = self.stitcher.stitch(self.image_paths)
                self.finished.emit(panorama)
        except InterruptedError:
            self.status.emit("Operation cancelled")
            self.progress.emit(0)
        except Exception as e:
            logger.error(f"Stitching error: {e}", exc_info=True)
            self.error.emit(str(e))


class GridAlignmentThread(QThread):
    """Thread for running grid alignment operations with overlap threshold"""
    
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    
    def __init__(self, stitcher: ImageStitcher, image_paths: List[Path], min_overlap_percent: float, max_overlap_percent: float = 100.0, spacing_factor: float = 1.3):
        super().__init__()
        self.stitcher = stitcher
        self.image_paths = image_paths
        self.min_overlap_percent = min_overlap_percent
        self.max_overlap_percent = max_overlap_percent
        self.spacing_factor = spacing_factor
        self._cancelled = False
    
    def cancel(self):
        """Cancel the operation"""
        self._cancelled = True
        self.stitcher.cancel()
    
    def run(self):
        try:
            # Set up progress callback
            def progress_callback(percentage: int, message: str = ""):
                self.progress.emit(percentage)
                if message:
                    self.status.emit(message)
            
            # Set up cancel flag
            def cancel_flag():
                return self._cancelled
            
            # Update stitcher with callbacks
            self.stitcher.progress_callback = progress_callback
            self.stitcher.cancel_flag = cancel_flag
            
            self.status.emit(f"Creating grid alignment (overlap: {self.min_overlap_percent}%-{self.max_overlap_percent}%, spacing: {self.spacing_factor}x)...")
            self.progress.emit(10)
            grid = self.stitcher.create_grid_alignment(
                self.image_paths, 
                min_overlap_percent=self.min_overlap_percent,
                max_overlap_percent=self.max_overlap_percent,
                spacing_factor=self.spacing_factor
            )
            self.progress.emit(100)
            self.finished.emit(grid)
        except InterruptedError:
            self.status.emit("Operation cancelled")
            self.progress.emit(0)
        except Exception as e:
            logger.error(f"Grid alignment error: {e}", exc_info=True)
            self.error.emit(str(e))


class COLMAPThread(QThread):
    """Thread for running COLMAP pipeline operations"""
    
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    
    def __init__(self, image_paths: List[Path], output_dir: Path, transform_type: str = "homography",
                 blend_method: str = "multiband", matching_strategy: str = "exhaustive",
                 sequential_overlap: int = 10, gpu_indices: List[str] = None,
                 num_threads: int = -1, max_features: int = 8192,
                 min_inliers: int = 0, max_images: int = 0, use_source_alpha: bool = False,
                 remove_duplicates: bool = False, duplicate_threshold: float = 0.92,
                 warp_interpolation: str = 'lanczos',
                 erode_border: bool = True, border_erosion_pixels: int = 5):
        super().__init__()
        self.image_paths = image_paths
        self.output_dir = output_dir
        self.transform_type = transform_type
        self.blend_method = blend_method
        self.matching_strategy = matching_strategy
        self.sequential_overlap = sequential_overlap
        self.gpu_indices = gpu_indices or ["auto"]
        self.num_threads = num_threads
        self.max_features = max_features
        self.min_inliers = min_inliers
        self.max_images = max_images
        self.use_source_alpha = use_source_alpha
        self.remove_duplicates = remove_duplicates
        self.duplicate_threshold = duplicate_threshold
        self.warp_interpolation = warp_interpolation
        self.erode_border = erode_border
        self.border_erosion_pixels = border_erosion_pixels
        self._cancelled = False
    
    def cancel(self):
        """Cancel the operation"""
        self._cancelled = True
    
    def run(self):
        try:
            from external.pipelines import COLMAPPipeline
            
            # Progress callback that emits signals
            # percentage=-1 means "status only, no progress change"
            def progress_callback(percentage: int, message: str = ""):
                if self._cancelled:
                    raise InterruptedError("Operation cancelled")
                if percentage >= 0:
                    self.progress.emit(percentage)
                if message:
                    self.status.emit(message)
            
            use_affine = self.transform_type == "affine"

            # Determine GPU usage
            use_gpu = True
            gpu_index = -1  # Auto-detect
            if self.gpu_indices:
                if "cpu" in self.gpu_indices:
                    use_gpu = False
                elif "auto" not in self.gpu_indices:
                    # Use first specified GPU
                    try:
                        gpu_index = int(self.gpu_indices[0])
                    except:
                        gpu_index = -1

            pipeline = COLMAPPipeline(
                use_gpu=use_gpu,
                use_affine=use_affine,
                blend_method=self.blend_method,
                matcher_type=self.matching_strategy,
                sequential_overlap=self.sequential_overlap,
                gpu_index=gpu_index,
                num_threads=self.num_threads,
                max_features=self.max_features,
                min_inliers=self.min_inliers,
                max_images=self.max_images,
                use_source_alpha=self.use_source_alpha,
                remove_duplicates=self.remove_duplicates,
                duplicate_threshold=self.duplicate_threshold,
                warp_interpolation=self.warp_interpolation,
                erode_border=self.erode_border,
                border_erosion_pixels=self.border_erosion_pixels,
                progress_callback=progress_callback
            )
            
            self.status.emit("Starting COLMAP 2D stitching pipeline...")
            result = pipeline.run_2d_stitch(self.image_paths, self.output_dir)
            
            self.finished.emit(result)
        except InterruptedError:
            self.status.emit("Operation cancelled")
            self.progress.emit(0)
        except Exception as e:
            logger.error(f"COLMAP error: {e}", exc_info=True)
            self.error.emit(str(e))


class HLOCInstallThread(QThread):
    """Thread for installing HLOC in the background"""
    
    status = pyqtSignal(str)
    finished = pyqtSignal(bool, str)  # success, message
    
    def __init__(self):
        super().__init__()
    
    def run(self):
        import subprocess
        import sys
        import shutil
        
        try:
            # Step 1: Check if pycolmap is installed
            self.status.emit("Checking pycolmap installation...")
            try:
                import pycolmap
                self.status.emit(f"pycolmap v{getattr(pycolmap, '__version__', 'unknown')} found")
            except ImportError:
                self.finished.emit(False, "pycolmap is required but not installed. Please install pycolmap first.")
                return
            
            # Step 2: Check if Git is available
            self.status.emit("Checking for Git...")
            git_path = shutil.which("git")
            if not git_path:
                self.finished.emit(False, "Git is required to install HLOC.\n\nPlease install Git from: https://git-scm.com/download/win\nThen restart the application.")
                return
            self.status.emit(f"Git found at {git_path}")

            # Step 3: Install PyTorch with CUDA support
            self.status.emit("Checking PyTorch installation...")
            try:
                import torch
                self.status.emit(f"PyTorch {torch.__version__} already installed")
            except ImportError:
                self.status.emit("PyTorch not found. Installing PyTorch with CUDA 11.8 support...")
                self.status.emit("This may take 10-15 minutes and download ~2GB...")

                # Install PyTorch with CUDA 11.8 (widely compatible)
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "torch", "torchvision",
                     "--index-url", "https://download.pytorch.org/whl/cu118"],
                    capture_output=True,
                    text=True,
                    timeout=1200  # 20 minute timeout for PyTorch with CUDA
                )

                if result.returncode != 0:
                    error_msg = result.stderr or result.stdout
                    self.finished.emit(False, f"PyTorch installation failed:\n{error_msg[:500]}")
                    return

                self.status.emit("PyTorch installed successfully!")

            # Step 4: Install additional dependencies
            self.status.emit("Installing additional HLOC dependencies...")
            additional_deps = ["plotly", "h5py", "kornia>=0.6.11", "gdown"]

            result = subprocess.run(
                [sys.executable, "-m", "pip", "install"] + additional_deps,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            if result.returncode != 0:
                # Non-critical, log but continue
                self.status.emit("Warning: Some additional dependencies may have failed to install")
            else:
                self.status.emit("Additional dependencies installed successfully!")

            # Step 5: Install HLOC + LightGlue from GitHub
            self.status.emit("Installing HLOC and LightGlue from GitHub (this may take several minutes)...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install",
                 "git+https://github.com/cvg/Hierarchical-Localization.git",
                 "lightglue@git+https://github.com/cvg/LightGlue"],
                capture_output=True,
                text=True,
                timeout=900  # 15 minute timeout
            )

            if result.returncode == 0:
                self.status.emit("HLOC installed successfully!")
                self.finished.emit(True, "HLOC and all dependencies installed successfully!")
            else:
                error_msg = result.stderr or result.stdout
                self.finished.emit(False, f"HLOC installation failed:\n{error_msg[:500]}")
                
        except subprocess.TimeoutExpired:
            self.finished.emit(False, "Installation timed out after 15 minutes. Please try again or install manually.")
        except Exception as e:
            self.finished.emit(False, f"Installation error: {e}")


class COLMAPInstallThread(QThread):
    """Thread for installing PyCOLMAP (with CUDA support) in the background"""

    status = pyqtSignal(str)
    finished = pyqtSignal(bool, str)  # success, message

    def __init__(self):
        super().__init__()

    def run(self):
        import subprocess
        import sys
        import platform

        try:
            system = platform.system()

            # Step 1: Check current pycolmap installation
            self.status.emit("Checking for existing pycolmap installation...")
            try:
                import pycolmap
                current_version = getattr(pycolmap, '__version__', 'unknown')
                has_cuda = getattr(pycolmap, 'has_cuda', False)
                self.status.emit(f"Found pycolmap v{current_version} (CUDA: {has_cuda})")

                if has_cuda:
                    self.finished.emit(True, f"PyCOLMAP v{current_version} with CUDA is already installed!")
                    return
            except ImportError:
                self.status.emit("pycolmap not found, will install fresh")

            # Step 2: Determine installation strategy based on OS
            if system == "Windows":
                # On Windows, try WSL CUDA installation
                self.status.emit("Detected Windows - will install pycolmap and setup WSL CUDA support...")

                # First install regular pycolmap for Windows
                self.status.emit("Installing pycolmap for Windows...")
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "--upgrade", "pycolmap"],
                    capture_output=True,
                    text=True,
                    timeout=600  # 10 minute timeout
                )

                if result.returncode != 0:
                    error_msg = result.stderr or result.stdout
                    self.finished.emit(False, f"PyCOLMAP installation failed:\n{error_msg[:500]}")
                    return

                self.status.emit("pycolmap installed successfully!")

                # Check if WSL is available
                self.status.emit("Checking for WSL (Windows Subsystem for Linux)...")
                wsl_check = subprocess.run(
                    ["wsl", "--status"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )

                if wsl_check.returncode == 0:
                    self.status.emit("WSL detected! Setting up GPU-accelerated COLMAP in WSL...")

                    # Install pycolmap-cuda in WSL
                    wsl_install = subprocess.run(
                        ["wsl", "bash", "-c", "pip3 install --upgrade pycolmap-cuda12"],
                        capture_output=True,
                        text=True,
                        timeout=600
                    )

                    if wsl_install.returncode == 0:
                        self.status.emit("WSL CUDA support configured successfully!")
                        self.finished.emit(True,
                            "PyCOLMAP installed with WSL GPU acceleration!\n\n"
                            "✓ Windows: pycolmap (CPU)\n"
                            "✓ WSL: pycolmap-cuda12 (GPU-accelerated)\n\n"
                            "The app will automatically use GPU acceleration when available.")
                    else:
                        # WSL install failed, but Windows install succeeded
                        self.status.emit("WSL CUDA setup failed, but Windows installation succeeded")
                        self.finished.emit(True,
                            "PyCOLMAP installed for Windows (CPU mode)!\n\n"
                            "Note: WSL GPU setup failed. You can install it manually with:\n"
                            "wsl bash -c 'pip3 install pycolmap-cuda12'")
                else:
                    self.finished.emit(True,
                        "PyCOLMAP installed for Windows (CPU mode)!\n\n"
                        "For GPU acceleration, install WSL2 and run:\n"
                        "wsl bash -c 'pip3 install pycolmap-cuda12'")

            elif system == "Linux":
                # On Linux, try to install CUDA version
                self.status.emit("Detected Linux - attempting GPU-accelerated installation...")

                # Try CUDA 12 first
                self.status.emit("Installing pycolmap-cuda12...")
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "--upgrade", "pycolmap-cuda12"],
                    capture_output=True,
                    text=True,
                    timeout=600
                )

                if result.returncode == 0:
                    self.status.emit("GPU-accelerated pycolmap installed successfully!")
                    self.finished.emit(True, "PyCOLMAP installed with CUDA 12 GPU acceleration!")
                else:
                    # Fall back to regular pycolmap
                    self.status.emit("CUDA installation failed, trying CPU version...")
                    result = subprocess.run(
                        [sys.executable, "-m", "pip", "install", "--upgrade", "pycolmap"],
                        capture_output=True,
                        text=True,
                        timeout=600
                    )

                    if result.returncode == 0:
                        self.finished.emit(True, "PyCOLMAP installed (CPU mode)!")
                    else:
                        error_msg = result.stderr or result.stdout
                        self.finished.emit(False, f"PyCOLMAP installation failed:\n{error_msg[:500]}")

            else:  # macOS
                self.status.emit("Detected macOS - installing pycolmap...")
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "--upgrade", "pycolmap"],
                    capture_output=True,
                    text=True,
                    timeout=600
                )

                if result.returncode == 0:
                    self.finished.emit(True, "PyCOLMAP installed successfully!")
                else:
                    error_msg = result.stderr or result.stdout
                    self.finished.emit(False, f"PyCOLMAP installation failed:\n{error_msg[:500]}")

        except subprocess.TimeoutExpired:
            self.finished.emit(False, "Installation timed out. Please try again or check your internet connection.")
        except Exception as e:
            self.finished.emit(False, f"Installation error: {e}")


class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.image_paths: List[Path] = []
        self.stitcher: Optional[ImageStitcher] = None
        self.thread: Optional[StitchingThread] = None
        self.colmap_thread: Optional[COLMAPThread] = None
        self.hloc_install_thread: Optional[HLOCInstallThread] = None
        self._preview_image: Optional[np.ndarray] = None
        self._preview_zoom: float = 1.0
        try:
            self.init_ui()
            # Initialize stitcher, but don't fail if it errors (can be retried later)
            try:
                self.init_stitcher()
            except Exception as e:
                logger.warning(f"Failed to initialize stitcher on startup: {e}", exc_info=True)
                # Stitcher will be initialized when user tries to use it
                self.stitcher = None
        except Exception as e:
            logger.error(f"Failed to initialize window UI: {e}", exc_info=True)
            raise
    
    def init_ui(self):
        """Initialize user interface"""
        self.setWindowTitle("Stitch2Stitch - Advanced Panoramic Image Stitching")
        self.setMinimumSize(1200, 800)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel - controls
        left_panel = self.create_control_panel()
        main_layout.addWidget(left_panel, 1)
        
        # Right panel - preview and logs
        right_panel = self.create_preview_panel()
        main_layout.addWidget(right_panel, 2)
        
        # Status bar
        self.statusBar().showMessage("Ready")
    
    def create_control_panel(self) -> QWidget:
        """Create control panel with scroll support"""
        # Create scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setMinimumWidth(320)
        
        # Create content widget
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # File selection
        file_group = QGroupBox("Image Selection")
        file_layout = QVBoxLayout()
        
        self.image_list = QListWidget()
        self.image_list.setMaximumHeight(200)
        file_layout.addWidget(QLabel("Selected Images:"))
        file_layout.addWidget(self.image_list)
        
        btn_layout = QHBoxLayout()
        self.btn_add_images = QPushButton("Select Folder...")
        self.btn_add_images.setToolTip("Select a folder containing images to stitch")
        self.btn_add_images.clicked.connect(self.add_images)
        self.btn_remove_image = QPushButton("Remove Selected")
        self.btn_remove_image.clicked.connect(self.remove_selected_image)
        self.btn_clear = QPushButton("Clear All")
        self.btn_clear.clicked.connect(self.clear_images)
        
        btn_layout.addWidget(self.btn_add_images)
        btn_layout.addWidget(self.btn_remove_image)
        btn_layout.addWidget(self.btn_clear)
        file_layout.addLayout(btn_layout)
        
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        # Settings
        settings_group = QGroupBox("Settings")
        settings_layout = QVBoxLayout()
        
        # Quality threshold
        quality_layout = QHBoxLayout()
        quality_layout.addWidget(QLabel("Quality Threshold:"))
        self.quality_spin = QDoubleSpinBox()
        self.quality_spin.setRange(0.0, 1.0)
        self.quality_spin.setSingleStep(0.1)
        self.quality_spin.setValue(0.3)  # Lower default threshold to allow more images
        self.quality_spin.setDecimals(2)
        quality_layout.addWidget(self.quality_spin)
        settings_layout.addLayout(quality_layout)
        
        # GPU acceleration
        self.gpu_checkbox = QCheckBox("Enable GPU Acceleration")
        self.gpu_checkbox.setChecked(False)
        settings_layout.addWidget(self.gpu_checkbox)
        
        # Memory efficient mode
        self.memory_efficient_checkbox = QCheckBox("Memory Efficient Mode")
        self.memory_efficient_checkbox.setChecked(True)  # Enabled by default
        self.memory_efficient_checkbox.setToolTip(
            "Reduces memory usage by 70-90% during image loading.\n"
            "Uses compressed image caching and lazy loading.\n"
            "Recommended for large image sets or limited RAM systems."
        )
        settings_layout.addWidget(self.memory_efficient_checkbox)
        
        # Feature detector
        detector_layout = QHBoxLayout()
        detector_layout.addWidget(QLabel("Feature Detector:"))
        self.detector_combo = QComboBox()
        self.detector_combo.addItems([
            "LP-SIFT (Recommended)",
            "SuperPoint (Deep Learning)",
            "SIFT",
            "ORB",
            "AKAZE"
        ])
        self.detector_combo.setToolTip(
            "Feature detection algorithm:\n\n"
            "• LP-SIFT (Recommended) - Enhanced SIFT with edge priority\n"
            "  Best all-around choice for most panoramas\n\n"
            "• SuperPoint - Deep learning detector (requires PyTorch)\n"
            "  Best for 500+ images and repetitive scenes\n\n"
            "• SIFT - Classic scale-invariant features\n"
            "• ORB - Fast binary features (less accurate)\n"
            "• AKAZE - Good for texture-less regions"
        )
        detector_layout.addWidget(self.detector_combo)
        settings_layout.addLayout(detector_layout)
        
        # Max features/control points
        features_layout = QHBoxLayout()
        features_layout.addWidget(QLabel("Max Features:"))
        self.max_features_spin = QSpinBox()
        self.max_features_spin.setRange(100, 50000)
        self.max_features_spin.setValue(5000)
        self.max_features_spin.setSingleStep(500)
        self.max_features_spin.setToolTip("Maximum number of features/control points to detect per image.\nMore features = better alignment accuracy but slower processing.")
        features_layout.addWidget(self.max_features_spin)
        settings_layout.addLayout(features_layout)
        
        # Allow scale checkbox
        self.allow_scale_checkbox = QCheckBox("Allow Scaling")
        self.allow_scale_checkbox.setChecked(False)
        self.allow_scale_checkbox.setToolTip("Allow images to be scaled (stretched/shrunk) during alignment.\nDisable for flatbed scans where only rotation and translation should be applied.\nRotation is always allowed.")
        settings_layout.addWidget(self.allow_scale_checkbox)
        
        # ============================================================
        # QUICK PRESETS - Guide users to right settings
        # ============================================================
        presets_group = QGroupBox("⚡ Quick Presets (Choose One)")
        presets_layout = QVBoxLayout()
        presets_layout.setSpacing(4)
        
        preset_info = QLabel("Select a preset based on your images:")
        preset_info.setStyleSheet("font-size: 10px; color: #666;")
        presets_layout.addWidget(preset_info)
        
        preset_buttons = QHBoxLayout()
        
        self.preset_few_btn = QPushButton("ðŸ“· Few Images\n(2-50)")
        self.preset_few_btn.setToolTip(
            "Best for: 2-50 high-quality photos\n\n"
            "Settings:\n"
            "• LP-SIFT detector\n"
            "• MAGSAC++ verification\n"
            "• Multiband blending\n"
            "• Fast processing"
        )
        self.preset_few_btn.clicked.connect(lambda: self._apply_preset("few"))
        preset_buttons.addWidget(self.preset_few_btn)
        
        self.preset_medium_btn = QPushButton("ðŸ–¼ï¸ Medium Set\n(50-200)")
        self.preset_medium_btn.setToolTip(
            "Best for: 50-200 images\n\n"
            "Settings:\n"
            "• LP-SIFT detector\n"
            "• MAGSAC++ verification\n"
            "• Bundle adjustment ON\n"
            "• Duplicate removal ON"
        )
        self.preset_medium_btn.clicked.connect(lambda: self._apply_preset("medium"))
        preset_buttons.addWidget(self.preset_medium_btn)
        
        self.preset_large_btn = QPushButton("ðŸ—ºï¸ Large Set\n(200-500)")
        self.preset_large_btn.setToolTip(
            "Best for: 200-500 images\n\n"
            "Settings:\n"
            "• SuperPoint detector (if available)\n"
            "• MAGSAC++ verification\n"
            "• Hierarchical matching\n"
            "• Memory-efficient mode"
        )
        self.preset_large_btn.clicked.connect(lambda: self._apply_preset("large"))
        preset_buttons.addWidget(self.preset_large_btn)
        
        self.preset_gigapixel_btn = QPushButton("ðŸŒ Gigapixel\n(500+)")
        self.preset_gigapixel_btn.setToolTip(
            "Best for: 500+ images (gigapixel panoramas)\n\n"
            "âš ï¸ RECOMMENDED: Use External Pipelines\n"
            "(COLMAP or HLOC below)\n\n"
            "Built-in settings:\n"
            "• SuperPoint + MAGSAC++\n"
            "• Hierarchical + Bundle Adjust\n"
            "• Very slow but may work"
        )
        self.preset_gigapixel_btn.clicked.connect(lambda: self._apply_preset("gigapixel"))
        self.preset_gigapixel_btn.setStyleSheet("background-color: #fff3cd;")
        preset_buttons.addWidget(self.preset_gigapixel_btn)
        
        presets_layout.addLayout(preset_buttons)
        presets_group.setLayout(presets_layout)
        settings_layout.addWidget(presets_group)
        
        # ============================================================
        # CORE SETTINGS (Simplified)
        # ============================================================
        core_group = QGroupBox("Core Settings")
        core_layout = QVBoxLayout()
        core_layout.setSpacing(4)
        
        # Row 1: Verification + Dedup
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Verification:"))
        self.geo_verify_combo = QComboBox()
        self.geo_verify_combo.addItems([
            "MAGSAC++ (Best)",
            "USAC++",
            "RANSAC",
            "None"
        ])
        self.geo_verify_combo.setToolTip(
            "Outlier rejection method:\n"
            "• MAGSAC++ - Most robust (recommended)\n"
            "• USAC++ - Fast and accurate\n"
            "• RANSAC - Classic, less robust\n"
            "• None - Skip (not recommended)"
        )
        row1.addWidget(self.geo_verify_combo)
        
        self.remove_duplicates_checkbox = QCheckBox("Remove Duplicates")
        self.remove_duplicates_checkbox.setChecked(False)  # OFF by default for panoramas
        self.remove_duplicates_checkbox.setToolTip(
            "Remove duplicate/similar images.\n"
            "• For burst photos: Enable to remove redundant frames\n"
            "• For panoramas: Usually DISABLE - overlapping images are NOT duplicates\n"
            "• Uses perceptual hashing + NCC verification"
        )
        row1.addWidget(self.remove_duplicates_checkbox)
        
        self.duplicate_threshold_spin = QDoubleSpinBox()
        self.duplicate_threshold_spin.setRange(0.85, 0.99)
        self.duplicate_threshold_spin.setValue(0.95)  # Very strict by default
        self.duplicate_threshold_spin.setDecimals(2)
        self.duplicate_threshold_spin.setFixedWidth(55)
        self.duplicate_threshold_spin.setToolTip(
            "Similarity threshold for duplicate detection.\n"
            "• 0.95+ = Only exact duplicates (recommended for panoramas)\n"
            "• 0.90 = Near-identical images\n"
            "• 0.85 = Similar images (for burst photos)"
        )
        row1.addWidget(self.duplicate_threshold_spin)
        row1.addStretch()
        core_layout.addLayout(row1)
        
        # Row 2: Bundle Adjust + Memory
        row2 = QHBoxLayout()
        self.bundle_adjust_checkbox = QCheckBox("Bundle Adjustment")
        self.bundle_adjust_checkbox.setChecked(False)
        self.bundle_adjust_checkbox.setToolTip(
            "Global optimization of camera poses.\n"
            "Improves accuracy but slower.\n"
            "Recommended for 50+ images."
        )
        row2.addWidget(self.bundle_adjust_checkbox)
        
        self.memory_efficient_checkbox = QCheckBox("Memory Efficient")
        self.memory_efficient_checkbox.setChecked(True)
        self.memory_efficient_checkbox.setToolTip("Use lazy loading to reduce RAM usage")
        row2.addWidget(self.memory_efficient_checkbox)
        row2.addStretch()
        core_layout.addLayout(row2)
        
        core_group.setLayout(core_layout)
        settings_layout.addWidget(core_group)
        
        # ============================================================
        # ADVANCED OPTIONS (Collapsed by default in spirit - less prominent)
        # ============================================================
        advanced_group = QGroupBox("Advanced Options")
        advanced_layout = QVBoxLayout()
        advanced_layout.setSpacing(2)
        
        adv_row1 = QHBoxLayout()
        
        # Hidden/less-used options
        self.exhaustive_match_checkbox = QCheckBox("Thorough Match")
        self.exhaustive_match_checkbox.setChecked(True)
        self.exhaustive_match_checkbox.setToolTip("Required for unsorted images (default ON)")
        adv_row1.addWidget(self.exhaustive_match_checkbox)
        
        self.hierarchical_checkbox = QCheckBox("Hierarchical")
        self.hierarchical_checkbox.setChecked(False)
        self.hierarchical_checkbox.setToolTip("Cluster-based matching for 200+ images")
        adv_row1.addWidget(self.hierarchical_checkbox)
        
        self.optimize_alignment_checkbox = QCheckBox("Pre-enhance")
        self.optimize_alignment_checkbox.setChecked(False)
        self.optimize_alignment_checkbox.setToolTip("Enhance images before feature detection")
        adv_row1.addWidget(self.optimize_alignment_checkbox)
        
        adv_row1.addStretch()
        advanced_layout.addLayout(adv_row1)
        
        adv_row2 = QHBoxLayout()
        
        self.ai_post_checkbox = QCheckBox("AI Post-Process")
        self.ai_post_checkbox.setChecked(False)
        self.ai_post_checkbox.setToolTip("Apply AI color/denoise after stitching")
        adv_row2.addWidget(self.ai_post_checkbox)
        
        self.super_res_checkbox = QCheckBox("2x Super-Res")
        self.super_res_checkbox.setChecked(False)
        self.super_res_checkbox.setToolTip("Double resolution (very slow)")
        adv_row2.addWidget(self.super_res_checkbox)
        
        adv_row2.addStretch()
        advanced_layout.addLayout(adv_row2)
        
        advanced_group.setLayout(advanced_layout)
        advanced_group.setStyleSheet("QGroupBox { color: #888; }")
        settings_layout.addWidget(advanced_group)
        
        # Hidden widgets that are still needed by the code but not shown
        # (These maintain compatibility with existing code)
        self.grid_topology_checkbox = QCheckBox()
        self.grid_topology_checkbox.setChecked(False)
        self.grid_topology_checkbox.setVisible(False)
        
        self.enhanced_detect_checkbox = QCheckBox()
        self.enhanced_detect_checkbox.setChecked(False)
        self.enhanced_detect_checkbox.setVisible(False)
        
        self.optimal_coverage_checkbox = QCheckBox()
        self.optimal_coverage_checkbox.setChecked(False)
        self.optimal_coverage_checkbox.setVisible(False)
        
        self.max_coverage_spin = QDoubleSpinBox()
        self.max_coverage_spin.setValue(0.5)
        self.max_coverage_spin.setVisible(False)
        
        self.optimization_level_combo = QComboBox()
        self.optimization_level_combo.addItems(["Light", "Balanced", "Aggressive"])
        self.optimization_level_combo.setCurrentIndex(1)
        self.optimization_level_combo.setVisible(False)
        
        self.ai_denoise_checkbox = QCheckBox()
        self.ai_denoise_checkbox.setChecked(True)
        self.ai_denoise_checkbox.setVisible(False)
        
        self.ai_color_checkbox = QCheckBox()
        self.ai_color_checkbox.setChecked(True)
        self.ai_color_checkbox.setVisible(False)
        
        # External Pipelines - Primary stitching workflows
        external_group = QGroupBox("🚀 Stitching Pipelines")
        external_layout = QVBoxLayout()
        external_layout.setSpacing(8)

        external_info = QLabel(
            "GPU-accelerated stitching pipelines.\n"
            "Click 'Install' to set up, then 'Run' to stitch your images."
        )
        external_info.setStyleSheet("color: #555; font-size: 11px; padding: 4px;")
        external_info.setWordWrap(True)
        external_layout.addWidget(external_info)
        
        # --- COLMAP Row ---
        colmap_row = QHBoxLayout()
        colmap_row.setSpacing(6)
        
        self.colmap_status = QLabel("?")
        self.colmap_status.setFixedWidth(20)
        colmap_row.addWidget(self.colmap_status)
        
        colmap_info = QLabel("<b>COLMAP</b> - Industry standard SfM (robust matching, bundle adjustment)")
        colmap_info.setStyleSheet("font-size: 11px;")
        colmap_row.addWidget(colmap_info, 1)
        
        self.colmap_install_btn = QPushButton("Install")
        self.colmap_install_btn.setFixedWidth(60)
        self.colmap_install_btn.setToolTip("Automatically install PyCOLMAP with GPU support (if available)")
        self.colmap_install_btn.clicked.connect(self._install_colmap)
        colmap_row.addWidget(self.colmap_install_btn)
        
        self.colmap_btn = QPushButton("Run")
        self.colmap_btn.setFixedWidth(50)
        self.colmap_btn.setToolTip("Run COLMAP pipeline on loaded images")
        self.colmap_btn.clicked.connect(self._run_colmap)
        colmap_row.addWidget(self.colmap_btn)

        self.reblend_btn = QPushButton("Reblend")
        self.reblend_btn.setFixedWidth(60)
        self.reblend_btn.setToolTip("Re-stitch last run with different blend/interpolation (fast, no re-matching)")
        self.reblend_btn.clicked.connect(self._reblend_last)
        self.reblend_btn.setEnabled(False)
        colmap_row.addWidget(self.reblend_btn)

        external_layout.addLayout(colmap_row)

        # Transform type selector for COLMAP
        transform_layout = QHBoxLayout()
        transform_layout.addWidget(QLabel("Transform:"))
        self.colmap_transform_combo = QComboBox()
        self.colmap_transform_combo.addItems([
            "Homography (3D perspective)",
            "Affine (2D planar)"
        ])
        self.colmap_transform_combo.setCurrentIndex(0)  # Default to homography
        self.colmap_transform_combo.setToolTip(
            "Homography: Full 3D perspective (8 DOF) - for photos/3D scenes\n"
            "Affine: 2D planar only (6 DOF) - for flatbed scans"
        )
        transform_layout.addWidget(self.colmap_transform_combo)
        external_layout.addLayout(transform_layout)

        # Blending method selector for COLMAP
        blend_layout = QHBoxLayout()
        blend_layout.addWidget(QLabel("Blending:"))
        self.colmap_blend_combo = QComboBox()
        self.colmap_blend_combo.addItems([
            "Multiband (Recommended)",
            "Feather",
            "AutoStitch",
            "Mosaic",
            "Linear"
        ])
        self.colmap_blend_combo.setCurrentIndex(0)  # Default to multiband
        self.colmap_blend_combo.setToolTip(
            "Multiband: Best quality, Laplacian pyramid blending\n"
            "Feather: Smooth feathering, faster\n"
            "AutoStitch: Puzzle-like selection (edge-distance priority)\n"
            "Mosaic: Puzzle-like selection (center-distance priority)\n"
            "Linear: Simple averaging, fastest"
        )
        blend_layout.addWidget(self.colmap_blend_combo)
        external_layout.addLayout(blend_layout)

        # Quality preset selector for COLMAP (affects feature extraction quality, not matching)
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("Quality Preset:"))
        self.colmap_preset_combo = QComboBox()
        self.colmap_preset_combo.addItems([
            "Original (Full Features - Slowest)",
            "Phase 1 (Optimized - 2x Faster)",
            "Phase 2 (Fast - 3-4x Faster)"
        ])
        self.colmap_preset_combo.setCurrentIndex(0)  # Default to original
        self.colmap_preset_combo.setToolTip(
            "Original: 8192 features, full resolution (slowest, highest quality)\n"
            "Phase 1: 4096 features, optimized settings (2x faster extraction)\n"
            "Phase 2: 2048 features, downsampled (3-4x faster extraction)\n\n"
            "Note: Matching strategy is configured separately below"
        )
        self.colmap_preset_combo.currentIndexChanged.connect(self._on_colmap_preset_changed)
        preset_layout.addWidget(self.colmap_preset_combo)
        external_layout.addLayout(preset_layout)

        # Advanced matching strategy selector (controlled by preset)
        matching_layout = QHBoxLayout()
        matching_layout.addWidget(QLabel("Matching:"))
        self.colmap_matching_combo = QComboBox()
        self.colmap_matching_combo.addItems([
            "Exhaustive (All Pairs)",
            "Sequential (Consecutive)",
            "Neighbor (Flatbed Scans)",
            "Vocabulary Tree (Visual Similarity)",
            "Grid-Aware (Grid Neighbors)"
        ])
        self.colmap_matching_combo.setCurrentIndex(0)
        self.colmap_matching_combo.setToolTip(
            "Exhaustive: Test all N*(N-1)/2 pairs (slow but thorough)\n"
            "Sequential: Match consecutive images only\n"
            "Neighbor: K nearest neighbors + skip connections (fast, ideal for flatbed scans)\n"
            "Vocabulary Tree: Match visually similar images\n"
            "Grid-Aware: Match detected grid neighbors"
        )
        matching_layout.addWidget(self.colmap_matching_combo)

        # Sequential overlap / neighbor K parameter
        matching_layout.addWidget(QLabel("Neighbors:"))
        self.colmap_overlap_spin = QSpinBox()
        self.colmap_overlap_spin.setMinimum(1)
        self.colmap_overlap_spin.setMaximum(100)
        self.colmap_overlap_spin.setValue(20)
        self.colmap_overlap_spin.setToolTip(
            "Number of nearby images to match:\n"
            "• Sequential: consecutive images to check\n"
            "• Neighbor: K nearest neighbors (20-30 recommended for flatbed scans)"
        )
        matching_layout.addWidget(self.colmap_overlap_spin)

        external_layout.addLayout(matching_layout)

        # GPU and thread settings for COLMAP
        perf_layout = QHBoxLayout()

        # Multi-GPU selection
        gpu_group = QVBoxLayout()
        gpu_group.addWidget(QLabel("GPUs:"))
        self.colmap_gpu_list = QListWidget()
        self.colmap_gpu_list.setMaximumHeight(80)
        self.colmap_gpu_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)

        # Detect available GPUs
        detected_gpus = self._detect_available_gpus()

        # Always add Auto and CPU Only options
        auto_item = QListWidgetItem("Auto (Detect Best)")
        self.colmap_gpu_list.addItem(auto_item)

        cpu_item = QListWidgetItem("CPU Only")
        self.colmap_gpu_list.addItem(cpu_item)

        # Add detected GPUs
        if detected_gpus:
            for gpu_info in detected_gpus:
                item = QListWidgetItem(gpu_info)
                self.colmap_gpu_list.addItem(item)
        else:
            # Fallback if detection fails
            for i in range(4):
                item = QListWidgetItem(f"GPU {i} (Not Verified)")
                self.colmap_gpu_list.addItem(item)

        # Select "Auto" by default
        self.colmap_gpu_list.item(0).setSelected(True)
        self.colmap_gpu_list.setToolTip(
            "Select GPUs to use for COLMAP processing\n"
            "Hold Ctrl/Cmd to select multiple GPUs\n"
            "Auto: Automatically detect and use available GPUs\n\n"
            f"Detected: {len(detected_gpus)} GPU(s) available" if detected_gpus else "GPU detection unavailable"
        )
        gpu_group.addWidget(self.colmap_gpu_list)
        perf_layout.addLayout(gpu_group)

        # Thread count and features
        params_group = QVBoxLayout()

        params_group.addWidget(QLabel("CPU Threads:"))
        self.colmap_threads_spin = QSpinBox()
        self.colmap_threads_spin.setMinimum(-1)  # -1 = auto
        self.colmap_threads_spin.setMaximum(128)
        self.colmap_threads_spin.setSpecialValueText("Auto")  # Show "Auto" for -1
        import os
        cpu_count = os.cpu_count() or 4
        self.colmap_threads_spin.setValue(-1)  # Default to auto
        self.colmap_threads_spin.setToolTip(f"Number of CPU threads for feature extraction\n-1/Auto: Use all available threads\n(System has {cpu_count} threads)")
        params_group.addWidget(self.colmap_threads_spin)

        params_group.addWidget(QLabel("Max Features:"))
        self.colmap_features_spin = QSpinBox()
        self.colmap_features_spin.setMinimum(512)
        self.colmap_features_spin.setMaximum(65536)
        self.colmap_features_spin.setValue(2048)  # Reduced from 8192 for faster matching
        self.colmap_features_spin.setSingleStep(512)
        self.colmap_features_spin.setToolTip(
            "Maximum SIFT features per image:\n\n"
            "Recommended by image count:\n"
            "• 2-50 images: 8192-16384 features\n"
            "• 50-200 images: 4096-8192 features\n"
            "• 200-500 images: 2048-4096 features\n"
            "• 500-1000 images: 1024-2048 features\n"
            "• 1000+ images: 512-1024 features\n\n"
            "Higher counts = better alignment but slower.\n"
            "Use Quality Preset dropdown for auto-adjustment."
        )
        params_group.addWidget(self.colmap_features_spin)

        perf_layout.addLayout(params_group)

        external_layout.addLayout(perf_layout)

        # Image filtering controls to reduce overlap
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Image Filtering:"))

        # Minimum inliers filter
        filter_layout.addWidget(QLabel("Min Inliers:"))
        self.colmap_min_inliers_spin = QSpinBox()
        self.colmap_min_inliers_spin.setMinimum(0)  # 0 = disabled
        self.colmap_min_inliers_spin.setMaximum(500)
        self.colmap_min_inliers_spin.setValue(0)  # Default: no filtering
        self.colmap_min_inliers_spin.setSpecialValueText("Off")
        self.colmap_min_inliers_spin.setToolTip(
            "Minimum feature matches required to include an image\n"
            "0/Off: Include all images\n"
            "30-50: Light filtering (recommended for overlap reduction)\n"
            "100+: Aggressive filtering (may exclude valid images)"
        )
        filter_layout.addWidget(self.colmap_min_inliers_spin)

        # Maximum images per cluster
        filter_layout.addWidget(QLabel("Max Images:"))
        self.colmap_max_images_spin = QSpinBox()
        self.colmap_max_images_spin.setMinimum(0)  # 0 = unlimited
        self.colmap_max_images_spin.setMaximum(1000)
        self.colmap_max_images_spin.setValue(0)  # Default: no limit
        self.colmap_max_images_spin.setSpecialValueText("All")
        self.colmap_max_images_spin.setToolTip(
            "Maximum images to include in panorama\n"
            "0/All: Include all registered images\n"
            "Set lower to reduce overlap blur (e.g., 50-100 for large datasets)\n"
            "Images with best quality/coverage are prioritized"
        )
        filter_layout.addWidget(self.colmap_max_images_spin)

        external_layout.addLayout(filter_layout)

        # Alpha/transparency handling for circular scans
        alpha_layout = QHBoxLayout()
        self.colmap_use_source_alpha = QCheckBox("Use source alpha (transparent backgrounds)")
        self.colmap_use_source_alpha.setChecked(False)
        self.colmap_use_source_alpha.setToolTip(
            "Enable for images with transparent backgrounds (e.g., circular scans)\n\n"
            "When CHECKED:\n"
            "  - Uses the image's alpha channel to determine valid pixels\n"
            "  - Transparent areas are excluded from the final panorama\n"
            "  - Best for circular scans or masked images\n\n"
            "When UNCHECKED (default):\n"
            "  - Treats all pixels as valid content\n"
            "  - Black areas are preserved as image content\n"
            "  - Best for rectangular photos"
        )
        alpha_layout.addWidget(self.colmap_use_source_alpha)
        alpha_layout.addStretch()
        external_layout.addLayout(alpha_layout)

        # Border erosion for circular images
        border_layout = QHBoxLayout()
        self.colmap_erode_border = QCheckBox("Remove black borders")
        self.colmap_erode_border.setChecked(True)  # On by default when using alpha
        self.colmap_erode_border.setToolTip(
            "Erode alpha mask to remove thin black borders from circular images\n\n"
            "When CHECKED:\n"
            "  - Shrinks the alpha mask inward by the specified pixels\n"
            "  - Removes dark edge artifacts from circular scans\n"
            "  - Only applies when 'Use source alpha' is enabled\n\n"
            "When UNCHECKED:\n"
            "  - Uses original alpha mask without modification"
        )
        border_layout.addWidget(self.colmap_erode_border)

        border_layout.addWidget(QLabel("Erosion:"))
        self.colmap_border_erosion_spin = QSpinBox()
        self.colmap_border_erosion_spin.setRange(1, 20)
        self.colmap_border_erosion_spin.setValue(5)
        self.colmap_border_erosion_spin.setSuffix(" px")
        self.colmap_border_erosion_spin.setFixedWidth(70)
        self.colmap_border_erosion_spin.setToolTip(
            "Number of pixels to erode from the alpha mask edge\n\n"
            "• 1-3 px: Light erosion (thin borders)\n"
            "• 4-6 px: Medium erosion (typical borders)\n"
            "• 7-10 px: Heavy erosion (thick borders)\n"
            "• 10+ px: Aggressive (use if borders still visible)\n\n"
            "Adjust based on your image resolution and border thickness"
        )
        border_layout.addWidget(self.colmap_border_erosion_spin)
        border_layout.addStretch()
        external_layout.addLayout(border_layout)

        # Duplicate removal option
        dedup_layout = QHBoxLayout()
        self.colmap_remove_duplicates = QCheckBox("Remove duplicates before processing")
        self.colmap_remove_duplicates.setChecked(False)
        self.colmap_remove_duplicates.setToolTip(
            "Scan for and remove duplicate/near-identical images before COLMAP processing\n\n"
            "When CHECKED:\n"
            "  - Uses perceptual hashing and NCC to detect duplicates\n"
            "  - Removes redundant images to speed up processing\n"
            "  - Good for burst photos or datasets with duplicates\n\n"
            "When UNCHECKED (default):\n"
            "  - All images are processed\n"
            "  - Best for curated panorama image sets"
        )
        dedup_layout.addWidget(self.colmap_remove_duplicates)

        # Threshold slider
        dedup_layout.addWidget(QLabel("Threshold:"))
        self.colmap_duplicate_threshold = QSlider(Qt.Orientation.Horizontal)
        self.colmap_duplicate_threshold.setMinimum(60)
        self.colmap_duplicate_threshold.setMaximum(98)
        self.colmap_duplicate_threshold.setValue(92)
        self.colmap_duplicate_threshold.setFixedWidth(100)
        self.colmap_duplicate_threshold.setToolTip(
            "Similarity threshold for duplicate detection (60-98%)\n\n"
            "Higher = stricter (only near-identical images)\n"
            "Lower = more aggressive (similar images removed)\n\n"
            "Recommended: 92% strict, 85% moderate, 75% aggressive\n"
            "Warning: Below 70% may remove panorama overlap images"
        )
        self.colmap_duplicate_threshold_label = QLabel("92%")
        self.colmap_duplicate_threshold.valueChanged.connect(
            lambda v: self.colmap_duplicate_threshold_label.setText(f"{v}%")
        )
        dedup_layout.addWidget(self.colmap_duplicate_threshold)
        dedup_layout.addWidget(self.colmap_duplicate_threshold_label)
        dedup_layout.addStretch()
        external_layout.addLayout(dedup_layout)

        # --- Warp Interpolation Row ---
        interp_layout = QHBoxLayout()
        interp_layout.setSpacing(6)

        interp_layout.addWidget(QLabel("Warp Interpolation:"))
        self.colmap_warp_interpolation = QComboBox()
        self.colmap_warp_interpolation.addItems(['Linear', 'Cubic', 'Lanczos', 'RealESRGAN'])
        self.colmap_warp_interpolation.setCurrentIndex(2)  # Lanczos default
        self.colmap_warp_interpolation.setFixedWidth(120)
        self.colmap_warp_interpolation.setToolTip(
            "Interpolation method for warping images during stitching:\n\n"
            "Linear (Fast):\n"
            "  - Bilinear interpolation\n"
            "  - Fastest but may appear slightly soft\n"
            "  - Good for quick previews\n\n"
            "Cubic (Balanced):\n"
            "  - Bicubic interpolation\n"
            "  - Good balance of speed and quality\n"
            "  - Sharper than linear\n\n"
            "Lanczos (Recommended):\n"
            "  - Highest quality traditional interpolation\n"
            "  - Best for preserving fine details\n"
            "  - Minimal blur, sharp results\n\n"
            "RealESRGAN (AI Enhanced):\n"
            "  - AI super-resolution enhancement\n"
            "  - Best quality for microscopy/scientific images\n"
            "  - Slower but preserves maximum detail\n"
            "  - Requires realesrgan-ncnn-vulkan installed"
        )
        interp_layout.addWidget(self.colmap_warp_interpolation)
        interp_layout.addStretch()
        external_layout.addLayout(interp_layout)

        # --- HLOC Row ---
        hloc_row = QHBoxLayout()
        hloc_row.setSpacing(6)

        self.hloc_status = QLabel("â¬œ")
        self.hloc_status.setFixedWidth(20)
        hloc_row.addWidget(self.hloc_status)

        hloc_info = QLabel("<b>HLOC</b> - SuperPoint + SuperGlue (best for repetitive scenes)")
        hloc_info.setStyleSheet("font-size: 11px;")
        hloc_row.addWidget(hloc_info, 1)

        self.hloc_install_btn = QPushButton("Install")
        self.hloc_install_btn.setFixedWidth(60)
        self.hloc_install_btn.setToolTip("Install HLOC via pip (requires PyTorch)")
        self.hloc_install_btn.clicked.connect(self._install_hloc)
        hloc_row.addWidget(self.hloc_install_btn)

        self.hloc_btn = QPushButton("Run")
        self.hloc_btn.setFixedWidth(50)
        self.hloc_btn.setToolTip("Run HLOC pipeline on loaded images")
        self.hloc_btn.clicked.connect(self._run_hloc)
        hloc_row.addWidget(self.hloc_btn)

        external_layout.addLayout(hloc_row)

        # --- Meshroom Row (Hidden - 3D scanning, not 2D stitching) ---
        self.meshroom_status = QLabel("â¬œ")
        self.meshroom_install_btn = QPushButton("Install")
        self.meshroom_install_btn.clicked.connect(self._install_meshroom)
        self.meshroom_btn = QPushButton("Run")
        self.meshroom_btn.clicked.connect(self._run_meshroom)

        # Install All button (Hidden)
        self.install_all_btn = QPushButton("ðŸ“¦ Install All Dependencies")
        self.install_all_btn.clicked.connect(self._install_all_pipelines)
        
        # Check availability and update button states
        self._update_external_pipeline_availability()

        external_group.setLayout(external_layout)
        # Store as instance variable - will be added to main layout separately
        self.external_group = external_group

        # ============================================================
        # OUTPUT POST-PROCESSING PANEL (visible, applies to COLMAP output)
        # ============================================================
        postproc_output_group = QGroupBox("🎨 Output Post-Processing")
        postproc_output_layout = QVBoxLayout()
        postproc_output_layout.setSpacing(6)

        # Auto-apply toggle
        auto_row = QHBoxLayout()
        self.auto_postproc_checkbox = QCheckBox("Auto-apply after stitching")
        self.auto_postproc_checkbox.setChecked(False)
        self.auto_postproc_checkbox.setToolTip(
            "Automatically apply post-processing after COLMAP completes.\n"
            "If unchecked, use 'Apply Post-Processing' button manually."
        )
        auto_row.addWidget(self.auto_postproc_checkbox)
        auto_row.addStretch()
        postproc_output_layout.addLayout(auto_row)

        # Row 1: Basic corrections
        basic_row = QHBoxLayout()

        self.pp_sharpen_checkbox = QCheckBox("Sharpen")
        self.pp_sharpen_checkbox.setToolTip("Enhance details with unsharp mask")
        basic_row.addWidget(self.pp_sharpen_checkbox)

        self.pp_sharpen_spin = QDoubleSpinBox()
        self.pp_sharpen_spin.setRange(0.1, 3.0)
        self.pp_sharpen_spin.setValue(1.0)
        self.pp_sharpen_spin.setSingleStep(0.1)
        self.pp_sharpen_spin.setFixedWidth(55)
        self.pp_sharpen_spin.setToolTip("Sharpening strength (0.5=subtle, 1.0=normal, 2.0=strong)")
        basic_row.addWidget(self.pp_sharpen_spin)

        self.pp_denoise_checkbox = QCheckBox("Denoise")
        self.pp_denoise_checkbox.setToolTip("Remove noise with non-local means")
        basic_row.addWidget(self.pp_denoise_checkbox)

        self.pp_denoise_spin = QSpinBox()
        self.pp_denoise_spin.setRange(1, 20)
        self.pp_denoise_spin.setValue(5)
        self.pp_denoise_spin.setFixedWidth(45)
        self.pp_denoise_spin.setToolTip("Denoise strength (3-5=subtle, 10+=aggressive)")
        basic_row.addWidget(self.pp_denoise_spin)

        basic_row.addStretch()
        postproc_output_layout.addLayout(basic_row)

        # Row 2: Contrast and color
        contrast_row = QHBoxLayout()

        self.pp_clahe_checkbox = QCheckBox("Contrast (CLAHE)")
        self.pp_clahe_checkbox.setToolTip("Adaptive histogram equalization for local contrast")
        contrast_row.addWidget(self.pp_clahe_checkbox)

        self.pp_clahe_spin = QDoubleSpinBox()
        self.pp_clahe_spin.setRange(1.0, 5.0)
        self.pp_clahe_spin.setValue(2.0)
        self.pp_clahe_spin.setSingleStep(0.5)
        self.pp_clahe_spin.setFixedWidth(55)
        self.pp_clahe_spin.setToolTip("CLAHE clip limit (1.0=subtle, 2.0=normal, 4.0=strong)")
        contrast_row.addWidget(self.pp_clahe_spin)

        self.pp_shadow_checkbox = QCheckBox("Fix Exposure")
        self.pp_shadow_checkbox.setToolTip("Equalize exposure across the panorama")
        contrast_row.addWidget(self.pp_shadow_checkbox)

        self.pp_shadow_spin = QDoubleSpinBox()
        self.pp_shadow_spin.setRange(0.1, 1.0)
        self.pp_shadow_spin.setValue(0.5)
        self.pp_shadow_spin.setSingleStep(0.1)
        self.pp_shadow_spin.setFixedWidth(55)
        self.pp_shadow_spin.setToolTip("Exposure fix strength (0.3=subtle, 0.5=moderate, 0.8=strong)")
        contrast_row.addWidget(self.pp_shadow_spin)

        contrast_row.addStretch()
        postproc_output_layout.addLayout(contrast_row)

        # Row 3: Advanced corrections
        advanced_row = QHBoxLayout()

        self.pp_deblur_checkbox = QCheckBox("Deblur")
        self.pp_deblur_checkbox.setToolTip("Wiener deconvolution to reduce blur")
        advanced_row.addWidget(self.pp_deblur_checkbox)

        self.pp_deblur_spin = QDoubleSpinBox()
        self.pp_deblur_spin.setRange(0.5, 5.0)
        self.pp_deblur_spin.setValue(1.5)
        self.pp_deblur_spin.setSingleStep(0.5)
        self.pp_deblur_spin.setFixedWidth(55)
        self.pp_deblur_spin.setToolTip("Estimated blur radius")
        advanced_row.addWidget(self.pp_deblur_spin)

        self.pp_white_balance_checkbox = QCheckBox("White Balance")
        self.pp_white_balance_checkbox.setToolTip("Auto white balance correction")
        advanced_row.addWidget(self.pp_white_balance_checkbox)

        self.pp_vignette_checkbox = QCheckBox("Remove Vignette")
        self.pp_vignette_checkbox.setToolTip("Remove lens vignetting (dark corners)")
        advanced_row.addWidget(self.pp_vignette_checkbox)

        advanced_row.addStretch()
        postproc_output_layout.addLayout(advanced_row)

        # Row 4: AI enhancements and upscale
        ai_row = QHBoxLayout()

        self.pp_ai_color_checkbox = QCheckBox("AI Color")
        self.pp_ai_color_checkbox.setToolTip("AI-powered automatic color correction")
        ai_row.addWidget(self.pp_ai_color_checkbox)

        self.pp_ai_denoise_checkbox = QCheckBox("AI Denoise")
        self.pp_ai_denoise_checkbox.setToolTip("AI-powered noise reduction")
        ai_row.addWidget(self.pp_ai_denoise_checkbox)

        self.pp_super_res_checkbox = QCheckBox("Super Res (AI)")
        self.pp_super_res_checkbox.setToolTip(
            "Real-ESRGAN AI super-resolution (2x or 4x)\n\n"
            "• Much better quality than Lanczos upscaling\n"
            "• Requires: pip install realesrgan\n"
            "• GPU recommended (slow on CPU)\n"
            "• Auto-downloads model on first use (~64MB)"
        )
        ai_row.addWidget(self.pp_super_res_checkbox)

        self.pp_super_res_scale_combo = QComboBox()
        self.pp_super_res_scale_combo.addItems(["2x", "4x"])
        self.pp_super_res_scale_combo.setToolTip("Super-resolution scale factor")
        self.pp_super_res_scale_combo.setFixedWidth(50)
        ai_row.addWidget(self.pp_super_res_scale_combo)

        ai_row.addWidget(QLabel("Lanczos:"))
        self.pp_upscale_combo = QComboBox()
        self.pp_upscale_combo.addItems(["1x", "1.5x", "2x", "3x", "4x"])
        self.pp_upscale_combo.setToolTip(
            "Traditional Lanczos upscaling (fast)\n"
            "Use INSTEAD of AI Super Res for speed"
        )
        self.pp_upscale_combo.setFixedWidth(55)
        ai_row.addWidget(self.pp_upscale_combo)

        ai_row.addStretch()
        postproc_output_layout.addLayout(ai_row)

        # Row 5: Action buttons
        button_row = QHBoxLayout()

        self.apply_postproc_btn = QPushButton("Apply Post-Processing")
        self.apply_postproc_btn.setToolTip("Apply post-processing to current result")
        self.apply_postproc_btn.clicked.connect(self._apply_postprocessing)
        self.apply_postproc_btn.setEnabled(False)  # Enabled when result exists
        button_row.addWidget(self.apply_postproc_btn)

        self.reset_postproc_btn = QPushButton("Reset to Original")
        self.reset_postproc_btn.setToolTip("Revert to original panorama (before post-processing)")
        self.reset_postproc_btn.clicked.connect(self._reset_postprocessing)
        self.reset_postproc_btn.setEnabled(False)
        button_row.addWidget(self.reset_postproc_btn)

        button_row.addStretch()
        postproc_output_layout.addLayout(button_row)

        postproc_output_group.setLayout(postproc_output_layout)
        self.postproc_output_group = postproc_output_group

        # Feature matcher
        matcher_layout = QHBoxLayout()
        matcher_layout.addWidget(QLabel("Feature Matcher:"))
        self.matcher_combo = QComboBox()
        self.matcher_combo.addItems([
            "FLANN (Recommended)",
            "LoFTR (Deep Learning)",
            "SuperGlue (Deep Learning)",
            "DISK (Deep Learning)"
        ])
        matcher_layout.addWidget(self.matcher_combo)
        settings_layout.addLayout(matcher_layout)
        
        # Blending method
        blend_layout = QHBoxLayout()
        blend_layout.addWidget(QLabel("Blending Method:"))
        self.blend_combo = QComboBox()
        self.blend_combo.addItems([
            "Multiband (Recommended)",
            "Feather",
            "Linear",
            "Semantic (Foreground-Aware)",
            "PixelStitch (Structure-Preserving)",
            "AutoStitch (Simple & Fast)",
            "Mosaic (Center Priority)",
            "Generative AI (Best Quality)"
        ])
        blend_layout.addWidget(self.blend_combo)
        settings_layout.addLayout(blend_layout)
        
        # Common blending options
        blend_options_group = QGroupBox("Blending Options")
        blend_options_layout = QVBoxLayout()
        
        # Common options row
        common_opts_layout = QHBoxLayout()
        self.hdr_checkbox = QCheckBox("HDR")
        self.hdr_checkbox.setToolTip("Combine multiple exposures for better dynamic range")
        common_opts_layout.addWidget(self.hdr_checkbox)
        
        self.antighost_checkbox = QCheckBox("Anti-Ghost")
        self.antighost_checkbox.setToolTip("Reduce ghosting artifacts in overlapping regions")
        common_opts_layout.addWidget(self.antighost_checkbox)
        common_opts_layout.addStretch()
        blend_options_layout.addLayout(common_opts_layout)
        
        # Overlap handling (unified for all methods)
        overlap_mode_layout = QHBoxLayout()
        overlap_mode_layout.addWidget(QLabel("Overlap Handling:"))
        self.semantic_pixel_combo = QComboBox()
        self.semantic_pixel_combo.addItems([
            "Blend (Smooth)",
            "Select (Crisp)",
            "Pairwise (Burst Fix)"
        ])
        self.semantic_pixel_combo.setToolTip(
            "How overlapping pixels are combined:\n"
            "• Blend: Weighted average (smooth transitions)\n"
            "• Select: Winner-take-all (crisp, no blur)\n"
            "• Pairwise: Best 2 only (fixes burst photo repetition)"
        )
        overlap_mode_layout.addWidget(self.semantic_pixel_combo)
        
        overlap_mode_layout.addWidget(QLabel("Padding:"))
        self.canvas_padding_spin = QSpinBox()
        self.canvas_padding_spin.setRange(0, 500)
        self.canvas_padding_spin.setValue(50)
        self.canvas_padding_spin.setSuffix("px")
        self.canvas_padding_spin.setToolTip("Extra canvas padding")
        overlap_mode_layout.addWidget(self.canvas_padding_spin)
        blend_options_layout.addLayout(overlap_mode_layout)
        
        # Hidden pixel_select_combo for backwards compatibility (uses same value as semantic_pixel_combo)
        self.pixel_select_combo = QComboBox()
        self.pixel_select_combo.addItems(["Weighted Average (Default)"])
        self.pixel_select_combo.hide()
        
        blend_options_group.setLayout(blend_options_layout)
        settings_layout.addWidget(blend_options_group)

        # Alpha Channel & Border Handling (for AutoStitch edge fix)
        alpha_group = QGroupBox("Alpha Channel & Border Handling")
        alpha_layout = QVBoxLayout()

        # Enable alpha channels checkbox
        self.create_alpha_checkbox = QCheckBox("Create alpha channels for warped images")
        self.create_alpha_checkbox.setChecked(True)  # On by default
        self.create_alpha_checkbox.setToolTip(
            "Create alpha channels to mark transformation borders\n\n"
            "FIXES AUTOSTITCH EDGE ISSUES:\n"
            "• Prevents black transformation borders from appearing in output\n"
            "• Eliminates edge misalignment and duplication\n"
            "• Essential for AutoStitch blending method\n\n"
            "Recommended: Keep ON (especially with AutoStitch)"
        )
        alpha_layout.addWidget(self.create_alpha_checkbox)

        # Auto-detect circular images checkbox
        self.auto_detect_circular_checkbox = QCheckBox("Auto-detect circular/round images")
        self.auto_detect_circular_checkbox.setChecked(True)  # On by default
        self.auto_detect_circular_checkbox.setToolTip(
            "Automatically detect and mask circular/round images\n\n"
            "Useful for:\n"
            "• Microscope images with circular apertures\n"
            "• Round scans with dark borders\n"
            "• Images with non-rectangular content\n\n"
            "Creates smooth geometric masks using ellipse fitting"
        )
        alpha_layout.addWidget(self.auto_detect_circular_checkbox)

        # Border erosion control
        border_erosion_layout = QHBoxLayout()
        border_erosion_layout.addWidget(QLabel("Border erosion:"))
        self.border_erosion_spin = QSpinBox()
        self.border_erosion_spin.setRange(0, 20)
        self.border_erosion_spin.setValue(5)
        self.border_erosion_spin.setSuffix(" px")
        self.border_erosion_spin.setFixedWidth(70)
        self.border_erosion_spin.setToolTip(
            "Pixels to erode from alpha mask edges\n\n"
            "Removes thin dark borders from image edges:\n"
            "• 0 px: No erosion\n"
            "• 1-3 px: Light erosion (thin borders)\n"
            "• 4-6 px: Medium erosion (typical borders)\n"
            "• 7-10 px: Heavy erosion (thick borders)\n"
            "• 10+ px: Aggressive (use if borders still visible)\n\n"
            "Adjust based on your image resolution and border thickness"
        )
        border_erosion_layout.addWidget(self.border_erosion_spin)
        border_erosion_layout.addStretch()
        alpha_layout.addLayout(border_erosion_layout)

        alpha_group.setLayout(alpha_layout)
        settings_layout.addWidget(alpha_group)

        # Semantic blending options (shown only when Semantic is selected)
        self.semantic_options_group = QGroupBox("Semantic Options")
        semantic_layout = QVBoxLayout()
        
        semantic_mode_layout = QHBoxLayout()
        semantic_mode_layout.addWidget(QLabel("Mode:"))
        self.semantic_mode_combo = QComboBox()
        self.semantic_mode_combo.addItems([
            "Auto",
            "SAM (Segment Anything)",
            "DeepLab (Neural Net)",
            "Hybrid (Microscopy)",
            "Texture",
            "Edge",
            "Superpixel"
        ])
        self.semantic_mode_combo.setToolTip(
            "• Auto: Best available method\n"
            "• SAM: Works on any image\n"
            "• DeepLab: Scenes, people, objects\n"
            "• Hybrid: Best for microscopy/geology\n"
            "• Texture/Edge/Superpixel: Classical methods"
        )
        semantic_mode_layout.addWidget(self.semantic_mode_combo)
        semantic_layout.addLayout(semantic_mode_layout)
        
        self.preserve_foreground_checkbox = QCheckBox("Preserve Foreground Objects")
        self.preserve_foreground_checkbox.setChecked(True)
        self.preserve_foreground_checkbox.setToolTip("Prioritize detected foreground objects")
        semantic_layout.addWidget(self.preserve_foreground_checkbox)
        
        self.semantic_options_group.setLayout(semantic_layout)
        self.semantic_options_group.hide()  # Hidden by default
        settings_layout.addWidget(self.semantic_options_group)
        
        # Generative AI options (shown only when Generative AI is selected)
        self.gen_ai_group = QGroupBox("Generative AI Options")
        gen_ai_layout = QVBoxLayout()
        
        gen_backend_layout = QHBoxLayout()
        gen_backend_layout.addWidget(QLabel("Backend:"))
        self.gen_backend_combo = QComboBox()
        self.gen_backend_combo.addItems([
            "Hybrid (Free)",
            "OpenAI (Best)",
            "Local (GPU)",
            "Replicate"
        ])
        self.gen_backend_combo.setToolTip(
            "• Hybrid: Traditional + AI seam repair (free)\n"
            "• OpenAI: DALL-E 3 (~$0.04/edit)\n"
            "• Local: Stable Diffusion (needs 8GB VRAM)\n"
            "• Replicate: Cloud GPU (pay-per-use)"
        )
        gen_backend_layout.addWidget(self.gen_backend_combo)
        gen_ai_layout.addLayout(gen_backend_layout)
        
        api_key_layout = QHBoxLayout()
        api_key_layout.addWidget(QLabel("API Key:"))
        self.gen_api_key_input = QLineEdit()
        self.gen_api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.gen_api_key_input.setPlaceholderText("OpenAI/Replicate key...")
        api_key_layout.addWidget(self.gen_api_key_input)
        gen_ai_layout.addLayout(api_key_layout)
        
        strength_layout = QHBoxLayout()
        strength_layout.addWidget(QLabel("Strength:"))
        self.gen_strength_slider = QSlider(Qt.Orientation.Horizontal)
        self.gen_strength_slider.setRange(1, 100)
        self.gen_strength_slider.setValue(75)
        strength_layout.addWidget(self.gen_strength_slider)
        self.gen_strength_label = QLabel("75%")
        self.gen_strength_slider.valueChanged.connect(
            lambda v: self.gen_strength_label.setText(f"{v}%")
        )
        strength_layout.addWidget(self.gen_strength_label)
        gen_ai_layout.addLayout(strength_layout)
        
        self.gen_ai_group.setLayout(gen_ai_layout)
        self.gen_ai_group.hide()  # Hidden by default
        settings_layout.addWidget(self.gen_ai_group)
        
        # Connect blending method change to show/hide options
        self.blend_combo.currentTextChanged.connect(self._on_blend_method_changed)
        
        # Max images
        max_img_layout = QHBoxLayout()
        max_img_layout.addWidget(QLabel("Max Images:"))
        self.max_images_spin = QSpinBox()
        self.max_images_spin.setRange(0, 10000)
        self.max_images_spin.setValue(0)
        self.max_images_spin.setSpecialValueText("Unlimited")
        max_img_layout.addWidget(self.max_images_spin)
        settings_layout.addLayout(max_img_layout)
        
        # Grid/alignment settings (collapsible)
        grid_group = QGroupBox("Grid Alignment")
        grid_layout = QVBoxLayout()
        
        overlap_layout = QHBoxLayout()
        overlap_layout.addWidget(QLabel("Min Overlap:"))
        self.overlap_spin = QDoubleSpinBox()
        self.overlap_spin.setRange(0.0, 100.0)
        self.overlap_spin.setValue(10.0)
        self.overlap_spin.setSuffix("%")
        self.overlap_spin.setToolTip("Minimum overlap % for grid alignment")
        overlap_layout.addWidget(self.overlap_spin)
        
        overlap_layout.addWidget(QLabel("Spacing:"))
        self.grid_spacing_spin = QDoubleSpinBox()
        self.grid_spacing_spin.setRange(1.0, 3.0)
        self.grid_spacing_spin.setValue(1.3)
        self.grid_spacing_spin.setToolTip("Grid spacing (1.0=touching, 1.5=50% gap)")
        overlap_layout.addWidget(self.grid_spacing_spin)
        grid_layout.addLayout(overlap_layout)
        
        # Hidden max_overlap_spin for backward compatibility
        self.max_overlap_spin = QDoubleSpinBox()
        self.max_overlap_spin.setValue(100.0)
        self.max_overlap_spin.hide()
        
        grid_group.setLayout(grid_layout)
        settings_layout.addWidget(grid_group)
        
        # Output settings group
        output_group = QGroupBox("Output Settings")
        output_layout = QVBoxLayout()
        
        # Output quality
        quality_layout = QHBoxLayout()
        quality_layout.addWidget(QLabel("Quality:"))
        self.output_quality_combo = QComboBox()
        self.output_quality_combo.addItems([
            "Ultra High (Uncompressed)",
            "High (Lossless)",
            "Medium (Balanced)",
            "Low (Compressed)",
            "Minimum (Most Compressed)"
        ])
        self.output_quality_combo.setCurrentIndex(1)  # Default to High
        self.output_quality_combo.setToolTip(
            "Ultra High: No compression, largest file size\n"
            "High: Lossless compression (PNG/TIFF LZW)\n"
            "Medium: Balanced quality and size\n"
            "Low: More compression, smaller files\n"
            "Minimum: Maximum compression, smallest files"
        )
        quality_layout.addWidget(self.output_quality_combo)
        output_layout.addLayout(quality_layout)
        
        # Size limits row
        limits_layout = QHBoxLayout()
        limits_layout.addWidget(QLabel("DPI:"))
        self.output_dpi_spin = QSpinBox()
        self.output_dpi_spin.setRange(72, 1200)
        self.output_dpi_spin.setValue(300)
        self.output_dpi_spin.setToolTip("72=screen, 300=print")
        limits_layout.addWidget(self.output_dpi_spin)
        
        limits_layout.addWidget(QLabel("Max:"))
        self.max_panorama_spin = QSpinBox()
        self.max_panorama_spin.setRange(0, 2000)
        self.max_panorama_spin.setValue(500)  # 500MP = ~22000x22000 (generous gigapixel)
        self.max_panorama_spin.setSuffix("MP")
        self.max_panorama_spin.setSpecialValueText("âˆž (No Limit)")
        self.max_panorama_spin.setToolTip(
            "Maximum final panorama size in megapixels.\n"
            "500MP = ~22000x22000 pixels (default)\n"
            "0 = Unlimited (warning: can use huge RAM!)\n"
            "Increase for larger gigapixel output."
        )
        limits_layout.addWidget(self.max_panorama_spin)
        
        self.scale_to_target_checkbox = QCheckBox("Fill")
        self.scale_to_target_checkbox.setChecked(True)  # Default: scale to fill target
        self.scale_to_target_checkbox.setToolTip(
            "Scale output to fill target size.\n"
            "ON: Output will be scaled UP or DOWN to match Max MP\n"
            "OFF: Only scale down if exceeds limit"
        )
        limits_layout.addWidget(self.scale_to_target_checkbox)
        output_layout.addLayout(limits_layout)
        
        # Warp limit on separate row (less commonly used)
        warp_layout = QHBoxLayout()
        warp_layout.addWidget(QLabel("Warp Limit:"))
        self.max_warp_spin = QSpinBox()
        self.max_warp_spin.setRange(0, 500)
        self.max_warp_spin.setValue(0)  # Let blender handle scaling globally
        self.max_warp_spin.setSuffix("MP")
        self.max_warp_spin.setSpecialValueText("âˆž (Full Res)")
        self.max_warp_spin.setToolTip(
            "Maximum size for individual warped images.\n"
            "0 = No limit (recommended for quality)\n"
            "Set a limit only if running out of RAM during alignment."
        )
        warp_layout.addWidget(self.max_warp_spin)
        warp_layout.addStretch()
        output_layout.addLayout(warp_layout)
        
        output_group.setLayout(output_layout)
        settings_layout.addWidget(output_group)
        
        # Post-processing options
        postproc_group = QGroupBox("Post-Processing")
        postproc_layout = QVBoxLayout()
        
        # Checkboxes row
        postproc_checks = QHBoxLayout()
        self.sharpen_checkbox = QCheckBox("Sharpen")
        self.sharpen_checkbox.setToolTip("Enhance details")
        postproc_checks.addWidget(self.sharpen_checkbox)
        
        self.denoise_checkbox = QCheckBox("Denoise")
        self.denoise_checkbox.setToolTip("Remove noise")
        postproc_checks.addWidget(self.denoise_checkbox)
        
        self.shadow_checkbox = QCheckBox("Fix Exposure")
        self.shadow_checkbox.setToolTip("Equalize exposure across image")
        postproc_checks.addWidget(self.shadow_checkbox)
        postproc_checks.addStretch()
        postproc_layout.addLayout(postproc_checks)
        
        # Strength controls row
        strength_layout = QHBoxLayout()
        strength_layout.addWidget(QLabel("Sharpen:"))
        self.sharpen_amount_spin = QDoubleSpinBox()
        self.sharpen_amount_spin.setRange(0.0, 3.0)
        self.sharpen_amount_spin.setValue(1.0)
        strength_layout.addWidget(self.sharpen_amount_spin)
        
        strength_layout.addWidget(QLabel("Denoise:"))
        self.denoise_strength_spin = QSpinBox()
        self.denoise_strength_spin.setRange(1, 20)
        self.denoise_strength_spin.setValue(5)
        strength_layout.addWidget(self.denoise_strength_spin)
        postproc_layout.addLayout(strength_layout)
        
        # Additional options row
        extra_checks = QHBoxLayout()
        self.clahe_checkbox = QCheckBox("Contrast")
        self.clahe_checkbox.setToolTip("CLAHE adaptive contrast enhancement")
        extra_checks.addWidget(self.clahe_checkbox)
        
        self.deblur_checkbox = QCheckBox("Deblur")
        self.deblur_checkbox.setToolTip("Wiener deconvolution")
        extra_checks.addWidget(self.deblur_checkbox)
        
        extra_checks.addWidget(QLabel("Upscale:"))
        self.upscale_combo = QComboBox()
        self.upscale_combo.addItems(["1x", "1.5x", "2x", "3x", "4x"])
        self.upscale_combo.setToolTip("Lanczos upscaling")
        extra_checks.addWidget(self.upscale_combo)
        postproc_layout.addLayout(extra_checks)
        
        # Hidden controls for backwards compatibility
        self.shadow_strength_spin = QDoubleSpinBox()
        self.shadow_strength_spin.setValue(0.5)
        self.shadow_strength_spin.hide()
        
        self.clahe_strength_spin = QDoubleSpinBox()
        self.clahe_strength_spin.setValue(2.0)
        self.clahe_strength_spin.hide()
        
        self.deblur_strength_spin = QDoubleSpinBox()
        self.deblur_strength_spin.setValue(1.5)
        self.deblur_strength_spin.hide()
        
        self.interp_combo = QComboBox()
        self.interp_combo.addItems(["Lanczos (Best)", "Cubic (High)", "Linear (Fast)", "Nearest (Fastest)"])
        self.interp_combo.hide()  # Hidden - Lanczos always used
        
        postproc_group.setLayout(postproc_layout)
        settings_layout.addWidget(postproc_group)
        
        # Connect post-processing controls to live preview update
        self.sharpen_checkbox.stateChanged.connect(self.update_postproc_preview)
        self.sharpen_amount_spin.valueChanged.connect(self.update_postproc_preview)
        self.denoise_checkbox.stateChanged.connect(self.update_postproc_preview)
        self.denoise_strength_spin.valueChanged.connect(self.update_postproc_preview)
        self.shadow_checkbox.stateChanged.connect(self.update_postproc_preview)
        self.shadow_strength_spin.valueChanged.connect(self.update_postproc_preview)
        self.clahe_checkbox.stateChanged.connect(self.update_postproc_preview)
        self.clahe_strength_spin.valueChanged.connect(self.update_postproc_preview)
        self.deblur_checkbox.stateChanged.connect(self.update_postproc_preview)
        self.deblur_strength_spin.valueChanged.connect(self.update_postproc_preview)
        # Note: upscale not in live preview (too slow for large images)
        
        settings_group.setLayout(settings_layout)

        # Add external pipelines group BEFORE settings (it was moved out of settings_group)
        layout.addWidget(self.external_group)

        # Add post-processing group after external pipelines
        layout.addWidget(self.postproc_output_group)

        # Legacy settings group (hidden by default - see end of function)
        layout.addWidget(settings_group)
        
        # Processing
        process_group = QGroupBox("Processing")
        process_layout = QVBoxLayout()
        
        self.btn_grid = QPushButton("Create Grid Alignment")
        self.btn_grid.clicked.connect(self.create_grid)
        process_layout.addWidget(self.btn_grid)
        
        self.btn_stitch = QPushButton("Stitch Images")
        self.btn_stitch.clicked.connect(self.start_stitching)
        process_layout.addWidget(self.btn_stitch)
        
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self.stop_processing)
        self.btn_stop.setEnabled(False)
        process_layout.addWidget(self.btn_stop)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setVisible(False)
        process_layout.addWidget(self.progress_bar)
        
        self.progress_label = QLabel("")
        self.progress_label.setVisible(False)
        process_layout.addWidget(self.progress_label)
        
        process_group.setLayout(process_layout)
        layout.addWidget(process_group)

        layout.addStretch()

        # ============================================================
        # HIDE LEGACY/NON-PIPELINE FEATURES
        # The COLMAP pipeline is now the primary workflow.
        # Legacy internal stitching features are hidden but preserved
        # for backward compatibility with existing code.
        # ============================================================

        # Hide legacy settings (kept visible: Image Selection, External Pipelines, Processing)
        settings_group.setVisible(False)  # Quality, GPU, Feature Detector, etc.

        # Hide legacy processing buttons (keep Stop button visible via process_group)
        self.btn_grid.setVisible(False)  # Grid Alignment button
        self.btn_stitch.setVisible(False)  # Legacy Stitch button

        # Rename process group to be clearer
        process_group.setTitle("Controls")

        # Set panel as scroll area content
        scroll_area.setWidget(panel)
        return scroll_area
    
    def create_preview_panel(self) -> QWidget:
        """Create preview panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Tabs
        tabs = QTabWidget()
        
        # Preview tab
        preview_tab = QWidget()
        preview_layout = QVBoxLayout(preview_tab)
        
        # Preview controls - zoom and clear
        preview_controls = QHBoxLayout()
        
        # Zoom controls
        zoom_label = QLabel("Zoom:")
        preview_controls.addWidget(zoom_label)
        
        self.btn_zoom_out = QPushButton("âˆ’")
        self.btn_zoom_out.setFixedWidth(30)
        self.btn_zoom_out.setToolTip("Zoom out (Ctrl+-)")
        self.btn_zoom_out.clicked.connect(self.zoom_out)
        preview_controls.addWidget(self.btn_zoom_out)
        
        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(10, 400)  # 10% to 400%
        self.zoom_slider.setValue(100)  # Default 100%
        self.zoom_slider.setFixedWidth(150)
        self.zoom_slider.setToolTip("Drag to adjust zoom level")
        self.zoom_slider.valueChanged.connect(self.on_zoom_changed)
        preview_controls.addWidget(self.zoom_slider)
        
        self.btn_zoom_in = QPushButton("+")
        self.btn_zoom_in.setFixedWidth(30)
        self.btn_zoom_in.setToolTip("Zoom in (Ctrl++)")
        self.btn_zoom_in.clicked.connect(self.zoom_in)
        preview_controls.addWidget(self.btn_zoom_in)
        
        self.zoom_percent_label = QLabel("100%")
        self.zoom_percent_label.setFixedWidth(45)
        preview_controls.addWidget(self.zoom_percent_label)
        
        self.btn_zoom_fit = QPushButton("Fit")
        self.btn_zoom_fit.setToolTip("Fit image to view (Ctrl+0)")
        self.btn_zoom_fit.clicked.connect(self.zoom_fit)
        preview_controls.addWidget(self.btn_zoom_fit)
        
        self.btn_zoom_100 = QPushButton("1:1")
        self.btn_zoom_100.setToolTip("Actual size (Ctrl+1)")
        self.btn_zoom_100.clicked.connect(self.zoom_actual)
        preview_controls.addWidget(self.btn_zoom_100)
        
        preview_controls.addStretch()
        
        self.btn_clear_preview = QPushButton("Clear Preview")
        self.btn_clear_preview.clicked.connect(self.clear_preview)
        preview_controls.addWidget(self.btn_clear_preview)
        
        preview_layout.addLayout(preview_controls)
        
        # Preview area with scroll support
        self.preview_scroll = QScrollArea()
        self.preview_scroll.setWidgetResizable(False)  # We control size manually
        self.preview_scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_scroll.setStyleSheet("border: 1px solid gray; background-color: #2d2d2d;")
        
        self.preview_label = QLabel("No preview available")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(800, 600)
        self.preview_label.setStyleSheet("background-color: transparent;")
        self.preview_scroll.setWidget(self.preview_label)
        
        preview_layout.addWidget(self.preview_scroll)
        
        tabs.addTab(preview_tab, "Preview")
        
        # Logs tab
        logs_tab = QWidget()
        logs_layout = QVBoxLayout(logs_tab)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Segoe UI Emoji", 9))  # Emoji-capable font
        logs_layout.addWidget(self.log_text)
        
        tabs.addTab(logs_tab, "Logs")
        
        layout.addWidget(tabs)
        
        # Output
        output_group = QGroupBox("Output")
        output_layout = QHBoxLayout()
        
        self.output_path_label = QLabel("Output: Not set")
        output_layout.addWidget(self.output_path_label)
        
        self.btn_browse_output = QPushButton("Browse...")
        self.btn_browse_output.clicked.connect(self.browse_output)
        output_layout.addWidget(self.btn_browse_output)
        
        self.btn_save = QPushButton("Save Result")
        self.btn_save.clicked.connect(self.save_result)
        self.btn_save.setEnabled(False)
        output_layout.addWidget(self.btn_save)
        
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)
        
        self.output_path: Optional[Path] = None
        self.current_result = None
        self.original_result = None  # Store original for post-processing reset

        return panel
    
    def init_stitcher(self):
        """Initialize stitcher with current settings"""
        try:
            use_gpu = self.gpu_checkbox.isChecked()
            quality_threshold = self.quality_spin.value()
            max_images = self.max_images_spin.value()
            if max_images == 0:
                max_images = None
            
            # Get algorithm selections
            detector_map = {
                "LP-SIFT (Recommended)": "lp_sift",
                "SuperPoint (Deep Learning)": "superpoint",
                "SIFT": "sift",
                "ORB": "orb",
                "AKAZE": "akaze"
            }
            matcher_map = {
                "FLANN (Recommended)": "flann",
                "LoFTR (Deep Learning)": "loftr",
                "SuperGlue (Deep Learning)": "superglue",
                "DISK (Deep Learning)": "disk"
            }
            blender_map = {
                "Multiband (Recommended)": "multiband",
                "Feather": "feather",
                "Linear": "linear",
                "Semantic (Foreground-Aware)": "semantic",
                "PixelStitch (Structure-Preserving)": "pixelstitch",
                "AutoStitch (Simple & Fast)": "autostitch",
                "Mosaic (Center Priority)": "mosaic",
                "Generative AI (Best Quality)": "generative"
            }
            
            feature_detector = detector_map.get(self.detector_combo.currentText(), "lp_sift")
            feature_matcher = matcher_map.get(self.matcher_combo.currentText(), "flann")
            blending_method = blender_map.get(self.blend_combo.currentText(), "multiband")
            max_features = self.max_features_spin.value()
            
            # Map semantic mode selection
            semantic_mode_map = {
                "Auto": "auto",
                "SAM (Segment Anything)": "sam",
                "DeepLab (Neural Net)": "deeplab",
                "Hybrid (Microscopy)": "hybrid",
                "Texture": "texture",
                "Edge": "edge",
                "Superpixel": "superpixel"
            }
            semantic_mode = semantic_mode_map.get(self.semantic_mode_combo.currentText(), "auto")
            
            # Map generative AI backend selection
            gen_backend_map = {
                "Hybrid (Free)": "hybrid",
                "OpenAI (Best)": "openai",
                "Local (GPU)": "local",
                "Replicate": "replicate"
            }
            gen_backend = gen_backend_map.get(self.gen_backend_combo.currentText(), "hybrid")
            
            # Get blending options
            blending_options = {
                'hdr_mode': self.hdr_checkbox.isChecked(),
                'anti_ghosting': self.antighost_checkbox.isChecked(),
                'pixel_selection': self.pixel_select_combo.currentText().split('(')[0].strip().lower().replace(' ', '_'),
                'padding': self.canvas_padding_spin.value(),
                'fit_all': True,
                'scale_to_target': self.scale_to_target_checkbox.isChecked(),  # Scale to fill target size
                # Semantic blending options
                'semantic_mode': semantic_mode,
                'preserve_foreground': self.preserve_foreground_checkbox.isChecked(),
                # Overlap pixel selection mode
                'semantic_pixel_selection': ['blend', 'select', 'pairwise'][self.semantic_pixel_combo.currentIndex()],
                # Generative AI options
                'gen_backend': gen_backend,
                'gen_api_key': self.gen_api_key_input.text() or None,
                'gen_strength': self.gen_strength_slider.value() / 100.0,
                # AI post-processing options (AutoPano Giga-style)
                'ai_post_processing': self.ai_post_checkbox.isChecked(),
                'ai_denoise': self.ai_denoise_checkbox.isChecked(),
                'ai_color_correct': self.ai_color_checkbox.isChecked(),
                'super_resolution': self.super_res_checkbox.isChecked()
            }
            
            allow_scale = self.allow_scale_checkbox.isChecked()
            
            # Get memory limit settings (convert MP to pixels, 0 = None for unlimited)
            max_panorama_mp = self.max_panorama_spin.value()
            max_panorama_pixels = max_panorama_mp * 1_000_000 if max_panorama_mp > 0 else None
            
            max_warp_mp = self.max_warp_spin.value()
            max_warp_pixels = max_warp_mp * 1_000_000 if max_warp_mp > 0 else None
            
            memory_efficient = self.memory_efficient_checkbox.isChecked()
            
            # Match filtering options
            geo_method_map = {
                "MAGSAC++ (Best)": "magsac",
                "USAC++": "usac",
                "RANSAC": "ransac",
                "None": None
            }
            geo_method = geo_method_map.get(self.geo_verify_combo.currentText(), "magsac")
            geometric_verify = geo_method is not None
            select_optimal_coverage = self.optimal_coverage_checkbox.isChecked()
            max_coverage_overlap = self.max_coverage_spin.value()
            
            # Duplicate detection options
            remove_duplicates = self.remove_duplicates_checkbox.isChecked()
            duplicate_threshold = self.duplicate_threshold_spin.value()
            
            self.stitcher = ImageStitcher(
                use_gpu=use_gpu,
                quality_threshold=quality_threshold,
                max_images=max_images,
                feature_detector=feature_detector,
                feature_matcher=feature_matcher,
                max_features=max_features,
                blending_method=blending_method,
                blending_options=blending_options,
                allow_scale=allow_scale,
                max_panorama_pixels=max_panorama_pixels,
                max_warp_pixels=max_warp_pixels,
                memory_efficient=memory_efficient,
                geometric_verify=geometric_verify,
                geometric_verify_method=geo_method,  # MAGSAC++, USAC++, etc.
                select_optimal_coverage=select_optimal_coverage,
                max_coverage_overlap=max_coverage_overlap,
                remove_duplicates=remove_duplicates,
                duplicate_threshold=duplicate_threshold,
                optimize_alignment=self.optimize_alignment_checkbox.isChecked(),
                # Alpha channel and border handling (autostitch edge fix)
                create_alpha_channels=self.create_alpha_checkbox.isChecked(),
                auto_detect_circular=self.auto_detect_circular_checkbox.isChecked(),
                border_erosion_pixels=self.border_erosion_spin.value(),
                alignment_optimization_level=self.optimization_level_combo.currentText().lower(),
                # AutoPano Giga-inspired features
                use_grid_topology=self.grid_topology_checkbox.isChecked(),
                use_bundle_adjustment=self.bundle_adjust_checkbox.isChecked(),
                use_hierarchical_stitching=self.hierarchical_checkbox.isChecked(),
                use_enhanced_detection=self.enhanced_detect_checkbox.isChecked(),
                exhaustive_matching=self.exhaustive_match_checkbox.isChecked()
            )
            logger.info(f"Stitcher initialized (detector={feature_detector}, geo_verify={geo_method}, max_panorama={max_panorama_mp}MP)")
            
            # #region agent log
            try:
                import json
                with open(r'c:\Users\ryanf\OneDrive - University of Maryland\Desktop\Stitch2Stitch\Stitch2Stitch-1\.cursor\debug.log', 'a') as f:
                    f.write(json.dumps({"hypothesisId":"UI","location":"main_window.py:_init_stitcher","message":"Stitcher settings","data":{"feature_detector":feature_detector,"geo_method":geo_method,"geo_verify":geometric_verify},"timestamp":__import__('time').time()}) + '\n')
            except: pass
            # #endregion
        except Exception as e:
            logger.error(f"Failed to initialize stitcher: {e}", exc_info=True)
            raise
    
    def add_images(self):
        """Add images from a selected folder"""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Folder Containing Images",
            "",
            QFileDialog.Option.ShowDirsOnly | QFileDialog.Option.DontResolveSymlinks
        )
        
        if not folder:
            return  # User cancelled
        
        folder_path = Path(folder)
        
        # Supported image extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.JPG', '.JPEG', '.PNG', '.TIF', '.TIFF', '.BMP'}
        
        # Find all image files in the folder
        image_files = []
        for ext in image_extensions:
            image_files.extend(folder_path.glob(f"*{ext}"))
        
        # Sort by filename for consistent ordering
        image_files.sort(key=lambda p: p.name.lower())
        
        if not image_files:
            QMessageBox.information(
                self,
                "No Images Found",
                f"No image files found in:\n{folder}\n\nSupported formats: JPG, PNG, TIFF, BMP"
            )
            return
        
        # Add images that aren't already in the list
        added_count = 0
        for image_path in image_files:
            if image_path not in self.image_paths:
                self.image_paths.append(image_path)
                self.image_list.addItem(image_path.name)
                added_count += 1
        
        self.log(f"Added {added_count} image(s) from folder. Total: {len(self.image_paths)}")

        if added_count < len(image_files):
            QMessageBox.information(
                self,
                "Some Images Skipped",
                f"Added {added_count} new image(s).\n{len(image_files) - added_count} image(s) were already in the list."
            )
    
    def remove_selected_image(self):
        """Remove selected image from list"""
        current = self.image_list.currentRow()
        if current >= 0:
            self.image_paths.pop(current)
            self.image_list.takeItem(current)
            self.log("Removed image from list")

    def clear_images(self):
        """Clear all images"""
        self.image_paths.clear()
        self.image_list.clear()
        self.log("Cleared all images")
    
    def clear_preview(self):
        """Clear the preview display"""
        self.preview_label.clear()
        self.preview_label.setText("No preview available")
        self.current_result = None
        self._preview_image = None
        self._preview_zoom = 1.0
        self.zoom_slider.setValue(100)
        self.zoom_percent_label.setText("100%")
        self.log("Preview cleared")
    
    def zoom_in(self):
        """Zoom in the preview by 25%"""
        current = self.zoom_slider.value()
        new_zoom = min(400, current + 25)
        self.zoom_slider.setValue(new_zoom)
    
    def zoom_out(self):
        """Zoom out the preview by 25%"""
        current = self.zoom_slider.value()
        new_zoom = max(10, current - 25)
        self.zoom_slider.setValue(new_zoom)
    
    def zoom_fit(self):
        """Fit the preview to the scroll area"""
        if self._preview_image is None:
            return
        
        h, w = self._preview_image.shape[:2]
        scroll_w = self.preview_scroll.viewport().width() - 20
        scroll_h = self.preview_scroll.viewport().height() - 20
        
        scale_w = scroll_w / w if w > 0 else 1
        scale_h = scroll_h / h if h > 0 else 1
        fit_scale = min(scale_w, scale_h) * 100
        
        self.zoom_slider.setValue(int(max(10, min(400, fit_scale))))
    
    def zoom_actual(self):
        """Set zoom to 100% (actual size)"""
        self.zoom_slider.setValue(100)
    
    def on_zoom_changed(self, value):
        """Handle zoom slider value change"""
        self._preview_zoom = value / 100.0
        self.zoom_percent_label.setText(f"{value}%")
        self._update_preview_zoom()
    
    def browse_output(self):
        """Browse for output file"""
        file, _ = QFileDialog.getSaveFileName(
            self,
            "Save Panorama",
            "",
            "TIFF Files (*.tif *.tiff);;PNG Files (*.png);;JPEG Files (*.jpg)"
        )
        
        if file:
            self.output_path = Path(file)
            self.output_path_label.setText(f"Output: {self.output_path.name}")
    
    def create_grid(self):
        """Create grid alignment"""
        if not self.image_paths:
            QMessageBox.warning(self, "No Images", "Please add images first.")
            return
        
        if not self.output_path:
            QMessageBox.warning(self, "No Output", "Please select output path first.")
            return
        
        self.init_stitcher()
        
        # Get overlap threshold from UI (percentage)
        min_overlap_percent = self.overlap_spin.value()
        max_overlap_percent = self.max_overlap_spin.value()
        self.log(f"Starting grid alignment creation (overlap: {min_overlap_percent}%-{max_overlap_percent}%)...")
        
        self.btn_grid.setEnabled(False)
        self.btn_stitch.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_label.setVisible(True)
        self.progress_label.setText("Initializing...")
        
        # Get grid spacing from UI
        spacing_factor = self.grid_spacing_spin.value()
        
        # Use GridAlignmentThread to pass overlap threshold and spacing
        self.thread = GridAlignmentThread(self.stitcher, self.image_paths, min_overlap_percent, max_overlap_percent, spacing_factor)
        self.thread.progress.connect(self.update_progress)
        self.thread.status.connect(self.update_status)
        self.thread.finished.connect(self.on_grid_finished)
        self.thread.error.connect(self.on_error)
        self.thread.start()
    
    def start_stitching(self):
        """Start stitching process"""
        if not self.image_paths:
            QMessageBox.warning(self, "No Images", "Please add images first.")
            return
        
        if not self.output_path:
            QMessageBox.warning(self, "No Output", "Please select output path first.")
            return
        
        self.init_stitcher()
        self.log("Starting stitching process...")
        
        self.btn_grid.setEnabled(False)
        self.btn_stitch.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_label.setVisible(True)
        self.progress_label.setText("Initializing...")
        
        self.thread = StitchingThread(self.stitcher, self.image_paths, grid_only=False)
        self.thread.progress.connect(self.update_progress)
        self.thread.status.connect(self.update_status)
        self.thread.finished.connect(self.on_stitching_finished)
        self.thread.error.connect(self.on_error)
        self.thread.start()
    
    def stop_processing(self):
        """Stop the current processing operation"""
        if hasattr(self, 'thread') and self.thread and self.thread.isRunning():
            reply = QMessageBox.question(
                self,
                "Confirm Stop",
                "Are you sure you want to stop the current operation?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.thread.cancel()
                self.log("Stopping operation...")
                # Wait a bit for thread to respond, then force if needed
                if not self.thread.wait(1000):  # Wait up to 1 second
                    logger.warning("Thread did not stop gracefully, forcing termination")
                    self.thread.terminate()
                    self.thread.wait(500)
                self.btn_stop.setEnabled(False)
                self.reset_ui_after_processing()
    
    def update_progress(self, percentage: int):
        """Update progress bar"""
        self.progress_bar.setValue(percentage)
    
    def update_status(self, message: str):
        """Update status message"""
        self.progress_label.setText(message)
        self.log(message)
    
    def on_grid_finished(self, grid_layout):
        """Handle grid creation completion"""
        try:
            self.log("Grid alignment created successfully!")
            
            # Get quality, DPI, and post-processing settings
            quality = self._get_quality_setting()
            dpi = self.output_dpi_spin.value()
            postproc = self._get_postproc_options()
            
            # Save grid first (this might modify grid_layout, so do it before preview)
            try:
                self.stitcher.save_grid(grid_layout, str(self.output_path), quality=quality, dpi=dpi, postproc=postproc)
            except Exception as e:
                logger.error(f"Error saving grid: {e}", exc_info=True)
                # Continue even if save fails
            
            self.current_result = grid_layout
            
            # Update preview (this might crash, so wrap it)
            try:
                self.update_preview_from_grid(grid_layout)
            except Exception as e:
                logger.error(f"Error updating grid preview: {e}", exc_info=True)
                self.preview_label.setText("Grid alignment created.\nPreview unavailable.\nUse 'Save Result' to export.")
            
            self.reset_ui_after_processing()
            self.btn_save.setEnabled(True)
            
            QMessageBox.information(self, "Success", "Grid alignment created successfully!")
        except Exception as e:
            logger.error(f"Error in on_grid_finished: {e}", exc_info=True)
            self.on_error(f"Failed to process grid result: {e}")
    
    def on_stitching_finished(self, panorama):
        """Handle stitching completion"""
        try:
            self.log("Stitching completed successfully!")
            
            # Validate panorama
            if panorama is None:
                raise ValueError("Panorama is None")
            
            if not isinstance(panorama, np.ndarray):
                raise ValueError(f"Panorama is not a numpy array: {type(panorama)}")
            
            if panorama.size == 0:
                raise ValueError("Panorama is empty")
            
            if len(panorama.shape) < 2:
                raise ValueError(f"Invalid panorama shape: {panorama.shape}")
            
            self.current_result = panorama
            self.original_result = panorama.copy()  # Store for post-processing reset
            self.update_preview(panorama)

            # Enable post-processing buttons
            self.apply_postproc_btn.setEnabled(True)
            self.reset_postproc_btn.setEnabled(True)

            self.reset_ui_after_processing()
            self.btn_save.setEnabled(True)

            QMessageBox.information(self, "Success", "Stitching completed successfully!")
        except Exception as e:
            logger.error(f"Error handling stitching completion: {e}", exc_info=True)
            self.on_error(f"Failed to display result: {e}")
    
    def on_error(self, error_msg):
        """Handle errors"""
        self.log(f"Error: {error_msg}")
        self.reset_ui_after_processing()
        
        if "cancelled" not in error_msg.lower():
            QMessageBox.critical(self, "Error", f"An error occurred:\n{error_msg}")
    
    def reset_ui_after_processing(self):
        """Reset UI elements after processing completes"""
        self.btn_grid.setEnabled(True)
        self.btn_stitch.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.progress_bar.setValue(0)
        self.progress_label.setVisible(False)
        self.progress_label.setText("")
    
    def _on_blend_method_changed(self, text: str):
        """Show/hide blending options based on selected method."""
        is_semantic = "Semantic" in text
        is_generative = "Generative" in text
        
        self.semantic_options_group.setVisible(is_semantic)
        self.gen_ai_group.setVisible(is_generative)
    
    def update_preview(self, image: np.ndarray):
        """Update preview with image (supports live zoom)"""
        try:
            # Validate input
            if image is None:
                logger.warning("Cannot update preview: image is None")
                self.preview_label.setText("No preview available")
                self._preview_image = None
                return
            
            if not isinstance(image, np.ndarray):
                logger.warning(f"Cannot update preview: invalid type {type(image)}")
                self.preview_label.setText("Invalid image data")
                self._preview_image = None
                return
            
            if image.size == 0:
                logger.warning("Cannot update preview: image is empty")
                self.preview_label.setText("Empty image")
                self._preview_image = None
                return
            
            h, w = image.shape[:2]
            if w == 0 or h == 0:
                logger.warning(f"Invalid image dimensions: {w}x{h}")
                self.preview_label.setText("Invalid image dimensions")
                return
            
            # Normalize image format
            if len(image.shape) == 3:
                if image.shape[2] != 3:
                    if image.shape[2] == 1:
                        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                    elif image.shape[2] == 4:
                        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            
            # Store the original image for zooming
            self._preview_image = image.copy()
            
            # Update the display with current zoom
            self._update_preview_zoom()
            
        except Exception as e:
            logger.error(f"Error updating preview: {e}", exc_info=True)
            self.preview_label.setText(f"Preview error: {str(e)}")
    
    def _update_preview_zoom(self):
        """Update the preview display with current zoom level"""
        try:
            if self._preview_image is None:
                return
            
            image = self._preview_image
            h, w = image.shape[:2]
            
            # Get the scroll area viewport size for auto-fit
            viewport = self.preview_scroll.viewport()
            viewport_w = viewport.width() - 20  # Leave some margin
            viewport_h = viewport.height() - 20
            
            # If zoom is 1.0 (default), auto-fit to preview area
            if self._preview_zoom == 1.0:
                # Calculate scale to fit in viewport
                scale_w = viewport_w / w if w > 0 else 1.0
                scale_h = viewport_h / h if h > 0 else 1.0
                auto_scale = min(scale_w, scale_h, 1.0)  # Don't upscale beyond 100%
                new_w = max(1, int(w * auto_scale))
                new_h = max(1, int(h * auto_scale))
            else:
                # Apply manual zoom
                new_w = max(1, int(w * self._preview_zoom))
                new_h = max(1, int(h * self._preview_zoom))
            
            # Use appropriate interpolation based on zoom direction
            actual_scale = new_w / w if w > 0 else 1.0
            if actual_scale < 1.0:
                interp = cv2.INTER_AREA  # Better for shrinking
            else:
                interp = cv2.INTER_LINEAR  # Better for enlarging
            
            if len(image.shape) == 3:
                resized = cv2.resize(image, (new_w, new_h), interpolation=interp)
                # Convert BGR to RGB for Qt
                resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                resized_rgb = np.ascontiguousarray(resized_rgb, dtype=np.uint8)
                bytes_per_line = resized_rgb.strides[0]
                qimage = QImage(resized_rgb.data, new_w, new_h, bytes_per_line, QImage.Format.Format_RGB888)
            else:
                resized = cv2.resize(image, (new_w, new_h), interpolation=interp)
                resized = np.ascontiguousarray(resized, dtype=np.uint8)
                bytes_per_line = resized.strides[0]
                qimage = QImage(resized.data, new_w, new_h, bytes_per_line, QImage.Format.Format_Grayscale8)
            
            if qimage.isNull():
                logger.warning("Failed to create QImage")
                return
            
            pixmap = QPixmap.fromImage(qimage)
            if pixmap.isNull():
                logger.warning("Failed to create QPixmap")
                return
            
            self.preview_label.setPixmap(pixmap)
            self.preview_label.resize(pixmap.size())
            
        except Exception as e:
            logger.error(f"Error updating zoom preview: {e}", exc_info=True)
    
    def update_preview_from_grid(self, grid_layout):
        """Update preview with grid layout"""
        try:
            if grid_layout is None:
                logger.warning("Cannot update preview: grid_layout is None")
                self.preview_label.setText("No grid layout available")
                return
            
            images = grid_layout.get('images', [])
            positions = grid_layout.get('positions', [])
            
            if not images or not positions:
                logger.warning("Cannot update preview: empty grid layout")
                self.preview_label.setText("Grid alignment created.\nUse 'Save Result' to export.")
                return
            
            # Import grid visualizer to create preview
            from core.grid_visualizer import GridVisualizer
            
            # Create a temporary preview image
            visualizer = GridVisualizer()
            preview_image = visualizer.create_preview_image(grid_layout)
            
            if preview_image is not None:
                # Use the same update_preview method to display it
                self.update_preview(preview_image)
            else:
                self.preview_label.setText("Grid alignment created.\nUse 'Save Result' to export.")
        except Exception as e:
            logger.error(f"Error updating grid preview: {e}", exc_info=True)
            self.preview_label.setText(f"Preview error: {str(e)}")
    
    def update_postproc_preview(self):
        """Update preview with post-processing applied"""
        try:
            # Check if we have a result to process
            if not hasattr(self, 'current_result') or self.current_result is None:
                return
            
            # Get post-processing options (without upscale for preview - too slow)
            postproc = self._get_postproc_options()
            postproc['upscale_factor'] = 1.0  # Don't upscale in preview
            
            # Check if any post-processing is enabled
            has_postproc = (
                postproc.get('sharpen', False) or
                postproc.get('denoise', False) or
                postproc.get('shadow_removal', False) or
                postproc.get('clahe', False) or
                postproc.get('deblur', False)
            )
            
            if isinstance(self.current_result, dict):
                # Grid layout - update grid preview
                self.update_preview_from_grid(self.current_result)
                # Note: post-processing for grid is applied at save time
                return
            
            elif isinstance(self.current_result, np.ndarray):
                # Panorama
                if not has_postproc:
                    # No post-processing, show original
                    self.update_preview(self.current_result)
                    return
                
                # Apply post-processing to a copy for preview
                from core.post_processing import apply_post_processing
                
                # For large images, process at reduced size for responsiveness
                h, w = self.current_result.shape[:2]
                max_preview_size = 1600  # Process at this max size for speed
                
                if max(h, w) > max_preview_size:
                    scale = max_preview_size / max(h, w)
                    preview_img = cv2.resize(
                        self.current_result, 
                        (int(w * scale), int(h * scale)),
                        interpolation=cv2.INTER_AREA
                    )
                else:
                    preview_img = self.current_result.copy()
                
                # Apply post-processing
                processed = apply_post_processing(preview_img, postproc)
                self.update_preview(processed)
                
        except Exception as e:
            logger.error(f"Error updating post-processing preview: {e}", exc_info=True)
    
    def _get_quality_setting(self) -> str:
        """Convert quality combo index to quality string"""
        quality_map = {
            0: 'ultra_high',  # Ultra High (Uncompressed)
            1: 'high',        # High (Lossless)
            2: 'medium',      # Medium (Balanced)
            3: 'low',         # Low (Compressed)
            4: 'minimum',     # Minimum (Most Compressed)
        }
        return quality_map.get(self.output_quality_combo.currentIndex(), 'high')
    
    def _get_postproc_options(self) -> dict:
        """Gather post-processing options from UI"""
        # Get upscale factor
        upscale_map = {0: 1.0, 1: 1.5, 2: 2.0, 3: 3.0, 4: 4.0}
        upscale_factor = upscale_map.get(self.upscale_combo.currentIndex(), 1.0)
        
        # Get interpolation method
        interp_map = {0: 'lanczos', 1: 'cubic', 2: 'linear', 3: 'nearest'}
        interpolation = interp_map.get(self.interp_combo.currentIndex(), 'lanczos')
        
        return {
            'sharpen': self.sharpen_checkbox.isChecked(),
            'sharpen_amount': self.sharpen_amount_spin.value(),
            'denoise': self.denoise_checkbox.isChecked(),
            'denoise_strength': self.denoise_strength_spin.value(),
            'shadow_removal': self.shadow_checkbox.isChecked(),
            'shadow_strength': self.shadow_strength_spin.value(),
            'clahe': self.clahe_checkbox.isChecked(),
            'clahe_strength': self.clahe_strength_spin.value(),
            'deblur': self.deblur_checkbox.isChecked(),
            'deblur_radius': self.deblur_strength_spin.value(),
            'upscale_factor': upscale_factor,
            'interpolation': interpolation
        }
    
    def save_result(self):
        """Save current result"""
        # Check if we have a result (handle numpy arrays correctly)
        has_result = False
        if isinstance(self.current_result, dict):
            has_result = True
        elif isinstance(self.current_result, np.ndarray):
            has_result = self.current_result.size > 0
        elif self.current_result is not None:
            has_result = True
        
        if not has_result:
            QMessageBox.warning(self, "No Result", "No result to save. Please stitch images first.")
            return
        
        if not self.output_path:
            self.browse_output()
            if not self.output_path:
                return
        
        # Ensure stitcher is initialized
        if not hasattr(self, 'stitcher') or self.stitcher is None:
            self.init_stitcher()
        
        # Get quality and DPI settings
        quality = self._get_quality_setting()
        dpi = self.output_dpi_spin.value()
        postproc = self._get_postproc_options()
        
        try:
            # Log what post-processing will be applied
            postproc_desc = []
            if postproc['sharpen']:
                postproc_desc.append(f"sharpen({postproc['sharpen_amount']})")
            if postproc['denoise']:
                postproc_desc.append(f"denoise({postproc['denoise_strength']})")
            if postproc['shadow_removal']:
                postproc_desc.append(f"shadow({postproc['shadow_strength']})")
            if postproc['clahe']:
                postproc_desc.append(f"clahe({postproc['clahe_strength']})")
            if postproc['deblur']:
                postproc_desc.append(f"deblur({postproc['deblur_radius']})")
            if postproc['upscale_factor'] > 1.0:
                postproc_desc.append(f"upscale({postproc['upscale_factor']}x)")
            
            postproc_str = ", ".join(postproc_desc) if postproc_desc else "none"
            self.log(f"Saving result to {self.output_path} (quality={quality}, dpi={dpi}, postproc=[{postproc_str}])...")
            
            if isinstance(self.current_result, dict):  # Grid layout
                self.stitcher.save_grid(self.current_result, str(self.output_path), quality=quality, dpi=dpi, postproc=postproc)
            else:  # Panorama (numpy array)
                panorama = self.current_result
                self.log(f"Panorama shape: {panorama.shape}, dtype: {panorama.dtype}")
                self.stitcher.save_panorama(panorama, str(self.output_path), quality=quality, dpi=dpi, postproc=postproc)
            
            # Verify save was successful
            if self.output_path.exists():
                file_size = self.output_path.stat().st_size
                size_mb = file_size / 1024 / 1024
                self.log(f"Result saved to {self.output_path} ({size_mb:.2f} MB)")
                QMessageBox.information(self, "Saved", f"Result saved to {self.output_path}\nFile size: {size_mb:.2f} MB")
            else:
                raise IOError("File was not created")
                
        except Exception as e:
            logger.error(f"Save error: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to save:\n{e}")
    
    def log(self, message: str):
        """Add message to log"""
        self.log_text.append(message)
        logger.info(message)
        self.statusBar().showMessage(message)
    
    def _update_external_pipeline_availability(self):
        """Check and update availability of external pipelines."""
        try:
            from external.pipelines import check_available_pipelines
            
            # Get detailed info including versions and paths
            available = check_available_pipelines(detailed=True)
            
            # Log detection results (only if log_text exists)
            can_log = hasattr(self, 'log_text') and self.log_text is not None
            if can_log:
                self.log("Checking external pipelines...")
            
            # COLMAP status
            colmap_info = available.get("colmap", {})
            if colmap_info.get("available", False):
                info_text = colmap_info.get("info", "COLMAP")
                path_text = colmap_info.get("path", "")
                self.colmap_status.setText("[OK]")
                self.colmap_status.setToolTip(f"COLMAP is installed and ready\n{info_text}\nPath: {path_text}")
                self.colmap_btn.setEnabled(True)
                self.colmap_btn.setStyleSheet("background-color: #4CAF50; color: white;")
                self.colmap_install_btn.setText("OK")
                self.colmap_install_btn.setEnabled(False)
                if can_log:
                    self.log(f"  [OK] COLMAP: {info_text}")
            else:
                self.colmap_status.setText("[X]")
                self.colmap_status.setToolTip("COLMAP is not installed\nChecked: PATH, conda envs, common locations")
                self.colmap_btn.setEnabled(False)
                self.colmap_btn.setStyleSheet("")
                self.colmap_install_btn.setText("Install")
                self.colmap_install_btn.setEnabled(True)
                if can_log:
                    self.log("  [X] COLMAP: Not found")
            
            # HLOC status
            hloc_info = available.get("hloc", {})
            if hloc_info.get("available", False):
                info_text = hloc_info.get("info", "HLOC")
                components = []
                if hloc_info.get("has_superpoint"):
                    components.append("SuperPoint")
                if hloc_info.get("has_superglue"):
                    components.append("SuperGlue")
                if hloc_info.get("has_netvlad"):
                    components.append("NetVLAD")
                    
                self.hloc_status.setText("[OK]")
                self.hloc_status.setToolTip(f"HLOC is installed and ready\n{info_text}")
                self.hloc_btn.setEnabled(True)
                self.hloc_btn.setStyleSheet("background-color: #4CAF50; color: white;")
                self.hloc_install_btn.setText("OK")
                self.hloc_install_btn.setEnabled(False)
                if can_log:
                    self.log(f"  ✅ HLOC: {info_text}")
            else:
                self.hloc_status.setText("[X]")
                self.hloc_status.setToolTip("HLOC is not installed\nInstall with: pip install hloc")
                self.hloc_btn.setEnabled(False)
                self.hloc_btn.setStyleSheet("")
                self.hloc_install_btn.setText("Install")
                self.hloc_install_btn.setEnabled(True)
                if can_log:
                    self.log("  [X] HLOC: Not found (pip install hloc)")
            
            # Meshroom/AliceVision status
            alice_info = available.get("alicevision", {})
            if alice_info.get("available", False):
                info_text = alice_info.get("info", "AliceVision")
                path_text = alice_info.get("path", "")
                self.meshroom_status.setText("[OK]")
                self.meshroom_status.setToolTip(f"AliceVision/Meshroom is installed\n{info_text}\nPath: {path_text}")
                self.meshroom_btn.setEnabled(True)
                self.meshroom_btn.setStyleSheet("background-color: #4CAF50; color: white;")
                self.meshroom_install_btn.setText("OK")
                self.meshroom_install_btn.setEnabled(False)
                if can_log:
                    self.log(f"  ✅ AliceVision: {info_text}")
            else:
                self.meshroom_status.setText("[X]")
                self.meshroom_status.setToolTip("AliceVision/Meshroom is not installed\nDownload from alicevision.org")
                self.meshroom_btn.setEnabled(False)
                self.meshroom_btn.setStyleSheet("")
                self.meshroom_install_btn.setText("Install")
                self.meshroom_install_btn.setEnabled(True)
                if can_log:
                    self.log("  [X] AliceVision: Not found")
                
        except ImportError as e:
            if hasattr(self, 'log_text') and self.log_text is not None:
                self.log(f"External pipelines module not available: {e}")
    
    def _install_colmap(self):
        """Install PyCOLMAP automatically in background."""
        # Disable button during installation
        self.colmap_install_btn.setEnabled(False)
        self.colmap_install_btn.setText("...")

        self.log("Starting PyCOLMAP installation...")
        self.statusBar().showMessage("Installing PyCOLMAP...")

        # Start background installation thread
        self.colmap_install_thread = COLMAPInstallThread()
        self.colmap_install_thread.status.connect(self._on_colmap_install_status)
        self.colmap_install_thread.finished.connect(self._on_colmap_install_finished)
        self.colmap_install_thread.start()

    def _on_colmap_install_status(self, message: str):
        """Handle COLMAP installation status updates."""
        self.log(f"COLMAP Install: {message}")
        self.statusBar().showMessage(f"COLMAP: {message}")

    def _on_colmap_install_finished(self, success: bool, message: str):
        """Handle COLMAP installation completion."""
        self.statusBar().showMessage("Ready")

        if success:
            self.log(f"âœ“ {message}")
            QMessageBox.information(self, "COLMAP Installed", message)
            self._update_external_pipeline_availability()
        else:
            self.log(f"✗ COLMAP installation failed: {message}")
            QMessageBox.critical(self, "COLMAP Installation Failed", message)
            # Re-enable button on failure
            self.colmap_install_btn.setEnabled(True)
            self.colmap_install_btn.setText("Install")
    
    def _install_hloc(self):
        """Install HLOC automatically in background using pre-existing pycolmap."""
        # Disable button during installation
        self.hloc_install_btn.setEnabled(False)
        self.hloc_install_btn.setText("...")
        
        self.log("Starting HLOC installation...")
        self.statusBar().showMessage("Installing HLOC...")
        
        # Start background installation thread
        self.hloc_install_thread = HLOCInstallThread()
        self.hloc_install_thread.status.connect(self._on_hloc_install_status)
        self.hloc_install_thread.finished.connect(self._on_hloc_install_finished)
        self.hloc_install_thread.start()
    
    def _on_hloc_install_status(self, message: str):
        """Handle HLOC installation status updates."""
        self.log(f"HLOC Install: {message}")
        self.statusBar().showMessage(f"HLOC: {message}")
    
    def _on_hloc_install_finished(self, success: bool, message: str):
        """Handle HLOC installation completion."""
        self.statusBar().showMessage("Ready")
        
        if success:
            self.log(f"âœ“ {message}")
            QMessageBox.information(self, "HLOC Installed", message)
            self._update_external_pipeline_availability()
        else:
            self.log(f"✗ HLOC installation failed: {message}")
            # Re-enable button on failure
            self.hloc_install_btn.setEnabled(True)
            self.hloc_install_btn.setText("Install")
            QMessageBox.warning(self, "HLOC Installation Failed", message)
    
    def _install_meshroom(self):
        """Open Meshroom download page."""
        import webbrowser
        
        reply = QMessageBox.question(
            self, "Install Meshroom",
            "Meshroom (AliceVision) needs to be downloaded manually.\n\n"
            "It's a standalone application that includes all required components.\n\n"
            "Open the Meshroom download page?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            webbrowser.open("https://alicevision.org/#meshroom")
    
    def _install_all_pipelines(self):
        """Install all available pipeline dependencies."""
        reply = QMessageBox.question(
            self, "Install All Dependencies",
            "This will attempt to install all external pipeline dependencies:\n\n"
            "• HLOC - Will be installed via pip\n"
            "• COLMAP - Download page will be opened\n"
            "• Meshroom - Download page will be opened\n\n"
            "Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # Install HLOC first (can be done automatically)
            self.log("Installing HLOC...")
            import subprocess
            import sys
            
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "hloc"],
                    capture_output=True,
                    text=True,
                    timeout=600
                )
                if result.returncode == 0:
                    self.log("HLOC installed successfully!")
                else:
                    self.log(f"HLOC installation failed: {result.stderr[:200]}")
            except Exception as e:
                self.log(f"HLOC installation error: {e}")
            
            # Open download pages for manual installers
            import webbrowser
            webbrowser.open("https://github.com/colmap/colmap/releases")
            webbrowser.open("https://alicevision.org/#meshroom")
            
            QMessageBox.information(
                self, "Installation Started",
                "HLOC installation attempted via pip.\n\n"
                "Download pages opened for:\n"
                "• COLMAP\n"
                "• Meshroom\n\n"
                "Please complete the manual installations and restart the app."
            )
            
            self._update_external_pipeline_availability()
    
    def _apply_preset(self, preset_name: str):
        """Apply a quick preset configuration."""
        self.log(f"Applying preset: {preset_name}")
        
        if preset_name == "few":
            # 2-50 images: Fast, simple settings (likely burst photos)
            self.detector_combo.setCurrentText("LP-SIFT (Recommended)")
            self.geo_verify_combo.setCurrentText("MAGSAC++ (Best)")
            self.remove_duplicates_checkbox.setChecked(True)  # Good for burst photos
            self.duplicate_threshold_spin.setValue(0.90)
            self.bundle_adjust_checkbox.setChecked(False)
            self.hierarchical_checkbox.setChecked(False)
            self.exhaustive_match_checkbox.setChecked(True)
            self.memory_efficient_checkbox.setChecked(False)
            self.blend_combo.setCurrentText("Multiband (Recommended)")
            self.max_features_spin.setValue(5000)
            
        elif preset_name == "medium":
            # 50-200 images: Balanced settings (panorama workflow)
            self.detector_combo.setCurrentText("LP-SIFT (Recommended)")
            self.geo_verify_combo.setCurrentText("MAGSAC++ (Best)")
            self.remove_duplicates_checkbox.setChecked(False)  # OFF for panoramas
            self.duplicate_threshold_spin.setValue(0.95)
            self.bundle_adjust_checkbox.setChecked(True)
            self.hierarchical_checkbox.setChecked(False)
            self.exhaustive_match_checkbox.setChecked(True)
            self.memory_efficient_checkbox.setChecked(True)
            self.blend_combo.setCurrentText("Multiband (Recommended)")
            self.max_features_spin.setValue(5000)
            
        elif preset_name == "large":
            # 200-500 images: Robust settings (large panorama)
            # Try SuperPoint if available
            if self.detector_combo.findText("SuperPoint (Deep Learning)") >= 0:
                self.detector_combo.setCurrentText("SuperPoint (Deep Learning)")
            else:
                self.detector_combo.setCurrentText("LP-SIFT (Recommended)")
            self.geo_verify_combo.setCurrentText("MAGSAC++ (Best)")
            self.remove_duplicates_checkbox.setChecked(False)  # OFF for panoramas
            self.duplicate_threshold_spin.setValue(0.95)
            self.bundle_adjust_checkbox.setChecked(True)
            self.hierarchical_checkbox.setChecked(True)
            self.exhaustive_match_checkbox.setChecked(True)
            self.memory_efficient_checkbox.setChecked(True)
            self.blend_combo.setCurrentText("AutoStitch (Simple & Fast)")
            self.max_features_spin.setValue(3000)  # Fewer features for speed
            
        elif preset_name == "gigapixel":
            # 500+ images: Maximum robustness
            # Recommend external pipelines
            reply = QMessageBox.question(
                self, "Gigapixel Recommendation",
                "For 500+ images, external pipelines (COLMAP or HLOC) are\n"
                "strongly recommended for best results.\n\n"
                "• COLMAP: Battle-tested, robust matching\n"
                "• HLOC: SuperPoint + SuperGlue + NetVLAD\n\n"
                "Apply built-in gigapixel settings anyway?\n"
                "(May be slow and less accurate than external tools)",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                if self.detector_combo.findText("SuperPoint (Deep Learning)") >= 0:
                    self.detector_combo.setCurrentText("SuperPoint (Deep Learning)")
                else:
                    self.detector_combo.setCurrentText("LP-SIFT (Recommended)")
                self.geo_verify_combo.setCurrentText("MAGSAC++ (Best)")
                self.remove_duplicates_checkbox.setChecked(False)  # OFF for panoramas
                self.duplicate_threshold_spin.setValue(0.95)
                self.bundle_adjust_checkbox.setChecked(True)
                self.hierarchical_checkbox.setChecked(True)
                self.exhaustive_match_checkbox.setChecked(True)
                self.memory_efficient_checkbox.setChecked(True)
                self.blend_combo.setCurrentText("AutoStitch (Simple & Fast)")
                self.max_features_spin.setValue(2000)  # Fewer features for memory
            else:
                # Scroll to external pipelines section
                self.log("Consider using COLMAP or HLOC for best results with 500+ images")
                return
        
        self.log(f"Preset '{preset_name}' applied successfully")

    def _detect_available_gpus(self):
        """Detect available CUDA GPUs using multiple methods"""
        detected = []

        # Helper to log safely (log_text may not exist yet during init)
        def safe_log(msg):
            if hasattr(self, 'log_text'):
                self.log(msg)
            else:
                logger.info(msg)

        # Method 1: Try PyTorch (most reliable, gives full info)
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    name = torch.cuda.get_device_name(i)
                    memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    detected.append(f"GPU {i}: {name} ({memory:.1f}GB)")
                safe_log(f"Detected {len(detected)} GPU(s) via PyTorch")
                return detected
        except Exception as e:
            safe_log(f"PyTorch GPU detection failed: {e}")

        # Method 2: Try nvidia-smi directly (works on Linux/Windows with CUDA)
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=3
            )
            if result.returncode == 0 and result.stdout:
                for line in result.stdout.strip().split('\n'):
                    if line:
                        parts = line.split(',')
                        if len(parts) >= 3:
                            idx = parts[0].strip()
                            name = parts[1].strip()
                            mem = parts[2].strip()
                            detected.append(f"GPU {idx}: {name} ({mem})")
                if detected:
                    safe_log(f"Detected {len(detected)} GPU(s) via nvidia-smi")
                    return detected
        except Exception as e:
            safe_log(f"nvidia-smi GPU detection failed: {e}")

        # Method 3: Try WSL nvidia-smi for Windows
        try:
            import platform
            import subprocess
            if platform.system() == "Windows":
                result = subprocess.run(
                    ["wsl", "bash", "-c", "nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader"],
                    capture_output=True,
                    text=True,
                    timeout=3
                )
                if result.returncode == 0 and result.stdout:
                    for line in result.stdout.strip().split('\n'):
                        if line:
                            parts = line.split(',')
                            if len(parts) >= 3:
                                idx = parts[0].strip()
                                name = parts[1].strip()
                                mem = parts[2].strip()
                                detected.append(f"GPU {idx} (WSL): {name} ({mem})")
                    if detected:
                        safe_log(f"Detected {len(detected)} GPU(s) via WSL")
                        return detected
        except Exception as e:
            safe_log(f"WSL GPU detection failed: {e}")

        # Method 4: Try checking if pycolmap has CUDA (fallback, no specific GPU info)
        try:
            import pycolmap
            if hasattr(pycolmap, 'has_cuda') and pycolmap.has_cuda:
                detected.append("GPU 0 (CUDA Available)")
                safe_log("CUDA available via pycolmap but cannot detect specific GPUs")
                return detected
        except:
            pass

        # No GPUs detected
        safe_log("No GPUs detected, CPU-only mode")
        return []

    def _on_colmap_preset_changed(self, index):
        """Handle COLMAP quality preset changes (only affects feature extraction, not matching)"""
        if index == 0:  # Original
            self.colmap_features_spin.setValue(4096)  # Good features
            self.log("COLMAP Quality: Original (8192 features, full resolution)")
        elif index == 1:  # Phase 1 - Optimized
            self.colmap_features_spin.setValue(4096)  # Reduced features
            self.log("COLMAP Quality: Phase 1 (4096 features, 2x faster extraction)")
        elif index == 2:  # Phase 2 - Fast
            self.colmap_features_spin.setValue(2048)  # Minimal features
            self.log("COLMAP Quality: Phase 2 (2048 features, 3-4x faster extraction)")

    def _run_colmap(self):
        """Run COLMAP pipeline on current images to create a 2D panorama."""
        if not self.image_paths:
            QMessageBox.warning(self, "No Images", "Please load images first.")
            return
        
        try:
            from external.pipelines import COLMAPPipeline
            
            # Quick availability check (runs on main thread, fast)
            pipeline = COLMAPPipeline()
            
            if not pipeline.is_available():
                reply = QMessageBox.question(
                    self, "PyCOLMAP Not Found",
                    "PyCOLMAP is not installed.\n\n" + pipeline.get_install_instructions() + 
                    "\n\nWould you like to install it now?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.Yes:
                    import webbrowser
                    webbrowser.open("https://pypi.org/project/pycolmap/")
                return
            
            # Get output directory
            output_dir = QFileDialog.getExistingDirectory(
                self, "Select Output Directory for COLMAP Panorama"
            )
            if not output_dir:
                return
            
            # Get transform type from dropdown
            transform_idx = self.colmap_transform_combo.currentIndex()
            transform_type = "affine" if transform_idx == 1 else "homography"

            # Get blending method from dropdown
            blend_map = {
                "Multiband (Recommended)": "multiband",
                "Feather": "feather",
                "AutoStitch": "autostitch",
                "Mosaic": "mosaic",
                "Linear": "linear"
            }
            blend_text = self.colmap_blend_combo.currentText()
            blend_method = blend_map.get(blend_text, "multiband")

            # Get matching strategy
            matching_idx = self.colmap_matching_combo.currentIndex()
            matching_strategies = ["exhaustive", "sequential", "neighbor", "vocab_tree", "grid"]
            matching_strategy = matching_strategies[matching_idx]

            # Get sequential overlap
            sequential_overlap = self.colmap_overlap_spin.value()

            # Get GPU selection
            selected_gpus = []
            for item in self.colmap_gpu_list.selectedItems():
                gpu_text = item.text()
                if "Auto" in gpu_text:
                    selected_gpus = ["auto"]
                    break
                elif "CPU Only" in gpu_text:
                    selected_gpus = ["cpu"]
                    break
                elif "GPU" in gpu_text:
                    # Extract GPU index from text like "GPU 0: RTX 3090" or "GPU 1 (WSL): Tesla"
                    import re
                    match = re.search(r'GPU (\d+)', gpu_text)
                    if match:
                        selected_gpus.append(match.group(1))

            if not selected_gpus:
                selected_gpus = ["auto"]

            # Get thread count and max features
            num_threads = self.colmap_threads_spin.value()
            max_features = self.colmap_features_spin.value()

            # Get image filtering parameters
            min_inliers = self.colmap_min_inliers_spin.value()
            max_images = self.colmap_max_images_spin.value()

            # Get alpha handling option
            use_source_alpha = self.colmap_use_source_alpha.isChecked()

            # Get duplicate removal settings
            remove_duplicates = self.colmap_remove_duplicates.isChecked()
            duplicate_threshold = self.colmap_duplicate_threshold.value() / 100.0

            # Get warp interpolation setting
            warp_interpolation = self.colmap_warp_interpolation.currentText().lower()

            # Get border erosion settings
            erode_border = self.colmap_erode_border.isChecked()
            border_erosion_pixels = self.colmap_border_erosion_spin.value()

            self.log("Starting COLMAP 2D stitching pipeline...")
            self.log(f"Processing {len(self.image_paths)} images...")
            self.log(f"Transform: {transform_type}")
            self.log(f"Blending: {blend_method}")
            self.log(f"Matching: {matching_strategy} (overlap={sequential_overlap if matching_strategy == 'sequential' else 'N/A'})")
            self.log(f"GPUs: {', '.join(selected_gpus)}")
            self.log(f"Threads: {num_threads}, Max Features: {max_features}")
            if min_inliers > 0:
                self.log(f"Filtering: Min inliers={min_inliers}")
            if max_images > 0:
                self.log(f"Filtering: Max images={max_images}")
            if use_source_alpha:
                self.log("Alpha: Using source image alpha (transparent backgrounds)")
                if erode_border:
                    self.log(f"Border erosion: {border_erosion_pixels}px")
            if remove_duplicates:
                self.log(f"Duplicate removal: Enabled (threshold={duplicate_threshold:.0%})")
            self.log(f"Warp interpolation: {warp_interpolation.capitalize()}")
            self.progress_bar.setValue(0)

            # Run in background thread to keep GUI responsive
            self.colmap_thread = COLMAPThread(
                self.image_paths,
                Path(output_dir),
                transform_type=transform_type,
                blend_method=blend_method,
                matching_strategy=matching_strategy,
                sequential_overlap=sequential_overlap,
                gpu_indices=selected_gpus,
                num_threads=num_threads,
                max_features=max_features,
                min_inliers=min_inliers,
                max_images=max_images,
                use_source_alpha=use_source_alpha,
                remove_duplicates=remove_duplicates,
                duplicate_threshold=duplicate_threshold,
                warp_interpolation=warp_interpolation,
                erode_border=erode_border,
                border_erosion_pixels=border_erosion_pixels
            )
            self.colmap_thread.progress.connect(self._on_colmap_progress)
            self.colmap_thread.status.connect(self._on_colmap_status)
            self.colmap_thread.finished.connect(self._on_colmap_finished)
            self.colmap_thread.error.connect(self._on_colmap_error)

            # Store parameters for reblend feature
            self._last_run_params = {
                "transform_type": transform_type,
                "blend_method": blend_method,
                "matching_strategy": matching_strategy,
                "sequential_overlap": sequential_overlap,
                "max_features": max_features,
                "min_inliers": min_inliers,
                "max_images": max_images,
                "warp_interpolation": warp_interpolation,
                "output_dir": Path(output_dir),
                "erode_border": erode_border,
                "border_erosion_pixels": border_erosion_pixels,
            }

            self.colmap_thread.start()

            # Disable buttons during processing
            self.colmap_btn.setEnabled(False)
            self.btn_stitch.setEnabled(False)
                
        except Exception as e:
            logger.error(f"COLMAP error: {e}", exc_info=True)
            QMessageBox.critical(self, "COLMAP Error", str(e))
    
    def _on_colmap_progress(self, percentage: int):
        """Handle COLMAP progress updates."""
        self.progress_bar.setValue(percentage)
    
    def _on_colmap_status(self, message: str):
        """Handle COLMAP status updates."""
        self.log(message)
        self.statusBar().showMessage(message)
    
    def _on_colmap_finished(self, result: dict):
        """Handle COLMAP completion."""
        # Re-enable buttons
        self.colmap_btn.setEnabled(True)
        self.btn_stitch.setEnabled(True)

        if result.get("success"):
            panorama = result.get("panorama")
            output_path = result.get("output_path", "")
            size = result.get("size", (0, 0))
            gpu_used = result.get("gpu_used", False)
            n_images = result.get("n_images", 0)
            n_images_original = result.get("n_images_original", 0)
            n_duplicates_removed = result.get("n_duplicates_removed", 0)

            mode = "GPU" if gpu_used else "CPU"
            self.log(f"COLMAP panorama complete ({mode})! Size: {size[1]}x{size[0]} pixels")

            # Report image filtering stats
            if n_duplicates_removed > 0:
                filter_msg = f"Processed {n_images}/{n_images_original} images ({n_duplicates_removed} duplicates removed)"
                self.log(filter_msg)

            self.log(f"Saved to: {output_path}")

            # Store result for display/saving
            # Load panorama from disk if not in result (WSL can't pass large arrays)
            if panorama is None and output_path:
                try:
                    panorama = cv2.imread(str(output_path), cv2.IMREAD_UNCHANGED)
                    if panorama is not None:
                        self.log("Loaded panorama from disk for preview.")
                except Exception as e:
                    self.log(f"Failed to load panorama from disk: {e}")
            
            if panorama is not None:
                self.current_result = panorama
                self.original_result = panorama.copy()  # Store original for reset
                self.output_path = Path(output_path)

                # Enable post-processing buttons
                self.apply_postproc_btn.setEnabled(True)
                self.reset_postproc_btn.setEnabled(True)

                # Auto-apply post-processing if enabled
                if self.auto_postproc_checkbox.isChecked() and self._should_apply_postproc():
                    self.log("Auto-applying post-processing...")
                    self._apply_postprocessing()
                else:
                    self.update_preview(panorama)

                # Enable reblend and store cache info for reuse
                try:
                    use_affine = self._last_run_params.get("transform_type") == "affine"
                    self._last_cache_key = compute_cache_key(
                        self.image_paths,
                        self._last_run_params.get("max_features", 8192),
                        matcher_type=self._last_run_params.get("matching_strategy", "exhaustive"),
                        use_affine=use_affine,
                        blend_method=self._last_run_params.get("blend_method", "multiband"),
                        warp_interpolation=self._last_run_params.get("warp_interpolation", "linear"),
                    )
                    self._last_image_paths = list(self.image_paths)
                    self._last_output_dir = self._last_run_params.get("output_dir", Path(output_path).parent)
                    self.reblend_btn.setEnabled(True)
                except Exception as e:
                    self.log(f'Reblend setup failed: {e}')  # Log but don't fail

            QMessageBox.information(
                self, "COLMAP Panorama Complete",
                f"Panorama created successfully!\n\n"
                f"Mode: {mode}\n"
                f"Images stitched: {result.get('n_images', 0)}\n"
                f"Size: {size[1]} x {size[0]} pixels\n"
                f"Output: {output_path}"
            )
        else:
            error_msg = result.get('error', 'Unknown error')
            self.log(f"COLMAP stitching failed: {error_msg}")
            QMessageBox.warning(self, "COLMAP Failed", f"Stitching failed:\n{error_msg}")
    
    def _on_colmap_error(self, error_msg: str):
        """Handle COLMAP errors."""
        # Re-enable buttons
        self.colmap_btn.setEnabled(True)
        self.btn_stitch.setEnabled(True)

        self.log(f"COLMAP error: {error_msg}")
        QMessageBox.critical(self, "COLMAP Error", error_msg)

    def _reblend_last(self):
        """Reblend using the last cached database (no feature extraction/matching)."""
        if not self._last_cache_key or not self._last_image_paths:
            QMessageBox.information(self, "Reblend", "No previous COLMAP run available to reblend.\nRun a full stitch first.")
            return
        try:
            from external.pipelines import COLMAPPipeline

            self.colmap_btn.setEnabled(False)
            self.btn_stitch.setEnabled(False)
            self.reblend_btn.setEnabled(False)
            self.statusBar().showMessage("Reblending panorama...")
            self.log("Starting reblend of last COLMAP run...")

            blend_text = self.colmap_blend_combo.currentText().split()[0].lower()
            blend_method = "multiband" if blend_text == "multiband" else ("feather" if blend_text == "feather" else ("autostitch" if blend_text == "autostitch" else "linear"))
            warp_text = self.warp_interp_combo.currentText().split()[0].lower()

            pipeline = COLMAPPipeline(
                use_gpu=True,
                use_affine=self._last_run_params.get("transform_type") == "affine",
                blend_method=blend_method,
                warp_interpolation=warp_text,
                erode_border=self._last_run_params.get("erode_border", True),
                border_erosion_pixels=self._last_run_params.get("border_erosion_pixels", 5),
                progress_callback=self._update_progress_slot,
            )
            pipeline._last_cache_key = self._last_cache_key
            out_dir = self._last_output_dir or (self.output_path.parent if hasattr(self, "output_path") and self.output_path else Path.home())
            result = pipeline.reblend_last(self._last_image_paths, out_dir)

            self.colmap_btn.setEnabled(True)
            self.btn_stitch.setEnabled(True)
            self.reblend_btn.setEnabled(True)

            if result.get("success") and result.get("output_path"):
                pano_path = Path(result["output_path"])
                img = cv2.imread(str(pano_path), cv2.IMREAD_UNCHANGED)
                if img is not None:
                    self.current_result = img
                    self.original_result = img.copy()
                    self.update_preview(img)
                    self.log(f"Reblend complete. Saved to: {pano_path}")
                    self.statusBar().showMessage("Reblend complete!")
                    return
            error_msg = result.get("error", "Unknown error")
            self.log(f"Reblend failed: {error_msg}")
            QMessageBox.warning(self, "Reblend Failed", error_msg)
            self.statusBar().showMessage("Reblend failed")
        except Exception as e:
            self.colmap_btn.setEnabled(True)
            self.btn_stitch.setEnabled(True)
            self.reblend_btn.setEnabled(True)
            self.log(f"Reblend error: {e}")
            QMessageBox.warning(self, "Reblend Error", str(e))
            self.statusBar().showMessage("Reblend error")

    # ============================================================
    # POST-PROCESSING METHODS
    # ============================================================

    def _should_apply_postproc(self) -> bool:
        """Check if any post-processing option is enabled."""
        return (
            self.pp_sharpen_checkbox.isChecked() or
            self.pp_denoise_checkbox.isChecked() or
            self.pp_clahe_checkbox.isChecked() or
            self.pp_shadow_checkbox.isChecked() or
            self.pp_deblur_checkbox.isChecked() or
            self.pp_white_balance_checkbox.isChecked() or
            self.pp_vignette_checkbox.isChecked() or
            self.pp_ai_color_checkbox.isChecked() or
            self.pp_ai_denoise_checkbox.isChecked() or
            self.pp_super_res_checkbox.isChecked() or
            self.pp_upscale_combo.currentIndex() > 0  # Not "1x"
        )

    def _get_postproc_options(self) -> dict:
        """Get current post-processing options from GUI."""
        # Parse Lanczos upscale factor
        upscale_text = self.pp_upscale_combo.currentText()
        if "1.5x" in upscale_text:
            upscale_factor = 1.5
        elif "2x" in upscale_text:
            upscale_factor = 2.0
        elif "3x" in upscale_text:
            upscale_factor = 3.0
        elif "4x" in upscale_text:
            upscale_factor = 4.0
        else:
            upscale_factor = 1.0

        # Parse super resolution scale
        super_res_text = self.pp_super_res_scale_combo.currentText()
        super_res_scale = 4 if "4x" in super_res_text else 2

        return {
            'sharpen': self.pp_sharpen_checkbox.isChecked(),
            'sharpen_amount': self.pp_sharpen_spin.value(),
            'denoise': self.pp_denoise_checkbox.isChecked(),
            'denoise_strength': self.pp_denoise_spin.value(),
            'clahe': self.pp_clahe_checkbox.isChecked(),
            'clahe_strength': self.pp_clahe_spin.value(),
            'shadow_removal': self.pp_shadow_checkbox.isChecked(),
            'shadow_strength': self.pp_shadow_spin.value(),
            'deblur': self.pp_deblur_checkbox.isChecked(),
            'deblur_radius': self.pp_deblur_spin.value(),
            'white_balance': self.pp_white_balance_checkbox.isChecked(),
            'vignette_removal': self.pp_vignette_checkbox.isChecked(),
            'ai_color': self.pp_ai_color_checkbox.isChecked(),
            'ai_denoise': self.pp_ai_denoise_checkbox.isChecked(),
            'super_resolution': self.pp_super_res_checkbox.isChecked(),
            'super_res_scale': super_res_scale,
            'upscale_factor': upscale_factor,
            'interpolation': 'lanczos'
        }

    def _apply_postprocessing(self):
        """Apply post-processing to current result."""
        if not hasattr(self, 'original_result') or self.original_result is None:
            self.log("No panorama result to process")
            return

        options = self._get_postproc_options()
        if not self._should_apply_postproc():
            self.log("No post-processing options enabled")
            return

        try:
            from core.post_processing import ImagePostProcessor
            from core.autopano_features import AIPostProcessor

            self.log("Applying post-processing...")
            self.statusBar().showMessage("Applying post-processing...")
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)

            # Start from original
            result = self.original_result.copy()

            # Apply standard post-processing
            processor = ImagePostProcessor()
            steps_done = 0
            total_steps = sum([
                options.get('shadow_removal', False),
                options.get('denoise', False),
                options.get('deblur', False),
                options.get('clahe', False),
                options.get('sharpen', False),
                options.get('white_balance', False),
                options.get('vignette_removal', False),
                options.get('ai_color', False),
                options.get('ai_denoise', False),
                options.get('super_resolution', False),
                options.get('upscale_factor', 1.0) > 1.0
            ])
            if total_steps == 0:
                total_steps = 1

            # Apply in order: shadow -> white balance -> vignette -> denoise -> deblur -> clahe -> sharpen -> ai -> upscale
            if options.get('shadow_removal'):
                self.log(f"  Fixing exposure (strength={options['shadow_strength']:.1f})...")
                self.statusBar().showMessage("Fixing exposure...")
                result = processor.apply_shadow_removal(result, options['shadow_strength'])
                steps_done += 1
                self.progress_bar.setValue(int(100 * steps_done / total_steps))

            if options.get('white_balance'):
                self.log("  Applying white balance...")
                self.statusBar().showMessage("Applying white balance...")
                result = processor.apply_white_balance(result)
                steps_done += 1
                self.progress_bar.setValue(int(100 * steps_done / total_steps))

            if options.get('vignette_removal'):
                self.log("  Removing vignette...")
                self.statusBar().showMessage("Removing vignette...")
                result = processor.apply_vignette_removal(result)
                steps_done += 1
                self.progress_bar.setValue(int(100 * steps_done / total_steps))

            if options.get('denoise'):
                self.log(f"  Denoising (strength={options['denoise_strength']})...")
                self.statusBar().showMessage("Denoising...")
                result = processor.apply_denoise(result, options['denoise_strength'])
                steps_done += 1
                self.progress_bar.setValue(int(100 * steps_done / total_steps))

            if options.get('deblur'):
                self.log(f"  Deblurring (radius={options['deblur_radius']:.1f})...")
                self.statusBar().showMessage("Deblurring...")
                result = processor.apply_deblur(result, options['deblur_radius'])
                steps_done += 1
                self.progress_bar.setValue(int(100 * steps_done / total_steps))

            if options.get('clahe'):
                self.log(f"  Enhancing contrast (CLAHE strength={options['clahe_strength']:.1f})...")
                self.statusBar().showMessage("Enhancing contrast...")
                result = processor.apply_clahe(result, options['clahe_strength'])
                steps_done += 1
                self.progress_bar.setValue(int(100 * steps_done / total_steps))

            if options.get('sharpen'):
                self.log(f"  Sharpening (amount={options['sharpen_amount']:.1f})...")
                self.statusBar().showMessage("Sharpening...")
                result = processor.apply_sharpen(result, options['sharpen_amount'])
                steps_done += 1
                self.progress_bar.setValue(int(100 * steps_done / total_steps))

            # Apply AI enhancements
            if options.get('ai_color') or options.get('ai_denoise'):
                ai_processor = AIPostProcessor()
                if options.get('ai_color'):
                    self.log("  Applying AI color correction...")
                    self.statusBar().showMessage("AI color correction...")
                    result = ai_processor._color_correct(result)
                    steps_done += 1
                    self.progress_bar.setValue(int(100 * steps_done / total_steps))
                if options.get('ai_denoise'):
                    self.log("  Applying AI denoising...")
                    self.statusBar().showMessage("AI denoising...")
                    result = ai_processor._denoise(result)
                    steps_done += 1
                    self.progress_bar.setValue(int(100 * steps_done / total_steps))

            # Apply AI super resolution (before Lanczos upscaling)
            if options.get('super_resolution'):
                scale = options.get('super_res_scale', 2)
                self.log(f"  Applying Real-ESRGAN super-resolution ({scale}x)...")
                self.statusBar().showMessage(f"AI Super-Resolution {scale}x (this may take a while)...")

                def sr_progress(pct, msg):
                    self.statusBar().showMessage(f"Super-Res: {msg}")

                result = processor.apply_super_resolution(
                    result,
                    scale=scale,
                    model_name='realesrgan-x4plus',
                    tile_size=512,
                    progress_callback=sr_progress
                )
                steps_done += 1
                self.progress_bar.setValue(int(100 * steps_done / total_steps))

            # Apply Lanczos upscaling (can be used in addition to or instead of AI)
            if options.get('upscale_factor', 1.0) > 1.0:
                factor = options['upscale_factor']
                self.log(f"  Upscaling {factor}x (Lanczos)...")
                self.statusBar().showMessage(f"Upscaling {factor}x...")
                result = processor.apply_upscale(result, factor, 'lanczos')
                steps_done += 1
                self.progress_bar.setValue(100)

            # Update result and display
            self.current_result = result
            self.update_preview(result)

            # Log size change
            orig_h, orig_w = self.original_result.shape[:2]
            new_h, new_w = result.shape[:2]
            self.log(f"Post-processing complete! Size: {orig_w}x{orig_h} -> {new_w}x{new_h}")

            self.progress_bar.setVisible(False)
            self.statusBar().showMessage("Post-processing complete", 3000)

        except ImportError as e:
            self.log(f"Post-processing error: Missing module - {e}")
            QMessageBox.warning(self, "Post-Processing Error", f"Missing module:\n{e}")
        except Exception as e:
            self.log(f"Post-processing error: {e}")
            QMessageBox.warning(self, "Post-Processing Error", str(e))
            import traceback
            traceback.print_exc()
        finally:
            self.progress_bar.setVisible(False)

    def _reset_postprocessing(self):
        """Reset to original panorama (before post-processing)."""
        if not hasattr(self, 'original_result') or self.original_result is None:
            self.log("No original result to restore")
            return

        self.current_result = self.original_result.copy()
        self.update_preview(self.current_result)
        self.log("Reset to original panorama (post-processing removed)")
        self.statusBar().showMessage("Reset to original", 2000)

    def _run_hloc(self):
        """Run HLOC pipeline on current images."""
        if not self.image_paths:
            QMessageBox.warning(self, "No Images", "Please load images first.")
            return
        
        try:
            from external.pipelines import HLOCPipeline
            
            pipeline = HLOCPipeline(progress_callback=self._update_progress_slot)
            
            if not pipeline.is_available():
                reply = QMessageBox.question(
                    self, "HLOC Not Found",
                    "HLOC is not installed.\n\n" + pipeline.get_install_instructions() +
                    "\n\nWould you like to open the HLOC GitHub page?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.Yes:
                    import webbrowser
                    webbrowser.open("https://github.com/cvg/Hierarchical-Localization")
                return
            
            # Get output directory
            output_dir = QFileDialog.getExistingDirectory(
                self, "Select Output Directory for HLOC"
            )
            if not output_dir:
                return
            
            self.log("Starting HLOC pipeline (SuperPoint + SuperGlue)...")
            self.progress_bar.setValue(0)
            
            result = pipeline.run(self.image_paths, Path(output_dir))
            
            if result.get("success"):
                self.log(f"HLOC complete! {result.get('n_images', 0)} images processed")
                QMessageBox.information(
                    self, "HLOC Complete",
                    f"Reconstruction complete!\n\n"
                    f"Images: {result.get('n_images', 0)}\n"
                    f"Output: {result.get('workspace', '')}"
                )
            else:
                self.log(f"HLOC failed: {result.get('message', 'Unknown error')}")
                
        except Exception as e:
            logger.error(f"HLOC error: {e}", exc_info=True)
            QMessageBox.critical(self, "HLOC Error", str(e))
    
    def _run_meshroom(self):
        """Run AliceVision/Meshroom pipeline or open Meshroom GUI."""
        if not self.image_paths:
            QMessageBox.warning(self, "No Images", "Please load images first.")
            return
        
        try:
            from external.pipelines import AliceVisionPipeline
            
            pipeline = AliceVisionPipeline()
            
            if not pipeline.is_available():
                reply = QMessageBox.question(
                    self, "Meshroom Not Found",
                    "Meshroom/AliceVision is not installed.\n\n" + 
                    pipeline.get_install_instructions() +
                    "\n\nWould you like to open the Meshroom download page?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.Yes:
                    import webbrowser
                    webbrowser.open("https://alicevision.org/#meshroom")
                return
            
            # For now, suggest using Meshroom GUI directly
            QMessageBox.information(
                self, "Meshroom",
                "For best results, please use Meshroom GUI directly:\n\n"
                "1. Open Meshroom\n"
                "2. Drag your images into the Images panel\n"
                "3. Click 'Start' to begin reconstruction\n\n"
                "Meshroom provides a visual pipeline editor and\n"
                "real-time progress monitoring."
            )
            
        except Exception as e:
            logger.error(f"Meshroom error: {e}", exc_info=True)
            QMessageBox.critical(self, "Meshroom Error", str(e))
    
    def _update_progress_slot(self, percent: int, message: str):
        """Progress callback for external pipelines."""
        self.progress_bar.setValue(percent)
        if message:
            self.log(message)

    def run_hloc_pipeline(self):
        """Run HLOC pipeline for reconstruction."""
        try:
            # Get output directory
            output_dir = QFileDialog.getExistingDirectory(
                self, "Select Output Directory for HLOC"
            )
            if not output_dir:
                return
            
            self.log("Starting HLOC pipeline (SuperPoint + SuperGlue)...")
            self.progress_bar.setValue(0)
            
            result = pipeline.run(self.image_paths, Path(output_dir))
            
            if result.get("success"):
                self.log(f"HLOC complete! {result.get('n_images', 0)} images processed")
                QMessageBox.information(
                    self, "HLOC Complete",
                    f"Reconstruction complete!\n\n"
                    f"Images: {result.get('n_images', 0)}\n"
                    f"Output: {result.get('workspace', '')}"
                )
            else:
                self.log(f"HLOC failed: {result.get('message', 'Unknown error')}")
                
        except Exception as e:
            logger.error(f"HLOC error: {e}", exc_info=True)
            QMessageBox.critical(self, "HLOC Error", str(e))
    
    def _run_meshroom(self):
        """Run AliceVision/Meshroom pipeline or open Meshroom GUI."""
        if not self.image_paths:
            QMessageBox.warning(self, "No Images", "Please load images first.")
            return
        
        try:
            from external.pipelines import AliceVisionPipeline
            
            pipeline = AliceVisionPipeline()
            
            if not pipeline.is_available():
                reply = QMessageBox.question(
                    self, "Meshroom Not Found",
                    "Meshroom/AliceVision is not installed.\n\n" + 
                    pipeline.get_install_instructions() +
                    "\n\nWould you like to open the Meshroom download page?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.Yes:
                    import webbrowser
                    webbrowser.open("https://alicevision.org/#meshroom")
                return
            
            # For now, suggest using Meshroom GUI directly
            QMessageBox.information(
                self, "Meshroom",
                "For best results, please use Meshroom GUI directly:\n\n"
                "1. Open Meshroom\n"
                "2. Drag your images into the Images panel\n"
                "3. Click 'Start' to begin reconstruction\n\n"
                "Meshroom provides a visual pipeline editor and\n"
                "real-time progress monitoring."
            )
            
        except Exception as e:
            logger.error(f"Meshroom error: {e}", exc_info=True)
            QMessageBox.critical(self, "Meshroom Error", str(e))
    
    def _update_progress_slot(self, percent: int, message: str):
        """Progress callback for external pipelines."""
        self.progress_bar.setValue(percent)
        if message:
            self.log(message)

    def run_hloc_pipeline(self):
        """Run HLOC pipeline for reconstruction."""
        try:
            # Get output directory
            output_dir = QFileDialog.getExistingDirectory(
                self, "Select Output Directory for HLOC"
            )
            if not output_dir:
                return
            
            self.log("Starting HLOC pipeline (SuperPoint + SuperGlue)...")
            self.progress_bar.setValue(0)
            
            result = pipeline.run(self.image_paths, Path(output_dir))
            
            if result.get("success"):
                self.log(f"HLOC complete! {result.get('n_images', 0)} images processed")
                QMessageBox.information(
                    self, "HLOC Complete",
                    f"Reconstruction complete!\n\n"
                    f"Images: {result.get('n_images', 0)}\n"
                    f"Output: {result.get('workspace', '')}"
                )
            else:
                self.log(f"HLOC failed: {result.get('message', 'Unknown error')}")
                
        except Exception as e:
            logger.error(f"HLOC error: {e}", exc_info=True)
            QMessageBox.critical(self, "HLOC Error", str(e))
    
    def _run_meshroom(self):
        """Run AliceVision/Meshroom pipeline or open Meshroom GUI."""
        if not self.image_paths:
            QMessageBox.warning(self, "No Images", "Please load images first.")
            return
        
        try:
            from external.pipelines import AliceVisionPipeline
            
            pipeline = AliceVisionPipeline()
            
            if not pipeline.is_available():
                reply = QMessageBox.question(
                    self, "Meshroom Not Found",
                    "Meshroom/AliceVision is not installed.\n\n" + 
                    pipeline.get_install_instructions() +
                    "\n\nWould you like to open the Meshroom download page?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.Yes:
                    import webbrowser
                    webbrowser.open("https://alicevision.org/#meshroom")
                return
            
            # For now, suggest using Meshroom GUI directly
            QMessageBox.information(
                self, "Meshroom",
                "For best results, please use Meshroom GUI directly:\n\n"
                "1. Open Meshroom\n"
                "2. Drag your images into the Images panel\n"
                "3. Click 'Start' to begin reconstruction\n\n"
                "Meshroom provides a visual pipeline editor and\n"
                "real-time progress monitoring."
            )
            
        except Exception as e:
            logger.error(f"Meshroom error: {e}", exc_info=True)
            QMessageBox.critical(self, "Meshroom Error", str(e))
    
    def _update_progress_slot(self, percent: int, message: str):
        """Progress callback for external pipelines."""
        self.progress_bar.setValue(percent)
        if message:
            self.log(message)

    def run_hloc_pipeline(self):
        """Run HLOC pipeline for reconstruction."""
        try:
            # Get output directory
            output_dir = QFileDialog.getExistingDirectory(
                self, "Select Output Directory for HLOC"
            )
            if not output_dir:
                return
            
            self.log("Starting HLOC pipeline (SuperPoint + SuperGlue)...")
            self.progress_bar.setValue(0)
            
            result = pipeline.run(self.image_paths, Path(output_dir))
            
            if result.get("success"):
                self.log(f"HLOC complete! {result.get('n_images', 0)} images processed")
                QMessageBox.information(
                    self, "HLOC Complete",
                    f"Reconstruction complete!\n\n"
                    f"Images: {result.get('n_images', 0)}\n"
                    f"Output: {result.get('workspace', '')}"
                )
            else:
                self.log(f"HLOC failed: {result.get('message', 'Unknown error')}")
                
        except Exception as e:
            logger.error(f"HLOC error: {e}", exc_info=True)
            QMessageBox.critical(self, "HLOC Error", str(e))
    
    def _run_meshroom(self):
        """Run AliceVision/Meshroom pipeline or open Meshroom GUI."""
        if not self.image_paths:
            QMessageBox.warning(self, "No Images", "Please load images first.")
            return
        
        try:
            from external.pipelines import AliceVisionPipeline
            
            pipeline = AliceVisionPipeline()
            
            if not pipeline.is_available():
                reply = QMessageBox.question(
                    self, "Meshroom Not Found",
                    "Meshroom/AliceVision is not installed.\n\n" + 
                    pipeline.get_install_instructions() +
                    "\n\nWould you like to open the Meshroom download page?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.Yes:
                    import webbrowser
                    webbrowser.open("https://alicevision.org/#meshroom")
                return
            
            # For now, suggest using Meshroom GUI directly
            QMessageBox.information(
                self, "Meshroom",
                "For best results, please use Meshroom GUI directly:\n\n"
                "1. Open Meshroom\n"
                "2. Drag your images into the Images panel\n"
                "3. Click 'Start' to begin reconstruction\n\n"
                "Meshroom provides a visual pipeline editor and\n"
                "real-time progress monitoring."
            )
            
        except Exception as e:
            logger.error(f"Meshroom error: {e}", exc_info=True)
            QMessageBox.critical(self, "Meshroom Error", str(e))
    
    def _update_progress_slot(self, percent: int, message: str):
        """Progress callback for external pipelines."""
        self.progress_bar.setValue(percent)
        if message:
            self.log(message)

    def run_hloc_pipeline(self):
        """Run HLOC pipeline for reconstruction."""
        try:
            # Get output directory
            output_dir = QFileDialog.getExistingDirectory(
                self, "Select Output Directory for HLOC"
            )
            if not output_dir:
                return
            
            self.log("Starting HLOC pipeline (SuperPoint + SuperGlue)...")
            self.progress_bar.setValue(0)
            
            result = pipeline.run(self.image_paths, Path(output_dir))
            
            if result.get("success"):
                self.log(f"HLOC complete! {result.get('n_images', 0)} images processed")
                QMessageBox.information(
                    self, "HLOC Complete",
                    f"Reconstruction complete!\n\n"
                    f"Images: {result.get('n_images', 0)}\n"
                    f"Output: {result.get('workspace', '')}"
                )
            else:
                self.log(f"HLOC failed: {result.get('message', 'Unknown error')}")
                
        except Exception as e:
            logger.error(f"HLOC error: {e}", exc_info=True)
            QMessageBox.critical(self, "HLOC Error", str(e))
    
    def _run_meshroom(self):
        """Run AliceVision/Meshroom pipeline or open Meshroom GUI."""
        if not self.image_paths:
            QMessageBox.warning(self, "No Images", "Please load images first.")
            return
        
        try:
            from external.pipelines import AliceVisionPipeline
            
            pipeline = AliceVisionPipeline()
            
            if not pipeline.is_available():
                reply = QMessageBox.question(
                    self, "Meshroom Not Found",
                    "Meshroom/AliceVision is not installed.\n\n" + 
                    pipeline.get_install_instructions() +
                    "\n\nWould you like to open the Meshroom download page?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.Yes:
                    import webbrowser
                    webbrowser.open("https://alicevision.org/#meshroom")
                return
            
            # For now, suggest using Meshroom GUI directly
            QMessageBox.information(
                self, "Meshroom",
                "For best results, please use Meshroom GUI directly:\n\n"
                "1. Open Meshroom\n"
                "2. Drag your images into the Images panel\n"
                "3. Click 'Start' to begin reconstruction\n\n"
                "Meshroom provides a visual pipeline editor and\n"
                "real-time progress monitoring."
            )
            
        except Exception as e:
            logger.error(f"Meshroom error: {e}", exc_info=True)
            QMessageBox.critical(self, "Meshroom Error", str(e))
    
    def _update_progress_slot(self, percent: int, message: str):
        """Progress callback for external pipelines."""
        self.progress_bar.setValue(percent)
        if message:
            self.log(message)

    def run_hloc_pipeline(self):
        """Run HLOC pipeline for reconstruction."""
        try:
            # Get output directory
            output_dir = QFileDialog.getExistingDirectory(
                self, "Select Output Directory for HLOC"
            )
            if not output_dir:
                return
            
            self.log("Starting HLOC pipeline (SuperPoint + SuperGlue)...")
            self.progress_bar.setValue(0)
            
            result = pipeline.run(self.image_paths, Path(output_dir))
            
            if result.get("success"):
                self.log(f"HLOC complete! {result.get('n_images', 0)} images processed")
                QMessageBox.information(
                    self, "HLOC Complete",
                    f"Reconstruction complete!\n\n"
                    f"Images: {result.get('n_images', 0)}\n"
                    f"Output: {result.get('workspace', '')}"
                )
            else:
                self.log(f"HLOC failed: {result.get('message', 'Unknown error')}")
                
        except Exception as e:
            logger.error(f"HLOC error: {e}", exc_info=True)
            QMessageBox.critical(self, "HLOC Error", str(e))
    
    def _run_meshroom(self):
        """Run AliceVision/Meshroom pipeline or open Meshroom GUI."""
        if not self.image_paths:
            QMessageBox.warning(self, "No Images", "Please load images first.")
            return
        
        try:
            from external.pipelines import AliceVisionPipeline
            
            pipeline = AliceVisionPipeline()
            
            if not pipeline.is_available():
                reply = QMessageBox.question(
                    self, "Meshroom Not Found",
                    "Meshroom/AliceVision is not installed.\n\n" + 
                    pipeline.get_install_instructions() +
                    "\n\nWould you like to open the Meshroom download page?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.Yes:
                    import webbrowser
                    webbrowser.open("https://alicevision.org/#meshroom")
                return
            
            # For now, suggest using Meshroom GUI directly
            QMessageBox.information(
                self, "Meshroom",
                "For best results, please use Meshroom GUI directly:\n\n"
                "1. Open Meshroom\n"
                "2. Drag your images into the Images panel\n"
                "3. Click 'Start' to begin reconstruction\n\n"
                "Meshroom provides a visual pipeline editor and\n"
                "real-time progress monitoring."
            )
            
        except Exception as e:
            logger.error(f"Meshroom error: {e}", exc_info=True)
            QMessageBox.critical(self, "Meshroom Error", str(e))
    
    def _update_progress_slot(self, percent: int, message: str):
        """Progress callback for external pipelines."""
        self.progress_bar.setValue(percent)
        if message:
            self.log(message)

