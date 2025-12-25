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
    QScrollArea
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QPixmap, QImage, QFont

from core.stitcher import ImageStitcher
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


class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.image_paths: List[Path] = []
        self.stitcher: Optional[ImageStitcher] = None
        self.thread: Optional[StitchingThread] = None
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
        
        # Matching memory optimization
        match_mem_layout = QHBoxLayout()
        match_mem_layout.addWidget(QLabel("Matching Memory:"))
        self.matching_memory_combo = QComboBox()
        self.matching_memory_combo.addItems([
            "Balanced (Recommended)",
            "Quality First",
            "Minimal Memory",
            "Standard (No Optimization)"
        ])
        self.matching_memory_combo.setCurrentIndex(0)
        self.matching_memory_combo.setToolTip(
            "Memory optimization for feature matching:\n"
            "• Balanced: PCA compression + cascade filter (~60% savings)\n"
            "• Quality First: PCA only, all pairs matched (~30% savings)\n"
            "• Minimal Memory: Maximum optimization + disk caching (~80% savings)\n"
            "• Standard: No optimization (use if matching seems incorrect)"
        )
        match_mem_layout.addWidget(self.matching_memory_combo)
        settings_layout.addLayout(match_mem_layout)
        
        # Duplicate detection
        dup_layout = QHBoxLayout()
        self.remove_duplicates_checkbox = QCheckBox("Remove Duplicates")
        self.remove_duplicates_checkbox.setChecked(False)
        self.remove_duplicates_checkbox.setToolTip(
            "Pre-scan images to detect and remove duplicates/similar photos.\n"
            "Reduces memory usage and improves stitching quality.\n"
            "Uses perceptual hashing and histogram comparison."
        )
        dup_layout.addWidget(self.remove_duplicates_checkbox)
        
        dup_layout.addWidget(QLabel("Threshold:"))
        self.duplicate_threshold_spin = QDoubleSpinBox()
        self.duplicate_threshold_spin.setRange(0.80, 0.99)
        self.duplicate_threshold_spin.setSingleStep(0.01)
        self.duplicate_threshold_spin.setValue(0.92)
        self.duplicate_threshold_spin.setDecimals(2)
        self.duplicate_threshold_spin.setToolTip(
            "Similarity threshold for duplicate detection.\n"
            "0.90 = 90% similar (catches more duplicates)\n"
            "0.95 = 95% similar (only near-identical)\n"
            "0.99 = 99% similar (only exact duplicates)"
        )
        dup_layout.addWidget(self.duplicate_threshold_spin)
        settings_layout.addLayout(dup_layout)
        
        # Feature detector
        detector_layout = QHBoxLayout()
        detector_layout.addWidget(QLabel("Feature Detector:"))
        self.detector_combo = QComboBox()
        self.detector_combo.addItems([
            "LP-SIFT (Recommended)",
            "SIFT",
            "ORB",
            "AKAZE"
        ])
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
        
        # Allow scale/rotation checkbox
        self.allow_scale_checkbox = QCheckBox("Allow Scale & Rotation")
        self.allow_scale_checkbox.setChecked(True)
        self.allow_scale_checkbox.setToolTip("Allow images to be scaled and rotated to match at overlaps.\nUseful when images have different resolutions, zoom levels, or orientations.")
        settings_layout.addWidget(self.allow_scale_checkbox)
        
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
            "AutoStitch (Simple & Fast)"
        ])
        blend_layout.addWidget(self.blend_combo)
        settings_layout.addLayout(blend_layout)
        
        # Blending options
        blend_options_group = QGroupBox("Blending Options")
        blend_options_layout = QVBoxLayout()
        
        self.hdr_checkbox = QCheckBox("HDR Mode (Exposure Fusion)")
        self.hdr_checkbox.setToolTip("Combine multiple exposures for better dynamic range")
        blend_options_layout.addWidget(self.hdr_checkbox)
        
        self.antighost_checkbox = QCheckBox("Anti-Ghosting")
        self.antighost_checkbox.setToolTip("Reduce ghosting artifacts in overlapping regions")
        blend_options_layout.addWidget(self.antighost_checkbox)
        
        pixel_select_layout = QHBoxLayout()
        pixel_select_layout.addWidget(QLabel("Pixel Selection:"))
        self.pixel_select_combo = QComboBox()
        self.pixel_select_combo.addItems([
            "Weighted Average (Default)",
            "Strongest Signal",
            "Median",
            "Maximum",
            "Minimum"
        ])
        self.pixel_select_combo.setToolTip("Method for selecting pixel values in overlapping regions")
        pixel_select_layout.addWidget(self.pixel_select_combo)
        blend_options_layout.addLayout(pixel_select_layout)
        
        blend_options_group.setLayout(blend_options_layout)
        settings_layout.addWidget(blend_options_group)
        
        # Max images
        max_img_layout = QHBoxLayout()
        max_img_layout.addWidget(QLabel("Max Images:"))
        self.max_images_spin = QSpinBox()
        self.max_images_spin.setRange(0, 10000)
        self.max_images_spin.setValue(0)
        self.max_images_spin.setSpecialValueText("Unlimited")
        max_img_layout.addWidget(self.max_images_spin)
        settings_layout.addLayout(max_img_layout)
        
        # Overlap threshold for grid alignment
        overlap_layout = QHBoxLayout()
        overlap_layout.addWidget(QLabel("Min Overlap:"))
        self.overlap_spin = QDoubleSpinBox()
        self.overlap_spin.setRange(0.0, 100.0)
        self.overlap_spin.setSingleStep(1.0)
        self.overlap_spin.setValue(10.0)  # Default 10%
        self.overlap_spin.setSuffix("%")
        self.overlap_spin.setDecimals(1)
        self.overlap_spin.setToolTip("Minimum overlap percentage for images to be included in grid alignment")
        overlap_layout.addWidget(self.overlap_spin)
        
        overlap_layout.addWidget(QLabel("Max:"))
        self.max_overlap_spin = QDoubleSpinBox()
        self.max_overlap_spin.setRange(0.0, 100.0)
        self.max_overlap_spin.setSingleStep(5.0)
        self.max_overlap_spin.setValue(100.0)  # Default 100% (no max limit)
        self.max_overlap_spin.setSuffix("%")
        self.max_overlap_spin.setDecimals(1)
        self.max_overlap_spin.setSpecialValueText("No limit")
        self.max_overlap_spin.setToolTip(
            "Maximum overlap percentage (filters out near-duplicates)\n"
            "100% = no limit, 90% = exclude >90% overlap (likely duplicates)"
        )
        overlap_layout.addWidget(self.max_overlap_spin)
        settings_layout.addLayout(overlap_layout)
        
        # Grid spacing for exploded view
        spacing_layout = QHBoxLayout()
        spacing_layout.addWidget(QLabel("Grid Spacing:"))
        self.grid_spacing_spin = QDoubleSpinBox()
        self.grid_spacing_spin.setRange(1.0, 3.0)
        self.grid_spacing_spin.setSingleStep(0.1)
        self.grid_spacing_spin.setValue(1.3)  # Default 30% gap
        self.grid_spacing_spin.setDecimals(1)
        self.grid_spacing_spin.setToolTip("Spacing between images in grid view.\n1.0 = touching, 1.5 = 50% gap between images")
        spacing_layout.addWidget(self.grid_spacing_spin)
        settings_layout.addLayout(spacing_layout)
        
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
        
        # Output DPI
        dpi_layout = QHBoxLayout()
        dpi_layout.addWidget(QLabel("DPI:"))
        self.output_dpi_spin = QSpinBox()
        self.output_dpi_spin.setRange(72, 1200)
        self.output_dpi_spin.setValue(300)
        self.output_dpi_spin.setSingleStep(50)
        self.output_dpi_spin.setToolTip("Output resolution in dots per inch.\n72 = screen, 150 = web, 300 = print, 600+ = high quality print")
        dpi_layout.addWidget(self.output_dpi_spin)
        output_layout.addLayout(dpi_layout)
        
        # Max panorama size (memory limit)
        max_pano_layout = QHBoxLayout()
        max_pano_layout.addWidget(QLabel("Max Panorama:"))
        self.max_panorama_spin = QSpinBox()
        self.max_panorama_spin.setRange(0, 500)
        self.max_panorama_spin.setValue(100)
        self.max_panorama_spin.setSingleStep(25)
        self.max_panorama_spin.setSuffix(" MP")
        self.max_panorama_spin.setSpecialValueText("Unlimited")
        self.max_panorama_spin.setToolTip(
            "Maximum panorama size in megapixels (0 = unlimited)\n"
            "Memory guidelines for blending:\n"
            "• 50MP: ~600MB (AutoStitch) / ~2.4GB (Multiband)\n"
            "• 100MP: ~1.2GB (AutoStitch) / ~4.8GB (Multiband)\n"
            "• 200MP: ~2.4GB (AutoStitch) / ~9.6GB (Multiband)\n"
            "Recommended: RAM(GB) × 15 for Multiband, RAM(GB) × 50 for AutoStitch"
        )
        max_pano_layout.addWidget(self.max_panorama_spin)
        output_layout.addLayout(max_pano_layout)
        
        # Max warp size
        max_warp_layout = QHBoxLayout()
        max_warp_layout.addWidget(QLabel("Max Warp:"))
        self.max_warp_spin = QSpinBox()
        self.max_warp_spin.setRange(0, 200)
        self.max_warp_spin.setValue(50)
        self.max_warp_spin.setSingleStep(10)
        self.max_warp_spin.setSuffix(" MP")
        self.max_warp_spin.setSpecialValueText("Unlimited")
        self.max_warp_spin.setToolTip(
            "Maximum size per warped image in megapixels (0 = unlimited)\n"
            "Memory: ~3MB per megapixel (RGB)\n"
            "• 50MP: ~150MB per image\n"
            "• 100MP: ~300MB per image\n"
            "Only applies when rotation/scale is used."
        )
        max_warp_layout.addWidget(self.max_warp_spin)
        output_layout.addLayout(max_warp_layout)
        
        output_group.setLayout(output_layout)
        settings_layout.addWidget(output_group)
        
        # Post-processing options
        postproc_group = QGroupBox("Post-Processing")
        postproc_layout = QVBoxLayout()
        
        # Sharpening
        sharpen_layout = QHBoxLayout()
        self.sharpen_checkbox = QCheckBox("Sharpen")
        self.sharpen_checkbox.setToolTip("Apply unsharp mask sharpening to enhance details")
        sharpen_layout.addWidget(self.sharpen_checkbox)
        
        self.sharpen_amount_spin = QDoubleSpinBox()
        self.sharpen_amount_spin.setRange(0.0, 3.0)
        self.sharpen_amount_spin.setValue(1.0)
        self.sharpen_amount_spin.setSingleStep(0.1)
        self.sharpen_amount_spin.setPrefix("Amount: ")
        self.sharpen_amount_spin.setToolTip("Sharpening strength (0.5=subtle, 1.0=normal, 2.0=strong)")
        sharpen_layout.addWidget(self.sharpen_amount_spin)
        postproc_layout.addLayout(sharpen_layout)
        
        # Denoising
        denoise_layout = QHBoxLayout()
        self.denoise_checkbox = QCheckBox("Denoise")
        self.denoise_checkbox.setToolTip("Remove noise using non-local means denoising")
        denoise_layout.addWidget(self.denoise_checkbox)
        
        self.denoise_strength_spin = QSpinBox()
        self.denoise_strength_spin.setRange(1, 20)
        self.denoise_strength_spin.setValue(5)
        self.denoise_strength_spin.setPrefix("h: ")
        self.denoise_strength_spin.setToolTip("Filter strength (3=light, 5=medium, 10=strong)")
        denoise_layout.addWidget(self.denoise_strength_spin)
        postproc_layout.addLayout(denoise_layout)
        
        # Shadow removal / exposure equalization
        shadow_layout = QHBoxLayout()
        self.shadow_checkbox = QCheckBox("Remove Shadows")
        self.shadow_checkbox.setToolTip("Equalize exposure and remove shadows across the image")
        shadow_layout.addWidget(self.shadow_checkbox)
        
        self.shadow_strength_spin = QDoubleSpinBox()
        self.shadow_strength_spin.setRange(0.1, 1.0)
        self.shadow_strength_spin.setValue(0.5)
        self.shadow_strength_spin.setSingleStep(0.1)
        self.shadow_strength_spin.setPrefix("Strength: ")
        self.shadow_strength_spin.setToolTip("Shadow removal strength (0.1=subtle, 0.5=medium, 1.0=full)")
        shadow_layout.addWidget(self.shadow_strength_spin)
        postproc_layout.addLayout(shadow_layout)
        
        # Contrast enhancement
        clahe_layout = QHBoxLayout()
        self.clahe_checkbox = QCheckBox("Enhance Contrast")
        self.clahe_checkbox.setToolTip("CLAHE - Adaptive histogram equalization for improved local contrast")
        clahe_layout.addWidget(self.clahe_checkbox)
        
        self.clahe_strength_spin = QDoubleSpinBox()
        self.clahe_strength_spin.setRange(1.0, 5.0)
        self.clahe_strength_spin.setValue(2.0)
        self.clahe_strength_spin.setSingleStep(0.5)
        self.clahe_strength_spin.setPrefix("Clip: ")
        self.clahe_strength_spin.setToolTip("Contrast clip limit (1.0=subtle, 2.0=default, 4.0=strong)")
        clahe_layout.addWidget(self.clahe_strength_spin)
        postproc_layout.addLayout(clahe_layout)
        
        # Deblur option
        deblur_layout = QHBoxLayout()
        self.deblur_checkbox = QCheckBox("Deblur")
        self.deblur_checkbox.setToolTip("Apply Wiener deconvolution to reduce blur")
        deblur_layout.addWidget(self.deblur_checkbox)
        
        self.deblur_strength_spin = QDoubleSpinBox()
        self.deblur_strength_spin.setRange(0.5, 5.0)
        self.deblur_strength_spin.setValue(1.5)
        self.deblur_strength_spin.setSingleStep(0.5)
        self.deblur_strength_spin.setPrefix("Radius: ")
        self.deblur_strength_spin.setToolTip("Blur radius estimate (larger = stronger correction)")
        deblur_layout.addWidget(self.deblur_strength_spin)
        postproc_layout.addLayout(deblur_layout)
        
        # Upscaling
        upscale_layout = QHBoxLayout()
        upscale_layout.addWidget(QLabel("Upscale:"))
        self.upscale_combo = QComboBox()
        self.upscale_combo.addItems([
            "1x (Original)",
            "1.5x",
            "2x",
            "3x",
            "4x"
        ])
        self.upscale_combo.setToolTip("Upscale output using high-quality Lanczos interpolation")
        upscale_layout.addWidget(self.upscale_combo)
        postproc_layout.addLayout(upscale_layout)
        
        # Interpolation quality
        interp_layout = QHBoxLayout()
        interp_layout.addWidget(QLabel("Warp Quality:"))
        self.interp_combo = QComboBox()
        self.interp_combo.addItems([
            "Lanczos (Best)",
            "Cubic (High)",
            "Linear (Fast)",
            "Nearest (Fastest)"
        ])
        self.interp_combo.setToolTip("Interpolation method used during image warping.\nLanczos is highest quality but slower.")
        interp_layout.addWidget(self.interp_combo)
        postproc_layout.addLayout(interp_layout)
        
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
        
        # Clear preview button
        preview_controls = QHBoxLayout()
        self.btn_clear_preview = QPushButton("Clear Preview")
        self.btn_clear_preview.clicked.connect(self.clear_preview)
        preview_controls.addWidget(self.btn_clear_preview)
        preview_controls.addStretch()
        preview_layout.addLayout(preview_controls)
        
        self.preview_label = QLabel("No preview available")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(800, 600)
        self.preview_label.setStyleSheet("border: 1px solid gray;")
        preview_layout.addWidget(self.preview_label)
        
        tabs.addTab(preview_tab, "Preview")
        
        # Logs tab
        logs_tab = QWidget()
        logs_layout = QVBoxLayout(logs_tab)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Courier", 9))
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
                "AutoStitch (Simple & Fast)": "autostitch"
            }
            
            feature_detector = detector_map.get(self.detector_combo.currentText(), "lp_sift")
            feature_matcher = matcher_map.get(self.matcher_combo.currentText(), "flann")
            blending_method = blender_map.get(self.blend_combo.currentText(), "multiband")
            max_features = self.max_features_spin.value()
            
            # Get blending options
            blending_options = {
                'hdr_mode': self.hdr_checkbox.isChecked(),
                'anti_ghosting': self.antighost_checkbox.isChecked(),
                'pixel_selection': self.pixel_select_combo.currentText().split('(')[0].strip().lower().replace(' ', '_')
            }
            
            allow_scale = self.allow_scale_checkbox.isChecked()
            
            # Get memory limit settings (convert MP to pixels, 0 = None for unlimited)
            max_panorama_mp = self.max_panorama_spin.value()
            max_panorama_pixels = max_panorama_mp * 1_000_000 if max_panorama_mp > 0 else None
            
            max_warp_mp = self.max_warp_spin.value()
            max_warp_pixels = max_warp_mp * 1_000_000 if max_warp_mp > 0 else None
            
            memory_efficient = self.memory_efficient_checkbox.isChecked()
            remove_duplicates = self.remove_duplicates_checkbox.isChecked()
            duplicate_threshold = self.duplicate_threshold_spin.value()
            
            # Get matching memory mode from combo box
            matching_memory_modes = ['balanced', 'quality', 'minimal', 'standard']
            matching_memory_mode = matching_memory_modes[self.matching_memory_combo.currentIndex()]
            
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
                remove_duplicates=remove_duplicates,
                duplicate_threshold=duplicate_threshold,
                matching_memory_mode=matching_memory_mode
            )
            logger.info(f"Stitcher initialized (max_panorama={max_panorama_mp}MP, max_warp={max_warp_mp}MP, "
                       f"memory_efficient={memory_efficient}, matching_mode={matching_memory_mode})")
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
        self.log("Preview cleared")
    
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
            self.update_preview(panorama)
            
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
    
    def update_preview(self, image: np.ndarray):
        """Update preview with image"""
        try:
            # Validate input
            if image is None:
                logger.warning("Cannot update preview: image is None")
                self.preview_label.setText("No preview available")
                return
            
            if not isinstance(image, np.ndarray):
                logger.warning(f"Cannot update preview: invalid type {type(image)}")
                self.preview_label.setText("Invalid image data")
                return
            
            if image.size == 0:
                logger.warning("Cannot update preview: image is empty")
                self.preview_label.setText("Empty image")
                return
            
            # Resize for preview
            h, w = image.shape[:2]
            if w == 0 or h == 0:
                logger.warning(f"Invalid image dimensions: {w}x{h}")
                self.preview_label.setText("Invalid image dimensions")
                return
            
            max_size = 800
            scale = min(max_size / w, max_size / h)
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            
            if len(image.shape) == 3:
                if image.shape[2] != 3:
                    logger.warning(f"Unexpected image shape: {image.shape}")
                    # Convert to 3-channel if needed
                    if image.shape[2] == 1:
                        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                    elif image.shape[2] == 4:
                        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
                
                resized = cv2.resize(image, (new_w, new_h))
                # Convert BGR to RGB for Qt (OpenCV uses BGR)
                resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                # Ensure contiguous array and correct dtype
                resized_rgb = np.ascontiguousarray(resized_rgb, dtype=np.uint8)
                if resized_rgb.size == 0:
                    logger.warning("Resized image is empty")
                    self.preview_label.setText("Failed to create preview")
                    return
                bytes_per_line = resized_rgb.strides[0]
                qimage = QImage(resized_rgb.data, new_w, new_h, bytes_per_line, QImage.Format.Format_RGB888)
            else:
                resized = cv2.resize(image, (new_w, new_h))
                resized = np.ascontiguousarray(resized, dtype=np.uint8)
                if resized.size == 0:
                    logger.warning("Resized image is empty")
                    self.preview_label.setText("Failed to create preview")
                    return
                bytes_per_line = resized.strides[0]
                qimage = QImage(resized.data, new_w, new_h, bytes_per_line, QImage.Format.Format_Grayscale8)
            
            if qimage.isNull():
                logger.warning("Failed to create QImage from numpy array")
                self.preview_label.setText("Failed to create preview")
                return
            
            pixmap = QPixmap.fromImage(qimage)
            if pixmap.isNull():
                logger.warning("Failed to create QPixmap from QImage")
                self.preview_label.setText("Failed to create preview")
                return
            
            self.preview_label.setPixmap(pixmap)
        except Exception as e:
            logger.error(f"Error updating preview: {e}", exc_info=True)
            self.preview_label.setText(f"Preview error: {str(e)}")
    
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

