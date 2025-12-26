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
        
        # Match filtering group
        filtering_group = QGroupBox("Image Filtering & Optimization")
        filtering_layout = QVBoxLayout()
        
        # Row 1: Checkboxes
        filter_checks_layout = QHBoxLayout()
        self.geo_verify_checkbox = QCheckBox("RANSAC")
        self.geo_verify_checkbox.setChecked(True)
        self.geo_verify_checkbox.setToolTip("Filter bad matches with geometric verification")
        filter_checks_layout.addWidget(self.geo_verify_checkbox)
        
        self.remove_duplicates_checkbox = QCheckBox("Dedup")
        self.remove_duplicates_checkbox.setChecked(True)
        self.remove_duplicates_checkbox.setToolTip("Remove duplicate burst photos")
        filter_checks_layout.addWidget(self.remove_duplicates_checkbox)
        
        self.optimal_coverage_checkbox = QCheckBox("Coverage")
        self.optimal_coverage_checkbox.setChecked(False)
        self.optimal_coverage_checkbox.setToolTip("Keep only necessary images")
        filter_checks_layout.addWidget(self.optimal_coverage_checkbox)
        
        self.optimize_alignment_checkbox = QCheckBox("Optimize Align")
        self.optimize_alignment_checkbox.setChecked(False)
        self.optimize_alignment_checkbox.setToolTip(
            "Preprocess images for better feature detection (like AutoPano Giga).\n"
            "Improves alignment for low-contrast or challenging images.\n"
            "Original images are preserved for final blending."
        )
        filter_checks_layout.addWidget(self.optimize_alignment_checkbox)
        filter_checks_layout.addStretch()
        filtering_layout.addLayout(filter_checks_layout)
        
        # Row 2: Thresholds and options
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Dup:"))
        self.duplicate_threshold_spin = QDoubleSpinBox()
        self.duplicate_threshold_spin.setRange(0.70, 0.98)
        self.duplicate_threshold_spin.setValue(0.88)
        self.duplicate_threshold_spin.setDecimals(2)
        self.duplicate_threshold_spin.setToolTip("Duplicate similarity (0.92=strict, 0.85=moderate)")
        threshold_layout.addWidget(self.duplicate_threshold_spin)
        
        threshold_layout.addWidget(QLabel("Overlap:"))
        self.max_coverage_spin = QDoubleSpinBox()
        self.max_coverage_spin.setRange(0.1, 0.9)
        self.max_coverage_spin.setValue(0.5)
        self.max_coverage_spin.setDecimals(1)
        self.max_coverage_spin.setToolTip("Max overlap for coverage mode")
        threshold_layout.addWidget(self.max_coverage_spin)
        
        threshold_layout.addWidget(QLabel("Opt Level:"))
        self.optimization_level_combo = QComboBox()
        self.optimization_level_combo.addItems(["Light", "Balanced", "Aggressive"])
        self.optimization_level_combo.setCurrentIndex(1)  # Balanced default
        self.optimization_level_combo.setToolTip(
            "Alignment optimization level:\n"
            "• Light: Subtle enhancement (fastest)\n"
            "• Balanced: Moderate enhancement (recommended)\n"
            "• Aggressive: Strong enhancement (for difficult images)"
        )
        threshold_layout.addWidget(self.optimization_level_combo)
        filtering_layout.addLayout(threshold_layout)
        
        filtering_group.setLayout(filtering_layout)
        settings_layout.addWidget(filtering_group)
        
        # AutoPano Giga-inspired advanced features
        autopano_group = QGroupBox("Advanced (AutoPano-Style)")
        autopano_layout = QVBoxLayout()
        
        # Row 1: Feature toggles
        autopano_checks = QHBoxLayout()
        
        self.grid_topology_checkbox = QCheckBox("Grid Detect")
        self.grid_topology_checkbox.setChecked(True)
        self.grid_topology_checkbox.setToolTip(
            "Auto-detect grid structure to reduce matching from O(n²) to O(n).\n"
            "Critical for large flat panoramas (microscope, drone, satellite)."
        )
        autopano_checks.addWidget(self.grid_topology_checkbox)
        
        self.bundle_adjust_checkbox = QCheckBox("Bundle Adjust")
        self.bundle_adjust_checkbox.setChecked(False)
        self.bundle_adjust_checkbox.setToolTip(
            "Global optimization of all camera poses.\n"
            "Minimizes reprojection error across all keypoints.\n"
            "Improves alignment accuracy but slower."
        )
        autopano_checks.addWidget(self.bundle_adjust_checkbox)
        
        self.enhanced_detect_checkbox = QCheckBox("Enhanced Detect")
        self.enhanced_detect_checkbox.setChecked(False)
        self.enhanced_detect_checkbox.setToolTip(
            "Enhanced feature detection for low-texture areas.\n"
            "Better for skies, walls, water, repetitive patterns."
        )
        autopano_checks.addWidget(self.enhanced_detect_checkbox)
        
        self.hierarchical_checkbox = QCheckBox("Hierarchical")
        self.hierarchical_checkbox.setChecked(False)
        self.hierarchical_checkbox.setToolTip(
            "Cluster-based stitching for 1000+ images.\n"
            "Stitches clusters independently, then merges."
        )
        autopano_checks.addWidget(self.hierarchical_checkbox)
        autopano_checks.addStretch()
        autopano_layout.addLayout(autopano_checks)
        
        # Row 2: AI post-processing options
        ai_post_layout = QHBoxLayout()
        
        self.ai_post_checkbox = QCheckBox("AI Post-Process")
        self.ai_post_checkbox.setChecked(False)
        self.ai_post_checkbox.setToolTip(
            "Apply AI-powered post-processing:\n"
            "• Color correction (auto white balance)\n"
            "• Denoising\n"
            "• Gap inpainting"
        )
        ai_post_layout.addWidget(self.ai_post_checkbox)
        
        self.ai_denoise_checkbox = QCheckBox("Denoise")
        self.ai_denoise_checkbox.setChecked(True)
        self.ai_denoise_checkbox.setToolTip("Apply AI denoising")
        ai_post_layout.addWidget(self.ai_denoise_checkbox)
        
        self.ai_color_checkbox = QCheckBox("Color Fix")
        self.ai_color_checkbox.setChecked(True)
        self.ai_color_checkbox.setToolTip("Automatic color correction")
        ai_post_layout.addWidget(self.ai_color_checkbox)
        
        self.super_res_checkbox = QCheckBox("2x Super-Res")
        self.super_res_checkbox.setChecked(False)
        self.super_res_checkbox.setToolTip(
            "Apply 2x super-resolution (SLOW).\n"
            "Doubles output resolution."
        )
        ai_post_layout.addWidget(self.super_res_checkbox)
        ai_post_layout.addStretch()
        autopano_layout.addLayout(ai_post_layout)
        
        autopano_group.setLayout(autopano_layout)
        settings_layout.addWidget(autopano_group)
        
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
        self.max_panorama_spin.setRange(0, 500)
        self.max_panorama_spin.setValue(100)
        self.max_panorama_spin.setSuffix("MP")
        self.max_panorama_spin.setSpecialValueText("∞")
        self.max_panorama_spin.setToolTip("Max panorama size (megapixels)")
        limits_layout.addWidget(self.max_panorama_spin)
        
        limits_layout.addWidget(QLabel("Warp:"))
        self.max_warp_spin = QSpinBox()
        self.max_warp_spin.setRange(0, 200)
        self.max_warp_spin.setValue(50)
        self.max_warp_spin.setSuffix("MP")
        self.max_warp_spin.setSpecialValueText("∞")
        self.max_warp_spin.setToolTip("Max warped image size")
        limits_layout.addWidget(self.max_warp_spin)
        output_layout.addLayout(limits_layout)
        
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
        
        # Preview controls - zoom and clear
        preview_controls = QHBoxLayout()
        
        # Zoom controls
        zoom_label = QLabel("Zoom:")
        preview_controls.addWidget(zoom_label)
        
        self.btn_zoom_out = QPushButton("−")
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
                "AutoStitch (Simple & Fast)": "autostitch",
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
            geometric_verify = self.geo_verify_checkbox.isChecked()
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
                select_optimal_coverage=select_optimal_coverage,
                max_coverage_overlap=max_coverage_overlap,
                remove_duplicates=remove_duplicates,
                duplicate_threshold=duplicate_threshold,
                optimize_alignment=self.optimize_alignment_checkbox.isChecked(),
                alignment_optimization_level=self.optimization_level_combo.currentText().lower(),
                # AutoPano Giga-inspired features
                use_grid_topology=self.grid_topology_checkbox.isChecked(),
                use_bundle_adjustment=self.bundle_adjust_checkbox.isChecked(),
                use_hierarchical_stitching=self.hierarchical_checkbox.isChecked(),
                use_enhanced_detection=self.enhanced_detect_checkbox.isChecked()
            )
            logger.info(f"Stitcher initialized (max_panorama={max_panorama_mp}MP, max_warp={max_warp_mp}MP, memory_efficient={memory_efficient}, geo_verify={geometric_verify}, optimal_coverage={select_optimal_coverage}, remove_dups={remove_duplicates}, optimize_align={self.optimize_alignment_checkbox.isChecked()})")
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
            
            # Apply zoom
            new_w = max(1, int(w * self._preview_zoom))
            new_h = max(1, int(h * self._preview_zoom))
            
            # Use appropriate interpolation based on zoom direction
            if self._preview_zoom < 1.0:
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

