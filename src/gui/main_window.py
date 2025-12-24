"""
Main GUI window for Stitch2Stitch
Modern, intuitive interface for panoramic image stitching
"""

import sys
from pathlib import Path
from typing import List, Optional
import logging

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QFileDialog, QLabel, QProgressBar, QTextEdit, QTabWidget,
    QGroupBox, QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox,
    QSlider, QMessageBox, QSplitter, QListWidget, QListWidgetItem
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QPixmap, QImage, QFont

from ..core.stitcher import ImageStitcher
from ..utils.logger import setup_logger

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


class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.image_paths: List[Path] = []
        self.stitcher: Optional[ImageStitcher] = None
        self.thread: Optional[StitchingThread] = None
        self.init_ui()
        self.init_stitcher()
    
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
        """Create control panel"""
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
        self.btn_add_images = QPushButton("Add Images...")
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
        self.quality_spin.setValue(0.7)
        self.quality_spin.setDecimals(2)
        quality_layout.addWidget(self.quality_spin)
        settings_layout.addLayout(quality_layout)
        
        # GPU acceleration
        self.gpu_checkbox = QCheckBox("Enable GPU Acceleration")
        self.gpu_checkbox.setChecked(False)
        settings_layout.addWidget(self.gpu_checkbox)
        
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
            "PixelStitch (Structure-Preserving)"
        ])
        blend_layout.addWidget(self.blend_combo)
        settings_layout.addLayout(blend_layout)
        
        # Max images
        max_img_layout = QHBoxLayout()
        max_img_layout.addWidget(QLabel("Max Images:"))
        self.max_images_spin = QSpinBox()
        self.max_images_spin.setRange(0, 10000)
        self.max_images_spin.setValue(0)
        self.max_images_spin.setSpecialValueText("Unlimited")
        max_img_layout.addWidget(self.max_images_spin)
        settings_layout.addLayout(max_img_layout)
        
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
        
        return panel
    
    def create_preview_panel(self) -> QWidget:
        """Create preview panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Tabs
        tabs = QTabWidget()
        
        # Preview tab
        preview_tab = QWidget()
        preview_layout = QVBoxLayout(preview_tab)
        
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
            "PixelStitch (Structure-Preserving)": "pixelstitch"
        }
        
        feature_detector = detector_map.get(self.detector_combo.currentText(), "lp_sift")
        feature_matcher = matcher_map.get(self.matcher_combo.currentText(), "flann")
        blending_method = blender_map.get(self.blend_combo.currentText(), "multiband")
        
        self.stitcher = ImageStitcher(
            use_gpu=use_gpu,
            quality_threshold=quality_threshold,
            max_images=max_images,
            feature_detector=feature_detector,
            feature_matcher=feature_matcher,
            blending_method=blending_method
        )
    
    def add_images(self):
        """Add images to the list"""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Images",
            "",
            "Image Files (*.jpg *.jpeg *.png *.tif *.tiff *.bmp);;All Files (*)"
        )
        
        for file in files:
            path = Path(file)
            if path not in self.image_paths:
                self.image_paths.append(path)
                self.image_list.addItem(path.name)
        
        self.log(f"Added {len(files)} image(s). Total: {len(self.image_paths)}")
    
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
        self.log("Starting grid alignment creation...")
        
        self.btn_grid.setEnabled(False)
        self.btn_stitch.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_label.setVisible(True)
        self.progress_label.setText("Initializing...")
        
        self.thread = StitchingThread(self.stitcher, self.image_paths, grid_only=True)
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
        if self.thread and self.thread.isRunning():
            reply = QMessageBox.question(
                self,
                "Confirm Stop",
                "Are you sure you want to stop the current operation?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.thread.cancel()
                self.log("Stopping operation...")
                self.btn_stop.setEnabled(False)
    
    def update_progress(self, percentage: int):
        """Update progress bar"""
        self.progress_bar.setValue(percentage)
    
    def update_status(self, message: str):
        """Update status message"""
        self.progress_label.setText(message)
        self.log(message)
    
    def on_grid_finished(self, grid_layout):
        """Handle grid creation completion"""
        self.log("Grid alignment created successfully!")
        self.stitcher.save_grid(grid_layout, str(self.output_path))
        self.current_result = grid_layout
        self.update_preview_from_grid(grid_layout)
        
        self.reset_ui_after_processing()
        self.btn_save.setEnabled(True)
        
        QMessageBox.information(self, "Success", "Grid alignment created successfully!")
    
    def on_stitching_finished(self, panorama):
        """Handle stitching completion"""
        self.log("Stitching completed successfully!")
        self.current_result = panorama
        self.update_preview(panorama)
        
        self.reset_ui_after_processing()
        self.btn_save.setEnabled(True)
        
        QMessageBox.information(self, "Success", "Stitching completed successfully!")
    
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
        import numpy as np
        
        # Resize for preview
        h, w = image.shape[:2]
        max_size = 800
        scale = min(max_size / w, max_size / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        if len(image.shape) == 3:
            resized = cv2.resize(image, (new_w, new_h))
            qimage = QImage(resized.data, new_w, new_h, QImage.Format.Format_BGR888)
        else:
            resized = cv2.resize(image, (new_w, new_h))
            qimage = QImage(resized.data, new_w, new_h, QImage.Format.Format_Grayscale8)
        
        pixmap = QPixmap.fromImage(qimage)
        self.preview_label.setPixmap(pixmap)
    
    def update_preview_from_grid(self, grid_layout):
        """Update preview with grid layout"""
        # Create a simple preview of the grid
        # In a full implementation, this would render the grid
        self.preview_label.setText("Grid alignment created.\nUse 'Save Result' to export.")
    
    def save_result(self):
        """Save current result"""
        if not self.current_result:
            QMessageBox.warning(self, "No Result", "No result to save.")
            return
        
        if not self.output_path:
            self.browse_output()
            if not self.output_path:
                return
        
        try:
            if isinstance(self.current_result, dict):  # Grid layout
                self.stitcher.save_grid(self.current_result, str(self.output_path))
            else:  # Panorama
                self.stitcher.save_panorama(self.current_result, str(self.output_path))
            
            self.log(f"Result saved to {self.output_path}")
            QMessageBox.information(self, "Saved", f"Result saved to {self.output_path}")
        except Exception as e:
            logger.error(f"Save error: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to save:\n{e}")
    
    def log(self, message: str):
        """Add message to log"""
        self.log_text.append(message)
        logger.info(message)
        self.statusBar().showMessage(message)

