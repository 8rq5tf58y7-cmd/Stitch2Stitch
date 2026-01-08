#!/usr/bin/env python3
"""
Stitch2Stitch - Advanced Panoramic Image Stitching Application
Main entry point for the application
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Check and install dependencies before importing heavy modules
def _ensure_dependencies():
    """Quick dependency check at startup."""
    try:
        from utils.dependency_installer import ensure_dependencies
        if not ensure_dependencies(silent=True):
            print("Installing missing dependencies...", file=sys.stderr)
            ensure_dependencies(silent=False)
    except ImportError:
        pass  # dependency_installer itself might need deps

_ensure_dependencies()

from gui.main_window import MainWindow
from core.stitcher import ImageStitcher
from utils.logger import setup_logger, get_log_file_path

logger = setup_logger(__name__)

# Print log file location at startup
log_path = get_log_file_path()
if log_path:
    print(f"Log file location: {log_path.absolute()}", file=sys.stderr)


def main():
    """Main entry point"""
    logger.info("=" * 60)
    logger.info("Stitch2Stitch starting...")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {Path.cwd()}")
    log_path = get_log_file_path()
    if log_path:
        logger.info(f"Log file: {log_path.absolute()}")
    logger.info("=" * 60)
    
    parser = argparse.ArgumentParser(
        description="Stitch2Stitch - Advanced Panoramic Image Stitching"
    )
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Run in command-line mode instead of GUI"
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Input directory containing images"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path for stitched panorama"
    )
    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=0.7,
        help="Quality threshold for image selection (0.0-1.0)"
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Enable GPU acceleration"
    )
    parser.add_argument(
        "--grid-only",
        action="store_true",
        help="Only create grid alignment, don't stitch"
    )
    parser.add_argument(
        "--detector",
        type=str,
        default="lp_sift",
        choices=["lp_sift", "sift", "orb", "akaze"],
        help="Feature detector algorithm (default: lp_sift)"
    )
    parser.add_argument(
        "--matcher",
        type=str,
        default="flann",
        choices=["flann", "loftr", "superglue", "disk"],
        help="Feature matcher algorithm (default: flann)"
    )
    parser.add_argument(
        "--blender",
        type=str,
        default="multiband",
        choices=["multiband", "feather", "linear", "semantic", "pixelstitch"],
        help="Blending algorithm (default: multiband)"
    )
    
    args = parser.parse_args()
    
    if args.cli:
        # Command-line mode
        if not args.input or not args.output:
            parser.error("--input and --output are required in CLI mode")
        
        logger.info("Starting CLI stitching process...")
        stitcher = ImageStitcher(
            use_gpu=args.gpu,
            quality_threshold=args.quality_threshold,
            feature_detector=args.detector,
            feature_matcher=args.matcher,
            blending_method=args.blender
        )
        
        input_path = Path(args.input)
        if not input_path.exists():
            logger.error(f"Input path does not exist: {input_path}")
            return 1
        
        images = list(input_path.glob("*.jpg")) + list(input_path.glob("*.png")) + \
                 list(input_path.glob("*.tif")) + list(input_path.glob("*.tiff"))
        
        if not images:
            logger.error(f"No images found in {input_path}")
            return 1
        
        logger.info(f"Found {len(images)} images")
        
        if args.grid_only:
            logger.info("Creating grid alignment...")
            grid = stitcher.create_grid_alignment(images)
            stitcher.save_grid(grid, args.output)
        else:
            logger.info("Starting stitching process...")
            panorama = stitcher.stitch(images)
            stitcher.save_panorama(panorama, args.output)
        
        logger.info("Process completed successfully!")
        return 0
    else:
        # GUI mode
        from PyQt6.QtWidgets import QApplication, QMessageBox
        
        try:
            app = QApplication(sys.argv)
            app.setApplicationName("Stitch2Stitch")
            app.setOrganizationName("Stitch2Stitch")
            
            try:
                logger.info("Creating main window...")
                window = MainWindow()
                logger.info("Main window created successfully")
                window.show()
                window.raise_()
                window.activateWindow()
                logger.info("Main window displayed")
                
                logger.info("Starting application event loop...")
                return app.exec()
            except Exception as e:
                logger.error(f"Failed to create window: {e}", exc_info=True)
                # Try to show error in a message box if QApplication is available
                try:
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Icon.Critical)
                    msg.setWindowTitle("Error")
                    msg.setText("Failed to start application")
                    log_path = get_log_file_path()
                    if log_path:
                        error_details = f"{str(e)}\n\nCheck the log file for details:\n{log_path.absolute()}"
                    else:
                        error_details = f"{str(e)}\n\nCheck the console for details."
                    msg.setDetailedText(error_details)
                    msg.exec()
                except:
                    pass
                print(f"ERROR: Failed to create window: {e}", file=sys.stderr)
                if log_path:
                    print(f"Check log file: {log_path.absolute()}", file=sys.stderr)
                import traceback
                traceback.print_exc()
                return 1
        except Exception as e:
            logger.error(f"Failed to start GUI: {e}", exc_info=True)
            print(f"ERROR: Failed to start GUI: {e}")
            import traceback
            traceback.print_exc()
            return 1


if __name__ == "__main__":
    sys.exit(main())

