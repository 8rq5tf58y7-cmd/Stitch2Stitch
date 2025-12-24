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

from gui.main_window import MainWindow
from core.stitcher import ImageStitcher
from utils.logger import setup_logger

logger = setup_logger(__name__)


def main():
    """Main entry point"""
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
        from PyQt6.QtWidgets import QApplication
        
        app = QApplication(sys.argv)
        app.setApplicationName("Stitch2Stitch")
        app.setOrganizationName("Stitch2Stitch")
        
        window = MainWindow()
        window.show()
        
        return app.exec()


if __name__ == "__main__":
    sys.exit(main())

