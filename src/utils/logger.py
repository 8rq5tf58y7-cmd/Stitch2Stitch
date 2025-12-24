"""Logging utilities"""

import logging
import sys
from pathlib import Path

# Import platform utilities for cross-platform log directory
try:
    from .platform_utils import get_logs_directory
except ImportError:
    # Fallback if platform_utils not available
    def get_logs_directory():
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        return log_dir


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Setup logger with consistent formatting"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler (cross-platform)
        try:
            log_dir = get_logs_directory()
            log_file = log_dir / "stitch2stitch.log"
            file_handler = logging.FileHandler(str(log_file))
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            # If file logging fails, continue without it
            logger.warning(f"Could not set up file logging: {e}")
    
    return logger

