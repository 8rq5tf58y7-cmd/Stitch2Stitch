"""Logging utilities"""

import logging
import sys
from pathlib import Path

# Global variable to store log file path
_log_file_path = None


def get_logs_directory() -> Path:
    """Get logs directory with fallbacks"""
    # Try platform-specific location first
    try:
        from utils.platform_utils import get_logs_directory as _get_logs_dir
        log_dir = _get_logs_dir()
        # Verify we can write to it
        test_file = log_dir / ".test_write"
        try:
            test_file.write_text("test")
            test_file.unlink()
            return log_dir
        except (PermissionError, OSError):
            # Can't write to platform directory, fall back to local
            pass
    except (ImportError, Exception):
        pass
    
    # Fallback 1: Try local logs directory
    try:
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        test_file = log_dir / ".test_write"
        test_file.write_text("test")
        test_file.unlink()
        return log_dir
    except (PermissionError, OSError):
        pass
    
    # Fallback 2: Use current directory
    return Path(".")


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Setup logger with consistent formatting"""
    global _log_file_path
    
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
            
            # Try to create/open the log file
            file_handler = logging.FileHandler(str(log_file), mode='a', encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
            _log_file_path = log_file
            # Log the log file location (but only once to avoid recursion)
            if name == "__main__" or "main" in name.lower():
                print(f"Log file: {log_file.absolute()}", file=sys.stderr)
                logger.info(f"Logging to: {log_file.absolute()}")
        except Exception as e:
            # If file logging fails, continue without it
            error_msg = f"Could not set up file logging: {e}"
            print(error_msg, file=sys.stderr)
            # Use warning level to avoid recursion issues
            try:
                logger.warning(error_msg)
            except:
                pass
    
    return logger


def get_log_file_path() -> Path:
    """Get the path to the log file"""
    return _log_file_path

