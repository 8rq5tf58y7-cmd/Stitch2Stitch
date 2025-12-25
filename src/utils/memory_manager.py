"""
Memory management utilities for image processing
"""

import psutil
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class MemoryManager:
    """Manage memory usage during image processing"""
    
    def __init__(self, memory_limit_gb: float = 16.0):
        """
        Initialize memory manager
        
        Args:
            memory_limit_gb: Memory limit in GB
        """
        self.memory_limit_gb = memory_limit_gb
        self.memory_limit_bytes = memory_limit_gb * 1024 * 1024 * 1024
        logger.info(f"Memory manager initialized (limit: {memory_limit_gb} GB)")
    
    def check_memory_available(self) -> bool:
        """
        Check if memory is available
        
        Returns:
            True if memory is available, False otherwise
        """
        try:
            available = psutil.virtual_memory().available
            return available > self.memory_limit_bytes * 0.1  # Keep 10% buffer
        except Exception as e:
            logger.warning(f"Could not check memory: {e}")
            return True  # Assume available if check fails
    
    def get_memory_usage(self) -> float:
        """
        Get current memory usage in GB
        
        Returns:
            Memory usage in GB
        """
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            return memory_info.rss / (1024 * 1024 * 1024)  # Convert to GB
        except Exception as e:
            logger.warning(f"Could not get memory usage: {e}")
            return 0.0
    
    def get_available_memory(self) -> float:
        """
        Get available system memory in GB
        
        Returns:
            Available memory in GB
        """
        try:
            available = psutil.virtual_memory().available
            return available / (1024 * 1024 * 1024)  # Convert to GB
        except Exception as e:
            logger.warning(f"Could not get available memory: {e}")
            return 16.0  # Default assumption

