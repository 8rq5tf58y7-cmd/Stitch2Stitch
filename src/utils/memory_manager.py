"""
Memory management utilities for image processing
"""

import psutil
import logging
import gc
from typing import Optional, Dict
from contextlib import contextmanager

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
        self._peak_usage_gb = 0.0
        self._checkpoints: Dict[str, float] = {}
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
            usage_gb = memory_info.rss / (1024 * 1024 * 1024)
            
            # Track peak usage
            if usage_gb > self._peak_usage_gb:
                self._peak_usage_gb = usage_gb
            
            return usage_gb
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
    
    def get_total_memory(self) -> float:
        """
        Get total system memory in GB
        
        Returns:
            Total memory in GB
        """
        try:
            total = psutil.virtual_memory().total
            return total / (1024 * 1024 * 1024)
        except Exception as e:
            logger.warning(f"Could not get total memory: {e}")
            return 16.0
    
    def get_peak_usage(self) -> float:
        """
        Get peak memory usage in GB
        
        Returns:
            Peak memory usage in GB
        """
        return self._peak_usage_gb
    
    def reset_peak_usage(self):
        """Reset peak usage tracker"""
        self._peak_usage_gb = self.get_memory_usage()
    
    def checkpoint(self, name: str):
        """
        Create a memory checkpoint for tracking
        
        Args:
            name: Checkpoint name
        """
        usage = self.get_memory_usage()
        self._checkpoints[name] = usage
        logger.debug(f"Memory checkpoint '{name}': {usage:.2f} GB")
    
    def get_checkpoint_diff(self, name: str) -> Optional[float]:
        """
        Get memory difference since checkpoint
        
        Args:
            name: Checkpoint name
            
        Returns:
            Memory difference in GB, or None if checkpoint doesn't exist
        """
        if name not in self._checkpoints:
            return None
        return self.get_memory_usage() - self._checkpoints[name]
    
    def log_memory_status(self, context: str = ""):
        """
        Log current memory status
        
        Args:
            context: Optional context string for the log
        """
        usage = self.get_memory_usage()
        available = self.get_available_memory()
        total = self.get_total_memory()
        peak = self.get_peak_usage()
        
        context_str = f" ({context})" if context else ""
        logger.info(
            f"Memory status{context_str}: "
            f"used={usage:.2f}GB, available={available:.2f}GB, "
            f"total={total:.2f}GB, peak={peak:.2f}GB"
        )
        
        # Warn if memory is getting low
        if available < total * 0.15:
            logger.warning(f"Low memory warning: only {available:.2f}GB available!")
    
    def force_gc(self):
        """Force garbage collection and log results"""
        before = self.get_memory_usage()
        gc.collect()
        after = self.get_memory_usage()
        freed = before - after
        if freed > 0.01:  # Only log if significant
            logger.debug(f"GC freed {freed*1024:.1f} MB")
    
    @contextmanager
    def track_operation(self, name: str):
        """
        Context manager to track memory usage of an operation
        
        Args:
            name: Operation name
            
        Example:
            with memory_manager.track_operation("feature_detection"):
                detect_features(images)
        """
        self.checkpoint(f"{name}_start")
        start_usage = self.get_memory_usage()
        
        try:
            yield
        finally:
            end_usage = self.get_memory_usage()
            diff = end_usage - start_usage
            logger.info(f"Operation '{name}': memory change {diff:+.2f} GB (now {end_usage:.2f} GB)")
    
    def estimate_can_process(self, estimated_mb: float) -> bool:
        """
        Check if there's enough memory to process estimated workload
        
        Args:
            estimated_mb: Estimated memory requirement in MB
            
        Returns:
            True if processing is likely safe, False otherwise
        """
        available_mb = self.get_available_memory() * 1024
        # Keep 20% buffer for safety
        safe_available = available_mb * 0.8
        
        can_process = estimated_mb < safe_available
        
        if not can_process:
            logger.warning(
                f"Estimated memory requirement ({estimated_mb:.0f} MB) "
                f"may exceed safe available memory ({safe_available:.0f} MB)"
            )
        
        return can_process

