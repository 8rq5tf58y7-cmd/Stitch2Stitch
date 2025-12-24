"""
Image quality assessment for blur detection and sharpness scoring
"""

import cv2
import numpy as np
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class ImageQualityAssessor:
    """Assess image quality using multiple metrics"""
    
    def __init__(self, use_gpu: bool = False):
        """
        Initialize quality assessor
        
        Args:
            use_gpu: Enable GPU acceleration
        """
        self.use_gpu = use_gpu
        logger.info("Image quality assessor initialized")
    
    def assess(self, image: np.ndarray) -> float:
        """
        Assess image quality and return score (0.0-1.0)
        
        Args:
            image: Input image (BGR)
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        if image is None or image.size == 0:
            return 0.0
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Calculate multiple quality metrics
        laplacian_var = self._laplacian_variance(gray)
        gradient_magnitude = self._gradient_magnitude(gray)
        contrast_score = self._contrast_score(gray)
        noise_score = self._noise_score(gray)
        
        # Normalize and combine scores
        scores = {
            'sharpness': self._normalize(laplacian_var, 0, 1000),
            'gradient': self._normalize(gradient_magnitude, 0, 100),
            'contrast': contrast_score,
            'noise': 1.0 - self._normalize(noise_score, 0, 50)  # Lower noise is better
        }
        
        # Weighted combination
        quality = (
            scores['sharpness'] * 0.4 +
            scores['gradient'] * 0.3 +
            scores['contrast'] * 0.2 +
            scores['noise'] * 0.1
        )
        
        return np.clip(quality, 0.0, 1.0)
    
    def _laplacian_variance(self, image: np.ndarray) -> float:
        """
        Calculate Laplacian variance (blur detection)
        Higher values indicate sharper images
        """
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        return laplacian.var()
    
    def _gradient_magnitude(self, image: np.ndarray) -> float:
        """Calculate mean gradient magnitude"""
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        return np.mean(magnitude)
    
    def _contrast_score(self, image: np.ndarray) -> float:
        """Calculate contrast score"""
        # Use standard deviation as contrast measure
        std = np.std(image)
        return self._normalize(std, 0, 100)
    
    def _noise_score(self, image: np.ndarray) -> float:
        """Estimate noise level"""
        # Use high-frequency content as noise indicator
        kernel = np.array([[-1, -1, -1],
                          [-1,  8, -1],
                          [-1, -1, -1]])
        filtered = cv2.filter2D(image.astype(np.float32), -1, kernel)
        return np.std(filtered)
    
    def _normalize(self, value: float, min_val: float, max_val: float) -> float:
        """Normalize value to 0-1 range"""
        if max_val == min_val:
            return 0.0
        normalized = (value - min_val) / (max_val - min_val)
        return np.clip(normalized, 0.0, 1.0)
    
    def is_blurry(self, image: np.ndarray, threshold: float = 0.5) -> bool:
        """
        Determine if image is blurry
        
        Args:
            image: Input image
            threshold: Quality threshold below which image is considered blurry
            
        Returns:
            True if image is blurry
        """
        quality = self.assess(image)
        return quality < threshold

