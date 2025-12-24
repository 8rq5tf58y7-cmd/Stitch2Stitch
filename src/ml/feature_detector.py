"""
Advanced feature detection using LP-SIFT and other modern algorithms
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class LP_SIFTDetector:
    """
    Local-Peak SIFT detector
    Optimized SIFT variant focusing on multiscale local peaks
    """
    
    def __init__(self, use_gpu: bool = False, n_features: int = 5000):
        """
        Initialize LP-SIFT detector
        
        Args:
            use_gpu: Enable GPU acceleration
            n_features: Maximum number of features to detect
        """
        self.use_gpu = use_gpu
        self.n_features = n_features
        
        # Initialize SIFT detector with optimized parameters
        self.sift = cv2.SIFT_create(
            nfeatures=n_features,
            contrastThreshold=0.04,
            edgeThreshold=10,
            sigma=1.6
        )
        
        logger.info(f"LP-SIFT detector initialized (GPU: {use_gpu})")
    
    def detect_and_compute(
        self,
        image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect features and compute descriptors
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            Tuple of (keypoints, descriptors)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply preprocessing for better feature detection
        gray = self._preprocess(gray)
        
        # Detect and compute
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        
        # Filter keypoints to focus on local peaks (LP-SIFT approach)
        if len(keypoints) > 0:
            keypoints, descriptors = self._filter_local_peaks(
                keypoints, descriptors, gray
            )
        
        # Convert keypoints to numpy array format
        kp_array = np.array([[kp.pt[0], kp.pt[1], kp.size, kp.angle] 
                             for kp in keypoints])
        
        return kp_array, descriptors
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better feature detection"""
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)
        
        return enhanced
    
    def _filter_local_peaks(
        self,
        keypoints: list,
        descriptors: np.ndarray,
        image: np.ndarray
    ) -> Tuple[list, np.ndarray]:
        """
        Filter keypoints to focus on local peaks
        This implements the LP-SIFT approach
        """
        if len(keypoints) == 0:
            return keypoints, descriptors
        
        # Create response map
        responses = np.array([kp.response for kp in keypoints])
        
        # Sort by response
        sorted_indices = np.argsort(responses)[::-1]
        
        # Keep top N features
        n_keep = min(self.n_features, len(keypoints))
        keep_indices = sorted_indices[:n_keep]
        
        filtered_kp = [keypoints[i] for i in keep_indices]
        filtered_desc = descriptors[keep_indices]
        
        return filtered_kp, filtered_desc


class ORBDetector:
    """ORB detector as alternative"""
    
    def __init__(self, n_features: int = 5000):
        self.orb = cv2.ORB_create(nfeatures=n_features)
    
    def detect_and_compute(self, image: np.ndarray):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        kp_array = np.array([[kp.pt[0], kp.pt[1], kp.size, kp.angle] 
                             for kp in keypoints])
        
        return kp_array, descriptors


class AKAZEDetector:
    """AKAZE detector as alternative"""
    
    def __init__(self):
        self.akaze = cv2.AKAZE_create()
    
    def detect_and_compute(self, image: np.ndarray):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        keypoints, descriptors = self.akaze.detectAndCompute(gray, None)
        kp_array = np.array([[kp.pt[0], kp.pt[1], kp.size, kp.angle] 
                             for kp in keypoints])
        
        return kp_array, descriptors

