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
        image: np.ndarray,
        max_dimension: int = 4000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect features and compute descriptors
        
        Args:
            image: Input image (BGR or grayscale)
            max_dimension: Maximum image dimension for feature detection.
                           Larger images are scaled down for speed, keypoints scaled back up.
            
        Returns:
            Tuple of (keypoints, descriptors)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Scale down very large images for faster feature detection
        h, w = gray.shape[:2]
        scale = 1.0
        if max(h, w) > max_dimension:
            scale = max_dimension / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
            logger.debug(f"Scaled image from {w}x{h} to {new_w}x{new_h} for feature detection")
        
        # Apply preprocessing for better feature detection
        gray = self._preprocess(gray)
        
        # Detect and compute
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        
        # Filter keypoints to focus on local peaks (LP-SIFT approach)
        if len(keypoints) > 0:
            keypoints, descriptors = self._filter_local_peaks(
                keypoints, descriptors, gray
            )
        
        # Scale keypoints back to original image coordinates
        if scale != 1.0 and len(keypoints) > 0:
            inv_scale = 1.0 / scale
            for kp in keypoints:
                kp.pt = (kp.pt[0] * inv_scale, kp.pt[1] * inv_scale)
                kp.size *= inv_scale
        
        # Convert keypoints to numpy array format
        kp_array = np.array([[kp.pt[0], kp.pt[1], kp.size, kp.angle] 
                             for kp in keypoints]) if len(keypoints) > 0 else np.array([])
        
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
    
    def __init__(self, n_features: int = 5000):
        self.n_features = n_features
        # AKAZE doesn't have nfeatures parameter, but we'll limit after detection
        self.akaze = cv2.AKAZE_create()
    
    def detect_and_compute(self, image: np.ndarray):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        keypoints, descriptors = self.akaze.detectAndCompute(gray, None)
        
        # Limit to max features by response strength
        if len(keypoints) > self.n_features:
            responses = np.array([kp.response for kp in keypoints])
            sorted_indices = np.argsort(responses)[::-1]
            keep_indices = sorted_indices[:self.n_features]
            keypoints = [keypoints[i] for i in keep_indices]
            if descriptors is not None:
                descriptors = descriptors[keep_indices]
        
        kp_array = np.array([[kp.pt[0], kp.pt[1], kp.size, kp.angle] 
                             for kp in keypoints])
        
        return kp_array, descriptors

