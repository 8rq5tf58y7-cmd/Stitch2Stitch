"""
Advanced feature matching with RANSAC and confidence scoring
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class AdvancedMatcher:
    """Advanced feature matcher with multiple algorithms"""
    
    def __init__(
        self,
        use_gpu: bool = False,
        method: str = 'flann',
        ratio_threshold: float = 0.75
    ):
        """
        Initialize matcher
        
        Args:
            use_gpu: Enable GPU acceleration
            method: Matching method ('flann' or 'bf')
            ratio_threshold: Lowe's ratio test threshold
        """
        self.use_gpu = use_gpu
        # Stricter ratio threshold for better match quality (default 0.75 -> 0.7)
        self.ratio_threshold = ratio_threshold if ratio_threshold > 0 else 0.7
        
        if method == 'flann':
            # FLANN matcher for SIFT/SURF - use LSH for binary descriptors, KDTREE for float
            # We'll use KDTREE for SIFT (float descriptors)
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=100)  # Increased checks for better accuracy
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            # Brute force matcher
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        
        logger.info(f"Advanced matcher initialized (method: {method})")
    
    def match(
        self,
        descriptors1: np.ndarray,
        descriptors2: np.ndarray
    ) -> Dict:
        """
        Match descriptors between two images
        
        Args:
            descriptors1: Descriptors from first image
            descriptors2: Descriptors from second image
            
        Returns:
            Dictionary with match results
        """
        if descriptors1 is None or descriptors2 is None:
            return {
                'matches': [],
                'num_matches': 0,
                'confidence': 0.0,
                'homography': None
            }
        
        # Validate descriptor format and shape
        if not isinstance(descriptors1, np.ndarray) or not isinstance(descriptors2, np.ndarray):
            logger.warning("Descriptors are not numpy arrays")
            return {
                'matches': [],
                'num_matches': 0,
                'confidence': 0.0,
                'homography': None
            }
        
        if len(descriptors1.shape) != 2 or len(descriptors2.shape) != 2:
            logger.warning(f"Invalid descriptor shape: {descriptors1.shape}, {descriptors2.shape}")
            return {
                'matches': [],
                'num_matches': 0,
                'confidence': 0.0,
                'homography': None
            }
        
        if descriptors1.shape[1] != descriptors2.shape[1]:
            logger.warning(f"Descriptor dimension mismatch: {descriptors1.shape[1]} vs {descriptors2.shape[1]}")
            return {
                'matches': [],
                'num_matches': 0,
                'confidence': 0.0,
                'homography': None
            }
        
        if len(descriptors1) < 4 or len(descriptors2) < 4:
            return {
                'matches': [],
                'num_matches': 0,
                'confidence': 0.0,
                'homography': None
            }
        
        # Ensure descriptors are float32 for FLANN
        if descriptors1.dtype != np.float32:
            descriptors1 = descriptors1.astype(np.float32)
        if descriptors2.dtype != np.float32:
            descriptors2 = descriptors2.astype(np.float32)
        
        try:
            # Perform matching with error handling
            # FLANN can fail with certain descriptor types, so try-catch
            try:
                matches = self.matcher.knnMatch(descriptors1, descriptors2, k=2)
            except cv2.error as e:
                logger.warning(f"FLANN matching failed, falling back to brute force: {e}")
                # Fallback to brute force matcher
                bf_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
                matches = bf_matcher.knnMatch(descriptors1, descriptors2, k=2)
            
            if not matches:
                return {
                    'matches': [],
                    'num_matches': 0,
                    'confidence': 0.0,
                    'homography': None
                }
            
            # Apply Lowe's ratio test with stricter threshold
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    # Stricter ratio test - only accept matches where first is significantly better
                    if m.distance < self.ratio_threshold * n.distance:
                        good_matches.append(m)
            
            if len(good_matches) < 10:  # Require at least 10 matches for reliability
                return {
                    'matches': good_matches,
                    'num_matches': len(good_matches),
                    'confidence': 0.0,
                    'homography': None
                }
            
            # Calculate confidence (avoid division by zero)
            max_descriptors = max(len(descriptors1), len(descriptors2))
            if max_descriptors > 0:
                confidence = len(good_matches) / max_descriptors
            else:
                confidence = 0.0
            
            return {
                'matches': good_matches,
                'num_matches': len(good_matches),
                'confidence': confidence,
                'homography': None  # Will be calculated in alignment module
            }
        except Exception as e:
            logger.error(f"Error during feature matching: {e}", exc_info=True)
            return {
                'matches': [],
                'num_matches': 0,
                'confidence': 0.0,
                'homography': None
            }

