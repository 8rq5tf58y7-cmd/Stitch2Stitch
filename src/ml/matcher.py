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
        self.ratio_threshold = ratio_threshold
        
        if method == 'flann':
            # FLANN matcher for SIFT/SURF
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
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
        
        if len(descriptors1) < 4 or len(descriptors2) < 4:
            return {
                'matches': [],
                'num_matches': 0,
                'confidence': 0.0,
                'homography': None
            }
        
        # Perform matching
        matches = self.matcher.knnMatch(descriptors1, descriptors2, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.ratio_threshold * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < 4:
            return {
                'matches': good_matches,
                'num_matches': len(good_matches),
                'confidence': 0.0,
                'homography': None
            }
        
        # Calculate confidence
        confidence = len(good_matches) / max(len(descriptors1), len(descriptors2))
        
        return {
            'matches': good_matches,
            'num_matches': len(good_matches),
            'confidence': confidence,
            'homography': None  # Will be calculated in alignment module
        }

