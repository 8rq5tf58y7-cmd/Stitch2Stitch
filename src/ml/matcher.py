"""
Advanced feature matching with RANSAC, confidence scoring, and color verification
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


def compute_color_similarity(desc1: np.ndarray, desc2: np.ndarray, color_dim: int = 48) -> float:
    """
    Compute color similarity between two descriptors that have color info appended.
    
    Args:
        desc1: First descriptor (SIFT + color)
        desc2: Second descriptor (SIFT + color)
        color_dim: Number of color descriptor dimensions (default 48 = 16 bins * 3 channels)
        
    Returns:
        Color similarity score (0-1, higher = more similar)
    """
    if len(desc1) <= color_dim or len(desc2) <= color_dim:
        return 1.0  # No color info, assume similar
    
    # Extract color portion (last color_dim elements)
    color1 = desc1[-color_dim:]
    color2 = desc2[-color_dim:]
    
    # Normalize
    norm1 = np.linalg.norm(color1)
    norm2 = np.linalg.norm(color2)
    
    if norm1 < 1e-6 or norm2 < 1e-6:
        return 1.0  # Empty color descriptors
    
    # Cosine similarity
    similarity = np.dot(color1, color2) / (norm1 * norm2)
    return max(0.0, similarity)


class AdvancedMatcher:
    """Advanced feature matcher with multiple algorithms and color verification"""
    
    def __init__(
        self,
        use_gpu: bool = False,
        method: str = 'flann',
        ratio_threshold: float = 0.75,
        color_threshold: float = 0.3,
        use_color_verification: bool = True
    ):
        """
        Initialize matcher
        
        Args:
            use_gpu: Enable GPU acceleration
            method: Matching method ('flann' or 'bf')
            ratio_threshold: Lowe's ratio test threshold
            color_threshold: Minimum color similarity for matches (0-1)
            use_color_verification: Whether to verify matches using color similarity
        """
        self.use_gpu = use_gpu
        self.method = method
        # Stricter ratio threshold for better match quality (default 0.75 -> 0.7)
        self.ratio_threshold = ratio_threshold if ratio_threshold > 0 else 0.7
        self.color_threshold = color_threshold
        self.use_color_verification = use_color_verification
        
        # Will create matcher on first use (to handle different descriptor dimensions)
        self.matcher = None
        self._last_dim = None
        
        logger.info(f"Advanced matcher initialized (method: {method}, color_verification: {use_color_verification})")
    
    def _create_matcher(self, descriptor_dim: int):
        """Create matcher appropriate for descriptor dimension"""
        if self.method == 'flann':
            # FLANN matcher for SIFT/SURF - use KDTREE for float descriptors
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=100)
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            # Brute force matcher
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        self._last_dim = descriptor_dim
    
    def match(
        self,
        descriptors1: np.ndarray,
        descriptors2: np.ndarray
    ) -> Dict:
        """
        Match descriptors between two images with optional color verification
        
        Args:
            descriptors1: Descriptors from first image (may include color info)
            descriptors2: Descriptors from second image (may include color info)
            
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
            # #region agent log
            try:
                import json
                with open(r'c:\Users\ryanf\OneDrive - University of Maryland\Desktop\Stitch2Stitch\Stitch2Stitch-1\.cursor\debug.log', 'a') as f:
                    f.write(json.dumps({"hypothesisId":"H2","location":"matcher.py:match","message":"Too few descriptors","data":{"len1":len(descriptors1),"len2":len(descriptors2)},"timestamp":__import__('time').time()}) + '\n')
            except: pass
            # #endregion
            return {
                'matches': [],
                'num_matches': 0,
                'confidence': 0.0,
                'homography': None
            }
        
        # #region agent log
        try:
            import json
            with open(r'c:\Users\ryanf\OneDrive - University of Maryland\Desktop\Stitch2Stitch\Stitch2Stitch-1\.cursor\debug.log', 'a') as f:
                f.write(json.dumps({"hypothesisId":"H2","location":"matcher.py:match","message":"Starting match","data":{"desc1_shape":list(descriptors1.shape),"desc2_shape":list(descriptors2.shape),"dtype":str(descriptors1.dtype)},"timestamp":__import__('time').time()}) + '\n')
        except: pass
        # #endregion
        
        # Check if descriptors include color info (SIFT=128, SIFT+color=176)
        has_color = descriptors1.shape[1] > 128
        color_dim = 48 if has_color else 0
        
        # Ensure descriptors are float32 for FLANN
        if descriptors1.dtype != np.float32:
            descriptors1 = descriptors1.astype(np.float32)
        if descriptors2.dtype != np.float32:
            descriptors2 = descriptors2.astype(np.float32)
        
        # Create or recreate matcher if descriptor dimension changed
        if self.matcher is None or self._last_dim != descriptors1.shape[1]:
            self._create_matcher(descriptors1.shape[1])
        
        try:
            # Perform matching with error handling
            try:
                matches = self.matcher.knnMatch(descriptors1, descriptors2, k=2)
            except cv2.error as e:
                logger.warning(f"FLANN matching failed, falling back to brute force: {e}")
                bf_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
                matches = bf_matcher.knnMatch(descriptors1, descriptors2, k=2)
            
            if not matches:
                return {
                    'matches': [],
                    'num_matches': 0,
                    'confidence': 0.0,
                    'homography': None
                }
            
            # Apply Lowe's ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < self.ratio_threshold * n.distance:
                        good_matches.append(m)
            
            # #region agent log
            try:
                import json
                with open(r'c:\Users\ryanf\OneDrive - University of Maryland\Desktop\Stitch2Stitch\Stitch2Stitch-1\.cursor\debug.log', 'a') as f:
                    f.write(json.dumps({"hypothesisId":"H2","location":"matcher.py:match","message":"After ratio test","data":{"raw_matches":len(matches),"good_matches":len(good_matches),"ratio_threshold":self.ratio_threshold},"timestamp":__import__('time').time()}) + '\n')
            except: pass
            # #endregion
            
            # Apply color verification if enabled and color info available
            if self.use_color_verification and has_color and len(good_matches) > 0:
                color_verified_matches = []
                rejected_by_color = 0
                
                for m in good_matches:
                    color_sim = compute_color_similarity(
                        descriptors1[m.queryIdx],
                        descriptors2[m.trainIdx],
                        color_dim
                    )
                    
                    if color_sim >= self.color_threshold:
                        color_verified_matches.append(m)
                    else:
                        rejected_by_color += 1
                
                if rejected_by_color > 0:
                    logger.debug(f"Color verification rejected {rejected_by_color} matches "
                               f"(kept {len(color_verified_matches)}/{len(good_matches)})")
                
                good_matches = color_verified_matches
            
            if len(good_matches) < 10:
                return {
                    'matches': good_matches,
                    'num_matches': len(good_matches),
                    'confidence': 0.0,
                    'homography': None
                }
            
            # Calculate confidence
            max_descriptors = max(len(descriptors1), len(descriptors2))
            confidence = len(good_matches) / max_descriptors if max_descriptors > 0 else 0.0
            
            return {
                'matches': good_matches,
                'num_matches': len(good_matches),
                'confidence': confidence,
                'homography': None
            }
        except Exception as e:
            logger.error(f"Error during feature matching: {e}", exc_info=True)
            return {
                'matches': [],
                'num_matches': 0,
                'confidence': 0.0,
                'homography': None
            }

