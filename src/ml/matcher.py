"""
Advanced feature matching with RANSAC, geometric verification, and confidence scoring
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


class AdvancedMatcher:
    """Advanced feature matcher with geometric verification"""
    
    def __init__(
        self,
        use_gpu: bool = False,
        method: str = 'flann',
        ratio_threshold: float = 0.70,
        geometric_verify: bool = True
    ):
        """
        Initialize matcher
        
        Args:
            use_gpu: Enable GPU acceleration
            method: Matching method ('flann' or 'bf')
            ratio_threshold: Lowe's ratio test threshold (lower = stricter)
            geometric_verify: Apply geometric verification to filter bad matches
        """
        self.use_gpu = use_gpu
        # Stricter default ratio threshold (0.70 instead of 0.75)
        self.ratio_threshold = ratio_threshold if ratio_threshold > 0 else 0.70
        self.geometric_verify = geometric_verify
        
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
        
        logger.info(f"Advanced matcher initialized (method: {method}, ratio: {self.ratio_threshold}, geo_verify: {geometric_verify})")
    
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
                    'homography': None,
                    'inliers': 0
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
                'homography': None,  # Will be calculated in alignment module
                'inliers': len(good_matches)  # Will be updated by geometric verification
            }
        except Exception as e:
            logger.error(f"Error during feature matching: {e}", exc_info=True)
            return {
                'matches': [],
                'num_matches': 0,
                'confidence': 0.0,
                'homography': None,
                'inliers': 0
            }
    
    def match_with_keypoints(
        self,
        keypoints1: np.ndarray,
        descriptors1: np.ndarray,
        keypoints2: np.ndarray,
        descriptors2: np.ndarray
    ) -> Dict:
        """
        Match descriptors with geometric verification using keypoint positions.
        
        This method filters out geometrically inconsistent matches using RANSAC
        to estimate a similarity transform and reject outliers.
        
        Args:
            keypoints1: Keypoints from first image (Nx2 array of [x, y])
            descriptors1: Descriptors from first image
            keypoints2: Keypoints from second image (Nx2 array of [x, y])
            descriptors2: Descriptors from second image
            
        Returns:
            Dictionary with verified match results including inlier information
        """
        # First get basic matches
        basic_result = self.match(descriptors1, descriptors2)
        
        if basic_result['num_matches'] < 10 or not self.geometric_verify:
            return basic_result
        
        # Extract matched keypoint positions
        pts1 = []
        pts2 = []
        for m in basic_result['matches']:
            idx1 = m.queryIdx
            idx2 = m.trainIdx
            if idx1 < len(keypoints1) and idx2 < len(keypoints2):
                pts1.append(keypoints1[idx1][:2])  # x, y
                pts2.append(keypoints2[idx2][:2])
        
        if len(pts1) < 4:
            return basic_result
        
        pts1 = np.float32(pts1)
        pts2 = np.float32(pts2)
        
        # Use RANSAC to find inliers via similarity transform
        # This filters out matches that don't fit the geometric model
        transform, inliers = cv2.estimateAffinePartial2D(
            pts1, pts2,
            method=cv2.RANSAC,
            ransacReprojThreshold=2.5,  # Stricter threshold (default is 3.0)
            confidence=0.995,
            maxIters=2000
        )
        
        if inliers is None:
            logger.warning("Geometric verification failed - no inliers found")
            return basic_result
        
        # Filter matches to keep only inliers
        inlier_mask = inliers.ravel().astype(bool)
        verified_matches = [m for m, is_inlier in zip(basic_result['matches'], inlier_mask) if is_inlier]
        
        inlier_count = len(verified_matches)
        total_matches = basic_result['num_matches']
        inlier_ratio = inlier_count / total_matches if total_matches > 0 else 0
        
        logger.debug(f"Geometric verification: {inlier_count}/{total_matches} inliers ({inlier_ratio:.1%})")
        
        # Reject if inlier ratio is too low (indicates bad match)
        if inlier_ratio < 0.35:
            logger.info(f"Rejecting match: low inlier ratio {inlier_ratio:.1%}")
            return {
                'matches': [],
                'num_matches': 0,
                'confidence': 0.0,
                'homography': None,
                'inliers': 0,
                'rejected_reason': f'low_inlier_ratio:{inlier_ratio:.2f}'
            }
        
        # Check transform validity
        if transform is not None:
            scale = np.sqrt(transform[0, 0]**2 + transform[0, 1]**2)
            rotation = np.abs(np.degrees(np.arctan2(transform[1, 0], transform[0, 0])))
            
            # Reject extreme transforms
            if scale < 0.3 or scale > 3.0:
                logger.info(f"Rejecting match: extreme scale {scale:.2f}")
                return {
                    'matches': [],
                    'num_matches': 0,
                    'confidence': 0.0,
                    'homography': None,
                    'inliers': 0,
                    'rejected_reason': f'extreme_scale:{scale:.2f}'
                }
        
        # Update confidence based on verified matches
        max_descriptors = max(len(descriptors1), len(descriptors2))
        confidence = inlier_count / max_descriptors if max_descriptors > 0 else 0.0
        
        return {
            'matches': verified_matches,
            'num_matches': inlier_count,
            'confidence': confidence,
            'homography': transform,
            'inliers': inlier_count,
            'inlier_ratio': inlier_ratio
        }
    
    def filter_spatial_outliers(
        self,
        matches: List,
        keypoints1: np.ndarray,
        keypoints2: np.ndarray,
        k_neighbors: int = 5
    ) -> List:
        """
        Filter matches based on spatial consistency using local neighborhood voting.
        
        A good match should have neighbors that agree on similar displacement.
        Bad matches have random displacements that differ from their neighbors.
        
        Args:
            matches: List of cv2.DMatch objects
            keypoints1: Keypoints from first image
            keypoints2: Keypoints from second image
            k_neighbors: Number of neighbors to consider
            
        Returns:
            Filtered list of matches
        """
        if len(matches) < k_neighbors + 1:
            return matches
        
        # Calculate displacement vectors for each match
        displacements = []
        positions1 = []
        for m in matches:
            idx1 = m.queryIdx
            idx2 = m.trainIdx
            if idx1 < len(keypoints1) and idx2 < len(keypoints2):
                pt1 = keypoints1[idx1][:2]
                pt2 = keypoints2[idx2][:2]
                displacements.append(pt2 - pt1)
                positions1.append(pt1)
        
        if len(displacements) < k_neighbors + 1:
            return matches
        
        displacements = np.array(displacements)
        positions1 = np.array(positions1)
        
        # For each match, check if its displacement is consistent with neighbors
        consistent_mask = []
        for i in range(len(displacements)):
            # Find k nearest neighbors in image 1
            distances = np.linalg.norm(positions1 - positions1[i], axis=1)
            distances[i] = np.inf  # Exclude self
            neighbor_indices = np.argsort(distances)[:k_neighbors]
            
            # Calculate median displacement of neighbors
            neighbor_displacements = displacements[neighbor_indices]
            median_displacement = np.median(neighbor_displacements, axis=0)
            
            # Check if this match's displacement is close to neighbor median
            diff = np.linalg.norm(displacements[i] - median_displacement)
            
            # Adaptive threshold based on neighbor variance
            neighbor_variance = np.std(neighbor_displacements)
            threshold = max(30, neighbor_variance * 2.5)  # At least 30 pixels tolerance
            
            consistent_mask.append(diff < threshold)
        
        # Keep only consistent matches
        filtered = [m for m, is_consistent in zip(matches, consistent_mask) if is_consistent]
        
        if len(filtered) < len(matches) * 0.5 and len(filtered) < 15:
            # If filtering removed too many, return original (might be a difficult case)
            logger.debug(f"Spatial filter would remove too many matches ({len(filtered)}/{len(matches)}), keeping original")
            return matches
        
        logger.debug(f"Spatial filter: {len(filtered)}/{len(matches)} matches kept")
        return filtered

