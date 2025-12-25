"""
Control Point Management System (PTGui-style)
Allows manual and automatic control point detection and editing
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ControlPoint:
    """Represents a control point pair between two images"""
    
    def __init__(
        self,
        image1_idx: int,
        image2_idx: int,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        confidence: float = 1.0,
        manual: bool = False
    ):
        self.image1_idx = image1_idx
        self.image2_idx = image2_idx
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.confidence = confidence
        self.manual = manual  # True if manually added/edited
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'image1_idx': self.image1_idx,
            'image2_idx': self.image2_idx,
            'x1': self.x1,
            'y1': self.y1,
            'x2': self.x2,
            'y2': self.y2,
            'confidence': self.confidence,
            'manual': self.manual
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ControlPoint':
        """Create from dictionary"""
        return cls(
            image1_idx=data['image1_idx'],
            image2_idx=data['image2_idx'],
            x1=data['x1'],
            y1=data['y1'],
            x2=data['x2'],
            y2=data['y2'],
            confidence=data.get('confidence', 1.0),
            manual=data.get('manual', False)
        )


class ControlPointManager:
    """Manages control points between images (PTGui-style)"""
    
    def __init__(self):
        self.control_points: List[ControlPoint] = []
        logger.info("Control point manager initialized")
    
    def add_control_point(
        self,
        image1_idx: int,
        image2_idx: int,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        manual: bool = True
    ) -> ControlPoint:
        """Add a control point"""
        cp = ControlPoint(image1_idx, image2_idx, x1, y1, x2, y2, manual=manual)
        self.control_points.append(cp)
        logger.info(f"Added control point: ({x1:.1f}, {y1:.1f}) <-> ({x2:.1f}, {y2:.1f})")
        return cp
    
    def remove_control_point(self, cp: ControlPoint):
        """Remove a control point"""
        if cp in self.control_points:
            self.control_points.remove(cp)
            logger.info("Removed control point")
    
    def get_control_points_for_pair(
        self,
        image1_idx: int,
        image2_idx: int
    ) -> List[ControlPoint]:
        """Get all control points for a specific image pair"""
        return [
            cp for cp in self.control_points
            if (cp.image1_idx == image1_idx and cp.image2_idx == image2_idx) or
               (cp.image1_idx == image2_idx and cp.image2_idx == image1_idx)
        ]
    
    def auto_detect_control_points(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
        image1_idx: int,
        image2_idx: int,
        max_points: int = 50
    ) -> List[ControlPoint]:
        """
        Automatically detect control points between two images
        Similar to PTGui's automatic control point detection
        """
        # Use SIFT to detect features
        sift = cv2.SIFT_create(nfeatures=max_points * 2)
        
        # Detect features
        kp1, desc1 = sift.detectAndCompute(image1, None)
        kp2, desc2 = sift.detectAndCompute(image2, None)
        
        if desc1 is None or desc2 is None or len(desc1) < 4 or len(desc2) < 4:
            logger.warning("Not enough features for control point detection")
            return []
        
        # Match features
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=100)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        matches = flann.knnMatch(desc1, desc2, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
        
        # Extract control points from good matches
        control_points = []
        for match in good_matches[:max_points]:
            pt1 = kp1[match.queryIdx].pt
            pt2 = kp2[match.trainIdx].pt
            
            # Calculate confidence based on match distance
            confidence = 1.0 - (match.distance / 300.0)  # Normalize
            confidence = max(0.0, min(1.0, confidence))
            
            cp = ControlPoint(
                image1_idx=image1_idx,
                image2_idx=image2_idx,
                x1=pt1[0],
                y1=pt1[1],
                x2=pt2[0],
                y2=pt2[1],
                confidence=confidence,
                manual=False
            )
            control_points.append(cp)
        
        # Add to manager
        self.control_points.extend(control_points)
        logger.info(f"Auto-detected {len(control_points)} control points")
        
        return control_points
    
    def estimate_homography_from_control_points(
        self,
        image1_idx: int,
        image2_idx: int,
        allow_scale: bool = True
    ) -> Optional[np.ndarray]:
        """
        Estimate similarity transform from control points
        
        Args:
            image1_idx: First image index
            image2_idx: Second image index
            allow_scale: Allow uniform scaling (default True)
            
        Returns:
            3x3 transform matrix or None
        """
        cps = self.get_control_points_for_pair(image1_idx, image2_idx)
        
        if len(cps) < 1:
            return None
        
        # Extract points
        pts1 = []
        pts2 = []
        
        for cp in cps:
            # Ensure correct order
            if cp.image1_idx == image1_idx:
                pts1.append([cp.x1, cp.y1])
                pts2.append([cp.x2, cp.y2])
            else:
                pts1.append([cp.x2, cp.y2])
                pts2.append([cp.x1, cp.y1])
        
        pts1 = np.float32(pts1)
        pts2 = np.float32(pts2)
        
        # If we have enough points, use similarity transform
        if len(pts1) >= 3 and allow_scale:
            similarity, inliers = cv2.estimateAffinePartial2D(
                pts1, pts2,
                method=cv2.RANSAC,
                ransacReprojThreshold=3.0,
                confidence=0.99
            )
            
            if similarity is not None:
                # Validate scale (rotation is allowed)
                scale = np.sqrt(similarity[0, 0]**2 + similarity[0, 1]**2)
                
                # Allow reasonable scale (0.5x to 2.0x), rotation is unlimited
                if 0.5 <= scale <= 2.0:
                    return np.vstack([similarity, [0, 0, 1]]).astype(np.float32)
        
        # Fall back to translation-only
        translations = pts2 - pts1
        median_translation = np.median(translations, axis=0)
        
        transform = np.array([
            [1, 0, median_translation[0]],
            [0, 1, median_translation[1]],
            [0, 0, 1]
        ], dtype=np.float32)
        
        return transform
    
    def validate_control_point(
        self,
        cp: ControlPoint,
        image1_shape: Tuple[int, int],
        image2_shape: Tuple[int, int]
    ) -> bool:
        """Validate that control point is within image bounds"""
        h1, w1 = image1_shape[:2]
        h2, w2 = image2_shape[:2]
        
        return (0 <= cp.x1 < w1 and 0 <= cp.y1 < h1 and
                0 <= cp.x2 < w2 and 0 <= cp.y2 < h2)
    
    def get_all_control_points(self) -> List[ControlPoint]:
        """Get all control points"""
        return self.control_points.copy()
    
    def clear_control_points(self):
        """Clear all control points"""
        self.control_points.clear()
        logger.info("Cleared all control points")
    
    def export_to_dict(self) -> List[Dict]:
        """Export control points to dictionary list"""
        return [cp.to_dict() for cp in self.control_points]
    
    def import_from_dict(self, data: List[Dict]):
        """Import control points from dictionary list"""
        self.control_points = [ControlPoint.from_dict(cp_dict) for cp_dict in data]
        logger.info(f"Imported {len(self.control_points)} control points")

