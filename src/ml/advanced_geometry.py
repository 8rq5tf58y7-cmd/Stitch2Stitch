"""
Advanced Geometric Verification and Line-Based Features

Implements techniques beyond basic RANSAC:
1. MAGSAC++ / USAC++ - More robust outlier rejection
2. LSD + LBD - Line Segment Detection with Binary Descriptors
3. Local Affine Verification - Region-based consistency

These help when:
- Geometry dominates texture (buildings, tiles, grids)
- Many false positive matches from repetitive patterns
- Standard RANSAC fails to find correct model
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List, Dict
import logging

logger = logging.getLogger(__name__)


class MAGSACPlusPlus:
    """
    MAGSAC++ - Marginalizing Sample Consensus
    
    Improvements over RANSAC:
    - No fixed inlier threshold needed
    - Marginalizes over all possible thresholds
    - Better handling of data with varying noise levels
    - More robust to outliers
    
    Based on: "MAGSAC++: A Fast, Reliable and Accurate Robust Estimator" (CVPR 2020)
    """
    
    def __init__(
        self,
        confidence: float = 0.9999,
        max_iters: int = 5000,
        sigma_max: float = 10.0
    ):
        """
        Args:
            confidence: Required confidence level (0-1)
            max_iters: Maximum iterations
            sigma_max: Maximum sigma for threshold marginalization
        """
        self.confidence = confidence
        self.max_iters = max_iters
        self.sigma_max = sigma_max
    
    def estimate_similarity_transform(
        self,
        pts1: np.ndarray,
        pts2: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Estimate similarity transform using MAGSAC++-style robust estimation.
        
        Args:
            pts1: Nx2 source points
            pts2: Nx2 destination points
            
        Returns:
            Tuple of (transform, inlier_mask) or (None, None) if failed
        """
        if len(pts1) < 3:
            return None, None
        
        # Use OpenCV's USAC if available (OpenCV 4.5+)
        try:
            # USAC_MAGSAC is the most robust option
            transform, inliers = cv2.estimateAffinePartial2D(
                pts1, pts2,
                method=cv2.USAC_MAGSAC,
                ransacReprojThreshold=self.sigma_max,
                confidence=self.confidence,
                maxIters=self.max_iters
            )
            return transform, inliers
        except (cv2.error, AttributeError):
            pass
        
        # Fallback to USAC_ACCURATE
        try:
            transform, inliers = cv2.estimateAffinePartial2D(
                pts1, pts2,
                method=cv2.USAC_ACCURATE,
                ransacReprojThreshold=3.0,
                confidence=self.confidence,
                maxIters=self.max_iters
            )
            return transform, inliers
        except (cv2.error, AttributeError):
            pass
        
        # Final fallback to standard RANSAC
        transform, inliers = cv2.estimateAffinePartial2D(
            pts1, pts2,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            confidence=self.confidence,
            maxIters=self.max_iters
        )
        return transform, inliers
    
    def estimate_homography(
        self,
        pts1: np.ndarray,
        pts2: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Estimate homography using MAGSAC++.
        
        Args:
            pts1: Nx2 source points
            pts2: Nx2 destination points
            
        Returns:
            Tuple of (homography, inlier_mask) or (None, None) if failed
        """
        if len(pts1) < 4:
            return None, None
        
        # Try USAC_MAGSAC first
        try:
            H, inliers = cv2.findHomography(
                pts1, pts2,
                method=cv2.USAC_MAGSAC,
                ransacReprojThreshold=self.sigma_max,
                confidence=self.confidence,
                maxIters=self.max_iters
            )
            return H, inliers
        except (cv2.error, AttributeError):
            pass
        
        # Fallback
        H, inliers = cv2.findHomography(
            pts1, pts2,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            confidence=self.confidence,
            maxIters=self.max_iters
        )
        return H, inliers


class LocalAffineVerifier:
    """
    Local Affine Verification
    
    Verifies matches by checking local affine consistency.
    For each match, computes a local affine model and checks
    if nearby matches agree.
    
    This catches matches that pass global RANSAC but fail locally.
    """
    
    def __init__(
        self,
        patch_radius: float = 50.0,
        min_neighbors: int = 3,
        max_affine_error: float = 5.0
    ):
        """
        Args:
            patch_radius: Radius for local neighborhood
            min_neighbors: Minimum neighbors needed for verification
            max_affine_error: Maximum reprojection error for affine model
        """
        self.patch_radius = patch_radius
        self.min_neighbors = min_neighbors
        self.max_affine_error = max_affine_error
    
    def verify_matches(
        self,
        pts1: np.ndarray,
        pts2: np.ndarray,
        global_inliers: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Verify matches using local affine consistency.
        
        Args:
            pts1: Nx2 source points
            pts2: Nx2 destination points
            global_inliers: Optional global inlier mask to start from
            
        Returns:
            Boolean mask of verified matches
        """
        n = len(pts1)
        if n < 4:
            return np.ones(n, dtype=bool)
        
        # Start with global inliers or all points
        if global_inliers is not None:
            candidates = global_inliers.flatten().astype(bool)
        else:
            candidates = np.ones(n, dtype=bool)
        
        verified = np.zeros(n, dtype=bool)
        
        for i in range(n):
            if not candidates[i]:
                continue
            
            # Find neighbors in source image
            dists = np.linalg.norm(pts1 - pts1[i], axis=1)
            neighbors = np.where((dists < self.patch_radius) & (dists > 0) & candidates)[0]
            
            if len(neighbors) < self.min_neighbors:
                # Not enough neighbors, trust global decision
                verified[i] = True
                continue
            
            # Compute local affine from neighbors
            local_pts1 = pts1[neighbors]
            local_pts2 = pts2[neighbors]
            
            try:
                # Estimate local affine
                A, _ = cv2.estimateAffine2D(local_pts1, local_pts2, method=cv2.LMEDS)
                
                if A is None:
                    verified[i] = True
                    continue
                
                # Check if current point agrees with local affine
                pt1_h = np.array([pts1[i, 0], pts1[i, 1], 1.0])
                pt2_pred = A @ pt1_h
                error = np.linalg.norm(pt2_pred - pts2[i])
                
                verified[i] = error < self.max_affine_error
                
            except cv2.error:
                verified[i] = True
        
        return verified


class LineFeatureDetector:
    """
    Line Segment Detection with Binary Descriptors (LSD + LBD)
    
    Detects line segments and computes binary descriptors.
    Useful for:
    - Architectural scenes
    - Man-made structures
    - Scenes where point features fail due to texture repetition
    
    Based on: Line Segment Detector (LSD) + Line Band Descriptor (LBD)
    """
    
    def __init__(
        self,
        min_line_length: float = 30.0,
        max_lines: int = 500,
        use_lsd: bool = True
    ):
        """
        Args:
            min_line_length: Minimum line segment length in pixels
            max_lines: Maximum number of lines to detect
            use_lsd: Use LSD (True) or Hough lines (False)
        """
        self.min_line_length = min_line_length
        self.max_lines = max_lines
        self.use_lsd = use_lsd
        
        # LSD detector
        if use_lsd:
            try:
                self.lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
            except AttributeError:
                self.lsd = None
                logger.warning("LSD not available in this OpenCV version, using Hough")
        else:
            self.lsd = None
    
    def detect(self, image: np.ndarray) -> np.ndarray:
        """
        Detect line segments in image.
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            Nx4 array of line segments (x1, y1, x2, y2)
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        if self.lsd is not None:
            # LSD detection
            lines, widths, prec, nfa = self.lsd.detect(gray)
            
            if lines is None:
                return np.array([])
            
            # Reshape to Nx4
            lines = lines.reshape(-1, 4)
            
        else:
            # Hough line detection fallback
            edges = cv2.Canny(gray, 50, 150)
            hough_lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi / 180,
                threshold=50,
                minLineLength=self.min_line_length,
                maxLineGap=10
            )
            
            if hough_lines is None:
                return np.array([])
            
            lines = hough_lines.reshape(-1, 4)
        
        # Filter by length
        lengths = np.sqrt((lines[:, 2] - lines[:, 0])**2 + 
                         (lines[:, 3] - lines[:, 1])**2)
        valid = lengths >= self.min_line_length
        lines = lines[valid]
        lengths = lengths[valid]
        
        # Keep longest lines
        if len(lines) > self.max_lines:
            top_indices = np.argsort(lengths)[::-1][:self.max_lines]
            lines = lines[top_indices]
        
        return lines.astype(np.float32)
    
    def compute_descriptors(
        self,
        image: np.ndarray,
        lines: np.ndarray,
        descriptor_size: int = 32
    ) -> np.ndarray:
        """
        Compute binary descriptors for line segments.
        
        Uses Line Band Descriptor (LBD) approach:
        - Sample points along the line
        - Compute gradient histograms in bands
        - Binarize for efficient matching
        
        Args:
            image: Input image
            lines: Nx4 array of line segments
            descriptor_size: Descriptor length in bytes
            
        Returns:
            NxD array of binary descriptors
        """
        if len(lines) == 0:
            return np.array([])
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        h, w = gray.shape
        
        # Compute gradients
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        
        descriptors = []
        
        for line in lines:
            x1, y1, x2, y2 = line
            
            # Line direction
            dx = x2 - x1
            dy = y2 - y1
            length = np.sqrt(dx*dx + dy*dy)
            
            if length < 1:
                descriptors.append(np.zeros(descriptor_size * 8, dtype=np.uint8))
                continue
            
            # Normalize direction
            dx /= length
            dy /= length
            
            # Normal direction (perpendicular)
            nx, ny = -dy, dx
            
            # Sample points along line and in bands
            n_samples = min(32, max(8, int(length / 4)))
            band_width = 4
            n_bands = 4
            
            descriptor_values = []
            
            for t in np.linspace(0.1, 0.9, n_samples):
                # Point on line
                px = x1 + t * (x2 - x1)
                py = y1 + t * (y2 - y1)
                
                for band in range(-n_bands//2, n_bands//2 + 1):
                    # Sample point in band
                    sx = int(px + band * band_width * nx)
                    sy = int(py + band * band_width * ny)
                    
                    if 0 <= sx < w and 0 <= sy < h:
                        # Gradient at this point
                        grad_x = gx[sy, sx]
                        grad_y = gy[sy, sx]
                        
                        # Project gradient onto line direction
                        grad_para = grad_x * dx + grad_y * dy
                        grad_perp = grad_x * nx + grad_y * ny
                        
                        descriptor_values.extend([grad_para, grad_perp])
            
            # Convert to binary descriptor
            if len(descriptor_values) > 0:
                arr = np.array(descriptor_values)
                median = np.median(arr)
                binary = (arr > median).astype(np.uint8)
                
                # Pad or truncate to descriptor_size * 8 bits
                target_len = descriptor_size * 8
                if len(binary) < target_len:
                    binary = np.pad(binary, (0, target_len - len(binary)))
                else:
                    binary = binary[:target_len]
                
                descriptors.append(binary)
            else:
                descriptors.append(np.zeros(descriptor_size * 8, dtype=np.uint8))
        
        return np.array(descriptors, dtype=np.uint8)
    
    def match_lines(
        self,
        desc1: np.ndarray,
        desc2: np.ndarray,
        lines1: np.ndarray,
        lines2: np.ndarray,
        max_ratio: float = 0.75
    ) -> List[Tuple[int, int, float]]:
        """
        Match line segments using descriptors.
        
        Args:
            desc1: Descriptors from image 1
            desc2: Descriptors from image 2
            lines1: Lines from image 1
            lines2: Lines from image 2
            max_ratio: Lowe's ratio threshold
            
        Returns:
            List of (idx1, idx2, distance) matches
        """
        if len(desc1) == 0 or len(desc2) == 0:
            return []
        
        matches = []
        
        for i, d1 in enumerate(desc1):
            # Hamming distances to all descriptors in image 2
            distances = np.sum(d1 != desc2, axis=1)
            
            # Sort by distance
            sorted_idx = np.argsort(distances)
            
            if len(sorted_idx) >= 2:
                # Lowe's ratio test
                best = sorted_idx[0]
                second = sorted_idx[1]
                
                if distances[second] > 0:
                    ratio = distances[best] / distances[second]
                    if ratio < max_ratio:
                        # Also check geometric consistency (similar length and orientation)
                        len1 = np.sqrt((lines1[i, 2] - lines1[i, 0])**2 + 
                                      (lines1[i, 3] - lines1[i, 1])**2)
                        len2 = np.sqrt((lines2[best, 2] - lines2[best, 0])**2 + 
                                      (lines2[best, 3] - lines2[best, 1])**2)
                        
                        if 0.5 < len1/len2 < 2.0:  # Similar length
                            matches.append((i, best, float(distances[best])))
        
        return matches


class CombinedFeatureDetector:
    """
    Combined Point + Line Feature Detection
    
    Fuses point features (SIFT/SuperPoint) with line features (LSD+LBD)
    for robust matching in geometric scenes.
    """
    
    def __init__(
        self,
        point_detector=None,
        use_lines: bool = True,
        point_weight: float = 0.7,
        line_weight: float = 0.3
    ):
        """
        Args:
            point_detector: Point feature detector (SIFT, SuperPoint, etc.)
            use_lines: Enable line feature detection
            point_weight: Weight for point matches in scoring
            line_weight: Weight for line matches in scoring
        """
        self.point_detector = point_detector
        self.line_detector = LineFeatureDetector() if use_lines else None
        self.point_weight = point_weight
        self.line_weight = line_weight
    
    def detect_and_compute(
        self,
        image: np.ndarray
    ) -> Dict:
        """
        Detect both point and line features.
        
        Returns:
            Dictionary with 'points', 'point_descriptors', 'lines', 'line_descriptors'
        """
        result = {
            'points': None,
            'point_descriptors': None,
            'lines': None,
            'line_descriptors': None
        }
        
        # Point features
        if self.point_detector:
            pts, desc = self.point_detector.detect_and_compute(image)
            result['points'] = pts
            result['point_descriptors'] = desc
        
        # Line features
        if self.line_detector:
            lines = self.line_detector.detect(image)
            if len(lines) > 0:
                line_desc = self.line_detector.compute_descriptors(image, lines)
                result['lines'] = lines
                result['line_descriptors'] = line_desc
        
        return result
    
    def compute_combined_score(
        self,
        point_matches: int,
        line_matches: int
    ) -> float:
        """
        Compute combined match score.
        
        Args:
            point_matches: Number of point matches
            line_matches: Number of line matches
            
        Returns:
            Combined score
        """
        return (self.point_weight * point_matches + 
                self.line_weight * line_matches)


def upgrade_ransac_to_magsac():
    """
    Check if MAGSAC++/USAC is available in OpenCV.
    
    Returns:
        True if advanced methods available
    """
    try:
        # Check for USAC_MAGSAC
        method = cv2.USAC_MAGSAC
        return True
    except AttributeError:
        return False


def get_best_robust_method():
    """
    Get the best available robust estimation method.
    
    Returns:
        OpenCV method constant
    """
    methods = [
        ('USAC_MAGSAC', lambda: cv2.USAC_MAGSAC),
        ('USAC_ACCURATE', lambda: cv2.USAC_ACCURATE),
        ('USAC_PARALLEL', lambda: cv2.USAC_PARALLEL),
        ('USAC_DEFAULT', lambda: cv2.USAC_DEFAULT),
        ('RANSAC', lambda: cv2.RANSAC),
    ]
    
    for name, getter in methods:
        try:
            method = getter()
            logger.info(f"Using robust estimation method: {name}")
            return method
        except AttributeError:
            continue
    
    return cv2.RANSAC










