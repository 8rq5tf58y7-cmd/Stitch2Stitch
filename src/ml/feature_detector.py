"""
Advanced feature detection using LP-SIFT and other modern algorithms
With color-aware feature extraction for better matching
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


def extract_color_at_keypoints(
    image: np.ndarray,
    keypoints: np.ndarray,
    patch_size: int = 11
) -> np.ndarray:
    """
    Extract color histograms at keypoint locations.
    
    Args:
        image: BGR color image
        keypoints: Nx4 array of keypoints (x, y, size, angle)
        patch_size: Size of patch around keypoint for color extraction
        
    Returns:
        Nx48 array of color descriptors (16 bins per channel in LAB space)
    """
    if len(keypoints) == 0 or len(image.shape) != 3:
        return np.array([])
    
    h, w = image.shape[:2]
    half_patch = patch_size // 2
    n_bins = 16
    
    # Convert to LAB for perceptually uniform color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    color_descs = []
    for kp in keypoints:
        x, y = int(kp[0]), int(kp[1])
        
        # Get patch bounds (clipped to image)
        x1 = max(0, x - half_patch)
        y1 = max(0, y - half_patch)
        x2 = min(w, x + half_patch + 1)
        y2 = min(h, y + half_patch + 1)
        
        if x2 <= x1 or y2 <= y1:
            # Fallback: use single pixel
            if 0 <= x < w and 0 <= y < h:
                pixel = lab[y, x]
                # Create histogram from single pixel
                hist = np.zeros(n_bins * 3, dtype=np.float32)
                for c in range(3):
                    bin_idx = int(pixel[c] / 256 * n_bins)
                    bin_idx = min(bin_idx, n_bins - 1)
                    hist[c * n_bins + bin_idx] = 1.0
                color_descs.append(hist)
            else:
                color_descs.append(np.zeros(n_bins * 3, dtype=np.float32))
            continue
        
        patch = lab[y1:y2, x1:x2]
        
        # Compute histogram for each channel
        hist = []
        for c in range(3):
            channel_hist, _ = np.histogram(patch[:, :, c], bins=n_bins, range=(0, 256))
            # Normalize
            channel_hist = channel_hist.astype(np.float32)
            if channel_hist.sum() > 0:
                channel_hist /= channel_hist.sum()
            hist.extend(channel_hist)
        
        color_descs.append(np.array(hist, dtype=np.float32))
    
    return np.array(color_descs, dtype=np.float32)


class LP_SIFTDetector:
    """
    Local-Peak SIFT detector with color-aware feature extraction
    Optimized SIFT variant focusing on multiscale local peaks
    """
    
    def __init__(self, use_gpu: bool = False, n_features: int = 5000, use_color: bool = True):
        """
        Initialize LP-SIFT detector
        
        Args:
            use_gpu: Enable GPU acceleration
            n_features: Maximum number of features to detect
            use_color: Extract color descriptors alongside SIFT (helps differentiate similar shapes)
        """
        self.use_gpu = use_gpu
        self.n_features = n_features
        self.use_color = use_color
        
        # Initialize SIFT detector with optimized parameters
        self.sift = cv2.SIFT_create(
            nfeatures=n_features,
            contrastThreshold=0.04,
            edgeThreshold=10,
            sigma=1.6
        )
        
        logger.info(f"LP-SIFT detector initialized (GPU: {use_gpu}, color: {use_color})")
    
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
            If use_color=True, descriptors include color information appended
        """
        # Keep original color image for color extraction
        color_image = image if len(image.shape) == 3 else None
        
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
        
        # Extract and append color descriptors if enabled
        if self.use_color and color_image is not None and len(kp_array) > 0 and descriptors is not None:
            color_descs = extract_color_at_keypoints(color_image, kp_array)
            if len(color_descs) == len(descriptors):
                # Append color descriptors (scaled to match SIFT descriptor range)
                # SIFT descriptors are ~0-255, color histograms are 0-1
                color_scaled = color_descs * 128  # Scale to similar range
                descriptors = np.hstack([descriptors, color_scaled])
                logger.debug(f"Appended color descriptors: {descriptors.shape}")
        
        return kp_array, descriptors
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better feature detection"""
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)
        
        return enhanced
    
    def _compute_edge_priority(self, keypoints: list, image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Compute edge-priority weights for keypoints.
        
        Features near image edges are more likely to be in overlap regions with adjacent
        images in a panorama, so they should be prioritized for detection.
        
        Args:
            keypoints: List of cv2.KeyPoint objects
            image_shape: (height, width) of the image
            
        Returns:
            Array of edge weights (1.0 = at edge, lower = toward center)
        """
        h, w = image_shape[:2]
        if len(keypoints) == 0:
            return np.array([])
        
        weights = []
        for kp in keypoints:
            x, y = kp.pt
            
            # Normalized distance from each edge (0 = at edge, 0.5 = at center)
            dist_left = x / w
            dist_right = 1 - x / w
            dist_top = y / h
            dist_bottom = 1 - y / h
            
            # Minimum distance to any edge
            min_dist = min(dist_left, dist_right, dist_top, dist_bottom)
            
            # Edge priority: exponential decay from edge
            # Points within 25% of edge get significant boost
            edge_weight = np.exp(-min_dist * 4)  # ~0.37 at 25% from edge
            
            weights.append(edge_weight)
        
        return np.array(weights)
    
    def _filter_local_peaks(
        self,
        keypoints: list,
        descriptors: np.ndarray,
        image: np.ndarray
    ) -> Tuple[list, np.ndarray]:
        """
        Filter keypoints to focus on local peaks with edge prioritization.
        
        This implements an enhanced LP-SIFT approach that:
        1. Boosts features near image edges (overlap regions)
        2. Maintains quality features throughout the image
        3. Ensures good coverage for panorama stitching
        """
        if len(keypoints) == 0:
            return keypoints, descriptors
        
        h, w = image.shape[:2]
        
        # Get feature responses
        responses = np.array([kp.response for kp in keypoints])
        
        # Normalize responses to 0-1 range
        if responses.max() > responses.min():
            norm_responses = (responses - responses.min()) / (responses.max() - responses.min())
        else:
            norm_responses = np.ones_like(responses)
        
        # Compute edge priority weights
        edge_weights = self._compute_edge_priority(keypoints, (h, w))
        
        # Combined score: 70% response quality + 30% edge priority
        # This ensures edge features are prioritized but still requires good feature quality
        combined_scores = 0.7 * norm_responses + 0.3 * edge_weights
        
        # Sort by combined score
        sorted_indices = np.argsort(combined_scores)[::-1]
        
        # Keep top N features, but ensure minimum 20% are from edge regions
        n_keep = min(self.n_features, len(keypoints))
        
        # First, identify edge features (within 15% of boundary)
        edge_threshold = 0.15
        is_edge_feature = edge_weights > np.exp(-edge_threshold * 4)  # ~0.55 threshold
        
        # Count how many edge features we'd get in top N
        top_indices = sorted_indices[:n_keep]
        edge_count_in_top = np.sum(is_edge_feature[top_indices])
        min_edge_features = int(n_keep * 0.2)  # Minimum 20% edge features
        
        # If not enough edge features, add more
        if edge_count_in_top < min_edge_features:
            # Get edge features not in top N, sorted by response
            edge_indices = np.where(is_edge_feature)[0]
            edge_not_in_top = [i for i in edge_indices if i not in top_indices]
            edge_responses = [(i, responses[i]) for i in edge_not_in_top]
            edge_responses.sort(key=lambda x: -x[1])
            
            # Add best edge features
            n_to_add = min_edge_features - edge_count_in_top
            additional_edge = [i for i, _ in edge_responses[:n_to_add]]
            
            # Replace lowest-scored non-edge features
            non_edge_in_top = [i for i in top_indices if not is_edge_feature[i]]
            non_edge_in_top.sort(key=lambda i: combined_scores[i])
            
            keep_indices = list(top_indices)
            for add_idx in additional_edge:
                if non_edge_in_top:
                    remove_idx = non_edge_in_top.pop(0)
                    keep_indices.remove(remove_idx)
                    keep_indices.append(add_idx)
        else:
            keep_indices = list(top_indices)
        
        filtered_kp = [keypoints[i] for i in keep_indices]
        filtered_desc = descriptors[keep_indices]
        
        # Log edge feature statistics
        final_edge_count = sum(1 for i in keep_indices if is_edge_feature[i])
        logger.debug(f"Features: {len(filtered_kp)} total, {final_edge_count} ({100*final_edge_count/len(filtered_kp):.0f}%) near edges")
        
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

