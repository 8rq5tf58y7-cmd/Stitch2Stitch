"""
Smart image selection for panorama stitching
Selects optimal subset of images for coverage with minimal overlap
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Callable
import logging

logger = logging.getLogger(__name__)


class ImageSelector:
    """
    Selects optimal subset of images for stitching.
    
    Goals:
    1. Full coverage - don't skip areas
    2. Minimal overlap - reduce redundancy (target: ≤25% overlap)
    3. Best quality - prefer higher quality images when choices exist
    """
    
    def __init__(
        self,
        max_overlap_percent: float = 25.0,
        min_overlap_percent: float = 5.0,
        thumbnail_size: int = 256
    ):
        """
        Initialize image selector.
        
        Args:
            max_overlap_percent: Maximum allowed overlap between adjacent images (0-100)
            min_overlap_percent: Minimum overlap needed to maintain connectivity (0-100)
            thumbnail_size: Size for thumbnail comparison
        """
        self.max_overlap = max_overlap_percent / 100.0
        self.min_overlap = min_overlap_percent / 100.0
        self.thumbnail_size = thumbnail_size
        
        logger.info(f"ImageSelector initialized (max_overlap: {max_overlap_percent}%, min_overlap: {min_overlap_percent}%)")
    
    def select_images(
        self,
        image_paths: List[Path],
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Tuple[List[Path], Dict]:
        """
        Select optimal subset of images for stitching.
        
        Args:
            image_paths: List of all image paths
            progress_callback: Optional callback(current, total, message)
            
        Returns:
            Tuple of (selected_paths, stats_dict)
        """
        if len(image_paths) <= 2:
            return image_paths, {'selected': len(image_paths), 'total': len(image_paths)}
        
        total = len(image_paths)
        logger.info(f"Selecting optimal images from {total} candidates...")
        
        # Step 1: Load thumbnails and compute features
        if progress_callback:
            progress_callback(0, total, "Loading thumbnails...")
        
        thumbnails = []
        for i, path in enumerate(image_paths):
            if progress_callback and i % 10 == 0:
                progress_callback(i, total, f"Loading thumbnail: {path.name}")
            
            img = cv2.imread(str(path))
            if img is not None:
                h, w = img.shape[:2]
                scale = self.thumbnail_size / max(h, w)
                thumb = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                thumbnails.append({
                    'path': path,
                    'thumbnail': thumb,
                    'index': i
                })
            else:
                logger.warning(f"Could not load: {path}")
        
        if len(thumbnails) <= 2:
            return [t['path'] for t in thumbnails], {'selected': len(thumbnails), 'total': total}
        
        # Step 2: Compute pairwise similarity/overlap
        if progress_callback:
            progress_callback(0, total, "Computing image relationships...")
        
        overlap_matrix = self._compute_overlap_matrix(thumbnails, progress_callback)
        
        # Step 3: Select optimal subset using greedy algorithm
        if progress_callback:
            progress_callback(0, total, "Selecting optimal images...")
        
        selected_indices = self._select_optimal_subset(thumbnails, overlap_matrix)
        
        # Get selected paths
        selected_paths = [thumbnails[i]['path'] for i in selected_indices]
        
        # Compute stats
        stats = {
            'selected': len(selected_paths),
            'total': total,
            'reduction_percent': (1 - len(selected_paths) / total) * 100,
            'avg_overlap': self._compute_avg_overlap(selected_indices, overlap_matrix)
        }
        
        logger.info(f"Selected {stats['selected']}/{stats['total']} images "
                   f"({stats['reduction_percent']:.1f}% reduction, "
                   f"avg overlap: {stats['avg_overlap']*100:.1f}%)")
        
        return selected_paths, stats
    
    def _compute_overlap_matrix(
        self,
        thumbnails: List[Dict],
        progress_callback: Optional[Callable] = None
    ) -> np.ndarray:
        """Compute pairwise overlap/similarity matrix"""
        n = len(thumbnails)
        overlap = np.zeros((n, n), dtype=np.float32)
        
        # Use feature matching to estimate overlap
        orb = cv2.ORB_create(nfeatures=500)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Compute features for all images
        features = []
        for t in thumbnails:
            gray = cv2.cvtColor(t['thumbnail'], cv2.COLOR_BGR2GRAY)
            kp, desc = orb.detectAndCompute(gray, None)
            features.append({
                'keypoints': kp,
                'descriptors': desc,
                'area': gray.shape[0] * gray.shape[1]
            })
        
        # Compute pairwise overlap
        # Only compare adjacent images (within window) for efficiency
        window_size = min(20, n)  # Compare up to 20 neighbors
        
        pair_count = 0
        total_pairs = n * min(window_size, n - 1)
        
        for i in range(n):
            if progress_callback and i % 5 == 0:
                progress_callback(pair_count, total_pairs, f"Analyzing image relationships ({i}/{n})...")
            
            for j in range(i + 1, min(i + window_size + 1, n)):
                pair_count += 1
                
                desc1 = features[i]['descriptors']
                desc2 = features[j]['descriptors']
                
                if desc1 is None or desc2 is None:
                    continue
                
                if len(desc1) < 10 or len(desc2) < 10:
                    continue
                
                try:
                    matches = bf.match(desc1, desc2)
                    
                    # Estimate overlap from match count and quality
                    if len(matches) > 0:
                        # Sort by distance
                        matches = sorted(matches, key=lambda x: x.distance)
                        good_matches = [m for m in matches if m.distance < 50]
                        
                        # Overlap estimate based on good matches
                        max_possible = min(len(desc1), len(desc2))
                        if max_possible > 0:
                            overlap_estimate = len(good_matches) / max_possible
                            
                            # Also consider spatial distribution of matches
                            if len(good_matches) >= 4:
                                # Check if matches cover a significant area
                                kp1 = features[i]['keypoints']
                                kp2 = features[j]['keypoints']
                                pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
                                pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
                                
                                # Bounding box of matched points
                                area1 = (pts1[:, 0].max() - pts1[:, 0].min()) * (pts1[:, 1].max() - pts1[:, 1].min())
                                area2 = (pts2[:, 0].max() - pts2[:, 0].min()) * (pts2[:, 1].max() - pts2[:, 1].min())
                                
                                # Spatial coverage factor
                                img_area = features[i]['area']
                                spatial_factor = min(area1, area2) / img_area if img_area > 0 else 0
                                
                                # Combined overlap estimate
                                overlap_estimate = overlap_estimate * 0.5 + spatial_factor * 0.5
                            
                            overlap[i, j] = overlap_estimate
                            overlap[j, i] = overlap_estimate
                            
                except Exception as e:
                    logger.debug(f"Error matching {i}-{j}: {e}")
                    continue
        
        return overlap
    
    def _select_optimal_subset(
        self,
        thumbnails: List[Dict],
        overlap_matrix: np.ndarray
    ) -> List[int]:
        """
        Select optimal subset using greedy coverage algorithm.
        
        Strategy:
        1. Start with first image
        2. Skip images that overlap too much with selected images
        3. Include images that add new coverage
        4. Ensure connectivity (min overlap with at least one selected image)
        """
        n = len(thumbnails)
        selected = [0]  # Always start with first image
        
        for i in range(1, n):
            # Check overlap with all selected images
            overlaps_with_selected = [overlap_matrix[i, j] for j in selected]
            max_overlap_with_selected = max(overlaps_with_selected) if overlaps_with_selected else 0
            
            # Decision logic:
            # 1. If overlap with any selected image > max_overlap, skip (too redundant)
            # 2. If overlap with any selected image >= min_overlap, include (maintains connectivity)
            # 3. If no overlap at all, include (new coverage)
            
            if max_overlap_with_selected > self.max_overlap:
                # Too much overlap - skip this image
                logger.debug(f"Skipping image {i}: overlap {max_overlap_with_selected*100:.1f}% > max {self.max_overlap*100:.1f}%")
                continue
            
            # Check if this image provides connectivity or new coverage
            has_connectivity = any(o >= self.min_overlap for o in overlaps_with_selected)
            
            if has_connectivity or max_overlap_with_selected == 0:
                selected.append(i)
                logger.debug(f"Including image {i}: max_overlap={max_overlap_with_selected*100:.1f}%")
        
        return selected
    
    def _compute_avg_overlap(
        self,
        selected_indices: List[int],
        overlap_matrix: np.ndarray
    ) -> float:
        """Compute average overlap between selected images"""
        if len(selected_indices) < 2:
            return 0.0
        
        overlaps = []
        for i, idx1 in enumerate(selected_indices):
            for idx2 in selected_indices[i + 1:]:
                overlaps.append(overlap_matrix[idx1, idx2])
        
        return np.mean(overlaps) if overlaps else 0.0


def select_optimal_images(
    image_paths: List[Path],
    max_overlap_percent: float = 25.0,
    min_overlap_percent: float = 5.0,
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> Tuple[List[Path], Dict]:
    """
    Convenience function to select optimal images.
    
    Args:
        image_paths: List of all image paths
        max_overlap_percent: Maximum allowed overlap (default: 25%)
        min_overlap_percent: Minimum overlap for connectivity (default: 5%)
        progress_callback: Optional progress callback
        
    Returns:
        Tuple of (selected_paths, stats_dict)
    """
    selector = ImageSelector(
        max_overlap_percent=max_overlap_percent,
        min_overlap_percent=min_overlap_percent
    )
    return selector.select_images(image_paths, progress_callback)

