"""
Duplicate and similar image detector for burst mode photos.

Uses multiple similarity metrics to identify truly duplicate/redundant images
while preserving sequential burst photos that cover different areas.
"""

import cv2
import numpy as np
from typing import List, Tuple, Set, Optional, Callable
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class DuplicateDetector:
    """
    Detect and remove duplicate/highly similar images from burst photo sets.
    
    Uses a combination of:
    - Perceptual hash (pHash) for content similarity
    - Difference hash (dHash) for structural similarity  
    - Normalized Cross-Correlation (NCC) for pixel-level comparison
    
    Designed to correctly handle burst mode photos where adjacent frames
    are similar but NOT duplicates.
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.92,
        hash_size: int = 16,
        comparison_window: int = 30,
        progress_callback: Optional[Callable] = None
    ):
        """
        Initialize duplicate detector.
        
        Args:
            similarity_threshold: Minimum similarity (0-1) to consider images duplicates.
                                  0.92 is strict - only near-identical images.
                                  0.85 is moderate - similar frames merged.
                                  0.75 is loose - more aggressive deduplication.
            hash_size: Size of perceptual hash (higher = more precise)
            comparison_window: For burst photos, only compare within this many positions
                              (0 = compare all pairs, which is slow for large sets)
            progress_callback: Optional callback(percent, message) for progress updates
        """
        self.similarity_threshold = similarity_threshold
        self.hash_size = hash_size
        self.comparison_window = comparison_window
        self.progress_callback = progress_callback
        
        logger.info(f"DuplicateDetector initialized: threshold={similarity_threshold:.2f}, "
                   f"hash_size={hash_size}, window={comparison_window}")
    
    def _update_progress(self, percent: float, message: str):
        """Update progress if callback is set."""
        if self.progress_callback:
            self.progress_callback(percent, message)
    
    def compute_phash(self, image: np.ndarray) -> np.ndarray:
        """
        Compute perceptual hash of image.
        
        pHash is robust to scaling and minor color changes.
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Resize to hash_size x hash_size
        resized = cv2.resize(gray, (self.hash_size, self.hash_size), 
                            interpolation=cv2.INTER_AREA)
        
        # Apply DCT
        dct = cv2.dct(np.float32(resized))
        
        # Use top-left 8x8 for hash (low frequencies)
        dct_low = dct[:8, :8]
        
        # Compute median and create binary hash
        median = np.median(dct_low)
        return (dct_low > median).flatten()
    
    def compute_dhash(self, image: np.ndarray) -> np.ndarray:
        """
        Compute difference hash of image.
        
        dHash captures edge/gradient information.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Resize to (hash_size+1) x hash_size
        resized = cv2.resize(gray, (self.hash_size + 1, self.hash_size),
                            interpolation=cv2.INTER_AREA)
        
        # Compute horizontal gradient (difference between adjacent pixels)
        diff = resized[:, 1:] > resized[:, :-1]
        return diff.flatten()
    
    def compute_ncc(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Compute Normalized Cross-Correlation between two images.
        
        This is the most reliable measure for pixel-level similarity.
        Returns value in range [-1, 1], where 1 = identical.
        """
        # Convert to grayscale
        if len(img1.shape) == 3:
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        else:
            gray1 = img1
            
        if len(img2.shape) == 3:
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        else:
            gray2 = img2
        
        # Resize to same size for comparison (use smaller of the two)
        size = (min(256, min(gray1.shape[1], gray2.shape[1])),
                min(256, min(gray1.shape[0], gray2.shape[0])))
        
        r1 = cv2.resize(gray1, size, interpolation=cv2.INTER_AREA).astype(np.float32)
        r2 = cv2.resize(gray2, size, interpolation=cv2.INTER_AREA).astype(np.float32)
        
        # Normalize
        r1 = (r1 - np.mean(r1)) / (np.std(r1) + 1e-10)
        r2 = (r2 - np.mean(r2)) / (np.std(r2) + 1e-10)
        
        # Compute NCC
        ncc = np.mean(r1 * r2)
        
        # Clamp to [0, 1] range
        return max(0.0, min(1.0, (ncc + 1) / 2))
    
    def hash_similarity(self, hash1: np.ndarray, hash2: np.ndarray) -> float:
        """Compute similarity between two hashes (0-1)."""
        # Hamming distance
        diff = np.sum(hash1 != hash2)
        max_diff = len(hash1)
        return 1.0 - (diff / max_diff)
    
    def compute_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Compute overall similarity between two images.
        
        Uses geometric mean of NCC and hash similarity for strict comparison.
        Both metrics must agree for high similarity score.
        """
        # Compute hashes
        phash1 = self.compute_phash(img1)
        phash2 = self.compute_phash(img2)
        dhash1 = self.compute_dhash(img1)
        dhash2 = self.compute_dhash(img2)
        
        phash_sim = self.hash_similarity(phash1, phash2)
        dhash_sim = self.hash_similarity(dhash1, dhash2)
        
        # Take best hash similarity
        hash_sim = max(phash_sim, dhash_sim)
        
        # Only compute NCC if hashes suggest potential match
        # This is an optimization for large datasets
        if hash_sim < 0.6:
            return hash_sim * 0.5  # Quick rejection
        
        # Compute pixel-level similarity
        ncc_sim = self.compute_ncc(img1, img2)
        
        # Use geometric mean - both must be high for high score
        # This prevents false positives from hash collisions
        combined = np.sqrt(hash_sim * ncc_sim)
        
        return combined
    
    def find_duplicates(
        self,
        images: List[np.ndarray],
        paths: Optional[List[Path]] = None
    ) -> Tuple[List[int], List[Tuple[int, int, float]]]:
        """
        Find duplicate images in a list.
        
        For burst mode photos, uses windowed comparison to be efficient
        and only marks direct duplicates (not transitively connected images).
        
        Args:
            images: List of images as numpy arrays
            paths: Optional list of paths for logging
            
        Returns:
            Tuple of:
            - List of indices to KEEP (non-duplicates)
            - List of (idx1, idx2, similarity) for duplicate pairs found
        """
        n = len(images)
        if n <= 1:
            return list(range(n)), []
        
        logger.info(f"Scanning {n} images for duplicates (threshold={self.similarity_threshold:.2f})")
        self._update_progress(0, f"Scanning {n} images for duplicates...")
        
        # Track which images are marked as duplicates
        is_duplicate = [False] * n
        duplicate_pairs = []
        
        # For each image, find if it's a duplicate of an earlier image
        comparisons_done = 0
        total_comparisons = self._estimate_comparisons(n)
        
        for i in range(n):
            if is_duplicate[i]:
                continue  # Already marked as duplicate
            
            # Determine comparison range
            if self.comparison_window > 0:
                # Only compare with nearby images (efficient for burst photos)
                start = max(0, i - self.comparison_window)
                end = i  # Only compare with earlier images
            else:
                # Compare with all earlier images
                start = 0
                end = i
            
            for j in range(start, end):
                if is_duplicate[j]:
                    continue  # Skip already-marked duplicates
                
                similarity = self.compute_similarity(images[i], images[j])
                comparisons_done += 1
                
                if similarity >= self.similarity_threshold:
                    # Image i is a duplicate of image j (which came first)
                    is_duplicate[i] = True
                    duplicate_pairs.append((j, i, similarity))
                    
                    name_i = paths[i].name if paths else f"image_{i}"
                    name_j = paths[j].name if paths else f"image_{j}"
                    logger.info(f"Duplicate found: {name_i} ≈ {name_j} (similarity={similarity:.3f})")
                    break  # Found a duplicate, no need to check more
            
            # Update progress
            if i % 10 == 0:
                percent = min(99, int(100 * comparisons_done / max(1, total_comparisons)))
                self._update_progress(percent, f"Checked {i+1}/{n} images...")
        
        # Build list of indices to keep
        keep_indices = [i for i in range(n) if not is_duplicate[i]]
        
        n_removed = n - len(keep_indices)
        logger.info(f"Duplicate detection complete: keeping {len(keep_indices)}/{n} images "
                   f"(removed {n_removed} duplicates)")
        self._update_progress(100, f"Removed {n_removed} duplicate images")
        
        return keep_indices, duplicate_pairs
    
    def _estimate_comparisons(self, n: int) -> int:
        """Estimate total number of comparisons for progress reporting."""
        if self.comparison_window > 0:
            # Windowed: each image compared with up to window_size earlier images
            return n * min(self.comparison_window, n // 2)
        else:
            # All pairs: n*(n-1)/2
            return n * (n - 1) // 2
    
    def remove_duplicates(
        self,
        images: List[np.ndarray],
        paths: Optional[List[Path]] = None
    ) -> Tuple[List[np.ndarray], List[Path], List[int]]:
        """
        Remove duplicate images from a list.
        
        Args:
            images: List of images
            paths: Optional list of paths
            
        Returns:
            Tuple of:
            - Filtered list of images
            - Filtered list of paths (or empty if not provided)
            - Original indices of kept images
        """
        keep_indices, _ = self.find_duplicates(images, paths)
        
        filtered_images = [images[i] for i in keep_indices]
        filtered_paths = [paths[i] for i in keep_indices] if paths else []
        
        return filtered_images, filtered_paths, keep_indices

