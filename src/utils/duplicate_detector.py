"""
Duplicate and similar image detector for burst mode photos.

Uses multiple similarity metrics to identify truly duplicate/redundant images
while preserving sequential burst photos that cover different areas.

OPTIMIZED for large datasets (500+ images):
- Uses locality-sensitive hashing for O(n) approximate nearest neighbor
- Progressive hash refinement (coarse to fine)
- Early termination with bounded duplicate removal
"""

import cv2
import numpy as np
from typing import List, Tuple, Set, Optional, Callable, Dict
import logging
from pathlib import Path
import time

logger = logging.getLogger(__name__)

# Debug log path
_DEBUG_LOG = r'c:\Users\ryanf\OneDrive - University of Maryland\Desktop\Stitch2Stitch\Stitch2Stitch-1\.cursor\debug.log'

def _log_debug(hypothesis_id: str, location: str, message: str, data: dict):
    """Helper to write debug log."""
    try:
        import json
        with open(_DEBUG_LOG, 'a') as f:
            f.write(json.dumps({
                "hypothesisId": hypothesis_id,
                "location": location,
                "message": message,
                "data": data,
                "timestamp": time.time()
            }) + '\n')
    except:
        pass


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
        comparison_window: int = 0,  # Default to 0 (compare all pairs) for unsorted images
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
                              (0 = compare all pairs - REQUIRED for unsorted images)
                              Note: For unsorted images, window must be 0 to find all duplicates
            progress_callback: Optional callback(percent, message) for progress updates
        """
        self.similarity_threshold = similarity_threshold
        self.hash_size = hash_size
        self.comparison_window = comparison_window
        self.progress_callback = progress_callback
        
        logger.info(f"DuplicateDetector initialized: threshold={similarity_threshold:.2f}, "
                   f"hash_size={hash_size}, window={comparison_window}")
        _log_debug("DUP0", "duplicate_detector:__init__", "DuplicateDetector config", {
            "threshold": similarity_threshold, "window": comparison_window, "hash_size": hash_size
        })
    
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
        
        # Resize to same size for comparison
        # Use 512x512 for better accuracy (256 is too coarse for panorama images)
        # This prevents similar textures from looking identical
        size = (min(512, min(gray1.shape[1], gray2.shape[1])),
                min(512, min(gray1.shape[0], gray2.shape[0])))
        
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
        
        Uses weighted combination for burst photo detection.
        NCC is more reliable for nearly-identical burst frames.
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
        
        # Compute pixel-level similarity (most reliable for burst photos)
        ncc_sim = self.compute_ncc(img1, img2)
        
        # For burst photos, NCC is the most important metric
        # Use weighted average favoring NCC, not geometric mean
        # This catches near-identical frames even with slight hash differences
        combined = 0.3 * hash_sim + 0.7 * ncc_sim
        
        # If NCC alone is very high (>0.95), trust it even if hashes differ
        # This catches burst photos with minor exposure changes
        if ncc_sim > 0.95:
            combined = max(combined, ncc_sim * 0.98)
        
        return combined
    
    def find_duplicates(
        self,
        images: List[np.ndarray],
        paths: Optional[List[Path]] = None
    ) -> Tuple[List[int], List[Tuple[int, int, float]]]:
        """
        Find duplicate images in a list.
        
        OPTIMIZED: Uses LSH-style bucketing for O(n) complexity instead of O(n²).
        
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
        
        start_time = time.time()
        logger.info(f"Scanning {n} images for duplicates (threshold={self.similarity_threshold:.2f})")
        self._update_progress(0, f"Scanning {n} images for duplicates...")
        
        _log_debug("DUP1", "find_duplicates:start", "Starting duplicate scan", {
            "n_images": n, "threshold": self.similarity_threshold, "window": self.comparison_window
        })
        
        # Track which images are marked as duplicates
        is_duplicate = [False] * n
        duplicate_pairs = []
        
        # SAFEGUARD: Never remove more than 20% of images as duplicates
        # Panorama images MUST have overlap - true duplicates are rare
        max_duplicates_allowed = max(1, n // 5)
        
        # For panorama workflows, use DCT-based perceptual hashing
        # This is more distinctive than mean-based hashing
        self._update_progress(5, "Computing perceptual hashes...")
        hash_time = time.time()
        
        # Compute pHash (DCT-based) for each image - more distinctive for panoramas
        phashes = []
        dhashes = []
        for idx, img in enumerate(images):
            phashes.append(self.compute_phash(img))
            dhashes.append(self.compute_dhash(img))
            
            if idx % 100 == 0:
                self._update_progress(5 + int(20 * idx / n), f"Hashing: {idx+1}/{n}")
        
        hash_elapsed = time.time() - hash_time
        _log_debug("DUP2", "find_duplicates:hashes", "Perceptual hashes computed", {
            "n_images": n, "hash_time_sec": round(hash_elapsed, 2)
        })
        
        # For panorama images: use VERY strict threshold
        # True duplicates have nearly identical hashes (hamming dist < 5 in 64 bits = 92%+ similar)
        # Regular panorama overlap images have 60-80% hash similarity - NOT duplicates
        self._update_progress(30, "Finding near-identical images...")
        bucket_time = time.time()
        
        # Only find pairs with VERY high hash similarity (strict duplicate detection)
        # This prevents false positives from overlapping panorama images
        strict_hash_threshold = 0.92  # 92%+ hash match = likely true duplicate
        close_candidates = []
        
        # Use sorted hashes for efficient comparison (avoid full O(n²))
        # Group by first few bits of pHash to reduce comparisons
        hash_groups: Dict[int, List[int]] = {}
        for idx, ph in enumerate(phashes):
            # Use first 8 bits as group key (256 groups)
            group_key = int(np.packbits(ph[:8].astype(np.uint8))[0])
            if group_key not in hash_groups:
                hash_groups[group_key] = []
            hash_groups[group_key].append(idx)
        
        # Compare only within groups
        for group_indices in hash_groups.values():
            if len(group_indices) < 2:
                continue
            for i_pos, i in enumerate(group_indices):
                for j in group_indices[:i_pos]:
                    # Check both pHash and dHash
                    phash_sim = self.hash_similarity(phashes[i], phashes[j])
                    dhash_sim = self.hash_similarity(dhashes[i], dhashes[j])
                    
                    # BOTH hashes must be very similar for true duplicates
                    if phash_sim >= strict_hash_threshold and dhash_sim >= strict_hash_threshold:
                        combined_sim = (phash_sim + dhash_sim) / 2
                        close_candidates.append((i, j, combined_sim))
        
        # Sort by similarity (highest first)
        close_candidates.sort(key=lambda x: -x[2])
        
        bucket_elapsed = time.time() - bucket_time
        _log_debug("DUP3", "find_duplicates:candidates", "Candidate search complete", {
            "n_groups": len(hash_groups), "n_candidates": len(close_candidates),
            "search_time_sec": round(bucket_elapsed, 2),
            "largest_group": max(len(g) for g in hash_groups.values()) if hash_groups else 0,
            "strict_threshold": strict_hash_threshold
        })
        
        # Verify with NCC only if we have candidates
        self._update_progress(50, f"Verifying {len(close_candidates)} potential duplicates...")
        verify_time = time.time()
        duplicates_found = 0
        comparisons = 0
        max_comparisons = min(len(close_candidates), n)  # At most n comparisons
        
        for i, j, hash_sim in close_candidates:
            if is_duplicate[i] or is_duplicate[j]:
                continue
            
            if duplicates_found >= max_duplicates_allowed:
                _log_debug("DUP5", "find_duplicates:safeguard", "Safeguard triggered", {
                    "duplicates_found": duplicates_found, "max_allowed": max_duplicates_allowed
                })
                break
            
            if comparisons >= max_comparisons:
                _log_debug("DUP5", "find_duplicates:max_comparisons", "Max comparisons reached", {
                    "comparisons": comparisons, "max_comparisons": max_comparisons
                })
                break
            
            comparisons += 1
            
            # Fast NCC on smaller images (256x256 is enough for duplicate detection)
            ncc_sim = self._fast_ncc(images[i], images[j])
            
            # Combine hash and NCC similarity
            combined_sim = 0.3 * hash_sim + 0.7 * ncc_sim
            
            if combined_sim >= self.similarity_threshold:
                is_duplicate[i] = True
                duplicate_pairs.append((j, i, combined_sim))
                duplicates_found += 1
                
                name_i = paths[i].name if paths else f"image_{i}"
                name_j = paths[j].name if paths else f"image_{j}"
                logger.info(f"Duplicate found: {name_i} ≈ {name_j} (similarity={combined_sim:.3f})")
            
            if comparisons % 50 == 0:
                percent = 45 + int(50 * comparisons / max_comparisons)
                self._update_progress(percent, f"Verified {comparisons} pairs...")
        
        verify_elapsed = time.time() - verify_time
        total_elapsed = time.time() - start_time
        
        # Build list of indices to keep
        keep_indices = [i for i in range(n) if not is_duplicate[i]]
        n_removed = n - len(keep_indices)
        
        logger.info(f"Duplicate detection complete: keeping {len(keep_indices)}/{n} images "
                   f"(removed {n_removed} duplicates) in {total_elapsed:.1f}s")
        self._update_progress(100, f"Removed {n_removed} duplicate images")
        
        _log_debug("DUP6", "find_duplicates:complete", "Duplicate detection complete", {
            "n_kept": len(keep_indices), "n_removed": n_removed,
            "total_time_sec": round(total_elapsed, 2),
            "hash_time_sec": round(hash_elapsed, 2),
            "verify_time_sec": round(verify_elapsed, 2),
            "comparisons_made": comparisons
        })
        
        return keep_indices, duplicate_pairs
    
    def _fast_ncc(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Fast NCC on downsampled images (256x256)."""
        # Convert to grayscale
        if len(img1.shape) == 3:
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        else:
            gray1 = img1
        if len(img2.shape) == 3:
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        else:
            gray2 = img2
        
        # Resize to 256x256 (fast but sufficient for duplicate detection)
        size = (256, 256)
        r1 = cv2.resize(gray1, size, interpolation=cv2.INTER_AREA).astype(np.float32)
        r2 = cv2.resize(gray2, size, interpolation=cv2.INTER_AREA).astype(np.float32)
        
        # Normalize and compute NCC
        r1 = (r1 - np.mean(r1)) / (np.std(r1) + 1e-10)
        r2 = (r2 - np.mean(r2)) / (np.std(r2) + 1e-10)
        ncc = np.mean(r1 * r2)
        
        return max(0.0, min(1.0, (ncc + 1) / 2))
    
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

