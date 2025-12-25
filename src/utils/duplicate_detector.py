"""
Duplicate and Similar Image Detection

Detects and removes duplicate or highly similar images before processing
to reduce memory usage and improve stitching quality.

Methods:
1. Perceptual Hash (pHash) - Fast, good for exact/near duplicates
2. Difference Hash (dHash) - Very fast, orientation sensitive
3. Histogram Comparison - Good for exposure variations
4. Combined scoring - Uses multiple methods for accuracy
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Callable
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class DuplicateDetector:
    """
    Detects duplicate and similar images using perceptual hashing
    and histogram comparison.
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.95,
        hash_size: int = 16,
        use_histogram: bool = True,
        use_phash: bool = True
    ):
        """
        Initialize duplicate detector.
        
        Args:
            similarity_threshold: Similarity score (0-1) above which images 
                                  are considered duplicates (default 0.95 = 95% similar)
            hash_size: Size of perceptual hash (larger = more sensitive)
            use_histogram: Include histogram comparison in scoring
            use_phash: Include perceptual hash in scoring
        """
        self.similarity_threshold = similarity_threshold
        self.hash_size = hash_size
        self.use_histogram = use_histogram
        self.use_phash = use_phash
        
        logger.info(f"DuplicateDetector initialized (threshold={similarity_threshold}, hash_size={hash_size})")
    
    def compute_phash(self, image: np.ndarray) -> np.ndarray:
        """
        Compute perceptual hash of an image.
        
        Uses DCT-based perceptual hashing which is robust to:
        - Scaling
        - Minor rotations
        - Compression artifacts
        - Small color changes
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            Binary hash as numpy array
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Resize to hash_size + 1 (for DCT)
        resized = cv2.resize(gray, (self.hash_size + 1, self.hash_size + 1), 
                            interpolation=cv2.INTER_AREA)
        
        # Convert to float for DCT
        float_img = resized.astype(np.float32)
        
        # Apply DCT
        dct = cv2.dct(float_img)
        
        # Keep top-left corner (low frequencies)
        dct_low = dct[:self.hash_size, :self.hash_size]
        
        # Compute median and create binary hash
        median = np.median(dct_low)
        hash_bits = (dct_low > median).flatten()
        
        return hash_bits
    
    def compute_dhash(self, image: np.ndarray) -> np.ndarray:
        """
        Compute difference hash of an image.
        
        Very fast, compares adjacent pixels.
        Good for detecting exact duplicates and minor edits.
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            Binary hash as numpy array
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Resize to hash_size + 1 x hash_size
        resized = cv2.resize(gray, (self.hash_size + 1, self.hash_size), 
                            interpolation=cv2.INTER_AREA)
        
        # Compute horizontal gradient (difference between adjacent pixels)
        diff = resized[:, 1:] > resized[:, :-1]
        
        return diff.flatten()
    
    def compute_histogram(self, image: np.ndarray) -> np.ndarray:
        """
        Compute normalized color histogram.
        
        Args:
            image: Input image (BGR)
            
        Returns:
            Normalized histogram
        """
        if len(image.shape) == 2:
            # Grayscale
            hist = cv2.calcHist([image], [0], None, [64], [0, 256])
        else:
            # Color - compute histogram for each channel
            hist_b = cv2.calcHist([image], [0], None, [64], [0, 256])
            hist_g = cv2.calcHist([image], [1], None, [64], [0, 256])
            hist_r = cv2.calcHist([image], [2], None, [64], [0, 256])
            hist = np.concatenate([hist_b, hist_g, hist_r])
        
        # Normalize
        hist = cv2.normalize(hist, hist).flatten()
        return hist
    
    def hash_similarity(self, hash1: np.ndarray, hash2: np.ndarray) -> float:
        """
        Compute similarity between two hashes (0-1, higher = more similar).
        
        Args:
            hash1: First hash
            hash2: Second hash
            
        Returns:
            Similarity score (0-1)
        """
        if len(hash1) != len(hash2):
            return 0.0
        
        # Hamming distance (number of different bits)
        hamming_dist = np.sum(hash1 != hash2)
        
        # Convert to similarity (0-1)
        similarity = 1 - (hamming_dist / len(hash1))
        return similarity
    
    def histogram_similarity(self, hist1: np.ndarray, hist2: np.ndarray) -> float:
        """
        Compute histogram similarity using correlation.
        
        Args:
            hist1: First histogram
            hist2: Second histogram
            
        Returns:
            Similarity score (0-1)
        """
        # Use correlation method (returns -1 to 1)
        correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        # Normalize to 0-1
        similarity = (correlation + 1) / 2
        return similarity
    
    def compute_pixel_similarity(self, image1: np.ndarray, image2: np.ndarray) -> float:
        """
        Compute pixel-level structural similarity.
        
        This catches cases where hash/histogram methods fail by doing
        direct image comparison after alignment.
        
        Args:
            image1, image2: Input images (should be same size or will be resized)
            
        Returns:
            Structural similarity (0-1)
        """
        # Ensure same size
        target_size = (64, 64)  # Small size for speed
        
        if len(image1.shape) == 3:
            gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        else:
            gray1 = image1
            
        if len(image2.shape) == 3:
            gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        else:
            gray2 = image2
        
        # Resize to same small size
        small1 = cv2.resize(gray1, target_size, interpolation=cv2.INTER_AREA)
        small2 = cv2.resize(gray2, target_size, interpolation=cv2.INTER_AREA)
        
        # Normalize
        small1 = small1.astype(np.float32) / 255.0
        small2 = small2.astype(np.float32) / 255.0
        
        # Compute normalized cross-correlation (NCC)
        mean1 = np.mean(small1)
        mean2 = np.mean(small2)
        std1 = np.std(small1)
        std2 = np.std(small2)
        
        if std1 < 0.02 or std2 < 0.02:
            # Low-variance image (nearly solid color or very flat)
            # Use direct pixel difference instead of NCC
            mse = np.mean((small1 - small2) ** 2)
            # MSE of 0 = identical, MSE of 1 = completely different
            # Typical different images have MSE ~0.1-0.3
            similarity = 1.0 - np.sqrt(mse) * 3  # Scale so MSE=0.11 gives ~0
            return float(np.clip(similarity, 0, 1))
        
        ncc = np.mean((small1 - mean1) * (small2 - mean2)) / (std1 * std2)
        
        # NCC ranges from -1 to 1, convert to 0-1
        similarity = (ncc + 1) / 2
        
        return float(np.clip(similarity, 0, 1))
    
    def compute_similarity(
        self, 
        image1: np.ndarray, 
        image2: np.ndarray,
        phash1: Optional[np.ndarray] = None,
        phash2: Optional[np.ndarray] = None,
        dhash1: Optional[np.ndarray] = None,
        dhash2: Optional[np.ndarray] = None,
        hist1: Optional[np.ndarray] = None,
        hist2: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute overall similarity between two images.
        
        Uses a STRICT approach: images must be similar in MULTIPLE metrics
        to be considered duplicates. This prevents false positives from
        images that happen to have similar histograms or patterns.
        
        Args:
            image1, image2: Input images
            phash1, phash2: Pre-computed perceptual hashes (optional)
            dhash1, dhash2: Pre-computed difference hashes (optional)
            hist1, hist2: Pre-computed histograms (optional)
            
        Returns:
            Combined similarity score (0-1)
        """
        scores = []
        
        if self.use_phash:
            # Perceptual hash similarity (DCT-based, good for structure)
            if phash1 is None:
                phash1 = self.compute_phash(image1)
            if phash2 is None:
                phash2 = self.compute_phash(image2)
            
            phash_sim = self.hash_similarity(phash1, phash2)
            scores.append(('phash', phash_sim))
            
            # Difference hash similarity (gradient-based)
            if dhash1 is None:
                dhash1 = self.compute_dhash(image1)
            if dhash2 is None:
                dhash2 = self.compute_dhash(image2)
            
            dhash_sim = self.hash_similarity(dhash1, dhash2)
            scores.append(('dhash', dhash_sim))
        
        # Always compute pixel similarity - this is the most reliable
        pixel_sim = self.compute_pixel_similarity(image1, image2)
        scores.append(('pixel', pixel_sim))
        
        if self.use_histogram:
            # Histogram similarity (only as supplementary check)
            if hist1 is None:
                hist1 = self.compute_histogram(image1)
            if hist2 is None:
                hist2 = self.compute_histogram(image2)
            
            hist_sim = self.histogram_similarity(hist1, hist2)
            scores.append(('hist', hist_sim))
        
        if not scores:
            return 0.0
        
        # BALANCED DUPLICATE DETECTION:
        # Pixel similarity is the most reliable metric - weight it heavily
        # Hashes are good for speed but can have false positives/negatives
        
        score_dict = {name: s for name, s in scores}
        
        pixel_sim = score_dict.get('pixel', 0)
        phash_sim = score_dict.get('phash', 0)
        dhash_sim = score_dict.get('dhash', 0)
        hist_sim = score_dict.get('hist', 0)
        
        # Strategy: Use geometric mean of pixel + best hash
        # This requires BOTH to be high to get a high score
        # But allows one to compensate slightly for the other
        
        if self.use_phash:
            # Take the better of the two hashes (they measure different things)
            best_hash = max(phash_sim, dhash_sim)
            
            # Geometric mean of pixel and best hash - both must be high
            structural_score = np.sqrt(pixel_sim * best_hash)
        else:
            structural_score = pixel_sim
        
        # Histogram only contributes if structural similarity is already decent
        # This prevents histogram from inflating scores of clearly different images
        if self.use_histogram and structural_score > 0.5:
            # Small histogram contribution (5%)
            combined_score = structural_score * 0.95 + hist_sim * 0.05
        else:
            combined_score = structural_score
        
        return float(combined_score)
    
    def find_duplicates(
        self,
        images: List[Tuple[Path, np.ndarray]],
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        max_comparison_distance: Optional[int] = None,
        fast_mode: bool = True
    ) -> Tuple[List[int], List[Tuple[int, int, float]]]:
        """
        Find duplicate/similar images in a list.
        
        Args:
            images: List of (path, image_array) tuples
            progress_callback: Optional callback(current, total, message)
            max_comparison_distance: Only compare images within this index distance.
                For burst photos, duplicates are usually adjacent. Set to 50-100
                to skip comparing image 1 with image 500 (they can't be duplicates).
                None = compare all pairs (slow for large sets).
            fast_mode: Use fast hash-only pre-filtering before full comparison.
                Reduces comparisons by ~90% for non-duplicate images.
            
        Returns:
            Tuple of:
            - List of indices to keep (unique images)
            - List of (idx1, idx2, similarity) for detected duplicates
        """
        n = len(images)
        if n <= 1:
            return list(range(n)), []
        
        # Auto-enable optimizations for large sets
        if n > 200 and max_comparison_distance is None:
            # For burst photos, duplicates are usually within 20-50 frames
            max_comparison_distance = 50
            logger.info(f"Large image set ({n} images): limiting comparison distance to {max_comparison_distance}")
        
        if n > 100 and fast_mode:
            logger.info(f"Fast mode enabled: using hash pre-filtering")
        
        logger.info(f"Scanning {n} images for duplicates (threshold={self.similarity_threshold})")
        
        # Pre-compute hashes and histograms for all images
        if progress_callback:
            progress_callback(0, n, "Computing image signatures...")
        
        signatures = []
        for i, (path, img) in enumerate(images):
            if progress_callback:
                progress_callback(i, n, f"Computing signature: {path.name}")
            
            # Resize for faster processing
            h, w = img.shape[:2]
            if max(h, w) > 512:
                scale = 512 / max(h, w)
                small = cv2.resize(img, (int(w * scale), int(h * scale)), 
                                  interpolation=cv2.INTER_AREA)
            else:
                small = img
            
            sig = {
                'phash': self.compute_phash(small) if self.use_phash else None,
                'dhash': self.compute_dhash(small) if self.use_phash else None,
                'hist': self.compute_histogram(small) if self.use_histogram else None
            }
            signatures.append(sig)
        
        # Compare pairs with optimizations for large sets
        duplicates = []  # List of (i, j, similarity) for pairs above threshold
        
        # Calculate actual pairs to compare based on max_comparison_distance
        if max_comparison_distance:
            # Only compare images within the distance window
            total_pairs = sum(min(max_comparison_distance, n - i - 1) for i in range(n - 1))
        else:
            total_pairs = n * (n - 1) // 2
        
        pair_count = 0
        skipped_by_distance = 0
        skipped_by_hash = 0
        
        if progress_callback:
            progress_callback(0, total_pairs, f"Comparing images (optimized: {total_pairs} pairs)...")
        
        # Hash similarity threshold for fast pre-filtering
        # If hashes are very different, skip expensive pixel comparison
        hash_prefilter_threshold = 0.5  # Must have at least 50% hash similarity
        
        for i in range(n):
            # Determine comparison range based on max_comparison_distance
            if max_comparison_distance:
                j_end = min(i + 1 + max_comparison_distance, n)
            else:
                j_end = n
            
            for j in range(i + 1, j_end):
                pair_count += 1
                
                if progress_callback and pair_count % 100 == 0:
                    progress_callback(pair_count, total_pairs, 
                                    f"Comparing: {pair_count}/{total_pairs} pairs")
                
                # FAST MODE: Pre-filter using hash similarity only
                if fast_mode and self.use_phash:
                    phash_sim = self.hash_similarity(signatures[i]['phash'], signatures[j]['phash'])
                    dhash_sim = self.hash_similarity(signatures[i]['dhash'], signatures[j]['dhash'])
                    max_hash_sim = max(phash_sim, dhash_sim)
                    
                    # Skip if hashes are too different (can't be duplicates)
                    if max_hash_sim < hash_prefilter_threshold:
                        skipped_by_hash += 1
                        continue
                
                # Full similarity computation
                similarity = self.compute_similarity(
                    images[i][1], images[j][1],
                    phash1=signatures[i]['phash'], phash2=signatures[j]['phash'],
                    dhash1=signatures[i]['dhash'], dhash2=signatures[j]['dhash'],
                    hist1=signatures[i]['hist'], hist2=signatures[j]['hist']
                )
                
                # Log high similarities for debugging
                if similarity >= 0.85:
                    logger.debug(f"High similarity: {images[i][0].name} <-> {images[j][0].name} = {similarity:.3f}")
                
                if similarity >= self.similarity_threshold:
                    duplicates.append((i, j, similarity))
                    logger.info(f"DUPLICATE PAIR: {images[i][0].name} <-> {images[j][0].name} ({similarity:.1%} >= {self.similarity_threshold:.1%} threshold)")
        
        # Track skipped comparisons
        if max_comparison_distance:
            full_pairs = n * (n - 1) // 2
            skipped_by_distance = full_pairs - total_pairs
        
        if skipped_by_distance > 0 or skipped_by_hash > 0:
            logger.info(f"Optimization stats: skipped {skipped_by_distance} by distance, {skipped_by_hash} by hash pre-filter")
        
        # Determine which images to keep
        # IMPORTANT: Use DIRECT duplicate pairs only, NOT transitive groups!
        # This prevents burst mode photos from being chained together.
        #
        # Strategy: For each duplicate pair, remove the lower quality one.
        # If an image is in multiple pairs, it may be removed if it's always
        # the lower quality option.
        
        to_remove = set()
        
        # Sort duplicates by similarity (highest first) to prioritize clear duplicates
        duplicates.sort(key=lambda x: x[2], reverse=True)
        
        for i, j, similarity in duplicates:
            # Skip if either image is already marked for removal
            if i in to_remove or j in to_remove:
                continue
            
            # Compare quality (resolution) to decide which to keep
            size_i = images[i][1].shape[0] * images[i][1].shape[1]
            size_j = images[j][1].shape[0] * images[j][1].shape[1]
            
            if size_i >= size_j:
                # Keep i, remove j
                to_remove.add(j)
                logger.debug(f"Removing {images[j][0].name} (duplicate of {images[i][0].name})")
            else:
                # Keep j, remove i
                to_remove.add(i)
                logger.debug(f"Removing {images[i][0].name} (duplicate of {images[j][0].name})")
        
        # Build list of indices to keep
        to_keep = [i for i in range(n) if i not in to_remove]
        
        logger.info(f"Duplicate detection complete: {len(to_keep)} UNIQUE images kept, {len(to_remove)} duplicates removed")
        
        # Log what's being kept vs removed
        for idx in to_keep:
            logger.debug(f"  KEEPING: {images[idx][0].name}")
        for idx in to_remove:
            logger.debug(f"  REMOVING: {images[idx][0].name}")
        
        if progress_callback:
            progress_callback(total_pairs, total_pairs, 
                            f"Keeping {len(to_keep)} unique images (removed {len(to_remove)} duplicates)")
        
        return to_keep, duplicates


def remove_duplicates_from_paths(
    paths: List[Path],
    similarity_threshold: float = 0.95,
    thumbnail_size: int = 256,
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> Tuple[List[Path], int]:
    """
    Convenience function to remove duplicates from a list of image paths.
    
    Args:
        paths: List of image paths
        similarity_threshold: Similarity threshold (0-1)
        thumbnail_size: Size to resize images for comparison
        progress_callback: Optional progress callback
        
    Returns:
        Tuple of (filtered_paths, num_removed)
    """
    if len(paths) <= 1:
        return paths, 0
    
    detector = DuplicateDetector(similarity_threshold=similarity_threshold)
    
    # Load thumbnails
    images = []
    for i, path in enumerate(paths):
        if progress_callback:
            progress_callback(i, len(paths), f"Loading: {path.name}")
        
        img = cv2.imread(str(path))
        if img is not None:
            # Resize for faster comparison
            h, w = img.shape[:2]
            if max(h, w) > thumbnail_size:
                scale = thumbnail_size / max(h, w)
                img = cv2.resize(img, (int(w * scale), int(h * scale)), 
                               interpolation=cv2.INTER_AREA)
            images.append((path, img))
    
    # Find duplicates
    keep_indices, duplicates = detector.find_duplicates(images, progress_callback)
    
    # Return filtered paths
    filtered_paths = [images[i][0] for i in keep_indices]
    num_removed = len(paths) - len(filtered_paths)
    
    return filtered_paths, num_removed

