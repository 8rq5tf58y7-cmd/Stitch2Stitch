"""
Duplicate and similar image detector for burst mode photos.

Uses imagehash library with SQLite caching for fast, persistent duplicate detection.
Optimized for large datasets (1000+ images) with O(n) complexity via hash bucketing.

FEATURES:
- Perceptual hashing (pHash, dHash, wHash) via imagehash library
- SQLite cache for instant re-runs on same image sets
- Hash bucketing for O(n) approximate nearest neighbor
- Safeguards to prevent over-deletion of panorama overlap images
"""

import cv2
import numpy as np
from typing import List, Tuple, Set, Optional, Callable, Dict
import logging
from pathlib import Path
import time
import sqlite3
import hashlib
import os

logger = logging.getLogger(__name__)


class HashCache:
    """
    SQLite-based cache for image hashes.

    Stores computed hashes persistently so subsequent runs are instant.
    Cache key is based on file path + modification time + file size.
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize hash cache.

        Args:
            cache_dir: Directory to store cache database.
                       If None, uses a default location in user's temp dir.
        """
        if cache_dir is None:
            import tempfile
            cache_dir = Path(tempfile.gettempdir()) / "stitch2stitch_cache"

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "image_hashes.db"
        self._init_db()

    def _init_db(self):
        """Initialize SQLite database with schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS hashes (
                    file_key TEXT PRIMARY KEY,
                    file_path TEXT,
                    phash TEXT,
                    dhash TEXT,
                    whash TEXT,
                    ahash TEXT,
                    created_at REAL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_phash ON hashes(phash)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_dhash ON hashes(dhash)")
            conn.commit()

    def _get_file_key(self, path: Path) -> str:
        """
        Generate a unique key for a file based on path + mtime + size.
        This ensures we recompute hashes if the file changes.
        """
        try:
            stat = path.stat()
            key_str = f"{path.absolute()}:{stat.st_mtime}:{stat.st_size}"
            return hashlib.md5(key_str.encode()).hexdigest()
        except:
            return hashlib.md5(str(path.absolute()).encode()).hexdigest()

    def get(self, path: Path) -> Optional[Dict[str, str]]:
        """
        Get cached hashes for a file.

        Returns:
            Dict with 'phash', 'dhash', 'whash', 'ahash' keys, or None if not cached.
        """
        file_key = self._get_file_key(path)
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT phash, dhash, whash, ahash FROM hashes WHERE file_key = ?",
                    (file_key,)
                )
                row = cursor.fetchone()
                if row:
                    return {
                        'phash': row[0],
                        'dhash': row[1],
                        'whash': row[2],
                        'ahash': row[3]
                    }
        except Exception as e:
            logger.debug(f"Cache read error: {e}")
        return None

    def set(self, path: Path, hashes: Dict[str, str]):
        """
        Store hashes for a file in the cache.

        Args:
            path: File path
            hashes: Dict with 'phash', 'dhash', 'whash', 'ahash' keys
        """
        file_key = self._get_file_key(path)
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO hashes
                    (file_key, file_path, phash, dhash, whash, ahash, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    file_key,
                    str(path.absolute()),
                    hashes.get('phash', ''),
                    hashes.get('dhash', ''),
                    hashes.get('whash', ''),
                    hashes.get('ahash', ''),
                    time.time()
                ))
                conn.commit()
        except Exception as e:
            logger.debug(f"Cache write error: {e}")

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM hashes")
                count = cursor.fetchone()[0]
                return {
                    'entries': count,
                    'db_path': str(self.db_path),
                    'db_size_mb': self.db_path.stat().st_size / (1024 * 1024) if self.db_path.exists() else 0,
                    'caching_enabled': True
                }
        except:
            return {'entries': 0, 'db_path': str(self.db_path), 'db_size_mb': 0, 'caching_enabled': True}


class DuplicateDetector:
    """
    Detect and remove duplicate/highly similar images using imagehash library.

    Features:
    - Multiple hash types (pHash, dHash, wHash, aHash) for robust detection
    - SQLite caching for instant re-runs
    - Hash bucketing for O(n) complexity
    - Safeguards for panorama workflows
    """

    def __init__(
        self,
        similarity_threshold: float = 0.92,
        hash_size: int = 16,
        comparison_window: int = 0,
        progress_callback: Optional[Callable] = None,
        cache_dir: Optional[Path] = None
    ):
        """
        Initialize duplicate detector.

        Args:
            similarity_threshold: Minimum similarity (0-1) to consider images duplicates.
                                  0.92 is strict - only near-identical images.
                                  0.85 is moderate - similar frames merged.
            hash_size: Size of perceptual hash (8, 16, or 32)
            comparison_window: For sorted images, only compare within this window
                              (0 = compare all pairs)
            progress_callback: Optional callback(percent, message) for progress
            cache_dir: Directory for SQLite hash cache (None = temp dir)
        """
        self.similarity_threshold = similarity_threshold
        self.hash_size = hash_size
        self.comparison_window = comparison_window
        self.progress_callback = progress_callback
        self.cache = HashCache(cache_dir)

        # Check if imagehash is available, try to install if not
        self._has_imagehash = False
        self._imagehash_status = "not_checked"
        try:
            import imagehash
            from PIL import Image
            self._has_imagehash = True
            self._imagehash_status = "available"
            logger.info("Using imagehash library for duplicate detection (caching enabled)")
        except ImportError:
            logger.warning("imagehash not installed, attempting to install...")
            self._imagehash_status = "installing"
            try:
                import subprocess
                import sys
                import importlib

                # Try different install strategies for different environments
                install_commands = [
                    # Standard install (works in venvs and Windows)
                    [sys.executable, "-m", "pip", "install", "imagehash"],
                    # User install (works on externally-managed Linux systems like WSL Ubuntu)
                    [sys.executable, "-m", "pip", "install", "--user", "imagehash"],
                    # Break system packages (last resort for WSL)
                    [sys.executable, "-m", "pip", "install", "--break-system-packages", "imagehash"],
                ]

                installed = False
                last_error = ""

                for cmd in install_commands:
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=120
                    )
                    if result.returncode == 0:
                        installed = True
                        break
                    last_error = result.stderr

                if installed:
                    # Clear import caches so newly installed packages are found
                    importlib.invalidate_caches()
                    try:
                        imagehash = importlib.import_module('imagehash')
                        Image = importlib.import_module('PIL.Image')
                        self._has_imagehash = True
                        self._imagehash_status = "installed"
                        logger.info("Successfully installed imagehash (caching enabled)")
                    except ImportError as ie:
                        # Package installed but not importable until restart
                        self._imagehash_status = "installed_needs_restart"
                        logger.warning(f"imagehash installed but requires restart to use: {ie}")
                        logger.warning("Using fallback OpenCV-based hashing for this session")
                else:
                    self._imagehash_status = f"install_failed: {last_error[:100]}"
                    logger.warning(f"Failed to install imagehash: {last_error}")
                    logger.warning("Using fallback OpenCV-based hashing (slower, NO caching)")
            except Exception as e:
                self._imagehash_status = f"install_error: {e}"
                logger.warning(f"Could not install imagehash: {e}")
                logger.warning("Using fallback OpenCV-based hashing (slower, NO caching)")

        logger.info(f"DuplicateDetector initialized: threshold={similarity_threshold:.2f}, "
                   f"hash_size={hash_size}, cached={self._has_imagehash}")

    def _update_progress(self, percent: float, message: str):
        """Update progress if callback is set."""
        if self.progress_callback:
            self.progress_callback(int(percent), message)

    def _compute_hashes_imagehash(self, path: Path) -> Dict[str, str]:
        """Compute hashes using imagehash library."""
        import imagehash
        from PIL import Image

        # Check cache first
        cached = self.cache.get(path)
        if cached:
            return cached

        # Load image with PIL
        try:
            img = Image.open(path)
            if img.mode == 'RGBA':
                # Convert RGBA to RGB for hashing
                img = img.convert('RGB')

            # Compute multiple hash types
            hashes = {
                'phash': str(imagehash.phash(img, hash_size=self.hash_size)),
                'dhash': str(imagehash.dhash(img, hash_size=self.hash_size)),
                'whash': str(imagehash.whash(img, hash_size=self.hash_size)),
                'ahash': str(imagehash.average_hash(img, hash_size=self.hash_size))
            }

            # Cache the results
            self.cache.set(path, hashes)

            return hashes

        except Exception as e:
            logger.warning(f"Failed to hash {path}: {e}")
            return {'phash': '', 'dhash': '', 'whash': '', 'ahash': ''}

    def _compute_hashes_fallback(self, image: np.ndarray) -> Dict[str, str]:
        """Fallback hash computation using OpenCV (no caching)."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Simple DCT-based hash
        resized = cv2.resize(gray, (self.hash_size, self.hash_size), interpolation=cv2.INTER_AREA)
        dct = cv2.dct(np.float32(resized))
        dct_low = dct[:8, :8]
        median = np.median(dct_low)
        hash_bits = (dct_low > median).flatten()
        phash = ''.join(['1' if b else '0' for b in hash_bits])

        # Difference hash
        resized = cv2.resize(gray, (self.hash_size + 1, self.hash_size), interpolation=cv2.INTER_AREA)
        diff = resized[:, 1:] > resized[:, :-1]
        dhash = ''.join(['1' if b else '0' for b in diff.flatten()[:64]])

        return {'phash': phash, 'dhash': dhash, 'whash': '', 'ahash': ''}

    def _hash_to_int(self, hash_str: str) -> int:
        """Convert hex hash string to integer for fast comparison."""
        if not hash_str:
            return 0
        try:
            return int(hash_str, 16)
        except ValueError:
            # Binary string
            try:
                return int(hash_str, 2)
            except:
                return 0

    def _hamming_distance(self, hash1: int, hash2: int) -> int:
        """Compute Hamming distance between two hash integers."""
        xor = hash1 ^ hash2
        return bin(xor).count('1')

    def _hash_similarity(self, hash1: str, hash2: str) -> float:
        """Compute similarity between two hash strings (0-1)."""
        if not hash1 or not hash2:
            return 0.0

        h1 = self._hash_to_int(hash1)
        h2 = self._hash_to_int(hash2)

        # Hamming distance
        distance = self._hamming_distance(h1, h2)

        # Max bits based on hash size
        max_bits = self.hash_size * self.hash_size

        return 1.0 - (distance / max_bits)

    def find_duplicates(
        self,
        images: List[np.ndarray],
        paths: Optional[List[Path]] = None
    ) -> Tuple[List[int], List[Tuple[int, int, float]]]:
        """
        Find duplicate images in a list.

        Args:
            images: List of images as numpy arrays
            paths: Optional list of paths (REQUIRED for caching)

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

        # Track duplicates
        is_duplicate = [False] * n
        duplicate_pairs = []

        # Safeguard: Never remove more than 20% as duplicates
        max_duplicates = max(1, n // 5)

        # Step 1: Compute hashes
        self._update_progress(5, "Computing image hashes...")
        hash_start = time.time()

        all_hashes = []
        cache_hits = 0

        for idx in range(n):
            if paths and self._has_imagehash:
                # Use imagehash with caching
                cached = self.cache.get(paths[idx])
                if cached:
                    cache_hits += 1
                    hashes = cached
                else:
                    hashes = self._compute_hashes_imagehash(paths[idx])
            else:
                # Fallback without caching - load image if None
                img = images[idx]
                if img is None and paths:
                    # Load image from path for fallback
                    img = cv2.imread(str(paths[idx]))
                    if img is not None:
                        # Resize to thumbnail for faster hashing
                        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
                if img is not None:
                    hashes = self._compute_hashes_fallback(img)
                else:
                    # Can't load image, use empty hash
                    hashes = {'phash': '', 'dhash': '', 'whash': '', 'ahash': ''}

            all_hashes.append(hashes)

            if idx % 100 == 0:
                pct = 5 + int(25 * idx / n)
                self._update_progress(pct, f"Hashing: {idx+1}/{n} (cache hits: {cache_hits})")

        hash_time = time.time() - hash_start
        logger.info(f"Hashing complete in {hash_time:.1f}s (cache hits: {cache_hits}/{n})")

        # Step 2: Build hash buckets for fast comparison
        self._update_progress(35, "Building hash index...")

        # Group by first 2 hex chars of phash (256 buckets)
        buckets: Dict[str, List[int]] = {}
        for idx, hashes in enumerate(all_hashes):
            phash = hashes.get('phash', '')
            if len(phash) >= 2:
                bucket_key = phash[:2]
            else:
                bucket_key = '00'

            if bucket_key not in buckets:
                buckets[bucket_key] = []
            buckets[bucket_key].append(idx)

        # Step 3: Compare within buckets
        self._update_progress(40, "Finding duplicates...")
        compare_start = time.time()

        candidates = []
        for bucket_key, indices in buckets.items():
            if len(indices) < 2:
                continue

            for i_pos, i in enumerate(indices):
                for j in indices[:i_pos]:
                    # Quick check with phash
                    phash_sim = self._hash_similarity(
                        all_hashes[i].get('phash', ''),
                        all_hashes[j].get('phash', '')
                    )

                    if phash_sim >= 0.85:  # Loose threshold for candidates
                        # Check other hashes
                        dhash_sim = self._hash_similarity(
                            all_hashes[i].get('dhash', ''),
                            all_hashes[j].get('dhash', '')
                        )
                        whash_sim = self._hash_similarity(
                            all_hashes[i].get('whash', ''),
                            all_hashes[j].get('whash', '')
                        )

                        # Combined similarity (average of available hashes)
                        sims = [phash_sim, dhash_sim]
                        if whash_sim > 0:
                            sims.append(whash_sim)
                        combined = sum(sims) / len(sims)

                        if combined >= self.similarity_threshold:
                            candidates.append((i, j, combined))

        # Sort by similarity (highest first)
        candidates.sort(key=lambda x: -x[2])

        compare_time = time.time() - compare_start
        logger.info(f"Found {len(candidates)} candidate pairs in {compare_time:.1f}s")

        # Step 4: Mark duplicates (keep lower index, remove higher)
        self._update_progress(80, f"Processing {len(candidates)} duplicate candidates...")

        duplicates_found = 0
        for i, j, similarity in candidates:
            if is_duplicate[i] or is_duplicate[j]:
                continue

            if duplicates_found >= max_duplicates:
                logger.warning(f"Safeguard: Stopped at {max_duplicates} duplicates (20% limit)")
                break

            # Mark higher index as duplicate (keep lower/earlier image)
            dup_idx = max(i, j)
            keep_idx = min(i, j)

            is_duplicate[dup_idx] = True
            duplicate_pairs.append((keep_idx, dup_idx, similarity))
            duplicates_found += 1

            if paths:
                logger.info(f"Duplicate: {paths[dup_idx].name} ≈ {paths[keep_idx].name} "
                           f"(similarity={similarity:.3f})")

        # Build keep list
        keep_indices = [i for i in range(n) if not is_duplicate[i]]

        total_time = time.time() - start_time
        logger.info(f"Duplicate detection complete: keeping {len(keep_indices)}/{n} images "
                   f"(removed {duplicates_found}) in {total_time:.1f}s")

        self._update_progress(100, f"Removed {duplicates_found} duplicates")

        # Log cache stats
        stats = self.cache.get_stats()
        logger.info(f"Hash cache: {stats['entries']} entries, {stats['db_size_mb']:.2f} MB")

        return keep_indices, duplicate_pairs

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


def find_duplicates_cached(
    paths: List[Path],
    similarity_threshold: float = 0.92,
    cache_dir: Optional[Path] = None,
    progress_callback: Optional[Callable] = None
) -> Tuple[List[int], List[Tuple[int, int, float]]]:
    """
    Convenience function to find duplicates using only file paths (no image loading).

    This is the fastest method for large image sets as it:
    1. Uses cached hashes when available
    2. Loads images only for uncached files
    3. Hashes directly from disk (no numpy arrays)

    Args:
        paths: List of image file paths
        similarity_threshold: Minimum similarity (0-1) to consider duplicates
        cache_dir: Directory for hash cache (None = temp dir)
        progress_callback: Optional callback(percent, message)

    Returns:
        Tuple of (keep_indices, duplicate_pairs)
    """
    detector = DuplicateDetector(
        similarity_threshold=similarity_threshold,
        cache_dir=cache_dir,
        progress_callback=progress_callback
    )

    # We don't need to load images if using imagehash - it reads from disk
    # Just pass empty images list but with paths
    n = len(paths)
    dummy_images = [None] * n  # Won't be used if imagehash is available

    return detector.find_duplicates(dummy_images, paths)
