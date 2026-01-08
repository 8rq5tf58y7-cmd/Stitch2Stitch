"""
COLMAP Per-Image Feature Cache Manager

Provides incremental caching for COLMAP feature extraction and matching.
Dramatically improves performance when adding new images to existing sets.

Performance targets:
- Adding 10 images to 100 cached: ~30s (vs 5min full extraction)
- Cache lookup overhead: <1s for 1000 images
- 100% backward compatible with global cache
"""

import hashlib
import json
import logging
import shutil
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class COLMAPCacheManager:
    """
    Manages per-image feature and match caching for COLMAP.

    Provides incremental caching with backward compatibility.
    Features and matches are cached independently per image/pair.
    """

    CACHE_FORMAT_VERSION = 1
    DEFAULT_MAX_SIZE_GB = 10.0

    def __init__(self, cache_root: Path, max_cache_size_gb: float = None):
        """
        Initialize cache manager.

        Args:
            cache_root: Root directory for per-image cache
            max_cache_size_gb: Maximum cache size in GB (default: 10.0)
        """
        self.cache_root = Path(cache_root)
        self.max_cache_size_gb = max_cache_size_gb or self.DEFAULT_MAX_SIZE_GB

        # Create cache directory structure
        self.features_dir = self.cache_root / "features"
        self.matches_dir = self.cache_root / "matches"
        self.index_db_path = self.cache_root / "index.db"

        self.features_dir.mkdir(parents=True, exist_ok=True)
        self.matches_dir.mkdir(parents=True, exist_ok=True)

        # Initialize SQLite index
        self._init_index_db()

        # Hit/miss counters for migration logging
        self.cache_hits_content = 0
        self.cache_hits_legacy = 0
        self.cache_misses = 0

    def _init_index_db(self):
        """Initialize SQLite index database with schema."""
        conn = sqlite3.connect(str(self.index_db_path))
        cursor = conn.cursor()

        # Feature cache index with migration (adds n_keypoints when missing)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feature_cache (
                cache_key TEXT PRIMARY KEY,
                image_name TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                max_features INTEGER NOT NULL,
                n_keypoints INTEGER,
                created_at REAL NOT NULL,
                last_used REAL NOT NULL,
                file_path TEXT
            )
        """)

        feat_cols = [r[1] for r in cursor.execute("PRAGMA table_info(feature_cache)").fetchall()]
        feat_required = {"cache_key", "image_name", "file_size", "max_features", "n_keypoints", "created_at", "last_used", "file_path"}
        if not feat_required.issubset(set(feat_cols)):
            legacy_name = f"feature_cache_legacy_{int(time.time())}"
            try:
                cursor.execute(f"ALTER TABLE feature_cache RENAME TO {legacy_name}")
            except Exception:
                cursor.execute("DROP TABLE IF EXISTS feature_cache")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feature_cache (
                    cache_key TEXT PRIMARY KEY,
                    image_name TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    max_features INTEGER NOT NULL,
                    n_keypoints INTEGER,
                    created_at REAL NOT NULL,
                    last_used REAL NOT NULL,
                    file_path TEXT
                )
            """)
            try:
                cursor.execute(f"""
                    INSERT OR IGNORE INTO feature_cache
                    (cache_key, image_name, file_size, max_features, n_keypoints, created_at, last_used, file_path)
                    SELECT cache_key, image_name, file_size, max_features,
                           NULL as n_keypoints, created_at, last_used, file_path
                    FROM {legacy_name}
                """)
            except Exception:
                pass

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_image_name ON feature_cache(image_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_last_used ON feature_cache(last_used)")

        # Match cache index (use img1_key/img2_key) with migration of older schemas.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS match_cache (
                cache_key TEXT PRIMARY KEY,
                img1_key TEXT NOT NULL,
                img2_key TEXT NOT NULL,
                matcher_type TEXT NOT NULL,
                n_matches INTEGER,
                n_inliers INTEGER,
                created_at REAL NOT NULL,
                last_used REAL NOT NULL
            )
        """)

        match_cols = [r[1] for r in cursor.execute("PRAGMA table_info(match_cache)").fetchall()]
        match_required = {"cache_key", "img1_key", "img2_key", "matcher_type", "n_matches", "n_inliers", "created_at", "last_used"}
        if not match_required.issubset(set(match_cols)):
            legacy_name = f"match_cache_legacy_{int(time.time())}"
            try:
                cursor.execute(f"ALTER TABLE match_cache RENAME TO {legacy_name}")
            except Exception:
                cursor.execute("DROP TABLE IF EXISTS match_cache")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS match_cache (
                    cache_key TEXT PRIMARY KEY,
                    img1_key TEXT NOT NULL,
                    img2_key TEXT NOT NULL,
                    matcher_type TEXT NOT NULL,
                    n_matches INTEGER,
                    n_inliers INTEGER,
                    created_at REAL NOT NULL,
                    last_used REAL NOT NULL
                )
            """)
            try:
                cursor.execute(f"""
                    INSERT OR IGNORE INTO match_cache
                    (cache_key, img1_key, img2_key, matcher_type, n_matches, n_inliers, created_at, last_used)
                    SELECT cache_key,
                           COALESCE(img1_key, image1_key),
                           COALESCE(img2_key, image2_key),
                           matcher_type,
                           COALESCE(n_matches, 0),
                           COALESCE(n_inliers, 0),
                           created_at,
                           last_used
                    FROM {legacy_name}
                """)
            except Exception:
                pass

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_image_keys ON match_cache(img1_key, img2_key)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_match_last_used ON match_cache(last_used)")

        conn.commit()
        conn.close()

    def _hash_file(self, image_path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
        """Return a short SHA256 digest of file contents (chunked to limit memory)."""
        h = hashlib.sha256()
        with image_path.open("rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                h.update(chunk)
        return h.hexdigest()[:32]

    def _legacy_image_cache_key(self, image_path: Path, max_features: int) -> str:
        """Legacy key based on filename and size (no content hash)."""
        stat = image_path.stat()
        key_data = f"{image_path.name}:{stat.st_size}:{max_features}:v{self.CACHE_FORMAT_VERSION}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:32]

    def compute_image_cache_key(self, image_path: Path, max_features: int) -> str:
        """
        Generate cache key for a single image's features.

        Uses content hash so the same image reused across different folders/names
        still hits the cache. Falls back to name+size if hashing fails.
        """
        try:
            stat = image_path.stat()
            content_digest = self._hash_file(image_path)
            key_data = f"{content_digest}:{stat.st_size}:{max_features}:v{self.CACHE_FORMAT_VERSION}"
            return hashlib.sha256(key_data.encode()).hexdigest()[:32]
        except Exception as e:
            logger.warning(f"Failed to compute content hash for {image_path}: {e}")
            # Fallback: use legacy name+size key
            try:
                stat = image_path.stat()
                key_data = f"{image_path.name}:{stat.st_size}:{max_features}:v{self.CACHE_FORMAT_VERSION}"
            except Exception:
                key_data = f"{image_path.name}:{max_features}:v{self.CACHE_FORMAT_VERSION}"
            return hashlib.sha256(key_data.encode()).hexdigest()[:32]

    def compute_match_cache_key(self, img1_key: str, img2_key: str,
                                matcher_type: str = "exhaustive") -> str:
        """
        Generate cache key for a match between two images.

        Args:
            img1_key: Cache key of first image
            img2_key: Cache key of second image
            matcher_type: Matcher type used

        Returns:
            32-character cache key (SHA256 truncated)
        """
        # Sort keys so (A,B) == (B,A)
        keys = tuple(sorted([img1_key, img2_key]))
        key_data = f"{keys[0]}:{keys[1]}:{matcher_type}:v{self.CACHE_FORMAT_VERSION}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:32]

    def get_cached_match(self, img1_key: str, img2_key: str, matcher_type: str) -> Optional[Dict]:
        """
        Check if a match (inlier correspondences) is cached for this image pair.

        Returns a dict with paths if found, otherwise None.
        """
        keys = tuple(sorted([img1_key, img2_key]))
        cache_key = self.compute_match_cache_key(keys[0], keys[1], matcher_type)

        try:
            conn = sqlite3.connect(str(self.index_db_path))
            cursor = conn.cursor()
            cursor.execute("""
                SELECT cache_key, n_matches, n_inliers, created_at
                FROM match_cache
                WHERE cache_key = ?
            """, (cache_key,))
            row = cursor.fetchone()
            conn.close()

            if not row:
                return None

            cache_dir = self.matches_dir / cache_key
            matches_path = cache_dir / "matches.npy"
            metadata_path = cache_dir / "metadata.json"

            if matches_path.exists() and metadata_path.exists():
                self._update_last_used_match(cache_key)
                return {
                    "cache_key": row[0],
                    "n_matches": row[1],
                    "n_inliers": row[2],
                    "created_at": row[3],
                    "matches_path": matches_path,
                    "metadata_path": metadata_path,
                }

            logger.warning(f"Match cache files missing for {cache_key}, removing index entry")
            self._delete_match_cache_entry(cache_key)
            return None
        except Exception as e:
            logger.error(f"Error checking match cache for pair {img1_key[:8]}.. {img2_key[:8]}..: {e}")
            return None

    def save_match(self, img1_key: str, img2_key: str, matcher_type: str,
                   matches: np.ndarray, metadata: dict) -> str:
        """
        Save inlier matches for an image pair.

        Args:
            img1_key/img2_key: per-image cache keys (order-independent)
            matcher_type: e.g. "exhaustive_affine_r0.75"
            matches: (K,2) uint32 array of (kp_idx1, kp_idx2)
            metadata: extra metadata (will be stored in metadata.json)
        """
        keys = tuple(sorted([img1_key, img2_key]))
        cache_key = self.compute_match_cache_key(keys[0], keys[1], matcher_type)
        cache_dir = self.matches_dir / cache_key

        try:
            cache_dir.mkdir(parents=True, exist_ok=True)

            # Atomic write for matches
            matches_tmp = cache_dir / "matches_tmp.npy"
            matches_final = cache_dir / "matches.npy"
            np.save(str(matches_tmp), matches.astype(np.uint32))
            matches_tmp.rename(matches_final)

            metadata_tmp = cache_dir / "metadata.json.tmp"
            metadata_final = cache_dir / "metadata.json"
            full_metadata = {
                "version": self.CACHE_FORMAT_VERSION,
                "image1_key": keys[0],
                "image2_key": keys[1],
                "matcher_type": matcher_type,
                "n_matches": int(len(matches)),
                "n_inliers": int(len(matches)),
                "created_at": time.time(),
                **(metadata or {}),
            }
            metadata_tmp.write_text(json.dumps(full_metadata, indent=2))
            metadata_tmp.rename(metadata_final)

            # Update index
            conn = sqlite3.connect(str(self.index_db_path))
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO match_cache
                (cache_key, img1_key, img2_key, matcher_type, n_matches, n_inliers, created_at, last_used)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                cache_key,
                keys[0],
                keys[1],
                matcher_type,
                int(full_metadata["n_matches"]),
                int(full_metadata["n_inliers"]),
                float(full_metadata["created_at"]),
                time.time(),
            ))
            conn.commit()
            conn.close()

            return cache_key
        except Exception as e:
            logger.error(f"Error saving match cache {cache_key}: {e}")
            if cache_dir.exists():
                shutil.rmtree(cache_dir, ignore_errors=True)
            raise

    def load_match(self, cache_key: str) -> Tuple[np.ndarray, dict]:
        """Load inlier matches for a pair from cache."""
        cache_dir = self.matches_dir / cache_key
        matches_path = cache_dir / "matches.npy"
        metadata_path = cache_dir / "metadata.json"

        if not (matches_path.exists() and metadata_path.exists()):
            raise FileNotFoundError(f"Match cache files not found for {cache_key}")

        metadata = json.loads(metadata_path.read_text())
        matches = np.load(str(matches_path))
        if matches.dtype != np.uint32 or matches.ndim != 2 or matches.shape[1] != 2:
            raise ValueError(f"Invalid matches array for {cache_key}: {matches.shape} {matches.dtype}")
        return matches, metadata

    def _update_last_used_match(self, cache_key: str):
        """Update last_used timestamp for match cache entry."""
        try:
            conn = sqlite3.connect(str(self.index_db_path))
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE match_cache
                SET last_used = ?
                WHERE cache_key = ?
            """, (time.time(), cache_key))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"Failed to update match last_used for {cache_key}: {e}")

    def _delete_match_cache_entry(self, cache_key: str):
        """Delete match cache entry and files."""
        try:
            conn = sqlite3.connect(str(self.index_db_path))
            cursor = conn.cursor()
            cursor.execute("DELETE FROM match_cache WHERE cache_key = ?", (cache_key,))
            conn.commit()
            conn.close()

            cache_dir = self.matches_dir / cache_key
            if cache_dir.exists():
                shutil.rmtree(cache_dir, ignore_errors=True)
        except Exception as e:
            logger.warning(f"Failed to delete match cache entry {cache_key}: {e}")

    def get_cached_features(self, image_path: Path, max_features: int) -> Optional[Dict]:
        """
        Check if features are cached for this image.

        Args:
            image_path: Path to image file
            max_features: Maximum features setting

        Returns:
            Cache entry dict if found, None otherwise
        """
        try:
            # Try both the content-hash key (new) and the legacy name+size key to reuse old caches
            candidate_keys = []
            new_key = self.compute_image_cache_key(image_path, max_features)
            candidate_keys.append((new_key, "content"))

            try:
                legacy_key = self._legacy_image_cache_key(image_path, max_features)
                if legacy_key != new_key:
                    candidate_keys.append((legacy_key, "legacy"))
            except Exception:
                pass

            conn = sqlite3.connect(str(self.index_db_path))
            cursor = conn.cursor()

            for cache_key, key_type in candidate_keys:
                cursor.execute("""
                    SELECT cache_key, image_name, n_keypoints, created_at
                    FROM feature_cache
                    WHERE cache_key = ?
                """, (cache_key,))

                row = cursor.fetchone()

                if row:
                    # Verify cache files exist
                    cache_dir = self.features_dir / cache_key
                    metadata_path = cache_dir / "metadata.json"
                    keypoints_path = cache_dir / "keypoints.npy"
                    descriptors_path = cache_dir / "descriptors.npy"

                    if metadata_path.exists() and keypoints_path.exists() and descriptors_path.exists():
                        # Update last_used timestamp
                        self._update_last_used_feature(cache_key)

                        if key_type == "content":
                            self.cache_hits_content += 1
                        else:
                            self.cache_hits_legacy += 1

                        conn.close()
                        return {
                            'cache_key': row[0],
                            'image_name': row[1],
                            'n_keypoints': row[2],
                            'created_at': row[3],
                            'metadata_path': metadata_path,
                            'keypoints_path': keypoints_path,
                            'descriptors_path': descriptors_path
                        }
                    else:
                        # Cache entry exists but files missing - clean up
                        logger.warning(f"Cache files missing for {cache_key}, removing index entry")
                        self._delete_feature_cache_entry(cache_key)

            conn.close()
            self.cache_misses += 1
            return None

        except Exception as e:
            logger.error(f"Error checking feature cache for {image_path}: {e}")
            return None

    def save_features(self, image_path: Path, max_features: int,
                     keypoints: np.ndarray, descriptors: np.ndarray, metadata: dict) -> str:
        """
        Save features to cache.

        Args:
            image_path: Path to original image
            max_features: Maximum features setting
            keypoints: Numpy array (N, 6) float32: [x, y, scale, orientation, score, octave]
            descriptors: Numpy array (N, 128) uint8: SIFT descriptors
            metadata: Additional metadata dict

        Returns:
            Cache key
        """
        cache_key = self.compute_image_cache_key(image_path, max_features)
        cache_dir = self.features_dir / cache_key

        try:
            # Create cache directory
            cache_dir.mkdir(parents=True, exist_ok=True)

            # Debug: verify directory exists and is writable
            if not cache_dir.exists():
                raise RuntimeError(f"Failed to create cache directory: {cache_dir}")

            # Test write permissions
            test_file = cache_dir / ".write_test"
            try:
                test_file.write_text("test")
                test_file.unlink()
            except Exception as e:
                raise RuntimeError(f"Cache directory not writable: {cache_dir}: {e}")

            # Save keypoints (atomic write)
            # NOTE: np.save() automatically adds .npy extension, so we use _tmp instead of .tmp
            keypoints_temp = cache_dir / "keypoints_tmp.npy"
            keypoints_final = cache_dir / "keypoints.npy"

            # np.save needs string path for compatibility
            import sys
            print(f"[CACHE DEBUG] Saving keypoints: shape={keypoints.shape}, dtype={keypoints.dtype}", file=sys.stderr, flush=True)
            np.save(str(keypoints_temp), keypoints)

            # Verify file was created before renaming
            if not keypoints_temp.exists():
                print(f"[CACHE DEBUG] File does not exist after np.save: {keypoints_temp}", file=sys.stderr, flush=True)
                print(f"[CACHE DEBUG] Directory contents: {list(cache_dir.iterdir())}", file=sys.stderr, flush=True)
                raise RuntimeError(f"np.save failed to create {keypoints_temp}")

            keypoints_temp.rename(keypoints_final)
            print(f"[CACHE DEBUG] Successfully saved keypoints to {keypoints_final}", file=sys.stderr, flush=True)

            # Save descriptors (atomic write)
            descriptors_temp = cache_dir / "descriptors_tmp.npy"
            descriptors_final = cache_dir / "descriptors.npy"

            print(f"[CACHE DEBUG] Saving descriptors: shape={descriptors.shape}, dtype={descriptors.dtype}", file=sys.stderr, flush=True)
            np.save(str(descriptors_temp), descriptors)

            if not descriptors_temp.exists():
                print(f"[CACHE DEBUG] File does not exist after np.save: {descriptors_temp}", file=sys.stderr, flush=True)
                raise RuntimeError(f"np.save failed to create {descriptors_temp}")

            descriptors_temp.rename(descriptors_final)
            print(f"[CACHE DEBUG] Successfully saved descriptors to {descriptors_final}", file=sys.stderr, flush=True)

            # Save metadata (atomic write)
            metadata_temp = cache_dir / "metadata.json.tmp"
            metadata_final = cache_dir / "metadata.json"

            full_metadata = {
                'version': self.CACHE_FORMAT_VERSION,
                'image_name': image_path.name,
                'file_size': image_path.stat().st_size,
                'max_features': max_features,
                'n_keypoints': len(keypoints),
                'created_at': time.time(),
                **metadata
            }

            metadata_temp.write_text(json.dumps(full_metadata, indent=2))
            metadata_temp.rename(metadata_final)

            # Update index
            conn = sqlite3.connect(str(self.index_db_path))
            cursor = conn.cursor()

            cursor.execute("""
                INSERT OR REPLACE INTO feature_cache
                (cache_key, image_name, file_size, max_features, n_keypoints,
                 created_at, last_used, file_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                cache_key,
                image_path.name,
                full_metadata['file_size'],
                max_features,
                len(keypoints),
                full_metadata['created_at'],
                time.time(),
                str(image_path)
            ))

            conn.commit()
            conn.close()

            logger.debug(f"Saved features to cache: {cache_key} ({len(keypoints)} keypoints)")
            return cache_key

        except Exception as e:
            logger.error(f"Error saving features to cache: {e}")
            # Clean up partial files
            if cache_dir.exists():
                shutil.rmtree(cache_dir, ignore_errors=True)
            raise

    def load_features(self, cache_key: str) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Load features from cache.

        Args:
            cache_key: Cache key from get_cached_features()

        Returns:
            (keypoints array, descriptors array, metadata dict)

        Raises:
            FileNotFoundError: If cache files don't exist
            ValueError: If cache data is corrupted
        """
        cache_dir = self.features_dir / cache_key
        metadata_path = cache_dir / "metadata.json"
        keypoints_path = cache_dir / "keypoints.npy"
        descriptors_path = cache_dir / "descriptors.npy"

        if not (metadata_path.exists() and keypoints_path.exists() and descriptors_path.exists()):
            raise FileNotFoundError(f"Cache files not found for {cache_key}")

        try:
            # Load metadata
            metadata = json.loads(metadata_path.read_text())

            # Validate version
            if metadata.get('version', 0) != self.CACHE_FORMAT_VERSION:
                raise ValueError(f"Cache format version mismatch: {metadata.get('version')} != {self.CACHE_FORMAT_VERSION}")

            # Load keypoints (use string path for compatibility)
            keypoints = np.load(str(keypoints_path))

            # Validate keypoints shape and dtype
            if keypoints.dtype != np.float32:
                raise ValueError(f"Invalid keypoints dtype: {keypoints.dtype}")

            if keypoints.ndim != 2 or keypoints.shape[1] != 6:
                raise ValueError(f"Invalid keypoints shape: {keypoints.shape}")

            if len(keypoints) != metadata['n_keypoints']:
                raise ValueError(f"Keypoint count mismatch: {len(keypoints)} != {metadata['n_keypoints']}")

            # Load descriptors
            descriptors = np.load(str(descriptors_path))

            # Validate descriptors shape and dtype
            if descriptors.dtype != np.uint8:
                raise ValueError(f"Invalid descriptors dtype: {descriptors.dtype}")

            if descriptors.ndim != 2 or descriptors.shape[1] != 128:
                raise ValueError(f"Invalid descriptors shape: {descriptors.shape}")

            if len(descriptors) != len(keypoints):
                raise ValueError(f"Descriptor/keypoint count mismatch: {len(descriptors)} != {len(keypoints)}")

            return keypoints, descriptors, metadata

        except Exception as e:
            logger.error(f"Error loading features from cache {cache_key}: {e}")
            raise

    def _update_last_used_feature(self, cache_key: str):
        """Update last_used timestamp for feature cache entry."""
        try:
            conn = sqlite3.connect(str(self.index_db_path))
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE feature_cache
                SET last_used = ?
                WHERE cache_key = ?
            """, (time.time(), cache_key))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"Failed to update last_used for {cache_key}: {e}")

    def _delete_feature_cache_entry(self, cache_key: str):
        """Delete feature cache entry and files."""
        try:
            # Delete index entry
            conn = sqlite3.connect(str(self.index_db_path))
            cursor = conn.cursor()
            cursor.execute("DELETE FROM feature_cache WHERE cache_key = ?", (cache_key,))
            conn.commit()
            conn.close()

            # Delete cache files
            cache_dir = self.features_dir / cache_key
            if cache_dir.exists():
                shutil.rmtree(cache_dir, ignore_errors=True)

        except Exception as e:
            logger.warning(f"Failed to delete cache entry {cache_key}: {e}")

    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        try:
            conn = sqlite3.connect(str(self.index_db_path))
            cursor = conn.cursor()

            # Feature cache stats
            cursor.execute("SELECT COUNT(*), SUM(n_keypoints) FROM feature_cache")
            feature_count, total_keypoints = cursor.fetchone()

            # Match cache stats
            cursor.execute("SELECT COUNT(*), SUM(n_matches) FROM match_cache")
            match_count, total_matches = cursor.fetchone()

            conn.close()

            # Calculate disk usage
            features_size = sum(f.stat().st_size for f in self.features_dir.rglob("*") if f.is_file())
            matches_size = sum(f.stat().st_size for f in self.matches_dir.rglob("*") if f.is_file())
            total_size = features_size + matches_size

            return {
                'feature_cache_entries': feature_count or 0,
                'total_keypoints': int(total_keypoints or 0),
                'match_cache_entries': match_count or 0,
                'total_matches': int(total_matches or 0),
                'features_size_mb': features_size / (1024 * 1024),
                'matches_size_mb': matches_size / (1024 * 1024),
                'total_size_mb': total_size / (1024 * 1024),
                'max_size_mb': self.max_cache_size_gb * 1024
            }

        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {}

    def get_hit_stats(self, reset: bool = False) -> Dict:
        """Return and optionally reset hit/miss counters (for migration logging)."""
        stats = {
            'content_hits': self.cache_hits_content,
            'legacy_hits': self.cache_hits_legacy,
            'misses': self.cache_misses
        }

        if reset:
            self.cache_hits_content = 0
            self.cache_hits_legacy = 0
            self.cache_misses = 0

        return stats

    def cleanup_old_cache(self, max_age_days: int = 90):
        """
        Remove cache entries older than max_age_days.

        Args:
            max_age_days: Maximum age in days
        """
        cutoff_time = time.time() - (max_age_days * 24 * 3600)

        try:
            conn = sqlite3.connect(str(self.index_db_path))
            cursor = conn.cursor()

            # Get old feature cache entries
            cursor.execute("""
                SELECT cache_key FROM feature_cache
                WHERE last_used < ?
            """, (cutoff_time,))

            old_features = [row[0] for row in cursor.fetchall()]

            # Delete them
            for cache_key in old_features:
                self._delete_feature_cache_entry(cache_key)

            logger.info(f"Cleaned up {len(old_features)} old feature cache entries")

            conn.close()

        except Exception as e:
            logger.error(f"Error cleaning up old cache: {e}")

    def enforce_size_limit(self):
        """Enforce cache size limit using LRU eviction."""
        stats = self.get_cache_stats()
        current_size_mb = stats.get('total_size_mb', 0)
        max_size_mb = self.max_cache_size_gb * 1024

        if current_size_mb <= max_size_mb:
            return

        logger.info(f"Cache size {current_size_mb:.1f}MB exceeds limit {max_size_mb:.1f}MB, cleaning up...")

        try:
            conn = sqlite3.connect(str(self.index_db_path))
            cursor = conn.cursor()

            # Get entries sorted by last_used (LRU)
            cursor.execute("""
                SELECT cache_key FROM feature_cache
                ORDER BY last_used ASC
            """)

            lru_entries = [row[0] for row in cursor.fetchall()]
            conn.close()

            # Delete until under limit
            for cache_key in lru_entries:
                self._delete_feature_cache_entry(cache_key)

                # Check size again
                stats = self.get_cache_stats()
                if stats.get('total_size_mb', 0) <= max_size_mb:
                    break

            logger.info(f"Cache cleanup complete, new size: {stats.get('total_size_mb', 0):.1f}MB")

        except Exception as e:
            logger.error(f"Error enforcing cache size limit: {e}")

    # =========================================================================
    # Match run tracking methods (for progress reporting during incremental matching)
    # =========================================================================

    def note_match_run_start(self):
        """Initialize counters for a new match run."""
        self._match_run_start_time = time.time()
        self._match_hits = 0
        self._match_computed = 0
        self._match_skipped = 0

    def note_match_hit(self, count: int = 1):
        """Record cache hits during matching."""
        self._match_hits = getattr(self, '_match_hits', 0) + count

    def note_match_computed(self, count: int = 1):
        """Record newly computed matches."""
        self._match_computed = getattr(self, '_match_computed', 0) + count

    def note_match_skipped(self, count: int = 1):
        """Record skipped pairs (insufficient features/matches)."""
        self._match_skipped = getattr(self, '_match_skipped', 0) + count

    def get_match_lookup_summary(self) -> Dict:
        """Return summary of match run statistics."""
        elapsed = time.time() - getattr(self, '_match_run_start_time', time.time())
        return {
            'hit': getattr(self, '_match_hits', 0),
            'computed': getattr(self, '_match_computed', 0),
            'skipped': getattr(self, '_match_skipped', 0),
            'time': elapsed,
        }

    def cleanup_old_entries(self, max_age_days: int = 30):
        """
        Remove cache entries older than max_age_days.

        Args:
            max_age_days: Maximum age in days
        """
        cutoff_time = time.time() - (max_age_days * 24 * 3600)

        try:
            conn = sqlite3.connect(str(self.index_db_path))
            cursor = conn.cursor()

            # Get old feature cache entries
            cursor.execute("""
                SELECT cache_key FROM feature_cache
                WHERE last_used < ?
            """, (cutoff_time,))

            old_features = [row[0] for row in cursor.fetchall()]

            # Delete them
            for cache_key in old_features:
                self._delete_feature_cache_entry(cache_key)

            logger.info(f"Cleaned up {len(old_features)} old feature cache entries")

            conn.close()

        except Exception as e:
            logger.error(f"Error cleaning up old cache: {e}")

    def enforce_size_limit(self):
        """Enforce cache size limit using LRU eviction."""
        stats = self.get_cache_stats()
        current_size_mb = stats.get('total_size_mb', 0)
        max_size_mb = self.max_cache_size_gb * 1024

        if current_size_mb <= max_size_mb:
            return

        logger.info(f"Cache size {current_size_mb:.1f}MB exceeds limit {max_size_mb:.1f}MB, cleaning up...")

        try:
            conn = sqlite3.connect(str(self.index_db_path))
            cursor = conn.cursor()

            # Get entries sorted by last_used (LRU)
            cursor.execute("""
                SELECT cache_key FROM feature_cache
                ORDER BY last_used ASC
            """)

            lru_entries = [row[0] for row in cursor.fetchall()]
            conn.close()

            # Delete until under limit
            for cache_key in lru_entries:
                self._delete_feature_cache_entry(cache_key)

                # Check size again
                stats = self.get_cache_stats()
                if stats.get('total_size_mb', 0) <= max_size_mb:
                    break

            logger.info(f"Cache cleanup complete, new size: {stats.get('total_size_mb', 0):.1f}MB")

        except Exception as e:
            logger.error(f"Error enforcing cache size limit: {e}")

    # =========================================================================
    # Match run tracking methods (for progress reporting during incremental matching)
    # =========================================================================

    def note_match_run_start(self):
        """Initialize counters for a new match run."""
        self._match_run_start_time = time.time()
        self._match_hits = 0
        self._match_computed = 0
        self._match_skipped = 0

    def note_match_hit(self, count: int = 1):
        """Record cache hits during matching."""
        self._match_hits = getattr(self, '_match_hits', 0) + count

    def note_match_computed(self, count: int = 1):
        """Record newly computed matches."""
        self._match_computed = getattr(self, '_match_computed', 0) + count

    def note_match_skipped(self, count: int = 1):
        """Record skipped pairs (insufficient features/matches)."""
        self._match_skipped = getattr(self, '_match_skipped', 0) + count

    def get_match_lookup_summary(self) -> Dict:
        """Return summary of match run statistics."""
        elapsed = time.time() - getattr(self, '_match_run_start_time', time.time())
        return {
            'hit': getattr(self, '_match_hits', 0),
            'computed': getattr(self, '_match_computed', 0),
            'skipped': getattr(self, '_match_skipped', 0),
            'time': elapsed,
        }
