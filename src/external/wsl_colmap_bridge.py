#!/usr/bin/env python3
"""
WSL COLMAP Bridge - GPU-accelerated pycolmap via WSL

This script runs inside WSL and provides GPU-accelerated COLMAP processing.
It's called from the Windows application via subprocess.
"""

import sys
import json
import os
import hashlib
import shutil
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np


def windows_to_wsl_path(win_path: str) -> str:
    """Convert Windows path to WSL path."""
    # C:\Users\... -> /mnt/c/Users/...
    path = win_path.replace('\\', '/')
    if len(path) >= 2 and path[1] == ':':
        drive = path[0].lower()
        path = f'/mnt/{drive}{path[2:]}'
    return path


def wsl_to_windows_path(wsl_path: str) -> str:
    """Convert WSL path to Windows path."""
    # /mnt/c/Users/... -> C:\Users\...
    if wsl_path.startswith('/mnt/') and len(wsl_path) > 6:
        drive = wsl_path[5].upper()
        path = wsl_path[6:].replace('/', '\\')
        return f'{drive}:{path}'
    return wsl_path


def compute_cache_key(image_paths: list, max_features: int,
                      matcher_type: str = "exhaustive",
                      use_affine: bool = False,
                      blend_method: str = "multiband",
                      warp_interpolation: str = "linear") -> str:
    """Compute a hash key for the current image set and settings.

    Uses file size instead of modification time so cache survives file copying.
    """
    # Include image paths, file sizes, and key settings
    cache_data = []
    for path in sorted(image_paths):
        try:
            size = path.stat().st_size
            cache_data.append(f"{path.name}:{size}")  # Use filename only, not full path
        except:
            cache_data.append(str(path.name))
    cache_data.append(f"max_features:{max_features}")
    # Include matching/blending settings and a cache format version so stale DBs don't get reused
    cache_data.append("global_cache_format:v2")
    cache_data.append(f"matcher_type:{matcher_type}")
    cache_data.append(f"use_affine:{bool(use_affine)}")
    cache_data.append(f"blend_method:{str(blend_method).lower()}")
    cache_data.append(f"warp_interp:{str(warp_interpolation).lower()}")
    cache_data.append(f"count:{len(image_paths)}")

    cache_str = "\n".join(cache_data)
    return hashlib.md5(cache_str.encode()).hexdigest()


def is_cache_valid(workspace: Path, cache_key: str, n_images: int) -> bool:
    """Check if the cached database is valid for the current image set."""
    cache_file = workspace / "cache_key.txt"
    database_path = workspace / "database.db"

    if not cache_file.exists():
        print(f"[DEBUG] Cache miss: cache_key.txt not found at {cache_file}", file=sys.stderr, flush=True)
        return False

    if not database_path.exists():
        print(f"[DEBUG] Cache miss: database.db not found at {database_path}", file=sys.stderr, flush=True)
        return False

    try:
        stored_key = cache_file.read_text().strip()
        if stored_key != cache_key:
            print(f"[DEBUG] Cache miss: key mismatch", file=sys.stderr, flush=True)
            print(f"[DEBUG]   Stored:  {stored_key[:16]}...", file=sys.stderr, flush=True)
            print(f"[DEBUG]   Current: {cache_key[:16]}...", file=sys.stderr, flush=True)
            return False

        # Verify database has the expected number of images
        import sqlite3
        conn = sqlite3.connect(str(database_path))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM images")
        db_image_count = cursor.fetchone()[0]
        conn.close()

        if db_image_count != n_images:
            print(f"[DEBUG] Cache miss: image count mismatch (cached={db_image_count}, current={n_images})", file=sys.stderr, flush=True)
            return False

        print(f"[DEBUG] Cache hit! Using cached features for {n_images} images", file=sys.stderr, flush=True)
        return True
    except Exception as e:
        print(f"[DEBUG] Cache validation error: {e}", file=sys.stderr, flush=True)
        return False


def save_cache_key(workspace: Path, cache_key: str):
    """Save the cache key for future validation."""
    cache_file = workspace / "cache_key.txt"
    tmp_file = workspace / "cache_key.txt.tmp"
    tmp_file.write_text(cache_key)
    tmp_file.replace(cache_file)

    # Best-effort fsync to ensure durability
    try:
        import os
        fd = os.open(str(cache_file), os.O_RDONLY)
        os.fsync(fd)
        os.close(fd)
    except Exception:
        pass

    print(f"[CACHE] Wrote global cache key file: {cache_file}", file=sys.stderr, flush=True)


class StreamingBlender:
    """
    Memory-efficient streaming blender that processes images one at a time.

    Instead of accumulating all warped images in memory, this blender maintains
    running accumulators and blends each image immediately after warping.

    Supports: feather, linear, autostitch blend modes (streaming)
    Fallback: multiband uses memory-mapped files
    """

    def __init__(self, output_h: int, output_w: int, blend_method: str = "feather"):
        """
        Initialize the streaming blender.

        Args:
            output_h: Output canvas height
            output_w: Output canvas width
            blend_method: 'feather', 'linear', or 'autostitch'
        """
        self.output_h = output_h
        self.output_w = output_w
        self.blend_method = blend_method.lower()

        # Initialize accumulators based on blend method
        if self.blend_method in ['feather', 'linear']:
            # Weighted average: accumulate weighted sum and weight sum
            self.color_accum = np.zeros((output_h, output_w, 3), dtype=np.float64)
            self.weight_accum = np.zeros((output_h, output_w), dtype=np.float64)
        elif self.blend_method == 'autostitch':
            # Classic AutoStitch: winner-takes-all based on distance from edge
            # Pixel with highest distance to edge wins (furthest from seam)
            self.canvas = np.zeros((output_h, output_w, 3), dtype=np.uint8)
            self.distance_map = np.full((output_h, output_w), -1, dtype=np.float32)
        else:
            # For multiband, we can't stream - will use memory-mapped fallback
            self.color_accum = None
            self.weight_accum = None

        self.images_blended = 0

    def _compute_distance_transform(self, alpha: np.ndarray) -> np.ndarray:
        """Compute distance from each pixel to the nearest edge."""
        import cv2
        # Distance transform gives distance to nearest zero pixel
        mask = (alpha > 127).astype(np.uint8)
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        return dist

    def _compute_feather_weights(self, alpha: np.ndarray, region_h: int, region_w: int) -> np.ndarray:
        """Compute feathering weights based on distance from edge."""
        dist = self._compute_distance_transform(alpha)
        # Normalize to 0-1, with falloff at edges
        max_dist = max(dist.max(), 1)
        weights = dist / max_dist
        # Apply smooth falloff
        weights = np.clip(weights, 0, 1)
        return weights.astype(np.float32)

    def add_image(self, warped: np.ndarray, alpha: np.ndarray, bbox: tuple):
        """
        Add a single warped image to the blend.

        Args:
            warped: Warped image (H, W, 3) uint8
            alpha: Alpha mask (H, W) uint8
            bbox: (x_min, y_min, x_max, y_max) position in output canvas
        """
        import cv2
        x_min, y_min, x_max, y_max = bbox
        region_h, region_w = warped.shape[:2]

        # Ensure bbox matches warped image size
        actual_h = y_max - y_min
        actual_w = x_max - x_min
        if actual_h != region_h or actual_w != region_w:
            # Adjust bbox to match actual warped size
            y_max = y_min + region_h
            x_max = x_min + region_w

        # Clamp to canvas bounds
        src_y_start = max(0, -y_min)
        src_x_start = max(0, -x_min)
        dst_y_start = max(0, y_min)
        dst_x_start = max(0, x_min)
        dst_y_end = min(self.output_h, y_max)
        dst_x_end = min(self.output_w, x_max)
        src_y_end = src_y_start + (dst_y_end - dst_y_start)
        src_x_end = src_x_start + (dst_x_end - dst_x_start)

        if dst_y_end <= dst_y_start or dst_x_end <= dst_x_start:
            return  # No overlap with canvas

        # Extract the valid region
        warped_region = warped[src_y_start:src_y_end, src_x_start:src_x_end]
        alpha_region = alpha[src_y_start:src_y_end, src_x_start:src_x_end]

        if self.blend_method in ['feather', 'linear']:
            # Compute weights
            if self.blend_method == 'feather':
                weights = self._compute_feather_weights(alpha_region, region_h, region_w)
            else:
                # Linear: uniform weight where alpha > 0
                weights = (alpha_region / 255.0).astype(np.float32)

            # Accumulate weighted colors
            warped_float = warped_region.astype(np.float64)
            for c in range(3):
                self.color_accum[dst_y_start:dst_y_end, dst_x_start:dst_x_end, c] += warped_float[:, :, c] * weights
            self.weight_accum[dst_y_start:dst_y_end, dst_x_start:dst_x_end] += weights

        elif self.blend_method == 'autostitch':
            # Classic AutoStitch: winner-takes-all based on distance from edge
            dist = self._compute_distance_transform(alpha_region)

            # Get current distance values in the destination region
            dst_dist = self.distance_map[dst_y_start:dst_y_end, dst_x_start:dst_x_end]

            # Mask for valid pixels (alpha > threshold)
            valid_mask = alpha_region > 127

            # Mask for pixels where this image has better (higher) distance
            better_mask = (dist > dst_dist) & valid_mask

            # Update canvas with winner pixels
            self.canvas[dst_y_start:dst_y_end, dst_x_start:dst_x_end][better_mask] = warped_region[better_mask]
            self.distance_map[dst_y_start:dst_y_end, dst_x_start:dst_x_end][better_mask] = dist[better_mask]

        self.images_blended += 1

    def finalize(self) -> np.ndarray:
        """Finalize the blend and return the panorama."""
        if self.blend_method in ['feather', 'linear']:
            # Normalize by weight sum
            weight_sum = np.maximum(self.weight_accum, 1e-6)
            panorama = np.zeros((self.output_h, self.output_w, 3), dtype=np.float64)
            for c in range(3):
                panorama[:, :, c] = self.color_accum[:, :, c] / weight_sum

            # Free accumulators
            del self.color_accum
            del self.weight_accum

            return np.clip(panorama, 0, 255).astype(np.uint8)

        elif self.blend_method == 'autostitch':
            # Classic AutoStitch: just return the canvas with winner pixels
            panorama = self.canvas.copy()

            # Free memory
            del self.canvas
            del self.distance_map

            return panorama

        return None

    def supports_streaming(self) -> bool:
        """Check if current blend method supports streaming."""
        return self.blend_method in ['feather', 'linear', 'autostitch']


class MemoryMappedImageStore:
    """
    Store warped images on disk using memory-mapped files.

    This allows processing thousands of images without running out of RAM.
    Images are written to temp files and read back during blending.
    """

    def __init__(self, temp_dir: Path):
        self.temp_dir = temp_dir
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.image_files = []
        self.metadata = []

    def add_image(self, warped: np.ndarray, alpha: np.ndarray, bbox: tuple, idx: int):
        """Save warped image to disk."""
        import cv2

        # Save image and alpha to disk
        img_path = self.temp_dir / f"warped_{idx:06d}.png"
        alpha_path = self.temp_dir / f"alpha_{idx:06d}.png"

        cv2.imwrite(str(img_path), warped)
        cv2.imwrite(str(alpha_path), alpha)

        self.image_files.append((img_path, alpha_path))
        self.metadata.append({
            'bbox': bbox,
            'shape': warped.shape
        })

    def get_image(self, idx: int) -> tuple:
        """Load warped image from disk."""
        import cv2

        img_path, alpha_path = self.image_files[idx]
        warped = cv2.imread(str(img_path))
        alpha = cv2.imread(str(alpha_path), cv2.IMREAD_GRAYSCALE)
        bbox = self.metadata[idx]['bbox']

        return warped, alpha, bbox

    def __len__(self):
        return len(self.image_files)

    def cleanup(self):
        """Remove all temp files."""
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass


def copy_single_image(args):
    """Copy a single image to the workspace (for parallel execution)."""
    import shutil
    i, path, images_dir = args
    dest_name = f"{i:06d}{path.suffix}"
    dest = images_dir / dest_name
    shutil.copy2(path, dest)
    return i, dest_name, path


def check_feature_cache(image_paths: list, max_features: int, cache_manager):
    """
    Check which images have cached features.

    Args:
        image_paths: List of Path objects for images
        max_features: Maximum features setting
        cache_manager: COLMAPCacheManager instance

    Returns:
        Tuple of (cached_features_dict, uncached_images_list)
        cached_features_dict: {image_index: {'cache_key': str, 'cache_entry': dict}}
        uncached_images_list: [(index, Path), ...]
    """
    cached_features = {}
    uncached_images = []

    for i, img_path in enumerate(image_paths):
        cache_entry = cache_manager.get_cached_features(img_path, max_features)

        if cache_entry:
            cached_features[i] = {
                'cache_key': cache_entry['cache_key'],
                'cache_entry': cache_entry
            }
        else:
            uncached_images.append((i, img_path))

    # One-time migration log per run: show hit breakdown and reset counters
    try:
        stats = cache_manager.get_hit_stats(reset=True)
        print(f"[CACHE] Cache lookup summary: {stats['content_hits']} content-hit, {stats['legacy_hits']} legacy-hit, {stats['misses']} miss", file=sys.stderr, flush=True)
    except Exception:
        pass

    return cached_features, uncached_images


def _pair_id(image_id1: int, image_id2: int) -> int:
    """COLMAP pair_id encoding used by our read_matches_from_db()."""
    # Must match decode in read_matches_from_db: id2 = pair_id % 2147483647, id1 = pair_id // 2147483647
    if image_id1 == image_id2:
        raise ValueError("pair_id requires two different image ids")
    if image_id1 > image_id2:
        image_id1, image_id2 = image_id2, image_id1
    return int(image_id1) * 2147483647 + int(image_id2)


def _generate_pairs(n_images: int, matcher_type: str, sequential_overlap: int) -> list:
    """Generate (i,j) pairs (0-based indices)."""
    pairs = []
    if matcher_type == "sequential":
        overlap = max(1, int(sequential_overlap))
        for i in range(n_images):
            for j in range(i + 1, min(n_images, i + 1 + overlap)):
                pairs.append((i, j))
    else:
        # exhaustive (default)
        for i in range(n_images):
            for j in range(i + 1, n_images):
                pairs.append((i, j))
    return pairs


def _write_two_view_geometry(conn, image_id1: int, image_id2: int, matches: np.ndarray):
    """Insert/replace an inlier match set into two_view_geometries."""
    pair_id = _pair_id(image_id1, image_id2)
    rows = int(matches.shape[0])
    cols = 2
    data_blob = matches.astype(np.uint32).tobytes()
    # config, F/E/H unused by our pipeline reader
    conn.execute("""
        INSERT OR REPLACE INTO two_view_geometries
        (pair_id, rows, cols, data, config, F, E, H)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (pair_id, rows, cols, data_blob, 0, None, None, None))


def match_pairs_incremental(database_path: Path,
                            image_paths: list,
                            max_features: int,
                            cache_manager,
                            matcher_type: str,
                            sequential_overlap: int,
                            use_affine: bool,
                            cached_features: dict,
                            newly_extracted: dict):
    """
    Incremental matching:
    - reuse cached inlier matches per pair
    - compute missing pairs with OpenCV matcher + RANSAC
    - write results into two_view_geometries
    """
    import cv2
    import sqlite3
    import time

    n_images = len(image_paths)
    pairs = _generate_pairs(n_images, matcher_type, sequential_overlap)
    total_pairs = len(pairs)

    # Match settings (baked into matcher_id)
    # Stricter defaults to avoid bad transforms that cause visible mis-stitching/seams.
    ratio = 0.70
    ransac_thresh = 2.0
    min_inliers_to_store = 40
    min_inlier_ratio = 0.25
    model_tag = "affine" if use_affine else "homography"
    matcher_id = f"{matcher_type}_{model_tag}_v2_r{ratio:.2f}_t{ransac_thresh:.1f}_min{min_inliers_to_store}"

    print(f"[CACHE] Match caching enabled: {total_pairs:,} pairs ({matcher_id})", file=sys.stderr, flush=True)
    print(f"[CACHE] Match cache directory: {cache_manager.matches_dir}", file=sys.stderr, flush=True)

    # Build image_key map (use existing keys when available)
    image_keys = {}
    for i, img_path in enumerate(image_paths):
        if i in cached_features:
            image_keys[i] = cached_features[i]["cache_key"]
        elif i in newly_extracted and newly_extracted[i].get("cache_key"):
            image_keys[i] = newly_extracted[i]["cache_key"]
        else:
            image_keys[i] = cache_manager.compute_image_cache_key(img_path, max_features)

    # Feature loader (memoized)
    feat_cache = {}

    def load_feats(idx: int):
        if idx in feat_cache:
            return feat_cache[idx]

        kp = None
        desc = None

        if idx in newly_extracted and "keypoints" in newly_extracted[idx] and "descriptors" in newly_extracted[idx]:
            kp = newly_extracted[idx]["keypoints"]
            desc = newly_extracted[idx]["descriptors"]
        elif idx in cached_features:
            ck = cached_features[idx]["cache_key"]
            kp, desc, _ = cache_manager.load_features(ck)
        else:
            # As fallback, try loading from cache by recomputed key
            ck = image_keys[idx]
            kp, desc, _ = cache_manager.load_features(ck)

        if kp is None or desc is None:
            raise RuntimeError(f"No features available for image {idx}")

        kp_xy = kp[:, :2].astype(np.float32)
        # Convert descriptors to float32 for FLANN
        desc_f = desc.astype(np.float32)
        feat_cache[idx] = (kp_xy, desc_f)
        return feat_cache[idx]

    # Exact matcher + mutual check (more stable than approximate FLANN for repetitive textures)
    bf = cv2.BFMatcher(cv2.NORM_L2)

    hits = 0
    computed = 0
    skipped = 0

    conn = sqlite3.connect(str(database_path))
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")

    start = time.time()
    last_report = start

    for k, (i, j) in enumerate(pairs):
        # Periodic progress
        now = time.time()
        if now - last_report >= 2.0 or k == 0 or k == total_pairs - 1:
            pct = int(100 * (k / max(1, total_pairs)))
            print(f"[PROGRESS] Feature matching: {k:,}/{total_pairs:,} pairs ({pct}%)", file=sys.stderr, flush=True)
            last_report = now

        img1_key = image_keys[i]
        img2_key = image_keys[j]

        cached = cache_manager.get_cached_match(img1_key, img2_key, matcher_id)
        if cached:
            try:
                matches_arr, _meta = cache_manager.load_match(cached["cache_key"])
                if matches_arr.size > 0:
                    _write_two_view_geometry(conn, i + 1, j + 1, matches_arr)
                hits += 1
                continue
            except Exception as e:
                print(f"[WARNING] Failed to load cached match for ({i},{j}): {e}", file=sys.stderr, flush=True)

        # Compute missing pair
        try:
            kp1, d1 = load_feats(i)
            kp2, d2 = load_feats(j)
            if d1.shape[0] < 2 or d2.shape[0] < 2:
                skipped += 1
                continue

            # Ratio test (forward)
            knn_f = bf.knnMatch(d1, d2, k=2)
            good_f = {}
            for m_n in knn_f:
                if len(m_n) != 2:
                    continue
                m, n = m_n
                if m.distance < ratio * n.distance:
                    good_f[(m.queryIdx, m.trainIdx)] = m

            # Ratio test (reverse) + mutual consistency
            knn_r = bf.knnMatch(d2, d1, k=2)
            good = []
            for m_n in knn_r:
                if len(m_n) != 2:
                    continue
                m, n = m_n
                if m.distance < ratio * n.distance:
                    # reverse match is (trainIdx -> queryIdx) in forward coordinates
                    q = m.trainIdx
                    t = m.queryIdx
                    mm = good_f.get((q, t))
                    if mm is not None:
                        good.append(mm)

            if len(good) < min_inliers_to_store:
                skipped += 1
                continue

            pts1 = np.float32([kp1[m.queryIdx] for m in good])
            pts2 = np.float32([kp2[m.trainIdx] for m in good])

            if use_affine:
                _A, inlier_mask = cv2.estimateAffine2D(
                    pts1, pts2, method=cv2.RANSAC,
                    ransacReprojThreshold=ransac_thresh,
                    maxIters=2000, confidence=0.999
                )
            else:
                _H, inlier_mask = cv2.findHomography(
                    pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh
                )

            if inlier_mask is None:
                skipped += 1
                continue

            inlier_mask = inlier_mask.ravel().astype(bool)
            n_inliers = int(inlier_mask.sum())
            if n_inliers < min_inliers_to_store:
                skipped += 1
                continue

            inlier_ratio = n_inliers / max(1, len(good))
            if inlier_ratio < min_inlier_ratio:
                skipped += 1
                continue

            pairs_idx = np.array([[m.queryIdx, m.trainIdx] for m in good], dtype=np.uint32)
            inlier_pairs = pairs_idx[inlier_mask]

            # Affine sanity checks to prevent wild transforms that create visible tearing
            if use_affine and _A is not None:
                M = _A[:, :2].astype(np.float64)
                tx, ty = float(_A[0, 2]), float(_A[1, 2])
                # Scale via singular values
                try:
                    svals = np.linalg.svd(M, compute_uv=False)
                    smin, smax = float(svals.min()), float(svals.max())
                except Exception:
                    smin, smax = 0.0, 0.0

                # Approx rotation (from polar-ish decomposition)
                rot_deg = 0.0
                try:
                    rot_deg = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
                except Exception:
                    pass

                # Heuristic bounds (microscope tiling assumptions)
                if not (0.6 <= smin <= 1.6 and 0.6 <= smax <= 1.6):
                    skipped += 1
                    continue
                if abs(rot_deg) > 25.0:
                    skipped += 1
                    continue

                # Translation bounds relative to image size inferred from keypoints
                w_est = float(kp1[:, 0].max() - kp1[:, 0].min())
                h_est = float(kp1[:, 1].max() - kp1[:, 1].min())
                max_trans = max(w_est, h_est) * 3.0
                if (tx * tx + ty * ty) > (max_trans * max_trans):
                    skipped += 1
                    continue

            # Save + write to DB
            meta = {
                "n_putative": int(len(good)),
                "n_inliers": int(len(inlier_pairs)),
                "inlier_ratio": float(inlier_ratio),
                "ratio": float(ratio),
                "ransac_thresh": float(ransac_thresh),
                "model": model_tag,
            }
            cache_manager.save_match(img1_key, img2_key, matcher_id, inlier_pairs, meta)
            _write_two_view_geometry(conn, i + 1, j + 1, inlier_pairs)
            computed += 1
        except Exception as e:
            skipped += 1
            print(f"[WARNING] Match compute failed for ({i},{j}): {e}", file=sys.stderr, flush=True)

    conn.commit()
    conn.close()

    elapsed = time.time() - start
    print(f"[CACHE] Match cache summary: {hits} hit, {computed} computed, {skipped} skipped in {elapsed:.1f}s", file=sys.stderr, flush=True)


def extract_features_incremental(uncached_images: list, workspace_dir: Path, database_path: Path,
                                 max_features: int, cache_manager, gpu_index: int = -1,
                                 num_threads: int = -1, has_cuda: bool = False):
    """
    Extract features only for uncached images and save to cache.

    Args:
        uncached_images: List of (index, Path) tuples for images without cached features
        workspace_dir: Workspace directory for temporary processing
        database_path: Path to COLMAP database
        max_features: Maximum features setting
        cache_manager: COLMAPCacheManager instance
        gpu_index: GPU index (-1 for auto)
        num_threads: Number of threads (-1 for auto)
        has_cuda: Whether CUDA is available

    Returns:
        Dict mapping image index to {'cache_key': str, 'keypoints': np.ndarray}: {index: {...}}
    """
    import pycolmap
    import sqlite3
    import time

    if not uncached_images:
        return {}

    # Create temporary workspace for uncached images only
    temp_images_dir = workspace_dir / "temp_extract"
    temp_images_dir.mkdir(parents=True, exist_ok=True)
    temp_db_path = workspace_dir / "temp_extract.db"

    # Copy uncached images to temp directory
    index_to_tempname = {}
    tempname_to_index = {}

    for idx, img_path in uncached_images:
        temp_name = f"{idx:06d}{img_path.suffix}"
        temp_dest = temp_images_dir / temp_name
        import shutil
        shutil.copy2(img_path, temp_dest)
        index_to_tempname[idx] = temp_name
        tempname_to_index[temp_name] = idx

    n_uncached = len(uncached_images)
    print(f"[CACHE] Extracting features for {n_uncached} uncached images (skipping {len([i for i in range(len(uncached_images) + n_uncached)]) - n_uncached} cached)", file=sys.stderr, flush=True)

    # Configure extraction
    sift_options = pycolmap.SiftExtractionOptions()
    sift_options.max_num_features = max_features
    sift_options.first_octave = 0
    sift_options.num_octaves = 4

    extraction_options = pycolmap.FeatureExtractionOptions()
    extraction_options.sift = sift_options

    # Configure device
    if gpu_index >= 0 and has_cuda:
        device = pycolmap.Device(f"cuda:{gpu_index}")
    elif has_cuda:
        device = pycolmap.Device.auto
    else:
        device = pycolmap.Device.cpu

    # Extract features
    extract_start = time.time()
    pycolmap.extract_features(
        temp_db_path,
        temp_images_dir,
        extraction_options=extraction_options,
        camera_mode=pycolmap.CameraMode.PER_IMAGE,
        camera_model="PINHOLE",
        device=device
    )
    extract_time = time.time() - extract_start
    print(f"[CACHE] Extracted {n_uncached} new features in {extract_time:.1f}s ({extract_time/n_uncached:.1f}s per image)", file=sys.stderr, flush=True)

    # Read extracted features from database and save to cache
    conn = sqlite3.connect(temp_db_path)
    cursor = conn.cursor()

    # Get image_id mapping
    cursor.execute("SELECT image_id, name FROM images")
    image_id_map = {name: img_id for img_id, name in cursor.fetchall()}

    index_to_cache_key = {}

    for idx, img_path in uncached_images:
        temp_name = index_to_tempname[idx]
        img_id = image_id_map.get(temp_name)

        if img_id is None:
            print(f"[WARNING] Image {temp_name} not found in database, skipping cache save", file=sys.stderr)
            continue

        # Read keypoints and descriptors from database
        cursor.execute("SELECT data FROM keypoints WHERE image_id = ?", (img_id,))
        kp_row = cursor.fetchone()

        if kp_row is None:
            print(f"[WARNING] No keypoints found for {temp_name}, skipping cache save", file=sys.stderr)
            continue

        cursor.execute("SELECT data FROM descriptors WHERE image_id = ?", (img_id,))
        desc_row = cursor.fetchone()

        if desc_row is None:
            print(f"[WARNING] No descriptors found for {temp_name}, skipping cache save", file=sys.stderr)
            continue

        # Decode keypoints (COLMAP stores as float32 binary blob)
        keypoints_blob = kp_row[0]
        keypoints = np.frombuffer(keypoints_blob, dtype=np.float32).reshape(-1, 6).copy()  # .copy() to own the data

        # Decode descriptors (COLMAP stores as uint8 binary blob, 128 dimensions per keypoint)
        descriptors_blob = desc_row[0]
        descriptors = np.frombuffer(descriptors_blob, dtype=np.uint8).reshape(-1, 128).copy()  # .copy() to own the data

        # Save to cache (non-fatal - continue if this fails)
        cache_key = None
        try:
            metadata = {
                'image_name': img_path.name,
                'extracted_at': time.time(),
                'device': str(device)
            }

            print(f"[CACHE] Saving {len(keypoints)} keypoints + {len(descriptors)} descriptors for {img_path.name}", file=sys.stderr)
            cache_key = cache_manager.save_features(img_path, max_features, keypoints, descriptors, metadata)
            print(f"[CACHE] ✓ Saved features for {img_path.name}: {cache_key[:16]}...", file=sys.stderr)
        except Exception as cache_err:
            print(f"[WARNING] Failed to save features to cache for {img_path.name}: {cache_err}", file=sys.stderr)
            print(f"[WARNING] Continuing without caching this image", file=sys.stderr)
            # Continue without caching - don't fail the whole pipeline

        # Store keypoints and descriptors in memory regardless of whether cache save succeeded
        index_to_cache_key[idx] = {
            'cache_key': cache_key,  # May be None if save failed
            'keypoints': keypoints,  # Always available
            'descriptors': descriptors,  # Always available
            'image_path': img_path
        }

    conn.close()

    # Cleanup temp files
    import shutil
    shutil.rmtree(temp_images_dir, ignore_errors=True)
    if temp_db_path.exists():
        temp_db_path.unlink()

    return index_to_cache_key


def build_database_from_cache(database_path: Path, image_paths: list, cached_features: dict,
                              newly_extracted: dict, cache_manager, images_dir: Path):
    """
    Build COLMAP database from mix of cached and newly extracted features.

    Args:
        database_path: Path to output COLMAP database
        image_paths: List of all image paths
        cached_features: Dict from check_feature_cache() {index: {'cache_key': str, 'cache_entry': dict}}
        newly_extracted: Dict from extract_features_incremental() {index: cache_key}
        cache_manager: COLMAPCacheManager instance
        images_dir: Directory where images were copied

    Returns:
        True if successful
    """
    import pycolmap
    import sqlite3
    from PIL import Image

    print(f"[CACHE] Building database from cache: {len(cached_features)} cached + {len(newly_extracted)} new", file=sys.stderr, flush=True)

    # NEW APPROACH: Create database using pycolmap's own extraction on a single image,
    # then replace all features with our cached ones. This ensures correct internal structure.

    # Remove old database
    if database_path.exists():
        database_path.unlink()

    # Create a minimal temporary workspace with just one image to initialize the database
    print(f"[CACHE] Initializing database structure via pycolmap extraction on first image...", file=sys.stderr, flush=True)

    temp_init_dir = images_dir.parent / "db_init"
    temp_init_dir.mkdir(exist_ok=True)

    try:
        # Copy first image to temp directory for initialization
        first_img = image_paths[0]
        temp_img_path = temp_init_dir / f"000000{first_img.suffix}"
        shutil.copy2(first_img, temp_img_path)

        # Run pycolmap feature extraction on just this one image to create database
        import pycolmap

        # Create SIFT options
        sift_opts = pycolmap.SiftExtractionOptions()
        sift_opts.max_num_features = max_features
        sift_opts.first_octave = -1
        sift_opts.num_octaves = 4
        sift_opts.octave_resolution = 3

        # Create FeatureExtractionOptions and set SIFT options
        extraction_opts = pycolmap.FeatureExtractionOptions()
        extraction_opts.sift = sift_opts

        print(f"[CACHE] Running pycolmap.extract_features to create database structure...", file=sys.stderr, flush=True)
        pycolmap.extract_features(
            str(database_path),
            str(temp_init_dir),
            extraction_options=extraction_opts,
            camera_mode=pycolmap.CameraMode.SINGLE,
            camera_model="PINHOLE"
        )

        print(f"[CACHE] Database initialized, now replacing features with cached data...", file=sys.stderr, flush=True)

    except Exception as e:
        print(f"[ERROR] Failed to initialize database via pycolmap: {e}", file=sys.stderr)
        print(f"[ERROR] Falling back to manual database creation", file=sys.stderr)
        # Continue with manual creation as fallback

    finally:
        # Clean up temp directory
        if temp_init_dir.exists():
            shutil.rmtree(temp_init_dir, ignore_errors=True)

    # Now open the database and clear the initialization data, then populate with our cached features
    conn = sqlite3.connect(str(database_path))
    cursor = conn.cursor()

    # Ensure tables exist (in case fallback was used)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cameras (
            camera_id INTEGER PRIMARY KEY,
            model INTEGER NOT NULL,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            params BLOB NOT NULL,
            prior_focal_length INTEGER NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS images (
            image_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            camera_id INTEGER NOT NULL,
            prior_qw REAL,
            prior_qx REAL,
            prior_qy REAL,
            prior_qz REAL,
            prior_tx REAL,
            prior_ty REAL,
            prior_tz REAL,
            FOREIGN KEY(camera_id) REFERENCES cameras(camera_id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS keypoints (
            image_id INTEGER PRIMARY KEY,
            rows INTEGER NOT NULL,
            cols INTEGER NOT NULL,
            data BLOB NOT NULL,
            FOREIGN KEY(image_id) REFERENCES images(image_id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS descriptors (
            image_id INTEGER PRIMARY KEY,
            rows INTEGER NOT NULL,
            cols INTEGER NOT NULL,
            data BLOB NOT NULL,
            FOREIGN KEY(image_id) REFERENCES images(image_id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS matches (
            pair_id INTEGER PRIMARY KEY,
            rows INTEGER NOT NULL,
            cols INTEGER NOT NULL,
            data BLOB NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS two_view_geometries (
            pair_id INTEGER PRIMARY KEY,
            rows INTEGER NOT NULL,
            cols INTEGER NOT NULL,
            data BLOB NOT NULL,
            config INTEGER NOT NULL,
            F BLOB,
            E BLOB,
            H BLOB
        )
    """)

    # Create indexes that COLMAP expects
    cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS index_images_name ON images(name)")
    cursor.execute("CREATE INDEX IF NOT EXISTS index_cameras_model ON cameras(model)")
    cursor.execute("CREATE INDEX IF NOT EXISTS index_keypoints_image_id ON keypoints(image_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS index_descriptors_image_id ON descriptors(image_id)")

    conn.commit()

    # Don't delete the pycolmap-initialized database! Instead, keep it and add our images to it.
    # First, check what pycolmap created
    cursor.execute("SELECT COUNT(*) FROM images")
    pycolmap_image_count = cursor.fetchone()[0]
    print(f"[CACHE] Database has {pycolmap_image_count} images from pycolmap initialization", file=sys.stderr, flush=True)

    # Get the camera_id from the pycolmap-created camera (we'll reuse it if dimensions match)
    cursor.execute("SELECT camera_id, width, height FROM cameras LIMIT 1")
    pycolmap_camera = cursor.fetchone()

    camera_id_map = {}
    if pycolmap_camera:
        pycolmap_camera_id, pycolmap_width, pycolmap_height = pycolmap_camera
        camera_key = (pycolmap_width, pycolmap_height)
        camera_id_map[camera_key] = pycolmap_camera_id
        print(f"[CACHE] Reusing pycolmap camera {pycolmap_camera_id}: {pycolmap_width}x{pycolmap_height}", file=sys.stderr, flush=True)

    # Now add/update images and their features
    print(f"[CACHE] Populating database with {len(image_paths)} images...", file=sys.stderr, flush=True)

    # PINHOLE camera model (model = 1 in COLMAP)
    camera_model = 1

    for i, img_path in enumerate(image_paths):
        # Get image dimensions
        with Image.open(img_path) as img:
            width, height = img.size

        # Get or create camera
        camera_key = (width, height)
        if camera_key not in camera_id_map:
            # PINHOLE camera model parameters: fx, fy, cx, cy
            focal_length = max(width, height) * 1.2
            params = np.array([focal_length, focal_length, width / 2, height / 2], dtype=np.float64)
            params_blob = params.tobytes()

            camera_id = len(camera_id_map) + 1
            cursor.execute("""
                INSERT INTO cameras (camera_id, model, width, height, params, prior_focal_length)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (camera_id, camera_model, width, height, params_blob, 0))

            camera_id_map[camera_key] = camera_id
            print(f"[CACHE] Created camera {camera_id}: {width}x{height}, f={focal_length:.1f}px", file=sys.stderr)
        else:
            camera_id = camera_id_map[camera_key]

        # Add/update image (use INSERT OR REPLACE to update if image_id=1 already exists from pycolmap)
        image_name = f"{i:06d}{img_path.suffix}"
        cursor.execute("""
            INSERT OR REPLACE INTO images (image_id, name, camera_id)
            VALUES (?, ?, ?)
        """, (i + 1, image_name, camera_id))

    conn.commit()
    print(f"[CACHE] Added/updated {len(camera_id_map)} cameras and {len(image_paths)} images", file=sys.stderr, flush=True)

    # Insert/update keypoints and descriptors for each image
    # For image_id=1 (pycolmap-initialized), use UPDATE to preserve rowid structure
    # For other images, use INSERT
    print(f"[CACHE] Populating keypoints and descriptors for {len(image_paths)} images...", file=sys.stderr, flush=True)

    for i, img_path in enumerate(image_paths):

        # Get keypoints and descriptors from cache or memory
        keypoints = None
        descriptors = None

        if i in newly_extracted:
            # Newly extracted - prefer in-memory data, fallback to cache
            extraction_data = newly_extracted[i]

            if 'keypoints' in extraction_data and 'descriptors' in extraction_data:
                # Use data from memory (always available)
                keypoints = extraction_data['keypoints']
                descriptors = extraction_data['descriptors']
                print(f"[CACHE] Using in-memory features for image {i} ({len(keypoints)} keypoints)", file=sys.stderr)
            elif extraction_data.get('cache_key'):
                # Try loading from cache if available
                try:
                    keypoints, descriptors, metadata = cache_manager.load_features(extraction_data['cache_key'])
                    print(f"[CACHE] Loaded from cache for image {i} ({len(keypoints)} keypoints)", file=sys.stderr)
                except Exception as e:
                    print(f"[WARNING] Failed to load newly extracted features for image {i}: {e}", file=sys.stderr)

        elif i in cached_features:
            # Load from existing cache
            cache_key = cached_features[i]['cache_key']
            try:
                keypoints, descriptors, metadata = cache_manager.load_features(cache_key)
                print(f"[CACHE] Loaded from cache for image {i} ({len(keypoints)} keypoints)", file=sys.stderr)
            except Exception as e:
                print(f"[WARNING] Failed to load cached features for image {i}: {e}", file=sys.stderr)

        if keypoints is None or descriptors is None:
            print(f"[ERROR] No features available for image {i}, database will be incomplete", file=sys.stderr)
            continue

        # Prepare blobs
        keypoints_blob = keypoints.astype(np.float32).tobytes()
        n_keypoints = len(keypoints)
        descriptors_blob = descriptors.astype(np.uint8).tobytes()
        n_descriptors = len(descriptors)

        # Validate counts match
        if n_keypoints != n_descriptors:
            print(f"[WARNING] Keypoint/descriptor count mismatch for image {i}: {n_keypoints} != {n_descriptors}", file=sys.stderr)

        # For image_id=1, UPDATE to preserve pycolmap's rowid structure
        # For others, INSERT new rows
        if i == 0:
            # Update the pycolmap-initialized image 1
            cursor.execute("""
                UPDATE keypoints
                SET rows = ?, cols = ?, data = ?
                WHERE image_id = 1
            """, (n_keypoints, 6, keypoints_blob))

            cursor.execute("""
                UPDATE descriptors
                SET rows = ?, cols = ?, data = ?
                WHERE image_id = 1
            """, (n_descriptors, 128, descriptors_blob))
            print(f"[CACHE] Updated image 1 features ({n_keypoints} keypoints)", file=sys.stderr)
        else:
            # Insert new images
            cursor.execute("""
                INSERT INTO keypoints (image_id, rows, cols, data)
                VALUES (?, ?, ?, ?)
            """, (i + 1, n_keypoints, 6, keypoints_blob))

            cursor.execute("""
                INSERT INTO descriptors (image_id, rows, cols, data)
                VALUES (?, ?, ?, ?)
            """, (i + 1, n_descriptors, 128, descriptors_blob))
            print(f"[CACHE] Inserted image {i+1} features ({n_keypoints} keypoints)", file=sys.stderr)

    # First commit the INSERT transaction
    conn.commit()
    print(f"[CACHE] Data committed to database", file=sys.stderr, flush=True)

    # Get raw file descriptor for explicit fsync
    import os
    db_fd = os.open(str(database_path), os.O_RDWR)
    os.fsync(db_fd)  # Force kernel to write all buffered data to disk
    os.close(db_fd)
    print(f"[CACHE] Database file synced to disk", file=sys.stderr, flush=True)

    # Now close the connection
    conn.close()

    # Critical: Give WSL filesystem time to flush writes to Windows
    # Large databases (16MB+) need more time for WSL->Windows bridge
    import time
    time.sleep(1.5)

    print(f"[CACHE] Waiting for WSL filesystem bridge...", file=sys.stderr, flush=True)

    # Verify database file exists and has content
    if database_path.exists():
        db_size = database_path.stat().st_size
        print(f"[CACHE] Database built successfully: {database_path} ({db_size / 1024:.1f} KB, {len(image_paths)} images)", file=sys.stderr, flush=True)
    else:
        print(f"[ERROR] Database file not found after building: {database_path}", file=sys.stderr)
        return False

    return True


def run_2d_stitch(image_paths_json: str, output_dir: str, use_affine: bool = False, blend_method: str = "multiband",
                  matcher_type: str = "exhaustive", sequential_overlap: int = 10,
                  gpu_index: int = -1, num_threads: int = -1, max_features: int = 8192,
                  min_inliers: int = 0, max_images: int = 0, use_source_alpha: bool = False,
                  remove_duplicates: bool = False, duplicate_threshold: float = 0.92,
                  warp_interpolation: str = 'linear',
                  erode_border: bool = True, border_erosion_pixels: int = 5) -> dict:
    """
    Run GPU-accelerated 2D stitching using pycolmap.

    Args:
        image_paths_json: JSON string of Windows image paths
        output_dir: Windows output directory path
        use_affine: Use affine transforms instead of homography
        blend_method: Blending method (multiband, feather, autostitch, linear)
        matcher_type: Matching strategy (exhaustive, sequential, vocab_tree, grid)
        sequential_overlap: Overlap for sequential matching
        gpu_index: GPU index to use (-1 = auto)
        num_threads: Number of threads (-1 = auto)
        max_features: Maximum SIFT features per image
        min_inliers: Minimum inliers to include image (0 = no filter)
        max_images: Maximum images in panorama (0 = no limit)
        use_source_alpha: Use source image alpha channel for transparent backgrounds
        remove_duplicates: Remove duplicate images before processing
        duplicate_threshold: Similarity threshold for duplicate detection (0.0-1.0)
        warp_interpolation: Interpolation method (linear, cubic, lanczos, realesrgan)
        erode_border: Erode alpha mask to remove black borders from circular images
        border_erosion_pixels: Number of pixels to erode from alpha mask edge

    Returns:
        dict with results
    """
    import pycolmap
    import numpy as np
    import signal

    # Set up signal handler for graceful shutdown on Ctrl+C
    def signal_handler(sig, frame):
        print(f"[DEBUG] Received signal {sig}, shutting down gracefully...", file=sys.stderr, flush=True)
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Log received parameters at entry point
    print(f"[DEBUG] WSL bridge called with: max_features={max_features}, matcher_type={matcher_type}, blend_method={blend_method}", file=sys.stderr, flush=True)
    import cv2
    import shutil
    import sqlite3
    import time

    start_time = time.time()  # Track total time

    # Map interpolation string to OpenCV flag
    use_realesrgan = (warp_interpolation.lower() == 'realesrgan')

    interp_map = {
        'nearest': cv2.INTER_NEAREST,
        'linear': cv2.INTER_LINEAR,
        'cubic': cv2.INTER_CUBIC,
        'lanczos': cv2.INTER_LANCZOS4,
        'realesrgan': cv2.INTER_LANCZOS4  # Use lanczos for initial warp, then upscale
    }
    interp_flag = interp_map.get(warp_interpolation.lower(), cv2.INTER_LINEAR)

    # Log warping settings
    if warp_interpolation != 'linear':
        print(f"[DEBUG] Warp interpolation: {warp_interpolation}", file=sys.stderr, flush=True)

    # Parse input
    image_paths_win = json.loads(image_paths_json)
    image_paths = [Path(windows_to_wsl_path(p)) for p in image_paths_win]
    output_path = Path(windows_to_wsl_path(output_dir))
    n_images_original = len(image_paths)

    # Track filtering stats for final report
    n_duplicates_removed = 0

    # Optional duplicate removal before processing
    if remove_duplicates and n_images_original > 1:
        # Time estimate: ~0.05-0.1s per image for hashing
        est_dup_time = n_images_original * 0.08
        dup_time_str = f"est. {est_dup_time:.0f}s" if est_dup_time < 60 else f"est. {est_dup_time/60:.1f}min"
        print(f"[PROGRESS] Scanning {n_images_original} images for duplicates (threshold={duplicate_threshold:.2f})... [{dup_time_str}]", file=sys.stderr, flush=True)
        dup_start = time.time()

        try:
            # Add parent directory to path for imports
            script_dir = Path(__file__).resolve().parent.parent
            if str(script_dir) not in sys.path:
                sys.path.insert(0, str(script_dir))

            from utils.duplicate_detector import DuplicateDetector

            # Progress callback that prints to stderr for GUI capture
            def dup_progress(percent: int, message: str):
                print(f"[PROGRESS] Dedup: {message}", file=sys.stderr, flush=True)

            # Run duplicate detection with imagehash + SQLite caching
            # Pass paths so imagehash can read directly from disk and use cache
            detector = DuplicateDetector(
                similarity_threshold=duplicate_threshold,
                comparison_window=0,  # Compare all pairs for unsorted images
                progress_callback=dup_progress
            )

            # Pass paths for imagehash caching (thumbnails only used as fallback)
            # imagehash reads directly from disk which is faster for cached runs
            keep_indices, duplicate_pairs = detector.find_duplicates([None] * n_images_original, image_paths)

            # Filter image paths
            n_removed = n_images_original - len(keep_indices)
            n_duplicates_removed = n_removed
            if n_removed > 0:
                image_paths = [image_paths[i] for i in keep_indices]
                print(f"[PROGRESS] Removed {n_removed} duplicate images (keeping {len(image_paths)}/{n_images_original})", file=sys.stderr, flush=True)

                # Log which images were removed
                for j, i, sim in duplicate_pairs[:5]:  # Log first 5
                    print(f"[DEBUG] Duplicate: {image_paths_win[i]} ~ {image_paths_win[j]} (similarity={sim:.3f})", file=sys.stderr, flush=True)
                if len(duplicate_pairs) > 5:
                    print(f"[DEBUG] ... and {len(duplicate_pairs) - 5} more duplicates", file=sys.stderr, flush=True)
            else:
                print(f"[PROGRESS] No duplicates found", file=sys.stderr, flush=True)

            dup_elapsed = time.time() - dup_start
            print(f"[DEBUG] Duplicate detection took {dup_elapsed:.1f}s", file=sys.stderr, flush=True)

            # Log cache statistics for debugging
            try:
                cache_stats = detector.cache.get_stats()
                imagehash_status = getattr(detector, '_imagehash_status', 'unknown')
                has_imagehash = getattr(detector, '_has_imagehash', False)

                if has_imagehash:
                    print(f"[DEBUG] Hash cache: {cache_stats['entries']} entries, {cache_stats['db_size_mb']:.2f} MB at {cache_stats['db_path']}", file=sys.stderr, flush=True)
                else:
                    print(f"[WARNING] Hash caching DISABLED - imagehash not available in WSL (status: {imagehash_status})", file=sys.stderr, flush=True)
                    print(f"[WARNING] To enable caching, run in WSL: pip install imagehash Pillow", file=sys.stderr, flush=True)
            except Exception as cache_err:
                print(f"[DEBUG] Could not get cache stats: {cache_err}", file=sys.stderr, flush=True)

        except ImportError as e:
            print(f"[WARNING] Could not import DuplicateDetector: {e}. Skipping duplicate removal.", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"[WARNING] Duplicate detection failed: {e}. Proceeding with all images.", file=sys.stderr, flush=True)
            import traceback
            print(f"[DEBUG] {traceback.format_exc()}", file=sys.stderr, flush=True)

    n_images = len(image_paths)

    print(f"[PROGRESS] WSL COLMAP Bridge - GPU Mode (has_cuda={pycolmap.has_cuda})", file=sys.stderr, flush=True)
    print(f"[PROGRESS] Processing {n_images} images", file=sys.stderr, flush=True)

    # Global cache directory (persistent across runs)
    # Each unique cache (based on images + settings) gets its own subdirectory
    import tempfile
    cache_key = compute_cache_key(
        image_paths,
        max_features,
        matcher_type=matcher_type,
        use_affine=use_affine,
        blend_method=blend_method,
        warp_interpolation=warp_interpolation,
    )
    cache_base = Path(os.environ.get("STITCH2STITCH_CACHE_DIR", Path.home() / ".stitch2stitch_cache"))
    cache_root = cache_base / "colmap"
    cache_root.mkdir(parents=True, exist_ok=True)

    # Use cache key as subdirectory name to allow multiple caches
    global_cache_dir = cache_root / cache_key
    global_cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"[DEBUG] Cache key: {cache_key} (max_features={max_features})", file=sys.stderr, flush=True)
    print(f"[DEBUG] Global cache directory: {global_cache_dir}", file=sys.stderr, flush=True)

    # Workspace on WSL native filesystem to avoid drvfs mmap/locking issues
    workspace_root = Path(tempfile.gettempdir()) / ".stitch2stitch_workspaces"
    workspace_root.mkdir(parents=True, exist_ok=True)

    import time as _time
    workspace = workspace_root / f"{cache_key}_run_{int(_time.time())}"
    if workspace.exists():
        shutil.rmtree(workspace, ignore_errors=True)
    workspace.mkdir(parents=True, exist_ok=True)

    images_dir = workspace / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    database_path = workspace / "database.db"

    # Check cache validity using global cache
    use_cache = is_cache_valid(global_cache_dir, cache_key, n_images)

    if use_cache:
        print(f"[PROGRESS] *** USING CACHED FEATURES (skipping extraction & matching) ***", file=sys.stderr, flush=True)
        print(f"[PROGRESS] Cache key: {cache_key[:12]}...", file=sys.stderr, flush=True)
        print(f"[CACHE] Global cache hit: skipping per-image feature cache and per-pair match cache", file=sys.stderr, flush=True)

        # Copy cached database to workspace
        cached_db = global_cache_dir / "database.db"
        if cached_db.exists():
            shutil.copy2(cached_db, database_path)
            print(f"[DEBUG] Copied cached database to workspace", file=sys.stderr, flush=True)

        # Copy cached images directory to workspace
        cached_images = global_cache_dir / "images"
        if cached_images.exists():
            if images_dir.exists():
                shutil.rmtree(images_dir)
            shutil.copytree(cached_images, images_dir)
            print(f"[DEBUG] Copied {len(list(cached_images.iterdir()))} cached images to workspace", file=sys.stderr, flush=True)
    else:
        print(f"[PROGRESS] Cache invalid or not found - will extract features", file=sys.stderr, flush=True)

        # Clean up old database if exists
        if database_path.exists():
            database_path.unlink()

        # CRITICAL: Clean images directory to avoid processing old images from previous runs
        if images_dir.exists():
            shutil.rmtree(images_dir)
            print(f"[DEBUG] Cleaned old images from workspace", file=sys.stderr, flush=True)

        images_dir.mkdir(exist_ok=True)

    # Build mapping (always needed for index lookup)
    image_name_to_original = {}
    image_name_to_index = {}

    if use_cache:
        # Rebuild mapping from existing images in workspace
        for i, path in enumerate(image_paths):
            dest_name = f"{i:06d}{path.suffix}"
            image_name_to_original[dest_name] = path
            image_name_to_index[dest_name] = i
    else:
        # PARALLEL IMAGE COPYING with ThreadPoolExecutor
        # Estimate: ~0.1s per image for local files, ~0.5s for network files
        est_copy_time = max(5, n_images * 0.1)
        print(f"[PROGRESS] Copying {n_images} images to workspace (parallel)... [est. {est_copy_time:.0f}s]", file=sys.stderr, flush=True)
        copy_start = time.time()

        # Prepare arguments for parallel copy
        copy_args = [(i, path, images_dir) for i, path in enumerate(image_paths)]

        # Use 8 workers for I/O-bound parallel copying
        n_workers = min(8, n_images)
        completed = 0

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(copy_single_image, args): args[0] for args in copy_args}

            for future in as_completed(futures):
                i, dest_name, path = future.result()
                image_name_to_original[dest_name] = path
                image_name_to_index[dest_name] = i
                completed += 1

                # Progress update every 10% or at end
                if completed % max(1, n_images // 10) == 0 or completed == n_images:
                    print(f"[PROGRESS] Copied {completed}/{n_images} images", file=sys.stderr, flush=True)

        copy_time = time.time() - copy_start
        print(f"[PROGRESS] Parallel copy complete in {copy_time:.1f}s ({n_images/copy_time:.1f} images/s)", file=sys.stderr, flush=True)

    # Feature extraction - check GPU availability
    has_cuda = pycolmap.has_cuda if hasattr(pycolmap, 'has_cuda') else False
    print(f"[DEBUG] pycolmap.has_cuda = {has_cuda}", file=sys.stderr, flush=True)

    # Skip extraction and matching if using global cache
    if not use_cache:
        # Initialize per-image cache manager
        per_image_cache_root = cache_root / "per_image"
        per_image_cache_root.mkdir(parents=True, exist_ok=True)

        try:
            # Add src to path if needed for imports
            src_path = Path(__file__).parent.parent
            if str(src_path) not in sys.path:
                sys.path.insert(0, str(src_path))

            from external.colmap_cache_manager import COLMAPCacheManager
            cache_manager = COLMAPCacheManager(per_image_cache_root)
            print(f"[CACHE] Initialized per-image cache at {per_image_cache_root}", file=sys.stderr, flush=True)
            use_per_image_cache = True
        except ImportError as e:
            print(f"[WARNING] Could not import COLMAPCacheManager: {e}. Using traditional extraction.", file=sys.stderr, flush=True)
            cache_manager = None
            use_per_image_cache = False

        # Check per-image cache
        cached_features = {}
        uncached_images = []

        if use_per_image_cache and cache_manager:
            print(f"[CACHE] Checking per-image cache for {n_images} images...", file=sys.stderr, flush=True)
            cached_features, uncached_images = check_feature_cache(image_paths, max_features, cache_manager)
            n_cached = len(cached_features)
            n_uncached = len(uncached_images)
            print(f"[CACHE] Found {n_cached} cached, {n_uncached} need extraction", file=sys.stderr, flush=True)
        else:
            # No per-image cache available, extract all
            uncached_images = list(enumerate(image_paths))

        # Extract features for uncached images only
        newly_extracted = {}

        if uncached_images:
            n_extract = len(uncached_images)

            # Time estimates for feature extraction
            if has_cuda:
                # GPU: ~0.2-0.5s per image depending on resolution and features
                est_extract_time = n_extract * (0.2 + max_features / 40000)
                time_str = f"est. {est_extract_time:.0f}s"
            else:
                # CPU: ~20-40s per image, much slower
                est_extract_time = n_extract * 30
                time_str = f"est. {est_extract_time:.0f}s (SLOW - CPU mode)"
                if n_extract == n_images:  # Only warn if extracting everything
                    print(f"[WARNING] CUDA not available in pycolmap! Feature extraction will be VERY SLOW on CPU.", file=sys.stderr, flush=True)
                    print(f"[WARNING] Expected time: {est_extract_time:.0f}s ({est_extract_time/60:.1f}min)", file=sys.stderr, flush=True)
                    print(f"[SUGGESTION] Install pycolmap-cuda12 in WSL for GPU acceleration", file=sys.stderr, flush=True)

            if use_per_image_cache and cache_manager:
                # Use incremental extraction with caching
                newly_extracted = extract_features_incremental(
                    uncached_images, workspace, database_path, max_features,
                    cache_manager, gpu_index, num_threads, has_cuda
                )
            else:
                # Traditional full extraction (fallback)
                print(f"[PROGRESS] WSL GPU: Extracting features from {n_extract} images (max_features={max_features})... [{time_str}]", file=sys.stderr, flush=True)
                extract_start = time.time()

                # Configure SIFT extraction options
                import pycolmap
                sift_options = pycolmap.SiftExtractionOptions()
                sift_options.max_num_features = max_features
                sift_options.first_octave = 0
                sift_options.num_octaves = 4

                extraction_options = pycolmap.FeatureExtractionOptions()
                extraction_options.sift = sift_options

                # Configure device
                if gpu_index >= 0 and has_cuda:
                    device = pycolmap.Device(f"cuda:{gpu_index}")
                elif has_cuda:
                    device = pycolmap.Device.auto
                else:
                    device = pycolmap.Device.cpu

                print(f"[PROGRESS] Starting feature extraction with {device}...", file=sys.stderr, flush=True)

                pycolmap.extract_features(
                    database_path,
                    images_dir,
                    extraction_options=extraction_options,
                    camera_mode=pycolmap.CameraMode.PER_IMAGE,
                    camera_model="PINHOLE",
                    device=device
                )

                extract_time = time.time() - extract_start
                avg_time = extract_time / n_extract
                print(f"[PROGRESS] Feature extraction complete in {extract_time:.1f}s ({avg_time:.1f}s per image)", file=sys.stderr, flush=True)

        # Build database from cache if using per-image cache
        # NOTE: Always build database when using per-image cache, even if cache saves failed
        # The incremental extraction uses a temp database, so we must rebuild the main one
        if use_per_image_cache and cache_manager:
            print(f"[CACHE] Building database from cache ({len(cached_features)} cached + {len(newly_extracted)} new)...", file=sys.stderr, flush=True)
            build_start = time.time()

            build_database_from_cache(
                database_path, image_paths, cached_features,
                newly_extracted, cache_manager, images_dir
            )

            build_time = time.time() - build_start
            print(f"[CACHE] Database built in {build_time:.1f}s", file=sys.stderr, flush=True)

        # Feature matching
        if use_per_image_cache and cache_manager:
            print(f"[PROGRESS] WSL: Matching via per-pair cache (incremental)", file=sys.stderr, flush=True)
            match_start = time.time()
            match_pairs_incremental(
                database_path=database_path,
                image_paths=image_paths,
                max_features=max_features,
                cache_manager=cache_manager,
                matcher_type=matcher_type,
                sequential_overlap=sequential_overlap,
                use_affine=use_affine,
                cached_features=cached_features,
                newly_extracted=newly_extracted
            )
            match_time = time.time() - match_start
            print(f"[PROGRESS] Feature matching complete in {match_time:.1f}s", file=sys.stderr, flush=True)
        else:
            # Calculate expected number of pairs for progress reporting
            if matcher_type == "sequential":
                expected_pairs = min(n_images * sequential_overlap, n_images * (n_images - 1) // 2)
            else:  # exhaustive
                expected_pairs = n_images * (n_images - 1) // 2

            # Time estimates for matching
            if has_cuda:
                # GPU: ~0.001-0.01s per pair depending on features
                est_match_time = expected_pairs * 0.005
            else:
                # CPU: ~0.05-0.1s per pair
                est_match_time = expected_pairs * 0.08

            # Format time nicely
            if est_match_time < 120:
                time_str = f"est. {est_match_time:.0f}s"
            elif est_match_time < 3600:
                time_str = f"est. {est_match_time/60:.1f}min"
            else:
                time_str = f"est. {est_match_time/3600:.1f}h"

            print(f"[PROGRESS] WSL GPU: Matching features using {matcher_type} matching ({expected_pairs:,} pairs)... [{time_str}]", file=sys.stderr, flush=True)
            match_start = time.time()

            # Start database progress monitoring for matching
            import threading
            matching_complete = threading.Event()

            def monitor_matching_progress():
                """Monitor database for feature matching progress."""
                last_count = 0
                last_report_time = time.time()
                while not matching_complete.is_set():
                    try:
                        conn = sqlite3.connect(database_path)
                        cursor = conn.execute("SELECT COUNT(*) FROM matches")
                        count = cursor.fetchone()[0]
                        conn.close()

                        # Report every 1000 pairs or every 2 seconds
                        if count > last_count and (count - last_count >= 1000 or time.time() - last_report_time >= 2.0):
                            pct = int(100 * count / expected_pairs) if expected_pairs > 0 else 0
                            print(f"[PROGRESS] Feature matching: {count:,}/{expected_pairs:,} pairs ({pct}%)", file=sys.stderr, flush=True)
                            last_count = count
                            last_report_time = time.time()
                    except:
                        pass
                    time.sleep(0.5)  # Check every 500ms

            match_progress_thread = threading.Thread(target=monitor_matching_progress, daemon=True)
            match_progress_thread.start()

            if matcher_type == "sequential":
                print(f"[PROGRESS] Sequential matching with overlap={sequential_overlap}", file=sys.stderr, flush=True)
                pycolmap.match_sequential(database_path, overlap=sequential_overlap)
            elif matcher_type == "vocab_tree":
                print(f"[PROGRESS] Vocab tree matching not implemented, falling back to exhaustive", file=sys.stderr, flush=True)
                pycolmap.match_exhaustive(database_path)
            elif matcher_type == "grid":
                print(f"[PROGRESS] Grid matching not implemented, falling back to sequential", file=sys.stderr, flush=True)
                pycolmap.match_sequential(database_path, overlap=sequential_overlap)
            else:  # exhaustive
                print(f"[PROGRESS] Exhaustive matching - {expected_pairs:,} image pairs...", file=sys.stderr, flush=True)
                pycolmap.match_exhaustive(database_path)

            # Stop progress monitoring
            matching_complete.set()
            match_progress_thread.join(timeout=1.0)

            match_time = time.time() - match_start
            print(f"[PROGRESS] Feature matching complete in {match_time:.1f}s", file=sys.stderr, flush=True)

        # Save to global cache for reuse across different output directories
        print(f"[PROGRESS] Saving features to global cache...", file=sys.stderr, flush=True)
        cache_save_start = time.time()

        # Copy database to global cache
        cached_db = global_cache_dir / "database.db"
        shutil.copy2(database_path, cached_db)

        # Copy images directory to global cache
        cached_images = global_cache_dir / "images"
        if cached_images.exists():
            shutil.rmtree(cached_images)
        shutil.copytree(images_dir, cached_images)

        # Save cache key
        save_cache_key(global_cache_dir, cache_key)

        cache_save_time = time.time() - cache_save_start
        print(f"[PROGRESS] Cache saved in {cache_save_time:.1f}s (will be reused for future runs with same images)", file=sys.stderr, flush=True)

    # Read matches and compute homographies (always run - reads from database)
    print("[PROGRESS] Reading matches from database...", file=sys.stderr, flush=True)
    matches_data = read_matches_from_db(database_path)
    
    if not matches_data:
        return {"success": False, "error": "No matches found"}
    
    print(f"[PROGRESS] Found {len(matches_data)} image pairs with matches", file=sys.stderr, flush=True)

    # MEMORY OPTIMIZATION: Don't load all images upfront
    # Only store paths and load images lazily during warping
    print(f"[PROGRESS] Preparing {n_images} images for stitching (lazy loading)...", file=sys.stderr, flush=True)
    if use_source_alpha:
        print(f"[PROGRESS] Will use source alpha channels (for transparent backgrounds)", file=sys.stderr, flush=True)

    # Compute transforms (homography or affine) - doesn't need image data, only matches
    transform_type = "affine" if use_affine else "homography"
    # Time estimate: ~0.01s per match pair for RANSAC
    n_match_pairs = len(matches_data)
    est_transform_time = n_match_pairs * 0.01
    if est_transform_time < 60:
        transform_time_str = f"est. {est_transform_time:.0f}s"
    else:
        transform_time_str = f"est. {est_transform_time/60:.1f}min"

    print(f"[PROGRESS] Computing {transform_type} transforms from {n_match_pairs:,} matches... [{transform_time_str}]", file=sys.stderr, flush=True)
    homographies = compute_homographies(matches_data, n_images, image_name_to_index, use_affine)

    if not homographies:
        transform_type = "affine transforms" if use_affine else "homographies"
        return {"success": False, "error": f"Could not compute {transform_type}"}

    transform_type = "affine transforms" if use_affine else "homographies"
    print(f"[PROGRESS] Computed {len(homographies)} valid {transform_type}", file=sys.stderr, flush=True)

    # Apply image filtering if requested - only filters indices, doesn't need image data
    if min_inliers > 0 or max_images > 0:
        image_paths, homographies = filter_images(image_paths, homographies, min_inliers, max_images, n_images)
        print(f"[PROGRESS] After filtering: {len(image_paths)} images, {len(homographies)} {transform_type}", file=sys.stderr, flush=True)

        if not homographies:
            return {"success": False, "error": "No images passed filtering criteria"}

    # Stitch - images loaded lazily during warping
    # Time estimate: ~0.5-2s per image for warping + blending time
    n_valid_images = len([h for h in homographies.values() if h is not None])
    est_stitch_time = n_valid_images * 1.5
    if est_stitch_time < 120:
        stitch_time_str = f"est. {est_stitch_time:.0f}s"
    else:
        stitch_time_str = f"est. {est_stitch_time/60:.1f}min"

    print(f"[PROGRESS] Stitching panorama (warping and blending {n_valid_images} images)... [{stitch_time_str}]", file=sys.stderr, flush=True)
    stitch_start = time.time()
    panorama = stitch_with_homographies(
        image_paths, homographies, use_affine, blend_method, use_source_alpha,
        warp_interpolation, erode_border, border_erosion_pixels, log_dir=output_path
    )
    stitch_time = time.time() - stitch_start
    
    if panorama is None:
        return {"success": False, "error": "Stitching failed"}
    
    print(f"[PROGRESS] Stitching complete in {stitch_time:.1f}s", file=sys.stderr, flush=True)
    print(f"[PROGRESS] Panorama size: {panorama.shape[1]}x{panorama.shape[0]} pixels", file=sys.stderr, flush=True)
    
    # Save
    output_file = output_path / "colmap_panorama.tiff"
    print(f"[PROGRESS] Saving panorama to {output_file}...", file=sys.stderr, flush=True)
    cv2.imwrite(str(output_file), panorama)
    
    total_time = time.time() - start_time
    print(f"[PROGRESS] Complete! Total time: {total_time:.1f}s", file=sys.stderr, flush=True)

    return {
        "success": True,
        "output_path": wsl_to_windows_path(str(output_file)),
        "n_images": len(image_paths),
        "n_images_original": n_images_original,
        "n_duplicates_removed": n_duplicates_removed,
        "size": list(panorama.shape[:2]),
        "gpu_used": pycolmap.has_cuda
    }


def read_matches_from_db(database_path: Path) -> dict:
    """Read keypoints and matches from COLMAP database."""
    import sqlite3
    import numpy as np
    
    conn = sqlite3.connect(str(database_path))
    cursor = conn.cursor()
    
    cursor.execute("SELECT image_id, name FROM images")
    image_id_to_name = {row[0]: row[1] for row in cursor.fetchall()}
    
    keypoints = {}
    cursor.execute("SELECT image_id, rows, cols, data FROM keypoints")
    for row in cursor.fetchall():
        image_id, rows, cols, data = row
        if data:
            arr = np.frombuffer(data, dtype=np.float32).reshape(rows, cols)
            keypoints[image_id] = arr[:, :2]
    
    matches = {}
    cursor.execute("SELECT pair_id, rows, cols, data FROM two_view_geometries WHERE rows > 0")
    
    for row in cursor.fetchall():
        pair_id, n_rows, cols, data = row
        if data and n_rows > 0:
            image_id2 = pair_id % 2147483647
            image_id1 = pair_id // 2147483647
            
            if image_id1 > image_id2:
                image_id1, image_id2 = image_id2, image_id1
            
            match_arr = np.frombuffer(data, dtype=np.uint32).reshape(n_rows, cols)
            
            name1 = image_id_to_name.get(image_id1)
            name2 = image_id_to_name.get(image_id2)
            
            if name1 and name2 and image_id1 in keypoints and image_id2 in keypoints:
                kp1 = keypoints[image_id1]
                kp2 = keypoints[image_id2]
                
                valid_mask = (match_arr[:, 0] < kp1.shape[0]) & (match_arr[:, 1] < kp2.shape[0])
                valid_matches = match_arr[valid_mask]
                
                if len(valid_matches) > 0:
                    matches[(name1, name2)] = {
                        'kp1': kp1,
                        'kp2': kp2,
                        'matches': valid_matches
                    }
    
    conn.close()
    return matches


def filter_images(image_paths: list, homographies: dict, min_inliers: int = 0, max_images: int = 0, n_images: int = None):
    """
    Filter images based on quality metrics to reduce overlap blur.

    Args:
        image_paths: List of image paths (not loaded images - for memory efficiency)
        homographies: Dict of (idx1, idx2) -> {H, n_inliers, n_matches}
        min_inliers: Minimum inliers required (0 = no filter)
        max_images: Maximum images to keep (0 = no limit)
        n_images: Total number of images (if different from len(image_paths))

    Returns:
        Filtered (image_paths, homographies)
    """
    import numpy as np

    if n_images is None:
        n_images = len(image_paths)

    # First filter by min_inliers if specified
    filtered_homographies = {}
    if min_inliers > 0:
        for key, data in homographies.items():
            if data['n_inliers'] >= min_inliers:
                filtered_homographies[key] = data
        print(f"[FILTER] Min inliers filter: {len(filtered_homographies)}/{len(homographies)} pairs kept (threshold={min_inliers})", file=sys.stderr, flush=True)
    else:
        filtered_homographies = homographies.copy()

    # Determine which images to keep based on connectivity
    if max_images > 0 and n_images > max_images:
        # Score images by total inliers across all their connections
        image_scores = {}
        for (i, j), data in filtered_homographies.items():
            inliers = data['n_inliers']
            image_scores[i] = image_scores.get(i, 0) + inliers
            image_scores[j] = image_scores.get(j, 0) + inliers

        # Also count connectivity (number of connections)
        connectivity = {}
        for (i, j) in filtered_homographies.keys():
            connectivity[i] = connectivity.get(i, 0) + 1
            connectivity[j] = connectivity.get(j, 0) + 1

        # Combined score: inliers + connectivity bonus
        combined_scores = {}
        for idx in image_scores:
            combined_scores[idx] = image_scores[idx] + (connectivity.get(idx, 0) * 100)

        # Sort by score and keep top max_images
        sorted_indices = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)
        indices_to_keep = set(sorted_indices[:max_images])

        print(f"[FILTER] Max images filter: Keeping top {len(indices_to_keep)} images by quality", file=sys.stderr, flush=True)

        # Filter image paths list
        filtered_paths = [path for idx, path in enumerate(image_paths) if idx in indices_to_keep]

        # Filter homographies to only include kept images
        final_homographies = {}
        for (i, j), data in filtered_homographies.items():
            if i in indices_to_keep and j in indices_to_keep:
                # Remap indices
                new_i = sorted(indices_to_keep).index(i)
                new_j = sorted(indices_to_keep).index(j)
                final_homographies[(new_i, new_j)] = data

        return filtered_paths, final_homographies

    return image_paths, filtered_homographies


def compute_homographies(matches_data: dict, n_images: int, image_name_to_index: dict, use_affine: bool = False) -> dict:
    """Compute homographies from matches.

    Note: This function doesn't need actual image data, only keypoints from matches.
    """
    import cv2
    import numpy as np

    homographies = {}
    total_pairs = len(matches_data)
    transform_type = "affine" if use_affine else "homography"
    print(f"[PROGRESS] Computing {transform_type} transforms for {total_pairs} image pairs...", file=sys.stderr, flush=True)

    for pair_idx, ((name1, name2), data) in enumerate(matches_data.items()):
        # Progress update every 10% or every 100 pairs
        if pair_idx % max(1, total_pairs // 10) == 0 or pair_idx % 100 == 0:
            pct = (pair_idx / total_pairs) * 100
            print(f"[PROGRESS] Transform estimation: {pair_idx}/{total_pairs} pairs ({pct:.0f}%)", file=sys.stderr, flush=True)
        idx1 = image_name_to_index.get(name1)
        idx2 = image_name_to_index.get(name2)
        
        if idx1 is None or idx2 is None:
            continue
        
        kp1 = data['kp1']
        kp2 = data['kp2']
        match_indices = data['matches']
        
        if len(match_indices) < 4:
            continue
        
        pts1 = kp1[match_indices[:, 0]]
        pts2 = kp2[match_indices[:, 1]]

        if use_affine:
            # Compute affine transform (2x3 matrix)
            M, mask = cv2.estimateAffine2D(pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=5.0)

            if M is not None:
                # Convert 2x3 to 3x3 for consistency
                H = np.vstack([M, [0, 0, 1]])
                n_inliers = np.sum(mask) if mask is not None else 0
                if n_inliers >= 10:
                    homographies[(idx1, idx2)] = {
                        'H': H,
                        'n_matches': len(match_indices),
                        'n_inliers': n_inliers,
                        'is_affine': True
                    }
        else:
            # Compute homography (3x3 matrix)
            H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)

            if H is not None:
                n_inliers = np.sum(mask) if mask is not None else 0
                if n_inliers >= 10:
                    homographies[(idx1, idx2)] = {
                        'H': H,
                        'n_matches': len(match_indices),
                        'n_inliers': n_inliers,
                        'is_affine': False
                    }

    print(f"[PROGRESS] Transform estimation complete: {len(homographies)} valid transforms from {total_pairs} pairs", file=sys.stderr, flush=True)
    return homographies


def stitch_with_homographies(image_paths: list, homographies: dict, use_affine: bool = False,
                             blend_method: str = "multiband", use_source_alpha: bool = False,
                             warp_interpolation: str = 'linear',
                             erode_border: bool = True, border_erosion_pixels: int = 5,
                             log_dir=None):
    """Stitch images using transforms (homography or affine) - optimized for ordered sequential captures.

    MEMORY OPTIMIZATION: Images are loaded lazily during warping to avoid OOM with large sets.

    Args:
        image_paths: List of Path objects to image files
        homographies: Dictionary of image pair transforms
        use_affine: Use affine transform instead of homography
        blend_method: Blending method (multiband, feather, autostitch, linear)
        use_source_alpha: If True, preserve alpha channel from source images (for transparent backgrounds)
        warp_interpolation: Interpolation method (linear, cubic, lanczos, realesrgan)
        erode_border: If True, erode alpha mask to remove black borders
        border_erosion_pixels: Number of pixels to erode from alpha mask edge
        log_dir: Directory to write diagnostic logs (accessible from Windows)
    """
    import cv2
    import numpy as np
    import traceback

    # Map interpolation string to OpenCV flag
    use_realesrgan = (warp_interpolation.lower() == 'realesrgan')
    interp_map = {
        'nearest': cv2.INTER_NEAREST,
        'linear': cv2.INTER_LINEAR,
        'cubic': cv2.INTER_CUBIC,
        'lanczos': cv2.INTER_LANCZOS4,
        'realesrgan': cv2.INTER_LANCZOS4  # Use lanczos for initial warp, then upscale
    }
    interp_flag = interp_map.get(warp_interpolation.lower(), cv2.INTER_LINEAR)

    # #region agent log - H1: Entry point
    n_images = len(image_paths)
    print(f"[DEBUG] stitch_with_homographies entry: n_images={n_images}, n_homographies={len(homographies)}", file=sys.stderr, flush=True)
    # #endregion

    if n_images == 0:
        return None
    if n_images == 1:
        # Single image - just load and return it
        img = cv2.imread(str(image_paths[0]), cv2.IMREAD_UNCHANGED if use_source_alpha else cv2.IMREAD_COLOR)
        if img is not None and img.ndim == 3 and img.shape[2] == 4:
            # Convert BGRA to BGR for output
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img
    
    # Build connectivity and find reference - use middle image for sequential captures
    connections = {}
    for (i, j) in homographies.keys():
        connections[i] = connections.get(i, 0) + 1
        connections[j] = connections.get(j, 0) + 1

    if not connections:
        # No connections found - just return first image
        img = cv2.imread(str(image_paths[0]), cv2.IMREAD_UNCHANGED if use_source_alpha else cv2.IMREAD_COLOR)
        if img is not None and img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    # Find connected components using BFS
    def find_connected_components(nodes, edges):
        """Find all connected components in the graph."""
        visited = set()
        components = []

        for start_node in nodes:
            if start_node in visited:
                continue

            # BFS to find component
            component = set()
            queue = [start_node]

            while queue:
                node = queue.pop(0)
                if node in visited:
                    continue
                visited.add(node)
                component.add(node)

                # Add neighbors
                for (i, j) in edges:
                    if i == node and j not in visited:
                        queue.append(j)
                    elif j == node and i not in visited:
                        queue.append(i)

            components.append(component)

        return components

    # Find connected components
    all_nodes = set(connections.keys())
    components = find_connected_components(all_nodes, homographies.keys())

    # Sort by size (largest first)
    components.sort(key=len, reverse=True)

    print(f"[DEBUG] H3: Found {len(components)} connected component(s):", file=sys.stderr, flush=True)
    for i, comp in enumerate(components):
        print(f"[DEBUG] H3:   Component {i}: {len(comp)} images - {sorted(comp)}", file=sys.stderr, flush=True)

    # Use largest component
    largest_component = components[0]
    print(f"[DEBUG] H3: Largest component has {len(largest_component)} images: {sorted(largest_component)}", file=sys.stderr, flush=True)

    # Choose reference from largest component - prefer middle image if in component
    middle_idx = n_images // 2
    ref_idx = middle_idx

    print(f"[DEBUG] H3: Checking if middle image {middle_idx} is in largest component...", file=sys.stderr, flush=True)
    print(f"[DEBUG] H3: largest_component type: {type(largest_component)}, contents: {largest_component}", file=sys.stderr, flush=True)
    print(f"[DEBUG] H3: Is {middle_idx} in largest_component? {middle_idx in largest_component}", file=sys.stderr, flush=True)

    if ref_idx not in largest_component:
        # Middle image not in largest component - find middle of largest component instead
        component_list = sorted(largest_component)
        ref_idx = component_list[len(component_list) // 2]
        print(f"[WARNING] H3: Middle image {middle_idx} not in largest component, using image {ref_idx} instead (middle of largest component)", file=sys.stderr, flush=True)
    else:
        print(f"[DEBUG] H3: Middle image {middle_idx} IS in largest component, using it as reference", file=sys.stderr, flush=True)
    
    # #region agent log - H3: Check connectivity
    print(f"[DEBUG] H3: ref_idx={ref_idx} (middle={n_images//2}), connections={connections}", file=sys.stderr, flush=True)
    print(f"[DEBUG] H3: Feature match pairs: {list(homographies.keys())}", file=sys.stderr, flush=True)
    for (i, j), data in homographies.items():
        print(f"[DEBUG] H3: Match ({i},{j}): {data['n_inliers']} inliers from {data['n_matches']} matches", file=sys.stderr, flush=True)
    # #endregion
    
    # Build adjacency graph - only include edges within largest component
    # H in homographies[(i,j)] transforms points from image i to image j: p_j = H @ p_i
    adjacency = {}
    filtered_homographies = {}

    # Minimum inlier threshold - weak matches produce unreliable (near-identity) homographies
    # that cause images to stack on top of each other instead of spreading across the panorama
    # NOTE: 100 was too aggressive - most microscope scans have fewer inliers per pair
    # Lowered to 10 to allow more connections while still filtering truly bad matches
    MIN_INLIERS = 10
    # Minimum translation threshold - filter near-identity transforms that would stack images
    # For microscope/flatbed scans, expect at least 30px translation between adjacent images
    MIN_TRANSLATION = 30.0
    weak_edges_filtered = 0
    identity_edges_filtered = 0

    for (i, j), data in homographies.items():
        # Only include edges where both nodes are in largest component
        if i not in largest_component or j not in largest_component:
            print(f"[DEBUG] H3: Skipping edge ({i},{j}) - outside largest component", file=sys.stderr, flush=True)
            continue

        # Filter out weak edges - they produce unreliable transforms
        n_inliers = data['n_inliers']
        if n_inliers < MIN_INLIERS:
            weak_edges_filtered += 1
            print(f"[DEBUG] H3: Filtering weak edge ({i},{j}) with only {n_inliers} inliers (< {MIN_INLIERS})", file=sys.stderr, flush=True)
            continue

        # Filter out near-identity transforms (translation too small)
        # These cause images to stack on top of each other
        H = data['H']
        tx, ty = H[0, 2], H[1, 2]
        translation_magnitude = np.sqrt(tx*tx + ty*ty)
        if translation_magnitude < MIN_TRANSLATION:
            identity_edges_filtered += 1
            print(f"[DEBUG] H3: Filtering near-identity edge ({i},{j}) with translation={translation_magnitude:.1f}px (< {MIN_TRANSLATION})", file=sys.stderr, flush=True)
            continue

        filtered_homographies[(i, j)] = data

        if i not in adjacency:
            adjacency[i] = []
        if j not in adjacency:
            adjacency[j] = []

        # Weight = number of inliers (let feature matching determine spatial adjacency)
        # Small sequential bonus for within-row connections, but don't override poor matches
        seq_bonus = 5 if abs(i - j) == 1 else 0
        weight = n_inliers + seq_bonus

        # Store both directions with correct transformations
        # From i to j: use H (transforms points i->j)
        # From j to i: use inv(H) (transforms points j->i)
        adjacency[i].append((j, data['H'], weight))
        adjacency[j].append((i, np.linalg.inv(data['H']), weight))

    print(f"[DEBUG] H3: Using {len(filtered_homographies)} edges (filtered {weak_edges_filtered} weak + {identity_edges_filtered} near-identity)", file=sys.stderr, flush=True)
    print(f"[DEBUG] H3: Adjacency has {len(adjacency)} nodes: {sorted(adjacency.keys())}", file=sys.stderr, flush=True)
    
    # Compute cumulative homographies using PROPER Dijkstra (maximum weight path)
    # Key fix: Don't mark nodes as "discovered" until they're PROCESSED (popped from queue)
    # This allows finding better (higher weight) paths to each node
    import heapq
    H_to_ref = {ref_idx: np.eye(3)}
    best_weight_to = {ref_idx: float('inf')}  # Best path weight to reach each node
    processed = set()  # Nodes whose neighbors we've expanded (finalized)
    parent_of = {ref_idx: None}  # Track which image each was reached from
    # Priority queue: (negative_weight, current_idx, parent_idx, H_transform)
    # Using negative weight so heapq (min-heap) gives us max-weight first
    pq = [(0, ref_idx, None, np.eye(3))]

    # DIAGNOSTIC: Log the BFS chain to file (use log_dir if available for Windows access)
    if log_dir:
        bfs_log_path = str(log_dir / "colmap_debug.log")
    else:
        bfs_log_path = "/tmp/colmap_bfs_chain.log"
    with open(bfs_log_path, 'w') as f:
        f.write(f"Reference image: {ref_idx}\n")
        f.write(f"Adjacency list:\n")
        for node, neighbors in sorted(adjacency.items()):
            neighbor_strs = [f"{n}(w={w})" for n, _, w in neighbors]
            f.write(f"  Image {node}: connects to {neighbor_strs}\n")
        f.write(f"\nBFS traversal (Dijkstra max-weight):\n")

    print(f"[PROGRESS] Building transform chain from reference image {ref_idx}...", file=sys.stderr, flush=True)
    last_progress_pct = 0

    while pq:
        neg_path_weight, current, parent, H_current = heapq.heappop(pq)
        path_weight = -neg_path_weight

        if current in processed:
            # Already finalized this node with a better path
            continue

        # Finalize this node - we've found the best path to it
        processed.add(current)
        H_to_ref[current] = H_current
        parent_of[current] = parent

        # Log this connection
        if parent is not None:
            with open(bfs_log_path, 'a') as f:
                tx, ty = H_current[0, 2], H_current[1, 2]
                f.write(f"  Image {current} reached via Image {parent} (path_weight={path_weight}), translation=({tx:.1f}, {ty:.1f})\n")

        # Progress update
        progress_pct = int((len(processed) / n_images) * 100)
        if progress_pct >= last_progress_pct + 10:
            print(f"[PROGRESS] Transform chain: {len(processed)}/{n_images} images connected ({progress_pct}%)", file=sys.stderr, flush=True)
            last_progress_pct = progress_pct

        # Explore neighbors
        for neighbor, H_edge, edge_weight in adjacency.get(current, []):
            if neighbor in processed:
                continue  # Already finalized

            # Path weight is the MINIMUM edge weight along the path (bottleneck)
            # This ensures we prefer paths with consistently strong edges
            new_path_weight = min(path_weight if path_weight > 0 else float('inf'), edge_weight)

            # Only add to queue if this is a better path (higher minimum weight)
            if new_path_weight > best_weight_to.get(neighbor, 0):
                best_weight_to[neighbor] = new_path_weight
                # Compute transform: neighbor->ref = current->ref @ inv(edge)
                # H_edge transforms current->neighbor, so inv(H_edge) transforms neighbor->current
                H_neighbor = H_current @ np.linalg.inv(H_edge)
                # Cap weight for heap
                capped_weight = int(min(new_path_weight, 100000))
                heapq.heappush(pq, (-capped_weight, neighbor, current, H_neighbor))

    # Log final transform summary
    with open(bfs_log_path, 'a') as f:
        f.write(f"\nFinal transforms (translation component only):\n")
        for idx in sorted(H_to_ref.keys()):
            H = H_to_ref[idx]
            tx, ty = H[0, 2], H[1, 2]
            parent = parent_of.get(idx, "N/A")
            f.write(f"  Image {idx}: tx={tx:.1f}, ty={ty:.1f} (via {parent})\n")

    print(f"[PROGRESS] BFS chain diagnostics written to: {bfs_log_path}", file=sys.stderr, flush=True)

    # Update discovered set to match processed for compatibility with rest of code
    discovered = processed

    # #region agent log - H3: Check discovered images
    print(f"[DEBUG] H3: discovered={sorted(discovered)}, H_to_ref_keys={sorted(H_to_ref.keys())}, n_images={n_images}", file=sys.stderr, flush=True)

    # Log which images are NOT included - write to file for Windows access
    all_indices = set(range(n_images))
    missing_indices = all_indices - discovered

    with open(bfs_log_path, 'a') as f:
        f.write(f"\n{'='*70}\n")
        f.write(f"STITCHING RESULT: {len(discovered)}/{n_images} images included\n")
        f.write(f"Discovered: {sorted(discovered)}\n")
        f.write(f"Missing: {sorted(missing_indices)}\n")
        f.write(f"{'='*70}\n\n")

        if missing_indices:
            f.write(f"*** {len(missing_indices)} IMAGES EXCLUDED - ANALYSIS ***\n\n")

            # Categorize missing images by why they were excluded
            for idx in sorted(missing_indices):
                if idx not in connections:
                    f.write(f"Image {idx}: ISOLATED - no feature matches with any other image\n")
                else:
                    # Check which component this image is in
                    in_component = None
                    for comp_idx, comp in enumerate(components):
                        if idx in comp:
                            in_component = comp_idx
                            break

                    if in_component is not None and in_component > 0:
                        connected_to = [j for (i, j) in homographies.keys() if i == idx] + [i for (i, j) in homographies.keys() if j == idx]
                        f.write(f"Image {idx}: In disconnected component #{in_component} ({len(components[in_component])} images), connects to {connected_to}\n")
                    else:
                        connected_to = [j for (i, j) in homographies.keys() if i == idx] + [i for (i, j) in homographies.keys() if j == idx]
                        in_adjacency = idx in adjacency
                        adjacency_neighbors = [n for n, _, _ in adjacency.get(idx, [])]
                        f.write(f"Image {idx}: has {connections[idx]} original connections to {connected_to}\n")
                        f.write(f"Image {idx}: in_adjacency={in_adjacency}, adjacency_neighbors={adjacency_neighbors}\n")

                        # Show what edges were filtered for this image
                        for (i, j), data in homographies.items():
                            if i == idx or j == idx:
                                n_inliers = data['n_inliers']
                                H = data['H']
                                tx, ty = H[0, 2], H[1, 2]
                                trans_mag = np.sqrt(tx*tx + ty*ty)
                                status = "KEPT" if (i, j) in filtered_homographies or (j, i) in filtered_homographies else "FILTERED"
                                f.write(f"  Edge ({i},{j}): {n_inliers} inliers, translation={trans_mag:.1f}px -> {status}\n")
                        f.write("\n")

        # Also log final transforms for ALL images (including included ones)
        f.write(f"\n{'='*70}\n")
        f.write(f"FINAL TRANSFORMS (translation component):\n")
        f.write(f"{'='*70}\n")
        for idx in sorted(H_to_ref.keys()):
            H = H_to_ref[idx]
            tx, ty = H[0, 2], H[1, 2]
            parent = parent_of.get(idx, "N/A")
            f.write(f"Image {idx}: tx={tx:.1f}, ty={ty:.1f} (via {parent})\n")
    # #endregion
    
    # Compute canvas size - LAZY LOADING: only read image dimensions, not full data
    print(f"[PROGRESS] Computing canvas size for {len(H_to_ref)} images (dimension check only)...", file=sys.stderr, flush=True)
    corners_all = []
    image_dimensions = {}  # Cache dimensions for warping phase
    try:
        for idx, H in H_to_ref.items():
            # LAZY: Get image dimensions without loading full image data
            img_path = str(image_paths[idx])
            # Read just to get shape, then immediately discard
            img_temp = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img_temp is None:
                print(f"[WARNING] Could not read image {idx}: {img_path}", file=sys.stderr, flush=True)
                continue
            h, w = img_temp.shape[:2]
            n_channels = img_temp.shape[2] if img_temp.ndim == 3 else 1
            image_dimensions[idx] = (h, w, n_channels)
            del img_temp  # Free memory immediately

            corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)

            # Use appropriate transform based on matrix type
            if use_affine:
                # For affine transforms (2x3 matrix), use cv2.transform
                M_affine = H[:2, :] if H.shape[0] == 3 else H
                transformed = cv2.transform(corners, M_affine)
            else:
                # For homography (3x3 matrix), use perspectiveTransform
                transformed = cv2.perspectiveTransform(corners, H)

            # #region agent log - H2: Check for invalid transform
            if np.any(np.isnan(transformed)) or np.any(np.isinf(transformed)):
                print(f"[DEBUG] H2: Invalid transform for idx={idx}, use_affine={use_affine}, H.shape={H.shape}, transformed={transformed.tolist()}", file=sys.stderr, flush=True)
            # #endregion
            corners_all.append(transformed)
    except Exception as e:
        print(f"[DEBUG] H2: Exception in transform: {e}\n{traceback.format_exc()}", file=sys.stderr, flush=True)
        raise
    
    corners_all = np.concatenate(corners_all, axis=0)
    x_min, y_min = corners_all.min(axis=0).ravel()
    x_max, y_max = corners_all.max(axis=0).ravel()

    print(f"[DEBUG] H1: ALL corners min/max: x=({x_min:.1f}, {x_max:.1f}), y=({y_min:.1f}, {y_max:.1f})", file=sys.stderr, flush=True)

    offset_x = -int(np.floor(x_min))
    offset_y = -int(np.floor(y_min))

    output_w = int(np.ceil(x_max - x_min))
    output_h = int(np.ceil(y_max - y_min))
    
    # #region agent log - H1: Canvas dimensions before limit
    print(f"[DEBUG] H1: Canvas before limit: output_w={output_w}, output_h={output_h}, x_range=({x_min:.1f},{x_max:.1f}), y_range=({y_min:.1f},{y_max:.1f})", file=sys.stderr, flush=True)
    # #endregion
    
    # Limit size - reduced from 20000 to 15000 to prevent OOM
    max_dim = 15000
    if output_w > max_dim or output_h > max_dim:
        scale = max_dim / max(output_w, output_h)
        output_w = int(output_w * scale)
        output_h = int(output_h * scale)
        offset_x = int(offset_x * scale)
        offset_y = int(offset_y * scale)
        S = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]])
        H_to_ref = {k: S @ v for k, v in H_to_ref.items()}
        print(f"[DEBUG] H1: Scaled down with scale={scale:.4f}", file=sys.stderr, flush=True)
    
    # #region agent log - H1/H5: Final canvas size and memory
    memory_mb = (output_h * output_w * 3 * 4) / (1024 * 1024)  # float32
    print(f"[DEBUG] H1/H5: Final canvas: {output_w}x{output_h}, estimated memory: {memory_mb:.1f}MB", file=sys.stderr, flush=True)
    # #endregion
    
    T = np.array([[1, 0, offset_x], [0, 1, offset_y], [0, 0, 1]], dtype=np.float64)

    # STREAMING BLEND: Process images one at a time to avoid memory exhaustion
    # For large image sets (1000+), accumulating all warped images would require 100+ GB RAM
    # Instead, we blend each image immediately after warping, keeping memory constant

    total_to_warp = len(H_to_ref)
    blend_method_lower = blend_method.lower()

    # Determine if we can use streaming blend (feather, linear, autostitch)
    # or need memory-mapped fallback (multiband)
    use_streaming = blend_method_lower in ['feather', 'linear', 'autostitch']

    if use_streaming:
        print(f"[PROGRESS] Using STREAMING BLEND for {total_to_warp} images (constant memory usage)", file=sys.stderr, flush=True)
        streaming_blender = StreamingBlender(output_h, output_w, blend_method_lower)
        mmap_store = None
    else:
        print(f"[PROGRESS] Using MEMORY-MAPPED storage for {total_to_warp} images (multiband requires all images)", file=sys.stderr, flush=True)
        streaming_blender = None
        mmap_temp_dir = Path(tempfile.mkdtemp(prefix="colmap_mmap_"))
        mmap_store = MemoryMappedImageStore(mmap_temp_dir)

    images_processed = 0

    for warp_idx, (idx, H) in enumerate(H_to_ref.items()):
        # Progress update every 10% or every 10 images
        if warp_idx % max(1, total_to_warp // 10) == 0:
            pct = (warp_idx / total_to_warp) * 100
            print(f"[PROGRESS] Warping: {warp_idx}/{total_to_warp} images ({pct:.0f}%)", file=sys.stderr, flush=True)

        # LAZY LOADING: Load image on-demand
        img_path = str(image_paths[idx])
        if use_source_alpha:
            img_raw = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        else:
            img_raw = cv2.imread(img_path, cv2.IMREAD_COLOR)

        if img_raw is None:
            print(f"[WARNING] Could not load image {idx}: {img_path}", file=sys.stderr, flush=True)
            continue

        # Extract alpha channel if present and requested
        if use_source_alpha and img_raw.ndim == 3 and img_raw.shape[2] == 4:
            img = img_raw[:, :, :3]  # BGR channels
            alpha_original = img_raw[:, :, 3]  # Alpha channel

            # Optionally erode alpha mask to remove thin black borders from circular images
            if erode_border and border_erosion_pixels > 0:
                kernel_size = border_erosion_pixels
                erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                alpha_original = cv2.erode(alpha_original, erode_kernel, iterations=2)
        else:
            img = img_raw if img_raw.ndim == 3 else cv2.cvtColor(img_raw, cv2.COLOR_GRAY2BGR)
            alpha_original = None

        del img_raw  # Free raw image memory

        H_final = T @ H

        # Get image dimensions and compute transformed corners
        h, w = img.shape[:2]
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)

        if use_affine:
            M_affine = H_final[:2, :]
            transformed = cv2.transform(corners, M_affine)
        else:
            transformed = cv2.perspectiveTransform(corners, H_final)

        x_min = int(np.floor(transformed[:, 0, 0].min()))
        y_min = int(np.floor(transformed[:, 0, 1].min()))
        x_max = int(np.ceil(transformed[:, 0, 0].max()))
        y_max = int(np.ceil(transformed[:, 0, 1].max()))

        # Clamp to canvas bounds
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(output_w, x_max)
        y_max = min(output_h, y_max)

        region_w = x_max - x_min
        region_h = y_max - y_min

        if region_w <= 0 or region_h <= 0:
            del img
            continue

        # Create alpha channel if not provided from source
        if alpha_original is None:
            alpha_original = np.ones((h, w), dtype=np.uint8) * 255

        # Warp to valid region only
        T_offset = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float64)
        H_cropped = T_offset @ H_final

        if use_affine:
            M_affine = H_cropped[:2, :]
            warped = cv2.warpAffine(img, M_affine, (region_w, region_h), flags=interp_flag)
            alpha = cv2.warpAffine(alpha_original, M_affine, (region_w, region_h), flags=interp_flag)
        else:
            warped = cv2.warpPerspective(img, H_cropped, (region_w, region_h), flags=interp_flag)
            alpha = cv2.warpPerspective(alpha_original, H_cropped, (region_w, region_h), flags=interp_flag)

        # Optional RealESRGAN enhancement for super-resolution quality
        if use_realesrgan:
            try:
                # Add parent directory to path for imports
                script_dir = Path(__file__).resolve().parent.parent
                if str(script_dir) not in sys.path:
                    sys.path.insert(0, str(script_dir))

                from core.post_processing import upscale_with_realesrgan

                # Upscale 2x, then downsample back to target size for detail enhancement
                warped_enhanced = upscale_with_realesrgan(warped, scale=2)
                # Downsample back to original size with high-quality interpolation
                warped = cv2.resize(warped_enhanced, (region_w, region_h), interpolation=cv2.INTER_AREA)
                del warped_enhanced
            except Exception as e:
                # If RealESRGAN fails, continue with regular warped image
                print(f"[WARNING] RealESRGAN enhancement failed: {e}, using standard interpolation", file=sys.stderr, flush=True)

        # Free source image memory after warping
        del img
        del alpha_original

        bbox = (x_min, y_min, x_max, y_max)

        # STREAMING: Blend immediately or store to disk
        if use_streaming:
            streaming_blender.add_image(warped, alpha, bbox)
            del warped  # Free immediately after blending
            del alpha
        else:
            # Memory-mapped: store to disk for later
            mmap_store.add_image(warped, alpha, bbox, images_processed)
            del warped
            del alpha

        images_processed += 1

        # Periodic progress for large sets (every 100 images)
        if images_processed % 100 == 0:
            print(f"[PROGRESS] Processed {images_processed}/{total_to_warp} images", file=sys.stderr, flush=True)

    print(f"[PROGRESS] Warping complete: {images_processed} images processed", file=sys.stderr, flush=True)

    # FINALIZE BLEND
    print(f"[PROGRESS] Finalizing blend using {blend_method} method...", file=sys.stderr, flush=True)
    sys.stderr.flush()

    if use_streaming:
        # Streaming blend: just finalize
        print(f"[PROGRESS] Finalizing streaming blend ({streaming_blender.images_blended} images)...", file=sys.stderr, flush=True)
        panorama = streaming_blender.finalize()
        print(f"[PROGRESS] Streaming blend complete!", file=sys.stderr, flush=True)
    else:
        # Memory-mapped: load images back and use ImageBlender for multiband
        print(f"[PROGRESS] Loading {len(mmap_store)} images from disk for multiband blending...", file=sys.stderr, flush=True)

        try:
            # Import ImageBlender for multiband
            script_dir = Path(__file__).resolve().parent.parent
            if str(script_dir) not in sys.path:
                sys.path.insert(0, str(script_dir))
            from core.blender import ImageBlender

            blender = ImageBlender(
                method=blend_method,
                options={'hdr_mode': False, 'anti_ghosting': False}
            )

            # Load images from disk in batches to blend
            aligned_images = []
            for i in range(len(mmap_store)):
                warped, alpha, bbox = mmap_store.get_image(i)
                aligned_images.append({
                    'image': warped,
                    'alpha': alpha,
                    'bbox': bbox,
                    'warped': True
                })

                # Progress for large sets
                if (i + 1) % 100 == 0:
                    print(f"[PROGRESS] Loaded {i + 1}/{len(mmap_store)} images from disk", file=sys.stderr, flush=True)

            print(f"[PROGRESS] Blending {len(aligned_images)} images with ImageBlender...", file=sys.stderr, flush=True)
            panorama = blender.blend(aligned_images, padding=0, fit_all=False)

            # Free aligned_images
            del aligned_images

        except Exception as e:
            print(f"[WARNING] ImageBlender failed: {e}, using fallback blend", file=sys.stderr, flush=True)
            # Fallback: simple weighted average
            panorama = np.zeros((output_h, output_w, 3), dtype=np.float32)
            weight_sum = np.zeros((output_h, output_w), dtype=np.float32)

            for i in range(len(mmap_store)):
                warped, alpha, bbox = mmap_store.get_image(i)
                x_min, y_min, x_max, y_max = bbox

                mask = (alpha / 255.0).astype(np.float32)
                warped_float = warped.astype(np.float32)

                h, w = warped.shape[:2]
                y_end = min(y_min + h, output_h)
                x_end = min(x_min + w, output_w)

                for c in range(3):
                    panorama[y_min:y_end, x_min:x_end, c] += warped_float[:y_end-y_min, :x_end-x_min, c] * mask[:y_end-y_min, :x_end-x_min]
                weight_sum[y_min:y_end, x_min:x_end] += mask[:y_end-y_min, :x_end-x_min]

                del warped, alpha

            weight_sum = np.maximum(weight_sum, 1e-6)
            for c in range(3):
                panorama[:, :, c] /= weight_sum
            panorama = np.clip(panorama, 0, 255).astype(np.uint8)

        # Cleanup temp files
        mmap_store.cleanup()
        print(f"[PROGRESS] Memory-mapped blend complete!", file=sys.stderr, flush=True)

    print(f"[DEBUG] stitch_with_homographies complete: panorama.shape={panorama.shape}", file=sys.stderr, flush=True)
    sys.stderr.flush()
    return panorama


if __name__ == "__main__":
    import argparse
    import signal
    
    # Handle signals gracefully
    def signal_handler(signum, frame):
        print(f"[DEBUG] Received signal {signum}", file=sys.stderr, flush=True)
        print(json.dumps({"success": False, "error": f"Process killed by signal {signum}"}))
        sys.exit(1)
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    parser = argparse.ArgumentParser(description='WSL COLMAP Bridge for GPU-accelerated stitching')
    parser.add_argument('--config', type=str, help='Path to config JSON file')
    parser.add_argument('image_paths_json', nargs='?', help='JSON string of image paths (legacy)')
    parser.add_argument('output_dir', nargs='?', help='Output directory (legacy)')
    
    args = parser.parse_args()
    
    try:
        if args.config:
            # New config file approach
            with open(args.config, 'r') as f:
                config = json.load(f)
            image_paths_json = json.dumps(config['image_paths'])
            output_dir = config['output_dir']
            use_affine = config.get('use_affine', False)
            blend_method = config.get('blend_method', 'multiband')
            matcher_type = config.get('matcher_type', 'exhaustive')
            sequential_overlap = config.get('sequential_overlap', 10)
            gpu_index = config.get('gpu_index', -1)
            num_threads = config.get('num_threads', -1)
            max_features = config.get('max_features', 8192)
            min_inliers = config.get('min_inliers', 0)
            max_images = config.get('max_images', 0)
            use_source_alpha = config.get('use_source_alpha', False)
            remove_duplicates = config.get('remove_duplicates', False)
            duplicate_threshold = config.get('duplicate_threshold', 0.92)
            warp_interpolation = config.get('warp_interpolation', 'linear')
            erode_border = config.get('erode_border', True)
            border_erosion_pixels = config.get('border_erosion_pixels', 5)
        elif args.image_paths_json and args.output_dir:
            # Legacy command line approach
            image_paths_json = args.image_paths_json
            output_dir = args.output_dir
            use_affine = False
            blend_method = 'multiband'
            matcher_type = 'exhaustive'
            sequential_overlap = 10
            gpu_index = -1
            num_threads = -1
            max_features = 8192
            min_inliers = 0
            max_images = 0
            use_source_alpha = False
            remove_duplicates = False
            duplicate_threshold = 0.92
            warp_interpolation = 'linear'
            erode_border = True
            border_erosion_pixels = 5
        else:
            print("Usage: wsl_colmap_bridge.py --config <config.json>", file=sys.stderr)
            print("   or: wsl_colmap_bridge.py <image_paths_json> <output_dir>", file=sys.stderr)
            sys.exit(1)

        print("[DEBUG] Starting run_2d_stitch...", file=sys.stderr, flush=True)
        print(f"[DEBUG] Parameters: matcher={matcher_type}, features={max_features}, threads={num_threads}, min_inliers={min_inliers}, max_images={max_images}, use_source_alpha={use_source_alpha}, remove_duplicates={remove_duplicates}, warp_interpolation={warp_interpolation}, erode_border={erode_border}, border_erosion_pixels={border_erosion_pixels}", file=sys.stderr, flush=True)
        result = run_2d_stitch(image_paths_json, output_dir, use_affine, blend_method,
                              matcher_type, sequential_overlap, gpu_index, num_threads, max_features,
                              min_inliers, max_images, use_source_alpha, remove_duplicates, duplicate_threshold,
                              warp_interpolation,
                              erode_border, border_erosion_pixels)
        print("[DEBUG] run_2d_stitch completed, outputting result...", file=sys.stderr, flush=True)
        print(json.dumps(result))
        sys.stdout.flush()
        print("[DEBUG] Result output complete", file=sys.stderr, flush=True)
    except MemoryError as e:
        import traceback
        print(f"[DEBUG] MemoryError: {e}", file=sys.stderr, flush=True)
        print(json.dumps({"success": False, "error": f"Out of memory: {e}", "traceback": traceback.format_exc()}))
    except Exception as e:
        import traceback
        print(f"[DEBUG] Exception: {e}", file=sys.stderr, flush=True)
        print(json.dumps({"success": False, "error": str(e), "traceback": traceback.format_exc()}))
        sys.exit(1)

