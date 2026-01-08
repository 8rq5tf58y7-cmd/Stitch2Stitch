#!/usr/bin/env python3
"""
WSL COLMAP Bridge - GPU-accelerated pycolmap via WSL

Runs inside WSL and is invoked by the Windows app (`src/external/pipelines.py`) via:
  wsl -e python3 -u wsl_colmap_bridge.py --config <config.json>

This restored version includes:
- Persistent **global cache** (database.db + copied images) keyed by image set + settings
- Persistent **per-image feature cache** (keypoints/descriptors) keyed by image content hash
- Persistent **per-pair match cache** (inlier correspondences) keyed by image keys + matcher params

Critical stability note:
- Workspaces (including `database.db`) are created on WSL native filesystem (e.g. /tmp) to avoid
  drvfs mmap/locking issues that can cause SQLite "disk I/O error" / SIGBUS.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# Debug instrumentation for mask detection
_DEBUG_MASK_DIR = None
_DEBUG_MASK_COUNTER = 0


def _get_debug_dir():
    """Get/create debug output directory for mask detection debugging."""
    global _DEBUG_MASK_DIR
    if _DEBUG_MASK_DIR is None:
        _DEBUG_MASK_DIR = Path(os.environ.get("TEMP", "/tmp")) / "stitch2stitch_mask_debug"
        _DEBUG_MASK_DIR.mkdir(parents=True, exist_ok=True)
        print(f"[DEBUG] Mask debug output: {_DEBUG_MASK_DIR}", file=sys.stderr, flush=True)
    return _DEBUG_MASK_DIR


def windows_to_wsl_path(win_path: str) -> str:
    """Convert Windows path to WSL path."""
    path = win_path.replace("\\", "/")
    if len(path) >= 2 and path[1] == ":":
        drive = path[0].lower()
        path = f"/mnt/{drive}{path[2:]}"
    return path


def wsl_to_windows_path(wsl_path: str) -> str:
    """Convert WSL path to Windows path."""
    if wsl_path.startswith("/mnt/") and len(wsl_path) > 6:
        drive = wsl_path[5].upper()
        rest = wsl_path[6:].replace("/", "\\")
        return f"{drive}:{rest}"
    return wsl_path


def _safe_imwrite(path, image):
    """Write image with error handling - never fails even if write fails."""
    try:
        import cv2
        cv2.imwrite(str(path), image)
    except Exception as e:
        print(f"[DEBUG WARNING] Could not save debug image: {e}", file=sys.stderr, flush=True)


def detect_content_mask(image: np.ndarray,
                        method: str = "ellipse",
                        threshold: int = 15,
                        image_name: str = "unknown") -> np.ndarray:
    """
    Detect valid content region in source image (handles round/circular images).
    WITH FULL DEBUG INSTRUMENTATION.

    For images with black borders (like microscope images with circular apertures),
    this function detects where actual content exists vs artificial black borders.

    Args:
        image: BGR source image
        method: Detection method - "ellipse" (recommended for round images),
                "contour", "flood", or "threshold"
        threshold: Brightness threshold for initial detection
        image_name: Name of the image for debug logging

    Returns:
        Binary mask (uint8) where 255 = content, 0 = border
    """
    import cv2
    global _DEBUG_MASK_COUNTER
    _DEBUG_MASK_COUNTER += 1
    img_id = _DEBUG_MASK_COUNTER

    try:
        debug_dir = _get_debug_dir()
    except Exception as e:
        print(f"[DEBUG WARNING] Could not get debug dir: {e}", file=sys.stderr, flush=True)
        debug_dir = None

    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # Log basic image stats
    print(f"\n{'='*60}", file=sys.stderr, flush=True)
    print(f"[DEBUG #{img_id}] Processing: {image_name}", file=sys.stderr, flush=True)
    print(f"[DEBUG #{img_id}] Image shape: {image.shape}", file=sys.stderr, flush=True)
    print(f"[DEBUG #{img_id}] Gray stats: min={gray.min()}, max={gray.max()}, mean={gray.mean():.1f}, std={gray.std():.1f}", file=sys.stderr, flush=True)

    # Save original grayscale
    if debug_dir:
        _safe_imwrite(debug_dir / f"{img_id:04d}_1_gray.png", gray)

    if method == "ellipse":
        # Step 1: Threshold
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        white_pixels_after_thresh = np.sum(binary > 0)
        white_pct = 100.0 * white_pixels_after_thresh / (h * w)
        print(f"[DEBUG #{img_id}] After threshold({threshold}): {white_pixels_after_thresh} white pixels ({white_pct:.1f}%)", file=sys.stderr, flush=True)
        if debug_dir:
            _safe_imwrite(debug_dir / f"{img_id:04d}_2_threshold.png", binary)

        # Step 2: Morphological ops
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        binary_closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary_opened = cv2.morphologyEx(binary_closed, cv2.MORPH_OPEN, kernel)

        white_after_close = np.sum(binary_closed > 0)
        white_after_open = np.sum(binary_opened > 0)
        print(f"[DEBUG #{img_id}] After CLOSE: {white_after_close} pixels ({100.0*white_after_close/(h*w):.1f}%)", file=sys.stderr, flush=True)
        print(f"[DEBUG #{img_id}] After OPEN:  {white_after_open} pixels ({100.0*white_after_open/(h*w):.1f}%)", file=sys.stderr, flush=True)
        if debug_dir:
            _safe_imwrite(debug_dir / f"{img_id:04d}_3_morph_close.png", binary_closed)
            _safe_imwrite(debug_dir / f"{img_id:04d}_4_morph_open.png", binary_opened)

        binary = binary_opened

        # Step 3: Find contours
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"[DEBUG #{img_id}] Found {len(contours)} contours", file=sys.stderr, flush=True)

        # Log all contour areas
        if contours:
            areas = [cv2.contourArea(c) for c in contours]
            areas_sorted = sorted(areas, reverse=True)[:5]  # Top 5
            print(f"[DEBUG #{img_id}] Top contour areas: {areas_sorted}", file=sys.stderr, flush=True)

            # Draw all contours for visualization
            if debug_dir:
                contour_vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                cv2.drawContours(contour_vis, contours, -1, (0, 255, 0), 2)
                _safe_imwrite(debug_dir / f"{img_id:04d}_5_contours.png", contour_vis)

        if not contours:
            print(f"[DEBUG #{img_id}] *** NO CONTOURS FOUND - RETURNING FULL MASK ***", file=sys.stderr, flush=True)
            mask = np.ones((h, w), dtype=np.uint8) * 255
            if debug_dir:
                _safe_imwrite(debug_dir / f"{img_id:04d}_6_FULL_MASK.png", mask)
            print(f"{'='*60}\n", file=sys.stderr, flush=True)
            return mask

        # Step 4: Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        largest_area = cv2.contourArea(largest_contour)
        print(f"[DEBUG #{img_id}] Largest contour: {len(largest_contour)} points, area={largest_area}", file=sys.stderr, flush=True)

        # Step 5: Fit ellipse or use hull
        if len(largest_contour) >= 5:
            ellipse = cv2.fitEllipse(largest_contour)
            (cx, cy), (axis1, axis2), angle = ellipse
            print(f"[DEBUG #{img_id}] Fitted ellipse: center=({cx:.1f},{cy:.1f}), axes=({axis1:.1f},{axis2:.1f}), angle={angle:.1f}", file=sys.stderr, flush=True)

            # Check if ellipse is reasonable
            ellipse_area = 3.14159 * (axis1/2) * (axis2/2)
            area_ratio = ellipse_area / (h * w)
            print(f"[DEBUG #{img_id}] Ellipse area: {ellipse_area:.0f} ({area_ratio*100:.1f}% of image)", file=sys.stderr, flush=True)

            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(mask, ellipse, 255, -1)

            # Visualize ellipse on original
            if debug_dir:
                ellipse_vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                cv2.ellipse(ellipse_vis, ellipse, (0, 255, 0), 3)
                _safe_imwrite(debug_dir / f"{img_id:04d}_6_ellipse_overlay.png", ellipse_vis)
        else:
            print(f"[DEBUG #{img_id}] Not enough points for ellipse ({len(largest_contour)}), using convex hull", file=sys.stderr, flush=True)
            mask = np.zeros((h, w), dtype=np.uint8)
            hull = cv2.convexHull(largest_contour)
            cv2.fillPoly(mask, [hull], 255)
            # Save hull visualization
            if debug_dir:
                hull_vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                cv2.drawContours(hull_vis, [hull], -1, (0, 255, 0), 3)
                _safe_imwrite(debug_dir / f"{img_id:04d}_6_hull_overlay.png", hull_vis)

        # Save final mask
        if debug_dir:
            _safe_imwrite(debug_dir / f"{img_id:04d}_7_final_mask.png", mask)

        # Log mask coverage
        mask_pixels = np.sum(mask > 0)
        mask_pct = 100.0 * mask_pixels / (h * w)
        print(f"[DEBUG #{img_id}] Final mask: {mask_pixels} pixels ({mask_pct:.1f}% coverage)", file=sys.stderr, flush=True)
        print(f"{'='*60}\n", file=sys.stderr, flush=True)

        return mask

    elif method == "contour":
        # Use largest contour directly (for non-elliptical shapes)
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"[DEBUG #{img_id}] contour method: Found {len(contours)} contours", file=sys.stderr, flush=True)

        if not contours:
            print(f"[DEBUG #{img_id}] *** NO CONTOURS - RETURNING FULL MASK ***", file=sys.stderr, flush=True)
            print(f"{'='*60}\n", file=sys.stderr, flush=True)
            return np.ones((h, w), dtype=np.uint8) * 255

        # Create mask from largest contour
        mask = np.zeros((h, w), dtype=np.uint8)
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.fillPoly(mask, [largest_contour], 255)
        print(f"{'='*60}\n", file=sys.stderr, flush=True)
        return mask

    elif method == "flood":
        # Flood fill from corners to find connected black regions
        mask = np.ones((h, w), dtype=np.uint8) * 255
        flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        # Flood from all 4 corners
        for seed in [(0, 0), (w-1, 0), (0, h-1), (w-1, h-1)]:
            if gray[seed[1], seed[0]] < threshold:
                cv2.floodFill(mask, flood_mask, seed, 0,
                             loDiff=threshold, upDiff=threshold)
        print(f"[DEBUG #{img_id}] flood method: mask coverage {100.0*np.sum(mask>0)/(h*w):.1f}%", file=sys.stderr, flush=True)
        print(f"{'='*60}\n", file=sys.stderr, flush=True)
        return mask

    else:  # threshold
        mask = (gray > threshold).astype(np.uint8) * 255
        print(f"[DEBUG #{img_id}] threshold method: mask coverage {100.0*np.sum(mask>0)/(h*w):.1f}%", file=sys.stderr, flush=True)
        print(f"{'='*60}\n", file=sys.stderr, flush=True)
        return mask


def create_content_mask_for_warp(image: np.ndarray,
                                  auto_detect: bool = True,
                                  image_name: str = "unknown") -> np.ndarray:
    """
    Create a content mask suitable for warping.

    For rectangular content: returns full white mask (fast path)
    For round/circular content: detects and masks the valid region

    Args:
        image: Source image (BGR)
        auto_detect: If True, auto-detect circular/non-rectangular content
        image_name: Name of the image for debug logging

    Returns:
        Mask (uint8) to be warped alongside the image
    """
    import cv2

    h, w = image.shape[:2]

    if not auto_detect:
        print(f"[DEBUG] {image_name}: auto_detect=False, returning full mask", file=sys.stderr, flush=True)
        return np.ones((h, w), dtype=np.uint8) * 255

    # Check if corners are dark (indicator of round/circular image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    corner_size = max(10, min(h, w) // 10)  # Sample 10% from each corner, min 10px
    corners = [
        gray[:corner_size, :corner_size],           # top-left
        gray[:corner_size, -corner_size:],          # top-right
        gray[-corner_size:, :corner_size],          # bottom-left
        gray[-corner_size:, -corner_size:],         # bottom-right
    ]
    corner_means = [float(c.mean()) for c in corners]

    # If 3+ corners are dark (mean < 20), it's likely a round image
    if sum(m < 20 for m in corner_means) >= 3:
        print(f"[DEBUG] {image_name}: Detected circular (corners: {corner_means})", file=sys.stderr, flush=True)
        # Use ellipse fitting for smooth geometric mask (not variance which creates irregular masks)
        return detect_content_mask(image, method="ellipse", image_name=image_name)
    else:
        # Rectangular content - full mask
        print(f"[DEBUG] {image_name}: Rectangular content (corners: {corner_means}), returning full mask", file=sys.stderr, flush=True)
        return np.ones((h, w), dtype=np.uint8) * 255


def compute_cache_key(
    image_paths: List[Path],
    max_features: int,
    matcher_type: str = "exhaustive",
    # Note: use_affine, blend_method, warp_interpolation are NOT included
    # because they don't affect feature extraction or matching (database contents).
    # They only affect post-database steps (homography computation, warping, blending).
    **kwargs,  # Accept but ignore other params for backward compatibility
) -> str:
    """
    Compute a cache key for the current image set and feature extraction settings.
    Uses filename+size (not full path) so cache survives folder moves.

    IMPORTANT: Only includes settings that affect database.db contents:
    - image_paths: which images to process
    - max_features: affects feature extraction
    - matcher_type: affects feature matching

    Does NOT include (these are handled separately):
    - use_affine: only affects homography computation (post-database)
    - blend_method: only affects final blending (post-warp)
    - warp_interpolation: only affects warping (post-homography)
    """
    cache_data: List[str] = []
    for p in sorted(image_paths, key=lambda x: x.name):
        try:
            cache_data.append(f"{p.name}:{p.stat().st_size}")
        except Exception:
            cache_data.append(p.name)

    cache_data.append(f"max_features:{int(max_features)}")
    cache_data.append("global_cache_format:v4")  # v4: Fixed image ordering bug - sorted paths
    cache_data.append(f"matcher_type:{matcher_type}")
    cache_data.append(f"count:{len(image_paths)}")
    cache_str = "\n".join(cache_data)
    return hashlib.md5(cache_str.encode("utf-8")).hexdigest()


def compute_cache_key_v2(
    image_paths: List[Path],
    max_features: int,
    matcher_type: str = "exhaustive",
    use_affine: bool = False,
    blend_method: str = "multiband",
    warp_interpolation: str = "linear"
) -> str:
    """
    Legacy v2 cache key computation for backward compatibility.
    Includes blend_method and warp settings that don't actually affect database.db.
    """
    cache_data = []
    for path in sorted(image_paths, key=lambda x: x.name):
        try:
            cache_data.append(f"{path.name}:{path.stat().st_size}")
        except Exception:
            cache_data.append(path.name)
    cache_data.append(f"max_features:{max_features}")
    cache_data.append("global_cache_format:v2")
    cache_data.append(f"matcher_type:{matcher_type}")
    cache_data.append(f"use_affine:{bool(use_affine)}")
    cache_data.append(f"blend_method:{str(blend_method).lower()}")
    cache_data.append(f"warp_interp:{str(warp_interpolation).lower()}")
    cache_data.append(f"count:{len(image_paths)}")
    cache_str = "\n".join(cache_data)
    return hashlib.md5(cache_str.encode("utf-8")).hexdigest()


def is_cache_valid(workspace: Path, cache_key: str, n_images: int) -> bool:
    """Check if global cached database matches the current key and expected image count."""
    cache_file = workspace / "cache_key.txt"
    database_path = workspace / "database.db"

    if not cache_file.exists():
        print(f"[DEBUG] Cache miss: cache_key.txt not found at {cache_file}", file=sys.stderr, flush=True)
        return False
    if not database_path.exists():
        print(f"[DEBUG] Cache miss: database.db not found at {database_path}", file=sys.stderr, flush=True)
        return False

    try:
        stored_key = cache_file.read_text(encoding="utf-8").strip()
        if stored_key != cache_key:
            print("[DEBUG] Cache miss: key mismatch", file=sys.stderr, flush=True)
            print(f"[DEBUG]   Stored:  {stored_key[:16]}...", file=sys.stderr, flush=True)
            print(f"[DEBUG]   Current: {cache_key[:16]}...", file=sys.stderr, flush=True)
            return False

        import sqlite3

        conn = sqlite3.connect(str(database_path))
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM images")
        db_image_count = int(cur.fetchone()[0])
        conn.close()

        if db_image_count != int(n_images):
            print(
                f"[DEBUG] Cache miss: image count mismatch (cached={db_image_count}, current={n_images})",
                file=sys.stderr,
                flush=True,
            )
            return False

        print(f"[DEBUG] Cache hit! Using cached features for {n_images} images", file=sys.stderr, flush=True)
        return True
    except Exception as e:
        print(f"[DEBUG] Cache validation error: {e}", file=sys.stderr, flush=True)
        return False


def save_cache_key(workspace: Path, cache_key: str) -> None:
    """Save the global cache key file (atomic + best-effort fsync)."""
    cache_file = workspace / "cache_key.txt"
    tmp_file = workspace / "cache_key.txt.tmp"
    tmp_file.write_text(cache_key, encoding="utf-8")
    os.replace(str(tmp_file), str(cache_file))
    try:
        fd = os.open(str(cache_file), os.O_RDONLY)
        os.fsync(fd)
        os.close(fd)
    except Exception:
        pass
    print(f"[CACHE] Wrote global cache key file: {cache_file}", file=sys.stderr, flush=True)


# ============================================================
# WARP CACHING - Skip warping on reblend if warp settings unchanged
# ============================================================

def compute_warp_key(
    use_affine: bool,
    warp_interpolation: str,
    use_source_alpha: bool,
    erode_border: bool,
    border_erosion_pixels: int,
    auto_detect_content: bool = False,
) -> str:
    """
    Compute a cache key for warp settings.
    If these settings match, we can skip warping and reuse cached warped images.
    """
    warp_data = [
        f"use_affine:{bool(use_affine)}",
        f"warp_interp:{str(warp_interpolation).lower()}",
        f"use_source_alpha:{bool(use_source_alpha)}",
        f"erode_border:{bool(erode_border)}",
        f"border_erosion_pixels:{int(border_erosion_pixels)}",
        f"auto_detect_content:{bool(auto_detect_content)}",
        "warp_cache_format:v3",  # v3: Added alpha erosion fix for autostitch duplication
    ]
    return hashlib.md5("\n".join(warp_data).encode("utf-8")).hexdigest()[:16]


def get_warp_cache_dir(global_cache_dir: Path, warp_key: str) -> Path:
    """Get the warp cache directory for a given global cache and warp key."""
    return global_cache_dir / f"warped_{warp_key}"


def is_warp_cache_valid(warp_cache_dir: Path, n_images: int) -> bool:
    """Check if warp cache exists and contains the expected number of images."""
    if not warp_cache_dir.exists():
        return False

    metadata_file = warp_cache_dir / "metadata.json"
    if not metadata_file.exists():
        return False

    try:
        import json
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        cached_count = metadata.get("n_warped", 0)
        if cached_count < 1:
            return False

        # Check that warped image files exist
        warped_files = list(warp_cache_dir.glob("warped_*.npz"))
        if len(warped_files) < cached_count:
            return False

        print(f"[CACHE] Warp cache valid: {cached_count} warped images found", file=sys.stderr, flush=True)
        return True
    except Exception as e:
        print(f"[DEBUG] Warp cache validation error: {e}", file=sys.stderr, flush=True)
        return False


def save_warp_cache(
    warp_cache_dir: Path,
    warped_images: List[Dict],
    canvas_size: Tuple[int, int],
    offset: Tuple[int, int],
) -> None:
    """
    Save warped images to cache for later reuse.
    Each warped image is saved as a compressed .npz file with image, alpha, and bbox.
    """
    import json

    warp_cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"[CACHE] Saving {len(warped_images)} warped images to cache...", file=sys.stderr, flush=True)

    for i, item in enumerate(warped_images):
        npz_path = warp_cache_dir / f"warped_{i:06d}.npz"
        np.savez_compressed(
            str(npz_path),
            image=item["image"],
            alpha=item["alpha"],
            bbox=np.array(item["bbox"]),
            idx=item.get("idx", i),
        )

    # Save metadata
    metadata = {
        "n_warped": len(warped_images),
        "canvas_h": canvas_size[0],
        "canvas_w": canvas_size[1],
        "offset_x": offset[0],
        "offset_y": offset[1],
    }
    metadata_file = warp_cache_dir / "metadata.json"
    metadata_file.write_text(json.dumps(metadata), encoding="utf-8")

    print(f"[CACHE] Warp cache saved: {len(warped_images)} images", file=sys.stderr, flush=True)


def load_warp_cache(warp_cache_dir: Path) -> Tuple[List[Dict], Dict]:
    """
    Load warped images from cache.
    Returns (warped_images, metadata) where warped_images is a list of dicts with image, alpha, bbox.
    """
    import json

    metadata_file = warp_cache_dir / "metadata.json"
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    n_warped = metadata["n_warped"]

    print(f"[PROGRESS] Loading {n_warped} cached warped images...", file=sys.stderr, flush=True)

    warped_images = []
    for i in range(n_warped):
        npz_path = warp_cache_dir / f"warped_{i:06d}.npz"
        if not npz_path.exists():
            continue

        data = np.load(str(npz_path))
        warped_images.append({
            "image": data["image"],
            "alpha": data["alpha"],
            "bbox": tuple(data["bbox"]),
            "idx": int(data["idx"]) if "idx" in data else i,
        })

        if (i + 1) % max(1, n_warped // 5) == 0:
            print(f"[PROGRESS] Loaded {i + 1}/{n_warped} warped images", file=sys.stderr, flush=True)

    print(f"[PROGRESS] ✅ Loaded {len(warped_images)} cached warped images", file=sys.stderr, flush=True)
    return warped_images, metadata


class StreamingBlender:
    """
    Memory-efficient streaming blender that processes images one at a time.
    Supports: feather, linear, autostitch, autostitch_feather (streaming).

    autostitch_feather: Hybrid mode that uses distance-based pixel selection
    (like autostitch) but blends at seam boundaries to hide misalignment.
    """

    def __init__(self, output_h: int, output_w: int, blend_method: str = "feather",
                 seam_feather_width: int = 15):
        self.output_h = output_h
        self.output_w = output_w
        self.blend_method = blend_method.lower()
        self.seam_feather_width = seam_feather_width  # Pixels to blend at seams

        if self.blend_method in ["feather", "linear"]:
            self.color_accum = np.zeros((output_h, output_w, 3), dtype=np.float64)
            self.weight_accum = np.zeros((output_h, output_w), dtype=np.float64)
        elif self.blend_method == "autostitch":
            self.canvas = np.zeros((output_h, output_w, 3), dtype=np.uint8)
            self.distance_map = np.full((output_h, output_w), -1, dtype=np.float32)
        elif self.blend_method == "mosaic":
            # Mosaic: center-priority winner-takes-all (fundamentally different from autostitch)
            self.canvas = np.zeros((output_h, output_w, 3), dtype=np.uint8)
            self.priority_map = np.full((output_h, output_w), -np.inf, dtype=np.float32)
        elif self.blend_method == "autostitch_feather":
            # Seam-feathered autostitch: accumulates weighted colors at seams
            self.canvas = np.zeros((output_h, output_w, 3), dtype=np.float64)
            self.distance_map = np.full((output_h, output_w), -1, dtype=np.float32)
            self.weight_accum = np.zeros((output_h, output_w), dtype=np.float64)
            # Track the "winning" image's color for sharp regions
            self.winner_canvas = np.zeros((output_h, output_w, 3), dtype=np.uint8)
        else:
            self.color_accum = None
            self.weight_accum = None

        self.images_blended = 0

    def _compute_distance_transform(self, alpha: np.ndarray) -> np.ndarray:
        import cv2

        mask = (alpha > 127).astype(np.uint8)
        return cv2.distanceTransform(mask, cv2.DIST_L2, 5)

    def _compute_feather_weights(self, alpha: np.ndarray) -> np.ndarray:
        dist = self._compute_distance_transform(alpha)
        max_dist = max(float(dist.max()), 1.0)
        weights = np.clip(dist / max_dist, 0, 1)
        return weights.astype(np.float32)

    def add_image(self, warped: np.ndarray, alpha: np.ndarray, bbox: Tuple[int, int, int, int]) -> None:
        import cv2
        import sys

        x_min, y_min, x_max, y_max = bbox
        region_h, region_w = warped.shape[:2]

        # CRITICAL: Validate alpha dimensions match warped image
        # Dimension mismatch causes massive shifts (100+ pixels)
        alpha_h, alpha_w = alpha.shape[:2]
        if alpha_h != region_h or alpha_w != region_w:
            print(f"[ERROR] Alpha/image dimension mismatch! warped={region_w}x{region_h}, alpha={alpha_w}x{alpha_h}",
                  file=sys.stderr, flush=True)
            print(f"[ERROR] bbox={bbox}, this will cause shift artifacts!", file=sys.stderr, flush=True)
            # Resize alpha to match (emergency fix)
            alpha = cv2.resize(alpha, (region_w, region_h), interpolation=cv2.INTER_NEAREST)

        # Clamp / align bbox to warped size
        y_max = y_min + region_h
        x_max = x_min + region_w

        src_y0 = max(0, -y_min)
        src_x0 = max(0, -x_min)
        dst_y0 = max(0, y_min)
        dst_x0 = max(0, x_min)
        dst_y1 = min(self.output_h, y_max)
        dst_x1 = min(self.output_w, x_max)
        src_y1 = src_y0 + (dst_y1 - dst_y0)
        src_x1 = src_x0 + (dst_x1 - dst_x0)

        if dst_y1 <= dst_y0 or dst_x1 <= dst_x0:
            return

        warped_region = warped[src_y0:src_y1, src_x0:src_x1]
        alpha_region = alpha[src_y0:src_y1, src_x0:src_x1]

        if self.blend_method in ["feather", "linear"]:
            if self.blend_method == "feather":
                weights = self._compute_feather_weights(alpha_region)
            else:
                weights = (alpha_region / 255.0).astype(np.float32)

            warped_f = warped_region.astype(np.float64)
            for c in range(3):
                self.color_accum[dst_y0:dst_y1, dst_x0:dst_x1, c] += warped_f[:, :, c] * weights
            self.weight_accum[dst_y0:dst_y1, dst_x0:dst_x1] += weights

        elif self.blend_method == "autostitch":
            dist = self._compute_distance_transform(alpha_region)
            dst_dist = self.distance_map[dst_y0:dst_y1, dst_x0:dst_x1]
            valid = alpha_region > 127
            better = (dist > dst_dist) & valid
            self.canvas[dst_y0:dst_y1, dst_x0:dst_x1][better] = warped_region[better]
            self.distance_map[dst_y0:dst_y1, dst_x0:dst_x1][better] = dist[better]

        elif self.blend_method == "mosaic":
            # Mosaic: center-priority winner-takes-all
            # Unlike autostitch which uses edge distance, this uses center distance
            h, w = warped_region.shape[:2]
            cy, cx = h / 2, w / 2
            y_coords, x_coords = np.ogrid[:h, :w]
            # Normalized distance from center (0 at center, 1 at corners)
            dist_from_center = np.sqrt(((x_coords - cx) / max(cx, 1)) ** 2 + ((y_coords - cy) / max(cy, 1)) ** 2)
            # Convert to priority: center = 1.0, corners = 0.0
            center_priority = (1.0 - np.clip(dist_from_center / np.sqrt(2), 0, 1)).astype(np.float32)

            valid = alpha_region > 127
            # Only valid pixels get priority (invalid = -inf)
            center_priority = np.where(valid, center_priority, -np.inf)

            dst_priority = self.priority_map[dst_y0:dst_y1, dst_x0:dst_x1]
            better = center_priority > dst_priority
            self.canvas[dst_y0:dst_y1, dst_x0:dst_x1][better] = warped_region[better]
            self.priority_map[dst_y0:dst_y1, dst_x0:dst_x1][better] = center_priority[better]

        elif self.blend_method == "autostitch_feather":
            # Seam-feathered autostitch: blend at seam boundaries, sharp elsewhere
            dist = self._compute_distance_transform(alpha_region)
            dst_dist = self.distance_map[dst_y0:dst_y1, dst_x0:dst_x1]
            valid = alpha_region > 127

            # Get current canvas regions
            dst_canvas = self.canvas[dst_y0:dst_y1, dst_x0:dst_x1]
            dst_weights = self.weight_accum[dst_y0:dst_y1, dst_x0:dst_x1]
            dst_winner = self.winner_canvas[dst_y0:dst_y1, dst_x0:dst_x1]

            # For pixels with no prior content, just write directly
            new_region = (dst_dist < 0) & valid
            if np.any(new_region):
                dst_canvas[new_region] = warped_region[new_region].astype(np.float64)
                dst_winner[new_region] = warped_region[new_region]
                dst_weights[new_region] = 1.0
                self.distance_map[dst_y0:dst_y1, dst_x0:dst_x1][new_region] = dist[new_region]

            # For overlapping regions, blend at seams
            overlap = (dst_dist >= 0) & valid
            if np.any(overlap):
                # Distance difference: positive = new image better, negative = old better
                dist_diff = dist - dst_dist

                # Seam zone: where distances are close (within seam_feather_width)
                seam_zone = overlap & (np.abs(dist_diff) < self.seam_feather_width)
                # Clear winner zones
                new_wins = overlap & (dist_diff >= self.seam_feather_width)
                old_wins = overlap & (dist_diff <= -self.seam_feather_width)

                # In seam zone: blend based on relative distance
                if np.any(seam_zone):
                    # Compute blend weight: 0.5 at equal distance, 1.0 when new is better by seam_width
                    blend_weight = 0.5 + (dist_diff[seam_zone] / (2.0 * self.seam_feather_width))
                    blend_weight = np.clip(blend_weight, 0.0, 1.0)

                    # Weighted accumulation at seams
                    old_colors = dst_canvas[seam_zone]
                    new_colors = warped_region[seam_zone].astype(np.float64)

                    # Blend: old_weight * old + new_weight * new
                    blended = old_colors * (1.0 - blend_weight[:, np.newaxis]) + new_colors * blend_weight[:, np.newaxis]
                    dst_canvas[seam_zone] = blended
                    dst_winner[seam_zone] = np.where(
                        blend_weight[:, np.newaxis] > 0.5,
                        warped_region[seam_zone],
                        dst_winner[seam_zone]
                    )

                    # Update distance map with the "winner" distance for future comparisons
                    new_dist = np.maximum(dist[seam_zone], dst_dist[seam_zone])
                    self.distance_map[dst_y0:dst_y1, dst_x0:dst_x1][seam_zone] = new_dist

                # Where new image clearly wins: overwrite
                if np.any(new_wins):
                    dst_canvas[new_wins] = warped_region[new_wins].astype(np.float64)
                    dst_winner[new_wins] = warped_region[new_wins]
                    self.distance_map[dst_y0:dst_y1, dst_x0:dst_x1][new_wins] = dist[new_wins]

                # Where old image wins: keep existing (no action needed)

        self.images_blended += 1

    def finalize(self) -> Optional[np.ndarray]:
        if self.blend_method in ["feather", "linear"]:
            wsum = np.maximum(self.weight_accum, 1e-6)
            out = np.zeros((self.output_h, self.output_w, 3), dtype=np.float64)
            for c in range(3):
                out[:, :, c] = self.color_accum[:, :, c] / wsum
            return np.clip(out, 0, 255).astype(np.uint8)
        if self.blend_method == "autostitch":
            return self.canvas.copy()
        if self.blend_method == "mosaic":
            return self.canvas.copy()
        if self.blend_method == "autostitch_feather":
            # The canvas already contains blended values
            return np.clip(self.canvas, 0, 255).astype(np.uint8)
        return None

    def supports_streaming(self) -> bool:
        return self.blend_method in ["feather", "linear", "autostitch", "mosaic", "autostitch_feather"]


class MemoryMappedImageStore:
    """Disk-backed store for multiband blending (fallback when streaming isn't possible)."""

    def __init__(self, temp_dir: Path):
        self.temp_dir = temp_dir
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.image_files: List[Tuple[Path, Path]] = []
        self.metadata: List[Dict] = []

    def add_image(self, warped: np.ndarray, alpha: np.ndarray, bbox: Tuple[int, int, int, int], idx: int) -> None:
        import cv2

        img_path = self.temp_dir / f"warped_{idx:06d}.png"
        alpha_path = self.temp_dir / f"alpha_{idx:06d}.png"
        cv2.imwrite(str(img_path), warped)
        cv2.imwrite(str(alpha_path), alpha)
        self.image_files.append((img_path, alpha_path))
        self.metadata.append({"bbox": bbox, "shape": warped.shape})

    def get_image(self, idx: int) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
        import cv2

        img_path, alpha_path = self.image_files[idx]
        warped = cv2.imread(str(img_path))
        alpha = cv2.imread(str(alpha_path), cv2.IMREAD_GRAYSCALE)
        bbox = tuple(self.metadata[idx]["bbox"])
        return warped, alpha, bbox

    def __len__(self) -> int:
        return len(self.image_files)

    def cleanup(self) -> None:
        try:
            shutil.rmtree(self.temp_dir)
        except Exception:
            pass


def copy_single_image(args):
    i, path, images_dir = args
    dest_name = f"{i:06d}{path.suffix}"
    dest = images_dir / dest_name
    shutil.copy2(path, dest)
    return i, dest_name, path


def check_feature_cache(image_paths: List[Path], max_features: int, cache_manager):
    cached_features: Dict[int, Dict] = {}
    uncached_images: List[Tuple[int, Path]] = []

    for i, img_path in enumerate(image_paths):
        entry = cache_manager.get_cached_features(img_path, max_features)
        if entry:
            cached_features[i] = {"cache_key": entry["cache_key"], "cache_entry": entry}
        else:
            uncached_images.append((i, img_path))

    # One-time migration log per run
    try:
        stats = cache_manager.get_hit_stats(reset=True)
        print(
            f"[CACHE] Cache lookup summary: {stats['content_hits']} content-hit, {stats['legacy_hits']} legacy-hit, {stats['misses']} miss",
            file=sys.stderr,
            flush=True,
        )
    except Exception:
        pass

    return cached_features, uncached_images


def _pair_id(image_id1: int, image_id2: int) -> int:
    """COLMAP pair_id encoding (must match decode in read_matches_from_db())."""
    if image_id1 == image_id2:
        raise ValueError("pair_id requires two different image ids")
    if image_id1 > image_id2:
        image_id1, image_id2 = image_id2, image_id1
    return int(image_id1) * 2147483647 + int(image_id2)


def _generate_pairs(n_images: int, matcher_type: str, sequential_overlap: int,
                   image_paths: Optional[List[Path]] = None) -> List[Tuple[int, int]]:
    """
    Generate image pairs for matching.

    matcher_types:
    - "exhaustive": Match all pairs (default, O(n²) - use for unknown layouts)
    - "sequential": Only match adjacent images within overlap window
    - "neighbor": Match each image with K nearest neighbors by filename order
                  (optimized for flatbed scans, reduces O(n²) to O(n*K))

    Args:
        n_images: Number of images
        matcher_type: Type of matching strategy
        sequential_overlap: Number of neighbors to match (for sequential/neighbor)
        image_paths: Optional list of image paths (used for neighbor detection)
    """
    pairs: List[Tuple[int, int]] = []

    if matcher_type == "sequential":
        # Simple sequential: match within overlap window
        overlap = max(1, int(sequential_overlap))
        for i in range(n_images):
            for j in range(i + 1, min(n_images, i + 1 + overlap)):
                pairs.append((i, j))

    elif matcher_type == "neighbor":
        # Neighbor matching: optimized for flatbed scans and similar layouts
        # Each image matches with K nearest neighbors (by filename sort order)
        # This assumes images are named in a sensible order (scan_001, scan_002, etc.)
        #
        # For flatbed scans with row-by-row or snake patterns:
        # - K=20 catches immediate neighbors and row transitions
        # - K=30-50 for large overlap or irregular patterns

        k_neighbors = max(10, int(sequential_overlap))  # Use sequential_overlap as K

        # Also add "skip connections" for robustness - connect to images further away
        # This helps when there are missed matches with immediate neighbors
        skip_interval = max(k_neighbors, n_images // 50)  # Every ~2% of images

        for i in range(n_images):
            # Match with K nearest neighbors (forward only to avoid duplicates)
            for j in range(i + 1, min(n_images, i + 1 + k_neighbors)):
                pairs.append((i, j))

            # Add skip connections for global consistency
            # These help detect/correct accumulated drift
            for skip in range(skip_interval, n_images - i, skip_interval):
                j = i + skip
                if j < n_images and (i, j) not in pairs:
                    pairs.append((i, j))

        # Remove duplicates and sort
        pairs = sorted(set(pairs))

        exhaustive_count = n_images * (n_images - 1) // 2
        reduction = (1 - len(pairs) / exhaustive_count) * 100
        print(f"[NEIGHBOR] K={k_neighbors} neighbors + skip connections: "
              f"{len(pairs):,} pairs vs {exhaustive_count:,} exhaustive ({reduction:.1f}% reduction)",
              file=sys.stderr, flush=True)

    else:
        # Exhaustive (default) - match all pairs
        for i in range(n_images):
            for j in range(i + 1, n_images):
                pairs.append((i, j))

    return pairs


def _match_single_pair(args):
    """
    Match a single pair of images. Used for parallel matching.

    Args:
        args: Tuple of (i, j, kp1, d1, kp2, d2, ratio, ransac_thresh, min_inliers, min_ratio, use_affine)

    Returns:
        Tuple of (i, j, inlier_pairs, n_putative, n_inliers, inlier_ratio) or None if failed

    OPTIMIZED: Uses FLANN matcher (~5x faster than BFMatcher)
    """
    import cv2
    import numpy as np

    (i, j, kp1, d1, kp2, d2, ratio, ransac_thresh, min_inliers, min_ratio, use_affine) = args

    if d1.shape[0] < 2 or d2.shape[0] < 2:
        return None

    try:
        # Use FLANN for ~5x faster matching than BFMatcher
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=4)
        search_params = dict(checks=32)  # Lower = faster
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        # Forward matching with ratio test
        knn_f = flann.knnMatch(d1, d2, k=2)
        good_f = {}
        for m_n in knn_f:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < ratio * n.distance:
                    good_f[(m.queryIdx, m.trainIdx)] = m

        # Reverse matching for mutual consistency
        knn_r = flann.knnMatch(d2, d1, k=2)
        good = []
        for m_n in knn_r:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < ratio * n.distance:
                    mm = good_f.get((m.trainIdx, m.queryIdx))
                    if mm is not None:
                        good.append(mm)

        if len(good) < min_inliers:
            return None

        pts1 = np.float32([kp1[m.queryIdx] for m in good])
        pts2 = np.float32([kp2[m.trainIdx] for m in good])

        # Use similarity transform for flatbed scans (constrains scale)
        if use_affine:
            # estimateAffinePartial2D gives similarity transform (scale + rotation + translation)
            # This is better for flatbed scans than full affine
            _, inlier_mask = cv2.estimateAffinePartial2D(
                pts1, pts2, method=cv2.RANSAC,
                ransacReprojThreshold=ransac_thresh,
                maxIters=500, confidence=0.99)
        else:
            _, inlier_mask = cv2.findHomography(
                pts1, pts2, method=cv2.RANSAC,
                ransacReprojThreshold=ransac_thresh)

        if inlier_mask is None:
            return None

        inlier_mask = inlier_mask.ravel().astype(bool)
        n_inliers = int(inlier_mask.sum())

        if n_inliers < min_inliers:
            return None

        inlier_ratio = n_inliers / max(1, len(good))
        if inlier_ratio < min_ratio:
            return None

        pairs_idx = np.array([[m.queryIdx, m.trainIdx] for m in good], dtype=np.uint32)
        inlier_pairs = pairs_idx[inlier_mask]

        return (i, j, inlier_pairs, len(good), n_inliers, inlier_ratio)

    except Exception:
        return None


def _write_two_view_geometry(conn, image_id1: int, image_id2: int, matches: np.ndarray) -> None:
    pair_id = _pair_id(image_id1, image_id2)
    rows = int(matches.shape[0])
    cols = 2
    data_blob = matches.astype(np.uint32).tobytes()
    conn.execute(
        """
        INSERT OR REPLACE INTO two_view_geometries
        (pair_id, rows, cols, data, config, F, E, H)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (pair_id, rows, cols, data_blob, 0, None, None, None),
    )


def match_pairs_incremental(
    database_path: Path,
    image_paths: List[Path],
    max_features: int,
    cache_manager,
    matcher_type: str,
    sequential_overlap: int,
    use_affine: bool,
    cached_features: Dict[int, Dict],
    newly_extracted: Dict[int, Dict],
):
    """
    Incremental matching with per-pair match cache:
    - Load cached inlier correspondences when available and write to `two_view_geometries`
    - Otherwise compute matches via OpenCV + RANSAC, save to match cache, write to DB

    OPTIMIZED VERSION:
    - Uses parallel processing for uncached pairs (multiprocessing)
    - Pre-loads all features before parallel matching
    - Provides significant speedup for large image sets
    """
    import cv2
    import sqlite3
    from multiprocessing import Pool, cpu_count

    n_images = len(image_paths)
    pairs = _generate_pairs(n_images, matcher_type, sequential_overlap, image_paths)
    total_pairs = len(pairs)

    # Match params
    ratio = 0.75
    ransac_thresh = 3.0
    min_inliers_to_store = 30
    min_inlier_ratio = 0.20
    model_tag = "affine" if use_affine else "homography"
    matcher_id = f"{matcher_type}_{model_tag}_v3_r{ratio:.2f}_t{ransac_thresh:.1f}_min{min_inliers_to_store}"

    print(f"[CACHE] Match caching enabled: {total_pairs:,} pairs ({matcher_id})", file=sys.stderr, flush=True)
    print(f"[CACHE] Match cache directory: {cache_manager.matches_dir}", file=sys.stderr, flush=True)

    cache_manager.note_match_run_start()

    # Build image_key map
    image_keys: Dict[int, str] = {}
    for i, img_path in enumerate(image_paths):
        if i in cached_features:
            image_keys[i] = cached_features[i]["cache_key"]
        elif i in newly_extracted and newly_extracted[i].get("cache_key"):
            image_keys[i] = newly_extracted[i]["cache_key"]
        else:
            image_keys[i] = cache_manager.compute_image_cache_key(img_path, max_features)

    # Feature loader (memoized)
    feat_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    def load_feats(idx: int) -> Tuple[np.ndarray, np.ndarray]:
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
            ck = image_keys[idx]
            kp, desc, _ = cache_manager.load_features(ck)

        if kp is None or desc is None:
            raise RuntimeError(f"No features available for image {idx}")

        kp_xy = kp[:, :2].astype(np.float32)
        desc_f = desc.astype(np.float32)
        feat_cache[idx] = (kp_xy, desc_f)
        return feat_cache[idx]

    conn = sqlite3.connect(str(database_path))
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")

    # Graceful shutdown handler - commit database on interrupt
    shutdown_requested = [False]
    original_sigint = signal.getsignal(signal.SIGINT)
    original_sigterm = signal.getsignal(signal.SIGTERM)

    def graceful_shutdown(sig, frame):
        if not shutdown_requested[0]:
            shutdown_requested[0] = True
            print(f"\n[SHUTDOWN] Signal {sig} received, committing database and exiting...",
                  file=sys.stderr, flush=True)
            try:
                conn.commit()
                print("[SHUTDOWN] Database committed successfully. Matches saved to cache are preserved.",
                      file=sys.stderr, flush=True)
            except Exception as e:
                print(f"[SHUTDOWN] Warning: Could not commit database: {e}", file=sys.stderr, flush=True)
            sys.exit(128 + sig)

    signal.signal(signal.SIGINT, graceful_shutdown)
    signal.signal(signal.SIGTERM, graceful_shutdown)

    # Phase 1: Check cache and collect uncached pairs
    print(f"[PROGRESS] Phase 1: Checking match cache for {total_pairs:,} pairs...", file=sys.stderr, flush=True)
    cached_count = 0
    uncached_pairs = []

    for i, j in pairs:
        img1_key = image_keys[i]
        img2_key = image_keys[j]

        cached = cache_manager.get_cached_match(img1_key, img2_key, matcher_id)
        if cached:
            try:
                matches_arr, _meta = cache_manager.load_match(cached["cache_key"])
                if matches_arr.size > 0:
                    _write_two_view_geometry(conn, i + 1, j + 1, matches_arr)
                cache_manager.note_match_hit(1)
                cached_count += 1
                continue
            except Exception:
                pass
        uncached_pairs.append((i, j))

    print(f"[CACHE] {cached_count:,} pairs from cache, {len(uncached_pairs):,} pairs need matching",
          file=sys.stderr, flush=True)

    if not uncached_pairs:
        conn.commit()
        conn.close()
        summary = cache_manager.get_match_lookup_summary()
        print(f"[CACHE] Match cache summary: {summary['hit']} hit, {summary['computed']} computed, {summary['skipped']} skipped",
              file=sys.stderr, flush=True)
        return

    # Phase 2: Pre-load all features needed for uncached pairs
    print(f"[PROGRESS] Phase 2: Pre-loading features...", file=sys.stderr, flush=True)
    needed_indices = set()
    for i, j in uncached_pairs:
        needed_indices.add(i)
        needed_indices.add(j)

    for idx in needed_indices:
        try:
            load_feats(idx)
        except Exception as e:
            print(f"[WARNING] Could not load features for image {idx}: {e}", file=sys.stderr, flush=True)

    print(f"[PROGRESS] Loaded features for {len(feat_cache)} images", file=sys.stderr, flush=True)

    # Phase 3: Parallel matching for uncached pairs
    # OPTIMIZED: Process in batches with incremental saving to avoid memory issues
    # and allow resumption after interruption
    n_workers = min(cpu_count(), 8)  # Cap at 8 workers
    use_parallel = len(uncached_pairs) > 100 and n_workers > 1

    if use_parallel:
        # Batch size: balance memory usage vs. efficiency
        # Each pair in match_args contains two sets of keypoints+descriptors
        # Memory per pair ≈ 2 * (num_features * (6 + 128) bytes) ≈ 1-2 MB
        # For 1000 pairs: ~1-2 GB; for 5000 pairs: ~5-10 GB
        #
        # Auto-adjust batch size based on available memory
        try:
            import psutil
            available_gb = psutil.virtual_memory().available / (1024**3)
            # Use ~25% of available memory for batch, assuming ~1.5MB per pair
            estimated_pairs = int(available_gb * 0.25 * 1024 / 1.5)
            BATCH_SIZE = max(200, min(5000, estimated_pairs))
            print(f"[MEMORY] Available: {available_gb:.1f} GB, batch size: {BATCH_SIZE} pairs",
                  file=sys.stderr, flush=True)
        except ImportError:
            BATCH_SIZE = 1000  # Conservative default without psutil
            print(f"[MEMORY] psutil not available, using conservative batch size: {BATCH_SIZE}",
                  file=sys.stderr, flush=True)

        total_uncached = len(uncached_pairs)
        n_batches = (total_uncached + BATCH_SIZE - 1) // BATCH_SIZE

        print(f"[PROGRESS] Phase 3: Batched parallel matching {total_uncached:,} pairs "
              f"in {n_batches} batches ({BATCH_SIZE} pairs/batch) with {n_workers} workers...",
              file=sys.stderr, flush=True)

        total_computed = 0
        total_skipped = 0
        batch_start_time = time.time()

        for batch_num in range(n_batches):
            batch_start = batch_num * BATCH_SIZE
            batch_end = min(batch_start + BATCH_SIZE, total_uncached)
            batch_pairs = uncached_pairs[batch_start:batch_end]

            # Prepare arguments for this batch only
            match_args = []
            for i, j in batch_pairs:
                if i in feat_cache and j in feat_cache:
                    kp1, d1 = feat_cache[i]
                    kp2, d2 = feat_cache[j]
                    match_args.append((i, j, kp1, d1, kp2, d2, ratio, ransac_thresh,
                                       min_inliers_to_store, min_inlier_ratio, use_affine))

            if not match_args:
                continue

            # Process batch with parallel matching
            batch_computed = 0
            batch_skipped = 0

            with Pool(n_workers) as pool:
                chunk_size = max(1, len(match_args) // (n_workers * 4))
                items_since_commit = 0
                COMMIT_INTERVAL = 500  # Commit every 500 pairs for durability

                # Process results incrementally within batch
                for result in pool.imap_unordered(_match_single_pair, match_args, chunksize=chunk_size):
                    # Check for shutdown request
                    if shutdown_requested[0]:
                        print("[SHUTDOWN] Shutdown requested, stopping batch processing...",
                              file=sys.stderr, flush=True)
                        break

                    if result is None:
                        batch_skipped += 1
                        cache_manager.note_match_skipped(1)
                        continue

                    i, j, inlier_pairs, n_putative, n_inliers, inlier_ratio = result
                    img1_key = image_keys[i]
                    img2_key = image_keys[j]

                    # Save to cache immediately (incremental)
                    meta = {
                        "n_putative": n_putative,
                        "n_inliers": n_inliers,
                        "inlier_ratio": inlier_ratio,
                        "ratio": ratio,
                        "ransac_thresh": ransac_thresh,
                        "model": model_tag,
                    }
                    cache_manager.save_match(img1_key, img2_key, matcher_id, inlier_pairs, meta)
                    _write_two_view_geometry(conn, i + 1, j + 1, inlier_pairs)
                    cache_manager.note_match_computed(1)
                    batch_computed += 1
                    items_since_commit += 1

                    # Periodic commit for durability
                    if items_since_commit >= COMMIT_INTERVAL:
                        conn.commit()
                        items_since_commit = 0

            # Commit database after each batch for durability
            conn.commit()

            # Check for shutdown after batch
            if shutdown_requested[0]:
                print(f"[SHUTDOWN] Exiting after batch {batch_num + 1}. Progress saved.",
                      file=sys.stderr, flush=True)
                break

            total_computed += batch_computed
            total_skipped += batch_skipped

            # Progress report
            elapsed = time.time() - batch_start_time
            pairs_done = batch_end
            pairs_per_sec = pairs_done / max(1, elapsed)
            eta_sec = (total_uncached - pairs_done) / max(1, pairs_per_sec)

            print(f"[PROGRESS] Batch {batch_num + 1}/{n_batches} complete: "
                  f"{pairs_done:,}/{total_uncached:,} pairs ({100*pairs_done//total_uncached}%), "
                  f"{pairs_per_sec:.1f} pairs/sec, ETA: {eta_sec/60:.1f} min",
                  file=sys.stderr, flush=True)

            # Clear batch data to free memory
            del match_args

        print(f"[PROGRESS] Parallel matching complete: {total_computed} computed, {total_skipped} skipped",
              file=sys.stderr, flush=True)

    else:
        # Sequential matching for small sets
        print(f"[PROGRESS] Phase 3: Sequential matching {len(uncached_pairs):,} pairs...",
              file=sys.stderr, flush=True)

        bf = cv2.BFMatcher(cv2.NORM_L2)
        last_report = time.time()

        for k, (i, j) in enumerate(uncached_pairs):
            now = time.time()
            if now - last_report >= 2.0 or k == 0 or k == len(uncached_pairs) - 1:
                pct = int(100 * (k / max(1, len(uncached_pairs))))
                print(f"[PROGRESS] Matching: {k:,}/{len(uncached_pairs):,} ({pct}%)",
                      file=sys.stderr, flush=True)
                last_report = now

            img1_key = image_keys[i]
            img2_key = image_keys[j]

            try:
                kp1, d1 = feat_cache.get(i, (None, None))
                kp2, d2 = feat_cache.get(j, (None, None))
                if d1 is None or d2 is None or d1.shape[0] < 2 or d2.shape[0] < 2:
                    cache_manager.note_match_skipped(1)
                    continue

                # Forward matching
                knn_f = bf.knnMatch(d1, d2, k=2)
                good_f = {}
                for m_n in knn_f:
                    if len(m_n) == 2:
                        m, n = m_n
                        if m.distance < ratio * n.distance:
                            good_f[(m.queryIdx, m.trainIdx)] = m

                # Reverse + mutual
                knn_r = bf.knnMatch(d2, d1, k=2)
                good = []
                for m_n in knn_r:
                    if len(m_n) == 2:
                        m, n = m_n
                        if m.distance < ratio * n.distance:
                            mm = good_f.get((m.trainIdx, m.queryIdx))
                            if mm is not None:
                                good.append(mm)

                if len(good) < min_inliers_to_store:
                    cache_manager.note_match_skipped(1)
                    continue

                pts1 = np.float32([kp1[m.queryIdx] for m in good])
                pts2 = np.float32([kp2[m.trainIdx] for m in good])

                if use_affine:
                    # Use similarity transform (4 DOF) - no shear for flatbed scans
                    _, inlier_mask = cv2.estimateAffinePartial2D(
                        pts1, pts2, method=cv2.RANSAC,
                        ransacReprojThreshold=ransac_thresh,
                        maxIters=1000, confidence=0.99)
                else:
                    _, inlier_mask = cv2.findHomography(
                        pts1, pts2, method=cv2.RANSAC,
                        ransacReprojThreshold=ransac_thresh)

                if inlier_mask is None:
                    cache_manager.note_match_skipped(1)
                    continue

                inlier_mask = inlier_mask.ravel().astype(bool)
                n_inliers = int(inlier_mask.sum())
                if n_inliers < min_inliers_to_store:
                    cache_manager.note_match_skipped(1)
                    continue

                inlier_ratio = n_inliers / max(1, len(good))
                if inlier_ratio < min_inlier_ratio:
                    cache_manager.note_match_skipped(1)
                    continue

                pairs_idx = np.array([[m.queryIdx, m.trainIdx] for m in good], dtype=np.uint32)
                inlier_pairs = pairs_idx[inlier_mask]

                meta = {
                    "n_putative": len(good),
                    "n_inliers": len(inlier_pairs),
                    "inlier_ratio": inlier_ratio,
                    "ratio": float(ratio),
                    "ransac_thresh": float(ransac_thresh),
                    "model": model_tag,
                }
                cache_manager.save_match(img1_key, img2_key, matcher_id, inlier_pairs, meta)
                _write_two_view_geometry(conn, i + 1, j + 1, inlier_pairs)
                cache_manager.note_match_computed(1)
            except Exception as e:
                cache_manager.note_match_skipped(1)
                print(f"[WARNING] Match compute failed for ({i},{j}): {e}", file=sys.stderr, flush=True)

    conn.commit()
    conn.close()

    summary = cache_manager.get_match_lookup_summary()
    print(
        f"[CACHE] Match cache summary: {summary['hit']} hit, {summary['computed']} computed, {summary['skipped']} skipped in {summary['time']:.1f}s",
        file=sys.stderr,
        flush=True,
    )


def extract_features_incremental(
    uncached_images: List[Tuple[int, Path]],
    workspace_dir: Path,
    max_features: int,
    cache_manager,
    gpu_index: int = -1,
    num_threads: int = -1,
    has_cuda: bool = False,
) -> Dict[int, Dict]:
    """Extract features only for uncached images (via a temp db) and save to per-image cache."""
    import pycolmap
    import sqlite3

    if not uncached_images:
        return {}

    temp_images_dir = workspace_dir / "temp_extract"
    temp_images_dir.mkdir(parents=True, exist_ok=True)
    temp_db_path = workspace_dir / "temp_extract.db"

    index_to_tempname: Dict[int, str] = {}
    tempname_to_index: Dict[str, int] = {}

    for idx, img_path in uncached_images:
        temp_name = f"{idx:06d}{img_path.suffix}"
        temp_dest = temp_images_dir / temp_name
        shutil.copy2(img_path, temp_dest)
        index_to_tempname[idx] = temp_name
        tempname_to_index[temp_name] = idx

    n_uncached = len(uncached_images)
    print(f"[CACHE] Extracting features for {n_uncached} uncached images", file=sys.stderr, flush=True)

    sift_options = pycolmap.SiftExtractionOptions()
    sift_options.max_num_features = int(max_features)
    sift_options.first_octave = 0
    sift_options.num_octaves = 4

    extraction_options = pycolmap.FeatureExtractionOptions()
    extraction_options.sift = sift_options

    if gpu_index >= 0 and has_cuda:
        device = pycolmap.Device(f"cuda:{gpu_index}")
    elif has_cuda:
        device = pycolmap.Device.auto
    else:
        device = pycolmap.Device.cpu

    t0 = time.time()
    pycolmap.extract_features(
        temp_db_path,
        temp_images_dir,
        extraction_options=extraction_options,
        camera_mode=pycolmap.CameraMode.PER_IMAGE,
        camera_model="PINHOLE",
        device=device,
    )
    dt = time.time() - t0
    print(f"[CACHE] Extracted {n_uncached} new features in {dt:.1f}s", file=sys.stderr, flush=True)

    conn = sqlite3.connect(str(temp_db_path))
    cur = conn.cursor()
    cur.execute("SELECT image_id, name FROM images")
    image_id_map = {name: int(img_id) for img_id, name in cur.fetchall()}

    out: Dict[int, Dict] = {}

    for idx, img_path in uncached_images:
        temp_name = index_to_tempname[idx]
        img_id = image_id_map.get(temp_name)
        if img_id is None:
            print(f"[WARNING] Image {temp_name} not found in temp db", file=sys.stderr, flush=True)
            continue

        cur.execute("SELECT data FROM keypoints WHERE image_id = ?", (img_id,))
        kp_row = cur.fetchone()
        cur.execute("SELECT data FROM descriptors WHERE image_id = ?", (img_id,))
        desc_row = cur.fetchone()

        if not kp_row or not desc_row:
            print(f"[WARNING] Missing features in temp db for {temp_name}", file=sys.stderr, flush=True)
            continue

        keypoints = np.frombuffer(kp_row[0], dtype=np.float32).reshape(-1, 6).copy()
        descriptors = np.frombuffer(desc_row[0], dtype=np.uint8).reshape(-1, 128).copy()

        cache_key = None
        try:
            meta = {"image_name": img_path.name, "extracted_at": time.time(), "device": str(device)}
            cache_key = cache_manager.save_features(img_path, max_features, keypoints, descriptors, meta)
        except Exception as e:
            print(f"[WARNING] Failed to save features to cache for {img_path.name}: {e}", file=sys.stderr, flush=True)

        out[idx] = {
            "cache_key": cache_key,
            "keypoints": keypoints,
            "descriptors": descriptors,
            "image_path": img_path,
        }

    conn.close()

    shutil.rmtree(temp_images_dir, ignore_errors=True)
    try:
        if temp_db_path.exists():
            temp_db_path.unlink()
    except Exception:
        pass

    return out


def build_database_from_cache(
    database_path: Path,
    image_paths: List[Path],
    cached_features: Dict[int, Dict],
    newly_extracted: Dict[int, Dict],
    cache_manager,
    images_dir: Path,
    max_features: int,
) -> bool:
    """
    Build a COLMAP database populated with keypoints/descriptors for all images.
    Uses pycolmap to initialize schema, then replaces/inserts feature blobs.
    """
    import sqlite3
    from PIL import Image
    import pycolmap

    print(
        f"[CACHE] Building database from cache: {len(cached_features)} cached + {len(newly_extracted)} new",
        file=sys.stderr,
        flush=True,
    )

    if database_path.exists():
        database_path.unlink()

    # Init DB structure via pycolmap on a single image (ensures correct schema)
    temp_init_dir = images_dir.parent / "db_init"
    temp_init_dir.mkdir(exist_ok=True)
    try:
        first_img = image_paths[0]
        temp_img_path = temp_init_dir / f"000000{first_img.suffix}"
        shutil.copy2(first_img, temp_img_path)

        sift_opts = pycolmap.SiftExtractionOptions()
        sift_opts.max_num_features = int(max_features)
        sift_opts.first_octave = -1
        sift_opts.num_octaves = 4
        sift_opts.octave_resolution = 3

        extraction_opts = pycolmap.FeatureExtractionOptions()
        extraction_opts.sift = sift_opts

        pycolmap.extract_features(
            str(database_path),
            str(temp_init_dir),
            extraction_options=extraction_opts,
            camera_mode=pycolmap.CameraMode.SINGLE,
            camera_model="PINHOLE",
        )
    finally:
        shutil.rmtree(temp_init_dir, ignore_errors=True)

    conn = sqlite3.connect(str(database_path))
    cur = conn.cursor()

    cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS index_images_name ON images(name)")
    conn.commit()

    # Reuse pycolmap camera if present (otherwise create as needed)
    cur.execute("SELECT camera_id, width, height FROM cameras LIMIT 1")
    row = cur.fetchone()
    camera_id_map: Dict[Tuple[int, int], int] = {}
    if row:
        cam_id, w0, h0 = int(row[0]), int(row[1]), int(row[2])
        camera_id_map[(w0, h0)] = cam_id

    # Add/update images
    camera_model = 1  # PINHOLE
    for i, img_path in enumerate(image_paths):
        with Image.open(img_path) as im:
            width, height = im.size
        cam_key = (width, height)
        if cam_key not in camera_id_map:
            focal = max(width, height) * 1.2
            params = np.array([focal, focal, width / 2, height / 2], dtype=np.float64).tobytes()
            cam_id = len(camera_id_map) + 1
            cur.execute(
                """
                INSERT OR REPLACE INTO cameras (camera_id, model, width, height, params, prior_focal_length)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (cam_id, camera_model, width, height, params, 0),
            )
            camera_id_map[cam_key] = cam_id
        else:
            cam_id = camera_id_map[cam_key]

        image_name = f"{i:06d}{img_path.suffix}"
        cur.execute("INSERT OR REPLACE INTO images (image_id, name, camera_id) VALUES (?, ?, ?)", (i + 1, image_name, cam_id))

    conn.commit()

    # Populate keypoints/descriptors
    for i, _img_path in enumerate(image_paths):
        keypoints = None
        descriptors = None

        if i in newly_extracted and "keypoints" in newly_extracted[i] and "descriptors" in newly_extracted[i]:
            keypoints = newly_extracted[i]["keypoints"]
            descriptors = newly_extracted[i]["descriptors"]
        elif i in cached_features:
            ck = cached_features[i]["cache_key"]
            keypoints, descriptors, _ = cache_manager.load_features(ck)
        else:
            # fallback: should not happen, but try loading by recomputed key
            ck = cache_manager.compute_image_cache_key(image_paths[i], max_features)
            keypoints, descriptors, _ = cache_manager.load_features(ck)

        if keypoints is None or descriptors is None:
            print(f"[ERROR] No features available for image {i}", file=sys.stderr, flush=True)
            continue

        kp_blob = keypoints.astype(np.float32).tobytes()
        desc_blob = descriptors.astype(np.uint8).tobytes()
        n_kp = int(len(keypoints))
        n_desc = int(len(descriptors))

        if i == 0:
            cur.execute("UPDATE keypoints SET rows=?, cols=?, data=? WHERE image_id=1", (n_kp, 6, kp_blob))
            cur.execute("UPDATE descriptors SET rows=?, cols=?, data=? WHERE image_id=1", (n_desc, 128, desc_blob))
        else:
            cur.execute("INSERT OR REPLACE INTO keypoints (image_id, rows, cols, data) VALUES (?, ?, ?, ?)", (i + 1, n_kp, 6, kp_blob))
            cur.execute("INSERT OR REPLACE INTO descriptors (image_id, rows, cols, data) VALUES (?, ?, ?, ?)", (i + 1, n_desc, 128, desc_blob))

    conn.commit()
    conn.close()

    # Ensure durability on WSL filesystem
    try:
        fd = os.open(str(database_path), os.O_RDWR)
        os.fsync(fd)
        os.close(fd)
    except Exception:
        pass

    return True


def run_2d_stitch(
    image_paths_json: str,
    output_dir: str,
    use_affine: bool = False,
    blend_method: str = "multiband",
    matcher_type: str = "exhaustive",
    sequential_overlap: int = 10,
    gpu_index: int = -1,
    num_threads: int = -1,
    max_features: int = 4096,  # Reduced from 8192 for faster processing
    min_inliers: int = 0,
    max_images: int = 0,
    use_source_alpha: bool = False,
    remove_duplicates: bool = False,
    duplicate_threshold: float = 0.92,
    warp_interpolation: str = "linear",
    erode_border: bool = True,
    border_erosion_pixels: int = 5,
    auto_detect_content: bool = False,
) -> Dict:
    import pycolmap
    import cv2

    def signal_handler(sig, _frame):
        print(f"[DEBUG] Received signal {sig}, shutting down gracefully...", file=sys.stderr, flush=True)
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print(
        f"[DEBUG] WSL bridge called with: max_features={max_features}, matcher_type={matcher_type}, blend_method={blend_method}",
        file=sys.stderr,
        flush=True,
    )

    start_time = time.time()

    image_paths_win = json.loads(image_paths_json)
    image_paths = [Path(windows_to_wsl_path(p)) for p in image_paths_win]
    output_path = Path(windows_to_wsl_path(output_dir))

    # CRITICAL: Sort image paths to ensure consistent ordering between runs.
    # The cache key is computed from sorted names, so we must also use sorted
    # paths for the index mapping. Otherwise, a cache hit with different
    # image ordering will map homographies to wrong images!
    image_paths = sorted(image_paths, key=lambda x: x.name)

    n_images_original = len(image_paths)

    # Optional duplicate removal (kept as-is; non-fatal if missing deps)
    n_duplicates_removed = 0
    if remove_duplicates and n_images_original > 1:
        print(
            f"[PROGRESS] Scanning {n_images_original} images for duplicates (threshold={duplicate_threshold:.2f})...",
            file=sys.stderr,
            flush=True,
        )
        try:
            script_dir = Path(__file__).resolve().parent.parent
            if str(script_dir) not in sys.path:
                sys.path.insert(0, str(script_dir))
            from utils.duplicate_detector import DuplicateDetector

            def dup_progress(_pct: int, msg: str):
                print(f"[PROGRESS] Dedup: {msg}", file=sys.stderr, flush=True)

            detector = DuplicateDetector(similarity_threshold=duplicate_threshold, comparison_window=0, progress_callback=dup_progress)
            keep_indices, _pairs = detector.find_duplicates([None] * n_images_original, image_paths)
            n_removed = n_images_original - len(keep_indices)
            n_duplicates_removed = n_removed
            if n_removed > 0:
                image_paths = [image_paths[i] for i in keep_indices]
                print(f"[PROGRESS] Removed {n_removed} duplicate images", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"[WARNING] Duplicate detection failed: {e}. Proceeding with all images.", file=sys.stderr, flush=True)

    n_images = len(image_paths)
    print(f"[PROGRESS] WSL COLMAP Bridge - GPU Mode (has_cuda={pycolmap.has_cuda})", file=sys.stderr, flush=True)
    print(f"[PROGRESS] Processing {n_images} images", file=sys.stderr, flush=True)

    # Persistent caches in WSL home
    # Try new v4 cache key first (doesn't include blend settings)
    cache_key = compute_cache_key(
        image_paths,
        max_features,
        matcher_type=matcher_type,
    )
    cache_base = Path(os.environ.get("STITCH2STITCH_CACHE_DIR", str(Path.home() / ".stitch2stitch_cache")))
    cache_root = cache_base / "colmap"
    cache_root.mkdir(parents=True, exist_ok=True)

    global_cache_dir = cache_root / cache_key

    # Backward compatibility: check for legacy v2 cache if v4 doesn't exist
    cache_key_v2 = compute_cache_key_v2(
        image_paths,
        max_features,
        matcher_type=matcher_type,
        use_affine=use_affine,
        blend_method=blend_method,
        warp_interpolation=warp_interpolation,
    )
    global_cache_dir_v2 = cache_root / cache_key_v2

    # Check which cache exists
    if is_cache_valid(global_cache_dir, cache_key, n_images):
        print(f"[DEBUG] Using v4 cache: {cache_key}", file=sys.stderr, flush=True)
    elif is_cache_valid(global_cache_dir_v2, cache_key_v2, n_images):
        print(f"[DEBUG] Using legacy v2 cache: {cache_key_v2}", file=sys.stderr, flush=True)
        # Use the v2 cache
        cache_key = cache_key_v2
        global_cache_dir = global_cache_dir_v2
    else:
        # No cache found - will create new v4 cache
        print(f"[DEBUG] No cache found, creating new v4 cache: {cache_key}", file=sys.stderr, flush=True)
        global_cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"[DEBUG] Cache key: {cache_key} (max_features={max_features})", file=sys.stderr, flush=True)
    print(f"[DEBUG] Global cache directory: {global_cache_dir}", file=sys.stderr, flush=True)

    # Workspace on WSL native filesystem
    workspace_root = Path(tempfile.gettempdir()) / ".stitch2stitch_workspaces"
    workspace_root.mkdir(parents=True, exist_ok=True)
    workspace = workspace_root / f"{cache_key}_run_{int(time.time())}"
    if workspace.exists():
        shutil.rmtree(workspace, ignore_errors=True)
    workspace.mkdir(parents=True, exist_ok=True)

    images_dir = workspace / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    database_path = workspace / "database.db"

    use_cache = is_cache_valid(global_cache_dir, cache_key, n_images)
    if use_cache:
        print("[PROGRESS] *** USING CACHED FEATURES (skipping extraction & matching) ***", file=sys.stderr, flush=True)
        print(f"[CACHE] Global cache hit: skipping per-image feature cache and per-pair match cache", file=sys.stderr, flush=True)

        cached_db = global_cache_dir / "database.db"
        if cached_db.exists():
            shutil.copy2(cached_db, database_path)
        cached_images = global_cache_dir / "images"
        if cached_images.exists():
            shutil.rmtree(images_dir, ignore_errors=True)
            shutil.copytree(cached_images, images_dir)
    else:
        print("[PROGRESS] Cache invalid or not found - will extract features", file=sys.stderr, flush=True)
        shutil.rmtree(images_dir, ignore_errors=True)
        images_dir.mkdir(parents=True, exist_ok=True)

        # Parallel copy images into workspace (COLMAP expects local names)
        print(f"[PROGRESS] Copying {n_images} images to workspace (parallel)...", file=sys.stderr, flush=True)
        copy_args = [(i, p, images_dir) for i, p in enumerate(image_paths)]
        n_workers = min(8, n_images)
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = {ex.submit(copy_single_image, a): a[0] for a in copy_args}
            done = 0
            for fut in as_completed(futures):
                fut.result()
                done += 1
                if done % max(1, n_images // 10) == 0 or done == n_images:
                    print(f"[PROGRESS] Copied {done}/{n_images} images", file=sys.stderr, flush=True)

    # Build name mappings for downstream steps
    image_name_to_index: Dict[str, int] = {f"{i:06d}{p.suffix}": i for i, p in enumerate(image_paths)}

    has_cuda = bool(pycolmap.has_cuda) if hasattr(pycolmap, "has_cuda") else False
    print(f"[DEBUG] pycolmap.has_cuda = {has_cuda}", file=sys.stderr, flush=True)

    if not use_cache:
        # Init cache manager
        try:
            src_path = Path(__file__).parent.parent
            if str(src_path) not in sys.path:
                sys.path.insert(0, str(src_path))
            from external.colmap_cache_manager import COLMAPCacheManager

            per_image_cache_root = cache_root / "per_image"
            per_image_cache_root.mkdir(parents=True, exist_ok=True)
            cache_manager = COLMAPCacheManager(per_image_cache_root)
            print(f"[CACHE] Initialized per-image/match cache at {per_image_cache_root}", file=sys.stderr, flush=True)
            use_per_image_cache = True
        except Exception as e:
            print(f"[WARNING] Could not import COLMAPCacheManager: {e}. Using traditional extraction.", file=sys.stderr, flush=True)
            cache_manager = None
            use_per_image_cache = False

        cached_features: Dict[int, Dict] = {}
        uncached_images: List[Tuple[int, Path]] = []
        if use_per_image_cache and cache_manager:
            print(f"[CACHE] Checking per-image cache for {n_images} images...", file=sys.stderr, flush=True)
            cached_features, uncached_images = check_feature_cache(image_paths, max_features, cache_manager)
            print(f"[CACHE] Found {len(cached_features)} cached, {len(uncached_images)} need extraction", file=sys.stderr, flush=True)
        else:
            uncached_images = list(enumerate(image_paths))

        newly_extracted: Dict[int, Dict] = {}
        if uncached_images and use_per_image_cache and cache_manager:
            newly_extracted = extract_features_incremental(
                uncached_images,
                workspace,
                max_features,
                cache_manager,
                gpu_index=gpu_index,
                num_threads=num_threads,
                has_cuda=has_cuda,
            )
        elif uncached_images:
            # Fallback: full pycolmap extraction on all images
            import pycolmap

            sift_options = pycolmap.SiftExtractionOptions()
            sift_options.max_num_features = int(max_features)
            sift_options.first_octave = 0
            sift_options.num_octaves = 4
            extraction_options = pycolmap.FeatureExtractionOptions()
            extraction_options.sift = sift_options

            if gpu_index >= 0 and has_cuda:
                device = pycolmap.Device(f"cuda:{gpu_index}")
            elif has_cuda:
                device = pycolmap.Device.auto
            else:
                device = pycolmap.Device.cpu

            print("[PROGRESS] Starting feature extraction (fallback)...", file=sys.stderr, flush=True)
            pycolmap.extract_features(
                database_path,
                images_dir,
                extraction_options=extraction_options,
                camera_mode=pycolmap.CameraMode.PER_IMAGE,
                camera_model="PINHOLE",
                device=device,
            )

        # Build database from cached+new features
        if use_per_image_cache and cache_manager:
            build_database_from_cache(
                database_path,
                image_paths,
                cached_features,
                newly_extracted,
                cache_manager,
                images_dir,
                max_features=max_features,
            )

            # Match via per-pair match cache
            print("[PROGRESS] WSL: Matching via per-pair cache (incremental)", file=sys.stderr, flush=True)
            match_pairs_incremental(
                database_path=database_path,
                image_paths=image_paths,
                max_features=max_features,
                cache_manager=cache_manager,
                matcher_type=matcher_type,
                sequential_overlap=sequential_overlap,
                use_affine=use_affine,
                cached_features=cached_features,
                newly_extracted=newly_extracted,
            )
        else:
            # Fallback: pycolmap matching
            if matcher_type == "sequential":
                pycolmap.match_sequential(database_path, overlap=int(sequential_overlap))
            else:
                pycolmap.match_exhaustive(database_path)

        # Save to global cache
        print("[PROGRESS] Saving features to global cache...", file=sys.stderr, flush=True)
        shutil.copy2(database_path, global_cache_dir / "database.db")
        cached_images = global_cache_dir / "images"
        shutil.rmtree(cached_images, ignore_errors=True)
        shutil.copytree(images_dir, cached_images)
        save_cache_key(global_cache_dir, cache_key)

    # Continue with existing stitch pipeline using DB matches
    print("[PROGRESS] Reading matches from database...", file=sys.stderr, flush=True)
    matches_data = read_matches_from_db(database_path)
    if not matches_data:
        return {"success": False, "error": "No matches found"}

    print(f"[PROGRESS] Found {len(matches_data)} image pairs with matches", file=sys.stderr, flush=True)

    transform_type = "affine" if use_affine else "homography"
    print(f"[PROGRESS] Computing {transform_type} transforms from matches...", file=sys.stderr, flush=True)
    homographies = compute_homographies(matches_data, n_images, image_name_to_index, use_affine)
    if not homographies:
        return {"success": False, "error": f"Could not compute {transform_type} transforms"}

    if min_inliers > 0 or max_images > 0:
        image_paths, homographies = filter_images(image_paths, homographies, min_inliers, max_images, n_images)
        if not homographies:
            return {"success": False, "error": "No images passed filtering criteria"}

    print("[PROGRESS] Stitching panorama...", file=sys.stderr, flush=True)
    panorama = stitch_with_homographies(
        image_paths,
        homographies,
        use_affine=use_affine,
        blend_method=blend_method,
        use_source_alpha=use_source_alpha,
        warp_interpolation=warp_interpolation,
        erode_border=erode_border,
        border_erosion_pixels=border_erosion_pixels,
        log_dir=output_path,
        auto_detect_content=auto_detect_content,
    )
    if panorama is None:
        return {"success": False, "error": "Stitching failed"}

    output_file = output_path / "colmap_panorama.tiff"
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
        "gpu_used": bool(getattr(pycolmap, "has_cuda", False)),
        "cache_key": cache_key,  # Return cache key for reblend
    }


# ============================================================
# REBLEND HELPER FUNCTIONS
# ============================================================

def blend_warped_images(warped_images: List[Dict], blend_method: str) -> Optional[np.ndarray]:
    """
    Blend pre-warped images without re-warping.
    Used when warp cache is valid and only blend method changed.
    """
    if not warped_images:
        return None

    print(f"[PROGRESS] 🎨 Blending with {blend_method}...", file=sys.stderr, flush=True)

    # Calculate canvas size from bboxes
    x_min = min(item["bbox"][0] for item in warped_images)
    y_min = min(item["bbox"][1] for item in warped_images)
    x_max = max(item["bbox"][2] for item in warped_images)
    y_max = max(item["bbox"][3] for item in warped_images)

    output_w = x_max - x_min
    output_h = y_max - y_min

    # Offset to adjust bboxes if needed
    offset_x = -x_min if x_min < 0 else 0
    offset_y = -y_min if y_min < 0 else 0

    blend_lower = str(blend_method).lower()

    # For streaming blenders (feather, linear, autostitch, autostitch_feather)
    if blend_lower in ["feather", "linear", "autostitch", "autostitch_feather"]:
        blender = StreamingBlender(output_h, output_w, blend_lower)
        for i, item in enumerate(warped_images):
            bbox = item["bbox"]
            adjusted_bbox = (
                bbox[0] + offset_x,
                bbox[1] + offset_y,
                bbox[2] + offset_x,
                bbox[3] + offset_y,
            )
            blender.add_image(item["image"], item["alpha"], adjusted_bbox)
            if (i + 1) % max(1, len(warped_images) // 5) == 0:
                print(f"[PROGRESS] Blending: {i + 1}/{len(warped_images)} images", file=sys.stderr, flush=True)
        return blender.finalize()

    # For multiband blending, use ImageBlender
    try:
        script_dir = Path(__file__).resolve().parent.parent
        if str(script_dir) not in sys.path:
            sys.path.insert(0, str(script_dir))
        from core.blender import ImageBlender

        aligned = []
        for item in warped_images:
            bbox = item["bbox"]
            adjusted_bbox = (
                bbox[0] + offset_x,
                bbox[1] + offset_y,
                bbox[2] + offset_x,
                bbox[3] + offset_y,
            )
            aligned.append({
                "image": item["image"],
                "alpha": item["alpha"],
                "bbox": adjusted_bbox,
                "warped": True,
            })

        blender = ImageBlender(method=blend_method, options={"hdr_mode": False, "anti_ghosting": False})
        return blender.blend(aligned, padding=0, fit_all=False)
    except ImportError as e:
        print(f"[WARNING] ImageBlender not available: {e}, using streaming blend", file=sys.stderr, flush=True)
        # Fallback to streaming blender
        blender = StreamingBlender(output_h, output_w, "feather")
        for item in warped_images:
            bbox = item["bbox"]
            adjusted_bbox = (
                bbox[0] + offset_x,
                bbox[1] + offset_y,
                bbox[2] + offset_x,
                bbox[3] + offset_y,
            )
            blender.add_image(item["image"], item["alpha"], adjusted_bbox)
        return blender.finalize()


def stitch_with_homographies_and_cache(
    image_paths: List[Path],
    homographies: Dict,
    warp_cache_dir: Path,
    use_affine: bool = False,
    blend_method: str = "multiband",
    use_source_alpha: bool = False,
    warp_interpolation: str = "linear",
    erode_border: bool = True,
    border_erosion_pixels: int = 5,
    log_dir: Optional[Path] = None,
    auto_detect_content: bool = False,
) -> Optional[np.ndarray]:
    """
    Stitch images with homographies and save warped images to cache for future reblends.
    This is like stitch_with_homographies but also saves the warp cache.
    """
    import cv2
    import heapq

    n_images = len(image_paths)
    if n_images == 0:
        return None
    if n_images == 1:
        img = cv2.imread(str(image_paths[0]), cv2.IMREAD_UNCHANGED if use_source_alpha else cv2.IMREAD_COLOR)
        if img is not None and img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    # Connectivity map
    connections: Dict[int, int] = {}
    for (i, j) in homographies.keys():
        connections[i] = connections.get(i, 0) + 1
        connections[j] = connections.get(j, 0) + 1

    if not connections:
        img = cv2.imread(str(image_paths[0]), cv2.IMREAD_UNCHANGED if use_source_alpha else cv2.IMREAD_COLOR)
        if img is not None and img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    # Connected components
    def find_components(nodes, edges):
        visited = set()
        comps = []
        for s in nodes:
            if s in visited:
                continue
            q = [s]
            comp = set()
            while q:
                u = q.pop(0)
                if u in visited:
                    continue
                visited.add(u)
                comp.add(u)
                for (a, b) in edges:
                    if a == u and b not in visited:
                        q.append(b)
                    elif b == u and a not in visited:
                        q.append(a)
            comps.append(comp)
        return comps

    nodes = set(connections.keys())
    comps = find_components(nodes, homographies.keys())
    comps.sort(key=len, reverse=True)
    largest = comps[0]

    # Improved reference selection: prefer well-connected images with higher resolution
    # This minimizes scale drift and preserves detail
    candidate_indices = sorted(largest)

    # Get image dimensions for candidates (quick header read)
    img_sizes = {}
    for idx in candidate_indices:
        try:
            img = cv2.imread(str(image_paths[idx]), cv2.IMREAD_UNCHANGED)
            if img is not None:
                h, w = img.shape[:2]
                img_sizes[idx] = (h, w, h * w)  # height, width, total pixels
                del img
        except Exception:
            pass

    if img_sizes:
        # Find median resolution (by total pixels)
        sorted_by_pixels = sorted(img_sizes.items(), key=lambda x: x[1][2])
        median_idx = len(sorted_by_pixels) // 2

        # Prefer images with higher resolution and good connectivity
        # Weight: 70% resolution rank, 30% connectivity
        scores = []
        max_conn = max(connections.get(idx, 1) for idx in img_sizes.keys())
        for rank, (idx, (h, w, pixels)) in enumerate(sorted_by_pixels):
            conn = connections.get(idx, 1)
            # Higher rank = larger resolution = better
            res_score = rank / len(sorted_by_pixels)
            conn_score = conn / max_conn
            combined = 0.7 * res_score + 0.3 * conn_score
            scores.append((combined, idx, h, w))

        # Pick highest scoring image
        scores.sort(reverse=True)
        ref_idx = scores[0][1]
        ref_h, ref_w = scores[0][2], scores[0][3]
        print(f"[PROGRESS] REF_SELECT: Chose image {ref_idx} ({ref_w}x{ref_h}) as reference "
              f"(connectivity={connections.get(ref_idx, 0)}, score={scores[0][0]:.3f})", file=sys.stderr, flush=True)
    else:
        # Fallback to original behavior
        ref_idx = n_images // 2
        if ref_idx not in largest:
            ref_idx = sorted(largest)[len(largest) // 2]
        print(f"[PROGRESS] REF_SELECT: Using fallback reference {ref_idx}", file=sys.stderr, flush=True)

    # Build adjacency with filtering
    adjacency: Dict[int, List[Tuple[int, np.ndarray, int]]] = {}
    MIN_INLIERS = 10
    MIN_TRANSLATION = 30.0
    for (i, j), data in homographies.items():
        if i not in largest or j not in largest:
            continue
        if int(data["n_inliers"]) < MIN_INLIERS:
            continue
        H = data["H"]
        tx, ty = float(H[0, 2]), float(H[1, 2])
        if float(np.hypot(tx, ty)) < MIN_TRANSLATION:
            continue
        adjacency.setdefault(i, [])
        adjacency.setdefault(j, [])
        w = int(data["n_inliers"]) + (5 if abs(i - j) == 1 else 0)
        adjacency[i].append((j, data["H"], w))
        adjacency[j].append((i, np.linalg.inv(data["H"]), w))

    # Dijkstra max-bottleneck
    H_to_ref = {ref_idx: np.eye(3)}
    hop_count = {ref_idx: 0}  # Track distance from reference
    best_weight_to = {ref_idx: float("inf")}
    processed = set()
    pq = [(0, ref_idx, np.eye(3), 0)]  # Added hop count to queue
    while pq:
        neg_w, cur_idx, H_cur, hops = heapq.heappop(pq)
        if cur_idx in processed:
            continue
        processed.add(cur_idx)
        H_to_ref[cur_idx] = H_cur
        hop_count[cur_idx] = hops
        for nb, H_edge, edge_w in adjacency.get(cur_idx, []):
            if nb in processed:
                continue
            path_w = min(best_weight_to.get(cur_idx, float("inf")), float(edge_w))
            if path_w > best_weight_to.get(nb, 0):
                best_weight_to[nb] = path_w
                H_nb = H_cur @ np.linalg.inv(H_edge)
                heapq.heappush(pq, (-int(min(path_w, 100000)), nb, H_nb, hops + 1))

    # Canvas from transformed corners
    corners_all = []
    canvas_dims = {}  # Store dimensions for verification during warping
    for idx, H in H_to_ref.items():
        img = cv2.imread(str(image_paths[idx]), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        h, w = img.shape[:2]
        canvas_dims[idx] = (h, w, img.shape[2] if img.ndim == 3 else 1)
        del img
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        if use_affine:
            M_affine = H[:2, :]
            transformed = cv2.transform(corners, M_affine)
        else:
            transformed = cv2.perspectiveTransform(corners, H)
        corners_all.append(transformed)

    if not corners_all:
        return None
    corners_all = np.concatenate(corners_all, axis=0)
    x_min, y_min = corners_all.min(axis=0).ravel()
    x_max, y_max = corners_all.max(axis=0).ravel()

    offset_x = -int(np.floor(x_min))
    offset_y = -int(np.floor(y_min))
    output_w = int(np.ceil(x_max - x_min))
    output_h = int(np.ceil(y_max - y_min))

    max_dim = 15000
    if output_w > max_dim or output_h > max_dim:
        scale = max_dim / max(output_w, output_h)
        output_w = int(output_w * scale)
        output_h = int(output_h * scale)
        offset_x = int(offset_x * scale)
        offset_y = int(offset_y * scale)
        S = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], dtype=np.float64)
        H_to_ref = {k: S @ v for k, v in H_to_ref.items()}

    T = np.array([[1, 0, offset_x], [0, 1, offset_y], [0, 0, 1]], dtype=np.float64)

    # ============================================================
    # SIMILARITY-CONSTRAINED BUNDLE ADJUSTMENT
    # Optimizes 4 parameters per image: (scale, theta, tx, ty)
    # This enforces uniform scaling (no shear) for flatbed scans
    # ============================================================
    print(f"[PROGRESS] ========== SIMILARITY BUNDLE ADJUSTMENT (4-DOF) ==========", file=sys.stderr, flush=True)

    try:
        from scipy.optimize import least_squares

        all_indices_ba = sorted(H_to_ref.keys())
        n_images_ba = len(all_indices_ba)
        idx_to_var_ba = {idx: i for i, idx in enumerate(all_indices_ba)}

        if n_images_ba > 2:
            # Get reference dimensions for expected scale calculation
            ref_h_ba, ref_w_ba, _ = canvas_dims.get(ref_idx, (1000, 1000, 3))
            ref_diag_ba = np.sqrt(ref_w_ba**2 + ref_h_ba**2)

            def H_to_similarity_params(H):
                """Extract (scale, theta, tx, ty) from similarity matrix."""
                a, b, tx = H[0, 0], H[0, 1], H[0, 2]
                c, d, ty = H[1, 0], H[1, 1], H[1, 2]
                # For similarity: a = s*cos(θ), b = -s*sin(θ), c = s*sin(θ), d = s*cos(θ)
                scale = np.sqrt(a**2 + c**2)  # or equivalently sqrt(b**2 + d**2)
                theta = np.arctan2(c, a)  # rotation angle
                return scale, theta, tx, ty

            def similarity_params_to_H(s, theta, tx, ty):
                """Convert (scale, theta, tx, ty) to 3x3 similarity matrix."""
                cos_t, sin_t = np.cos(theta), np.sin(theta)
                return np.array([
                    [s * cos_t, -s * sin_t, tx],
                    [s * sin_t,  s * cos_t, ty],
                    [0,          0,          1]
                ], dtype=np.float64)

            def params_to_H_ba(params, var_i):
                """Extract similarity params for image var_i and convert to matrix."""
                off = var_i * 4  # 4 params per image now
                s, theta, tx, ty = params[off:off+4]
                return similarity_params_to_H(s, theta, tx, ty)

            # Extract initial similarity parameters: [s, theta, tx, ty] per image
            initial_params = []
            for idx in all_indices_ba:
                H = H_to_ref[idx]
                s, theta, tx, ty = H_to_similarity_params(H)
                initial_params.extend([s, theta, tx, ty])
            initial_params = np.array(initial_params, dtype=np.float64)

            # Pre-compute valid pairs for consistent residual array size
            # First get all candidate pairs
            candidate_pairs = [(idx_i, idx_j, h_data) for (idx_i, idx_j), h_data in homographies.items()
                               if idx_i in idx_to_var_ba and idx_j in idx_to_var_ba]

            # Filter pairs by initial residual (100px threshold)
            valid_pairs = []
            ba_outlier_threshold = 50  # pixels
            for idx_i, idx_j, h_data in candidate_pairs:
                try:
                    H_i = H_to_ref[idx_i]
                    H_j = H_to_ref[idx_j]
                    H_ij = h_data["H"]
                    H_i_inv = np.linalg.inv(H_i)
                    residual_matrix = H_j @ H_ij @ H_i_inv
                    res_mag = np.sqrt(residual_matrix[0, 2]**2 + residual_matrix[1, 2]**2)
                    if res_mag < ba_outlier_threshold:
                        valid_pairs.append((idx_i, idx_j, h_data))
                except:
                    pass  # Skip pairs that fail

            n_filtered = len(candidate_pairs) - len(valid_pairs)
            print(f"[DEBUG] BA pair filtering: {len(candidate_pairs)} candidates, {len(valid_pairs)} kept, {n_filtered} filtered (>{ba_outlier_threshold}px)", file=sys.stderr, flush=True)
            n_pair_residuals = len(valid_pairs) * 4  # 4 residuals per pair (scale, theta, tx, ty)
            n_scale_residuals = len(all_indices_ba)  # 1 scale residual per image (uniform scale!)
            n_anchor_residuals = 4  # 4 params for reference anchor
            total_residuals = n_pair_residuals + n_scale_residuals + n_anchor_residuals

            def residual_function_ba(params):
                residuals = np.zeros(total_residuals, dtype=np.float64)
                res_idx = 0

                # Pairwise consistency residuals
                for idx_i, idx_j, h_data in valid_pairs:
                    var_i, var_j = idx_to_var_ba[idx_i], idx_to_var_ba[idx_j]
                    H_i, H_j = params_to_H_ba(params, var_i), params_to_H_ba(params, var_j)
                    H_ij = h_data["H"]
                    n_inliers = h_data.get("n_inliers", 10)
                    weight = np.sqrt(max(n_inliers, 1)) * 0.1
                    try:
                        H_i_inv = np.linalg.inv(H_i)
                        rm = H_j @ H_ij @ H_i_inv
                        # For similarity, rm should be close to identity
                        # Extract similarity residuals: scale deviation, rotation deviation, translation
                        rm_s, rm_theta, rm_tx, rm_ty = H_to_similarity_params(rm)
                        residuals[res_idx:res_idx+4] = [
                            (rm_s - 1.0) * weight * 2.0,  # scale should be 1
                            rm_theta * weight * 2.0,       # rotation should be 0
                            np.clip(rm_tx * weight * 0.01, -50, 50),
                            np.clip(rm_ty * weight * 0.01, -50, 50)
                        ]
                    except:
                        pass  # Leave as zeros
                    res_idx += 4

                # Scale regularization - RESOLUTION-AWARE (single residual per image)
                # Strong weight to prevent drift while allowing real magnification variations
                scale_weight = 10.0
                for idx in all_indices_ba:
                    off = idx_to_var_ba[idx] * 4
                    s = params[off]  # Direct access to scale parameter

                    # Calculate expected scale based on resolution ratio to reference
                    if idx in canvas_dims:
                        img_h, img_w, _ = canvas_dims[idx]
                        img_diag = np.sqrt(img_w**2 + img_h**2)
                        exp_s = ref_diag_ba / img_diag
                    else:
                        exp_s = 1.0

                    if s > 0.01:
                        residuals[res_idx] = scale_weight * (np.log(s) - np.log(exp_s))
                    res_idx += 1

                # Reference anchor: scale=expected, theta=0, tx=ty=0
                off_ref = idx_to_var_ba[ref_idx] * 4
                s_ref, theta_ref, tx_ref, ty_ref = params[off_ref:off_ref+4]
                aw = 10.0
                # Reference expected scale is 1.0 (it's the reference!)
                residuals[res_idx:res_idx+4] = [
                    (s_ref - 1.0) * aw,
                    theta_ref * aw,
                    tx_ref * aw * 0.01,
                    ty_ref * aw * 0.01
                ]
                return residuals

            initial_cost = np.sqrt(np.mean(residual_function_ba(initial_params)**2))
            n_params = len(initial_params)
            print(f"[PROGRESS] BA_INIT: {n_images_ba} imgs, {len(valid_pairs)} pairs, {n_params} params, {total_residuals} residuals, RMS={initial_cost:.4f}",
                  file=sys.stderr, flush=True)
            print(f"[DEBUG] BA settings: max_nfev=10, ftol=1e-6, xtol=1e-6, method=lm", file=sys.stderr, flush=True)

            import time as _time
            _ba_start = _time.time()
            result = least_squares(residual_function_ba, initial_params, method='lm', max_nfev=10,
                                   ftol=1e-6, xtol=1e-6, verbose=0)
            _ba_elapsed = _time.time() - _ba_start

            final_cost = np.sqrt(np.mean(residual_function_ba(result.x)**2))
            print(f"[DEBUG] BA completed: nfev={result.nfev}, njev={result.njev}, status={result.status}, time={_ba_elapsed:.2f}s", file=sys.stderr, flush=True)
            print(f"[DEBUG] BA message: {result.message}", file=sys.stderr, flush=True)
            if final_cost < initial_cost:
                improvement = (initial_cost - final_cost) / initial_cost * 100
                print(f"[PROGRESS] BA_DONE: RMS {initial_cost:.4f} -> {final_cost:.4f} ({improvement:.1f}%)",
                      file=sys.stderr, flush=True)
                for idx in all_indices_ba:
                    H_to_ref[idx] = params_to_H_ba(result.x, idx_to_var_ba[idx])
                # Recalculate canvas
                corners_ba = []
                for idx, H in H_to_ref.items():
                    if idx not in canvas_dims: continue
                    h, w, _ = canvas_dims[idx]
                    c = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
                    corners_ba.append(cv2.transform(c, H[:2,:]) if use_affine else cv2.perspectiveTransform(c, H))
                if corners_ba:
                    corners_ba = np.concatenate(corners_ba, axis=0)
                    x_min, y_min = corners_ba.min(axis=0).ravel()
                    x_max, y_max = corners_ba.max(axis=0).ravel()
                    offset_x, offset_y = -int(np.floor(x_min)), -int(np.floor(y_min))
                    output_w, output_h = int(np.ceil(x_max-x_min)), int(np.ceil(y_max-y_min))
                    T = np.array([[1,0,offset_x],[0,1,offset_y],[0,0,1]], dtype=np.float64)
            else:
                print(f"[PROGRESS] BA_SKIP: No improvement", file=sys.stderr, flush=True)
    except ImportError as e:
        print(f"[PROGRESS] BA_SKIP: scipy not available ({e})", file=sys.stderr, flush=True)
    except Exception as e:
        import traceback
        print(f"[PROGRESS] BA_ERROR: {e}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)

    print(f"[PROGRESS] ========== SCALE VERIFICATION (POST-BA) ==========", file=sys.stderr, flush=True)
    # Verify scale factors after BA - BA should now handle resolution-aware scaling
    # This is diagnostic only - no transform modifications
    ref_h, ref_w, _ = canvas_dims.get(ref_idx, (1, 1, 1))
    ref_diag = np.sqrt(ref_w**2 + ref_h**2)

    scale_issues = []
    all_scales = []
    for idx, H in H_to_ref.items():
        if idx not in canvas_dims:
            continue

        # Calculate expected scale based on resolution ratio to reference
        img_h, img_w, _ = canvas_dims[idx]
        img_diag = np.sqrt(img_w**2 + img_h**2)
        expected_scale = ref_diag / img_diag

        # Extract actual scale factors
        a, b = H[0, 0], H[0, 1]
        c, d = H[1, 0], H[1, 1]
        scale_x = np.sqrt(a**2 + c**2)
        scale_y = np.sqrt(b**2 + d**2)
        hops = hop_count.get(idx, -1)

        # Check deviation from expected (not from 1.0!)
        dev_x = abs(scale_x - expected_scale) / expected_scale * 100
        dev_y = abs(scale_y - expected_scale) / expected_scale * 100
        all_scales.append((idx, scale_x, scale_y, expected_scale, hops))

        # Flag significant deviations (>5% from expected)
        if dev_x > 5.0 or dev_y > 5.0:
            scale_issues.append((idx, scale_x, scale_y, expected_scale, hops, max(dev_x, dev_y)))

    # Print summary
    if all_scales:
        scales_x = [s[1] for s in all_scales]
        scales_y = [s[2] for s in all_scales]
        print(f"[PROGRESS] SCALE_DIAG: ref={ref_idx}, n={len(all_scales)}, "
              f"sx=[{min(scales_x):.3f}-{max(scales_x):.3f}], sy=[{min(scales_y):.3f}-{max(scales_y):.3f}]",
              file=sys.stderr, flush=True)

    if scale_issues:
        print(f"[PROGRESS] SCALE_WARNING: {len(scale_issues)} images have scale >5% off from expected", file=sys.stderr, flush=True)
        for idx, sx, sy, exp, hops, dev in sorted(scale_issues, key=lambda x: -x[5])[:10]:
            img_name = Path(image_paths[idx]).name if idx < len(image_paths) else f"idx_{idx}"
            print(f"[PROGRESS] SCALE_ISSUE: {img_name} hops={hops} sx={sx:.3f} sy={sy:.3f} expected={exp:.3f} dev={dev:.1f}%",
                  file=sys.stderr, flush=True)
    else:
        print(f"[PROGRESS] SCALE_OK: All images within 5% of expected scale", file=sys.stderr, flush=True)

    # Note: No transform modifications here - BA should handle scale correctly now
    # If scale issues persist, the BA parameters may need tuning

    if True:  # Always enter this block for canvas recalculation
        # ============================================================
        # GLOBAL TRANSLATION OPTIMIZATION (Iterative Linear Least Squares)
        # Based on MegaStitch approach: minimize pairwise consistency error
        # For each pair (i,j) with H_ij: ideally H_i = H_j @ H_ij
        # We solve for translation adjustments that minimize residuals
        # Uses iterative refinement for better convergence
        # ============================================================
        print(f"[PROGRESS] GLOBAL_OPT: Starting global translation optimization...", file=sys.stderr, flush=True)

        try:
            from scipy import sparse
            from scipy.sparse.linalg import lsqr

            all_indices = sorted(H_to_ref.keys())
            idx_to_var = {idx: i for i, idx in enumerate(all_indices)}
            n_vars = len(all_indices) * 2  # tx, ty for each image

            # Track per-image residuals for debugging
            def compute_per_image_residuals():
                """Compute average residual per image"""
                img_residuals = {idx: [] for idx in all_indices}
                for (idx_i, idx_j), h_data in homographies.items():
                    if idx_i not in H_to_ref or idx_j not in H_to_ref:
                        continue
                    H_ij = h_data["H"]
                    H_i = H_to_ref[idx_i]
                    H_j = H_to_ref[idx_j]
                    try:
                        H_i_inv = np.linalg.inv(H_i)
                        residual_matrix = H_j @ H_ij @ H_i_inv
                        res_tx = residual_matrix[0, 2]
                        res_ty = residual_matrix[1, 2]
                        res_mag = np.sqrt(res_tx**2 + res_ty**2)
                        if res_mag < 50:  # Skip outliers
                            img_residuals[idx_i].append(res_mag)
                            img_residuals[idx_j].append(res_mag)
                    except Exception:
                        pass
                # Compute mean per image
                return {idx: np.mean(r) if r else 0 for idx, r in img_residuals.items()}

            # Robust weighting function (Huber-like)
            # Down-weights constraints with high residuals to prevent outliers from dominating
            def huber_weight(residual_mag, threshold=20.0):
                """Huber weight: 1 for small residuals, decreases for large ones"""
                if residual_mag <= threshold:
                    return 1.0
                else:
                    return threshold / residual_mag

            def compute_image_confidence():
                """Compute per-image confidence based on average residual across all its constraints"""
                img_residuals = {idx: [] for idx in all_indices}
                for (idx_i, idx_j), h_data in homographies.items():
                    if idx_i not in H_to_ref or idx_j not in H_to_ref:
                        continue
                    H_ij = h_data["H"]
                    H_i = H_to_ref[idx_i]
                    H_j = H_to_ref[idx_j]
                    try:
                        H_i_inv = np.linalg.inv(H_i)
                        residual_matrix = H_j @ H_ij @ H_i_inv
                        res_mag = np.sqrt(residual_matrix[0, 2]**2 + residual_matrix[1, 2]**2)
                        if res_mag < 50:
                            img_residuals[idx_i].append(res_mag)
                            img_residuals[idx_j].append(res_mag)
                    except Exception:
                        pass
                # Compute confidence: high avg residual = low confidence
                confidence = {}
                residual_threshold = 50.0  # Images with avg > 50px get down-weighted
                for idx, residuals in img_residuals.items():
                    if residuals:
                        avg = np.mean(residuals)
                        # Confidence decreases as avg residual increases
                        if avg <= residual_threshold:
                            confidence[idx] = 1.0
                        else:
                            confidence[idx] = residual_threshold / avg
                    else:
                        confidence[idx] = 1.0
                return confidence

            # Iterative optimization with robust reweighting (IRLS) + per-image confidence
            max_iterations = 10  # More iterations for better convergence
            convergence_threshold = 0.005  # Tighter convergence
            prev_avg_residual = float('inf')
            huber_threshold = 20.0  # Lower threshold for more aggressive down-weighting
            image_confidence = {idx: 1.0 for idx in all_indices}  # Start with full confidence
            excluded_images = set()  # Images with consistently terrible alignment

            for iteration in range(max_iterations):
                # After iteration 3, identify and exclude images with very high residuals
                if iteration == 3:
                    per_img = compute_per_image_residuals()
                    # Exclude images with avg residual > 100px - they have fundamentally bad matches
                    for idx, res in per_img.items():
                        if res > 100:
                            excluded_images.add(idx)
                    if excluded_images:
                        excluded_names = [Path(image_paths[idx]).name if idx < len(image_paths) else f"idx_{idx}"
                                         for idx in excluded_images]
                        print(f"[PROGRESS] GLOBAL_OPT: Excluding {len(excluded_images)} images with bad matches: {', '.join(excluded_names[:5])}", file=sys.stderr, flush=True)

                # Update per-image confidence after first 2 iterations
                if iteration >= 2:
                    image_confidence = compute_image_confidence()
                    n_low_conf = sum(1 for c in image_confidence.values() if c < 1.0)
                    if iteration == 2:
                        print(f"[PROGRESS] GLOBAL_OPT: {n_low_conf} images have reduced confidence", file=sys.stderr, flush=True)

                rows = []
                cols = []
                data = []
                b_vec = []
                n_constraints = 0
                total_residual = 0.0
                n_downweighted = 0

                for (idx_i, idx_j), h_data in homographies.items():
                    if idx_i not in H_to_ref or idx_j not in H_to_ref:
                        continue

                    # Skip constraints involving excluded (bad match) images
                    if idx_i in excluded_images or idx_j in excluded_images:
                        continue

                    H_ij = h_data["H"]  # Transform from i to j
                    n_inliers = h_data.get("n_inliers", 10)

                    H_i = H_to_ref[idx_i]
                    H_j = H_to_ref[idx_j]

                    try:
                        H_i_inv = np.linalg.inv(H_i)
                        residual_matrix = H_j @ H_ij @ H_i_inv

                        res_tx = residual_matrix[0, 2]
                        res_ty = residual_matrix[1, 2]
                        res_mag = np.sqrt(res_tx**2 + res_ty**2)

                        # Skip if residual is extreme (likely completely wrong match)
                        if res_mag > 50:
                            continue

                        total_residual += res_mag
                        n_constraints += 1

                        # Robust Huber weighting - down-weight high-residual constraints
                        robust_weight = huber_weight(res_mag, huber_threshold)
                        if robust_weight < 1.0:
                            n_downweighted += 1

                        # Combined weight: inlier confidence * robust weight * per-image confidence
                        # Per-image confidence down-weights images with consistently high residuals
                        img_conf = np.sqrt(image_confidence.get(idx_i, 1.0) * image_confidence.get(idx_j, 1.0))
                        weight = np.sqrt(n_inliers) * robust_weight * img_conf

                        var_i_tx = idx_to_var[idx_i] * 2
                        var_i_ty = idx_to_var[idx_i] * 2 + 1
                        var_j_tx = idx_to_var[idx_j] * 2
                        var_j_ty = idx_to_var[idx_j] * 2 + 1

                        # X constraint: adj_tx_i - adj_tx_j = res_tx
                        row = len(b_vec)
                        rows.extend([row, row])
                        cols.extend([var_i_tx, var_j_tx])
                        data.extend([weight, -weight])
                        b_vec.append(weight * res_tx)

                        # Y constraint: adj_ty_i - adj_ty_j = res_ty
                        row = len(b_vec)
                        rows.extend([row, row])
                        cols.extend([var_i_ty, var_j_ty])
                        data.extend([weight, -weight])
                        b_vec.append(weight * res_ty)

                    except Exception:
                        continue

                if n_constraints == 0:
                    print(f"[PROGRESS] GLOBAL_OPT: No valid constraints, skipping", file=sys.stderr, flush=True)
                    break

                avg_residual = total_residual / n_constraints

                # Check convergence
                if iteration > 0 and (prev_avg_residual - avg_residual) < convergence_threshold:
                    print(f"[PROGRESS] GLOBAL_OPT: Converged at iteration {iteration+1} (improvement < {convergence_threshold}px)", file=sys.stderr, flush=True)
                    break

                print(f"[PROGRESS] GLOBAL_OPT: Iter {iteration+1}: {n_constraints} constraints ({n_downweighted} down-weighted), avg residual={avg_residual:.2f}px", file=sys.stderr, flush=True)

                # Very low regularization - let the constraints drive the solution
                reg_weight = 0.01
                for i in range(n_vars):
                    row = len(b_vec)
                    rows.append(row)
                    cols.append(i)
                    data.append(reg_weight)
                    b_vec.append(0)

                # Fix reference image (zero adjustment)
                if ref_idx in idx_to_var:
                    big_weight = 1000
                    var_ref_tx = idx_to_var[ref_idx] * 2
                    var_ref_ty = idx_to_var[ref_idx] * 2 + 1

                    row = len(b_vec)
                    rows.append(row)
                    cols.append(var_ref_tx)
                    data.append(big_weight)
                    b_vec.append(0)

                    row = len(b_vec)
                    rows.append(row)
                    cols.append(var_ref_ty)
                    data.append(big_weight)
                    b_vec.append(0)

                # Build sparse matrix and solve
                A = sparse.csr_matrix((data, (rows, cols)), shape=(len(b_vec), n_vars))
                b = np.array(b_vec)

                result = lsqr(A, b, atol=1e-10, btol=1e-10)
                adjustments = result[0]

                # Apply adjustments
                max_adj = 0
                adjustments_applied = []
                for idx in all_indices:
                    var_i = idx_to_var[idx]
                    adj_tx = adjustments[var_i * 2]
                    adj_ty = adjustments[var_i * 2 + 1]

                    H_to_ref[idx][0, 2] += adj_tx
                    H_to_ref[idx][1, 2] += adj_ty

                    adj_mag = np.sqrt(adj_tx**2 + adj_ty**2)
                    max_adj = max(max_adj, adj_mag)
                    adjustments_applied.append((idx, adj_mag))

                mean_adj = np.mean([a[1] for a in adjustments_applied])
                print(f"[PROGRESS] GLOBAL_OPT: Iter {iteration+1}: Applied adjustments (max={max_adj:.2f}px, mean={mean_adj:.2f}px)", file=sys.stderr, flush=True)

                prev_avg_residual = avg_residual

            # Final verification and per-image reporting
            per_img_res = compute_per_image_residuals()
            worst_images = sorted(per_img_res.items(), key=lambda x: -x[1])[:10]

            # Compute final average
            final_residuals = [r for r in per_img_res.values() if r > 0]
            if final_residuals:
                final_avg = np.mean(final_residuals)
                print(f"[PROGRESS] GLOBAL_OPT: Final avg residual={final_avg:.2f}px", file=sys.stderr, flush=True)

                # Report worst aligned images
                print(f"[PROGRESS] GLOBAL_OPT: Worst aligned images:", file=sys.stderr, flush=True)
                for idx, res in worst_images[:5]:
                    if res > 0:
                        img_name = Path(image_paths[idx]).name if idx < len(image_paths) else f"idx_{idx}"
                        print(f"[PROGRESS] GLOBAL_OPT:   {img_name}: avg_residual={res:.1f}px", file=sys.stderr, flush=True)

        except ImportError:
            print(f"[PROGRESS] GLOBAL_OPT: scipy not available, skipping global optimization", file=sys.stderr, flush=True)
        except Exception as e:
            import traceback
            print(f"[PROGRESS] GLOBAL_OPT: Error during optimization: {e}", file=sys.stderr, flush=True)
            traceback.print_exc()

        # ALWAYS recompute canvas after scale analysis to catch any drift
        # (even if normalized_count = 0, small accumulated errors could cause truncation)
        corners_all_new = []
        for idx, H in H_to_ref.items():
            if idx not in canvas_dims:
                continue
            h, w, _ = canvas_dims[idx]
            corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
            # Use raw H without old T - we'll compute fresh bounds
            if use_affine:
                transformed = cv2.transform(corners, H[:2, :])
            else:
                transformed = cv2.perspectiveTransform(corners, H)
            corners_all_new.append(transformed)

        if corners_all_new:
            corners_all_new = np.concatenate(corners_all_new, axis=0)
            x_min_new, y_min_new = corners_all_new.min(axis=0).ravel()
            x_max_new, y_max_new = corners_all_new.max(axis=0).ravel()

            # Add safety margin to prevent truncation from floating point rounding
            canvas_margin = 20  # pixels
            x_min_new -= canvas_margin
            y_min_new -= canvas_margin
            x_max_new += canvas_margin
            y_max_new += canvas_margin

            offset_x = -int(np.floor(x_min_new))
            offset_y = -int(np.floor(y_min_new))
            output_w = int(np.ceil(x_max_new - x_min_new))
            output_h = int(np.ceil(y_max_new - y_min_new))
            T = np.array([[1, 0, offset_x], [0, 1, offset_y], [0, 0, 1]], dtype=np.float64)
            print(f"[PROGRESS] SCALE_FIX: Recalculated canvas size {output_w}x{output_h} (with {canvas_margin}px margin)", file=sys.stderr, flush=True)

    interp_map = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4,
        "realesrgan": cv2.INTER_LANCZOS4,
    }
    interp_flag = interp_map.get(str(warp_interpolation).lower(), cv2.INTER_LINEAR)

    # Warp images and collect for caching
    warped_images_for_cache: List[Dict] = []
    total_to_warp = len(H_to_ref)
    warped_count = 0
    print(f"[PROGRESS] Warping: Starting {total_to_warp} images (will cache)...", file=sys.stderr, flush=True)

    for idx, H in H_to_ref.items():
        warped_count += 1
        if warped_count % max(1, total_to_warp // 10) == 0 or warped_count == total_to_warp:
            pct = int(100 * warped_count / total_to_warp)
            print(f"[PROGRESS] Warping: {warped_count}/{total_to_warp} images ({pct}%)", file=sys.stderr, flush=True)

        img_path = str(image_paths[idx])
        img_raw = cv2.imread(img_path, cv2.IMREAD_UNCHANGED if use_source_alpha else cv2.IMREAD_COLOR)
        if img_raw is None:
            continue
        if use_source_alpha and img_raw.ndim == 3 and img_raw.shape[2] == 4:
            img = img_raw[:, :, :3]
            alpha_original = img_raw[:, :, 3]

            # DIAGNOSTIC: Check alpha coverage vs full image
            h_raw, w_raw = alpha_original.shape
            alpha_pixels = np.sum(alpha_original > 0)
            total_pixels = h_raw * w_raw
            coverage_pct = 100.0 * alpha_pixels / total_pixels

            # Find bounding box of non-zero alpha
            nonzero_rows = np.any(alpha_original > 0, axis=1)
            nonzero_cols = np.any(alpha_original > 0, axis=0)
            if np.any(nonzero_rows) and np.any(nonzero_cols):
                y_min, y_max = np.where(nonzero_rows)[0][[0, -1]]
                x_min, x_max = np.where(nonzero_cols)[0][[0, -1]]
                alpha_w = x_max - x_min + 1
                alpha_h = y_max - y_min + 1
                # Log if alpha region is significantly smaller than image
                if coverage_pct < 95 or alpha_w < w_raw - 10 or alpha_h < h_raw - 10:
                    img_name = Path(image_paths[idx]).name
                    print(f"[PROGRESS] ALPHA_SMALL: {img_name} covers {coverage_pct:.1f}%, "
                          f"bbox=({x_min},{y_min})-({x_max},{y_max}) vs {w_raw}x{h_raw}",
                          file=sys.stderr, flush=True)

            if erode_border and border_erosion_pixels > 0:
                ksz = int(border_erosion_pixels)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
                alpha_original = cv2.erode(alpha_original, kernel, iterations=2)
        else:
            img = img_raw if img_raw.ndim == 3 else cv2.cvtColor(img_raw, cv2.COLOR_GRAY2BGR)
            alpha_original = None
        del img_raw

        H_final = T @ H
        h, w = img.shape[:2]

        # CRITICAL: Verify dimensions match what was used for canvas calculation
        if idx in canvas_dims:
            canvas_h, canvas_w, canvas_ch = canvas_dims[idx]
            if h != canvas_h or w != canvas_w:
                print(f"[ERROR] Image {idx} DIMENSION MISMATCH! Canvas used {canvas_w}x{canvas_h} but warp got {w}x{h}",
                      file=sys.stderr, flush=True)
                print(f"[ERROR] This will cause scale/shift artifacts! "
                      f"(use_source_alpha={use_source_alpha}, canvas_ch={canvas_ch}, warp_ch={img.shape[2] if img.ndim==3 else 1})",
                      file=sys.stderr, flush=True)
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        transformed = cv2.transform(corners, H_final[:2, :]) if use_affine else cv2.perspectiveTransform(corners, H_final)
        x0 = int(np.floor(transformed[:, 0, 0].min()))
        y0 = int(np.floor(transformed[:, 0, 1].min()))
        x1 = int(np.ceil(transformed[:, 0, 0].max()))
        y1 = int(np.ceil(transformed[:, 0, 1].max()))
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(output_w, x1)
        y1 = min(output_h, y1)
        region_w = x1 - x0
        region_h = y1 - y0
        if region_w <= 0 or region_h <= 0:
            del img
            continue

        if alpha_original is None:
            if auto_detect_content:
                # Use content detection to find valid region (handles round images)
                img_name = Path(image_paths[idx]).name
                alpha_original = create_content_mask_for_warp(img, auto_detect=True, image_name=img_name)
            else:
                alpha_original = np.ones((h, w), dtype=np.uint8) * 255

        T_offset = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], dtype=np.float64)
        H_crop = T_offset @ H_final

        # Use NEAREST interpolation for alpha mask to preserve hard edges
        # (LANCZOS can create ringing artifacts on binary masks)
        alpha_interp = cv2.INTER_NEAREST

        if use_affine:
            M = H_crop[:2, :]
            warped = cv2.warpAffine(img, M, (region_w, region_h), flags=interp_flag)
            alpha = cv2.warpAffine(alpha_original, M, (region_w, region_h), flags=alpha_interp)
        else:
            warped = cv2.warpPerspective(img, H_crop, (region_w, region_h), flags=interp_flag)
            alpha = cv2.warpPerspective(alpha_original, H_crop, (region_w, region_h), flags=alpha_interp)

        del img
        del alpha_original

        # CRITICAL FIX for autostitch duplication/shift artifacts:
        # RGB warp uses LINEAR/CUBIC interpolation which causes edge pixels to be
        # interpolated with black border (borderValue=0), creating darkened/shifted edges.
        # Alpha uses INTER_NEAREST so it doesn't know about this.
        # Solution: Erode alpha by 1-2 pixels to exclude those interpolated RGB edges.
        # This only affects the mask, not the RGB content.
        if interp_flag != cv2.INTER_NEAREST:
            # Only erode if RGB used interpolation (not NEAREST)
            edge_erosion_px = 2  # 2px erosion to safely exclude interpolated edge
            if edge_erosion_px > 0:
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (edge_erosion_px * 2 + 1, edge_erosion_px * 2 + 1)
                )
                alpha = cv2.erode(alpha, kernel, iterations=1)

        bbox = (x0, y0, x1, y1)
        warped_images_for_cache.append({
            "image": warped.copy(),
            "alpha": alpha.copy(),
            "bbox": bbox,
            "idx": idx,
        })

    # Save warp cache for future reblends
    if warped_images_for_cache:
        try:
            save_warp_cache(
                warp_cache_dir,
                warped_images_for_cache,
                canvas_size=(output_h, output_w),
                offset=(offset_x, offset_y),
            )
        except Exception as e:
            print(f"[WARNING] Failed to save warp cache: {e}", file=sys.stderr, flush=True)

    # Now blend
    print(f"[PROGRESS] 🎨 Blending {len(warped_images_for_cache)} images...", file=sys.stderr, flush=True)
    return blend_warped_images(warped_images_for_cache, blend_method)


def run_reblend(
    cache_key: str,
    image_paths_json: str,
    output_dir: str,
    use_affine: bool = False,
    blend_method: str = "multiband",
    use_source_alpha: bool = False,
    warp_interpolation: str = "linear",
    erode_border: bool = True,
    border_erosion_pixels: int = 5,
    max_features: int = 8192,
    auto_detect_content: bool = False,
) -> Dict:
    """
    Reblend a previous COLMAP run using cached database and matches.

    This skips feature extraction and matching, reusing the cached database.
    If warp settings haven't changed, it also skips warping and only re-blends.

    Args:
        cache_key: The cache key from a previous successful run
        image_paths_json: JSON string of original image paths
        output_dir: Output directory
        blend_method: New blend method to use
        Other args: Same as run_2d_stitch

    Returns:
        Dict with success status and output path
    """
    import cv2
    import time

    start_time = time.time()

    print(f"[PROGRESS] 🔄 Reblend: Using cached run {cache_key[:16]}...", file=sys.stderr, flush=True)
    print(f"[PROGRESS] 🎨 New blend method: {blend_method}", file=sys.stderr, flush=True)

    # Parse image paths
    image_paths_win = json.loads(image_paths_json)
    image_paths = [Path(windows_to_wsl_path(p)) for p in image_paths_win]
    output_path = Path(windows_to_wsl_path(output_dir))

    # CRITICAL: Sort image paths to ensure consistent ordering with cache.
    # The cache was created with sorted paths, so we must use the same order.
    image_paths = sorted(image_paths, key=lambda x: x.name)

    n_images = len(image_paths)
    print(f"[PROGRESS] 📊 Processing {n_images} images", file=sys.stderr, flush=True)

    # Find cached database
    print(f"[PROGRESS] 🔍 Looking for cached database...", file=sys.stderr, flush=True)
    cache_base = Path(os.environ.get("STITCH2STITCH_CACHE_DIR", str(Path.home() / ".stitch2stitch_cache")))
    cache_root = cache_base / "colmap"
    global_cache_dir = cache_root / cache_key

    if not global_cache_dir.exists():
        return {"success": False, "error": f"Cache not found for key {cache_key[:16]}..."}

    cached_db = global_cache_dir / "database.db"
    if not cached_db.exists():
        return {"success": False, "error": "Cached database not found"}

    print(f"[PROGRESS] ✅ Found cached database", file=sys.stderr, flush=True)

    # Compute warp key to check if we can skip warping
    warp_key = compute_warp_key(
        use_affine=use_affine,
        warp_interpolation=warp_interpolation,
        use_source_alpha=use_source_alpha,
        erode_border=erode_border,
        border_erosion_pixels=border_erosion_pixels,
        auto_detect_content=auto_detect_content,
    )
    warp_cache_dir = get_warp_cache_dir(global_cache_dir, warp_key)

    # Check if we can skip warping (only re-blend)
    if is_warp_cache_valid(warp_cache_dir, n_images):
        print(f"[PROGRESS] ⚡ Warp cache hit! Skipping warping, blend only...", file=sys.stderr, flush=True)

        # Load cached warped images
        warped_images, metadata = load_warp_cache(warp_cache_dir)

        if len(warped_images) > 0:
            # Blend only - no warping needed!
            print(f"[PROGRESS] 🎨 Blending {len(warped_images)} cached warped images...", file=sys.stderr, flush=True)

            panorama = blend_warped_images(warped_images, blend_method)

            if panorama is not None:
                print(f"[PROGRESS] Stitching complete! Panorama size: {panorama.shape[1]}x{panorama.shape[0]}", file=sys.stderr, flush=True)

                # Save result
                print(f"[PROGRESS] Saving panorama...", file=sys.stderr, flush=True)
                output_file = output_path / f"colmap_panorama_reblend.tiff"
                cv2.imwrite(str(output_file), panorama)

                total_time = time.time() - start_time
                print(f"[PROGRESS] ✨ Reblend Complete! (blend-only) Time: {total_time:.1f}s", file=sys.stderr, flush=True)
                print(f"[PROGRESS] 📁 Output: {output_file.name}", file=sys.stderr, flush=True)

                return {
                    "success": True,
                    "output_path": wsl_to_windows_path(str(output_file)),
                    "n_images": n_images,
                    "size": list(panorama.shape[:2]),
                    "blend_method": blend_method,
                    "reblend": True,
                    "skipped_warping": True,
                }

        print(f"[WARNING] Warp cache blend failed, falling back to full re-warp", file=sys.stderr, flush=True)

    # Full reblend path: read matches, compute homographies, warp, blend
    print(f"[PROGRESS] 🔄 Full reblend: warping + blending required", file=sys.stderr, flush=True)

    # Build name mapping
    image_name_to_index: Dict[str, int] = {f"{i:06d}{p.suffix}": i for i, p in enumerate(image_paths)}

    # Read matches from cached database
    print("[PROGRESS] Reading matches from cached database...", file=sys.stderr, flush=True)
    matches_data = read_matches_from_db(cached_db)
    if not matches_data:
        return {"success": False, "error": "No matches found in cached database"}

    print(f"[PROGRESS] Found {len(matches_data)} image pairs with matches", file=sys.stderr, flush=True)

    # Compute homographies
    transform_type = "affine" if use_affine else "homography"
    print(f"[PROGRESS] Computing homographies ({transform_type} transforms)...", file=sys.stderr, flush=True)
    homographies = compute_homographies(matches_data, n_images, image_name_to_index, use_affine)
    if not homographies:
        return {"success": False, "error": f"Could not compute {transform_type} transforms"}

    print(f"[PROGRESS] Computed {len(homographies)} homographies", file=sys.stderr, flush=True)

    # Stitch with new blend settings (and save warp cache)
    print(f"[PROGRESS] Stitching panorama with {blend_method} blending...", file=sys.stderr, flush=True)
    print(f"[PROGRESS] 🖼️ Warping {len(homographies)} images...", file=sys.stderr, flush=True)

    panorama = stitch_with_homographies_and_cache(
        image_paths,
        homographies,
        warp_cache_dir=warp_cache_dir,
        use_affine=use_affine,
        blend_method=blend_method,
        use_source_alpha=use_source_alpha,
        warp_interpolation=warp_interpolation,
        erode_border=erode_border,
        border_erosion_pixels=border_erosion_pixels,
        log_dir=output_path,
        auto_detect_content=auto_detect_content,
    )
    if panorama is None:
        return {"success": False, "error": "Reblend stitching failed"}

    print(f"[PROGRESS] Stitching complete! Panorama size: {panorama.shape[1]}x{panorama.shape[0]}", file=sys.stderr, flush=True)

    # Save with "_reblend" suffix to not overwrite original
    print(f"[PROGRESS] Saving panorama...", file=sys.stderr, flush=True)
    output_file = output_path / f"colmap_panorama_reblend.tiff"
    cv2.imwrite(str(output_file), panorama)

    total_time = time.time() - start_time
    print(f"[PROGRESS] ✨ Reblend Complete! Time: {total_time:.1f}s", file=sys.stderr, flush=True)
    print(f"[PROGRESS] 📁 Output: {output_file.name}", file=sys.stderr, flush=True)

    return {
        "success": True,
        "output_path": wsl_to_windows_path(str(output_file)),
        "n_images": n_images,
        "size": list(panorama.shape[:2]),
        "blend_method": blend_method,
        "reblend": True,
        "skipped_warping": False,
    }


def read_matches_from_db(database_path: Path) -> Dict:
    import sqlite3

    conn = sqlite3.connect(str(database_path))
    cur = conn.cursor()
    cur.execute("SELECT image_id, name FROM images")
    image_id_to_name = {int(i): n for i, n in cur.fetchall()}

    keypoints: Dict[int, np.ndarray] = {}
    cur.execute("SELECT image_id, rows, cols, data FROM keypoints")
    for image_id, rows, cols, data in cur.fetchall():
        if data:
            arr = np.frombuffer(data, dtype=np.float32).reshape(int(rows), int(cols))
            keypoints[int(image_id)] = arr[:, :2]

    matches: Dict[Tuple[str, str], Dict] = {}
    cur.execute("SELECT pair_id, rows, cols, data FROM two_view_geometries WHERE rows > 0")
    for pair_id, n_rows, cols, data in cur.fetchall():
        if not data or int(n_rows) <= 0:
            continue
        pair_id = int(pair_id)
        image_id2 = pair_id % 2147483647
        image_id1 = pair_id // 2147483647
        if image_id1 > image_id2:
            image_id1, image_id2 = image_id2, image_id1
        match_arr = np.frombuffer(data, dtype=np.uint32).reshape(int(n_rows), int(cols))
        name1 = image_id_to_name.get(int(image_id1))
        name2 = image_id_to_name.get(int(image_id2))
        if not name1 or not name2:
            continue
        if int(image_id1) not in keypoints or int(image_id2) not in keypoints:
            continue
        kp1 = keypoints[int(image_id1)]
        kp2 = keypoints[int(image_id2)]
        valid = (match_arr[:, 0] < kp1.shape[0]) & (match_arr[:, 1] < kp2.shape[0])
        valid_matches = match_arr[valid]
        if len(valid_matches) > 0:
            matches[(name1, name2)] = {"kp1": kp1, "kp2": kp2, "matches": valid_matches}

    conn.close()
    return matches


def filter_images(
    image_paths: List[Path],
    homographies: Dict,
    min_inliers: int = 0,
    max_images: int = 0,
    n_images: Optional[int] = None,
):
    import numpy as _np

    if n_images is None:
        n_images = len(image_paths)

    filtered = dict(homographies)
    if min_inliers > 0:
        filtered = {k: v for k, v in filtered.items() if int(v.get("n_inliers", 0)) >= int(min_inliers)}
        print(
            f"[FILTER] Min inliers filter: {len(filtered)}/{len(homographies)} pairs kept (threshold={min_inliers})",
            file=sys.stderr,
            flush=True,
        )

    if max_images > 0 and n_images > max_images:
        image_scores: Dict[int, int] = {}
        connectivity: Dict[int, int] = {}
        for (i, j), data in filtered.items():
            inl = int(data.get("n_inliers", 0))
            image_scores[i] = image_scores.get(i, 0) + inl
            image_scores[j] = image_scores.get(j, 0) + inl
            connectivity[i] = connectivity.get(i, 0) + 1
            connectivity[j] = connectivity.get(j, 0) + 1

        combined = {idx: image_scores.get(idx, 0) + connectivity.get(idx, 0) * 100 for idx in image_scores.keys()}
        keep = sorted(combined.keys(), key=lambda x: combined[x], reverse=True)[: int(max_images)]
        keep_set = set(keep)
        filtered_paths = [p for idx, p in enumerate(image_paths) if idx in keep_set]

        final_h = {}
        keep_sorted = sorted(keep_set)
        for (i, j), data in filtered.items():
            if i in keep_set and j in keep_set:
                new_i = keep_sorted.index(i)
                new_j = keep_sorted.index(j)
                final_h[(new_i, new_j)] = data
        return filtered_paths, final_h

    return image_paths, filtered


def compute_homographies(matches_data: Dict, n_images: int, image_name_to_index: Dict[str, int], use_affine: bool = False) -> Dict:
    import cv2

    homographies: Dict = {}
    for (name1, name2), data in matches_data.items():
        idx1 = image_name_to_index.get(name1)
        idx2 = image_name_to_index.get(name2)
        if idx1 is None or idx2 is None:
            continue
        kp1 = data["kp1"]
        kp2 = data["kp2"]
        mi = data["matches"]
        if len(mi) < 4:
            continue
        pts1 = kp1[mi[:, 0]]
        pts2 = kp2[mi[:, 1]]
        if use_affine:
            # Use similarity transform (4 DOF: uniform scale + rotation + translation)
            # NOT full affine (6 DOF) which allows shear - flatbed scans shouldn't have shear
            M, mask = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC,
                                                   ransacReprojThreshold=5.0,
                                                   maxIters=2000, confidence=0.99)
            if M is None:
                continue
            H = np.vstack([M, [0, 0, 1]])
            n_inl = int(mask.sum()) if mask is not None else 0
            if n_inl >= 10:
                homographies[(idx1, idx2)] = {"H": H, "n_matches": int(len(mi)), "n_inliers": n_inl, "is_affine": True}
        else:
            H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
            if H is None:
                continue
            n_inl = int(mask.sum()) if mask is not None else 0
            if n_inl >= 10:
                homographies[(idx1, idx2)] = {"H": H, "n_matches": int(len(mi)), "n_inliers": n_inl, "is_affine": False}
    return homographies


def stitch_with_homographies(
    image_paths: List[Path],
    homographies: Dict,
    use_affine: bool = False,
    blend_method: str = "multiband",
    use_source_alpha: bool = False,
    warp_interpolation: str = "linear",
    erode_border: bool = True,
    border_erosion_pixels: int = 5,
    log_dir: Optional[Path] = None,
    auto_detect_content: bool = False,
):
    import cv2
    import heapq

    n_images = len(image_paths)
    if n_images == 0:
        return None
    if n_images == 1:
        img = cv2.imread(str(image_paths[0]), cv2.IMREAD_UNCHANGED if use_source_alpha else cv2.IMREAD_COLOR)
        if img is not None and img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    # Connectivity map
    connections: Dict[int, int] = {}
    for (i, j) in homographies.keys():
        connections[i] = connections.get(i, 0) + 1
        connections[j] = connections.get(j, 0) + 1

    if not connections:
        img = cv2.imread(str(image_paths[0]), cv2.IMREAD_UNCHANGED if use_source_alpha else cv2.IMREAD_COLOR)
        if img is not None and img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    # Connected components
    def find_components(nodes, edges):
        visited = set()
        comps = []
        for s in nodes:
            if s in visited:
                continue
            q = [s]
            comp = set()
            while q:
                u = q.pop(0)
                if u in visited:
                    continue
                visited.add(u)
                comp.add(u)
                for (a, b) in edges:
                    if a == u and b not in visited:
                        q.append(b)
                    elif b == u and a not in visited:
                        q.append(a)
            comps.append(comp)
        return comps

    nodes = set(connections.keys())
    comps = find_components(nodes, homographies.keys())
    comps.sort(key=len, reverse=True)
    largest = comps[0]

    # Improved reference selection: prefer well-connected images with higher resolution
    # This minimizes scale drift and preserves detail
    candidate_indices = sorted(largest)

    # Get image dimensions for candidates (quick header read)
    img_sizes = {}
    for idx in candidate_indices:
        try:
            img = cv2.imread(str(image_paths[idx]), cv2.IMREAD_UNCHANGED)
            if img is not None:
                h, w = img.shape[:2]
                img_sizes[idx] = (h, w, h * w)  # height, width, total pixels
                del img
        except Exception:
            pass

    if img_sizes:
        # Find median resolution (by total pixels)
        sorted_by_pixels = sorted(img_sizes.items(), key=lambda x: x[1][2])
        median_idx = len(sorted_by_pixels) // 2

        # Prefer images with higher resolution and good connectivity
        # Weight: 70% resolution rank, 30% connectivity
        scores = []
        max_conn = max(connections.get(idx, 1) for idx in img_sizes.keys())
        for rank, (idx, (h, w, pixels)) in enumerate(sorted_by_pixels):
            conn = connections.get(idx, 1)
            # Higher rank = larger resolution = better
            res_score = rank / len(sorted_by_pixels)
            conn_score = conn / max_conn
            combined = 0.7 * res_score + 0.3 * conn_score
            scores.append((combined, idx, h, w))

        # Pick highest scoring image
        scores.sort(reverse=True)
        ref_idx = scores[0][1]
        ref_h, ref_w = scores[0][2], scores[0][3]
        print(f"[PROGRESS] REF_SELECT: Chose image {ref_idx} ({ref_w}x{ref_h}) as reference "
              f"(connectivity={connections.get(ref_idx, 0)}, score={scores[0][0]:.3f})", file=sys.stderr, flush=True)
    else:
        # Fallback to original behavior
        ref_idx = n_images // 2
        if ref_idx not in largest:
            ref_idx = sorted(largest)[len(largest) // 2]
        print(f"[PROGRESS] REF_SELECT: Using fallback reference {ref_idx}", file=sys.stderr, flush=True)

    # Build adjacency with filtering
    adjacency: Dict[int, List[Tuple[int, np.ndarray, int]]] = {}
    MIN_INLIERS = 10
    MIN_TRANSLATION = 30.0
    filtered_edges = {}
    for (i, j), data in homographies.items():
        if i not in largest or j not in largest:
            continue
        if int(data["n_inliers"]) < MIN_INLIERS:
            continue
        H = data["H"]
        tx, ty = float(H[0, 2]), float(H[1, 2])
        if float(np.hypot(tx, ty)) < MIN_TRANSLATION:
            continue
        filtered_edges[(i, j)] = data
        adjacency.setdefault(i, [])
        adjacency.setdefault(j, [])
        w = int(data["n_inliers"]) + (5 if abs(i - j) == 1 else 0)
        adjacency[i].append((j, data["H"], w))
        adjacency[j].append((i, np.linalg.inv(data["H"]), w))

    # Dijkstra max-bottleneck
    H_to_ref = {ref_idx: np.eye(3)}
    hop_count = {ref_idx: 0}  # Track distance from reference
    best_weight_to = {ref_idx: float("inf")}
    processed = set()
    pq = [(0, ref_idx, np.eye(3), 0)]  # Added hop count to queue
    while pq:
        neg_w, cur_idx, H_cur, hops = heapq.heappop(pq)
        if cur_idx in processed:
            continue
        processed.add(cur_idx)
        H_to_ref[cur_idx] = H_cur
        hop_count[cur_idx] = hops
        for nb, H_edge, edge_w in adjacency.get(cur_idx, []):
            if nb in processed:
                continue
            path_w = min(best_weight_to.get(cur_idx, float("inf")), float(edge_w))
            if path_w > best_weight_to.get(nb, 0):
                best_weight_to[nb] = path_w
                H_nb = H_cur @ np.linalg.inv(H_edge)
                heapq.heappush(pq, (-int(min(path_w, 100000)), nb, H_nb, hops + 1))

    # Canvas from transformed corners
    corners_all = []
    canvas_dims = {}  # Store dimensions for verification during warping
    for idx, H in H_to_ref.items():
        img = cv2.imread(str(image_paths[idx]), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        h, w = img.shape[:2]
        canvas_dims[idx] = (h, w, img.shape[2] if img.ndim == 3 else 1)
        del img
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        if use_affine:
            M_affine = H[:2, :]
            transformed = cv2.transform(corners, M_affine)
        else:
            transformed = cv2.perspectiveTransform(corners, H)
        corners_all.append(transformed)

    if not corners_all:
        return None
    corners_all = np.concatenate(corners_all, axis=0)
    x_min, y_min = corners_all.min(axis=0).ravel()
    x_max, y_max = corners_all.max(axis=0).ravel()

    offset_x = -int(np.floor(x_min))
    offset_y = -int(np.floor(y_min))
    output_w = int(np.ceil(x_max - x_min))
    output_h = int(np.ceil(y_max - y_min))

    max_dim = 15000
    if output_w > max_dim or output_h > max_dim:
        scale = max_dim / max(output_w, output_h)
        output_w = int(output_w * scale)
        output_h = int(output_h * scale)
        offset_x = int(offset_x * scale)
        offset_y = int(offset_y * scale)
        S = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], dtype=np.float64)
        H_to_ref = {k: S @ v for k, v in H_to_ref.items()}

    T = np.array([[1, 0, offset_x], [0, 1, offset_y], [0, 0, 1]], dtype=np.float64)

    # ============================================================
    # SIMILARITY-CONSTRAINED BUNDLE ADJUSTMENT
    # Optimizes 4 parameters per image: (scale, theta, tx, ty)
    # This enforces uniform scaling (no shear) for flatbed scans
    # ============================================================
    print(f"[PROGRESS] ========== SIMILARITY BUNDLE ADJUSTMENT (4-DOF) ==========", file=sys.stderr, flush=True)

    try:
        from scipy.optimize import least_squares

        all_indices_ba = sorted(H_to_ref.keys())
        n_images_ba = len(all_indices_ba)
        idx_to_var_ba = {idx: i for i, idx in enumerate(all_indices_ba)}

        if n_images_ba > 2:
            # Get reference dimensions for expected scale calculation
            ref_h_ba, ref_w_ba, _ = canvas_dims.get(ref_idx, (1000, 1000, 3))
            ref_diag_ba = np.sqrt(ref_w_ba**2 + ref_h_ba**2)

            def H_to_similarity_params(H):
                """Extract (scale, theta, tx, ty) from similarity matrix."""
                a, b, tx = H[0, 0], H[0, 1], H[0, 2]
                c, d, ty = H[1, 0], H[1, 1], H[1, 2]
                # For similarity: a = s*cos(θ), b = -s*sin(θ), c = s*sin(θ), d = s*cos(θ)
                scale = np.sqrt(a**2 + c**2)  # or equivalently sqrt(b**2 + d**2)
                theta = np.arctan2(c, a)  # rotation angle
                return scale, theta, tx, ty

            def similarity_params_to_H(s, theta, tx, ty):
                """Convert (scale, theta, tx, ty) to 3x3 similarity matrix."""
                cos_t, sin_t = np.cos(theta), np.sin(theta)
                return np.array([
                    [s * cos_t, -s * sin_t, tx],
                    [s * sin_t,  s * cos_t, ty],
                    [0,          0,          1]
                ], dtype=np.float64)

            def params_to_H_ba(params, var_i):
                """Extract similarity params for image var_i and convert to matrix."""
                off = var_i * 4  # 4 params per image now
                s, theta, tx, ty = params[off:off+4]
                return similarity_params_to_H(s, theta, tx, ty)

            # Extract initial similarity parameters: [s, theta, tx, ty] per image
            initial_params = []
            for idx in all_indices_ba:
                H = H_to_ref[idx]
                s, theta, tx, ty = H_to_similarity_params(H)
                initial_params.extend([s, theta, tx, ty])
            initial_params = np.array(initial_params, dtype=np.float64)

            # Pre-compute valid pairs for consistent residual array size
            # First get all candidate pairs
            candidate_pairs = [(idx_i, idx_j, h_data) for (idx_i, idx_j), h_data in homographies.items()
                               if idx_i in idx_to_var_ba and idx_j in idx_to_var_ba]

            # Filter pairs by initial residual (100px threshold)
            valid_pairs = []
            ba_outlier_threshold = 50  # pixels
            for idx_i, idx_j, h_data in candidate_pairs:
                try:
                    H_i = H_to_ref[idx_i]
                    H_j = H_to_ref[idx_j]
                    H_ij = h_data["H"]
                    H_i_inv = np.linalg.inv(H_i)
                    residual_matrix = H_j @ H_ij @ H_i_inv
                    res_mag = np.sqrt(residual_matrix[0, 2]**2 + residual_matrix[1, 2]**2)
                    if res_mag < ba_outlier_threshold:
                        valid_pairs.append((idx_i, idx_j, h_data))
                except:
                    pass  # Skip pairs that fail

            n_filtered = len(candidate_pairs) - len(valid_pairs)
            print(f"[DEBUG] BA pair filtering: {len(candidate_pairs)} candidates, {len(valid_pairs)} kept, {n_filtered} filtered (>{ba_outlier_threshold}px)", file=sys.stderr, flush=True)
            n_pair_residuals = len(valid_pairs) * 4  # 4 residuals per pair (scale, theta, tx, ty)
            n_scale_residuals = len(all_indices_ba)  # 1 scale residual per image (uniform scale!)
            n_anchor_residuals = 4  # 4 params for reference anchor
            total_residuals = n_pair_residuals + n_scale_residuals + n_anchor_residuals

            def residual_function_ba(params):
                residuals = np.zeros(total_residuals, dtype=np.float64)
                res_idx = 0

                # Pairwise consistency residuals
                for idx_i, idx_j, h_data in valid_pairs:
                    var_i, var_j = idx_to_var_ba[idx_i], idx_to_var_ba[idx_j]
                    H_i, H_j = params_to_H_ba(params, var_i), params_to_H_ba(params, var_j)
                    H_ij = h_data["H"]
                    n_inliers = h_data.get("n_inliers", 10)
                    weight = np.sqrt(max(n_inliers, 1)) * 0.1
                    try:
                        H_i_inv = np.linalg.inv(H_i)
                        rm = H_j @ H_ij @ H_i_inv
                        # For similarity, rm should be close to identity
                        # Extract similarity residuals: scale deviation, rotation deviation, translation
                        rm_s, rm_theta, rm_tx, rm_ty = H_to_similarity_params(rm)
                        residuals[res_idx:res_idx+4] = [
                            (rm_s - 1.0) * weight * 2.0,  # scale should be 1
                            rm_theta * weight * 2.0,       # rotation should be 0
                            np.clip(rm_tx * weight * 0.01, -50, 50),
                            np.clip(rm_ty * weight * 0.01, -50, 50)
                        ]
                    except:
                        pass  # Leave as zeros
                    res_idx += 4

                # Scale regularization - RESOLUTION-AWARE (single residual per image)
                # Strong weight to prevent drift while allowing real magnification variations
                scale_weight = 10.0
                for idx in all_indices_ba:
                    off = idx_to_var_ba[idx] * 4
                    s = params[off]  # Direct access to scale parameter

                    # Calculate expected scale based on resolution ratio to reference
                    if idx in canvas_dims:
                        img_h, img_w, _ = canvas_dims[idx]
                        img_diag = np.sqrt(img_w**2 + img_h**2)
                        exp_s = ref_diag_ba / img_diag
                    else:
                        exp_s = 1.0

                    if s > 0.01:
                        residuals[res_idx] = scale_weight * (np.log(s) - np.log(exp_s))
                    res_idx += 1

                # Reference anchor: scale=expected, theta=0, tx=ty=0
                off_ref = idx_to_var_ba[ref_idx] * 4
                s_ref, theta_ref, tx_ref, ty_ref = params[off_ref:off_ref+4]
                aw = 10.0
                # Reference expected scale is 1.0 (it's the reference!)
                residuals[res_idx:res_idx+4] = [
                    (s_ref - 1.0) * aw,
                    theta_ref * aw,
                    tx_ref * aw * 0.01,
                    ty_ref * aw * 0.01
                ]
                return residuals

            initial_cost = np.sqrt(np.mean(residual_function_ba(initial_params)**2))
            n_params = len(initial_params)
            print(f"[PROGRESS] BA_INIT: {n_images_ba} imgs, {len(valid_pairs)} pairs, {n_params} params, {total_residuals} residuals, RMS={initial_cost:.4f}",
                  file=sys.stderr, flush=True)
            print(f"[DEBUG] BA settings: max_nfev=10, ftol=1e-6, xtol=1e-6, method=lm", file=sys.stderr, flush=True)

            import time as _time
            _ba_start = _time.time()
            result = least_squares(residual_function_ba, initial_params, method='lm', max_nfev=10,
                                   ftol=1e-6, xtol=1e-6, verbose=0)
            _ba_elapsed = _time.time() - _ba_start

            final_cost = np.sqrt(np.mean(residual_function_ba(result.x)**2))
            print(f"[DEBUG] BA completed: nfev={result.nfev}, njev={result.njev}, status={result.status}, time={_ba_elapsed:.2f}s", file=sys.stderr, flush=True)
            print(f"[DEBUG] BA message: {result.message}", file=sys.stderr, flush=True)
            if final_cost < initial_cost:
                improvement = (initial_cost - final_cost) / initial_cost * 100
                print(f"[PROGRESS] BA_DONE: RMS {initial_cost:.4f} -> {final_cost:.4f} ({improvement:.1f}%)",
                      file=sys.stderr, flush=True)
                for idx in all_indices_ba:
                    H_to_ref[idx] = params_to_H_ba(result.x, idx_to_var_ba[idx])
                # Recalculate canvas
                corners_ba = []
                for idx, H in H_to_ref.items():
                    if idx not in canvas_dims: continue
                    h, w, _ = canvas_dims[idx]
                    c = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
                    corners_ba.append(cv2.transform(c, H[:2,:]) if use_affine else cv2.perspectiveTransform(c, H))
                if corners_ba:
                    corners_ba = np.concatenate(corners_ba, axis=0)
                    x_min, y_min = corners_ba.min(axis=0).ravel()
                    x_max, y_max = corners_ba.max(axis=0).ravel()
                    offset_x, offset_y = -int(np.floor(x_min)), -int(np.floor(y_min))
                    output_w, output_h = int(np.ceil(x_max-x_min)), int(np.ceil(y_max-y_min))
                    T = np.array([[1,0,offset_x],[0,1,offset_y],[0,0,1]], dtype=np.float64)
            else:
                print(f"[PROGRESS] BA_SKIP: No improvement", file=sys.stderr, flush=True)
    except ImportError as e:
        print(f"[PROGRESS] BA_SKIP: scipy not available ({e})", file=sys.stderr, flush=True)
    except Exception as e:
        import traceback
        print(f"[PROGRESS] BA_ERROR: {e}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)

    print(f"[PROGRESS] ========== SCALE VERIFICATION (POST-BA) ==========", file=sys.stderr, flush=True)
    # Verify scale factors after BA - BA should now handle resolution-aware scaling
    # This is diagnostic only - no transform modifications
    ref_h, ref_w, _ = canvas_dims.get(ref_idx, (1, 1, 1))
    ref_diag = np.sqrt(ref_w**2 + ref_h**2)

    scale_issues = []
    all_scales = []
    for idx, H in H_to_ref.items():
        if idx not in canvas_dims:
            continue

        # Calculate expected scale based on resolution ratio to reference
        img_h, img_w, _ = canvas_dims[idx]
        img_diag = np.sqrt(img_w**2 + img_h**2)
        expected_scale = ref_diag / img_diag

        # Extract actual scale factors
        a, b = H[0, 0], H[0, 1]
        c, d = H[1, 0], H[1, 1]
        scale_x = np.sqrt(a**2 + c**2)
        scale_y = np.sqrt(b**2 + d**2)
        hops = hop_count.get(idx, -1)

        # Check deviation from expected (not from 1.0!)
        dev_x = abs(scale_x - expected_scale) / expected_scale * 100
        dev_y = abs(scale_y - expected_scale) / expected_scale * 100
        all_scales.append((idx, scale_x, scale_y, expected_scale, hops))

        # Flag significant deviations (>5% from expected)
        if dev_x > 5.0 or dev_y > 5.0:
            scale_issues.append((idx, scale_x, scale_y, expected_scale, hops, max(dev_x, dev_y)))

    # Print summary
    if all_scales:
        scales_x = [s[1] for s in all_scales]
        scales_y = [s[2] for s in all_scales]
        print(f"[PROGRESS] SCALE_DIAG: ref={ref_idx}, n={len(all_scales)}, "
              f"sx=[{min(scales_x):.3f}-{max(scales_x):.3f}], sy=[{min(scales_y):.3f}-{max(scales_y):.3f}]",
              file=sys.stderr, flush=True)

    if scale_issues:
        print(f"[PROGRESS] SCALE_WARNING: {len(scale_issues)} images have scale >5% off from expected", file=sys.stderr, flush=True)
        for idx, sx, sy, exp, hops, dev in sorted(scale_issues, key=lambda x: -x[5])[:10]:
            img_name = Path(image_paths[idx]).name if idx < len(image_paths) else f"idx_{idx}"
            print(f"[PROGRESS] SCALE_ISSUE: {img_name} hops={hops} sx={sx:.3f} sy={sy:.3f} expected={exp:.3f} dev={dev:.1f}%",
                  file=sys.stderr, flush=True)
    else:
        print(f"[PROGRESS] SCALE_OK: All images within 5% of expected scale", file=sys.stderr, flush=True)

    # Note: No transform modifications here - BA should handle scale correctly now
    # If scale issues persist, the BA parameters may need tuning

    if True:  # Always enter this block for canvas recalculation
        # ============================================================
        # GLOBAL TRANSLATION OPTIMIZATION (Iterative Linear Least Squares)
        # Based on MegaStitch approach: minimize pairwise consistency error
        # For each pair (i,j) with H_ij: ideally H_i = H_j @ H_ij
        # We solve for translation adjustments that minimize residuals
        # Uses iterative refinement for better convergence
        # ============================================================
        print(f"[PROGRESS] GLOBAL_OPT: Starting global translation optimization...", file=sys.stderr, flush=True)

        try:
            from scipy import sparse
            from scipy.sparse.linalg import lsqr

            all_indices = sorted(H_to_ref.keys())
            idx_to_var = {idx: i for i, idx in enumerate(all_indices)}
            n_vars = len(all_indices) * 2  # tx, ty for each image

            # Track per-image residuals for debugging
            def compute_per_image_residuals():
                """Compute average residual per image"""
                img_residuals = {idx: [] for idx in all_indices}
                for (idx_i, idx_j), h_data in homographies.items():
                    if idx_i not in H_to_ref or idx_j not in H_to_ref:
                        continue
                    H_ij = h_data["H"]
                    H_i = H_to_ref[idx_i]
                    H_j = H_to_ref[idx_j]
                    try:
                        H_i_inv = np.linalg.inv(H_i)
                        residual_matrix = H_j @ H_ij @ H_i_inv
                        res_tx = residual_matrix[0, 2]
                        res_ty = residual_matrix[1, 2]
                        res_mag = np.sqrt(res_tx**2 + res_ty**2)
                        if res_mag < 50:  # Skip outliers
                            img_residuals[idx_i].append(res_mag)
                            img_residuals[idx_j].append(res_mag)
                    except Exception:
                        pass
                # Compute mean per image
                return {idx: np.mean(r) if r else 0 for idx, r in img_residuals.items()}

            # Robust weighting function (Huber-like)
            # Down-weights constraints with high residuals to prevent outliers from dominating
            def huber_weight(residual_mag, threshold=20.0):
                """Huber weight: 1 for small residuals, decreases for large ones"""
                if residual_mag <= threshold:
                    return 1.0
                else:
                    return threshold / residual_mag

            def compute_image_confidence():
                """Compute per-image confidence based on average residual across all its constraints"""
                img_residuals = {idx: [] for idx in all_indices}
                for (idx_i, idx_j), h_data in homographies.items():
                    if idx_i not in H_to_ref or idx_j not in H_to_ref:
                        continue
                    H_ij = h_data["H"]
                    H_i = H_to_ref[idx_i]
                    H_j = H_to_ref[idx_j]
                    try:
                        H_i_inv = np.linalg.inv(H_i)
                        residual_matrix = H_j @ H_ij @ H_i_inv
                        res_mag = np.sqrt(residual_matrix[0, 2]**2 + residual_matrix[1, 2]**2)
                        if res_mag < 50:
                            img_residuals[idx_i].append(res_mag)
                            img_residuals[idx_j].append(res_mag)
                    except Exception:
                        pass
                # Compute confidence: high avg residual = low confidence
                confidence = {}
                residual_threshold = 50.0  # Images with avg > 50px get down-weighted
                for idx, residuals in img_residuals.items():
                    if residuals:
                        avg = np.mean(residuals)
                        # Confidence decreases as avg residual increases
                        if avg <= residual_threshold:
                            confidence[idx] = 1.0
                        else:
                            confidence[idx] = residual_threshold / avg
                    else:
                        confidence[idx] = 1.0
                return confidence

            # Iterative optimization with robust reweighting (IRLS) + per-image confidence
            max_iterations = 10  # More iterations for better convergence
            convergence_threshold = 0.005  # Tighter convergence
            prev_avg_residual = float('inf')
            huber_threshold = 20.0  # Lower threshold for more aggressive down-weighting
            image_confidence = {idx: 1.0 for idx in all_indices}  # Start with full confidence
            excluded_images = set()  # Images with consistently terrible alignment

            for iteration in range(max_iterations):
                # After iteration 3, identify and exclude images with very high residuals
                if iteration == 3:
                    per_img = compute_per_image_residuals()
                    # Exclude images with avg residual > 100px - they have fundamentally bad matches
                    for idx, res in per_img.items():
                        if res > 100:
                            excluded_images.add(idx)
                    if excluded_images:
                        excluded_names = [Path(image_paths[idx]).name if idx < len(image_paths) else f"idx_{idx}"
                                         for idx in excluded_images]
                        print(f"[PROGRESS] GLOBAL_OPT: Excluding {len(excluded_images)} images with bad matches: {', '.join(excluded_names[:5])}", file=sys.stderr, flush=True)

                # Update per-image confidence after first 2 iterations
                if iteration >= 2:
                    image_confidence = compute_image_confidence()
                    n_low_conf = sum(1 for c in image_confidence.values() if c < 1.0)
                    if iteration == 2:
                        print(f"[PROGRESS] GLOBAL_OPT: {n_low_conf} images have reduced confidence", file=sys.stderr, flush=True)

                rows = []
                cols = []
                data = []
                b_vec = []
                n_constraints = 0
                total_residual = 0.0
                n_downweighted = 0

                for (idx_i, idx_j), h_data in homographies.items():
                    if idx_i not in H_to_ref or idx_j not in H_to_ref:
                        continue

                    # Skip constraints involving excluded (bad match) images
                    if idx_i in excluded_images or idx_j in excluded_images:
                        continue

                    H_ij = h_data["H"]  # Transform from i to j
                    n_inliers = h_data.get("n_inliers", 10)

                    H_i = H_to_ref[idx_i]
                    H_j = H_to_ref[idx_j]

                    try:
                        H_i_inv = np.linalg.inv(H_i)
                        residual_matrix = H_j @ H_ij @ H_i_inv

                        res_tx = residual_matrix[0, 2]
                        res_ty = residual_matrix[1, 2]
                        res_mag = np.sqrt(res_tx**2 + res_ty**2)

                        # Skip if residual is extreme (likely completely wrong match)
                        if res_mag > 50:
                            continue

                        total_residual += res_mag
                        n_constraints += 1

                        # Robust Huber weighting - down-weight high-residual constraints
                        robust_weight = huber_weight(res_mag, huber_threshold)
                        if robust_weight < 1.0:
                            n_downweighted += 1

                        # Combined weight: inlier confidence * robust weight * per-image confidence
                        # Per-image confidence down-weights images with consistently high residuals
                        img_conf = np.sqrt(image_confidence.get(idx_i, 1.0) * image_confidence.get(idx_j, 1.0))
                        weight = np.sqrt(n_inliers) * robust_weight * img_conf

                        var_i_tx = idx_to_var[idx_i] * 2
                        var_i_ty = idx_to_var[idx_i] * 2 + 1
                        var_j_tx = idx_to_var[idx_j] * 2
                        var_j_ty = idx_to_var[idx_j] * 2 + 1

                        # X constraint: adj_tx_i - adj_tx_j = res_tx
                        row = len(b_vec)
                        rows.extend([row, row])
                        cols.extend([var_i_tx, var_j_tx])
                        data.extend([weight, -weight])
                        b_vec.append(weight * res_tx)

                        # Y constraint: adj_ty_i - adj_ty_j = res_ty
                        row = len(b_vec)
                        rows.extend([row, row])
                        cols.extend([var_i_ty, var_j_ty])
                        data.extend([weight, -weight])
                        b_vec.append(weight * res_ty)

                    except Exception:
                        continue

                if n_constraints == 0:
                    print(f"[PROGRESS] GLOBAL_OPT: No valid constraints, skipping", file=sys.stderr, flush=True)
                    break

                avg_residual = total_residual / n_constraints

                # Check convergence
                if iteration > 0 and (prev_avg_residual - avg_residual) < convergence_threshold:
                    print(f"[PROGRESS] GLOBAL_OPT: Converged at iteration {iteration+1} (improvement < {convergence_threshold}px)", file=sys.stderr, flush=True)
                    break

                print(f"[PROGRESS] GLOBAL_OPT: Iter {iteration+1}: {n_constraints} constraints ({n_downweighted} down-weighted), avg residual={avg_residual:.2f}px", file=sys.stderr, flush=True)

                # Very low regularization - let the constraints drive the solution
                reg_weight = 0.01
                for i in range(n_vars):
                    row = len(b_vec)
                    rows.append(row)
                    cols.append(i)
                    data.append(reg_weight)
                    b_vec.append(0)

                # Fix reference image (zero adjustment)
                if ref_idx in idx_to_var:
                    big_weight = 1000
                    var_ref_tx = idx_to_var[ref_idx] * 2
                    var_ref_ty = idx_to_var[ref_idx] * 2 + 1

                    row = len(b_vec)
                    rows.append(row)
                    cols.append(var_ref_tx)
                    data.append(big_weight)
                    b_vec.append(0)

                    row = len(b_vec)
                    rows.append(row)
                    cols.append(var_ref_ty)
                    data.append(big_weight)
                    b_vec.append(0)

                # Build sparse matrix and solve
                A = sparse.csr_matrix((data, (rows, cols)), shape=(len(b_vec), n_vars))
                b = np.array(b_vec)

                result = lsqr(A, b, atol=1e-10, btol=1e-10)
                adjustments = result[0]

                # Apply adjustments
                max_adj = 0
                adjustments_applied = []
                for idx in all_indices:
                    var_i = idx_to_var[idx]
                    adj_tx = adjustments[var_i * 2]
                    adj_ty = adjustments[var_i * 2 + 1]

                    H_to_ref[idx][0, 2] += adj_tx
                    H_to_ref[idx][1, 2] += adj_ty

                    adj_mag = np.sqrt(adj_tx**2 + adj_ty**2)
                    max_adj = max(max_adj, adj_mag)
                    adjustments_applied.append((idx, adj_mag))

                mean_adj = np.mean([a[1] for a in adjustments_applied])
                print(f"[PROGRESS] GLOBAL_OPT: Iter {iteration+1}: Applied adjustments (max={max_adj:.2f}px, mean={mean_adj:.2f}px)", file=sys.stderr, flush=True)

                prev_avg_residual = avg_residual

            # Final verification and per-image reporting
            per_img_res = compute_per_image_residuals()
            worst_images = sorted(per_img_res.items(), key=lambda x: -x[1])[:10]

            # Compute final average
            final_residuals = [r for r in per_img_res.values() if r > 0]
            if final_residuals:
                final_avg = np.mean(final_residuals)
                print(f"[PROGRESS] GLOBAL_OPT: Final avg residual={final_avg:.2f}px", file=sys.stderr, flush=True)

                # Report worst aligned images
                print(f"[PROGRESS] GLOBAL_OPT: Worst aligned images:", file=sys.stderr, flush=True)
                for idx, res in worst_images[:5]:
                    if res > 0:
                        img_name = Path(image_paths[idx]).name if idx < len(image_paths) else f"idx_{idx}"
                        print(f"[PROGRESS] GLOBAL_OPT:   {img_name}: avg_residual={res:.1f}px", file=sys.stderr, flush=True)

        except ImportError:
            print(f"[PROGRESS] GLOBAL_OPT: scipy not available, skipping global optimization", file=sys.stderr, flush=True)
        except Exception as e:
            import traceback
            print(f"[PROGRESS] GLOBAL_OPT: Error during optimization: {e}", file=sys.stderr, flush=True)
            traceback.print_exc()

        # ALWAYS recompute canvas after scale analysis to catch any drift
        # (even if normalized_count = 0, small accumulated errors could cause truncation)
        corners_all_new = []
        for idx, H in H_to_ref.items():
            if idx not in canvas_dims:
                continue
            h, w, _ = canvas_dims[idx]
            corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
            # Use raw H without old T - we'll compute fresh bounds
            if use_affine:
                transformed = cv2.transform(corners, H[:2, :])
            else:
                transformed = cv2.perspectiveTransform(corners, H)
            corners_all_new.append(transformed)

        if corners_all_new:
            corners_all_new = np.concatenate(corners_all_new, axis=0)
            x_min_new, y_min_new = corners_all_new.min(axis=0).ravel()
            x_max_new, y_max_new = corners_all_new.max(axis=0).ravel()

            # Add safety margin to prevent truncation from floating point rounding
            canvas_margin = 20  # pixels
            x_min_new -= canvas_margin
            y_min_new -= canvas_margin
            x_max_new += canvas_margin
            y_max_new += canvas_margin

            offset_x = -int(np.floor(x_min_new))
            offset_y = -int(np.floor(y_min_new))
            output_w = int(np.ceil(x_max_new - x_min_new))
            output_h = int(np.ceil(y_max_new - y_min_new))
            T = np.array([[1, 0, offset_x], [0, 1, offset_y], [0, 0, 1]], dtype=np.float64)
            print(f"[PROGRESS] SCALE_FIX: Recalculated canvas size {output_w}x{output_h} (with {canvas_margin}px margin)", file=sys.stderr, flush=True)

    interp_map = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4,
        "realesrgan": cv2.INTER_LANCZOS4,
    }
    interp_flag = interp_map.get(str(warp_interpolation).lower(), cv2.INTER_LINEAR)

    blend_lower = str(blend_method).lower()
    use_streaming = blend_lower in ["feather", "linear", "autostitch", "mosaic", "autostitch_feather"]
    if use_streaming:
        blender = StreamingBlender(output_h, output_w, blend_lower)
        mmap_store = None
    else:
        blender = None
        mmap_store = MemoryMappedImageStore(Path(tempfile.mkdtemp(prefix="colmap_mmap_")))

    # Warping progress tracking
    total_to_warp = len(H_to_ref)
    warped_count = 0
    print(f"[PROGRESS] Warping: Starting {total_to_warp} images...", file=sys.stderr, flush=True)

    for idx, H in H_to_ref.items():
        warped_count += 1
        if warped_count % max(1, total_to_warp // 10) == 0 or warped_count == total_to_warp:
            pct = int(100 * warped_count / total_to_warp)
            print(f"[PROGRESS] Warping: {warped_count}/{total_to_warp} images ({pct}%)", file=sys.stderr, flush=True)

        img_path = str(image_paths[idx])
        img_raw = cv2.imread(img_path, cv2.IMREAD_UNCHANGED if use_source_alpha else cv2.IMREAD_COLOR)
        if img_raw is None:
            continue
        if use_source_alpha and img_raw.ndim == 3 and img_raw.shape[2] == 4:
            img = img_raw[:, :, :3]
            alpha_original = img_raw[:, :, 3]

            # DIAGNOSTIC: Check alpha coverage vs full image
            h_raw, w_raw = alpha_original.shape
            alpha_pixels = np.sum(alpha_original > 0)
            total_pixels = h_raw * w_raw
            coverage_pct = 100.0 * alpha_pixels / total_pixels

            # Find bounding box of non-zero alpha
            nonzero_rows = np.any(alpha_original > 0, axis=1)
            nonzero_cols = np.any(alpha_original > 0, axis=0)
            if np.any(nonzero_rows) and np.any(nonzero_cols):
                y_min, y_max = np.where(nonzero_rows)[0][[0, -1]]
                x_min, x_max = np.where(nonzero_cols)[0][[0, -1]]
                alpha_w = x_max - x_min + 1
                alpha_h = y_max - y_min + 1
                # Log if alpha region is significantly smaller than image
                if coverage_pct < 95 or alpha_w < w_raw - 10 or alpha_h < h_raw - 10:
                    img_name = Path(image_paths[idx]).name
                    print(f"[PROGRESS] ALPHA_SMALL: {img_name} covers {coverage_pct:.1f}%, "
                          f"bbox=({x_min},{y_min})-({x_max},{y_max}) vs {w_raw}x{h_raw}",
                          file=sys.stderr, flush=True)

            if erode_border and border_erosion_pixels > 0:
                ksz = int(border_erosion_pixels)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
                alpha_original = cv2.erode(alpha_original, kernel, iterations=2)
        else:
            img = img_raw if img_raw.ndim == 3 else cv2.cvtColor(img_raw, cv2.COLOR_GRAY2BGR)
            alpha_original = None
        del img_raw

        H_final = T @ H
        h, w = img.shape[:2]

        # CRITICAL: Verify dimensions match what was used for canvas calculation
        if idx in canvas_dims:
            canvas_h, canvas_w, canvas_ch = canvas_dims[idx]
            if h != canvas_h or w != canvas_w:
                print(f"[ERROR] Image {idx} DIMENSION MISMATCH! Canvas used {canvas_w}x{canvas_h} but warp got {w}x{h}",
                      file=sys.stderr, flush=True)
                print(f"[ERROR] This will cause scale/shift artifacts! "
                      f"(use_source_alpha={use_source_alpha}, canvas_ch={canvas_ch}, warp_ch={img.shape[2] if img.ndim==3 else 1})",
                      file=sys.stderr, flush=True)
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        transformed = cv2.transform(corners, H_final[:2, :]) if use_affine else cv2.perspectiveTransform(corners, H_final)
        x0 = int(np.floor(transformed[:, 0, 0].min()))
        y0 = int(np.floor(transformed[:, 0, 1].min()))
        x1 = int(np.ceil(transformed[:, 0, 0].max()))
        y1 = int(np.ceil(transformed[:, 0, 1].max()))
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(output_w, x1)
        y1 = min(output_h, y1)
        region_w = x1 - x0
        region_h = y1 - y0
        if region_w <= 0 or region_h <= 0:
            del img
            continue

        if alpha_original is None:
            if auto_detect_content:
                # Use content detection to find valid region (handles round images)
                img_name = Path(image_paths[idx]).name
                alpha_original = create_content_mask_for_warp(img, auto_detect=True, image_name=img_name)
            else:
                alpha_original = np.ones((h, w), dtype=np.uint8) * 255

        T_offset = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], dtype=np.float64)
        H_crop = T_offset @ H_final

        # Use NEAREST interpolation for alpha mask to preserve hard edges
        # (LANCZOS can create ringing artifacts on binary masks)
        alpha_interp = cv2.INTER_NEAREST

        if use_affine:
            M = H_crop[:2, :]
            warped = cv2.warpAffine(img, M, (region_w, region_h), flags=interp_flag)
            alpha = cv2.warpAffine(alpha_original, M, (region_w, region_h), flags=alpha_interp)
        else:
            warped = cv2.warpPerspective(img, H_crop, (region_w, region_h), flags=interp_flag)
            alpha = cv2.warpPerspective(alpha_original, H_crop, (region_w, region_h), flags=alpha_interp)

        del img
        del alpha_original

        # CRITICAL FIX for autostitch duplication/shift artifacts:
        # RGB warp uses LINEAR/CUBIC interpolation which causes edge pixels to be
        # interpolated with black border (borderValue=0), creating darkened/shifted edges.
        # Alpha uses INTER_NEAREST so it doesn't know about this.
        # Solution: Erode alpha by 1-2 pixels to exclude those interpolated RGB edges.
        if interp_flag != cv2.INTER_NEAREST:
            edge_erosion_px = 2
            if edge_erosion_px > 0:
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (edge_erosion_px * 2 + 1, edge_erosion_px * 2 + 1)
                )
                alpha = cv2.erode(alpha, kernel, iterations=1)

        bbox = (x0, y0, x1, y1)
        if use_streaming:
            blender.add_image(warped, alpha, bbox)
        else:
            mmap_store.add_image(warped, alpha, bbox, idx)

        del warped
        del alpha

    if use_streaming:
        pano = blender.finalize()
        return pano

    # Multiband path: load stored warps and blend using ImageBlender if available
    try:
        script_dir = Path(__file__).resolve().parent.parent
        if str(script_dir) not in sys.path:
            sys.path.insert(0, str(script_dir))
        from core.blender import ImageBlender

        aligned = []
        for i in range(len(mmap_store)):
            warped, alpha, bbox = mmap_store.get_image(i)
            aligned.append({"image": warped, "alpha": alpha, "bbox": bbox, "warped": True})
        blender2 = ImageBlender(method=blend_method, options={"hdr_mode": False, "anti_ghosting": False})
        pano = blender2.blend(aligned, padding=0, fit_all=False)
        return pano
    finally:
        mmap_store.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WSL COLMAP Bridge for GPU-accelerated stitching")
    parser.add_argument("--config", type=str, help="Path to config JSON file")
    parser.add_argument("image_paths_json", nargs="?", help="JSON string of image paths (legacy)")
    parser.add_argument("output_dir", nargs="?", help="Output directory (legacy)")
    args = parser.parse_args()

    try:
        if args.config:
            with open(args.config, "r", encoding="utf-8") as f:
                config = json.load(f)
            image_paths_json = json.dumps(config["image_paths"])
            output_dir = config["output_dir"]

            # Check if this is a reblend operation
            if config.get("mode") == "reblend" and config.get("cache_key"):
                result = run_reblend(
                    cache_key=config["cache_key"],
                    image_paths_json=image_paths_json,
                    output_dir=output_dir,
                    use_affine=config.get("use_affine", False),
                    blend_method=config.get("blend_method", "multiband"),
                    use_source_alpha=config.get("use_source_alpha", False),
                    warp_interpolation=config.get("warp_interpolation", "linear"),
                    erode_border=config.get("erode_border", True),
                    border_erosion_pixels=config.get("border_erosion_pixels", 5),
                    max_features=config.get("max_features", 8192),
                    auto_detect_content=config.get("auto_detect_content", False),
                )
            else:
                result = run_2d_stitch(
                    image_paths_json=image_paths_json,
                    output_dir=output_dir,
                    use_affine=config.get("use_affine", False),
                    blend_method=config.get("blend_method", "multiband"),
                    matcher_type=config.get("matcher_type", "exhaustive"),
                    sequential_overlap=config.get("sequential_overlap", 10),
                    gpu_index=config.get("gpu_index", -1),
                    num_threads=config.get("num_threads", -1),
                    max_features=config.get("max_features", 8192),
                    min_inliers=config.get("min_inliers", 0),
                    max_images=config.get("max_images", 0),
                    use_source_alpha=config.get("use_source_alpha", False),
                    remove_duplicates=config.get("remove_duplicates", False),
                    duplicate_threshold=config.get("duplicate_threshold", 0.92),
                    warp_interpolation=config.get("warp_interpolation", "linear"),
                    erode_border=config.get("erode_border", True),
                    border_erosion_pixels=config.get("border_erosion_pixels", 5),
                    auto_detect_content=config.get("auto_detect_content", False),
                )
        elif args.image_paths_json and args.output_dir:
            result = run_2d_stitch(args.image_paths_json, args.output_dir)
        else:
            print("Usage: wsl_colmap_bridge.py --config <config.json>", file=sys.stderr)
            sys.exit(1)

        print(json.dumps(result))
        sys.stdout.flush()
    except MemoryError as e:
        import traceback

        print(json.dumps({"success": False, "error": f"Out of memory: {e}", "traceback": traceback.format_exc()}))
        sys.exit(1)
    except Exception as e:
        import traceback

        print(json.dumps({"success": False, "error": str(e), "traceback": traceback.format_exc()}))
        sys.exit(1)