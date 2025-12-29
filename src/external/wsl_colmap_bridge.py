#!/usr/bin/env python3
"""
WSL COLMAP Bridge - GPU-accelerated pycolmap via WSL

This script runs inside WSL and provides GPU-accelerated COLMAP processing.
It's called from the Windows application via subprocess.
"""

import sys
import json
import os
from pathlib import Path


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


def run_2d_stitch(image_paths_json: str, output_dir: str, use_affine: bool = False, blend_method: str = "multiband") -> dict:
    """
    Run GPU-accelerated 2D stitching using pycolmap.

    Args:
        image_paths_json: JSON string of Windows image paths
        output_dir: Windows output directory path
        use_affine: Use affine transforms instead of homography
        blend_method: Blending method (multiband, feather, autostitch, linear)

    Returns:
        dict with results
    """
    import pycolmap
    import numpy as np
    import cv2
    import shutil
    import sqlite3
    import time
    
    # Parse input
    image_paths_win = json.loads(image_paths_json)
    image_paths = [Path(windows_to_wsl_path(p)) for p in image_paths_win]
    output_path = Path(windows_to_wsl_path(output_dir))
    n_images = len(image_paths)
    
    print(f"[PROGRESS] WSL COLMAP Bridge - GPU Mode (has_cuda={pycolmap.has_cuda})", file=sys.stderr, flush=True)
    print(f"[PROGRESS] Processing {n_images} images", file=sys.stderr, flush=True)
    
    # Create workspace
    workspace = output_path / "colmap_workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    
    images_dir = workspace / "images"
    database_path = workspace / "database.db"
    
    # Clean up old database if exists
    if database_path.exists():
        database_path.unlink()
    
    images_dir.mkdir(exist_ok=True)
    
    # Build mapping
    image_name_to_original = {}
    image_name_to_index = {}
    
    # Copy images with progress
    print(f"[PROGRESS] Copying {n_images} images to workspace...", file=sys.stderr, flush=True)
    start_time = time.time()
    for i, path in enumerate(image_paths):
        dest_name = f"{i:06d}{path.suffix}"
        dest = images_dir / dest_name
        shutil.copy2(path, dest)
        image_name_to_original[dest_name] = path
        image_name_to_index[dest_name] = i
        if (i + 1) % 10 == 0 or i == n_images - 1:
            print(f"[PROGRESS] Copied {i+1}/{n_images} images", file=sys.stderr, flush=True)
    
    # Feature extraction with GPU
    print(f"[PROGRESS] Extracting features (GPU) from {n_images} images...", file=sys.stderr, flush=True)
    print(f"[PROGRESS] This may take a few minutes (typically 0.1-2s per image depending on resolution and GPU)...", file=sys.stderr, flush=True)
    extract_start = time.time()
    pycolmap.extract_features(
        database_path,
        images_dir,
        camera_mode=pycolmap.CameraMode.PER_IMAGE,
        camera_model="PINHOLE"
    )
    extract_time = time.time() - extract_start
    print(f"[PROGRESS] Feature extraction complete in {extract_time:.1f}s ({extract_time/n_images:.2f}s per image)", file=sys.stderr, flush=True)
    
    # Feature matching with GPU
    n_pairs = n_images * (n_images - 1) // 2
    print(f"[PROGRESS] Matching features (GPU) - {n_pairs} image pairs...", file=sys.stderr, flush=True)
    match_start = time.time()
    pycolmap.match_exhaustive(database_path)
    match_time = time.time() - match_start
    print(f"[PROGRESS] Feature matching complete in {match_time:.1f}s", file=sys.stderr, flush=True)
    
    # Read matches and compute homographies
    print("[PROGRESS] Reading matches from database...", file=sys.stderr, flush=True)
    matches_data = read_matches_from_db(database_path)
    
    if not matches_data:
        return {"success": False, "error": "No matches found"}
    
    print(f"[PROGRESS] Found {len(matches_data)} image pairs with matches", file=sys.stderr, flush=True)
    
    # Load images
    print(f"[PROGRESS] Loading {n_images} images for stitching...", file=sys.stderr, flush=True)
    images = []
    for i, path in enumerate(image_paths):
        img = cv2.imread(str(path))
        if img is None:
            return {"success": False, "error": f"Failed to load: {path}"}
        images.append(img)
        if (i + 1) % 10 == 0 or i == n_images - 1:
            print(f"[PROGRESS] Loaded {i+1}/{n_images} images", file=sys.stderr, flush=True)
    
    # Compute transforms (homography or affine)
    transform_type = "affine" if use_affine else "homography"
    print(f"[PROGRESS] Computing {transform_type} transforms from matches...", file=sys.stderr, flush=True)
    homographies = compute_homographies(matches_data, images, image_name_to_index, use_affine)
    
    if not homographies:
        transform_type = "affine transforms" if use_affine else "homographies"
        return {"success": False, "error": f"Could not compute {transform_type}"}

    transform_type = "affine transforms" if use_affine else "homographies"
    print(f"[PROGRESS] Computed {len(homographies)} valid {transform_type}", file=sys.stderr, flush=True)
    
    # Stitch
    print("[PROGRESS] Stitching panorama (warping and blending)...", file=sys.stderr, flush=True)
    stitch_start = time.time()
    panorama = stitch_with_homographies(images, homographies, use_affine, blend_method)
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
        "n_images": len(images),
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


def compute_homographies(matches_data: dict, images: list, image_name_to_index: dict, use_affine: bool = False) -> dict:
    """Compute homographies from matches."""
    import cv2
    import numpy as np
    
    homographies = {}
    
    for (name1, name2), data in matches_data.items():
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

    return homographies


def stitch_with_homographies(images: list, homographies: dict, use_affine: bool = False, blend_method: str = "multiband"):
    """Stitch images using transforms (homography or affine) - optimized for ordered sequential captures."""
    import cv2
    import numpy as np
    import traceback
    
    # #region agent log - H1: Entry point
    print(f"[DEBUG] stitch_with_homographies entry: n_images={len(images)}, n_homographies={len(homographies)}", file=sys.stderr, flush=True)
    # #endregion
    
    n_images = len(images)
    if n_images == 0:
        return None
    if n_images == 1:
        return images[0]
    
    # Build connectivity and find reference - use middle image for sequential captures
    connections = {}
    for (i, j) in homographies.keys():
        connections[i] = connections.get(i, 0) + 1
        connections[j] = connections.get(j, 0) + 1
    
    if not connections:
        return images[0]
    
    # Use middle image as reference to minimize error accumulation
    ref_idx = n_images // 2
    if ref_idx not in connections:
        # Fallback to most connected if middle not in graph
        ref_idx = max(connections.keys(), key=lambda x: connections[x])
    
    # #region agent log - H3: Check connectivity
    print(f"[DEBUG] H3: ref_idx={ref_idx} (middle={n_images//2}), connections={connections}", file=sys.stderr, flush=True)
    # #endregion
    
    # Build adjacency graph
    # H in homographies[(i,j)] transforms points from image i to image j: p_j = H @ p_i
    adjacency = {}
    for (i, j), data in homographies.items():
        if i not in adjacency:
            adjacency[i] = []
        if j not in adjacency:
            adjacency[j] = []
        
        # Weight = number of inliers (let feature matching determine spatial adjacency)
        # Small sequential bonus for within-row connections, but don't override poor matches
        seq_bonus = 5 if abs(i - j) == 1 else 0
        weight = data['n_inliers'] + seq_bonus
        
        # Store both directions with correct transformations
        # From i to j: use H (transforms points i->j)
        # From j to i: use inv(H) (transforms points j->i)
        adjacency[i].append((j, data['H'], weight))
        adjacency[j].append((i, np.linalg.inv(data['H']), weight))
    
    # Compute cumulative homographies using BFS with priority (Dijkstra-like)
    import heapq
    H_to_ref = {ref_idx: np.eye(3)}
    visited = {ref_idx}
    # Priority queue: (negative_weight, distance_from_ref, current_idx)
    pq = [(0, 0, ref_idx)]
    
    while pq:
        neg_weight, dist, current = heapq.heappop(pq)
        
        if current != ref_idx and current in H_to_ref:
            # Already processed with a better path
            continue
        
        H_current = H_to_ref[current]
        
        for neighbor, H_edge, weight in adjacency.get(current, []):
            if neighbor not in visited:
                # H_edge transforms from current to neighbor
                # To get neighbor->ref: first go neighbor->current (inv(H_edge)), then current->ref (H_current)
                # But we want the transform that takes points in neighbor and maps to ref
                # So: p_ref = H_current @ inv(H_edge) @ p_neighbor
                # Simplified: H_to_ref[neighbor] = H_current @ inv(H_edge)
                # But H_edge already is current->neighbor, so neighbor->ref = inv(H_edge) @ H_current? No.
                # Let's think: H_current maps points in current to ref: p_ref = H_current @ p_current
                # H_edge maps points in current to neighbor: p_neighbor = H_edge @ p_current
                # So: p_current = inv(H_edge) @ p_neighbor
                # Therefore: p_ref = H_current @ inv(H_edge) @ p_neighbor
                H_neighbor = H_current @ np.linalg.inv(H_edge)
                H_to_ref[neighbor] = H_neighbor
                visited.add(neighbor)
                heapq.heappush(pq, (-weight, dist + 1, neighbor))
    
    # #region agent log - H3: Check visited images
    print(f"[DEBUG] H3: visited={sorted(visited)}, H_to_ref_keys={sorted(H_to_ref.keys())}, n_images={n_images}", file=sys.stderr, flush=True)
    # #endregion
    
    # Compute canvas size
    corners_all = []
    try:
        for idx, H in H_to_ref.items():
            h, w = images[idx].shape[:2]
            corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
            transformed = cv2.perspectiveTransform(corners, H)
            # #region agent log - H2: Check for invalid homography
            if np.any(np.isnan(transformed)) or np.any(np.isinf(transformed)):
                print(f"[DEBUG] H2: Invalid transform for idx={idx}, H={H.tolist()}, transformed={transformed.tolist()}", file=sys.stderr, flush=True)
            # #endregion
            corners_all.append(transformed)
    except Exception as e:
        print(f"[DEBUG] H2: Exception in perspectiveTransform: {e}\n{traceback.format_exc()}", file=sys.stderr, flush=True)
        raise
    
    corners_all = np.concatenate(corners_all, axis=0)
    x_min, y_min = corners_all.min(axis=0).ravel()
    x_max, y_max = corners_all.max(axis=0).ravel()
    
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

    # Prepare aligned images for ImageBlender
    print(f"[DEBUG] H5: Preparing {len(H_to_ref)} images for blending...", file=sys.stderr, flush=True)
    aligned_images = []

    for idx, H in H_to_ref.items():
        img = images[idx]
        H_final = T @ H

        # Compute bbox for this image
        h, w = img.shape[:2]
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)

        if use_affine:
            M_affine = H_final[:2, :]
            transformed = cv2.transform(corners, M_affine)
        else:
            transformed = cv2.perspectiveTransform(corners, H_final)

        x_min = int(transformed[0, :, 0].min())
        y_min = int(transformed[0, :, 1].min())
        x_max = int(transformed[0, :, 0].max())
        y_max = int(transformed[0, :, 1].max())

        # Warp the image
        if use_affine:
            M_affine = H_final[:2, :]
            warped = cv2.warpAffine(img, M_affine, (output_w, output_h))
        else:
            warped = cv2.warpPerspective(img, H_final, (output_w, output_h))

        aligned_images.append({
            'image': warped,
            'bbox': (x_min, y_min, x_max, y_max),
            'transform': H_final
        })

        print(f"[DEBUG] H6: Prepared image {idx} for blending ({len(aligned_images)}/{len(H_to_ref)})", file=sys.stderr, flush=True)

    # Use ImageBlender to blend
    print(f"[DEBUG] H7: Blending {len(aligned_images)} images using {blend_method} method...", file=sys.stderr, flush=True)
    sys.stderr.flush()

    # Try to import ImageBlender - if it fails, fall back to simple blending
    try:
        # Import ImageBlender - need to add parent directory to path for imports
        script_dir = Path(__file__).resolve().parent.parent
        print(f"[DEBUG] H7: script_dir={script_dir}", file=sys.stderr, flush=True)
        print(f"[DEBUG] H7: sys.path before={sys.path[:3]}", file=sys.stderr, flush=True)

        if str(script_dir) not in sys.path:
            sys.path.insert(0, str(script_dir))

        print(f"[DEBUG] H7: Attempting to import ImageBlender from core.blender...", file=sys.stderr, flush=True)
        from core.blender import ImageBlender
        print(f"[DEBUG] H7: ImageBlender imported successfully", file=sys.stderr, flush=True)

        blender = ImageBlender(
            method=blend_method,
            hdr_mode=False,
            anti_ghosting=False
        )

        panorama = blender.blend(aligned_images, padding=0, fit_all=False)
        print(f"[DEBUG] H7: ImageBlender completed successfully", file=sys.stderr, flush=True)

    except Exception as e:
        # Fall back to simple weighted averaging if ImageBlender fails
        print(f"[DEBUG] H7: ImageBlender failed ({e}), falling back to simple blending", file=sys.stderr, flush=True)
        import traceback
        print(f"[DEBUG] H7: Traceback: {traceback.format_exc()}", file=sys.stderr, flush=True)

        # Simple weighted average blending (fallback)
        panorama = np.zeros((output_h, output_w, 3), dtype=np.float32)
        weight_sum = np.zeros((output_h, output_w), dtype=np.float32)

        for aligned in aligned_images:
            warped = aligned['image'].astype(np.float32)

            # Create simple weight mask
            mask = (warped.sum(axis=2) > 0).astype(np.float32)

            # Accumulate
            for c in range(3):
                panorama[:, :, c] += warped[:, :, c] * mask
            weight_sum += mask

        # Normalize
        weight_sum = np.maximum(weight_sum, 1e-6)
        for c in range(3):
            panorama[:, :, c] /= weight_sum

        panorama = np.clip(panorama, 0, 255).astype(np.uint8)
        print(f"[DEBUG] H7: Simple blending completed", file=sys.stderr, flush=True)

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
            use_affine = config.get('use_affine', False)  # Default to False for backward compatibility
            blend_method = config.get('blend_method', 'multiband')  # Default to multiband
        elif args.image_paths_json and args.output_dir:
            # Legacy command line approach
            image_paths_json = args.image_paths_json
            output_dir = args.output_dir
            use_affine = False
            blend_method = 'multiband'
        else:
            print("Usage: wsl_colmap_bridge.py --config <config.json>", file=sys.stderr)
            print("   or: wsl_colmap_bridge.py <image_paths_json> <output_dir>", file=sys.stderr)
            sys.exit(1)

        print("[DEBUG] Starting run_2d_stitch...", file=sys.stderr, flush=True)
        result = run_2d_stitch(image_paths_json, output_dir, use_affine, blend_method)
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

