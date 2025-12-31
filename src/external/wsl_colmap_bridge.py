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


def run_2d_stitch(image_paths_json: str, output_dir: str, use_affine: bool = False, blend_method: str = "multiband",
                  matcher_type: str = "exhaustive", sequential_overlap: int = 10,
                  gpu_index: int = -1, num_threads: int = -1, max_features: int = 8192,
                  min_inliers: int = 0, max_images: int = 0, use_source_alpha: bool = False) -> dict:
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

    # CRITICAL: Clean images directory to avoid processing old images from previous runs
    if images_dir.exists():
        shutil.rmtree(images_dir)
        print(f"[DEBUG] Cleaned old images from workspace", file=sys.stderr, flush=True)

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
    
    # Feature extraction - check GPU availability
    has_cuda = pycolmap.has_cuda if hasattr(pycolmap, 'has_cuda') else False
    print(f"[DEBUG] pycolmap.has_cuda = {has_cuda}", file=sys.stderr, flush=True)

    if not has_cuda:
        print(f"[WARNING] CUDA not available in pycolmap! Feature extraction will be VERY SLOW on CPU.", file=sys.stderr, flush=True)
        print(f"[WARNING] Expected time: {n_images * 40}s+ (CPU mode)", file=sys.stderr, flush=True)
        print(f"[SUGGESTION] Install pycolmap-cuda12 in WSL for GPU acceleration", file=sys.stderr, flush=True)

    print(f"[PROGRESS] Extracting features from {n_images} images (max_features={max_features})...", file=sys.stderr, flush=True)
    extract_start = time.time()

    # Configure SIFT extraction options with aggressive optimization
    sift_options = pycolmap.SiftExtractionOptions()

    # Log DEFAULT values before modification
    print(f"[DEBUG] SIFT defaults: max_num_features={sift_options.max_num_features}, first_octave={sift_options.first_octave}, num_octaves={sift_options.num_octaves}", file=sys.stderr, flush=True)

    sift_options.max_num_features = max_features

    # Optimize extraction speed
    sift_options.first_octave = 0  # Don't upscale (2x faster)
    sift_options.num_octaves = 4   # Reduce octaves for speed
    sift_options.edge_threshold = 10.0  # Default, but explicit

    # Log CONFIGURED values after modification
    print(f"[DEBUG] SIFT configured: max_num_features={sift_options.max_num_features}, first_octave={sift_options.first_octave}, num_octaves={sift_options.num_octaves}", file=sys.stderr, flush=True)

    # Try to set num_threads (may not exist in all pycolmap versions)
    try:
        if not has_cuda:
            import os
            cpu_threads = num_threads if num_threads > 0 else os.cpu_count()
            sift_options.num_threads = cpu_threads
            print(f"[DEBUG] CPU mode: using {cpu_threads} threads", file=sys.stderr, flush=True)
        else:
            # GPU mode: single thread per GPU
            sift_options.num_threads = 1
            print(f"[DEBUG] GPU mode: using 1 thread", file=sys.stderr, flush=True)
    except AttributeError:
        # num_threads not supported in this version - skip it
        print(f"[DEBUG] num_threads not supported in this pycolmap version (GPU handles threading)", file=sys.stderr, flush=True)

    # Create extraction options
    extraction_options = pycolmap.FeatureExtractionOptions()
    extraction_options.sift = sift_options

    # Try to set num_threads on extraction_options instead (newer pycolmap API)
    if not has_cuda and num_threads > 0:
        try:
            extraction_options.num_threads = num_threads
            print(f"[DEBUG] Set num_threads={num_threads} on extraction_options", file=sys.stderr, flush=True)
        except AttributeError:
            pass  # Not supported, GPU will handle threading

    # Configure device
    if gpu_index >= 0 and has_cuda:
        device = pycolmap.Device(f"cuda:{gpu_index}")
        print(f"[DEBUG] Device: CUDA GPU {gpu_index}", file=sys.stderr, flush=True)
    elif has_cuda:
        device = pycolmap.Device.auto
        print(f"[DEBUG] Device: CUDA auto", file=sys.stderr, flush=True)
    else:
        device = pycolmap.Device.cpu
        print(f"[DEBUG] Device: CPU (slow!)", file=sys.stderr, flush=True)

    # Test GPU availability at runtime
    if has_cuda:
        try:
            import subprocess
            gpu_test = subprocess.run(['nvidia-smi', '--query-gpu=name,utilization.gpu', '--format=csv,noheader'],
                                     capture_output=True, text=True, timeout=5)
            if gpu_test.returncode == 0:
                print(f"[DEBUG] GPU status: {gpu_test.stdout.strip()}", file=sys.stderr, flush=True)
        except:
            pass

    print(f"[PROGRESS] Starting feature extraction with {device}...", file=sys.stderr, flush=True)
    print(f"[PROGRESS] For {n_images} images, expect ~{n_images * (2 if has_cuda else 40)}s", file=sys.stderr, flush=True)

    # FINAL verification before extraction call
    print(f"[DEBUG] FINAL CHECK - extraction_options.sift.max_num_features = {extraction_options.sift.max_num_features}", file=sys.stderr, flush=True)
    print(f"[DEBUG] FINAL CHECK - extraction_options.sift.first_octave = {extraction_options.sift.first_octave}", file=sys.stderr, flush=True)
    print(f"[DEBUG] FINAL CHECK - extraction_options.sift.num_octaves = {extraction_options.sift.num_octaves}", file=sys.stderr, flush=True)
    sys.stderr.flush()

    pycolmap.extract_features(
        database_path,
        images_dir,
        extraction_options=extraction_options,
        camera_mode=pycolmap.CameraMode.PER_IMAGE,
        camera_model="PINHOLE",
        device=device
    )
    extract_time = time.time() - extract_start
    avg_time = extract_time / n_images
    print(f"[PROGRESS] Feature extraction complete in {extract_time:.1f}s ({avg_time:.1f}s per image)", file=sys.stderr, flush=True)

    if avg_time > 10.0:
        print(f"[WARNING] Extraction is very slow ({avg_time:.1f}s/image). GPU may not be working.", file=sys.stderr, flush=True)

    # Feature matching with GPU
    print(f"[PROGRESS] Matching features (GPU) using {matcher_type} matching...", file=sys.stderr, flush=True)
    match_start = time.time()

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
        n_pairs = n_images * (n_images - 1) // 2
        print(f"[PROGRESS] Exhaustive matching - {n_pairs} image pairs...", file=sys.stderr, flush=True)
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
    if use_source_alpha:
        print(f"[PROGRESS] Using source alpha channels (for transparent backgrounds)", file=sys.stderr, flush=True)
    images = []
    source_alphas = []  # Store source alpha channels for transparent background images
    for i, path in enumerate(image_paths):
        if use_source_alpha:
            # Load with alpha channel preserved
            img_with_alpha = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if img_with_alpha is None:
                return {"success": False, "error": f"Failed to load: {path}"}

            if img_with_alpha.shape[2] == 4:
                # Has alpha channel - extract it
                img = img_with_alpha[:, :, :3]  # BGR
                alpha = img_with_alpha[:, :, 3]  # Alpha
                source_alphas.append(alpha)
            else:
                # No alpha channel - use all-white
                img = img_with_alpha
                source_alphas.append(None)
        else:
            img = cv2.imread(str(path))
            if img is None:
                return {"success": False, "error": f"Failed to load: {path}"}
            source_alphas.append(None)

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

    # Apply image filtering if requested
    if min_inliers > 0 or max_images > 0:
        images, homographies = filter_images(images, homographies, min_inliers, max_images)
        print(f"[PROGRESS] After filtering: {len(images)} images, {len(homographies)} {transform_type}", file=sys.stderr, flush=True)

        if not homographies:
            return {"success": False, "error": "No images passed filtering criteria"}

    # Stitch
    print("[PROGRESS] Stitching panorama (warping and blending)...", file=sys.stderr, flush=True)
    stitch_start = time.time()
    panorama = stitch_with_homographies(images, homographies, use_affine, blend_method, source_alphas)
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


def filter_images(images: list, homographies: dict, min_inliers: int = 0, max_images: int = 0):
    """
    Filter images based on quality metrics to reduce overlap blur.

    Args:
        images: List of images
        homographies: Dict of (idx1, idx2) -> {H, n_inliers, n_matches}
        min_inliers: Minimum inliers required (0 = no filter)
        max_images: Maximum images to keep (0 = no limit)

    Returns:
        Filtered (images, homographies)
    """
    import numpy as np

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
    if max_images > 0 and len(images) > max_images:
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

        # Filter images list
        filtered_images = [img for idx, img in enumerate(images) if idx in indices_to_keep]

        # Filter homographies to only include kept images
        final_homographies = {}
        for (i, j), data in filtered_homographies.items():
            if i in indices_to_keep and j in indices_to_keep:
                # Remap indices
                new_i = sorted(indices_to_keep).index(i)
                new_j = sorted(indices_to_keep).index(j)
                final_homographies[(new_i, new_j)] = data

        return filtered_images, final_homographies

    return images, filtered_homographies


def compute_homographies(matches_data: dict, images: list, image_name_to_index: dict, use_affine: bool = False) -> dict:
    """Compute homographies from matches."""
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


def stitch_with_homographies(images: list, homographies: dict, use_affine: bool = False, blend_method: str = "multiband", source_alphas: list = None):
    """Stitch images using transforms (homography or affine) - optimized for ordered sequential captures.

    Args:
        source_alphas: Optional list of alpha channels for each image (for transparent backgrounds)
    """
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
    MIN_INLIERS = 100
    weak_edges_filtered = 0

    for (i, j), data in homographies.items():
        # Only include edges where both nodes are in largest component
        if i not in largest_component or j not in largest_component:
            print(f"[DEBUG] H3: Skipping edge ({i},{j}) - outside largest component", file=sys.stderr, flush=True)
            continue

        # Filter out weak edges - they produce unreliable transforms
        n_inliers = data['n_inliers']
        if n_inliers < MIN_INLIERS:
            weak_edges_filtered += 1
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

    print(f"[DEBUG] H3: Using {len(filtered_homographies)} edges (filtered {weak_edges_filtered} weak edges < {MIN_INLIERS} inliers)", file=sys.stderr, flush=True)
    
    # Compute cumulative homographies using BFS with priority (Dijkstra-like)
    import heapq
    H_to_ref = {ref_idx: np.eye(3)}
    discovered = {ref_idx}  # Nodes we've seen (added to queue)
    processed = set()  # Nodes whose neighbors we've expanded
    parent_of = {ref_idx: None}  # Track which image each was reached from
    # Priority queue: (negative_weight, distance_from_ref, current_idx)
    pq = [(0, 0, ref_idx)]

    # DIAGNOSTIC: Log the BFS chain to file
    bfs_log_path = "/tmp/colmap_bfs_chain.log"
    with open(bfs_log_path, 'w') as f:
        f.write(f"Reference image: {ref_idx}\n")
        f.write(f"Adjacency list:\n")
        for node, neighbors in sorted(adjacency.items()):
            neighbor_strs = [f"{n}(w={w})" for n, _, w in neighbors]
            f.write(f"  Image {node}: connects to {neighbor_strs}\n")
        f.write(f"\nBFS traversal:\n")

    print(f"[PROGRESS] Building transform chain from reference image {ref_idx}...", file=sys.stderr, flush=True)
    last_progress_pct = 0

    while pq:
        neg_weight, dist, current = heapq.heappop(pq)

        if current in processed:
            # Already expanded this node's neighbors
            continue
        processed.add(current)

        # Progress update
        progress_pct = int((len(discovered) / n_images) * 100)
        if progress_pct >= last_progress_pct + 10:
            print(f"[PROGRESS] Transform chain: {len(discovered)}/{n_images} images connected ({progress_pct}%)", file=sys.stderr, flush=True)
            last_progress_pct = progress_pct

        H_current = H_to_ref[current]

        for neighbor, H_edge, weight in adjacency.get(current, []):
            if neighbor not in discovered:
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
                discovered.add(neighbor)
                parent_of[neighbor] = current
                # Cap weight and convert to Python int to prevent numpy overflow when negating
                capped_weight = int(min(weight, 100000))
                heapq.heappush(pq, (-capped_weight, dist + 1, neighbor))

                # Log this connection
                with open(bfs_log_path, 'a') as f:
                    # Show the translation component of the final transform
                    tx, ty = H_neighbor[0, 2], H_neighbor[1, 2]
                    f.write(f"  Image {neighbor} reached via Image {current} (weight={weight}), translation=({tx:.1f}, {ty:.1f})\n")
    
    # Log final transform summary
    with open(bfs_log_path, 'a') as f:
        f.write(f"\nFinal transforms (translation component only):\n")
        for idx in sorted(H_to_ref.keys()):
            H = H_to_ref[idx]
            tx, ty = H[0, 2], H[1, 2]
            parent = parent_of.get(idx, "N/A")
            f.write(f"  Image {idx}: tx={tx:.1f}, ty={ty:.1f} (via {parent})\n")

    print(f"[PROGRESS] BFS chain diagnostics written to: {bfs_log_path}", file=sys.stderr, flush=True)

    # #region agent log - H3: Check discovered images
    print(f"[DEBUG] H3: discovered={sorted(discovered)}, H_to_ref_keys={sorted(H_to_ref.keys())}, n_images={n_images}", file=sys.stderr, flush=True)

    # Log which images are NOT included
    all_indices = set(range(n_images))
    missing_indices = all_indices - discovered

    if missing_indices:
        print(f"\n{'='*70}", file=sys.stderr, flush=True)
        print(f"[WARNING] *** {len(missing_indices)} of {n_images} images EXCLUDED ***", file=sys.stderr, flush=True)
        print(f"[WARNING] Excluded image indices: {sorted(missing_indices)}", file=sys.stderr, flush=True)
        print(f"[WARNING] Included image indices: {sorted(discovered)} ({len(discovered)} images)", file=sys.stderr, flush=True)
        print(f"{'='*70}\n", file=sys.stderr, flush=True)

        # Categorize missing images by why they were excluded
        for idx in sorted(missing_indices):
            if idx not in connections:
                print(f"[WARNING] Image {idx}: ISOLATED - no feature matches with any other image", file=sys.stderr, flush=True)
            else:
                # Check which component this image is in
                in_component = None
                for comp_idx, comp in enumerate(components):
                    if idx in comp:
                        in_component = comp_idx
                        break

                if in_component is not None and in_component > 0:
                    # In a smaller disconnected component
                    connected_to = [j for (i, j) in homographies.keys() if i == idx] + [i for (i, j) in homographies.keys() if j == idx]
                    print(f"[WARNING] Image {idx}: In disconnected component #{in_component} ({len(components[in_component])} images: {sorted(components[in_component])}), connects to {connected_to}", file=sys.stderr, flush=True)
                else:
                    # Has connections but wasn't reached (shouldn't happen with component filtering)
                    connected_to = [j for (i, j) in homographies.keys() if i == idx] + [i for (i, j) in homographies.keys() if j == idx]
                    print(f"[WARNING] Image {idx}: has {connections[idx]} connections to {connected_to}, but wasn't reached", file=sys.stderr, flush=True)
    else:
        print(f"[DEBUG] H3: All images in largest component successfully connected! ({len(discovered)}/{n_images} total images)", file=sys.stderr, flush=True)
    # #endregion
    
    # Compute canvas size
    corners_all = []
    try:
        for idx, H in H_to_ref.items():
            h, w = images[idx].shape[:2]
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

    # Prepare aligned images for ImageBlender
    # MEMORY OPTIMIZATION: Warp directly to valid region instead of full canvas
    # This reduces memory from O(n * canvas_size) to O(n * avg_image_size)
    total_to_warp = len(H_to_ref)
    print(f"[PROGRESS] Warping {total_to_warp} images (memory-optimized)...", file=sys.stderr, flush=True)
    aligned_images = []

    for warp_idx, (idx, H) in enumerate(H_to_ref.items()):
        # Progress update every 10% or every 10 images
        if warp_idx % max(1, total_to_warp // 10) == 0:
            pct = (warp_idx / total_to_warp) * 100
            print(f"[PROGRESS] Warping: {warp_idx}/{total_to_warp} images ({pct:.0f}%)", file=sys.stderr, flush=True)

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

        # Fix: use [:, 0, :] to get all 4 corners (not [0, :, :] which is only first corner)
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
            print(f"[DEBUG] H5: Image {idx} has invalid region, skipping", file=sys.stderr, flush=True)
            continue

        # Create alpha channel BEFORE warping
        # If source_alphas provided and this image has alpha, use it (for transparent backgrounds)
        # Otherwise, all original pixels are valid
        if source_alphas and idx < len(source_alphas) and source_alphas[idx] is not None:
            alpha_original = source_alphas[idx]
            print(f"[DEBUG] H5: Image {idx} using source alpha (transparent background)", file=sys.stderr, flush=True)
        else:
            alpha_original = np.ones((h, w), dtype=np.uint8) * 255

        # MEMORY OPTIMIZATION: Warp directly to the valid region only
        # Adjust transform to account for the crop offset
        T_offset = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float64)
        H_cropped = T_offset @ H_final

        if use_affine:
            M_affine = H_cropped[:2, :]
            warped = cv2.warpAffine(img, M_affine, (region_w, region_h))
            alpha = cv2.warpAffine(alpha_original, M_affine, (region_w, region_h))
        else:
            warped = cv2.warpPerspective(img, H_cropped, (region_w, region_h))
            alpha = cv2.warpPerspective(alpha_original, H_cropped, (region_w, region_h))

        # Now alpha correctly marks transformed content (255) vs warping borders (0)
        # bbox stores the position in the final canvas
        aligned_images.append({
            'image': warped,
            'alpha': alpha,
            'bbox': (x_min, y_min, x_max, y_max),
            'transform': H_final,
            'warped': True
        })

        print(f"[DEBUG] H5: Image {idx} warped to {region_w}x{region_h}, bbox: ({x_min},{y_min})-({x_max},{y_max})", file=sys.stderr, flush=True)
        print(f"[DEBUG] H6: Prepared image {idx} for blending ({len(aligned_images)}/{len(H_to_ref)})", file=sys.stderr, flush=True)

    print(f"[PROGRESS] Warping complete: {len(aligned_images)} images warped to {output_w}x{output_h} canvas", file=sys.stderr, flush=True)

    # Use ImageBlender to blend
    print(f"[PROGRESS] Blending {len(aligned_images)} images using {blend_method} method...", file=sys.stderr, flush=True)
    print(f"[DEBUG] H7: Blending {len(aligned_images)} images using {blend_method} method...", file=sys.stderr, flush=True)
    sys.stderr.flush()

    # Try to import ImageBlender - if it fails, fall back to simple blending
    try:
        # Import ImageBlender - need to add parent directory to path for imports
        script_dir = Path(__file__).resolve().parent.parent
        print(f"[PROGRESS] Importing ImageBlender...", file=sys.stderr, flush=True)
        print(f"[DEBUG] H7: script_dir={script_dir}", file=sys.stderr, flush=True)
        print(f"[DEBUG] H7: sys.path before={sys.path[:3]}", file=sys.stderr, flush=True)

        if str(script_dir) not in sys.path:
            sys.path.insert(0, str(script_dir))

        print(f"[DEBUG] H7: Attempting to import ImageBlender from core.blender...", file=sys.stderr, flush=True)
        from core.blender import ImageBlender
        print(f"[PROGRESS] ImageBlender imported successfully", file=sys.stderr, flush=True)
        print(f"[DEBUG] H7: ImageBlender imported successfully", file=sys.stderr, flush=True)

        blender = ImageBlender(
            method=blend_method,
            options={
                'hdr_mode': False,
                'anti_ghosting': False
            }
        )

        # Images are already warped to their valid regions with proper bboxes
        # Just filter out any empty images and log debug info
        valid_region_images = []
        total_aligned = len(aligned_images)
        print(f"[PROGRESS] Validating {total_aligned} warped images...", file=sys.stderr, flush=True)

        # DIAGNOSTIC: Write bbox info to file
        debug_log_path = "/tmp/colmap_extraction_debug.log"
        with open(debug_log_path, 'w') as debug_f:
            debug_f.write(f"Canvas size: {output_w}x{output_h}\n")
            debug_f.write(f"Processing {len(aligned_images)} aligned images...\n\n")

        for idx, aligned_img in enumerate(aligned_images):
            img = aligned_img['image']
            alpha = aligned_img['alpha']
            bbox = aligned_img['bbox']

            # Check if image has valid content
            if np.count_nonzero(alpha) == 0:
                print(f"[DEBUG] H7: Image {idx} has no valid pixels, skipping", file=sys.stderr, flush=True)
                with open(debug_log_path, 'a') as debug_f:
                    debug_f.write(f"Image {idx}: NO VALID PIXELS\n")
                continue

            # Log bbox info
            with open(debug_log_path, 'a') as debug_f:
                debug_f.write(f"Image {idx}: bbox={bbox}, size={img.shape[1]}x{img.shape[0]}\n")

            valid_region_images.append(aligned_img)

        # Log bbox summary
        if valid_region_images:
            bboxes = [img['bbox'] for img in valid_region_images]
            x_mins = [b[0] for b in bboxes]
            y_mins = [b[1] for b in bboxes]

            debug_file = "/tmp/colmap_bbox_debug.txt"
            with open(debug_file, 'w') as f:
                f.write(f"Number of images: {len(valid_region_images)}\n")
                f.write(f"Bbox X range: {min(x_mins)} to {max(x_mins)}\n")
                f.write(f"Bbox Y range: {min(y_mins)} to {max(y_mins)}\n\n")
                f.write("Individual bboxes:\n")
                for i, bbox in enumerate(bboxes):
                    f.write(f"  Image {i}: {bbox}\n")

            print(f"[PROGRESS] {len(valid_region_images)} valid images, Bbox X: {min(x_mins)}-{max(x_mins)}, Y: {min(y_mins)}-{max(y_mins)}", file=sys.stderr, flush=True)

        # Final diagnostic message with file location
        print(f"[PROGRESS] Extraction diagnostics written to: {debug_log_path}", file=sys.stderr, flush=True)
        print(f"[PROGRESS] Calling ImageBlender with {len(valid_region_images)} cropped images...", file=sys.stderr, flush=True)
        print(f"[DEBUG] H7: Calling blender.blend() with {len(valid_region_images)} cropped images...", file=sys.stderr, flush=True)

        # fit_all=False because we want to preserve the exact canvas size we calculated
        panorama = blender.blend(valid_region_images, padding=0, fit_all=False)
        print(f"[PROGRESS] ImageBlender completed! Panorama size: {panorama.shape[1]}x{panorama.shape[0]}", file=sys.stderr, flush=True)
        print(f"[DEBUG] H7: ImageBlender.blend() completed successfully, panorama.shape={panorama.shape}", file=sys.stderr, flush=True)

    except Exception as e:
        # Fall back to simple weighted averaging if ImageBlender fails
        print(f"[WARNING] ImageBlender failed: {e} - using fallback blending", file=sys.stderr, flush=True)
        print(f"[DEBUG] H7: ImageBlender failed ({e}), falling back to simple blending", file=sys.stderr, flush=True)
        import traceback
        print(f"[DEBUG] H7: Traceback: {traceback.format_exc()}", file=sys.stderr, flush=True)

        # Simple weighted average blending (fallback)
        panorama = np.zeros((output_h, output_w, 3), dtype=np.float32)
        weight_sum = np.zeros((output_h, output_w), dtype=np.float32)

        for aligned in aligned_images:
            warped = aligned['image'].astype(np.float32)

            # Use the warped alpha channel instead of detecting black pixels
            # This preserves legitimate black content in the original images
            if 'alpha' in aligned:
                mask = (aligned['alpha'] / 255.0).astype(np.float32)
            else:
                # Fallback if no alpha (shouldn't happen with our code)
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
        print(f"[DEBUG] H7: Simple blending completed (fallback)", file=sys.stderr, flush=True)

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
        else:
            print("Usage: wsl_colmap_bridge.py --config <config.json>", file=sys.stderr)
            print("   or: wsl_colmap_bridge.py <image_paths_json> <output_dir>", file=sys.stderr)
            sys.exit(1)

        print("[DEBUG] Starting run_2d_stitch...", file=sys.stderr, flush=True)
        print(f"[DEBUG] Parameters: matcher={matcher_type}, features={max_features}, threads={num_threads}, min_inliers={min_inliers}, max_images={max_images}, use_source_alpha={use_source_alpha}", file=sys.stderr, flush=True)
        result = run_2d_stitch(image_paths_json, output_dir, use_affine, blend_method,
                              matcher_type, sequential_overlap, gpu_index, num_threads, max_features,
                              min_inliers, max_images, use_source_alpha)
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

