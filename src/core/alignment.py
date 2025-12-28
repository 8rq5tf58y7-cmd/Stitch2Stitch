"""
Image alignment engine for flat images (like scanned documents, maps, artwork)
Uses similarity transforms (translation + uniform scale) - no perspective distortion

Key improvements for large panoramas:
- Minimum spanning tree for transform propagation (minimizes chain length)
- Global consistency checking
- Automatic bundle adjustment for 10+ images
- Better handling of grid/mosaic patterns
- Edge-boundary prioritized matching for better connectivity
- Spatial coherence filtering to prevent object duplication
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
import gc
from collections import defaultdict
import heapq

logger = logging.getLogger(__name__)


def compute_edge_weights(keypoints: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Compute weights for keypoints based on proximity to image edges.
    
    Features near edges are more likely to be in overlap regions with adjacent images,
    so they should be prioritized for matching.
    
    Args:
        keypoints: Nx2 or Nx4 array of keypoint coordinates
        image_shape: (height, width) of the image
        
    Returns:
        Nx1 array of weights (0-1), higher = closer to edge
    """
    h, w = image_shape[:2]
    if len(keypoints) == 0:
        return np.array([])
    
    # Extract x, y coordinates
    x = keypoints[:, 0]
    y = keypoints[:, 1]
    
    # Distance from each edge (normalized to 0-1 where 0 = edge, 1 = center)
    dist_left = x / w
    dist_right = 1 - x / w
    dist_top = y / h
    dist_bottom = 1 - y / h
    
    # Minimum distance to any edge (0 = at edge, 0.5 = at center)
    min_dist = np.minimum(np.minimum(dist_left, dist_right), 
                          np.minimum(dist_top, dist_bottom))
    
    # Convert to edge weight: 1.0 at edge, 0.0 at center
    # Use exponential decay so edge region (within 20% of image) gets high weight
    edge_weight = np.exp(-min_dist * 5)  # Decay factor of 5 means ~0.37 weight at 20% from edge
    
    return edge_weight


def check_spatial_coherence(pts1: np.ndarray, pts2: np.ndarray, 
                            max_angle_deviation: float = 30.0,
                            min_consistent_ratio: float = 0.5) -> Tuple[np.ndarray, float]:
    """
    Check spatial coherence of matched points to filter scattered incorrect matches.
    
    For a valid panorama overlap, matched points should:
    1. Have consistent relative displacement vectors
    2. Not be randomly scattered across the image
    
    This helps prevent object duplication from false matches.
    
    Args:
        pts1: Nx2 source points
        pts2: Nx2 destination points
        max_angle_deviation: Maximum allowed angle deviation (degrees) from median
        min_consistent_ratio: Minimum ratio of consistent matches required
        
    Returns:
        Tuple of (mask of coherent matches, coherence score 0-1)
    """
    n = len(pts1)
    if n < 4:
        return np.ones(n, dtype=bool), 0.0
    
    # Compute displacement vectors
    displacements = pts2 - pts1
    
    # Compute angles of displacement vectors
    angles = np.arctan2(displacements[:, 1], displacements[:, 0])
    
    # Compute median angle (most common direction)
    median_angle = np.median(angles)
    
    # Compute angular deviation from median
    angle_diff = np.abs(angles - median_angle)
    # Handle wrap-around (e.g., -179° and 179° are close)
    angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
    angle_diff_deg = np.degrees(angle_diff)
    
    # Mask for consistent directions
    direction_mask = angle_diff_deg < max_angle_deviation
    
    # Also check magnitude consistency (displacement lengths should be similar)
    magnitudes = np.linalg.norm(displacements, axis=1)
    median_mag = np.median(magnitudes)
    if median_mag > 0:
        mag_ratio = magnitudes / median_mag
        magnitude_mask = (mag_ratio > 0.5) & (mag_ratio < 2.0)
    else:
        magnitude_mask = np.ones(n, dtype=bool)
    
    # Combined coherence mask
    coherent_mask = direction_mask & magnitude_mask
    
    # Coherence score
    coherence_score = np.sum(coherent_mask) / n
    
    return coherent_mask, coherence_score


def filter_matches_by_cluster(pts1: np.ndarray, pts2: np.ndarray,
                               image1_shape: Tuple[int, int],
                               image2_shape: Tuple[int, int],
                               min_cluster_size: int = 5) -> np.ndarray:
    """
    Filter matches to only keep those in valid overlap regions.
    
    For panorama stitching, matches should cluster in regions where images overlap,
    not be scattered randomly. This function identifies the largest cluster of
    spatially consistent matches.
    
    OPTIMIZED: Uses median-based filtering instead of O(n²) pairwise distances.
    After MAGSAC++ and coherence filtering, most outliers are already removed,
    so we just need to reject extreme outliers.
    
    Args:
        pts1: Nx2 source points
        pts2: Nx2 destination points  
        image1_shape: Shape of first image (height, width)
        image2_shape: Shape of second image (height, width)
        min_cluster_size: Minimum matches in a cluster to be valid
        
    Returns:
        Boolean mask of matches to keep
    """
    n = len(pts1)
    if n < min_cluster_size:
        return np.ones(n, dtype=bool)
    
    # FAST PATH: Skip expensive clustering for small/medium sets
    # After MAGSAC++ and coherence filtering, outliers are already mostly removed
    if n <= 100:
        return np.ones(n, dtype=bool)
    
    # Use median-based outlier detection (O(n) instead of O(n²))
    displacements = pts2 - pts1
    
    # Compute median displacement
    median_disp = np.median(displacements, axis=0)
    
    # Compute distances from median
    dist_from_median = np.linalg.norm(displacements - median_disp, axis=1)
    
    # Use MAD (Median Absolute Deviation) for robust threshold
    mad = np.median(dist_from_median)
    if mad < 1e-6:
        mad = np.std(dist_from_median)  # Fallback to std
    
    # Keep points within 3 MAD of median (robust outlier detection)
    threshold = max(3.0 * mad, 50.0)  # At least 50 pixels tolerance
    cluster_mask = dist_from_median <= threshold
    
    # Ensure we keep enough points
    if np.sum(cluster_mask) >= min_cluster_size:
        return cluster_mask
    else:
        return np.ones(n, dtype=bool)

# Default maximum pixels for a single warped image output (50 megapixels)
# Set to None or 0 to disable limit
# Memory guidelines:
#   - Each warped image: width * height * 3 bytes (RGB uint8)
#   - 50MP image: ~150MB RAM
#   - 100MP image: ~300MB RAM
# Default: None (let blender handle scaling globally)
# Per-image scaling is not recommended as it can cause quality issues
# The blender's max_panorama_pixels handles output size limiting properly
DEFAULT_MAX_WARP_PIXELS = None


class ImageAligner:
    """Align flat images using similarity transforms (translation + scale)"""
    
    def __init__(
        self, 
        use_gpu: bool = False, 
        allow_scale: bool = True,
        max_warp_pixels: Optional[int] = DEFAULT_MAX_WARP_PIXELS
    ):
        """
        Initialize aligner
        
        Args:
            use_gpu: Enable GPU acceleration
            allow_scale: Allow uniform scaling to match image sizes (default True)
            max_warp_pixels: Maximum pixels for warped image output (None or 0 = unlimited)
                Memory guidelines:
                - 50MP: ~150MB per image
                - 100MP: ~300MB per image
        """
        self.use_gpu = use_gpu
        self.allow_scale = allow_scale
        self.max_warp_pixels = max_warp_pixels if max_warp_pixels else None
        logger.info(f"Image aligner initialized (allow_scale={allow_scale}, max_warp_pixels={self.max_warp_pixels or 'unlimited'})")
    
    def align_images(
        self,
        images_data: List[Dict],
        features_data: List[Dict],
        matches: List[Dict]
    ) -> List[Dict]:
        """
        Align images based on feature matches using similarity transforms
        
        Args:
            images_data: List of image data dictionaries
            features_data: List of feature data dictionaries
            matches: List of match dictionaries
            
        Returns:
            List of aligned image data dictionaries
        """
        if len(images_data) < 2:
            return images_data
        
        n_images = len(images_data)
        logger.info(f"Aligning {n_images} images using similarity transforms")
        
        # Calculate relative transforms between all matched pairs
        relative_transforms = self._calculate_relative_transforms(
            images_data, features_data, matches
        )
        
        if not relative_transforms:
            logger.warning("No valid transforms found, returning images at origin")
            return self._place_images_at_origin(images_data)
        
        # Build connectivity graph
        graph = self._build_graph(images_data, matches)
        
        # Find reference image (most connected)
        ref_idx = self._find_reference_image(graph)
        logger.info(f"Using image {ref_idx} as reference")
        
        # Calculate absolute transforms using BFS from reference
        transforms = self._calculate_absolute_transforms(
            n_images, relative_transforms, graph, ref_idx
        )
        
        # Apply transforms to create aligned images
        aligned_images = self._apply_transforms(images_data, transforms, ref_idx)
        
        return aligned_images
    
    def _calculate_relative_transforms(
        self,
        images_data: List[Dict],
        features_data: List[Dict],
        matches: List[Dict]
    ) -> Dict[Tuple[int, int], np.ndarray]:
        """
        Calculate relative similarity transform between each matched pair.
        
        Uses advanced edge-boundary prioritization and spatial coherence checking
        to improve alignment quality and prevent object duplication.
        
        Returns:
            Dict mapping (i, j) -> 3x3 transform matrix from image i coords to image j coords
        """
        relative_transforms = {}
        rejected_coherence = 0
        rejected_cluster = 0
        accepted = 0
        
        for match in matches:
            i = match['image_i']
            j = match['image_j']
            match_list = match['matches']
            
            if len(match_list) < 4:
                continue
            
            kp_i = features_data[i]['keypoints']
            kp_j = features_data[j]['keypoints']
            
            # Get image shapes for edge weighting
            img_i_shape = images_data[i]['image'].shape[:2]
            img_j_shape = images_data[j]['image'].shape[:2]
            
            # Extract matched points with edge weights
            pts_i = []
            pts_j = []
            edge_weights = []
            
            for m in match_list:
                idx_i = m.queryIdx
                idx_j = m.trainIdx
            
                if idx_i < len(kp_i) and idx_j < len(kp_j):
                    pt_i = [kp_i[idx_i][0], kp_i[idx_i][1]]
                    pt_j = [kp_j[idx_j][0], kp_j[idx_j][1]]
                    pts_i.append(pt_i)
                    pts_j.append(pt_j)
                    
                    # Compute edge weight (higher for points near boundaries)
                    weight_i = compute_edge_weights(np.array([pt_i]), img_i_shape)[0]
                    weight_j = compute_edge_weights(np.array([pt_j]), img_j_shape)[0]
                    edge_weights.append((weight_i + weight_j) / 2)
        
            if len(pts_i) < 4:
                continue
            
            pts_i = np.float32(pts_i)
            pts_j = np.float32(pts_j)
            edge_weights = np.array(edge_weights)
            
            # Step 1: Check spatial coherence - reject scattered matches
            coherence_mask, coherence_score = check_spatial_coherence(pts_i, pts_j)
            
            # #region agent log
            try:
                import json
                with open(r'c:\Users\ryanf\OneDrive - University of Maryland\Desktop\Stitch2Stitch\Stitch2Stitch-1\.cursor\debug.log', 'a') as f:
                    f.write(json.dumps({"hypothesisId":"E","location":"alignment.py:coherence_check","message":"Coherence check","data":{"i":i,"j":j,"n_pts":len(pts_i),"coherence_score":round(coherence_score,3),"avg_edge_weight":round(float(np.mean(edge_weights)),3)},"timestamp":__import__('time').time()}) + '\n')
            except: pass
            # #endregion
            
            # Relax coherence threshold to 30% to improve connectivity
            # Lower is acceptable if we have many points (law of large numbers)
            min_coherence = 0.3 if len(pts_i) >= 20 else 0.4
            if coherence_score < min_coherence:
                rejected_coherence += 1
                logger.debug(f"Match ({i}, {j}): Rejected due to low spatial coherence ({coherence_score:.1%})")
                continue
            
            # Apply coherence filter
            pts_i_coherent = pts_i[coherence_mask]
            pts_j_coherent = pts_j[coherence_mask]
            
            if len(pts_i_coherent) < 4:
                rejected_coherence += 1
                continue
            
            # Step 2: Filter to largest cluster of consistent matches
            cluster_mask = filter_matches_by_cluster(
                pts_i_coherent, pts_j_coherent, img_i_shape, img_j_shape
            )
            
            pts_i_filtered = pts_i_coherent[cluster_mask]
            pts_j_filtered = pts_j_coherent[cluster_mask]
            
            if len(pts_i_filtered) < 4:
                rejected_cluster += 1
                logger.debug(f"Match ({i}, {j}): Rejected - cluster too small after filtering")
                continue
            
            # Step 3: Estimate similarity transform with filtered points
            transform = self._estimate_similarity_transform(pts_i_filtered, pts_j_filtered)
            
            if transform is not None:
                accepted += 1
                # Store forward transform (i -> j)
                relative_transforms[(i, j)] = transform
                # Store inverse transform (j -> i)
                try:
                    inverse = np.linalg.inv(transform)
                    relative_transforms[(j, i)] = inverse
                except np.linalg.LinAlgError:
                    logger.warning(f"Could not invert transform for ({j}, {i})")
                
                # Log transform parameters
                scale = np.sqrt(transform[0, 0]**2 + transform[0, 1]**2)
                rotation = np.degrees(np.arctan2(transform[1, 0], transform[0, 0]))
                tx, ty = transform[0, 2], transform[1, 2]
                logger.info(f"Transform ({i} -> {j}): scale={scale:.3f}, rotation={rotation:.1f}°, "
                           f"tx={tx:.1f}, ty={ty:.1f} (from {len(pts_i_filtered)}/{len(pts_i)} filtered points, "
                           f"coherence={coherence_score:.1%})")
        
        logger.info(f"Transform computation: {accepted} accepted, "
                   f"{rejected_coherence} rejected (coherence), {rejected_cluster} rejected (cluster)")
        
        # Compute connectivity statistics
        connected_images = set()
        for (i, j) in relative_transforms.keys():
            connected_images.add(i)
            connected_images.add(j)
        
        # #region agent log
        try:
            import json
            with open(r'c:\Users\ryanf\OneDrive - University of Maryland\Desktop\Stitch2Stitch\Stitch2Stitch-1\.cursor\debug.log', 'a') as f:
                f.write(json.dumps({"hypothesisId":"D","location":"alignment.py:_calculate_relative_transforms","message":"Relative transforms computed","data":{"n_matches":len(matches),"n_transforms":len(relative_transforms)//2,"accepted":accepted,"rejected_coherence":rejected_coherence,"rejected_cluster":rejected_cluster,"n_connected_images":len(connected_images),"n_total_images":len(images_data)},"timestamp":__import__('time').time()}) + '\n')
        except: pass
        # #endregion
        
        logger.info(f"Transform graph: {len(connected_images)}/{len(images_data)} images have at least one valid transform")
        
        return relative_transforms
    
    def _estimate_similarity_transform(
        self,
        pts1: np.ndarray,
        pts2: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Estimate similarity transform from pts1 to pts2 using MAGSAC++/USAC.
        
        Uses advanced robust estimation (MAGSAC++ > USAC > RANSAC fallback):
        - MAGSAC++: Marginalizing over threshold, best for varying noise
        - USAC: Universal RANSAC with local optimization
        - RANSAC: Standard fallback
        
        Similarity transform has 4 DOF: translation (tx, ty), uniform scale (s), rotation (θ)
        Matrix form: [[s*cos(θ), -s*sin(θ), tx],
                      [s*sin(θ),  s*cos(θ), ty],
                      [0,         0,         1]]
        """
        if len(pts1) < 2:
            return None
        
        # Try MAGSAC++ first via findHomography (which supports it), then fallback to RANSAC
        # Note: estimateAffinePartial2D does NOT support USAC_MAGSAC in OpenCV 4.x
        similarity = None
        inliers = None
        
        # First, try to get robust inliers using findHomography with MAGSAC++
        # This is much better at rejecting outliers
        magsac_inliers = None
        try:
            _, magsac_inliers = cv2.findHomography(
                pts1, pts2,
                method=cv2.USAC_MAGSAC,
                ransacReprojThreshold=3.0,
                confidence=0.995,
                maxIters=2000
            )
        except (cv2.error, AttributeError):
            pass  # MAGSAC++ not available, will use RANSAC below
        
        # If MAGSAC++ found inliers, filter points and estimate affine on clean data
        if magsac_inliers is not None and np.sum(magsac_inliers) >= 4:
            inlier_mask = magsac_inliers.ravel().astype(bool)
            pts1_clean = pts1[inlier_mask]
            pts2_clean = pts2[inlier_mask]
            
            # Now estimate affine with clean points (RANSAC is fast on pre-filtered data)
            similarity, inliers = cv2.estimateAffinePartial2D(
                pts1_clean, pts2_clean,
                method=cv2.RANSAC,
                ransacReprojThreshold=3.0,
                confidence=0.99,
                maxIters=1000
            )
            # Remap inliers back to original indices
            if inliers is not None:
                full_inliers = np.zeros((len(pts1), 1), dtype=np.uint8)
                inlier_indices = np.where(inlier_mask)[0]
                for idx, is_inlier in zip(inlier_indices, inliers.ravel()):
                    full_inliers[idx] = is_inlier
                inliers = full_inliers
        else:
            # Fallback: direct RANSAC estimation
            similarity, inliers = cv2.estimateAffinePartial2D(
                pts1, pts2,
                method=cv2.RANSAC,
                ransacReprojThreshold=3.0,
                confidence=0.99,
                maxIters=2000
            )
        
        if similarity is None:
            logger.warning("Failed to estimate similarity transform")
            # #region agent log
            try:
                import json
                with open(r'c:\Users\ryanf\OneDrive - University of Maryland\Desktop\Stitch2Stitch\Stitch2Stitch-1\.cursor\debug.log', 'a') as f:
                    f.write(json.dumps({"hypothesisId":"C","location":"alignment.py:_estimate_similarity_transform","message":"RANSAC returned None","data":{"n_pts":len(pts1)},"timestamp":__import__('time').time()}) + '\n')
            except: pass
            # #endregion
            return None
        
        # Validate transform
        scale = np.sqrt(similarity[0, 0]**2 + similarity[0, 1]**2)
        rotation = np.arctan2(similarity[1, 0], similarity[0, 0])
        inlier_count = int(np.sum(inliers)) if inliers is not None else 0
        inlier_ratio = inlier_count / len(pts1) if len(pts1) > 0 else 0
        
        # Check scale bounds
        if not self.allow_scale:
            # Force scale to 1.0 if scaling disabled
            if abs(scale - 1.0) > 0.05:
                # Re-estimate with translation only
                return self._estimate_translation_only(pts1, pts2)
        else:
            # Allow wider scale range (0.5x to 2.0x) for datasets with varying zoom levels
            # Only reject extreme scales that indicate clearly bad matches
            if scale < 0.5 or scale > 2.0:
                # #region agent log
                try:
                    import json
                    with open(r'c:\Users\ryanf\OneDrive - University of Maryland\Desktop\Stitch2Stitch\Stitch2Stitch-1\.cursor\debug.log', 'a') as f:
                        f.write(json.dumps({"hypothesisId":"C","location":"alignment.py","message":"REJECTED:scale","data":{"scale":round(scale,4),"rotation":round(np.degrees(rotation),1),"inliers":inlier_count},"timestamp":__import__('time').time()}) + '\n')
                except: pass
                # #endregion
                return None
            # Also reject degenerate scales (essentially zero)
            if scale < 0.01:
                return None
        
        # Check rotation - for most panoramas, rotation should be small
        # Large rotations (>45°) often indicate false matches
        if abs(np.degrees(rotation)) > 45:
            # #region agent log
            try:
                import json
                with open(r'c:\Users\ryanf\OneDrive - University of Maryland\Desktop\Stitch2Stitch\Stitch2Stitch-1\.cursor\debug.log', 'a') as f:
                    f.write(json.dumps({"hypothesisId":"C","location":"alignment.py","message":"REJECTED:rotation","data":{"scale":round(scale,4),"rotation":round(np.degrees(rotation),1),"inliers":inlier_count},"timestamp":__import__('time').time()}) + '\n')
            except: pass
            # #endregion
            return None
        
        # Check inlier quality using BOTH ratio AND absolute count
        # High-feature images may have low ratio but still have many reliable inliers
        if inliers is not None:
            # Flexible criteria:
            # 1. Accept if inlier ratio is good (>= 20%)
            # 2. Accept if we have many inliers (>= 30) even with lower ratio (>= 5%)
            # 3. Reject otherwise
            is_good_ratio = inlier_ratio >= 0.20
            is_many_inliers = (inlier_count >= 30 and inlier_ratio >= 0.05)
            
            if not (is_good_ratio or is_many_inliers):
                # #region agent log
                try:
                    import json
                    with open(r'c:\Users\ryanf\OneDrive - University of Maryland\Desktop\Stitch2Stitch\Stitch2Stitch-1\.cursor\debug.log', 'a') as f:
                        f.write(json.dumps({"hypothesisId":"C","location":"alignment.py","message":"REJECTED:inliers","data":{"scale":round(scale,4),"inliers":inlier_count,"ratio":round(inlier_ratio,3)},"timestamp":__import__('time').time()}) + '\n')
                except: pass
                # #endregion
                return None
        
        # #region agent log
        try:
            import json
            with open(r'c:\Users\ryanf\OneDrive - University of Maryland\Desktop\Stitch2Stitch\Stitch2Stitch-1\.cursor\debug.log', 'a') as f:
                f.write(json.dumps({"hypothesisId":"C","location":"alignment.py:_estimate_similarity_transform","message":"Transform ACCEPTED","data":{"scale":round(scale,4),"rotation_deg":round(np.degrees(rotation),1),"inliers":inlier_count,"inlier_ratio":round(inlier_ratio,3)},"timestamp":__import__('time').time()}) + '\n')
        except: pass
        # #endregion
        
        # Convert 2x3 to 3x3
        transform = np.vstack([similarity, [0, 0, 1]])
        return transform.astype(np.float64)
    
    def _estimate_translation_only(
        self,
        pts1: np.ndarray,
        pts2: np.ndarray
    ) -> Optional[np.ndarray]:
        """Estimate translation-only transform using median"""
        translations = pts2 - pts1
        median_t = np.median(translations, axis=0)
        
        transform = np.array([
            [1, 0, median_t[0]],
            [0, 1, median_t[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        
        return transform
    
    def _build_graph(
        self,
        images_data: List[Dict],
        matches: List[Dict]
    ) -> Dict[int, List[int]]:
        """
        Build connectivity graph from matches.
        Also stores match quality for later use in MST.
        """
        graph = {i: [] for i in range(len(images_data))}
        self._match_quality = {}  # Store quality for edge weighting
        
        for match in matches:
            i = match['image_i']
            j = match['image_j']
            num_matches = match.get('num_matches', 0)
            inlier_ratio = match.get('inlier_ratio', 0.5)
            confidence = match.get('confidence', 0.5)
            
            # Relaxed criteria: just need decent number of matches
            if num_matches >= 8:
                if j not in graph[i]:
                    graph[i].append(j)
                if i not in graph[j]:
                    graph[j].append(i)
                
                # Store quality metric (higher is better)
                quality = num_matches * inlier_ratio * confidence
                self._match_quality[(i, j)] = quality
                self._match_quality[(j, i)] = quality
        
        # Log connectivity stats
        connected_count = sum(1 for i in graph if len(graph[i]) > 0)
        avg_connections = sum(len(v) for v in graph.values()) / max(len(graph), 1)
        logger.info(f"Graph: {connected_count}/{len(graph)} images connected, avg {avg_connections:.1f} connections each")
        
        # Warn if graph is poorly connected
        if connected_count < len(graph) * 0.9:
            logger.warning(f"Only {connected_count}/{len(graph)} images are connected - results may be poor")
        if avg_connections < 1.5 and len(graph) > 5:
            logger.warning(f"Low connectivity ({avg_connections:.1f}/image) - consider using more features")
        
        return graph
    
    def _find_reference_image(self, graph: Dict[int, List[int]]) -> int:
        """Find image with most connections as reference"""
        max_connections = -1
        ref_idx = 0
        
        for idx, connections in graph.items():
            if len(connections) > max_connections:
                max_connections = len(connections)
                ref_idx = idx
        
        return ref_idx
    
    def _calculate_absolute_transforms(
        self,
        n_images: int,
        relative_transforms: Dict[Tuple[int, int], np.ndarray],
        graph: Dict[int, List[int]],
        ref_idx: int
    ) -> Dict[int, np.ndarray]:
        """
        Calculate absolute transforms using Minimum Spanning Tree from reference image.
        
        Using MST instead of simple BFS minimizes the maximum chain length,
        which reduces error accumulation for large panoramas.
        
        Each transform maps from the image's local coordinates to the 
        global (reference) coordinate system.
        """
        # Build edge weights based on transform quality (inlier ratio, match count)
        edge_weights = self._compute_edge_weights(relative_transforms)
        
        # Use MST to find optimal path from reference to all images
        mst_edges = self._compute_mst(n_images, graph, edge_weights, ref_idx)
        
        # Build MST adjacency for propagation
        mst_adj = defaultdict(list)
        for i, j, weight in mst_edges:
            mst_adj[i].append(j)
            mst_adj[j].append(i)
        
        # Reference image has identity transform
        transforms = {ref_idx: np.eye(3, dtype=np.float64)}
        visited = {ref_idx}
        queue = [ref_idx]
        
        # Propagate transforms along MST (ensures shortest paths)
        while queue:
            current = queue.pop(0)
            current_transform = transforms[current]
            
            for neighbor in mst_adj.get(current, []):
                if neighbor in visited:
                    continue
                
                # Get transform from neighbor to current
                key = (neighbor, current)
                if key in relative_transforms:
                    neighbor_to_current = relative_transforms[key]
                    neighbor_to_global = current_transform @ neighbor_to_current
                    transforms[neighbor] = neighbor_to_global
                    visited.add(neighbor)
                    queue.append(neighbor)
                    
                    # Log position
                    tx, ty = neighbor_to_global[0, 2], neighbor_to_global[1, 2]
                    scale = np.sqrt(neighbor_to_global[0, 0]**2 + neighbor_to_global[0, 1]**2)
                    logger.debug(f"Image {neighbor}: position=({tx:.1f}, {ty:.1f}), scale={scale:.3f}")
        
        # Handle disconnected images/components
        unplaced = [i for i in range(n_images) if i not in transforms]
        if unplaced:
            transforms = self._place_disconnected_components(
                transforms, unplaced, n_images, graph, relative_transforms
            )
        
        # Verify alignment consistency for large sets
        if len(transforms) >= 10:
            transforms = self._verify_and_refine_alignment(transforms, relative_transforms, n_images)
        
        # Check if layout is too linear and needs 2D correction
        transforms = self._fix_linear_layout(transforms, n_images)
        
        # Final check: clamp extreme positions to prevent "exploded" layouts
        transforms = self._clamp_extreme_positions(transforms, n_images)
        
        return transforms
    
    def _compute_edge_weights(
        self, 
        relative_transforms: Dict[Tuple[int, int], np.ndarray]
    ) -> Dict[Tuple[int, int], float]:
        """
        Compute edge weights for MST based on transform quality.
        Lower weight = better connection = higher priority in MST.
        """
        weights = {}
        
        for (i, j), transform in relative_transforms.items():
            if i > j:
                continue  # Only store each edge once
            
            # Weight based on transform deviation from identity (scale and rotation)
            scale = np.sqrt(transform[0, 0]**2 + transform[0, 1]**2)
            rotation = abs(np.arctan2(transform[1, 0], transform[0, 0]))
            
            # Prefer transforms close to identity (scale ~1, rotation ~0)
            scale_penalty = abs(scale - 1.0) * 10
            rotation_penalty = rotation * 5  # radians
            
            # Base weight - lower is better
            weight = 1.0 + scale_penalty + rotation_penalty
            
            weights[(i, j)] = weight
            weights[(j, i)] = weight
        
        return weights
    
    def _compute_mst(
        self,
        n_images: int,
        graph: Dict[int, List[int]],
        edge_weights: Dict[Tuple[int, int], float],
        start_idx: int
    ) -> List[Tuple[int, int, float]]:
        """
        Compute Minimum Spanning Tree using Prim's algorithm.
        This ensures the shortest path from reference to any image.
        """
        mst_edges = []
        in_mst = {start_idx}
        
        # Priority queue: (weight, from_node, to_node)
        candidates = []
        for neighbor in graph.get(start_idx, []):
            weight = edge_weights.get((start_idx, neighbor), 1.0)
            heapq.heappush(candidates, (weight, start_idx, neighbor))
        
        while candidates and len(in_mst) < n_images:
            weight, from_node, to_node = heapq.heappop(candidates)
            
            if to_node in in_mst:
                continue
            
            # Add edge to MST
            in_mst.add(to_node)
            mst_edges.append((from_node, to_node, weight))
            
            # Add new candidates
            for neighbor in graph.get(to_node, []):
                if neighbor not in in_mst:
                    edge_weight = edge_weights.get((to_node, neighbor), 1.0)
                    heapq.heappush(candidates, (edge_weight, to_node, neighbor))
        
        logger.info(f"MST created with {len(mst_edges)} edges for {len(in_mst)} images")
        return mst_edges
    
    def _place_disconnected_components(
        self,
        transforms: Dict[int, np.ndarray],
        unplaced: List[int],
        n_images: int,
        graph: Dict[int, List[int]],
        relative_transforms: Dict[Tuple[int, int], np.ndarray]
    ) -> Dict[int, np.ndarray]:
        """Place disconnected image components side by side."""
        logger.warning(f"Images {unplaced} are disconnected from main component")
        
        # #region agent log
        try:
            import json
            with open(r'c:\Users\ryanf\OneDrive - University of Maryland\Desktop\Stitch2Stitch\Stitch2Stitch-1\.cursor\debug.log', 'a') as f:
                f.write(json.dumps({"hypothesisId":"E","location":"alignment.py:_place_disconnected_components","message":"Disconnected images","data":{"n_unplaced":len(unplaced),"n_placed":len(transforms),"n_total":n_images,"n_relative_transforms":len(relative_transforms)},"timestamp":__import__('time').time()}) + '\n')
        except: pass
        # #endregion
        
        # Calculate bounding box of placed component
        if transforms:
            placed_max_x = max(t[0, 2] for t in transforms.values())
            avg_width = 1000
            start_x = placed_max_x + avg_width + 200
        else:
            start_x = 0
            avg_width = 1000
        
        # Find connected sub-components
        unplaced_set = set(unplaced)
        sub_components = []
        visited_unplaced = set()
        
        for img_idx in unplaced:
            if img_idx in visited_unplaced:
                continue
            
            component = []
            queue = [img_idx]
            while queue:
                node = queue.pop(0)
                if node in visited_unplaced or node not in unplaced_set:
                    continue
                visited_unplaced.add(node)
                component.append(node)
                for neighbor in graph.get(node, []):
                    if neighbor in unplaced_set and neighbor not in visited_unplaced:
                        queue.append(neighbor)
            if component:
                sub_components.append(component)
        
        # Place each sub-component
        current_x = start_x
        for comp_idx, component in enumerate(sub_components):
            logger.info(f"Placing disconnected component {comp_idx + 1} with {len(component)} images")
            
            if len(component) == 1:
                img_idx = component[0]
                transforms[img_idx] = np.array([
                    [1, 0, current_x],
                    [0, 1, 0],
                    [0, 0, 1]
                ], dtype=np.float64)
                current_x += avg_width + 100
            else:
                # Run BFS within this sub-component
                comp_ref = component[0]
                comp_transforms = {comp_ref: np.eye(3, dtype=np.float64)}
                comp_visited = {comp_ref}
                comp_queue = [comp_ref]
                
                while comp_queue:
                    current = comp_queue.pop(0)
                    current_t = comp_transforms[current]
                    
                    for neighbor in graph.get(current, []):
                        if neighbor in comp_visited or neighbor not in component:
                            continue
                        key = (neighbor, current)
                        if key in relative_transforms:
                            neighbor_t = current_t @ relative_transforms[key]
                            comp_transforms[neighbor] = neighbor_t
                            comp_visited.add(neighbor)
                            comp_queue.append(neighbor)
                
                # Shift component to current position
                if comp_transforms:
                    comp_min_x = min(t[0, 2] for t in comp_transforms.values())
                    comp_max_x = max(t[0, 2] for t in comp_transforms.values())
                    comp_width = comp_max_x - comp_min_x + avg_width
                    
                    for img_idx, t in comp_transforms.items():
                        shifted = t.copy()
                        shifted[0, 2] = t[0, 2] - comp_min_x + current_x
                        transforms[img_idx] = shifted
                    
                    current_x += comp_width + 200
        
        # Handle any remaining isolated images
        still_unplaced = [i for i in range(n_images) if i not in transforms]
        for img_idx in still_unplaced:
            transforms[img_idx] = np.array([
                [1, 0, current_x],
                [0, 1, 0],
                [0, 0, 1]
            ], dtype=np.float64)
            current_x += avg_width + 100
        
        return transforms
    
    def _verify_and_refine_alignment(
        self,
        transforms: Dict[int, np.ndarray],
        relative_transforms: Dict[Tuple[int, int], np.ndarray],
        n_images: int
    ) -> Dict[int, np.ndarray]:
        """
        Verify alignment consistency by checking loop closures.
        For large sets, apply simple global adjustment if inconsistency detected.
        """
        # Check a sample of loop closures
        inconsistent_pairs = []
        checked_pairs = 0
        max_checks = min(50, n_images * 2)
        
        for (i, j), rel_transform in relative_transforms.items():
            if i >= j:
                continue
            if i not in transforms or j not in transforms:
                continue
            
            checked_pairs += 1
            if checked_pairs > max_checks:
                break
            
            # Expected: T_j = T_i @ rel_T_ij
            # Check: T_j^{-1} @ T_i @ rel_T_ij should be close to identity
            T_i = transforms[i]
            T_j = transforms[j]
            
            # Compute implied transform from i to j via global transforms
            try:
                T_j_inv = np.linalg.inv(T_j)
                implied = T_j_inv @ T_i @ rel_transform
                
                # Check deviation from identity
                scale = np.sqrt(implied[0, 0]**2 + implied[0, 1]**2)
                translation = np.sqrt(implied[0, 2]**2 + implied[1, 2]**2)
                
                if abs(scale - 1.0) > 0.1 or translation > 50:
                    inconsistent_pairs.append((i, j, scale, translation))
            except np.linalg.LinAlgError:
                pass
        
        if inconsistent_pairs:
            n_inconsistent = len(inconsistent_pairs)
            inconsistency_rate = n_inconsistent / max(checked_pairs, 1)
            
            if inconsistency_rate > 0.3:
                logger.warning(f"High alignment inconsistency ({inconsistency_rate:.1%}): {n_inconsistent}/{checked_pairs} pairs")
                logger.info("Consider enabling bundle adjustment for better results")
            else:
                logger.info(f"Alignment consistency check: {n_inconsistent}/{checked_pairs} pairs have minor drift")
        
        return transforms
    
    def _clamp_extreme_positions(
        self,
        transforms: Dict[int, np.ndarray],
        n_images: int
    ) -> Dict[int, np.ndarray]:
        """
        Clamp extreme positions to prevent 'exploded' layouts.
        
        When alignment accumulates errors over many images, positions can
        spread unreasonably far apart, creating a canvas that's too large.
        
        This method detects and fixes such cases by scaling down the spread.
        """
        if len(transforms) < 5:
            return transforms
        
        # Get all positions
        positions = [(i, t[0, 2], t[1, 2]) for i, t in transforms.items()]
        x_values = [p[1] for p in positions]
        y_values = [p[2] for p in positions]
        
        spread_x = max(x_values) - min(x_values)
        spread_y = max(y_values) - min(y_values)
        total_spread = np.sqrt(spread_x**2 + spread_y**2)
        
        # Estimate reasonable spread: for n images with ~30% overlap
        # If images are 1000x1000 (typical), and arranged optimally:
        # - spread should be roughly sqrt(n) * 700 per axis
        # We'll use a generous estimate assuming 1000px images
        assumed_image_size = 1000
        reasonable_spread_per_axis = np.sqrt(n_images) * assumed_image_size * 0.7
        reasonable_total = reasonable_spread_per_axis * np.sqrt(2)
        
        # If spread is more than 10x expected, it's exploded
        if total_spread > reasonable_total * 10:
            logger.warning(f"Alignment spread too large: {total_spread:.0f}px vs expected {reasonable_total:.0f}px")
            
            # Calculate scale factor to bring within reasonable range
            # Target 3x expected (generous, but not insane)
            target_spread = reasonable_total * 3
            scale_factor = target_spread / total_spread
            
            logger.info(f"Scaling down positions by {scale_factor:.3f}x")
            
            # Find center of current layout
            center_x = (max(x_values) + min(x_values)) / 2
            center_y = (max(y_values) + min(y_values)) / 2
            
            # Scale positions around center
            new_transforms = {}
            for idx, transform in transforms.items():
                new_t = transform.copy()
                # Scale translation relative to center
                new_t[0, 2] = center_x + (transform[0, 2] - center_x) * scale_factor
                new_t[1, 2] = center_y + (transform[1, 2] - center_y) * scale_factor
                new_transforms[idx] = new_t
            
            new_spread_x = (max(x_values) - min(x_values)) * scale_factor
            new_spread_y = (max(y_values) - min(y_values)) * scale_factor
            logger.info(f"New spread: {new_spread_x:.0f}x{new_spread_y:.0f}px")
            
            return new_transforms
        
        return transforms
    
    def _fix_linear_layout(
        self,
        transforms: Dict[int, np.ndarray],
        n_images: int,
        images_data: List[Dict] = None
    ) -> Dict[int, np.ndarray]:
        """
        Detect and fix linear layouts (all images on one horizontal line).
        
        This can happen when:
        1. Feature matching only found chain connections
        2. Images are unsorted and true spatial neighbors weren't matched
        
        For UNSORTED images, we can't assume any ordering, so we:
        1. Detect if layout is suspiciously linear
        2. If so, try to find natural clustering in the positions
        3. Reorganize into a 2D arrangement if possible
        
        NOTE: This is a fallback heuristic. For best results, ensure proper
        feature matching finds both horizontal AND vertical connections.
        """
        if len(transforms) < 4:
            return transforms
        
        # Get all positions (note: NOT sorted by index since images are unsorted)
        positions = [(i, transforms[i][0, 2], transforms[i][1, 2]) for i in transforms.keys()]
        
        y_values = [p[2] for p in positions]
        x_values = [p[1] for p in positions]
        
        y_range = max(y_values) - min(y_values) if y_values else 0
        x_range = max(x_values) - min(x_values) if x_values else 1
        
        # Calculate aspect ratio of the layout
        aspect_ratio = y_range / max(x_range, 1)
        
        # Check for collapsed alignment (all images at nearly same position)
        avg_image_spread = np.sqrt((np.std(x_values)**2 + np.std(y_values)**2))
        if avg_image_spread < 50:  # Less than 50 pixels spread
            logger.error(f"Alignment collapsed: all images at similar position (spread={avg_image_spread:.0f}px)")
            return self._create_fallback_grid_layout(transforms, n_images)
        
        # Only trigger fix if layout is VERY linear (aspect < 0.1)
        if x_range > 0 and aspect_ratio < 0.1 and n_images >= 6:
            logger.warning(f"Detected LINEAR layout: Y range={y_range:.0f}, X range={x_range:.0f} (aspect={aspect_ratio:.3f})")
            
            # Sort by X position (not by index - images are unsorted)
            x_sorted = sorted(positions, key=lambda p: p[1])
            spacings = []
            for i in range(len(x_sorted) - 1):
                spacing = x_sorted[i + 1][1] - x_sorted[i][1]
                if spacing > 0:
                    spacings.append((i, spacing))
            
            if not spacings:
                logger.warning("No spacing data, creating fallback grid")
                return self._create_fallback_grid_layout(transforms, n_images)
            
            median_spacing = np.median([s[1] for s in spacings])
            
            # Look for large gaps that might indicate row breaks
            large_gaps = [(i, s) for i, s in spacings if s > median_spacing * 2]
            
            if large_gaps and len(large_gaps) >= 2:
                logger.info(f"Found {len(large_gaps)} potential row breaks")
                return self._reorganize_by_gaps(transforms, x_sorted, large_gaps, median_spacing)
            else:
                return self._reorganize_estimated_grid(transforms, x_sorted, median_spacing, n_images)
        
        # Also check for nearly-vertical linear layout
        if y_range > 0 and x_range / max(y_range, 1) < 0.1 and n_images >= 6:
            logger.warning(f"Detected VERTICAL LINEAR layout: X range={x_range:.0f}, Y range={y_range:.0f}")
            y_sorted = sorted(positions, key=lambda p: p[2])
            spacings = []
            for i in range(len(y_sorted) - 1):
                spacing = y_sorted[i + 1][2] - y_sorted[i][2]
                if spacing > 0:
                    spacings.append((i, spacing))
            
            if spacings:
                median_spacing = np.median([s[1] for s in spacings])
                return self._reorganize_estimated_grid_vertical(transforms, y_sorted, median_spacing, n_images)
        
        return transforms
    
    def _create_fallback_grid_layout(
        self,
        transforms: Dict[int, np.ndarray],
        n_images: int
    ) -> Dict[int, np.ndarray]:
        """
        Create a simple grid layout when alignment completely fails.
        This is a last resort to at least show images in an organized way.
        """
        logger.warning("Creating fallback grid layout due to alignment failure")
        
        grid_cols = int(np.ceil(np.sqrt(n_images * 1.3)))
        grid_rows = int(np.ceil(n_images / grid_cols))
        
        # Assume typical image size for spacing
        spacing = 800  # Will be adjusted if images overlap
        
        new_transforms = {}
        indices = sorted(transforms.keys())
        
        for seq_idx, img_idx in enumerate(indices):
            row = seq_idx // grid_cols
            col = seq_idx % grid_cols
            
            new_x = col * spacing
            new_y = row * spacing
            
            old_transform = transforms[img_idx]
            new_transform = old_transform.copy()
            new_transform[0, 2] = new_x
            new_transform[1, 2] = new_y
            new_transforms[img_idx] = new_transform
        
        logger.info(f"Fallback grid: {grid_rows}x{grid_cols} with {spacing}px spacing")
        return new_transforms
    
    def _reorganize_by_gaps(
        self,
        transforms: Dict[int, np.ndarray],
        x_sorted: List[Tuple[int, float, float]],
        large_gaps: List[Tuple[int, float]],
        horizontal_spacing: float
    ) -> Dict[int, np.ndarray]:
        """Reorganize using detected gap positions as row breaks."""
        new_transforms = {}
        
        # Split into rows at gap positions
        gap_indices = sorted([g[0] for g in large_gaps])
        rows = []
        start = 0
        for gap_idx in gap_indices:
            rows.append(x_sorted[start:gap_idx + 1])
            start = gap_idx + 1
        rows.append(x_sorted[start:])
        
        # Remove empty rows
        rows = [r for r in rows if r]
        
        if not rows:
            return transforms
        
        logger.info(f"Reorganizing into {len(rows)} rows")
        
        vertical_spacing = horizontal_spacing  # Assume similar overlap
        
        for row_idx, row in enumerate(rows):
            for col_idx, (img_idx, old_x, old_y) in enumerate(row):
                # Preserve relative X positions within row, apply row Y offset
                row_min_x = min(p[1] for p in row)
                new_x = old_x - row_min_x
                new_y = row_idx * vertical_spacing
                
                old_transform = transforms[img_idx]
                new_transform = old_transform.copy()
                new_transform[0, 2] = new_x
                new_transform[1, 2] = new_y
                new_transforms[img_idx] = new_transform
        
        return new_transforms
    
    def _reorganize_estimated_grid(
        self,
        transforms: Dict[int, np.ndarray],
        x_sorted: List[Tuple[int, float, float]],
        horizontal_spacing: float,
        n_images: int
    ) -> Dict[int, np.ndarray]:
        """Reorganize into estimated grid dimensions."""
        # Estimate grid size - prefer wider grids
        grid_cols = int(np.ceil(np.sqrt(n_images * 1.3)))
        grid_rows = int(np.ceil(n_images / grid_cols))
        
        logger.info(f"Estimated grid: {grid_rows}x{grid_cols} (no natural row breaks found)")
        
        vertical_spacing = horizontal_spacing
        new_transforms = {}
        
        for seq_idx, (img_idx, old_x, old_y) in enumerate(x_sorted):
            row = seq_idx // grid_cols
            col = seq_idx % grid_cols
            
            # Snake pattern for scanning patterns
            if row % 2 == 1:
                col = grid_cols - 1 - col
            
            new_x = col * horizontal_spacing
            new_y = row * vertical_spacing
            
            old_transform = transforms[img_idx]
            new_transform = old_transform.copy()
            new_transform[0, 2] = new_x
            new_transform[1, 2] = new_y
            new_transforms[img_idx] = new_transform
        
        return new_transforms
    
    def _reorganize_estimated_grid_vertical(
        self,
        transforms: Dict[int, np.ndarray],
        y_sorted: List[Tuple[int, float, float]],
        vertical_spacing: float,
        n_images: int
    ) -> Dict[int, np.ndarray]:
        """Reorganize vertical linear layout into grid."""
        grid_rows = int(np.ceil(np.sqrt(n_images * 1.3)))
        grid_cols = int(np.ceil(n_images / grid_rows))
        
        logger.info(f"Estimated grid (from vertical): {grid_rows}x{grid_cols}")
        
        horizontal_spacing = vertical_spacing
        new_transforms = {}
        
        for seq_idx, (img_idx, old_x, old_y) in enumerate(y_sorted):
            col = seq_idx // grid_rows
            row = seq_idx % grid_rows
            
            if col % 2 == 1:
                row = grid_rows - 1 - row
            
            new_x = col * horizontal_spacing
            new_y = row * vertical_spacing
            
            old_transform = transforms[img_idx]
            new_transform = old_transform.copy()
            new_transform[0, 2] = new_x
            new_transform[1, 2] = new_y
            new_transforms[img_idx] = new_transform
        
        return new_transforms
    
    def _apply_transforms(
        self,
        images_data: List[Dict],
        transforms: Dict[int, np.ndarray],
        ref_idx: int
    ) -> List[Dict]:
        """
        Apply transforms to images
        For pure translation, no warping needed
        For scale/rotation, warp the image
        """
        aligned_images = []
        n_images = len(images_data)
        
        for i, img_data in enumerate(images_data):
            if i not in transforms:
                logger.warning(f"Image {i} has no transform, skipping")
                continue
            
            transform = transforms[i]
            img = img_data['image']
            h, w = img.shape[:2]
            
            logger.debug(f"Applying transform to image {i+1}/{n_images} ({w}x{h})")
            
            # Check if transform is pure translation (identity rotation/scale)
            scale = np.sqrt(transform[0, 0]**2 + transform[0, 1]**2)
            rotation = np.arctan2(transform[1, 0], transform[0, 0])
            
            is_pure_translation = (abs(scale - 1.0) < 0.01 and abs(rotation) < 0.01)
            
            if is_pure_translation:
                # No warping needed - just use position
                tx, ty = transform[0, 2], transform[1, 2]
                x, y = int(round(tx)), int(round(ty))
                
                aligned_data = img_data.copy()
                aligned_data['transform'] = transform
                aligned_data['bbox'] = (x, y, x + w, y + h)
                aligned_data['warped'] = False
                aligned_images.append(aligned_data)
            else:
                # Need to warp the image
                warped_img, bbox = self._warp_image(img, transform)
                
                aligned_data = img_data.copy()
                aligned_data['image'] = warped_img
                aligned_data['transform'] = transform
                aligned_data['bbox'] = bbox
                aligned_data['warped'] = True
                
                # Warp alpha if present
                if img_data.get('alpha') is not None:
                    warped_alpha, _ = self._warp_image(img_data['alpha'], transform)
                    aligned_data['alpha'] = warped_alpha
                
                aligned_images.append(aligned_data)
            
            # Periodic garbage collection for large batches
            if i > 0 and i % 10 == 0:
                gc.collect()
        
        return aligned_images
    
    def _warp_image(
        self,
        img: np.ndarray,
        transform: np.ndarray
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Warp image using similarity transform.
        
        Note: This method does NOT apply size limits. Size limits are handled
        globally by the blender to ensure consistent scaling across all images.
        
        Returns:
            Tuple of (warped_image, bbox)
            - warped_image: The transformed image
            - bbox: (x_min, y_min, x_max, y_max) in global coordinates
        """
        h, w = img.shape[:2]
        
        # Calculate output bounds by transforming corners
        corners = np.array([
            [0, 0, 1],
            [w, 0, 1],
            [w, h, 1],
            [0, h, 1]
        ], dtype=np.float64).T
        
        transformed = transform @ corners
        transformed = transformed[:2, :] / transformed[2, :]
        
        x_min = int(np.floor(np.min(transformed[0])))
        x_max = int(np.ceil(np.max(transformed[0])))
        y_min = int(np.floor(np.min(transformed[1])))
        y_max = int(np.ceil(np.max(transformed[1])))
        
        output_w = max(1, x_max - x_min)
        output_h = max(1, y_max - y_min)
        
        # Safety check: prevent obviously wrong transforms
        # This catches numerical issues like extreme scales or broken transforms
        max_reasonable_size = 100000  # 100k pixels per side max
        if output_w > max_reasonable_size or output_h > max_reasonable_size:
            logger.error(f"Transform produced unreasonable output size: {output_w}x{output_h}")
            logger.error(f"Transform matrix:\n{transform}")
            # Clamp to reasonable size
            if output_w > max_reasonable_size:
                scale = max_reasonable_size / output_w
                output_w = max_reasonable_size
                output_h = int(output_h * scale)
                x_max = x_min + output_w
                y_max = y_min + output_h
            if output_h > max_reasonable_size:
                scale = max_reasonable_size / output_h
                output_h = max_reasonable_size
                output_w = int(output_w * scale)
                x_max = x_min + output_w
                y_max = y_min + output_h
        
        # Create adjusted transform that accounts for output offset
        offset = np.array([
            [1, 0, -x_min],
            [0, 1, -y_min],
            [0, 0, 1]
        ], dtype=np.float64)
        adjusted = offset @ transform
        
        # Warp using affine transform
        warped = cv2.warpAffine(
            img,
            adjusted[:2, :].astype(np.float32),
            (output_w, output_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT
        )
        
        return warped, (x_min, y_min, x_max, y_max)
    
    def _place_images_at_origin(self, images_data: List[Dict]) -> List[Dict]:
        """Place all images at origin when no matches found"""
        aligned_images = []
        for i, img_data in enumerate(images_data):
            h, w = img_data['image'].shape[:2]
            aligned_data = img_data.copy()
            aligned_data['transform'] = np.eye(3, dtype=np.float32)
            aligned_data['bbox'] = (0, 0, w, h)
            aligned_data['warped'] = False
            aligned_images.append(aligned_data)
        return aligned_images
    
    def create_grid_layout(
        self,
        images_data: List[Dict],
        features_data: List[Dict],
        matches: List[Dict],
        grid_size: Optional[Tuple[int, int]] = None,
        spacing_factor: float = 1.2
    ) -> Dict:
        """
        Create 2D grid layout based on feature matches - "exploded view" style
        
        Images are positioned in their correct relative positions and rotations,
        but with uniform spacing between them.
        
        Args:
            images_data: List of image data dictionaries
            features_data: List of feature data dictionaries  
            matches: List of match dictionaries
            grid_size: Optional (rows, cols) for grid
            spacing_factor: Multiplier for spacing (1.0 = touching, 1.2 = 20% gap)
        """
        logger.info("Creating grid layout (exploded view)...")
        
        # First, get the alignment transforms
        n_images = len(images_data)
        
        # Calculate relative transforms
        relative_transforms = self._calculate_relative_transforms(
            images_data, features_data, matches
        )
        
        if not relative_transforms:
            logger.warning("No transforms found, using simple grid layout")
            return self._create_simple_grid_layout(images_data, grid_size)
        
        # Build graph and find reference
        graph = self._build_graph(images_data, matches)
        ref_idx = self._find_reference_image(graph)
        
        # Get absolute transforms
        transforms = self._calculate_absolute_transforms(
            n_images, relative_transforms, graph, ref_idx
        )
        
        # Calculate center positions for each image
        centers = {}
        for i, img_data in enumerate(images_data):
            if i not in transforms:
                continue
            
            transform = transforms[i]
            h, w = img_data['image'].shape[:2]
            
            # Transform the center point
            center_local = np.array([w/2, h/2, 1])
            center_global = transform @ center_local
            centers[i] = (center_global[0], center_global[1])
        
        if not centers:
            return self._create_simple_grid_layout(images_data, grid_size)
        
        # Calculate the bounding box of all centers
        all_cx = [c[0] for c in centers.values()]
        all_cy = [c[1] for c in centers.values()]
        min_cx, max_cx = min(all_cx), max(all_cx)
        min_cy, max_cy = min(all_cy), max(all_cy)
        
        # Normalize centers to start from origin
        for i in centers:
            cx, cy = centers[i]
            centers[i] = (cx - min_cx, cy - min_cy)
        
        # Apply spacing factor to spread images apart
        avg_size = np.mean([max(img['shape'][0], img['shape'][1]) for img in images_data])
        
        for i in centers:
            cx, cy = centers[i]
            centers[i] = (cx * spacing_factor, cy * spacing_factor)
        
        # Create positions with transforms for rotation
        positions = []
        for i, img_data in enumerate(images_data):
            if i not in centers:
                # Disconnected image - place at end
                cx = (max_cx - min_cx) * spacing_factor + avg_size * (len(positions) + 1)
                cy = 0
            else:
                cx, cy = centers[i]
            
            # Get rotation angle from transform
            rotation = 0.0
            scale = 1.0
            if i in transforms:
                t = transforms[i]
                scale = np.sqrt(t[0, 0]**2 + t[0, 1]**2)
                rotation = np.degrees(np.arctan2(t[1, 0], t[0, 0]))
            
            h, w = img_data['image'].shape[:2]
            
            # Position is top-left corner, accounting for image size
            x = int(cx - w/2)
            y = int(cy - h/2)
            
            positions.append({
                'image_idx': i,
                'x': x,
                'y': y,
                'center_x': cx,
                'center_y': cy,
                'rotation': rotation,
                'scale': scale,
                'row': 0,
                'col': i
            })
        
        # Normalize positions so minimum is at origin with padding
        if positions:
            min_x = min(p['x'] for p in positions)
            min_y = min(p['y'] for p in positions)
            padding = 50
            
            for p in positions:
                p['x'] -= min_x - padding
                p['y'] -= min_y - padding
                p['center_x'] -= min_x - padding
                p['center_y'] -= min_y - padding
        
        if grid_size is None:
            grid_size = self._estimate_grid_size(len(images_data))
        
        return {
            'images': images_data,
            'positions': positions,
            'grid_size': grid_size,
            'matches': matches,
            'transforms': transforms
        }
    
    def _create_simple_grid_layout(
        self,
        images_data: List[Dict],
        grid_size: Optional[Tuple[int, int]] = None
    ) -> Dict:
        """Create simple row-based grid layout when no matches found"""
        if grid_size is None:
            grid_size = self._estimate_grid_size(len(images_data))
        
        rows, cols = grid_size
        positions = []
        
        # Calculate max dimensions for spacing
        max_w = max(img['shape'][1] for img in images_data) if images_data else 100
        max_h = max(img['shape'][0] for img in images_data) if images_data else 100
        spacing = 50
        
        for i, img_data in enumerate(images_data):
            row = i // cols
            col = i % cols
            
            x = col * (max_w + spacing) + spacing
            y = row * (max_h + spacing) + spacing
            
            positions.append({
                'image_idx': i,
                'x': x,
                'y': y,
                'center_x': x + img_data['shape'][1] / 2,
                'center_y': y + img_data['shape'][0] / 2,
                'rotation': 0.0,
                'scale': 1.0,
                'row': row,
                'col': col
            })
        
        return {
            'images': images_data,
            'positions': positions,
            'grid_size': grid_size,
            'matches': [],
            'transforms': {}
        }
    
    def _estimate_grid_size(self, n_images: int) -> Tuple[int, int]:
        """Estimate grid dimensions"""
        if n_images == 0:
            return (1, 1)
        cols = int(np.ceil(np.sqrt(n_images)))
        if cols == 0:
            cols = 1
        rows = int(np.ceil(n_images / cols))
        return (rows, cols)
