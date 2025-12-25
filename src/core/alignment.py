"""
Image alignment engine for flat images (like scanned documents, maps, artwork)
Uses similarity transforms (translation + uniform scale) - no perspective distortion
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
import gc

logger = logging.getLogger(__name__)

# Default maximum pixels for a single warped image output (50 megapixels)
# Set to None or 0 to disable limit
# Memory guidelines:
#   - Each warped image: width * height * 3 bytes (RGB uint8)
#   - 50MP image: ~150MB RAM
#   - 100MP image: ~300MB RAM
DEFAULT_MAX_WARP_PIXELS = 50_000_000


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
            logger.warning(f"No valid transforms found from {len(matches)} matches - placing images in grid")
            return self._place_images_at_origin(images_data)
        
        logger.info(f"Found {len(relative_transforms)} valid transforms from {len(matches)} matches")
        
        # Build connectivity graph
        graph = self._build_graph(images_data, matches)
        
        # Find reference image (most connected)
        ref_idx = self._find_reference_image(graph)
        logger.info(f"Using image {ref_idx} as reference")
        
        # Calculate absolute transforms using BFS from reference
        transforms = self._calculate_absolute_transforms(
            n_images, relative_transforms, graph, ref_idx
        )
        
        # Refine transforms using edge matching in overlap regions
        transforms = self._refine_transforms_with_edges(
            images_data, transforms, relative_transforms
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
        Calculate relative similarity transform between each matched pair
        
        Returns:
            Dict mapping (i, j) -> 3x3 transform matrix from image i coords to image j coords
        """
        relative_transforms = {}
        
        for match in matches:
            i = match['image_i']
            j = match['image_j']
            match_list = match['matches']
            
            if len(match_list) < 4:
                continue
            
            kp_i = features_data[i]['keypoints']
            kp_j = features_data[j]['keypoints']
            
            # Extract matched points
            pts_i = []
            pts_j = []
            
            for m in match_list:
                idx_i = m.queryIdx
                idx_j = m.trainIdx
                
                if idx_i < len(kp_i) and idx_j < len(kp_j):
                    pts_i.append([kp_i[idx_i][0], kp_i[idx_i][1]])
                    pts_j.append([kp_j[idx_j][0], kp_j[idx_j][1]])
            
            if len(pts_i) < 4:
                logger.debug(f"Match ({i}, {j}): only {len(pts_i)} valid points, skipping")
                continue
            
            pts_i = np.float32(pts_i)
            pts_j = np.float32(pts_j)
            
            # Estimate similarity transform (translation + uniform scale + rotation)
            transform = self._estimate_similarity_transform(pts_i, pts_j)
            
            if transform is not None:
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
                logger.info(f"Transform ({i} -> {j}): scale={scale:.3f}, rotation={rotation:.1f}°, tx={tx:.1f}, ty={ty:.1f} (from {len(pts_i)} points)")
        
        return relative_transforms
    
    def _estimate_similarity_transform(
        self,
        pts1: np.ndarray,
        pts2: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Estimate similarity transform from pts1 to pts2 with robust outlier rejection.
        
        Similarity transform has 4 DOF: translation (tx, ty), uniform scale (s), rotation (θ)
        Matrix form: [[s*cos(θ), -s*sin(θ), tx],
                      [s*sin(θ),  s*cos(θ), ty],
                      [0,         0,         1]]
        
        Uses multiple rounds of RANSAC with decreasing threshold for thorough outlier removal.
        """
        if len(pts1) < 4:
            return None
        
        # Multi-round RANSAC with decreasing threshold for robust estimation
        current_pts1 = pts1.copy()
        current_pts2 = pts2.copy()
        best_transform = None
        best_inlier_count = 0
        
        # Round 1: Coarse estimation (loose threshold)
        similarity, inliers = cv2.estimateAffinePartial2D(
            current_pts1, current_pts2,
            method=cv2.RANSAC,
            ransacReprojThreshold=5.0,  # Looser threshold first
            confidence=0.99,
            maxIters=1000
        )
        
        if similarity is not None and inliers is not None:
            # Keep only inliers for refinement
            inlier_mask = inliers.ravel().astype(bool)
            if np.sum(inlier_mask) >= 4:
                current_pts1 = current_pts1[inlier_mask]
                current_pts2 = current_pts2[inlier_mask]
        
        # Round 2: Fine estimation (strict threshold)
        if len(current_pts1) >= 4:
            similarity, inliers = cv2.estimateAffinePartial2D(
                current_pts1, current_pts2,
                method=cv2.RANSAC,
                ransacReprojThreshold=2.5,  # Stricter threshold
                confidence=0.995,
                maxIters=2000
            )
        
        if similarity is None:
            logger.warning("Failed to estimate similarity transform")
            return None
        
        # Validate transform
        scale = np.sqrt(similarity[0, 0]**2 + similarity[0, 1]**2)
        rotation = np.arctan2(similarity[1, 0], similarity[0, 0])
        rotation_deg = np.degrees(rotation)
        
        # Check scale bounds
        if not self.allow_scale:
            # Force scale to 1.0 if scaling disabled
            if abs(scale - 1.0) > 0.05:
                # Re-estimate with translation only
                return self._estimate_translation_only(pts1, pts2)
        else:
            # Allow reasonable scale range (0.5x to 2.0x)
            if scale < 0.5 or scale > 2.0:
                logger.warning(f"Scale {scale:.3f} out of range, rejecting transform")
                return None
        
        # Rotation is allowed - but warn for large rotations
        if abs(rotation_deg) > 45:
            logger.info(f"Large rotation detected: {rotation_deg:.1f}°")
        
        # Check inlier ratio - be lenient to avoid rejecting valid transforms
        if inliers is not None:
            inlier_count = np.sum(inliers)
            inlier_ratio = inlier_count / len(pts1)  # Use original point count
            
            # Only reject if both ratio is very low AND we have few absolute inliers
            if inlier_ratio < 0.15 and inlier_count < 10:
                logger.warning(f"Very low inlier ratio {inlier_ratio:.1%} ({inlier_count}/{len(pts1)}), rejecting transform")
                return None
            
            logger.debug(f"Transform validated: scale={scale:.3f}, rotation={rotation_deg:.1f}°, inliers={inlier_count}/{len(pts1)} ({inlier_ratio:.1%})")
        
        # Compute reprojection error for quality assessment (informational only, don't reject)
        if inliers is not None:
            inlier_mask = inliers.ravel().astype(bool)
            if np.any(inlier_mask):
                pts1_inliers = current_pts1[inlier_mask] if len(current_pts1) == len(inlier_mask) else current_pts1
                pts2_inliers = current_pts2[inlier_mask] if len(current_pts2) == len(inlier_mask) else current_pts2
                
                if len(pts1_inliers) > 0:
                    # Transform pts1 and compute distance to pts2
                    pts1_h = np.hstack([pts1_inliers, np.ones((len(pts1_inliers), 1))])
                    projected = (similarity @ pts1_h.T).T
                    errors = np.linalg.norm(projected - pts2_inliers, axis=1)
                    mean_error = np.mean(errors)
                    max_error = np.max(errors)
                    
                    # Only reject for extremely high error (indicates wrong transform)
                    if mean_error > 50.0:  # Very high threshold - only reject clearly wrong transforms
                        logger.warning(f"Very high reprojection error: mean={mean_error:.1f}px, rejecting")
                        return None
                    
                    if mean_error > 15.0:
                        logger.warning(f"High reprojection error: mean={mean_error:.1f}px, max={max_error:.1f}px (accepting anyway)")
                    else:
                        logger.debug(f"Reprojection error: mean={mean_error:.2f}px, max={max_error:.2f}px")
        
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
    ) -> Dict[int, List[Tuple[int, float]]]:
        """
        Build weighted connectivity graph from matches.
        
        For burst photos, sequential connections (i, i+1) are weighted higher.
        Returns dict mapping image_idx -> list of (neighbor_idx, weight) tuples.
        Weight represents connection quality (higher = better).
        """
        n = len(images_data)
        graph = {i: [] for i in range(n)}
        
        # Build match quality map
        match_quality = {}
        for match in matches:
            i = match['image_i']
            j = match['image_j']
            if match['confidence'] > 0.1 and match.get('num_matches', 0) >= 10:
                # Quality based on number of matches and inlier ratio
                num_matches = match.get('num_matches', 0)
                inlier_ratio = match.get('inlier_ratio', 0.5)
                quality = num_matches * inlier_ratio
                
                # Boost for sequential connections (burst photo priority)
                if abs(i - j) == 1:
                    quality *= 2.0  # Double weight for adjacent images
                elif abs(i - j) <= 3:
                    quality *= 1.5  # 1.5x weight for near-sequential
                
                match_quality[(i, j)] = quality
                match_quality[(j, i)] = quality
        
        # Build graph with weighted edges
        for (i, j), quality in match_quality.items():
            if i < j:  # Only process each pair once
                graph[i].append((j, quality))
                graph[j].append((i, quality))
        
        # Sort neighbors by weight (highest first) for priority traversal
        for i in graph:
            graph[i].sort(key=lambda x: x[1], reverse=True)
        
        return graph
    
    def _find_reference_image(self, graph: Dict[int, List[Tuple[int, float]]]) -> int:
        """
        Find best reference image based on connection quality.
        
        Prefers images with high-quality connections to neighbors,
        especially sequential neighbors for burst photos.
        """
        best_score = -1
        ref_idx = 0
        
        for idx, connections in graph.items():
            # Score based on total connection quality
            total_quality = sum(weight for _, weight in connections)
            # Bonus for being in the middle of the sequence (for burst photos)
            n_images = len(graph)
            center_bonus = 1.0 - abs(idx - n_images // 2) / (n_images / 2 + 1)
            score = total_quality * (1.0 + 0.2 * center_bonus)
            
            if score > best_score:
                best_score = score
                ref_idx = idx
        
        logger.info(f"Selected reference image {ref_idx} (score={best_score:.1f})")
        return ref_idx
    
    def _calculate_absolute_transforms(
        self,
        n_images: int,
        relative_transforms: Dict[Tuple[int, int], np.ndarray],
        graph: Dict[int, List[Tuple[int, float]]],
        ref_idx: int
    ) -> Dict[int, np.ndarray]:
        """
        Calculate absolute transforms using priority BFS from reference image.
        
        Prioritizes sequential connections (for burst photos) and high-quality matches.
        Each transform maps from the image's local coordinates to the 
        global (reference) coordinate system.
        """
        import heapq
        
        # Reference image has identity transform
        transforms = {ref_idx: np.eye(3, dtype=np.float64)}
        visited = {ref_idx}
        
        # Priority queue: (negative_quality, image_idx, parent_idx)
        # Use negative because heapq is min-heap
        pq = []
        for neighbor, weight in graph.get(ref_idx, []):
            heapq.heappush(pq, (-weight, neighbor, ref_idx))
        
        while pq:
            neg_quality, neighbor, parent = heapq.heappop(pq)
            
            if neighbor in visited:
                continue
            
            # Get transform from neighbor to parent
            key = (neighbor, parent)
            if key not in relative_transforms:
                # Try reverse direction
                reverse_key = (parent, neighbor)
                if reverse_key in relative_transforms:
                    try:
                        neighbor_to_parent = np.linalg.inv(relative_transforms[reverse_key])
                    except np.linalg.LinAlgError:
                        continue
                else:
                    continue
            else:
                neighbor_to_parent = relative_transforms[key]
            
            parent_transform = transforms[parent]
            neighbor_to_global = parent_transform @ neighbor_to_parent
            
            # Accept this placement
            transforms[neighbor] = neighbor_to_global
            visited.add(neighbor)
            
            # Log position
            tx, ty = neighbor_to_global[0, 2], neighbor_to_global[1, 2]
            scale = np.sqrt(neighbor_to_global[0, 0]**2 + neighbor_to_global[0, 1]**2)
            logger.info(f"Image {neighbor}: position=({tx:.1f}, {ty:.1f}), scale={scale:.3f} (via {parent})")
            
            # Add neighbors to queue
            for next_neighbor, weight in graph.get(neighbor, []):
                if next_neighbor not in visited:
                    heapq.heappush(pq, (-weight, next_neighbor, neighbor))
        
        # Handle disconnected images using sequential chain
        unplaced = [i for i in range(n_images) if i not in transforms]
        if unplaced:
            logger.warning(f"{len(unplaced)} images disconnected, using sequential chain")
            self._place_disconnected_sequentially(
                unplaced, transforms, relative_transforms, n_images
            )
        
        return transforms
    
    def _place_disconnected_sequentially(
        self,
        unplaced: List[int],
        transforms: Dict[int, np.ndarray],
        relative_transforms: Dict[Tuple[int, int], np.ndarray],
        n_images: int
    ):
        """Place disconnected images by following sequential chain."""
        # Sort unplaced by distance to nearest placed image
        for img_idx in sorted(unplaced):
            if img_idx in transforms:
                continue
                
            # Try sequential neighbors first
            placed = False
            for offset in [-1, 1, -2, 2, -3, 3, -5, 5]:
                seq_neighbor = img_idx + offset
                if 0 <= seq_neighbor < n_images and seq_neighbor in transforms:
                    key = (img_idx, seq_neighbor)
                    if key in relative_transforms:
                        seq_transform = transforms[seq_neighbor]
                        rel_transform = relative_transforms[key]
                        transforms[img_idx] = seq_transform @ rel_transform
                        tx, ty = transforms[img_idx][0, 2], transforms[img_idx][1, 2]
                        logger.info(f"Image {img_idx}: placed via sequential neighbor {seq_neighbor} at ({tx:.0f}, {ty:.0f})")
                        placed = True
                        break
                    
                    # Try inverse
                    reverse_key = (seq_neighbor, img_idx)
                    if reverse_key in relative_transforms:
                        try:
                            seq_transform = transforms[seq_neighbor]
                            rel_transform = np.linalg.inv(relative_transforms[reverse_key])
                            transforms[img_idx] = seq_transform @ rel_transform
                            tx, ty = transforms[img_idx][0, 2], transforms[img_idx][1, 2]
                            logger.info(f"Image {img_idx}: placed via sequential neighbor {seq_neighbor} (inverse) at ({tx:.0f}, {ty:.0f})")
                            placed = True
                            break
                        except np.linalg.LinAlgError:
                            continue
            
            if not placed:
                # Interpolate position from nearest placed neighbors
                prev_placed = None
                next_placed = None
                
                for p in range(img_idx - 1, -1, -1):
                    if p in transforms:
                        prev_placed = p
                        break
                
                for n in range(img_idx + 1, n_images):
                    if n in transforms:
                        next_placed = n
                        break
                
                if prev_placed is not None and next_placed is not None:
                    # Interpolate
                    prev_pos = transforms[prev_placed][:2, 2]
                    next_pos = transforms[next_placed][:2, 2]
                    t = (img_idx - prev_placed) / (next_placed - prev_placed)
                    interp_pos = prev_pos + t * (next_pos - prev_pos)
                    transforms[img_idx] = np.array([
                        [1, 0, interp_pos[0]],
                        [0, 1, interp_pos[1]],
                        [0, 0, 1]
                    ], dtype=np.float64)
                    logger.info(f"Image {img_idx}: interpolated between {prev_placed} and {next_placed} at ({interp_pos[0]:.0f}, {interp_pos[1]:.0f})")
                elif prev_placed is not None:
                    # Extrapolate from previous
                    prev_pos = transforms[prev_placed][:2, 2]
                    # Estimate offset from earlier images
                    offset = np.array([200, 0])  # Default horizontal offset
                    if prev_placed > 0 and prev_placed - 1 in transforms:
                        offset = prev_pos - transforms[prev_placed - 1][:2, 2]
                    new_pos = prev_pos + offset * (img_idx - prev_placed)
                    transforms[img_idx] = np.array([
                        [1, 0, new_pos[0]],
                        [0, 1, new_pos[1]],
                        [0, 0, 1]
                    ], dtype=np.float64)
                    logger.info(f"Image {img_idx}: extrapolated from {prev_placed} at ({new_pos[0]:.0f}, {new_pos[1]:.0f})")
                else:
                    # Place at origin
                    transforms[img_idx] = np.eye(3, dtype=np.float64)
                    logger.warning(f"Image {img_idx}: no neighbors, placed at origin")
    
    def _refine_transforms_with_edges(
        self,
        images_data: List[Dict],
        transforms: Dict[int, np.ndarray],
        relative_transforms: Dict[Tuple[int, int], np.ndarray]
    ) -> Dict[int, np.ndarray]:
        """
        Refine transforms by verifying edge alignment in overlap regions.
        
        For each pair of overlapping images:
        1. Compute the overlap region
        2. Extract edges using Canny
        3. Verify edges match after transform
        4. Apply small corrections if needed
        """
        n_images = len(images_data)
        
        # Get image sizes
        sizes = {}
        for i, img_data in enumerate(images_data):
            if i in transforms:
                h, w = img_data['image'].shape[:2]
                sizes[i] = (w, h)
        
        # For each sequential pair, verify and refine
        refinements_made = 0
        for i in range(n_images - 1):
            if i not in transforms or i + 1 not in transforms:
                continue
            
            # Check if we have relative transform between them
            key = (i, i + 1)
            if key not in relative_transforms and (i + 1, i) not in relative_transforms:
                continue
            
            try:
                correction = self._compute_edge_correction(
                    images_data[i]['image'],
                    images_data[i + 1]['image'],
                    transforms[i],
                    transforms[i + 1],
                    sizes.get(i),
                    sizes.get(i + 1)
                )
                
                if correction is not None and np.linalg.norm(correction) > 0.5:
                    # Apply correction to image i+1 and all subsequent images
                    correction_matrix = np.array([
                        [1, 0, correction[0]],
                        [0, 1, correction[1]],
                        [0, 0, 1]
                    ], dtype=np.float64)
                    
                    for j in range(i + 1, n_images):
                        if j in transforms:
                            transforms[j] = correction_matrix @ transforms[j]
                    
                    refinements_made += 1
                    logger.debug(f"Applied edge correction ({correction[0]:.1f}, {correction[1]:.1f}) to images {i+1}+")
                    
            except Exception as e:
                logger.debug(f"Edge refinement failed for pair ({i}, {i+1}): {e}")
        
        if refinements_made > 0:
            logger.info(f"Applied {refinements_made} edge-based refinements")
        
        return transforms
    
    def _compute_edge_correction(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        transform1: np.ndarray,
        transform2: np.ndarray,
        size1: Tuple[int, int],
        size2: Tuple[int, int]
    ) -> Optional[np.ndarray]:
        """
        Compute correction needed to align edges in overlap region.
        
        Returns (dx, dy) correction or None if edges already align well.
        """
        if size1 is None or size2 is None:
            return None
        
        # Compute bounding boxes in global coordinates
        w1, h1 = size1
        w2, h2 = size2
        
        # Image 1 bbox
        x1_min, y1_min = transform1[0, 2], transform1[1, 2]
        x1_max, y1_max = x1_min + w1, y1_min + h1
        
        # Image 2 bbox
        x2_min, y2_min = transform2[0, 2], transform2[1, 2]
        x2_max, y2_max = x2_min + w2, y2_min + h2
        
        # Compute overlap
        overlap_x_min = max(x1_min, x2_min)
        overlap_x_max = min(x1_max, x2_max)
        overlap_y_min = max(y1_min, y2_min)
        overlap_y_max = min(y1_max, y2_max)
        
        overlap_w = overlap_x_max - overlap_x_min
        overlap_h = overlap_y_max - overlap_y_min
        
        # Need at least 50px overlap
        if overlap_w < 50 or overlap_h < 50:
            return None
        
        # Extract overlap regions from both images
        # Local coordinates in image 1
        local1_x = int(overlap_x_min - x1_min)
        local1_y = int(overlap_y_min - y1_min)
        local1_w = int(min(overlap_w, w1 - local1_x))
        local1_h = int(min(overlap_h, h1 - local1_y))
        
        # Local coordinates in image 2
        local2_x = int(overlap_x_min - x2_min)
        local2_y = int(overlap_y_min - y2_min)
        local2_w = int(min(overlap_w, w2 - local2_x))
        local2_h = int(min(overlap_h, h2 - local2_y))
        
        # Ensure valid regions
        if local1_w < 50 or local1_h < 50 or local2_w < 50 or local2_h < 50:
            return None
        
        # Extract regions
        region1 = img1[local1_y:local1_y + local1_h, local1_x:local1_x + local1_w]
        region2 = img2[local2_y:local2_y + local2_h, local2_x:local2_x + local2_w]
        
        # Ensure same size for comparison
        min_w = min(region1.shape[1], region2.shape[1])
        min_h = min(region1.shape[0], region2.shape[0])
        region1 = region1[:min_h, :min_w]
        region2 = region2[:min_h, :min_w]
        
        if region1.size == 0 or region2.size == 0:
            return None
        
        # Convert to grayscale
        if len(region1.shape) == 3:
            gray1 = cv2.cvtColor(region1, cv2.COLOR_BGR2GRAY)
        else:
            gray1 = region1
            
        if len(region2.shape) == 3:
            gray2 = cv2.cvtColor(region2, cv2.COLOR_BGR2GRAY)
        else:
            gray2 = region2
        
        # Detect edges
        edges1 = cv2.Canny(gray1, 50, 150)
        edges2 = cv2.Canny(gray2, 50, 150)
        
        # Use phase correlation to find sub-pixel shift
        try:
            # Convert to float for phase correlation
            f1 = np.float32(edges1)
            f2 = np.float32(edges2)
            
            # Phase correlation
            shift, response = cv2.phaseCorrelate(f1, f2)
            
            # Only apply if response is strong (good match)
            if response > 0.3:
                # Limit correction to small values (max 10px)
                dx = np.clip(shift[0], -10, 10)
                dy = np.clip(shift[1], -10, 10)
                
                if abs(dx) > 0.5 or abs(dy) > 0.5:
                    return np.array([dx, dy])
        except Exception:
            pass
        
        return None
    
    def _detect_and_fix_outliers(
        self,
        transforms: Dict[int, np.ndarray],
        relative_transforms: Dict[Tuple[int, int], np.ndarray],
        n_images: int
    ) -> Dict[int, np.ndarray]:
        """
        Detect images that are placed inconsistently with their sequential neighbors
        and attempt to fix them.
        """
        positions = {i: (t[0, 2], t[1, 2]) for i, t in transforms.items()}
        
        # Calculate expected spacing from sequential neighbors
        sequential_distances = []
        for i in range(n_images - 1):
            if i in positions and i + 1 in positions:
                dist = np.sqrt((positions[i+1][0] - positions[i][0])**2 + 
                              (positions[i+1][1] - positions[i][1])**2)
                sequential_distances.append(dist)
        
        if len(sequential_distances) < 3:
            return transforms
        
        median_distance = np.median(sequential_distances)
        logger.info(f"Median sequential distance: {median_distance:.0f}px")
        
        # Find outliers (images far from both sequential neighbors)
        outliers = []
        for i in range(1, n_images - 1):
            if i not in positions:
                continue
            
            prev_ok = i - 1 not in positions
            next_ok = i + 1 not in positions
            
            if i - 1 in positions:
                dist_prev = np.sqrt((positions[i][0] - positions[i-1][0])**2 + 
                                   (positions[i][1] - positions[i-1][1])**2)
                prev_ok = dist_prev < median_distance * 4
            
            if i + 1 in positions:
                dist_next = np.sqrt((positions[i][0] - positions[i+1][0])**2 + 
                                   (positions[i][1] - positions[i+1][1])**2)
                next_ok = dist_next < median_distance * 4
            
            if not prev_ok and not next_ok:
                outliers.append(i)
        
        # Try to fix outliers by re-placing them based on sequential transforms
        for i in outliers:
            logger.warning(f"Image {i} is an outlier, attempting to re-place")
            
            # Try sequential placement
            for offset in [-1, 1]:
                neighbor = i + offset
                if neighbor in transforms:
                    key = (i, neighbor)
                    if key in relative_transforms:
                        neighbor_transform = transforms[neighbor]
                        rel_transform = relative_transforms[key]
                        new_transform = neighbor_transform @ rel_transform
                        
                        new_x, new_y = new_transform[0, 2], new_transform[1, 2]
                        old_x, old_y = transforms[i][0, 2], transforms[i][1, 2]
                        
                        logger.info(f"Image {i}: repositioned from ({old_x:.0f}, {old_y:.0f}) to ({new_x:.0f}, {new_y:.0f})")
                        transforms[i] = new_transform
                        break
        
        return transforms
    
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
        """Warp image using similarity transform with size limits"""
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
        
        # Check if output would be too large (only if limit is set)
        total_pixels = output_w * output_h
        if self.max_warp_pixels and total_pixels > self.max_warp_pixels:
            # Scale down to fit within limit
            scale = np.sqrt(self.max_warp_pixels / total_pixels)
            logger.warning(f"Warped image would be too large ({output_w}x{output_h}), scaling by {scale:.2f}")
            output_w = int(output_w * scale)
            output_h = int(output_h * scale)
            # Adjust bounds accordingly
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
        """Place images in a grid when no matches found (fallback)"""
        logger.warning("No valid transforms - placing images in grid layout")
        aligned_images = []
        
        # Calculate grid layout
        n_images = len(images_data)
        cols = int(np.ceil(np.sqrt(n_images)))
        
        # Get max dimensions for spacing
        max_w = max(img['image'].shape[1] for img in images_data) if images_data else 100
        max_h = max(img['image'].shape[0] for img in images_data) if images_data else 100
        spacing = 50
        
        for i, img_data in enumerate(images_data):
            h, w = img_data['image'].shape[:2]
            
            # Calculate grid position
            row = i // cols
            col = i % cols
            x = col * (max_w + spacing)
            y = row * (max_h + spacing)
            
            aligned_data = img_data.copy()
            aligned_data['transform'] = np.array([
                [1, 0, x],
                [0, 1, y],
                [0, 0, 1]
            ], dtype=np.float32)
            aligned_data['bbox'] = (x, y, x + w, y + h)
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
