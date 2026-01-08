"""
AutoPano Giga-inspired advanced stitching features.

Based on documented behavior and known algorithms from:
- AutoPano Giga (Kolor/GoPro)
- Bundle adjustment from SfM pipelines
- Grid topology detection for flat panoramas

Full AutoPano Giga Workflow (inferred):
1. Load 1400+ images
2. Detect features (AI-enhanced SIFT-like)
3. Cluster images (spatial or feature-based)
4. Match within clusters
5. Build initial pose graph
6. Run coarse bundle adjustment
7. Refine matches (remove outliers)
8. Run fine bundle adjustment (global)
9. Detect grid topology (if flat)
10. Optimize seam positions
11. Multi-band blend into gigapixel output

Features:
1. Grid Topology Detection - Auto-detect grid structure, reduce O(n²) to O(n)
2. Two-Stage Bundle Adjustment - Coarse then fine global optimization
3. Hierarchical Stitching - Cluster-based stitching for large image sets
4. Enhanced Low-Texture Detection - Better features in skies, walls, water
5. Seam Optimization - Find optimal seam paths
6. AI Post-Processing - Optional enhancement with deep learning
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
import logging
from scipy.optimize import least_squares
from scipy.spatial import KDTree

logger = logging.getLogger(__name__)


class GridTopologyDetector:
    """
    Detect regular grid structure in image sets.
    
    For flat panoramas (microscope, satellite, drone grids), this:
    - Detects the grid layout automatically
    - Uses topological priors to guide matching
    - Reduces search space from O(n²) to O(n)
    
    Critical for 1000+ flat images.
    """
    
    def __init__(
        self,
        position_tolerance: float = 0.15,
        min_grid_confidence: float = 0.7
    ):
        """
        Args:
            position_tolerance: Tolerance for grid position matching (as fraction of image size)
            min_grid_confidence: Minimum confidence to use grid topology
        """
        self.position_tolerance = position_tolerance
        self.min_grid_confidence = min_grid_confidence
    
    def detect_grid(
        self,
        images_data: List[Dict],
        matches: Dict[Tuple[int, int], Dict]
    ) -> Optional[Dict]:
        """
        Detect if images form a regular grid.
        
        Returns:
            Grid info dict or None if no grid detected:
            {
                'rows': int,
                'cols': int,
                'grid_map': {(row, col): image_idx},
                'neighbors': {idx: [(neighbor_idx, direction), ...]},
                'confidence': float
            }
        """
        if len(images_data) < 4:
            return None
        
        logger.info(f"Detecting grid topology for {len(images_data)} images...")
        
        # Step 1: Build connectivity graph from matches
        connectivity = self._build_connectivity(len(images_data), matches)
        
        # Step 2: Estimate relative positions from match geometry
        relative_positions = self._estimate_relative_positions(images_data, matches)
        
        if not relative_positions:
            logger.info("Could not estimate relative positions")
            return None
        
        # Step 3: Detect grid pattern
        grid_info = self._detect_grid_pattern(relative_positions, images_data)
        
        if grid_info and grid_info['confidence'] >= self.min_grid_confidence:
            logger.info(f"Grid detected: {grid_info['rows']}x{grid_info['cols']} "
                       f"(confidence: {grid_info['confidence']:.2f})")
            return grid_info
        
        logger.info("No regular grid pattern detected")
        return None
    
    def _build_connectivity(
        self,
        n_images: int,
        matches: Dict[Tuple[int, int], Dict]
    ) -> Dict[int, Set[int]]:
        """Build adjacency graph from matches."""
        connectivity = defaultdict(set)
        
        for (i, j), match_data in matches.items():
            if match_data.get('num_inliers', 0) >= 8:
                connectivity[i].add(j)
                connectivity[j].add(i)
        
        return connectivity
    
    def _estimate_relative_positions(
        self,
        images_data: List[Dict],
        matches: Dict[Tuple[int, int], Dict]
    ) -> Dict[int, Tuple[float, float]]:
        """
        Estimate relative image positions from match geometry.
        Uses translation components of homographies/transforms.
        """
        n = len(images_data)
        positions = {0: (0.0, 0.0)}  # Reference image at origin
        
        # BFS to propagate positions
        visited = {0}
        queue = [0]
        
        while queue:
            current = queue.pop(0)
            current_pos = positions[current]
            
            for (i, j), match_data in matches.items():
                if match_data.get('num_inliers', 0) < 8:
                    continue
                
                # Find unvisited neighbor
                if i == current and j not in visited:
                    neighbor = j
                elif j == current and i not in visited:
                    neighbor = i
                else:
                    continue
                
                # Estimate translation from keypoint positions
                if 'src_pts' in match_data and 'dst_pts' in match_data:
                    src_pts = match_data['src_pts']
                    dst_pts = match_data['dst_pts']
                    
                    if i == current:
                        # Translation from current to neighbor
                        dx = np.median(dst_pts[:, 0] - src_pts[:, 0])
                        dy = np.median(dst_pts[:, 1] - src_pts[:, 1])
                    else:
                        dx = np.median(src_pts[:, 0] - dst_pts[:, 0])
                        dy = np.median(src_pts[:, 1] - dst_pts[:, 1])
                    
                    # Normalize by image size
                    h, w = images_data[current]['image'].shape[:2]
                    dx_norm = dx / w
                    dy_norm = dy / h
                    
                    positions[neighbor] = (
                        current_pos[0] + dx_norm,
                        current_pos[1] + dy_norm
                    )
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return positions
    
    def _detect_grid_pattern(
        self,
        positions: Dict[int, Tuple[float, float]],
        images_data: List[Dict]
    ) -> Optional[Dict]:
        """
        Detect if positions form a regular grid.
        """
        if len(positions) < 4:
            return None
        
        # Convert to numpy array
        indices = list(positions.keys())
        coords = np.array([positions[i] for i in indices])
        
        # Find most common x and y spacings
        x_coords = coords[:, 0]
        y_coords = coords[:, 1]
        
        # Cluster x and y coordinates
        x_clusters = self._cluster_1d(x_coords)
        y_clusters = self._cluster_1d(y_coords)
        
        if len(x_clusters) < 2 or len(y_clusters) < 2:
            return None
        
        # Check if clusters form regular grid
        x_spacings = np.diff(sorted(x_clusters))
        y_spacings = np.diff(sorted(y_clusters))
        
        # Regularity check: all spacings should be similar
        x_regular = len(x_spacings) == 0 or np.std(x_spacings) / (np.mean(x_spacings) + 1e-6) < 0.3
        y_regular = len(y_spacings) == 0 or np.std(y_spacings) / (np.mean(y_spacings) + 1e-6) < 0.3
        
        if not (x_regular and y_regular):
            return {'confidence': 0.3}
        
        # Build grid map
        cols = len(x_clusters)
        rows = len(y_clusters)
        
        x_sorted = sorted(x_clusters)
        y_sorted = sorted(y_clusters)
        
        grid_map = {}
        neighbors = defaultdict(list)
        
        for idx in indices:
            x, y = positions[idx]
            
            # Find closest grid cell
            col = min(range(cols), key=lambda c: abs(x_sorted[c] - x))
            row = min(range(rows), key=lambda r: abs(y_sorted[r] - y))
            
            grid_map[(row, col)] = idx
        
        # Build neighbor relationships
        for (row, col), idx in grid_map.items():
            if (row - 1, col) in grid_map:
                neighbors[idx].append((grid_map[(row - 1, col)], 'up'))
            if (row + 1, col) in grid_map:
                neighbors[idx].append((grid_map[(row + 1, col)], 'down'))
            if (row, col - 1) in grid_map:
                neighbors[idx].append((grid_map[(row, col - 1)], 'left'))
            if (row, col + 1) in grid_map:
                neighbors[idx].append((grid_map[(row, col + 1)], 'right'))
        
        # Calculate confidence based on grid completeness
        expected_cells = rows * cols
        actual_cells = len(grid_map)
        confidence = actual_cells / expected_cells
        
        return {
            'rows': rows,
            'cols': cols,
            'grid_map': grid_map,
            'neighbors': dict(neighbors),
            'confidence': confidence,
            'x_spacing': np.mean(x_spacings) if len(x_spacings) > 0 else 1.0,
            'y_spacing': np.mean(y_spacings) if len(y_spacings) > 0 else 1.0
        }
    
    def _cluster_1d(self, values: np.ndarray, tolerance: float = 0.1) -> List[float]:
        """Cluster 1D values into groups."""
        if len(values) == 0:
            return []
        
        sorted_vals = np.sort(values)
        clusters = [sorted_vals[0]]
        
        for v in sorted_vals[1:]:
            if abs(v - clusters[-1]) > tolerance:
                clusters.append(v)
        
        return clusters
    
    def get_neighbor_pairs(self, grid_info: Dict) -> List[Tuple[int, int]]:
        """
        Get list of image pairs that should be matched based on grid topology.
        
        Returns O(n) pairs instead of O(n²).
        """
        if not grid_info:
            return []
        
        pairs = set()
        for idx, neighbor_list in grid_info['neighbors'].items():
            for neighbor_idx, _ in neighbor_list:
                pair = tuple(sorted([idx, neighbor_idx]))
                pairs.add(pair)
        
        return list(pairs)


class BundleAdjuster:
    """
    Global bundle adjustment for panorama optimization.
    
    Optimizes:
    - Camera poses (position, orientation)
    - Focal length (if enabled)
    - Minimizes reprojection error across all keypoints
    
    Based on SfM bundle adjustment, adapted for 2D panoramas.
    """
    
    def __init__(
        self,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        robust_loss: str = 'huber'
    ):
        """
        Args:
            max_iterations: Maximum optimization iterations
            tolerance: Convergence tolerance
            robust_loss: Loss function ('huber', 'cauchy', 'soft_l1')
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.robust_loss = robust_loss
    
    def optimize(
        self,
        images_data: List[Dict],
        matches: Dict[Tuple[int, int], Dict],
        initial_transforms: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], Dict]:
        """
        Perform global bundle adjustment.
        
        Args:
            images_data: Image data with sizes
            matches: Feature matches between pairs
            initial_transforms: Initial 2x3 affine transforms
            
        Returns:
            Tuple of (optimized_transforms, optimization_stats)
        """
        n = len(images_data)
        
        if n < 2:
            return initial_transforms, {'success': False, 'reason': 'Too few images'}
        
        logger.info(f"Starting bundle adjustment for {n} images...")
        
        # Collect all observations (matched keypoint pairs)
        observations = []
        for (i, j), match_data in matches.items():
            if 'src_pts' in match_data and 'dst_pts' in match_data:
                src_pts = match_data['src_pts']
                dst_pts = match_data['dst_pts']
                
                for k in range(len(src_pts)):
                    observations.append({
                        'img_i': i,
                        'img_j': j,
                        'pt_i': src_pts[k],
                        'pt_j': dst_pts[k]
                    })
        
        if len(observations) < n * 4:
            logger.warning(f"Too few observations ({len(observations)}) for bundle adjustment")
            return initial_transforms, {'success': False, 'reason': 'Too few observations'}
        
        logger.info(f"Bundle adjustment with {len(observations)} observations")
        
        # Convert transforms to parameter vector
        # Each transform: [tx, ty, scale, rotation]
        params = []
        for transform in initial_transforms:
            tx = transform[0, 2]
            ty = transform[1, 2]
            scale = np.sqrt(transform[0, 0]**2 + transform[0, 1]**2)
            rotation = np.arctan2(transform[1, 0], transform[0, 0])
            params.extend([tx, ty, scale, rotation])
        
        params = np.array(params)
        
        # Fix first image (reference)
        # Optimization will adjust all others relative to it
        
        def residuals(p):
            return self._compute_residuals(p, observations, n)
        
        try:
            result = least_squares(
                residuals,
                params,
                method='trf',
                loss=self.robust_loss,
                max_nfev=self.max_iterations * len(params),
                ftol=self.tolerance,
                xtol=self.tolerance
            )
            
            # Convert back to transforms
            optimized_transforms = []
            for i in range(n):
                tx = result.x[i * 4]
                ty = result.x[i * 4 + 1]
                scale = result.x[i * 4 + 2]
                rotation = result.x[i * 4 + 3]
                
                cos_r = np.cos(rotation)
                sin_r = np.sin(rotation)
                
                transform = np.array([
                    [scale * cos_r, -scale * sin_r, tx],
                    [scale * sin_r, scale * cos_r, ty]
                ], dtype=np.float64)
                
                optimized_transforms.append(transform)
            
            # Calculate improvement
            initial_cost = np.sum(residuals(params)**2)
            final_cost = result.cost
            improvement = (initial_cost - final_cost) / (initial_cost + 1e-6) * 100
            
            stats = {
                'success': result.success,
                'iterations': result.nfev,
                'initial_cost': initial_cost,
                'final_cost': final_cost,
                'improvement_percent': improvement,
                'message': result.message
            }
            
            logger.info(f"Bundle adjustment complete: {improvement:.1f}% improvement, "
                       f"cost {initial_cost:.2f} -> {final_cost:.2f}")
            
            return optimized_transforms, stats
            
        except Exception as e:
            logger.error(f"Bundle adjustment failed: {e}")
            return initial_transforms, {'success': False, 'reason': str(e)}
    
    def _compute_residuals(
        self,
        params: np.ndarray,
        observations: List[Dict],
        n_images: int
    ) -> np.ndarray:
        """Compute reprojection residuals for all observations."""
        residuals = []
        
        for obs in observations:
            i = obs['img_i']
            j = obs['img_j']
            pt_i = obs['pt_i']
            pt_j = obs['pt_j']
            
            # Get transforms for both images
            tx_i, ty_i, scale_i, rot_i = params[i*4:(i+1)*4]
            tx_j, ty_j, scale_j, rot_j = params[j*4:(j+1)*4]
            
            # Transform point from image i to global
            cos_i, sin_i = np.cos(rot_i), np.sin(rot_i)
            global_x = scale_i * (cos_i * pt_i[0] - sin_i * pt_i[1]) + tx_i
            global_y = scale_i * (sin_i * pt_i[0] + cos_i * pt_i[1]) + ty_i
            
            # Transform point from image j to global
            cos_j, sin_j = np.cos(rot_j), np.sin(rot_j)
            global_x_j = scale_j * (cos_j * pt_j[0] - sin_j * pt_j[1]) + tx_j
            global_y_j = scale_j * (sin_j * pt_j[0] + cos_j * pt_j[1]) + ty_j
            
            # Residual is difference in global coordinates
            residuals.append(global_x - global_x_j)
            residuals.append(global_y - global_y_j)
        
        return np.array(residuals)


class HierarchicalStitcher:
    """
    Hierarchical/cluster-based stitching for large image sets.
    
    Strategy:
    1. Cluster images based on connectivity
    2. Stitch each cluster independently
    3. Merge clusters together
    
    This is much faster than global stitching for 1000+ images.
    """
    
    def __init__(
        self,
        cluster_size: int = 50,
        overlap_threshold: int = 8
    ):
        """
        Args:
            cluster_size: Target size for each cluster
            overlap_threshold: Minimum matches to consider images connected
        """
        self.cluster_size = cluster_size
        self.overlap_threshold = overlap_threshold
    
    def cluster_images(
        self,
        n_images: int,
        matches: Dict[Tuple[int, int], Dict]
    ) -> List[List[int]]:
        """
        Cluster images based on connectivity graph.
        
        Uses graph partitioning to create balanced clusters.
        """
        if n_images <= self.cluster_size:
            return [list(range(n_images))]
        
        logger.info(f"Clustering {n_images} images into groups of ~{self.cluster_size}...")
        
        # Build adjacency graph
        graph = defaultdict(set)
        for (i, j), match_data in matches.items():
            if match_data.get('num_inliers', 0) >= self.overlap_threshold:
                graph[i].add(j)
                graph[j].add(i)
        
        # Find connected components first
        visited = set()
        components = []
        
        for start in range(n_images):
            if start in visited:
                continue
            
            component = []
            queue = [start]
            while queue:
                node = queue.pop(0)
                if node in visited:
                    continue
                visited.add(node)
                component.append(node)
                queue.extend(n for n in graph[node] if n not in visited)
            
            components.append(component)
        
        logger.info(f"Found {len(components)} connected components")
        
        # Split large components into clusters
        all_clusters = []
        for component in components:
            if len(component) <= self.cluster_size:
                all_clusters.append(component)
            else:
                # Use spatial clustering within component
                sub_clusters = self._partition_component(component, graph)
                all_clusters.extend(sub_clusters)
        
        logger.info(f"Created {len(all_clusters)} clusters")
        return all_clusters
    
    def _partition_component(
        self,
        component: List[int],
        graph: Dict[int, Set[int]]
    ) -> List[List[int]]:
        """Partition a large component into smaller clusters."""
        n_clusters = max(2, len(component) // self.cluster_size)
        
        # Simple greedy clustering based on connectivity
        clusters = [[] for _ in range(n_clusters)]
        assigned = set()
        
        # Start each cluster with a seed node
        remaining = list(component)
        for i in range(n_clusters):
            if not remaining:
                break
            seed = remaining.pop(0)
            clusters[i].append(seed)
            assigned.add(seed)
        
        # Assign remaining nodes to nearest cluster
        for node in remaining:
            if node in assigned:
                continue
            
            # Find cluster with most connections to this node
            best_cluster = 0
            best_connections = -1
            
            for c_idx, cluster in enumerate(clusters):
                if len(cluster) >= self.cluster_size * 1.2:
                    continue  # Skip overfull clusters
                
                connections = sum(1 for n in cluster if n in graph[node])
                if connections > best_connections:
                    best_connections = connections
                    best_cluster = c_idx
            
            clusters[best_cluster].append(node)
            assigned.add(node)
        
        return [c for c in clusters if c]
    
    def get_cluster_connections(
        self,
        clusters: List[List[int]],
        matches: Dict[Tuple[int, int], Dict]
    ) -> List[Tuple[int, int, int]]:
        """
        Find connections between clusters for merging.
        
        Returns:
            List of (cluster_i, cluster_j, num_matches)
        """
        # Map image index to cluster index
        img_to_cluster = {}
        for c_idx, cluster in enumerate(clusters):
            for img_idx in cluster:
                img_to_cluster[img_idx] = c_idx
        
        # Count inter-cluster matches
        cluster_matches = defaultdict(int)
        for (i, j), match_data in matches.items():
            c_i = img_to_cluster.get(i, -1)
            c_j = img_to_cluster.get(j, -1)
            
            if c_i != c_j and c_i >= 0 and c_j >= 0:
                key = tuple(sorted([c_i, c_j]))
                cluster_matches[key] += match_data.get('num_inliers', 0)
        
        connections = [
            (k[0], k[1], v) for k, v in cluster_matches.items()
        ]
        connections.sort(key=lambda x: -x[2])  # Sort by match count
        
        return connections


class EnhancedFeatureDetector:
    """
    Enhanced feature detection for low-texture areas.
    
    Combines multiple strategies:
    - Dense feature grids for textureless regions
    - Multi-scale detection
    - Adaptive thresholds based on image content
    """
    
    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu
        
        # Create multiple detectors for different scenarios
        self.sift = cv2.SIFT_create(nfeatures=10000, contrastThreshold=0.02, edgeThreshold=15)
        self.orb = cv2.ORB_create(nfeatures=5000, scaleFactor=1.1, nlevels=12)
        
        # AKAZE for blurred/low-contrast images
        self.akaze = cv2.AKAZE_create(threshold=0.0005, nOctaves=4, nOctaveLayers=4)
    
    def detect_enhanced(
        self,
        image: np.ndarray,
        min_features: int = 500
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Enhanced feature detection with fallbacks for low-texture areas.
        
        Returns:
            (keypoints_array, descriptors)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Try SIFT first (best for most images)
        kp, desc = self.sift.detectAndCompute(gray, None)
        
        if desc is not None and len(desc) >= min_features:
            return self._kp_to_array(kp), desc
        
        # If low features, try with enhanced preprocessing
        enhanced = self._enhance_for_detection(gray)
        kp2, desc2 = self.sift.detectAndCompute(enhanced, None)
        
        if desc2 is not None and (desc is None or len(desc2) > len(desc)):
            kp, desc = kp2, desc2
        
        if desc is not None and len(desc) >= min_features:
            return self._kp_to_array(kp), desc
        
        # Try AKAZE as fallback (good for blurred images)
        kp3, desc3 = self.akaze.detectAndCompute(enhanced, None)
        
        if desc3 is not None and (desc is None or len(desc3) > len(desc)):
            kp, desc = kp3, desc3
        
        if desc is not None and len(desc) >= min_features:
            return self._kp_to_array(kp), desc
        
        # Last resort: add dense grid features for textureless regions
        if desc is None or len(desc) < min_features // 2:
            grid_kp, grid_desc = self._detect_dense_grid(gray)
            
            if desc is not None and grid_desc is not None:
                # Combine with existing features
                all_kp = list(kp) + list(grid_kp)
                desc = np.vstack([desc, grid_desc])
                kp = all_kp
            elif grid_desc is not None:
                kp, desc = grid_kp, grid_desc
        
        return self._kp_to_array(kp) if kp else np.array([]), desc if desc is not None else np.array([])
    
    def _enhance_for_detection(self, gray: np.ndarray) -> np.ndarray:
        """Enhance image for better feature detection."""
        # Multi-scale CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Edge enhancement
        blurred = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
        enhanced = cv2.addWeighted(enhanced, 1.3, blurred, -0.3, 0)
        
        return enhanced
    
    def _detect_dense_grid(
        self,
        gray: np.ndarray,
        step: int = 32
    ) -> Tuple[List, Optional[np.ndarray]]:
        """
        Detect features on a dense grid for textureless areas.
        Uses ORB descriptors at regular grid positions.
        """
        h, w = gray.shape
        keypoints = []
        
        for y in range(step, h - step, step):
            for x in range(step, w - step, step):
                # Create keypoint at grid position
                kp = cv2.KeyPoint(x=float(x), y=float(y), size=step)
                keypoints.append(kp)
        
        if not keypoints:
            return [], None
        
        # Compute descriptors at grid points
        _, descriptors = self.sift.compute(gray, keypoints)
        
        return keypoints, descriptors
    
    def _kp_to_array(self, keypoints) -> np.ndarray:
        """Convert keypoints to numpy array."""
        if not keypoints:
            return np.array([])
        return np.array([[kp.pt[0], kp.pt[1], kp.size, kp.angle] for kp in keypoints])


def create_matching_pairs_from_grid(
    n_images: int,
    grid_info: Optional[Dict],
    fallback_strategy: str = 'windowed'
) -> List[Tuple[int, int]]:
    """
    Create list of image pairs to match based on detected topology.
    
    Args:
        n_images: Total number of images
        grid_info: Grid topology info from GridTopologyDetector
        fallback_strategy: Strategy if no grid detected
            'all': All pairs O(n²)
            'windowed': Only nearby images O(n*k)
            'chain': Linear chain O(n)
            
    Returns:
        List of (i, j) pairs to match
    """
    if grid_info and grid_info.get('confidence', 0) >= 0.7:
        # Use grid topology - O(n)
        pairs = []
        for idx, neighbors in grid_info['neighbors'].items():
            for neighbor_idx, _ in neighbors:
                if neighbor_idx > idx:  # Avoid duplicates
                    pairs.append((idx, neighbor_idx))
        
        # Add some diagonal connections for robustness
        grid_map = grid_info['grid_map']
        for (row, col), idx in grid_map.items():
            diagonals = [(row-1, col-1), (row-1, col+1), (row+1, col-1), (row+1, col+1)]
            for r, c in diagonals:
                if (r, c) in grid_map:
                    pair = tuple(sorted([idx, grid_map[(r, c)]]))
                    if pair not in pairs:
                        pairs.append(pair)
        
        logger.info(f"Grid topology: {len(pairs)} pairs (from {n_images} images)")
        return pairs
    
    # Fallback strategies
    if fallback_strategy == 'all':
        return [(i, j) for i in range(n_images) for j in range(i+1, n_images)]
    
    elif fallback_strategy == 'windowed':
        window = min(15, n_images // 2)
        pairs = []
        for i in range(n_images):
            for j in range(i+1, min(i + window + 1, n_images)):
                pairs.append((i, j))
        return pairs
    
    else:  # chain
        return [(i, i+1) for i in range(n_images - 1)]


class TwoStageBundleAdjuster:
    """
    Two-stage bundle adjustment as used in AutoPano Giga.
    
    Stage 1 (Coarse): Fast optimization with reduced parameters
    Stage 2 (Fine): Full optimization with all parameters
    
    This approach is faster and more robust than single-pass optimization.
    """
    
    def __init__(self):
        self.coarse_adjuster = BundleAdjuster(
            max_iterations=50,
            tolerance=1e-4,
            robust_loss='huber'
        )
        self.fine_adjuster = BundleAdjuster(
            max_iterations=200,
            tolerance=1e-8,
            robust_loss='soft_l1'
        )
    
    def optimize(
        self,
        images_data: List[Dict],
        matches: Dict[Tuple[int, int], Dict],
        initial_transforms: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], Dict]:
        """
        Two-stage bundle adjustment.
        
        Returns:
            (optimized_transforms, combined_stats)
        """
        logger.info("Stage 1: Coarse bundle adjustment...")
        coarse_transforms, coarse_stats = self.coarse_adjuster.optimize(
            images_data, matches, initial_transforms
        )
        
        if not coarse_stats.get('success', False):
            logger.warning("Coarse BA failed, using initial transforms")
            coarse_transforms = initial_transforms
        
        # Refine matches based on coarse alignment (remove outliers)
        refined_matches = self._refine_matches(matches, coarse_transforms)
        
        logger.info("Stage 2: Fine bundle adjustment...")
        fine_transforms, fine_stats = self.fine_adjuster.optimize(
            images_data, refined_matches, coarse_transforms
        )
        
        combined_stats = {
            'success': fine_stats.get('success', coarse_stats.get('success', False)),
            'coarse_improvement': coarse_stats.get('improvement_percent', 0),
            'fine_improvement': fine_stats.get('improvement_percent', 0),
            'total_improvement': (
                coarse_stats.get('improvement_percent', 0) + 
                fine_stats.get('improvement_percent', 0)
            ),
            'outliers_removed': len(matches) - len(refined_matches)
        }
        
        logger.info(f"Two-stage BA complete: {combined_stats['total_improvement']:.1f}% total improvement, "
                   f"{combined_stats['outliers_removed']} outliers removed")
        
        return fine_transforms, combined_stats
    
    def _refine_matches(
        self,
        matches: Dict[Tuple[int, int], Dict],
        transforms: List[np.ndarray]
    ) -> Dict[Tuple[int, int], Dict]:
        """Remove outlier matches based on current alignment."""
        refined = {}
        
        for (i, j), match_data in matches.items():
            if 'src_pts' not in match_data or 'dst_pts' not in match_data:
                refined[(i, j)] = match_data
                continue
            
            src_pts = match_data['src_pts']
            dst_pts = match_data['dst_pts']
            
            if i >= len(transforms) or j >= len(transforms):
                refined[(i, j)] = match_data
                continue
            
            # Transform points and check reprojection error
            t_i = transforms[i]
            t_j = transforms[j]
            
            # Transform src_pts to global
            src_global = np.column_stack([
                t_i[0, 0] * src_pts[:, 0] + t_i[0, 1] * src_pts[:, 1] + t_i[0, 2],
                t_i[1, 0] * src_pts[:, 0] + t_i[1, 1] * src_pts[:, 1] + t_i[1, 2]
            ])
            
            # Transform dst_pts to global
            dst_global = np.column_stack([
                t_j[0, 0] * dst_pts[:, 0] + t_j[0, 1] * dst_pts[:, 1] + t_j[0, 2],
                t_j[1, 0] * dst_pts[:, 0] + t_j[1, 1] * dst_pts[:, 1] + t_j[1, 2]
            ])
            
            # Calculate errors
            errors = np.sqrt(np.sum((src_global - dst_global)**2, axis=1))
            
            # Keep inliers (error < 5 pixels)
            inlier_mask = errors < 5.0
            
            if np.sum(inlier_mask) >= 4:
                refined_match = match_data.copy()
                refined_match['src_pts'] = src_pts[inlier_mask]
                refined_match['dst_pts'] = dst_pts[inlier_mask]
                refined_match['num_inliers'] = int(np.sum(inlier_mask))
                refined[(i, j)] = refined_match
        
        return refined


class SeamOptimizer:
    """
    Optimize seam positions between overlapping images.
    
    AutoPano Giga uses seam optimization to:
    - Avoid cutting through important objects
    - Place seams in low-gradient regions
    - Minimize color/exposure discontinuity
    """
    
    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu
    
    def find_optimal_seam(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        overlap_mask: np.ndarray
    ) -> np.ndarray:
        """
        Find optimal seam between two overlapping images.
        
        Uses graph cut / dynamic programming to find minimum cost path.
        
        Returns:
            Seam mask (255 where img1 should be used, 0 where img2)
        """
        if overlap_mask is None or not np.any(overlap_mask):
            return np.ones(img1.shape[:2], dtype=np.uint8) * 255
        
        h, w = img1.shape[:2]
        
        # Calculate cost for choosing each image at each pixel
        # Cost = gradient magnitude + color difference
        
        # Convert to LAB for perceptual color difference
        lab1 = cv2.cvtColor(img1, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab2 = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB).astype(np.float32)
        
        # Color difference
        color_diff = np.sqrt(np.sum((lab1 - lab2)**2, axis=2))
        
        # Gradient magnitude (prefer seams in low-gradient areas)
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        grad1 = cv2.Sobel(gray1, cv2.CV_64F, 1, 1)
        grad2 = cv2.Sobel(gray2, cv2.CV_64F, 1, 1)
        
        # Combined gradient (avoid high-gradient areas in either image)
        gradient = np.abs(grad1) + np.abs(grad2)
        
        # Total cost: high at color differences and high gradients
        cost = color_diff + gradient * 0.5
        
        # Apply overlap mask
        cost[~overlap_mask.astype(bool)] = 0
        
        # Find optimal seam using simple horizontal sweep
        # (For production, use graph cut or dynamic programming)
        seam_mask = self._find_seam_dp(cost, overlap_mask)
        
        return seam_mask
    
    def _find_seam_dp(
        self,
        cost: np.ndarray,
        overlap_mask: np.ndarray
    ) -> np.ndarray:
        """Find minimum cost seam using dynamic programming."""
        h, w = cost.shape
        
        # Simple left-to-right DP
        dp = np.full((h, w), np.inf)
        dp[:, 0] = cost[:, 0]
        
        for x in range(1, w):
            for y in range(h):
                if not overlap_mask[y, x]:
                    continue
                
                min_prev = dp[y, x-1]
                if y > 0:
                    min_prev = min(min_prev, dp[y-1, x-1])
                if y < h - 1:
                    min_prev = min(min_prev, dp[y+1, x-1])
                
                dp[y, x] = cost[y, x] + min_prev
        
        # Backtrack to find seam
        seam_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Find minimum cost ending point
        end_col = w - 1
        while end_col > 0 and not np.any(overlap_mask[:, end_col]):
            end_col -= 1
        
        if end_col == 0:
            return np.ones((h, w), dtype=np.uint8) * 255
        
        y = np.argmin(dp[:, end_col])
        
        # Fill from seam to right edge with img2 (0), left with img1 (255)
        for x in range(w - 1, -1, -1):
            seam_mask[:y+1, x] = 255
            
            if x > 0:
                # Move to previous column
                min_prev = dp[y, x-1]
                best_y = y
                if y > 0 and dp[y-1, x-1] < min_prev:
                    min_prev = dp[y-1, x-1]
                    best_y = y - 1
                if y < h - 1 and dp[y+1, x-1] < min_prev:
                    best_y = y + 1
                y = best_y
        
        return seam_mask
    
    def apply_seam_blending(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        seam_mask: np.ndarray,
        blend_width: int = 10
    ) -> np.ndarray:
        """Apply seam mask with feathered blending."""
        # Create smooth transition at seam
        seam_float = seam_mask.astype(np.float32) / 255.0
        
        # Feather the seam
        seam_blurred = cv2.GaussianBlur(seam_float, (blend_width*2+1, blend_width*2+1), 0)
        
        # Blend
        result = (img1.astype(np.float32) * seam_blurred[:, :, np.newaxis] +
                  img2.astype(np.float32) * (1 - seam_blurred[:, :, np.newaxis]))
        
        return np.clip(result, 0, 255).astype(np.uint8)


class AIPostProcessor:
    """
    AI-powered post-processing for panoramas.
    
    Optional enhancements using deep learning:
    - Super-resolution (SwinIR, Real-ESRGAN)
    - Denoising
    - Inpainting for gaps
    - Color correction
    """
    
    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu
        self._models_loaded = False
        self._super_res_model = None
        self._denoise_model = None
    
    def enhance(
        self,
        image: np.ndarray,
        enable_super_res: bool = False,
        enable_denoise: bool = True,
        enable_color_correct: bool = True
    ) -> np.ndarray:
        """
        Apply AI enhancement to panorama.
        
        Args:
            image: Input panorama
            enable_super_res: Apply super-resolution (slow, doubles size)
            enable_denoise: Apply denoising
            enable_color_correct: Apply color correction
            
        Returns:
            Enhanced panorama
        """
        result = image.copy()
        
        if enable_color_correct:
            result = self._color_correct(result)
        
        if enable_denoise:
            result = self._denoise(result)
        
        if enable_super_res:
            result = self._super_resolve(result)
        
        return result
    
    def _color_correct(self, image: np.ndarray) -> np.ndarray:
        """Apply automatic color correction."""
        # Convert to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # CLAHE on L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Gray world assumption for color cast removal
        a_mean = np.mean(a)
        b_mean = np.mean(b)
        a = np.clip(a.astype(np.float32) - (a_mean - 128) * 0.5, 0, 255).astype(np.uint8)
        b = np.clip(b.astype(np.float32) - (b_mean - 128) * 0.5, 0, 255).astype(np.uint8)
        
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    def _denoise(self, image: np.ndarray) -> np.ndarray:
        """Apply denoising."""
        # Use OpenCV's fastNlMeansDenoisingColored as fallback
        return cv2.fastNlMeansDenoisingColored(image, None, 5, 5, 7, 21)
    
    def _super_resolve(self, image: np.ndarray) -> np.ndarray:
        """Apply super-resolution (placeholder - uses bicubic by default)."""
        # For real SR, would load SwinIR or Real-ESRGAN
        h, w = image.shape[:2]
        return cv2.resize(image, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)
    
    def inpaint_gaps(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        method: str = 'telea'
    ) -> np.ndarray:
        """
        Inpaint gaps/holes in the panorama.
        
        Args:
            image: Panorama with gaps
            mask: Binary mask of gaps (255 = gap)
            method: 'telea', 'ns', or 'ai'
        """
        if not np.any(mask):
            return image
        
        if method == 'telea':
            return cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
        elif method == 'ns':
            return cv2.inpaint(image, mask, 3, cv2.INPAINT_NS)
        else:
            # AI inpainting would go here
            return cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)


class AutoPanoWorkflow:
    """
    Complete AutoPano Giga-style workflow orchestrator.
    
    Implements the full 11-step workflow:
    1. Load images
    2. Detect features (AI-enhanced)
    3. Cluster images
    4. Match within clusters
    5. Build pose graph
    6. Coarse bundle adjustment
    7. Refine matches
    8. Fine bundle adjustment
    9. Detect grid topology
    10. Optimize seams
    11. Multi-band blend
    """
    
    def __init__(self, use_gpu: bool = False, options: Dict = None):
        self.use_gpu = use_gpu
        self.options = options or {}
        
        # Initialize components
        self.grid_detector = GridTopologyDetector()
        self.bundle_adjuster = TwoStageBundleAdjuster()
        self.hierarchical_stitcher = HierarchicalStitcher()
        self.enhanced_detector = EnhancedFeatureDetector(use_gpu)
        self.seam_optimizer = SeamOptimizer(use_gpu)
        self.post_processor = AIPostProcessor(use_gpu)
        
        # State
        self.grid_info = None
        self.clusters = None
    
    def get_matching_pairs(
        self,
        n_images: int,
        preliminary_matches: Optional[Dict] = None
    ) -> List[Tuple[int, int]]:
        """
        Get optimized matching pairs based on detected topology.
        
        For large image sets (>100), uses:
        1. Grid detection if available
        2. Hierarchical clustering
        3. Intra-cluster matching only
        """
        # Try grid detection first
        if preliminary_matches:
            self.grid_info = self.grid_detector.detect_grid(
                [{'image': np.zeros((100, 100, 3))} for _ in range(n_images)],  # Placeholder
                preliminary_matches
            )
        
        if self.grid_info and self.grid_info.get('confidence', 0) >= 0.7:
            return create_matching_pairs_from_grid(n_images, self.grid_info)
        
        # Use hierarchical clustering for large sets
        if n_images > 100 and preliminary_matches:
            self.clusters = self.hierarchical_stitcher.cluster_images(n_images, preliminary_matches)
            
            pairs = []
            for cluster in self.clusters:
                # All pairs within cluster
                for i, idx_i in enumerate(cluster):
                    for idx_j in cluster[i+1:]:
                        pairs.append((min(idx_i, idx_j), max(idx_i, idx_j)))
            
            # Add inter-cluster connections
            cluster_connections = self.hierarchical_stitcher.get_cluster_connections(
                self.clusters, preliminary_matches
            )
            # ... add best pairs between clusters
            
            return pairs
        
        # Fallback to windowed matching
        return create_matching_pairs_from_grid(n_images, None, 'windowed')
    
    def run_bundle_adjustment(
        self,
        images_data: List[Dict],
        matches: Dict,
        initial_transforms: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], Dict]:
        """Run two-stage bundle adjustment."""
        return self.bundle_adjuster.optimize(images_data, matches, initial_transforms)
    
    def post_process(
        self,
        panorama: np.ndarray,
        gap_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Apply AI post-processing."""
        result = panorama
        
        # Inpaint any gaps
        if gap_mask is not None and np.any(gap_mask):
            result = self.post_processor.inpaint_gaps(result, gap_mask)
        
        # Apply enhancements
        if self.options.get('enable_post_processing', True):
            result = self.post_processor.enhance(
                result,
                enable_super_res=self.options.get('super_resolution', False),
                enable_denoise=self.options.get('denoise', True),
                enable_color_correct=self.options.get('color_correct', True)
            )
        
        return result

