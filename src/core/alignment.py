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
        Estimate similarity transform from pts1 to pts2
        
        Similarity transform has 4 DOF: translation (tx, ty), uniform scale (s), rotation (θ)
        Matrix form: [[s*cos(θ), -s*sin(θ), tx],
                      [s*sin(θ),  s*cos(θ), ty],
                      [0,         0,         1]]
        """
        if len(pts1) < 2:
            return None
        
        # Use OpenCV's estimateAffinePartial2D for similarity transform estimation
        similarity, inliers = cv2.estimateAffinePartial2D(
            pts1, pts2,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            confidence=0.995,
            maxIters=2000
        )
        
        if similarity is None:
            logger.warning("Failed to estimate similarity transform")
            return None
        
        # Validate transform
        scale = np.sqrt(similarity[0, 0]**2 + similarity[0, 1]**2)
        rotation = np.arctan2(similarity[1, 0], similarity[0, 0])
        
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
        
        # Rotation is allowed - no limits (images may be rotated)
        
        # Check inlier ratio
        if inliers is not None:
            inlier_ratio = np.sum(inliers) / len(pts1)
            if inlier_ratio < 0.3:
                logger.warning(f"Low inlier ratio {inlier_ratio:.1%}, rejecting transform")
                return None
        
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
        """Build connectivity graph from matches"""
        graph = {i: [] for i in range(len(images_data))}
        
        for match in matches:
            i = match['image_i']
            j = match['image_j']
            if match['confidence'] > 0.1 and match.get('num_matches', 0) >= 10:
                if j not in graph[i]:
                    graph[i].append(j)
                if i not in graph[j]:
                    graph[j].append(i)
        
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
        Calculate absolute transforms using BFS from reference image
        
        Each transform maps from the image's local coordinates to the 
        global (reference) coordinate system.
        """
        # Reference image has identity transform
        transforms = {ref_idx: np.eye(3, dtype=np.float64)}
        visited = {ref_idx}
        queue = [ref_idx]
        
        while queue:
            current = queue.pop(0)
            current_transform = transforms[current]
            
            for neighbor in graph.get(current, []):
                if neighbor in visited:
                    continue
                
                # Get transform from neighbor to current
                # We want: T_neighbor_to_global = T_current_to_global @ T_neighbor_to_current
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
                    logger.info(f"Image {neighbor}: position=({tx:.1f}, {ty:.1f}), scale={scale:.3f}")
        
        # Handle disconnected images
        unplaced = [i for i in range(n_images) if i not in transforms]
        if unplaced:
            logger.warning(f"Images {unplaced} are disconnected, placing separately")
            max_x = max(t[0, 2] for t in transforms.values()) if transforms else 0
            for i, img_idx in enumerate(unplaced):
                transforms[img_idx] = np.array([
                    [1, 0, max_x + 500 + i * 500],
                    [0, 1, 0],
                    [0, 0, 1]
                ], dtype=np.float64)
        
        # Check if layout is too linear (all on one row) and needs 2D correction
        transforms = self._fix_linear_layout(transforms, n_images)
        
        return transforms
    
    def _fix_linear_layout(
        self,
        transforms: Dict[int, np.ndarray],
        n_images: int,
        images_data: List[Dict] = None
    ) -> Dict[int, np.ndarray]:
        """
        Detect and fix linear layouts (all images on one horizontal line).
        
        For burst photos scanned in rows, the feature matching often only finds
        horizontal connections (1-2-3-4...) missing the vertical/row transitions.
        This detects that pattern and reorganizes into a 2D grid WITH PROPER OVERLAP.
        """
        if len(transforms) < 4:
            return transforms
        
        # Get all positions
        positions = [(i, transforms[i][0, 2], transforms[i][1, 2]) for i in sorted(transforms.keys())]
        
        y_values = [p[2] for p in positions]
        x_values = [p[1] for p in positions]
        
        y_range = max(y_values) - min(y_values) if y_values else 0
        x_range = max(x_values) - min(x_values) if x_values else 1
        
        # If Y range is very small compared to X range, layout is linear
        if x_range > 0 and y_range / max(x_range, 1) < 0.15:
            logger.warning(f"Detected LINEAR layout: Y range={y_range:.0f}, X range={x_range:.0f}")
            logger.info("Reorganizing into 2D grid with overlap...")
            
            # Estimate grid dimensions
            grid_cols = int(np.ceil(np.sqrt(n_images * 1.5)))  # Wider than tall
            grid_rows = int(np.ceil(n_images / grid_cols))
            
            logger.info(f"Creating {grid_rows} rows x {grid_cols} columns grid")
            
            # Calculate the actual horizontal spacing from feature matches
            # This tells us how much images actually overlap
            x_sorted = sorted(positions, key=lambda p: p[1])
            if len(x_sorted) > 1:
                spacings = []
                for i in range(min(len(x_sorted) - 1, grid_cols * 2)):
                    spacing = x_sorted[i + 1][1] - x_sorted[i][1]
                    if spacing > 0:
                        spacings.append(spacing)
                horizontal_spacing = np.median(spacings) if spacings else 1000
            else:
                horizontal_spacing = 1000
            
            # The horizontal_spacing represents the actual offset between images
            # This already accounts for overlap! We should use the same spacing vertically
            # to maintain consistent overlap in both directions
            vertical_spacing = horizontal_spacing  # Same overlap ratio for Y
            
            logger.info(f"Using spacing: {horizontal_spacing:.0f}px horizontal, {vertical_spacing:.0f}px vertical")
            
            # Reorganize positions in 2D grid with proper overlap
            new_transforms = {}
            sorted_by_idx = sorted(positions, key=lambda p: p[0])
            
            for seq_idx, (img_idx, old_x, old_y) in enumerate(sorted_by_idx):
                row = seq_idx // grid_cols
                col = seq_idx % grid_cols
                
                # Snake pattern: alternate direction each row
                if row % 2 == 1:
                    col = grid_cols - 1 - col
                
                # Use the feature-derived spacing (which accounts for overlap)
                new_x = col * horizontal_spacing
                new_y = row * vertical_spacing
                
                # Preserve rotation/scale from original transform
                old_transform = transforms[img_idx]
                new_transform = old_transform.copy()
                new_transform[0, 2] = new_x
                new_transform[1, 2] = new_y
                new_transforms[img_idx] = new_transform
                
                if seq_idx < 8 or seq_idx >= n_images - 2:
                    logger.debug(f"Image {img_idx}: grid[{row},{col}] -> ({new_x:.0f}, {new_y:.0f})")
            
            return new_transforms
        
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
