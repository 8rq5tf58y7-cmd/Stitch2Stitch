"""
Image alignment engine with support for flat images and transparency masks
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ImageAligner:
    """Align images using homography estimation"""
    
    def __init__(self, use_gpu: bool = False):
        """
        Initialize aligner
        
        Args:
            use_gpu: Enable GPU acceleration
        """
        self.use_gpu = use_gpu
        logger.info("Image aligner initialized")
    
    def align_images(
        self,
        images_data: List[Dict],
        features_data: List[Dict],
        matches: List[Dict]
    ) -> List[Dict]:
        """
        Align images based on feature matches
        
        Args:
            images_data: List of image data dictionaries
            matches: List of match dictionaries
            
        Returns:
            List of aligned image data dictionaries
        """
        if len(images_data) < 2:
            return images_data
        
        logger.info(f"Aligning {len(images_data)} images")
        
        # Build connectivity graph from matches
        graph = self._build_graph(images_data, matches)
        
        # Find reference image (image with most connections)
        ref_idx = self._find_reference_image(graph)
        
        logger.info(f"Using image {ref_idx} as reference")
        
        # Calculate transforms relative to reference
        transforms = self._calculate_transforms(
            images_data, features_data, matches, graph, ref_idx
        )
        
        # Apply transforms
        aligned_images = []
        for i, img_data in enumerate(images_data):
            if i == ref_idx:
                aligned_data = img_data.copy()
                aligned_data['transform'] = np.eye(3)
                h, w = img_data['image'].shape[:2]
                aligned_data['bbox'] = (0, 0, w, h)
                aligned_images.append(aligned_data)
            else:
                transform = transforms.get(i)
                if transform is not None:
                    aligned_img = self._apply_transform(img_data, transform)
                    aligned_images.append(aligned_img)
                else:
                    logger.warning(f"Could not align image {i}, skipping")
        
        return aligned_images
    
    def create_grid_layout(
        self,
        images_data: List[Dict],
        features_data: List[Dict],
        matches: List[Dict],
        grid_size: Optional[Tuple[int, int]] = None
    ) -> Dict:
        """
        Create 2D grid layout without merging
        
        Args:
            images_data: List of image data dictionaries
            matches: List of match dictionaries
            grid_size: Optional (rows, cols) for grid
            
        Returns:
            Dictionary with grid layout information
        """
        logger.info("Creating grid layout...")
        
        # Build graph
        graph = self._build_graph(images_data, matches)
        
        # Estimate grid dimensions if not provided
        if grid_size is None:
            grid_size = self._estimate_grid_size(len(images_data))
        
        # Arrange images in grid based on spatial relationships
        positions = self._arrange_in_grid(images_data, matches, grid_size)
        
        return {
            'images': images_data,
            'positions': positions,
            'grid_size': grid_size,
            'matches': matches
        }
    
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
            if match['confidence'] > 0.1:  # Threshold
                graph[i].append(j)
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
    
    def _calculate_transforms(
        self,
        images_data: List[Dict],
        features_data: List[Dict],
        matches: List[Dict],
        graph: Dict[int, List[int]],
        ref_idx: int
    ) -> Dict[int, np.ndarray]:
        """Calculate transformation matrices"""
        transforms = {ref_idx: np.eye(3)}
        
        # BFS to calculate transforms
        queue = [ref_idx]
        visited = {ref_idx}
        
        while queue:
            current = queue.pop(0)
            
            for neighbor in graph[current]:
                if neighbor in visited:
                    continue
                
                # Find match between current and neighbor
                match_data = self._find_match(matches, current, neighbor)
                if match_data:
                    # Calculate transform using keypoints
                    transform = self._estimate_transform_from_match(
                        features_data[current],
                        features_data[neighbor],
                        match_data
                    )
                    
                    if transform is not None:
                        # Compose with current transform
                        if current in transforms:
                            transforms[neighbor] = transforms[current] @ transform
                        else:
                            transforms[neighbor] = transform
                        
                        visited.add(neighbor)
                        queue.append(neighbor)
        
        return transforms
    
    def _find_match(
        self,
        matches: List[Dict],
        i: int,
        j: int
    ) -> Optional[Dict]:
        """Find match between two images"""
        for match in matches:
            if (match['image_i'] == i and match['image_j'] == j) or \
               (match['image_i'] == j and match['image_j'] == i):
                return match
        return None
    
    def _estimate_transform_from_match(
        self,
        features1: Dict,
        features2: Dict,
        match_data: Dict
    ) -> Optional[np.ndarray]:
        """
        Estimate homography from match data using keypoints
        """
        kp1 = features1['keypoints']
        kp2 = features2['keypoints']
        matches = match_data['matches']
        
        if len(matches) < 4:
            return None
        
        # Extract matched keypoint coordinates
        pts1 = []
        pts2 = []
        
        for match in matches:
            idx1 = match.queryIdx
            idx2 = match.trainIdx
            
            if idx1 < len(kp1) and idx2 < len(kp2):
                pts1.append([kp1[idx1][0], kp1[idx1][1]])
                pts2.append([kp2[idx2][0], kp2[idx2][1]])
        
        if len(pts1) < 4:
            return None
        
        pts1 = np.float32(pts1)
        pts2 = np.float32(pts2)
        
        # Estimate homography using RANSAC
        homography, mask = cv2.findHomography(
            pts1, pts2,
            method=cv2.RANSAC,
            ransacReprojThreshold=5.0
        )
        
        return homography
    
    def _apply_transform(self, img_data: Dict, transform: np.ndarray) -> Dict:
        """Apply transformation to image"""
        img = img_data['image']
        h, w = img.shape[:2]
        
        # Calculate output size
        corners = np.array([
            [0, 0, 1],
            [w, 0, 1],
            [w, h, 1],
            [0, h, 1]
        ]).T
        
        transformed_corners = transform @ corners
        transformed_corners = transformed_corners[:2] / transformed_corners[2]
        
        x_min = int(np.min(transformed_corners[0]))
        x_max = int(np.max(transformed_corners[0]))
        y_min = int(np.min(transformed_corners[1]))
        y_max = int(np.max(transformed_corners[1]))
        
        # Adjust transform for translation
        translation = np.array([[1, 0, -x_min],
                               [0, 1, -y_min],
                               [0, 0, 1]])
        adjusted_transform = translation @ transform
        
        # Warp image
        output_w = x_max - x_min
        output_h = y_max - y_min
        
        warped = cv2.warpPerspective(
            img,
            adjusted_transform,
            (output_w, output_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT
        )
        
        # Warp alpha channel if present
        warped_alpha = None
        if img_data.get('alpha') is not None:
            warped_alpha = cv2.warpPerspective(
                img_data['alpha'],
                adjusted_transform,
                (output_w, output_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT
            )
        
        result = img_data.copy()
        result['image'] = warped
        result['alpha'] = warped_alpha
        result['transform'] = adjusted_transform
        result['bbox'] = (x_min, y_min, x_max, y_max)
        
        return result
    
    def _estimate_grid_size(self, n_images: int) -> Tuple[int, int]:
        """Estimate grid dimensions"""
        cols = int(np.ceil(np.sqrt(n_images)))
        rows = int(np.ceil(n_images / cols))
        return (rows, cols)
    
    def _arrange_in_grid(
        self,
        images_data: List[Dict],
        matches: List[Dict],
        grid_size: Tuple[int, int]
    ) -> List[Dict]:
        """Arrange images in grid based on spatial relationships"""
        rows, cols = grid_size
        
        # Simple grid arrangement (can be improved with spatial analysis)
        positions = []
        idx = 0
        
        for row in range(rows):
            for col in range(cols):
                if idx < len(images_data):
                    positions.append({
                        'image_idx': idx,
                        'row': row,
                        'col': col,
                        'x': col * images_data[idx]['shape'][1],
                        'y': row * images_data[idx]['shape'][0]
                    })
                    idx += 1
        
        return positions

