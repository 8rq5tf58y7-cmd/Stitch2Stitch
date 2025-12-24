"""
PixelStitch-inspired structure-preserving blending
Based on: "PixelStitch: Structure-Preserving Pixel-Wise Bidirectional Warps"
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class PixelStitchBlender:
    """
    Structure-preserving pixel-wise bidirectional warps
    Based on: "PixelStitch: Structure-Preserving Pixel-Wise Bidirectional Warps 
    for Unsupervised Image Stitching"
    """
    
    def __init__(self, use_gpu: bool = False):
        """
        Initialize PixelStitch blender
        
        Args:
            use_gpu: Enable GPU acceleration
        """
        self.use_gpu = use_gpu
        logger.info("PixelStitch blender initialized")
    
    def blend_images(self, aligned_images: List[Dict]) -> np.ndarray:
        """
        Blend images using structure-preserving pixel-wise warps
        
        Args:
            aligned_images: List of aligned image data dictionaries
            
        Returns:
            Blended panorama
        """
        if not aligned_images:
            raise ValueError("No images to blend")
        
        if len(aligned_images) == 1:
            return aligned_images[0]['image']
        
        logger.info(f"PixelStitch blending {len(aligned_images)} images")
        
        # Calculate bounding box
        bbox = self._calculate_bbox(aligned_images)
        x_min, y_min, x_max, y_max = bbox
        
        output_h = y_max - y_min
        output_w = x_max - x_min
        
        # Create output canvas
        panorama = np.zeros((output_h, output_w, 3), dtype=np.float32)
        weight_sum = np.zeros((output_h, output_w), dtype=np.float32)
        
        # Process each image pair
        for idx, img_data in enumerate(aligned_images):
            img = img_data['image'].astype(np.float32)
            alpha = img_data.get('alpha')
            
            if alpha is not None:
                alpha = alpha.astype(np.float32) / 255.0
            else:
                alpha = np.ones((img.shape[0], img.shape[1]), dtype=np.float32)
            
            # Create bidirectional optical flow-based weight
            weight = self._create_bidirectional_weight(img.shape[:2], alpha, idx, aligned_images)
            
            # Place in panorama
            bbox_img = img_data.get('bbox', (0, 0, img.shape[1], img.shape[0]))
            x_off = bbox_img[0] - x_min
            y_off = bbox_img[1] - y_min
            
            h, w = img.shape[:2]
            x_end = min(x_off + w, output_w)
            y_end = min(y_off + h, output_h)
            
            if x_off >= 0 and y_off >= 0:
                img_crop = img[:y_end-y_off, :x_end-x_off]
                weight_crop = weight[:y_end-y_off, :x_end-x_off]
                alpha_crop = alpha[:y_end-y_off, :x_end-x_off]
                
                # Apply alpha and weight
                weight_crop = weight_crop * alpha_crop
                
                for c in range(3):
                    panorama[y_off:y_end, x_off:x_end, c] += \
                        img_crop[:, :, c] * weight_crop
                
                weight_sum[y_off:y_end, x_off:x_end] += weight_crop
        
        # Normalize by weight sum
        weight_sum[weight_sum == 0] = 1
        for c in range(3):
            panorama[:, :, c] /= weight_sum
        
        return panorama.astype(np.uint8)
    
    def _create_bidirectional_weight(
        self,
        shape: Tuple[int, int],
        alpha: np.ndarray,
        idx: int,
        all_images: List[Dict]
    ) -> np.ndarray:
        """
        Create weight mask using bidirectional optical flow concept
        """
        h, w = shape
        weight = np.ones((h, w), dtype=np.float32)
        
        # Find overlapping regions with other images
        overlap_mask = self._compute_overlap_mask(idx, all_images, shape)
        
        # In overlap regions, use bidirectional consistency
        if overlap_mask is not None:
            # Create smooth transition in overlap regions
            weight = weight * (1.0 - 0.3 * overlap_mask)
        
        # Structure-preserving: enhance weights near edges
        # (In full implementation, would use actual optical flow)
        edge_weight = self._structure_preserving_weight(shape)
        weight = weight * (1.0 + 0.2 * edge_weight)
        
        # Apply alpha
        weight = weight * alpha
        
        # Distance-based feathering
        dist_y = np.minimum(
            np.arange(h)[:, None],
            np.arange(h)[::-1, None]
        )
        dist_x = np.minimum(
            np.arange(w)[None, :],
            np.arange(w)[None, ::-1]
        )
        dist = np.minimum(dist_y, dist_x)
        dist = np.clip(dist / 100.0, 0, 1)
        weight = weight * dist
        
        return weight
    
    def _compute_overlap_mask(
        self,
        idx: int,
        all_images: List[Dict],
        shape: Tuple[int, int]
    ) -> Optional[np.ndarray]:
        """Compute mask for overlapping regions"""
        # Simplified overlap detection
        # In full implementation, would use actual overlap computation
        h, w = shape
        
        # Create a mask that assumes center regions are more likely to overlap
        center_y, center_x = h // 2, w // 2
        y_coords, x_coords = np.ogrid[:h, :w]
        
        # Distance from center
        dist_from_center = np.sqrt(
            (y_coords - center_y)**2 + (x_coords - center_x)**2
        )
        max_dist = np.sqrt(center_y**2 + center_x**2)
        
        # Normalize to 0-1
        overlap_mask = 1.0 - np.clip(dist_from_center / max_dist, 0, 1)
        
        return overlap_mask
    
    def _structure_preserving_weight(self, shape: Tuple[int, int]) -> np.ndarray:
        """Create weight that preserves structure (edges)"""
        h, w = shape
        # Placeholder: would use actual edge detection
        # For now, create a simple gradient
        weight = np.ones((h, w), dtype=np.float32)
        return weight
    
    def _calculate_bbox(self, aligned_images: List[Dict]) -> Tuple[int, int, int, int]:
        """Calculate bounding box for all images"""
        x_min = y_min = float('inf')
        x_max = y_max = float('-inf')
        
        for img_data in aligned_images:
            bbox = img_data.get('bbox')
            if bbox:
                x_min = min(x_min, bbox[0])
                y_min = min(y_min, bbox[1])
                x_max = max(x_max, bbox[2])
                y_max = max(y_max, bbox[3])
            else:
                h, w = img_data['image'].shape[:2]
                x_max = max(x_max, w)
                y_max = max(y_max, h)
        
        return (int(x_min), int(y_min), int(x_max), int(y_max))

