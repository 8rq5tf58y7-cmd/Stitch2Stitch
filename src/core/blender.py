"""
Image blending engine with advanced algorithms
"""

import cv2
import numpy as np
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class ImageBlender:
    """Blend aligned images into final panorama"""
    
    def __init__(self, use_gpu: bool = False, method: str = 'multiband'):
        """
        Initialize blender
        
        Args:
            use_gpu: Enable GPU acceleration
            method: Blending method ('multiband', 'feather', 'linear')
        """
        self.use_gpu = use_gpu
        self.method = method
        logger.info(f"Image blender initialized (method: {method})")
    
    def blend_images(self, aligned_images: List[Dict]) -> np.ndarray:
        """
        Blend aligned images into panorama
        
        Args:
            aligned_images: List of aligned image data dictionaries
            
        Returns:
            Blended panorama
        """
        if not aligned_images:
            raise ValueError("No images to blend")
        
        if len(aligned_images) == 1:
            return aligned_images[0]['image']
        
        logger.info(f"Blending {len(aligned_images)} images using {self.method} method")
        
        if self.method == 'multiband':
            return self._multiband_blend(aligned_images)
        elif self.method == 'feather':
            return self._feather_blend(aligned_images)
        else:
            return self._linear_blend(aligned_images)
    
    def _multiband_blend(self, aligned_images: List[Dict]) -> np.ndarray:
        """Multi-band blending for seamless transitions"""
        # Calculate bounding box
        bbox = self._calculate_bbox(aligned_images)
        x_min, y_min, x_max, y_max = bbox
        
        output_h = y_max - y_min
        output_w = x_max - x_min
        
        # Create output canvas
        panorama = np.zeros((output_h, output_w, 3), dtype=np.float32)
        weight_sum = np.zeros((output_h, output_w), dtype=np.float32)
        
        # Blend each image
        for img_data in aligned_images:
            img = img_data['image'].astype(np.float32)
            alpha = img_data.get('alpha')
            
            if alpha is not None:
                alpha = alpha.astype(np.float32) / 255.0
            else:
                alpha = np.ones((img.shape[0], img.shape[1]), dtype=np.float32)
            
            # Create weight mask (distance from edges)
            weight = self._create_weight_mask(img.shape[:2], alpha)
            
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
        weight_sum[weight_sum == 0] = 1  # Avoid division by zero
        for c in range(3):
            panorama[:, :, c] /= weight_sum
        
        return panorama.astype(np.uint8)
    
    def _feather_blend(self, aligned_images: List[Dict]) -> np.ndarray:
        """Feather blending (simpler, faster)"""
        bbox = self._calculate_bbox(aligned_images)
        x_min, y_min, x_max, y_max = bbox
        
        output_h = y_max - y_min
        output_w = x_max - x_min
        
        panorama = np.zeros((output_h, output_w, 3), dtype=np.float32)
        weight_sum = np.zeros((output_h, output_w), dtype=np.float32)
        
        for img_data in aligned_images:
            img = img_data['image'].astype(np.float32)
            alpha = img_data.get('alpha')
            
            if alpha is not None:
                alpha = alpha.astype(np.float32) / 255.0
            else:
                alpha = np.ones((img.shape[0], img.shape[1]), dtype=np.float32)
            
            # Simple distance-based weight
            weight = self._create_weight_mask(img.shape[:2], alpha, feather_size=50)
            
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
                
                weight_crop = weight_crop * alpha_crop
                
                for c in range(3):
                    panorama[y_off:y_end, x_off:x_end, c] += \
                        img_crop[:, :, c] * weight_crop
                
                weight_sum[y_off:y_end, x_off:x_end] += weight_crop
        
        weight_sum[weight_sum == 0] = 1
        for c in range(3):
            panorama[:, :, c] /= weight_sum
        
        return panorama.astype(np.uint8)
    
    def _linear_blend(self, aligned_images: List[Dict]) -> np.ndarray:
        """Simple linear blending"""
        return self._feather_blend(aligned_images)
    
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
    
    def _create_weight_mask(
        self,
        shape: Tuple[int, int],
        alpha: np.ndarray,
        feather_size: int = 100
    ) -> np.ndarray:
        """Create weight mask for blending"""
        h, w = shape
        weight = np.ones((h, w), dtype=np.float32)
        
        # Create distance transform from edges
        # Apply alpha mask
        weight = weight * alpha
        
        # Distance from edges
        dist_y = np.minimum(
            np.arange(h)[:, None],
            np.arange(h)[::-1, None]
        )
        dist_x = np.minimum(
            np.arange(w)[None, :],
            np.arange(w)[None, ::-1]
        )
        dist = np.minimum(dist_y, dist_x)
        
        # Feather edges
        dist = np.clip(dist / feather_size, 0, 1)
        weight = weight * dist
        
        return weight

