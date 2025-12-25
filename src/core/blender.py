"""
Image blending engine with advanced algorithms
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
import gc

logger = logging.getLogger(__name__)

# Default maximum pixels for output panorama (100 megapixels)
# Set to None or 0 to disable limit
# Memory guidelines:
#   - 50MP panorama: ~600MB RAM for blending (uint8) or ~2.4GB (float32)
#   - 100MP panorama: ~1.2GB RAM for blending (uint8) or ~4.8GB (float32)
#   - 200MP panorama: ~2.4GB RAM for blending (uint8) or ~9.6GB (float32)
# Recommended: Set limit based on available RAM / 6 for float32 blending methods
DEFAULT_MAX_PANORAMA_PIXELS = 100_000_000


class ImageBlender:
    """Blend aligned images into final panorama"""
    
    def __init__(
        self, 
        use_gpu: bool = False, 
        method: str = 'multiband', 
        options: Dict = None,
        max_panorama_pixels: Optional[int] = DEFAULT_MAX_PANORAMA_PIXELS
    ):
        """
        Initialize blender
        
        Args:
            use_gpu: Enable GPU acceleration
            method: Blending method ('multiband', 'feather', 'linear', 'autostitch')
            options: Blending options dict with keys:
                - hdr_mode: bool - Enable HDR/exposure fusion
                - anti_ghosting: bool - Enable anti-ghosting
                - pixel_selection: str - Pixel selection method ('weighted_average', 'strongest_signal', 'median', 'maximum', 'minimum')
            max_panorama_pixels: Maximum output pixels (None or 0 = unlimited)
                Memory guidelines:
                - 50MP: ~600MB (uint8) / ~2.4GB (float32)
                - 100MP: ~1.2GB (uint8) / ~4.8GB (float32)
                - 200MP: ~2.4GB (uint8) / ~9.6GB (float32)
                Recommended: available_RAM_GB * 150_000_000 for float32 methods
        """
        self.use_gpu = use_gpu
        self.method = method
        self.options = options or {}
        self.max_panorama_pixels = max_panorama_pixels if max_panorama_pixels else None
        logger.info(f"Image blender initialized (method: {method}, max_pixels: {self.max_panorama_pixels or 'unlimited'})")
    
    def blend_images(self, aligned_images: List[Dict], options: Dict = None) -> np.ndarray:
        """
        Blend aligned images into panorama
        
        Args:
            aligned_images: List of aligned image data dictionaries
            options: Optional blending options (overrides instance options)
            
        Returns:
            Blended panorama
        """
        if not aligned_images:
            raise ValueError("No images to blend")
        
        if len(aligned_images) == 1:
            return aligned_images[0]['image']
        
        # Merge options
        blend_options = {**self.options, **(options or {})}
        
        logger.info(f"Blending {len(aligned_images)} images using {self.method} method")
        
        if self.method == 'autostitch':
            return self._autostitch_blend(aligned_images, blend_options)
        elif self.method == 'multiband':
            return self._multiband_blend(aligned_images, blend_options)
        elif self.method == 'feather':
            return self._feather_blend(aligned_images, blend_options)
        else:
            return self._linear_blend(aligned_images, blend_options)
    
    def _autostitch_blend(self, aligned_images: List[Dict], options: Dict) -> np.ndarray:
        """
        AutoStitch-style blending: select pixels from one image in overlap regions
        No averaging/blending - images fit together like a puzzle
        Memory-optimized version.
        """
        if not aligned_images:
            raise ValueError("No images to blend")
        
        # Calculate bounding box
        bbox = self._calculate_bbox(aligned_images)
        x_min, y_min, x_max, y_max = bbox
        
        output_h = y_max - y_min
        output_w = x_max - x_min
        
        if output_h <= 0 or output_w <= 0:
            logger.error(f"Invalid bounding box dimensions: {bbox}")
            return aligned_images[0]['image'].copy()
        
        # Check size and scale down if needed (only if limit is set)
        total_pixels = output_h * output_w
        if self.max_panorama_pixels and total_pixels > self.max_panorama_pixels:
            scale = np.sqrt(self.max_panorama_pixels / total_pixels)
            output_h = int(output_h * scale)
            output_w = int(output_w * scale)
            logger.warning(f"Panorama too large ({total_pixels/1e6:.1f}MP), scaling to {output_w}x{output_h}")
        
        logger.info(f"Creating panorama canvas: {output_w}x{output_h} ({output_w*output_h/1e6:.1f}MP)")
        
        # Create output canvas with white background (uint8 to save memory)
        panorama = np.ones((output_h, output_w, 3), dtype=np.uint8) * 255
        # Use uint8 for filled mask too (saves memory vs bool on large arrays)
        filled_mask = np.zeros((output_h, output_w), dtype=np.uint8)
        
        # Sort images by quality/confidence (best first)
        sorted_images = sorted(
            aligned_images,
            key=lambda x: x.get('quality', 0.5),
            reverse=True
        )
        
        # Process each image
        for idx, img_data in enumerate(sorted_images):
            logger.debug(f"Blending image {idx+1}/{len(sorted_images)}")
            
            img = img_data['image']
            alpha = img_data.get('alpha')
            was_warped = img_data.get('warped', False)
            
            h, w = img.shape[:2]
            
            # Create content mask efficiently
            if alpha is not None:
                alpha_mask = alpha > 127
            else:
                alpha_mask = np.ones((h, w), dtype=bool)
            
            # Filter warp artifacts
            if was_warped and len(img.shape) == 3:
                # Use max channel for faster intensity check
                intensity = np.max(img, axis=2)
                alpha_mask = alpha_mask & (intensity > 10) & (intensity < 250)
            
            bbox_img = img_data.get('bbox', (0, 0, w, h))
            x_off = bbox_img[0] - x_min
            y_off = bbox_img[1] - y_min
            
            # Calculate valid region
            src_x_start = max(0, -x_off)
            src_y_start = max(0, -y_off)
            dst_x_start = max(0, x_off)
            dst_y_start = max(0, y_off)
            
            src_x_end = min(w, output_w - x_off)
            src_y_end = min(h, output_h - y_off)
            dst_x_end = min(output_w, x_off + w)
            dst_y_end = min(output_h, y_off + h)
            
            if src_x_end <= src_x_start or src_y_end <= src_y_start:
                continue
            
            # Get source and destination regions
            src_region = img[src_y_start:src_y_end, src_x_start:src_x_end]
            src_alpha = alpha_mask[src_y_start:src_y_end, src_x_start:src_x_end]
            dst_filled = filled_mask[dst_y_start:dst_y_end, dst_x_start:dst_x_end]
            
            # Write mask: valid alpha and not already filled
            write_mask = src_alpha & (dst_filled == 0)
            
            if np.any(write_mask):
                # Use numpy advanced indexing (faster than per-channel loop)
                dst_region = panorama[dst_y_start:dst_y_end, dst_x_start:dst_x_end]
                dst_region[write_mask] = src_region[write_mask]
                
                # Update filled mask
                filled_mask[dst_y_start:dst_y_end, dst_x_start:dst_x_end][write_mask] = 1
            
            # Free alpha_mask memory
            del alpha_mask, write_mask
        
        # Cleanup
        del filled_mask
        gc.collect()
        
        logger.info("Blending complete")
        return panorama
    
    def _multiband_blend(self, aligned_images: List[Dict], options: Dict) -> np.ndarray:
        """Multi-band blending for seamless transitions (memory-optimized)"""
        if not aligned_images:
            raise ValueError("No images to blend")
        
        # Calculate bounding box
        bbox = self._calculate_bbox(aligned_images)
        x_min, y_min, x_max, y_max = bbox
        
        output_h = y_max - y_min
        output_w = x_max - x_min
        
        # Validate dimensions
        if output_h <= 0 or output_w <= 0:
            logger.error(f"Invalid bounding box dimensions: {bbox}")
            return aligned_images[0]['image'].copy()
        
        # Limit maximum size to prevent memory issues (only if limit is set)
        total_pixels = output_h * output_w
        if self.max_panorama_pixels and total_pixels > self.max_panorama_pixels:
            scale = np.sqrt(self.max_panorama_pixels / total_pixels)
            output_h = int(output_h * scale)
            output_w = int(output_w * scale)
            logger.warning(f"Panorama too large, scaling to {output_w}x{output_h}")
        
        logger.info(f"Multiband blending: {output_w}x{output_h} ({output_w*output_h/1e6:.1f}MP)")
        
        # Check if pixel_selection requires special handling (and warn about performance)
        pixel_selection = options.get('pixel_selection', 'weighted_average')
        if pixel_selection not in ('weighted_average', 'strongest_signal'):
            logger.warning(f"Pixel selection '{pixel_selection}' may be slow on large images")
        
        # Create output canvas (use float32 for weighted average)
        panorama = np.zeros((output_h, output_w, 3), dtype=np.float32)
        weight_sum = np.zeros((output_h, output_w), dtype=np.float32)
        
        # Blend each image
        for idx, img_data in enumerate(aligned_images):
            logger.debug(f"Multiband blending image {idx+1}/{len(aligned_images)}")
            
            img = img_data['image']
            h, w = img.shape[:2]
            alpha = img_data.get('alpha')
            
            if alpha is not None:
                alpha_f = alpha.astype(np.float32) / 255.0
            else:
                alpha_f = np.ones((h, w), dtype=np.float32)
            
            # Create weight mask (distance from edges)
            weight = self._create_weight_mask((h, w), alpha_f)
            
            # Calculate placement
            bbox_img = img_data.get('bbox', (0, 0, w, h))
            x_off = bbox_img[0] - x_min
            y_off = bbox_img[1] - y_min
            
            # Calculate valid region
            src_x_start = max(0, -x_off)
            src_y_start = max(0, -y_off)
            dst_x_start = max(0, x_off)
            dst_y_start = max(0, y_off)
            
            src_x_end = min(w, output_w - x_off)
            src_y_end = min(h, output_h - y_off)
            dst_x_end = min(output_w, x_off + w)
            dst_y_end = min(output_h, y_off + h)
            
            if src_x_end <= src_x_start or src_y_end <= src_y_start:
                continue
            
            # Extract regions
            img_region = img[src_y_start:src_y_end, src_x_start:src_x_end].astype(np.float32)
            weight_region = weight[src_y_start:src_y_end, src_x_start:src_x_end]
            alpha_region = alpha_f[src_y_start:src_y_end, src_x_start:src_x_end]
            
            # Combined weight
            combined_weight = weight_region * alpha_region
            
            # Add to panorama (vectorized)
            panorama[dst_y_start:dst_y_end, dst_x_start:dst_x_end] += \
                img_region * combined_weight[:, :, np.newaxis]
            weight_sum[dst_y_start:dst_y_end, dst_x_start:dst_x_end] += combined_weight
            
            # Cleanup
            del alpha_f, weight, combined_weight
        
        # Normalize
            weight_sum[weight_sum == 0] = 1  # Avoid division by zero
        panorama /= weight_sum[:, :, np.newaxis]
        
        # Cleanup
        del weight_sum
        gc.collect()
        
        logger.info("Multiband blending complete")
        return np.clip(panorama, 0, 255).astype(np.uint8)
    
    def _feather_blend(self, aligned_images: List[Dict], options: Dict) -> np.ndarray:
        """Feather blending (simpler, faster, memory-optimized)"""
        bbox = self._calculate_bbox(aligned_images)
        x_min, y_min, x_max, y_max = bbox
        
        output_h = y_max - y_min
        output_w = x_max - x_min
        
        # Size limit (only if limit is set)
        total_pixels = output_h * output_w
        if self.max_panorama_pixels and total_pixels > self.max_panorama_pixels:
            scale = np.sqrt(self.max_panorama_pixels / total_pixels)
            output_h = int(output_h * scale)
            output_w = int(output_w * scale)
            logger.warning(f"Panorama too large, scaling to {output_w}x{output_h}")
        
        logger.info(f"Feather blending: {output_w}x{output_h} ({output_w*output_h/1e6:.1f}MP)")
        
        panorama = np.zeros((output_h, output_w, 3), dtype=np.float32)
        weight_sum = np.zeros((output_h, output_w), dtype=np.float32)
        
        for idx, img_data in enumerate(aligned_images):
            logger.debug(f"Feather blending image {idx+1}/{len(aligned_images)}")
            
            img = img_data['image']
            h, w = img.shape[:2]
            alpha = img_data.get('alpha')
            
            if alpha is not None:
                alpha_f = alpha.astype(np.float32) / 255.0
            else:
                alpha_f = np.ones((h, w), dtype=np.float32)
            
            # Simple distance-based weight
            weight = self._create_weight_mask((h, w), alpha_f, feather_size=50)
            
            bbox_img = img_data.get('bbox', (0, 0, w, h))
            x_off = bbox_img[0] - x_min
            y_off = bbox_img[1] - y_min
            
            # Calculate valid region
            src_x_start = max(0, -x_off)
            src_y_start = max(0, -y_off)
            dst_x_start = max(0, x_off)
            dst_y_start = max(0, y_off)
            
            src_x_end = min(w, output_w - x_off)
            src_y_end = min(h, output_h - y_off)
            dst_x_end = min(output_w, x_off + w)
            dst_y_end = min(output_h, y_off + h)
            
            if src_x_end <= src_x_start or src_y_end <= src_y_start:
                continue
            
            # Extract regions
            img_region = img[src_y_start:src_y_end, src_x_start:src_x_end].astype(np.float32)
            weight_region = weight[src_y_start:src_y_end, src_x_start:src_x_end]
            alpha_region = alpha_f[src_y_start:src_y_end, src_x_start:src_x_end]
            
            combined_weight = weight_region * alpha_region
            
            # Add to panorama (vectorized)
            panorama[dst_y_start:dst_y_end, dst_x_start:dst_x_end] += \
                img_region * combined_weight[:, :, np.newaxis]
            weight_sum[dst_y_start:dst_y_end, dst_x_start:dst_x_end] += combined_weight
            
            # Cleanup
            del alpha_f, weight
        
        # Normalize
        weight_sum[weight_sum == 0] = 1
        panorama /= weight_sum[:, :, np.newaxis]
        
        del weight_sum
        gc.collect()
        
        logger.info("Feather blending complete")
        return np.clip(panorama, 0, 255).astype(np.uint8)
    
    def _linear_blend(self, aligned_images: List[Dict], options: Dict) -> np.ndarray:
        """Simple linear blending"""
        return self._feather_blend(aligned_images, options)
    
    def _calculate_bbox(self, aligned_images: List[Dict]) -> Tuple[int, int, int, int]:
        """Calculate bounding box for all images"""
        if not aligned_images:
            return (0, 0, 100, 100)
        
        x_min = y_min = float('inf')
        x_max = y_max = float('-inf')
        
        for img_data in aligned_images:
            bbox = img_data.get('bbox')
            if bbox and len(bbox) >= 4:
                x_min = min(x_min, bbox[0])
                y_min = min(y_min, bbox[1])
                x_max = max(x_max, bbox[2])
                y_max = max(y_max, bbox[3])
            else:
                h, w = img_data['image'].shape[:2]
                if x_min == float('inf'):
                    x_min = 0
                if y_min == float('inf'):
                    y_min = 0
                x_max = max(x_max, w)
                y_max = max(y_max, h)
        
        # Validate bounding box
        if x_min == float('inf') or y_min == float('inf') or x_max == float('-inf') or y_max == float('-inf'):
            # Fallback: use first image dimensions
            h, w = aligned_images[0]['image'].shape[:2]
            return (0, 0, w, h)
        
        # Ensure valid dimensions
        if x_max <= x_min:
            x_max = x_min + 100
        if y_max <= y_min:
            y_max = y_min + 100
        
        return (int(x_min), int(y_min), int(x_max), int(y_max))
    
    def _apply_pixel_selection(
        self,
        panorama: np.ndarray,
        weight_sum: np.ndarray,
        aligned_images: List[Dict],
        bbox: Tuple[int, int, int, int],
        method: str
    ) -> np.ndarray:
        """
        Apply pixel selection method (strongest signal, median, max, min)
        Optimized version using numpy operations.
        
        Memory guidelines for median mode:
        - Stores all pixel values in a 4D array: (height, width, n_images, 3)
        - 1000x1000 panorama with 10 images: ~120MB
        - 5000x5000 panorama with 10 images: ~3GB
        - 10000x10000 panorama with 10 images: ~12GB
        """
        x_min, y_min, x_max, y_max = bbox
        output_h, output_w = panorama.shape[:2]
        n_images = len(aligned_images)
        
        logger.info(f"Applying pixel selection: {method}")
        
        if method == 'median':
            # Estimate memory requirement for median (stores all pixel values)
            estimated_memory_gb = (output_h * output_w * n_images * 3 * 4) / (1024**3)
            logger.warning(f"Median mode requires ~{estimated_memory_gb:.1f}GB RAM for {output_w}x{output_h} with {n_images} images")
            
            # Use chunked processing for very large images
            if estimated_memory_gb > 2.0:
                logger.info("Using chunked processing for median (large image)")
                return self._apply_median_chunked(aligned_images, bbox, output_h, output_w)
            else:
                return self._apply_median_full(aligned_images, bbox, output_h, output_w)
        
        # For strongest_signal, maximum, minimum - use efficient numpy operations
        result = np.zeros((output_h, output_w, 3), dtype=np.float32)
        
        if method == 'strongest_signal':
            # Track maximum luminance at each pixel
            max_luminance = np.zeros((output_h, output_w), dtype=np.float32) - 1
        elif method == 'maximum':
            result.fill(-np.inf)
        elif method == 'minimum':
            result.fill(np.inf)
        
        for img_data in aligned_images:
            img = img_data['image']
            h, w = img.shape[:2]
            
            bbox_img = img_data.get('bbox', (0, 0, w, h))
            x_off = bbox_img[0] - x_min
            y_off = bbox_img[1] - y_min
            
            # Calculate valid region
            src_x_start = max(0, -x_off)
            src_y_start = max(0, -y_off)
            dst_x_start = max(0, x_off)
            dst_y_start = max(0, y_off)
            
            src_x_end = min(w, output_w - x_off)
            src_y_end = min(h, output_h - y_off)
            dst_x_end = min(output_w, x_off + w)
            dst_y_end = min(output_h, y_off + h)
            
            if src_x_end <= src_x_start or src_y_end <= src_y_start:
                continue
            
            src_region = img[src_y_start:src_y_end, src_x_start:src_x_end].astype(np.float32)
            dst_slice = (slice(dst_y_start, dst_y_end), slice(dst_x_start, dst_x_end))
            
            if method == 'strongest_signal':
                # Calculate luminance of source region
                luminance = np.mean(src_region, axis=2)
                # Mask where this image has higher luminance
                mask = luminance > max_luminance[dst_slice]
                # Update result where mask is True
                result_region = result[dst_slice]
                result_region[mask] = src_region[mask]
                max_luminance[dst_slice] = np.maximum(max_luminance[dst_slice], luminance)
            elif method == 'maximum':
                result[dst_slice] = np.maximum(result[dst_slice], src_region)
            elif method == 'minimum':
                result[dst_slice] = np.minimum(result[dst_slice], src_region)
        
        # Handle pixels with no coverage
        if method in ('maximum', 'minimum'):
            no_coverage = np.isinf(result[:, :, 0])
            result[no_coverage] = 0
        
        return result
    
    def _apply_median_full(
        self,
        aligned_images: List[Dict],
        bbox: Tuple[int, int, int, int],
        output_h: int,
        output_w: int
    ) -> np.ndarray:
        """Apply median pixel selection using full array (for smaller images)"""
        x_min, y_min, x_max, y_max = bbox
        n_images = len(aligned_images)
        
        # Create array to hold all pixel values: (h, w, n_images, 3)
        all_pixels = np.full((output_h, output_w, n_images, 3), np.nan, dtype=np.float32)
        
        for img_idx, img_data in enumerate(aligned_images):
            img = img_data['image']
            h, w = img.shape[:2]
            
            bbox_img = img_data.get('bbox', (0, 0, w, h))
            x_off = bbox_img[0] - x_min
            y_off = bbox_img[1] - y_min
            
            # Calculate valid region
            src_x_start = max(0, -x_off)
            src_y_start = max(0, -y_off)
            dst_x_start = max(0, x_off)
            dst_y_start = max(0, y_off)
            
            src_x_end = min(w, output_w - x_off)
            src_y_end = min(h, output_h - y_off)
            dst_x_end = min(output_w, x_off + w)
            dst_y_end = min(output_h, y_off + h)
            
            if src_x_end <= src_x_start or src_y_end <= src_y_start:
                continue
            
            src_region = img[src_y_start:src_y_end, src_x_start:src_x_end].astype(np.float32)
            all_pixels[dst_y_start:dst_y_end, dst_x_start:dst_x_end, img_idx] = src_region
        
        # Compute median ignoring NaN values
        result = np.nanmedian(all_pixels, axis=2)
        
        # Handle pixels with no coverage
        no_coverage = np.all(np.isnan(all_pixels[:, :, :, 0]), axis=2)
        result[no_coverage] = 0
        
        del all_pixels
        gc.collect()
        
        return result
    
    def _apply_median_chunked(
        self,
        aligned_images: List[Dict],
        bbox: Tuple[int, int, int, int],
        output_h: int,
        output_w: int,
        chunk_size: int = 1000
    ) -> np.ndarray:
        """Apply median pixel selection using chunked processing (for large images)"""
        x_min, y_min, x_max, y_max = bbox
        n_images = len(aligned_images)
        
        result = np.zeros((output_h, output_w, 3), dtype=np.float32)
        
        # Process in chunks
        for y_start in range(0, output_h, chunk_size):
            y_end = min(y_start + chunk_size, output_h)
            chunk_h = y_end - y_start
            
            logger.debug(f"Processing median chunk: rows {y_start}-{y_end}")
            
            for x_start in range(0, output_w, chunk_size):
                x_end = min(x_start + chunk_size, output_w)
                chunk_w = x_end - x_start
                
                # Create chunk array
                chunk_pixels = np.full((chunk_h, chunk_w, n_images, 3), np.nan, dtype=np.float32)
                
                for img_idx, img_data in enumerate(aligned_images):
                    img = img_data['image']
                    h, w = img.shape[:2]
                    
                    bbox_img = img_data.get('bbox', (0, 0, w, h))
                    x_off = bbox_img[0] - x_min
                    y_off = bbox_img[1] - y_min
                    
                    # Calculate intersection with this chunk
                    src_x_start = max(0, x_start - x_off)
                    src_y_start = max(0, y_start - y_off)
                    src_x_end = min(w, x_end - x_off)
                    src_y_end = min(h, y_end - y_off)
                    
                    dst_x_start = max(0, x_off - x_start)
                    dst_y_start = max(0, y_off - y_start)
                    dst_x_end = min(chunk_w, x_off + w - x_start)
                    dst_y_end = min(chunk_h, y_off + h - y_start)
                    
                    if (src_x_end <= src_x_start or src_y_end <= src_y_start or
                        dst_x_end <= dst_x_start or dst_y_end <= dst_y_start):
                        continue
                    
                    # Ensure sizes match
                    copy_h = min(src_y_end - src_y_start, dst_y_end - dst_y_start)
                    copy_w = min(src_x_end - src_x_start, dst_x_end - dst_x_start)
                    
                    if copy_h > 0 and copy_w > 0:
                        src_region = img[src_y_start:src_y_start+copy_h, src_x_start:src_x_start+copy_w]
                        chunk_pixels[dst_y_start:dst_y_start+copy_h, dst_x_start:dst_x_start+copy_w, img_idx] = \
                            src_region.astype(np.float32)
                
                # Compute median for this chunk
                chunk_result = np.nanmedian(chunk_pixels, axis=2)
                
                # Handle no coverage
                no_coverage = np.all(np.isnan(chunk_pixels[:, :, :, 0]), axis=2)
                chunk_result[no_coverage] = 0
                
                result[y_start:y_end, x_start:x_end] = chunk_result
                
                del chunk_pixels
            
            gc.collect()
        
        return result
    
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
        
        # Feather edges (avoid division by zero)
        if feather_size > 0:
            dist = np.clip(dist / feather_size, 0, 1)
        else:
            dist = np.ones_like(dist)
        weight = weight * dist
        
        return weight

