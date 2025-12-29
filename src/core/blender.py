"""
Image blending engine with advanced algorithms
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
import gc

logger = logging.getLogger(__name__)

# Default maximum pixels for output panorama (500 megapixels = ~22000x22000)
# This is generous for gigapixel output while preventing memory blowup
# Set to None or 0 to disable limit (only if you have lots of RAM!)
# Memory guidelines:
#   - 100MP panorama: ~1.2GB RAM for blending (uint8) or ~4.8GB (float32)
#   - 200MP panorama: ~2.4GB RAM for blending (uint8) or ~9.6GB (float32)
#   - 500MP panorama: ~6GB RAM for blending (uint8) or ~24GB (float32)
DEFAULT_MAX_PANORAMA_PIXELS = 500_000_000


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

    def blend(self, aligned_images: List[Dict], **kwargs) -> np.ndarray:
        """
        Blend aligned images (alias for blend_images with kwargs support)

        Args:
            aligned_images: List of aligned image data dictionaries
            **kwargs: Blending options (padding, fit_all, zoom, etc.)

        Returns:
            Blended panorama
        """
        return self.blend_images(aligned_images, options=kwargs)

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
        
        Options:
            padding: Extra pixels around edges (default 50)
            zoom: Output scale factor (default 1.0, <1 = zoom out, >1 = zoom in)
            fit_all: Ensure all images fit completely (default True)
        """
        if not aligned_images:
            raise ValueError("No images to blend")
        
        # Get options
        padding = options.get('padding', 50)  # Default 50px padding
        zoom = options.get('zoom', 1.0)  # Default no zoom
        fit_all = options.get('fit_all', True)  # Default ensure all images fit
        
        # Calculate bounding box with padding
        bbox = self._calculate_bbox(aligned_images, padding=padding if fit_all else 0)
        x_min, y_min, x_max, y_max = bbox
        
        output_h = y_max - y_min
        output_w = x_max - x_min
        
        if output_h <= 0 or output_w <= 0:
            logger.error(f"Invalid bounding box dimensions: {bbox}")
            return aligned_images[0]['image'].copy()
        
        # Sanity check: detect if alignment has "exploded" (canvas way too large vs content)
        aligned_images, bbox = self._check_and_fix_exploded_alignment(aligned_images, bbox)
        x_min, y_min, x_max, y_max = bbox
        output_h = y_max - y_min
        output_w = x_max - x_min
        
        # Apply zoom factor (zoom < 1.0 = zoom out to show more, zoom > 1.0 = zoom in)
        if zoom != 1.0 and zoom > 0:
            # Adjust bounding box to show more/less area
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            half_w = output_w / 2 / zoom
            half_h = output_h / 2 / zoom
            x_min = int(center_x - half_w)
            y_min = int(center_y - half_h)
            x_max = int(center_x + half_w)
            y_max = int(center_y + half_h)
            output_w = x_max - x_min
            output_h = y_max - y_min
            logger.info(f"Zoom {zoom}x applied: canvas size {output_w}x{output_h}")
        
        # Note: We blend at native resolution first, then scale after cropping
        # This ensures the content fills the target size properly
        total_pixels = output_h * output_w
        scale_factor = 1.0  # No pre-scaling
        
        # Only scale down if MUCH larger than target to prevent memory issues
        max_blend_pixels = 200_000_000  # Max 200MP during blending to save memory
        if total_pixels > max_blend_pixels:
            scale_factor = np.sqrt(max_blend_pixels / total_pixels)
            output_h = int(output_h * scale_factor)
            output_w = int(output_w * scale_factor)
            logger.info(f"Blending at reduced size to save memory: {output_w}x{output_h}")
        
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
            
            # Create content mask - use alpha channel if provided
            if alpha is not None:
                # Trust the provided alpha channel completely
                # It already correctly distinguishes content from transformation borders
                alpha_mask = alpha > 127
            else:
                # No alpha provided - default to all pixels valid
                # (Legacy behavior for non-COLMAP pipelines)
                alpha_mask = np.ones((h, w), dtype=bool)

            # NOTE: We no longer filter dark/black pixels by color
            # The warped alpha channel already handles transformation borders correctly
            # This preserves legitimate black content in the original images
            
            bbox_img = img_data.get('bbox', (0, 0, w, h))
            
            # Apply scale factor if canvas was scaled down due to size limits
            if scale_factor != 1.0:
                x_off = int((bbox_img[0] - x_min) * scale_factor)
                y_off = int((bbox_img[1] - y_min) * scale_factor)
                # Scale the image down too
                scaled_w = max(1, int(w * scale_factor))
                scaled_h = max(1, int(h * scale_factor))
                img = cv2.resize(img, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
                alpha_mask = cv2.resize(alpha_mask.astype(np.uint8), (scaled_w, scaled_h), 
                                       interpolation=cv2.INTER_NEAREST).astype(bool)
                h, w = scaled_h, scaled_w
            else:
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
            
            # Log if image is being cut off
            if src_x_start > 0 or src_y_start > 0 or src_x_end < w or src_y_end < h:
                cut_pixels = max(src_x_start, src_y_start, w - src_x_end, h - src_y_end)
                if cut_pixels > 10:  # Only warn if significant
                    logger.warning(f"Image {idx} partially outside canvas (cut by ~{cut_pixels}px)")
            
            if src_x_end <= src_x_start or src_y_end <= src_y_start:
                logger.warning(f"Image {idx} completely outside canvas, skipping")
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
        
        # Auto-crop to remove white borders (unfilled areas)
        panorama = self._autocrop_panorama(panorama, filled_mask)
        
        # Cleanup
        del filled_mask
        gc.collect()
        
        # Now scale the CROPPED result to target size (if scale_to_target is enabled)
        scale_to_target = options.get('scale_to_target', True)
        if self.max_panorama_pixels and scale_to_target:
            h, w = panorama.shape[:2]
            current_pixels = h * w
            if current_pixels > 0:
                target_scale = np.sqrt(self.max_panorama_pixels / current_pixels)
                if abs(target_scale - 1.0) > 0.01:  # Only scale if needed
                    new_w = int(w * target_scale)
                    new_h = int(h * target_scale)
                    interp = cv2.INTER_CUBIC if target_scale > 1.0 else cv2.INTER_AREA
                    panorama = cv2.resize(panorama, (new_w, new_h), interpolation=interp)
                    logger.info(f"Scaled to target: {w}x{h} -> {new_w}x{new_h} ({new_w*new_h/1e6:.1f}MP)")
        
        logger.info("Blending complete")
        return panorama
    
    def _autocrop_panorama(self, panorama: np.ndarray, filled_mask: np.ndarray) -> np.ndarray:
        """
        Auto-crop panorama to remove unfilled borders.
        
        Args:
            panorama: The blended panorama
            filled_mask: Mask showing which pixels were filled
            
        Returns:
            Cropped panorama with borders removed
        """
        # Find content bounds from filled mask
        rows_with_content = np.any(filled_mask > 0, axis=1)
        cols_with_content = np.any(filled_mask > 0, axis=0)
        
        if not np.any(rows_with_content) or not np.any(cols_with_content):
            logger.warning("No content found in panorama, returning as-is")
            return panorama
        
        row_indices = np.where(rows_with_content)[0]
        col_indices = np.where(cols_with_content)[0]
        
        min_row, max_row = row_indices[0], row_indices[-1]
        min_col, max_col = col_indices[0], col_indices[-1]
        
        # Add small padding (5 pixels) for safety
        padding = 5
        min_row = max(0, min_row - padding)
        max_row = min(panorama.shape[0] - 1, max_row + padding)
        min_col = max(0, min_col - padding)
        max_col = min(panorama.shape[1] - 1, max_col + padding)
        
        cropped = panorama[min_row:max_row + 1, min_col:max_col + 1]
        
        original_size = panorama.shape[0] * panorama.shape[1]
        cropped_size = cropped.shape[0] * cropped.shape[1]
        
        if cropped_size < original_size:
            reduction = (1 - cropped_size / original_size) * 100
            logger.info(f"Auto-cropped: {panorama.shape[1]}x{panorama.shape[0]} -> {cropped.shape[1]}x{cropped.shape[0]} ({reduction:.1f}% reduction)")
        
        return cropped
    
    def _multiband_blend(self, aligned_images: List[Dict], options: Dict) -> np.ndarray:
        """Multi-band blending for seamless transitions (memory-optimized)"""
        if not aligned_images:
            raise ValueError("No images to blend")
        
        # Get options
        padding = options.get('padding', 50)
        zoom = options.get('zoom', 1.0)
        fit_all = options.get('fit_all', True)
        
        # Calculate bounding box with padding
        bbox = self._calculate_bbox(aligned_images, padding=padding if fit_all else 0)
        x_min, y_min, x_max, y_max = bbox
        
        output_h = y_max - y_min
        output_w = x_max - x_min
        
        # Validate dimensions
        if output_h <= 0 or output_w <= 0:
            logger.error(f"Invalid bounding box dimensions: {bbox}")
            return aligned_images[0]['image'].copy()
        
        # Sanity check: detect if alignment has "exploded"
        aligned_images, bbox = self._check_and_fix_exploded_alignment(aligned_images, bbox)
        x_min, y_min, x_max, y_max = bbox
        output_h = y_max - y_min
        output_w = x_max - x_min
        
        # Apply zoom factor
        if zoom != 1.0 and zoom > 0:
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            half_w = output_w / 2 / zoom
            half_h = output_h / 2 / zoom
            x_min = int(center_x - half_w)
            y_min = int(center_y - half_h)
            x_max = int(center_x + half_w)
            y_max = int(center_y + half_h)
            output_w = x_max - x_min
            output_h = y_max - y_min
            logger.info(f"Zoom {zoom}x applied: canvas size {output_w}x{output_h}")
        
        # Note: We blend at native resolution first, then scale after
        total_pixels = output_h * output_w
        scale_factor = 1.0  # No pre-scaling
        
        # Only scale down if MUCH larger than target to prevent memory issues
        max_blend_pixels = 200_000_000  # Max 200MP during blending to save memory
        if total_pixels > max_blend_pixels:
            scale_factor = np.sqrt(max_blend_pixels / total_pixels)
            output_h = int(output_h * scale_factor)
            output_w = int(output_w * scale_factor)
            logger.info(f"Blending at reduced size to save memory: {output_w}x{output_h}")
        
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
            
            # Calculate placement
            bbox_img = img_data.get('bbox', (0, 0, w, h))
            
            # Apply scale factor if canvas was scaled
            if scale_factor != 1.0:
                x_off = int((bbox_img[0] - x_min) * scale_factor)
                y_off = int((bbox_img[1] - y_min) * scale_factor)
                scaled_w = max(1, int(w * scale_factor))
                scaled_h = max(1, int(h * scale_factor))
                img = cv2.resize(img, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
                alpha_f = cv2.resize(alpha_f, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
                h, w = scaled_h, scaled_w
            else:
                x_off = bbox_img[0] - x_min
                y_off = bbox_img[1] - y_min
            
            # Create weight mask (distance from edges)
            weight = self._create_weight_mask((h, w), alpha_f)
            
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

        # Convert to uint8
        panorama = np.clip(panorama, 0, 255).astype(np.uint8)
        
        # Scale the result to target size (if scale_to_target is enabled)
        scale_to_target = options.get('scale_to_target', True)
        if self.max_panorama_pixels and scale_to_target:
            h, w = panorama.shape[:2]
            current_pixels = h * w
            if current_pixels > 0:
                target_scale = np.sqrt(self.max_panorama_pixels / current_pixels)
                if abs(target_scale - 1.0) > 0.01:  # Only scale if needed
                    new_w = int(w * target_scale)
                    new_h = int(h * target_scale)
                    interp = cv2.INTER_CUBIC if target_scale > 1.0 else cv2.INTER_AREA
                    panorama = cv2.resize(panorama, (new_w, new_h), interpolation=interp)
                    logger.info(f"Scaled to target: {w}x{h} -> {new_w}x{new_h} ({new_w*new_h/1e6:.1f}MP)")
        
        logger.info("Multiband blending complete")
        return panorama
    
    def _feather_blend(self, aligned_images: List[Dict], options: Dict) -> np.ndarray:
        """Feather blending (simpler, faster, memory-optimized)"""
        # Get options
        padding = options.get('padding', 50)
        zoom = options.get('zoom', 1.0)
        fit_all = options.get('fit_all', True)
        
        bbox = self._calculate_bbox(aligned_images, padding=padding if fit_all else 0)
        x_min, y_min, x_max, y_max = bbox
        
        output_h = y_max - y_min
        output_w = x_max - x_min
        
        # Sanity check: detect if alignment has "exploded"
        aligned_images, bbox = self._check_and_fix_exploded_alignment(aligned_images, bbox)
        x_min, y_min, x_max, y_max = bbox
        output_h = y_max - y_min
        output_w = x_max - x_min
        
        # Apply zoom factor
        if zoom != 1.0 and zoom > 0:
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            half_w = output_w / 2 / zoom
            half_h = output_h / 2 / zoom
            x_min = int(center_x - half_w)
            y_min = int(center_y - half_h)
            x_max = int(center_x + half_w)
            y_max = int(center_y + half_h)
            output_w = x_max - x_min
            output_h = y_max - y_min
        
        # Note: We blend at native resolution first, then scale after
        total_pixels = output_h * output_w
        scale_factor = 1.0  # No pre-scaling
        
        # Only scale down if MUCH larger than target to prevent memory issues
        max_blend_pixels = 200_000_000  # Max 200MP during blending to save memory
        if total_pixels > max_blend_pixels:
            scale_factor = np.sqrt(max_blend_pixels / total_pixels)
            output_h = int(output_h * scale_factor)
            output_w = int(output_w * scale_factor)
            logger.info(f"Blending at reduced size to save memory: {output_w}x{output_h}")
        
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
            
            bbox_img = img_data.get('bbox', (0, 0, w, h))
            
            # Apply scale factor if canvas was scaled
            if scale_factor != 1.0:
                x_off = int((bbox_img[0] - x_min) * scale_factor)
                y_off = int((bbox_img[1] - y_min) * scale_factor)
                scaled_w = max(1, int(w * scale_factor))
                scaled_h = max(1, int(h * scale_factor))
                img = cv2.resize(img, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
                alpha_f = cv2.resize(alpha_f, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
                h, w = scaled_h, scaled_w
            else:
                x_off = bbox_img[0] - x_min
                y_off = bbox_img[1] - y_min
            
            # Simple distance-based weight
            weight = self._create_weight_mask((h, w), alpha_f, feather_size=50)
            
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

        # Convert to uint8
        panorama = np.clip(panorama, 0, 255).astype(np.uint8)
        
        # Scale the result to target size (if scale_to_target is enabled)
        scale_to_target = options.get('scale_to_target', True)
        if self.max_panorama_pixels and scale_to_target:
            h, w = panorama.shape[:2]
            current_pixels = h * w
            if current_pixels > 0:
                target_scale = np.sqrt(self.max_panorama_pixels / current_pixels)
                if abs(target_scale - 1.0) > 0.01:  # Only scale if needed
                    new_w = int(w * target_scale)
                    new_h = int(h * target_scale)
                    interp = cv2.INTER_CUBIC if target_scale > 1.0 else cv2.INTER_AREA
                    panorama = cv2.resize(panorama, (new_w, new_h), interpolation=interp)
                    logger.info(f"Scaled to target: {w}x{h} -> {new_w}x{new_h} ({new_w*new_h/1e6:.1f}MP)")
        
        logger.info("Feather blending complete")
        return panorama
    
    def _linear_blend(self, aligned_images: List[Dict], options: Dict) -> np.ndarray:
        """Simple linear blending"""
        return self._feather_blend(aligned_images, options)
    
    def _calculate_bbox(self, aligned_images: List[Dict], padding: int = 0) -> Tuple[int, int, int, int]:
        """
        Calculate bounding box for all images with optional padding.
        
        Args:
            aligned_images: List of aligned image data
            padding: Extra pixels to add around all edges
            
        Returns:
            Tuple of (x_min, y_min, x_max, y_max)
        """
        if not aligned_images:
            return (0, 0, 100, 100)
        
        x_min = y_min = float('inf')
        x_max = y_max = float('-inf')
        
        for idx, img_data in enumerate(aligned_images):
            bbox = img_data.get('bbox')
            h, w = img_data['image'].shape[:2]
            
            if bbox and len(bbox) >= 4:
                # Use provided bounding box
                img_x_min, img_y_min, img_x_max, img_y_max = bbox
                x_min = min(x_min, img_x_min)
                y_min = min(y_min, img_y_min)
                x_max = max(x_max, img_x_max)
                y_max = max(y_max, img_y_max)
                logger.debug(f"Image {idx}: bbox=({img_x_min}, {img_y_min}, {img_x_max}, {img_y_max})")
            else:
                # No bbox provided, assume origin
                x_min = min(x_min, 0)
                y_min = min(y_min, 0)
                x_max = max(x_max, w)
                y_max = max(y_max, h)
                logger.debug(f"Image {idx}: no bbox, using (0, 0, {w}, {h})")
        
        # Validate bounding box
        if x_min == float('inf') or y_min == float('inf') or x_max == float('-inf') or y_max == float('-inf'):
            # Fallback: use first image dimensions
            h, w = aligned_images[0]['image'].shape[:2]
            logger.warning("Invalid bounding box, using first image dimensions")
            return (0, 0, w, h)
        
        # Ensure valid dimensions
        if x_max <= x_min:
            x_max = x_min + 100
        if y_max <= y_min:
            y_max = y_min + 100
        
        # Apply padding
        if padding > 0:
            x_min -= padding
            y_min -= padding
            x_max += padding
            y_max += padding
        
        logger.info(f"Total bounding box: ({int(x_min)}, {int(y_min)}) to ({int(x_max)}, {int(y_max)}) = {int(x_max-x_min)}x{int(y_max-y_min)} pixels")
        
        return (int(x_min), int(y_min), int(x_max), int(y_max))
    
    def _check_and_fix_exploded_alignment(
        self, 
        aligned_images: List[Dict], 
        bbox: Tuple[int, int, int, int]
    ) -> Tuple[List[Dict], Tuple[int, int, int, int]]:
        """
        Check if alignment has 'exploded' (images spread way too far apart).
        This happens when alignment errors accumulate with many images.
        
        If detected, compact the layout by creating a grid arrangement
        that preserves approximate relative positions.
        
        Returns:
            Tuple of (possibly modified aligned_images, possibly modified bbox)
        """
        if len(aligned_images) < 5:
            return aligned_images, bbox
        
        x_min, y_min, x_max, y_max = bbox
        canvas_area = (x_max - x_min) * (y_max - y_min)
        
        # Calculate total image content area
        total_content_area = 0
        for img_data in aligned_images:
            h, w = img_data['image'].shape[:2]
            total_content_area += h * w
        
        # Calculate expected canvas area with reasonable overlap (30%)
        # For n images with 30% overlap, expected area ≈ content_area * 0.7 * sqrt(n)/n * n = content_area * 0.7 * sqrt(n)
        n_images = len(aligned_images)
        expected_area = total_content_area * 0.7 * np.sqrt(n_images)
        
        # If canvas is way larger than expected (>10x), alignment has exploded
        area_ratio = canvas_area / max(expected_area, 1)
        
        if area_ratio > 10:
            logger.warning(f"Alignment exploded! Canvas area {canvas_area/1e6:.1f}MP is {area_ratio:.1f}x expected {expected_area/1e6:.1f}MP")
            logger.info("Compacting layout to create usable panorama...")
            
            # Get average image dimensions
            avg_h = int(np.mean([img['image'].shape[0] for img in aligned_images]))
            avg_w = int(np.mean([img['image'].shape[1] for img in aligned_images]))
            
            # Create compact grid layout
            # Grid dimensions that approximate square arrangement
            grid_cols = int(np.ceil(np.sqrt(n_images * avg_w / max(avg_h, 1))))
            grid_cols = max(1, min(grid_cols, n_images))
            grid_rows = int(np.ceil(n_images / grid_cols))
            
            # Overlap factor (0.2 = 20% overlap between adjacent images)
            overlap = 0.2
            step_x = int(avg_w * (1 - overlap))
            step_y = int(avg_h * (1 - overlap))
            
            # Get original positions and sort by them to maintain some spatial coherence
            positions = []
            for idx, img_data in enumerate(aligned_images):
                img_bbox = img_data.get('bbox', (0, 0, avg_w, avg_h))
                cx = (img_bbox[0] + img_bbox[2]) / 2
                cy = (img_bbox[1] + img_bbox[3]) / 2
                positions.append((idx, cx, cy))
            
            # Sort by y then x to create a reasonable ordering
            positions.sort(key=lambda p: (p[2] // (avg_h * 2), p[1]))
            
            # Assign new grid positions
            new_min_x, new_min_y = 0, 0
            new_max_x, new_max_y = 0, 0
            
            for grid_idx, (orig_idx, _, _) in enumerate(positions):
                row = grid_idx // grid_cols
                col = grid_idx % grid_cols
                
                new_x = col * step_x
                new_y = row * step_y
                
                img_data = aligned_images[orig_idx]
                h, w = img_data['image'].shape[:2]
                
                # Update bbox
                img_data['bbox'] = (new_x, new_y, new_x + w, new_y + h)
                
                new_max_x = max(new_max_x, new_x + w)
                new_max_y = max(new_max_y, new_y + h)
            
            # Update overall bbox
            padding = 50
            new_bbox = (new_min_x - padding, new_min_y - padding, 
                       new_max_x + padding, new_max_y + padding)
            
            logger.info(f"Compacted to {grid_cols}x{grid_rows} grid: {new_max_x}x{new_max_y} pixels")
            
            return aligned_images, new_bbox
        
        return aligned_images, bbox
    
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

