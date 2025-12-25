"""
Grid visualization for 2D alignment preview - "exploded view" style
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class GridVisualizer:
    """Visualize grid alignment as exploded view - images in relative positions with spacing"""
    
    def __init__(self, max_display_size: int = 600, spacing_pixels: int = 30):
        """
        Initialize grid visualizer
        
        Args:
            max_display_size: Maximum width/height for displayed images
            spacing_pixels: Minimum spacing between images in pixels
        """
        self.max_display_size = max_display_size
        self.spacing_pixels = spacing_pixels
    
    def create_preview_image(self, grid_layout: Dict) -> np.ndarray:
        """
        Create preview image from grid layout
        
        Args:
            grid_layout: Grid layout dictionary with images, positions, transforms
            
        Returns:
            Preview image as numpy array (RGB format), or None if error
        """
        try:
            images = grid_layout.get('images', [])
            positions = grid_layout.get('positions', [])
            transforms = grid_layout.get('transforms', {})
            
            if not images or not positions:
                logger.warning("Empty grid layout")
                return None
            
            canvas = self._create_canvas(images, positions, transforms)
            return canvas
        except Exception as e:
            logger.error(f"Error creating preview image: {e}", exc_info=True)
            return None
    
    def _rotate_image(
        self,
        img: np.ndarray,
        angle_degrees: float,
        scale: float = 1.0
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Rotate image around its center
        
        Returns:
            Rotated image and (new_width, new_height)
        """
        h, w = img.shape[:2]
        center = (w / 2, h / 2)
        
        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, -angle_degrees, scale)  # Negative for correct direction
        
        # Calculate new bounding box size
        cos = abs(M[0, 0])
        sin = abs(M[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)
        
        # Adjust rotation matrix for new size
        M[0, 2] += (new_w - w) / 2
        M[1, 2] += (new_h - h) / 2
        
        # Rotate with transparent/white background
        if len(img.shape) == 3:
            rotated = cv2.warpAffine(img, M, (new_w, new_h), 
                                      borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=(255, 255, 255))
                else:
            rotated = cv2.warpAffine(img, M, (new_w, new_h),
                                      borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=255)
        
        return rotated, (new_w, new_h)
    
    def _create_canvas(
        self,
        images: List[Dict],
        positions: List[Dict],
        transforms: Dict[int, np.ndarray]
    ) -> np.ndarray:
        """Create canvas with images placed and rotated according to positions"""
        
        if not positions:
            logger.error("No positions in grid layout")
            return None
        
        # First pass: calculate sizes after rotation/scaling for each image
        processed_images = {}
        for pos in positions:
            idx = pos.get('image_idx')
            if idx is None or idx >= len(images) or idx < 0:
                continue
            
            img_data = images[idx]
            img = img_data.get('image')
            if img is None or not isinstance(img, np.ndarray):
                continue
            
            rotation = pos.get('rotation', 0.0)
            scale = pos.get('scale', 1.0)
            
            # Scale image for display
            h, w = img.shape[:2]
            display_scale = min(self.max_display_size / max(w, h), 1.0) * scale
            
            if display_scale != 1.0:
                new_w = max(1, int(w * display_scale))
                new_h = max(1, int(h * display_scale))
                img_scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                img_scaled = img.copy()
            
            # Rotate image
            if abs(rotation) > 0.5:  # Only rotate if > 0.5 degrees
                img_rotated, (rot_w, rot_h) = self._rotate_image(img_scaled, rotation)
            else:
                img_rotated = img_scaled
                rot_h, rot_w = img_scaled.shape[:2]
            
            processed_images[idx] = {
                'image': img_rotated,
                'width': rot_w,
                'height': rot_h,
                'rotation': rotation,
                'display_scale': display_scale
            }
        
        if not processed_images:
            logger.error("No valid images to display")
            return None
        
        # Calculate canvas size based on center positions + image sizes
        max_x = 0
        max_y = 0
        min_x = float('inf')
        min_y = float('inf')
        
        for pos in positions:
            idx = pos.get('image_idx')
            if idx not in processed_images:
                    continue
                
            proc = processed_images[idx]
            cx = pos.get('center_x', pos.get('x', 0) + proc['width'] / 2)
            cy = pos.get('center_y', pos.get('y', 0) + proc['height'] / 2)
            
            # Account for scaled/rotated dimensions
            half_w = proc['width'] / 2
            half_h = proc['height'] / 2
            
            min_x = min(min_x, cx - half_w)
            min_y = min(min_y, cy - half_h)
            max_x = max(max_x, cx + half_w)
            max_y = max(max_y, cy + half_h)
        
        # Normalize so min is at padding
        padding = self.spacing_pixels
        offset_x = -min_x + padding
        offset_y = -min_y + padding
        
        canvas_w = int(max_x - min_x + 2 * padding)
        canvas_h = int(max_y - min_y + 2 * padding)
        
        # Limit canvas size
        max_canvas = 8000
        if canvas_w > max_canvas or canvas_h > max_canvas:
            scale_down = min(max_canvas / canvas_w, max_canvas / canvas_h)
            canvas_w = int(canvas_w * scale_down)
            canvas_h = int(canvas_h * scale_down)
            offset_x *= scale_down
            offset_y *= scale_down
            
            # Also scale processed images
            for idx in processed_images:
                proc = processed_images[idx]
                new_w = max(1, int(proc['width'] * scale_down))
                new_h = max(1, int(proc['height'] * scale_down))
                proc['image'] = cv2.resize(proc['image'], (new_w, new_h))
                proc['width'] = new_w
                proc['height'] = new_h
        
        # Create canvas with white background
        canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255
        
        # Place images on canvas
        for pos in positions:
            idx = pos.get('image_idx')
            if idx not in processed_images:
                    continue
                
            proc = processed_images[idx]
            img = proc['image']
            
            # Get center position with offset
            cx = pos.get('center_x', pos.get('x', 0) + proc['width'] / 2)
            cy = pos.get('center_y', pos.get('y', 0) + proc['height'] / 2)
            
            # Apply canvas normalization
            cx = int(cx + offset_x)
            cy = int(cy + offset_y)
            
            # Calculate top-left corner
            x = int(cx - proc['width'] / 2)
            y = int(cy - proc['height'] / 2)
            
            # Clip to canvas bounds
            src_x1 = max(0, -x)
            src_y1 = max(0, -y)
            dst_x1 = max(0, x)
            dst_y1 = max(0, y)
            
            src_x2 = min(proc['width'], canvas_w - x)
            src_y2 = min(proc['height'], canvas_h - y)
            dst_x2 = min(canvas_w, x + proc['width'])
            dst_y2 = min(canvas_h, y + proc['height'])
            
            if src_x2 <= src_x1 or src_y2 <= src_y1:
                        continue
            if dst_x2 <= dst_x1 or dst_y2 <= dst_y1:
                    continue
                
            # Convert to RGB if needed
            if len(img.shape) == 2:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    else:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Place image (handle white background from rotation as transparent)
            img_region = img_rgb[src_y1:src_y2, src_x1:src_x2]
            canvas_region = canvas[dst_y1:dst_y2, dst_x1:dst_x2]
            
            # Create mask for non-white pixels (from rotation background)
            if abs(pos.get('rotation', 0)) > 0.5:
                # Mask out white pixels (rotation background)
                white_threshold = 250
                mask = np.any(img_region < white_threshold, axis=2)
                mask_3d = mask[:, :, np.newaxis]
                
                # Blend: keep canvas where mask is False, use image where mask is True
                blended = np.where(mask_3d, img_region, canvas_region)
                canvas[dst_y1:dst_y2, dst_x1:dst_x2] = blended
            else:
                canvas[dst_y1:dst_y2, dst_x1:dst_x2] = img_region
            
            # Draw border
            border_color = (80, 80, 80)
            cv2.rectangle(canvas, (dst_x1, dst_y1), (dst_x2 - 1, dst_y2 - 1), border_color, 2)
            
            # Draw image index label
            label = str(idx)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
            
            label_x = dst_x1 + 5
            label_y = dst_y1 + text_h + 5
            
            # Background for label
            cv2.rectangle(canvas, 
                         (label_x - 2, label_y - text_h - 2),
                         (label_x + text_w + 2, label_y + 2),
                         (255, 255, 255), -1)
            cv2.putText(canvas, label, (label_x, label_y), font, font_scale, (0, 0, 0), thickness)
        
        return canvas
    
    def save_grid(self, grid_layout: Dict, output_path: str, quality: str = 'high', dpi: int = 300, postproc: Dict = None):
        """
        Save grid layout visualization
        
        Args:
            grid_layout: Grid layout dictionary
            output_path: Output file path
            quality: Quality preset ('ultra_high', 'high', 'medium', 'low', 'minimum')
            dpi: Output DPI (dots per inch)
            postproc: Post-processing options dict
        """
        canvas = self.create_preview_image(grid_layout)
        
        if canvas is None:
            logger.error("Failed to create canvas")
            return
        
        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert RGB to BGR for OpenCV
        canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
        
        # Apply post-processing if requested (on BGR image)
        if postproc:
            from core.post_processing import apply_post_processing
            canvas_bgr = apply_post_processing(canvas_bgr, postproc)
            # Update canvas (RGB) for TIFF saving
            canvas = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB)
            logger.info(f"Applied post-processing to grid")
        
        # Quality settings
        quality_settings = {
            'ultra_high': {'tiff_compression': None, 'png_level': 0, 'jpeg_quality': 100},
            'high': {'tiff_compression': 'lzw', 'png_level': 3, 'jpeg_quality': 95},
            'medium': {'tiff_compression': 'lzw', 'png_level': 6, 'jpeg_quality': 85},
            'low': {'tiff_compression': 'lzw', 'png_level': 8, 'jpeg_quality': 70},
            'minimum': {'tiff_compression': 'lzw', 'png_level': 9, 'jpeg_quality': 50},
        }
        settings = quality_settings.get(quality, quality_settings['high'])
        
        suffix = output_path.suffix.lower()
        
        if suffix in ['.tif', '.tiff']:
            self._save_tiff(canvas, output_path, settings['tiff_compression'], dpi)
        elif suffix in ['.png']:
            self._save_png(canvas_bgr, output_path, settings['png_level'], dpi)
        elif suffix in ['.jpg', '.jpeg']:
            self._save_jpeg(canvas_bgr, output_path, settings['jpeg_quality'], dpi)
        else:
            cv2.imwrite(str(output_path), canvas_bgr)
        
        logger.info(f"Grid visualization saved to {output_path}")

    def _save_tiff(self, image: np.ndarray, path: Path, compression: str, dpi: int):
        """Save as TIFF with proper fallback"""
        try:
            import tifffile
            
            # Calculate resolution (pixels per cm for TIFF)
            resolution = (dpi / 2.54, dpi / 2.54)
            
            try:
                if compression:
                    tifffile.imwrite(
                        str(path), 
                        image,
                        compression=compression,
                        photometric='rgb',
                        resolution=resolution,
                        resolutionunit='CENTIMETER'
                    )
                else:
                    tifffile.imwrite(
                        str(path), 
                        image,
                        photometric='rgb',
                        resolution=resolution,
                        resolutionunit='CENTIMETER'
                    )
                return
            except KeyError as e:
                if 'imagecodecs' in str(e):
                    logger.warning(f"Compression codec not available, saving uncompressed TIFF")
                    tifffile.imwrite(
                        str(path), 
                        image,
                        photometric='rgb',
                        resolution=resolution,
                        resolutionunit='CENTIMETER'
                    )
                    return
                raise
                
        except ImportError:
            logger.warning("tifffile not installed, using OpenCV")
        except Exception as e:
            logger.warning(f"tifffile failed: {e}, using OpenCV")
        
        # Fallback to OpenCV
        canvas_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(path), canvas_bgr)
    
    def _save_png(self, image: np.ndarray, path: Path, compression_level: int, dpi: int):
        """Save as PNG with DPI metadata"""
        cv2.imwrite(str(path), image, [cv2.IMWRITE_PNG_COMPRESSION, compression_level])
        try:
            from PIL import Image
            img = Image.open(str(path))
            img.save(str(path), dpi=(dpi, dpi))
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Could not set PNG DPI: {e}")
    
    def _save_jpeg(self, image: np.ndarray, path: Path, quality: int, dpi: int):
        """Save as JPEG with quality and DPI"""
        try:
            from PIL import Image
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(image_rgb)
            img.save(str(path), quality=quality, dpi=(dpi, dpi))
            return
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"PIL save failed: {e}, using OpenCV")
        
        cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, quality])
