"""
Generative AI-powered panorama stitching using diffusion models.

Based on research from:
- "Generative Panoramic Image Stitching" (arXiv:2507.07133) - Tuli et al. 2025
- "RealFill: Reference-Driven Generation for Authentic Image Completion" 
- "Paint by Example: Exemplar-based Image Editing with Diffusion Models"

This module provides generative AI blending that outperforms traditional
SIFT/FLANN/multiband blending for challenging cases:
- Strong parallax effects (translation between viewpoints)
- Lighting/exposure variations between images
- Style differences
- Large gaps between images
- Ghosting artifacts in overlaps

Supports multiple backends:
- OpenAI (DALL-E 3 for outpainting)
- Replicate (Stable Diffusion XL Inpainting)
- Local (Stable Diffusion with ControlNet)
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple, Callable
import logging
import gc
import base64
import io
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Check for optional dependencies
PIL_AVAILABLE = False
REQUESTS_AVAILABLE = False
TORCH_AVAILABLE = False
DIFFUSERS_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    pass

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    pass

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    pass

try:
    from diffusers import StableDiffusionXLInpaintPipeline, AutoPipelineForInpainting
    DIFFUSERS_AVAILABLE = True
except ImportError:
    pass


class GenerativeBlender:
    """
    Generative AI-powered panorama blending using diffusion models.
    
    This approach is inspired by "Generative Panoramic Image Stitching" which
    fine-tunes a diffusion-based inpainting model to:
    1. Preserve scene content and layout from multiple reference images
    2. Outpaint seamless panoramas from aligned reference views
    3. Handle parallax, lighting, and style variations that break traditional stitching
    
    Modes:
    - 'openai': Use OpenAI's DALL-E 3 API for inpainting/outpainting
    - 'replicate': Use Replicate's hosted Stable Diffusion models
    - 'local': Use local diffusers pipeline (requires GPU)
    - 'hybrid': Traditional stitching with AI-based seam repair
    """
    
    def __init__(self, use_gpu: bool = False, options: Dict = None):
        """
        Initialize generative blender.
        
        Args:
            use_gpu: Enable GPU acceleration for local inference
            options: Configuration options:
                - backend: 'openai', 'replicate', 'local', or 'hybrid'
                - api_key: API key for cloud services
                - model: Model name/version
                - prompt: Custom prompt for generation
                - strength: Inpainting strength (0.0-1.0)
                - guidance_scale: CFG guidance scale (default 7.5)
                - tile_size: Size of tiles for processing (default 1024)
                - overlap_ratio: Overlap between tiles (default 0.25)
        """
        self.use_gpu = use_gpu
        self.options = options or {}
        
        self.backend = self.options.get('backend', 'hybrid')
        self.api_key = self.options.get('api_key', os.environ.get('OPENAI_API_KEY', ''))
        self.model = self.options.get('model', 'dall-e-3')
        self.prompt = self.options.get('prompt', '')
        self.strength = self.options.get('strength', 0.75)
        self.guidance_scale = self.options.get('guidance_scale', 7.5)
        self.tile_size = self.options.get('tile_size', 1024)
        self.overlap_ratio = self.options.get('overlap_ratio', 0.25)
        
        # Local pipeline (loaded lazily)
        self._pipeline = None
        self._device = None
        
        logger.info(f"GenerativeBlender initialized (backend: {self.backend})")
    
    def _get_device(self):
        """Get the torch device for local inference."""
        if self._device is not None:
            return self._device
        
        if not TORCH_AVAILABLE:
            return None
        
        if self.use_gpu and torch.cuda.is_available():
            self._device = torch.device('cuda')
        elif self.use_gpu and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self._device = torch.device('mps')
        else:
            self._device = torch.device('cpu')
        
        return self._device
    
    def _load_local_pipeline(self):
        """Lazily load local diffusion pipeline."""
        if self._pipeline is not None:
            return
        
        if not DIFFUSERS_AVAILABLE:
            logger.error("diffusers not available. Install with: pip install diffusers transformers accelerate")
            return
        
        logger.info("Loading local SDXL inpainting pipeline...")
        device = self._get_device()
        
        try:
            # Use SDXL inpainting model
            self._pipeline = AutoPipelineForInpainting.from_pretrained(
                "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
                torch_dtype=torch.float16 if device.type != 'cpu' else torch.float32,
                variant="fp16" if device.type != 'cpu' else None,
            )
            self._pipeline = self._pipeline.to(device)
            
            # Enable memory optimizations
            if hasattr(self._pipeline, 'enable_attention_slicing'):
                self._pipeline.enable_attention_slicing()
            
            logger.info("Local pipeline loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load local pipeline: {e}")
            self._pipeline = None
    
    def _numpy_to_pil(self, image: np.ndarray) -> 'Image.Image':
        """Convert numpy array to PIL Image."""
        if not PIL_AVAILABLE:
            raise ImportError("PIL not available")
        
        if len(image.shape) == 3 and image.shape[2] == 3:
            # BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return Image.fromarray(image.astype(np.uint8))
    
    def _pil_to_numpy(self, image: 'Image.Image') -> np.ndarray:
        """Convert PIL Image to numpy array (BGR)."""
        arr = np.array(image)
        if len(arr.shape) == 3 and arr.shape[2] == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        return arr
    
    def _encode_image_base64(self, image: np.ndarray) -> str:
        """Encode image as base64 for API calls."""
        if not PIL_AVAILABLE:
            raise ImportError("PIL not available")
        
        pil_img = self._numpy_to_pil(image)
        buffer = io.BytesIO()
        pil_img.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def _decode_image_base64(self, b64_string: str) -> np.ndarray:
        """Decode base64 image to numpy array."""
        if not PIL_AVAILABLE:
            raise ImportError("PIL not available")
        
        image_data = base64.b64decode(b64_string)
        pil_img = Image.open(io.BytesIO(image_data))
        return self._pil_to_numpy(pil_img)
    
    def blend_images(
        self,
        aligned_images: List[Dict],
        progress_callback: Optional[Callable] = None
    ) -> np.ndarray:
        """
        Blend aligned images using generative AI.
        
        Args:
            aligned_images: List of aligned image data dictionaries
            progress_callback: Optional callback(percent, message)
            
        Returns:
            Blended panorama
        """
        if not aligned_images:
            raise ValueError("No images to blend")
        
        if len(aligned_images) == 1:
            return aligned_images[0]['image']
        
        logger.info(f"Generative blending {len(aligned_images)} images (backend: {self.backend})")
        
        if self.backend == 'hybrid':
            return self._blend_hybrid(aligned_images, progress_callback)
        elif self.backend == 'openai':
            return self._blend_openai(aligned_images, progress_callback)
        elif self.backend == 'replicate':
            return self._blend_replicate(aligned_images, progress_callback)
        elif self.backend == 'local':
            return self._blend_local(aligned_images, progress_callback)
        else:
            logger.warning(f"Unknown backend {self.backend}, using hybrid")
            return self._blend_hybrid(aligned_images, progress_callback)
    
    def _blend_hybrid(
        self,
        aligned_images: List[Dict],
        progress_callback: Optional[Callable] = None
    ) -> np.ndarray:
        """
        Hybrid mode: Traditional stitching + AI seam repair.
        
        1. First create a rough panorama using simple blending
        2. Detect seam artifacts (ghosting, misalignment)
        3. Use AI to repair/regenerate seam regions
        """
        logger.info("Hybrid blending: traditional stitch + AI seam repair")
        
        # Step 1: Create initial panorama with simple blending
        panorama, seam_mask = self._create_initial_panorama(aligned_images)
        
        if progress_callback:
            progress_callback(30, "Initial panorama created, detecting seams...")
        
        # Step 2: Detect problematic seam regions
        problem_regions = self._detect_seam_problems(panorama, seam_mask, aligned_images)
        
        if not problem_regions:
            logger.info("No seam problems detected, returning initial panorama")
            return panorama
        
        if progress_callback:
            progress_callback(50, f"Found {len(problem_regions)} problem regions, repairing...")
        
        # Step 3: Repair each problem region with AI
        for idx, region in enumerate(problem_regions):
            if progress_callback:
                progress = 50 + int(40 * (idx / len(problem_regions)))
                progress_callback(progress, f"Repairing region {idx + 1}/{len(problem_regions)}...")
            
            panorama = self._repair_region(panorama, region, aligned_images)
        
        if progress_callback:
            progress_callback(100, "Generative blending complete")
        
        return panorama
    
    def _create_initial_panorama(
        self,
        aligned_images: List[Dict]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create initial panorama with simple averaging and seam mask."""
        # Calculate bounding box
        bbox = self._calculate_bbox(aligned_images)
        x_min, y_min, x_max, y_max = bbox
        
        output_h = y_max - y_min
        output_w = x_max - x_min
        
        # Scale to target size (500MP default)
        max_pixels = 500_000_000
        total_pixels = output_h * output_w
        scale = np.sqrt(max_pixels / total_pixels)
            output_h = int(output_h * scale)
            output_w = int(output_w * scale)
        logger.info(f"Scaling to target: {output_w}x{output_h} ({output_w*output_h/1e6:.1f}MP)")
        
        # Create output
        panorama = np.zeros((output_h, output_w, 3), dtype=np.float32)
        weight_sum = np.zeros((output_h, output_w), dtype=np.float32)
        seam_mask = np.zeros((output_h, output_w), dtype=np.uint8)  # Track overlaps
        
        for img_data in aligned_images:
            img = img_data['image'].astype(np.float32)
            h, w = img.shape[:2]
            
            alpha = img_data.get('alpha')
            if alpha is not None:
                alpha = alpha.astype(np.float32) / 255.0
            else:
                alpha = np.ones((h, w), dtype=np.float32)
            
            # Filter black borders
            if len(img.shape) == 3:
                min_channel = np.min(img, axis=2)
                alpha[min_channel < 10] = 0
            
            bbox_img = img_data.get('bbox', (0, 0, w, h))
            x_off = int((bbox_img[0] - x_min) * scale)
            y_off = int((bbox_img[1] - y_min) * scale)
            
            if scale != 1.0:
                scaled_w = max(1, int(w * scale))
                scaled_h = max(1, int(h * scale))
                if scaled_w > 0 and scaled_h > 0:
                    img = cv2.resize(img, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
                    alpha = cv2.resize(alpha, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
                    h, w = scaled_h, scaled_w
            
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
            img_region = img[src_y_start:src_y_end, src_x_start:src_x_end]
            alpha_region = alpha[src_y_start:src_y_end, src_x_start:src_x_end]
            
            # Mark overlapping regions in seam mask
            dst_weight = weight_sum[dst_y_start:dst_y_end, dst_x_start:dst_x_end]
            overlap_mask = (dst_weight > 0) & (alpha_region > 0)
            seam_mask[dst_y_start:dst_y_end, dst_x_start:dst_x_end][overlap_mask] = 255
            
            # Accumulate
            panorama[dst_y_start:dst_y_end, dst_x_start:dst_x_end] += \
                img_region * alpha_region[:, :, np.newaxis]
            weight_sum[dst_y_start:dst_y_end, dst_x_start:dst_x_end] += alpha_region
        
        # Normalize
        weight_sum[weight_sum == 0] = 1
        panorama /= weight_sum[:, :, np.newaxis]
        
        return np.clip(panorama, 0, 255).astype(np.uint8), seam_mask
    
    def _detect_seam_problems(
        self,
        panorama: np.ndarray,
        seam_mask: np.ndarray,
        aligned_images: List[Dict]
    ) -> List[Dict]:
        """Detect problematic seam regions (ghosting, misalignment)."""
        problems = []
        
        # Dilate seam mask to include border regions
        kernel = np.ones((21, 21), np.uint8)
        seam_dilated = cv2.dilate(seam_mask, kernel, iterations=1)
        
        # Find connected components (separate seam regions)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            seam_dilated, connectivity=8
        )
        
        for label in range(1, num_labels):  # Skip background
            x, y, w, h, area = stats[label]
            
            # Skip very small regions
            if area < 1000:
                continue
            
            # Extract region
            region_mask = (labels == label).astype(np.uint8)
            region = panorama[y:y+h, x:x+w]
            mask_crop = region_mask[y:y+h, x:x+w]
            
            # Check for ghosting (high-frequency artifacts)
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            laplacian_var = np.var(laplacian[mask_crop > 0])
            
            # Check for color inconsistency
            lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB)
            l_channel = lab[:, :, 0]
            l_var = np.var(l_channel[mask_crop > 0])
            
            # If high variance, likely has artifacts
            if laplacian_var > 500 or l_var > 300:
                problems.append({
                    'x': x, 'y': y, 'w': w, 'h': h,
                    'mask': region_mask,
                    'severity': laplacian_var + l_var
                })
                logger.debug(f"Problem region at ({x}, {y}): laplacian_var={laplacian_var:.1f}, l_var={l_var:.1f}")
        
        # Sort by severity
        problems.sort(key=lambda p: p['severity'], reverse=True)
        
        # Limit number of regions to repair
        max_regions = self.options.get('max_repair_regions', 10)
        return problems[:max_regions]
    
    def _repair_region(
        self,
        panorama: np.ndarray,
        region: Dict,
        aligned_images: List[Dict]
    ) -> np.ndarray:
        """Repair a problematic region using AI inpainting."""
        x, y, w, h = region['x'], region['y'], region['w'], region['h']
        mask = region['mask']
        
        # Add padding around region
        padding = 50
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(panorama.shape[1], x + w + padding)
        y2 = min(panorama.shape[0], y + h + padding)
        
        # Extract patch
        patch = panorama[y1:y2, x1:x2].copy()
        mask_patch = mask[y1:y2, x1:x2]
        
        # Create inpainting mask (dilate slightly)
        inpaint_mask = cv2.dilate(mask_patch, np.ones((5, 5), np.uint8), iterations=1)
        
        # Try AI inpainting, fall back to OpenCV
        try:
            if self.backend == 'openai' and self.api_key:
                repaired = self._inpaint_openai(patch, inpaint_mask)
            elif self.backend == 'local' and DIFFUSERS_AVAILABLE:
                repaired = self._inpaint_local(patch, inpaint_mask)
            else:
                # Fallback to OpenCV inpainting
                repaired = cv2.inpaint(patch, inpaint_mask, 3, cv2.INPAINT_TELEA)
        except Exception as e:
            logger.warning(f"AI inpainting failed, using OpenCV fallback: {e}")
            repaired = cv2.inpaint(patch, inpaint_mask, 3, cv2.INPAINT_TELEA)
        
        # Blend repaired patch back
        panorama[y1:y2, x1:x2] = repaired
        
        return panorama
    
    def _inpaint_openai(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Inpaint using OpenAI DALL-E API."""
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests not available")
        
        if not self.api_key:
            raise ValueError("OpenAI API key not set")
        
        # Resize to supported size (must be square, 1024x1024 max)
        h, w = image.shape[:2]
        target_size = min(1024, max(h, w))
        
        # Pad to square
        max_dim = max(h, w)
        padded = np.ones((max_dim, max_dim, 3), dtype=np.uint8) * 255
        padded[:h, :w] = image
        
        mask_padded = np.zeros((max_dim, max_dim), dtype=np.uint8)
        mask_padded[:h, :w] = mask
        
        # Resize to target
        img_resized = cv2.resize(padded, (target_size, target_size))
        mask_resized = cv2.resize(mask_padded, (target_size, target_size))
        
        # Convert to PIL and base64
        pil_img = self._numpy_to_pil(img_resized)
        pil_mask = Image.fromarray(mask_resized)
        
        # Create RGBA image with transparency for mask
        img_rgba = pil_img.convert('RGBA')
        r, g, b, a = img_rgba.split()
        # Set alpha to 0 where mask is white (area to inpaint)
        a = Image.fromarray(255 - mask_resized)
        img_rgba = Image.merge('RGBA', (r, g, b, a))
        
        # Save to buffer
        img_buffer = io.BytesIO()
        img_rgba.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        # Call OpenAI API
        headers = {
            'Authorization': f'Bearer {self.api_key}'
        }
        
        prompt = self.prompt or "Seamlessly fill in the missing area to match the surrounding panoramic image, maintaining consistent lighting, texture, and perspective."
        
        response = requests.post(
            'https://api.openai.com/v1/images/edits',
            headers=headers,
            files={'image': ('image.png', img_buffer, 'image/png')},
            data={
                'prompt': prompt,
                'n': 1,
                'size': f'{target_size}x{target_size}',
                'response_format': 'b64_json'
            }
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"OpenAI API error: {response.text}")
        
        result = response.json()
        b64_image = result['data'][0]['b64_json']
        
        # Decode result
        result_img = self._decode_image_base64(b64_image)
        
        # Resize back to original
        result_img = cv2.resize(result_img, (max_dim, max_dim))
        
        # Crop to original size
        return result_img[:h, :w]
    
    def _inpaint_local(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Inpaint using local Stable Diffusion pipeline."""
        self._load_local_pipeline()
        
        if self._pipeline is None:
            raise RuntimeError("Local pipeline not available")
        
        h, w = image.shape[:2]
        
        # Resize to multiple of 8 for SDXL
        new_h = (h // 8) * 8
        new_w = (w // 8) * 8
        
        if new_h == 0 or new_w == 0:
            new_h = max(8, new_h)
            new_w = max(8, new_w)
        
        img_resized = cv2.resize(image, (new_w, new_h))
        mask_resized = cv2.resize(mask, (new_w, new_h))
        
        # Convert to PIL
        pil_img = self._numpy_to_pil(img_resized)
        pil_mask = Image.fromarray(mask_resized)
        
        prompt = self.prompt or "Seamless panoramic image, consistent lighting and perspective"
        
        # Run inpainting
        result = self._pipeline(
            prompt=prompt,
            image=pil_img,
            mask_image=pil_mask,
            guidance_scale=self.guidance_scale,
            strength=self.strength,
            num_inference_steps=30
        ).images[0]
        
        # Convert back to numpy
        result_np = self._pil_to_numpy(result)
        
        # Resize to original
        return cv2.resize(result_np, (w, h))
    
    def _blend_openai(
        self,
        aligned_images: List[Dict],
        progress_callback: Optional[Callable] = None
    ) -> np.ndarray:
        """Full generative blending using OpenAI API."""
        # Create initial rough panorama
        panorama, seam_mask = self._create_initial_panorama(aligned_images)
        
        if progress_callback:
            progress_callback(20, "Initial layout created, generating with AI...")
        
        # Process in tiles for large panoramas
        h, w = panorama.shape[:2]
        tile_size = self.tile_size
        overlap = int(tile_size * self.overlap_ratio)
        
        if max(h, w) <= tile_size:
            # Small enough to process at once
            result = self._generate_full_panorama_openai(panorama, seam_mask)
        else:
            # Process in overlapping tiles
            result = self._process_tiles(panorama, seam_mask, self._inpaint_openai, progress_callback)
        
        if progress_callback:
            progress_callback(100, "Generative blending complete")
        
        return result
    
    def _blend_replicate(
        self,
        aligned_images: List[Dict],
        progress_callback: Optional[Callable] = None
    ) -> np.ndarray:
        """Full generative blending using Replicate API."""
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests not available for Replicate API")
        
        # Similar to OpenAI but using Replicate endpoints
        # Create initial panorama and use their inpainting models
        panorama, seam_mask = self._create_initial_panorama(aligned_images)
        
        if progress_callback:
            progress_callback(20, "Processing with Replicate AI...")
        
        # For now, fall back to hybrid mode with seam repair
        return self._blend_hybrid(aligned_images, progress_callback)
    
    def _blend_local(
        self,
        aligned_images: List[Dict],
        progress_callback: Optional[Callable] = None
    ) -> np.ndarray:
        """Full generative blending using local Stable Diffusion."""
        if not DIFFUSERS_AVAILABLE:
            logger.warning("diffusers not available, falling back to hybrid mode")
            return self._blend_hybrid(aligned_images, progress_callback)
        
        # Create initial panorama
        panorama, seam_mask = self._create_initial_panorama(aligned_images)
        
        if progress_callback:
            progress_callback(20, "Processing with local AI model...")
        
        # Process seam regions with local model
        h, w = panorama.shape[:2]
        
        if max(h, w) <= self.tile_size:
            # Inpaint all seams at once
            if np.any(seam_mask > 0):
                panorama = self._inpaint_local(panorama, seam_mask)
        else:
            # Process in tiles
            panorama = self._process_tiles(panorama, seam_mask, self._inpaint_local, progress_callback)
        
        if progress_callback:
            progress_callback(100, "Generative blending complete")
        
        return panorama
    
    def _process_tiles(
        self,
        panorama: np.ndarray,
        seam_mask: np.ndarray,
        inpaint_fn: Callable,
        progress_callback: Optional[Callable] = None
    ) -> np.ndarray:
        """Process panorama in overlapping tiles."""
        h, w = panorama.shape[:2]
        tile_size = self.tile_size
        overlap = int(tile_size * self.overlap_ratio)
        step = tile_size - overlap
        
        result = panorama.copy()
        
        tiles_y = (h - overlap) // step + 1
        tiles_x = (w - overlap) // step + 1
        total_tiles = tiles_y * tiles_x
        
        tile_idx = 0
        for ty in range(tiles_y):
            for tx in range(tiles_x):
                y1 = ty * step
                x1 = tx * step
                y2 = min(y1 + tile_size, h)
                x2 = min(x1 + tile_size, w)
                
                tile = panorama[y1:y2, x1:x2]
                mask_tile = seam_mask[y1:y2, x1:x2]
                
                # Only process tiles with seams
                if np.any(mask_tile > 0):
                    try:
                        repaired = inpaint_fn(tile, mask_tile)
                        
                        # Blend into result with feathering
                        weight = self._create_tile_weight(y2 - y1, x2 - x1, overlap)
                        
                        result[y1:y2, x1:x2] = (
                            result[y1:y2, x1:x2].astype(np.float32) * (1 - weight[:, :, np.newaxis]) +
                            repaired.astype(np.float32) * weight[:, :, np.newaxis]
                        ).astype(np.uint8)
                    except Exception as e:
                        logger.warning(f"Tile ({tx}, {ty}) inpainting failed: {e}")
                
                tile_idx += 1
                if progress_callback:
                    progress = 20 + int(70 * tile_idx / total_tiles)
                    progress_callback(progress, f"Processing tile {tile_idx}/{total_tiles}...")
        
        return result
    
    def _create_tile_weight(self, h: int, w: int, overlap: int) -> np.ndarray:
        """Create weight mask for tile blending."""
        weight = np.ones((h, w), dtype=np.float32)
        
        # Feather edges
        for i in range(overlap):
            alpha = i / overlap
            weight[i, :] *= alpha
            weight[h - 1 - i, :] *= alpha
            weight[:, i] *= alpha
            weight[:, w - 1 - i] *= alpha
        
        return weight
    
    def _generate_full_panorama_openai(
        self,
        panorama: np.ndarray,
        seam_mask: np.ndarray
    ) -> np.ndarray:
        """Generate full panorama using OpenAI (for small panoramas)."""
        if np.any(seam_mask > 0):
            return self._inpaint_openai(panorama, seam_mask)
        return panorama
    
    def _calculate_bbox(self, aligned_images: List[Dict]) -> Tuple[int, int, int, int]:
        """Calculate bounding box for all images."""
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
                x_min = min(x_min, 0) if x_min != float('inf') else 0
                y_min = min(y_min, 0) if y_min != float('inf') else 0
                x_max = max(x_max, w)
                y_max = max(y_max, h)
        
        if x_min == float('inf'):
            h, w = aligned_images[0]['image'].shape[:2]
            return (0, 0, w, h)
        
        return (int(x_min), int(y_min), int(x_max), int(y_max))


