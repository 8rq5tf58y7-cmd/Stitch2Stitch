"""
AI-powered semantic blending with deep learning models
Supports SAM (Segment Anything), DeepLabV3, and texture-aware segmentation
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
import gc

logger = logging.getLogger(__name__)

# Check for scikit-image (texture analysis)
try:
    from skimage.segmentation import slic, felzenszwalb
    from skimage.filters import gabor
    from skimage.feature import local_binary_pattern
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    logger.warning("scikit-image not available. Install with: pip install scikit-image")

# Check for PyTorch and torchvision (deep learning)
TORCH_AVAILABLE = False
TORCHVISION_AVAILABLE = False
SAM_AVAILABLE = False
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
    try:
        import torchvision
        from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101
        from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, DeepLabV3_ResNet101_Weights
        TORCHVISION_AVAILABLE = True
    except ImportError:
        logger.warning("torchvision not available for DeepLabV3 segmentation")
    
    # Check for Segment Anything Model
    try:
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
        SAM_AVAILABLE = True
    except ImportError:
        # Try mobile_sam as alternative
        try:
            from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
            SAM_AVAILABLE = True
            logger.info("Using MobileSAM for segmentation")
        except ImportError:
            logger.info("SAM not available. Install with: pip install segment-anything")
except ImportError:
    logger.warning("PyTorch not available. Deep learning features disabled.")


class SemanticBlender:
    """
    AI-powered semantic blending with multiple segmentation backends.
    
    Modes:
    - 'sam': Segment Anything Model (best for any images, auto-downloads)
    - 'deeplab': DeepLabV3 neural network (best for general scenes, people, objects)
    - 'hybrid': Combined edge + texture + color + superpixel (best for microscopy)
    - 'texture': Gabor + LBP texture analysis
    - 'edge': Multi-scale edge detection
    - 'superpixel': SLIC superpixel boundaries
    - 'auto': Automatically selects best method based on available libraries
    
    Models are loaded lazily on first use and can be auto-downloaded.
    """
    
    def __init__(self, use_gpu: bool = False, options: Dict = None):
        """
        Initialize semantic blender
        
        Args:
            use_gpu: Enable GPU acceleration for deep learning
            options: Configuration options:
                - mode: 'auto', 'sam', 'deeplab', 'hybrid', 'texture', 'edge', 'superpixel'
                - model: Model variant:
                    - For SAM: 'mobile_sam', 'sam_vit_b', 'sam_vit_l', 'sam_vit_h'
                    - For DeepLab: 'resnet50', 'resnet101'
                - auto_download: Auto-download models if missing (default True)
                - edge_weight: Weight for edge preservation (0.0-1.0, default 0.7)
                - texture_weight: Weight for texture boundaries (0.0-1.0, default 0.5)
                - superpixel_segments: Number of superpixels (default 500)
                - gabor_frequencies: Gabor filter frequencies
                - foreground_boost: Extra weight for detected foreground (default 1.5)
                - sam_points_per_side: Grid density for SAM (default 32)
                - pixel_selection: 'blend' (weighted average) or 'select' (autostitch-style winner-take-all)
        """
        self.use_gpu = use_gpu
        self.options = options or {}
        self.mode = self.options.get('mode', 'auto')
        self.model_name = self.options.get('model', 'mobile_sam')
        self.auto_download = self.options.get('auto_download', True)
        self.edge_weight = self.options.get('edge_weight', 0.7)
        self.texture_weight = self.options.get('texture_weight', 0.5)
        self.superpixel_segments = self.options.get('superpixel_segments', 500)
        self.foreground_boost = self.options.get('foreground_boost', 1.5)
        self.sam_points_per_side = self.options.get('sam_points_per_side', 32)
        # Pixel selection mode: 
        # - 'blend': weighted average (default)
        # - 'select': autostitch-style winner-take-all
        # - 'pairwise': only consider best 2 images per overlap point (reduces burst photo noise)
        # Check both 'pixel_selection' and 'semantic_pixel_selection' for GUI compatibility
        self.pixel_selection = self.options.get('semantic_pixel_selection', 
                                                 self.options.get('pixel_selection', 'blend'))
        
        # Deep learning models (loaded lazily)
        self._dl_model = None
        self._sam_model = None
        self._sam_mask_generator = None
        self._device = None
        
        # Determine actual mode based on availability
        if self.mode == 'auto':
            if SAM_AVAILABLE:
                self.mode = 'sam'
            elif TORCHVISION_AVAILABLE:
                self.mode = 'deeplab'
            elif SKIMAGE_AVAILABLE:
                self.mode = 'hybrid'
            else:
                self.mode = 'edge'
        
        # Validate mode
        if self.mode == 'sam' and not SAM_AVAILABLE:
            logger.warning("SAM requested but not available, falling back to deeplab")
            self.mode = 'deeplab' if TORCHVISION_AVAILABLE else 'hybrid'
        
        if self.mode == 'deeplab' and not TORCHVISION_AVAILABLE:
            logger.warning("DeepLab requested but torchvision not available, falling back to hybrid")
            self.mode = 'hybrid' if SKIMAGE_AVAILABLE else 'edge'
        
        logger.info(f"Semantic blender initialized (mode: {self.mode}, GPU: {use_gpu})")
    
    def _get_device(self):
        """Get the torch device for inference"""
        if self._device is not None:
            return self._device
        
        if not TORCH_AVAILABLE:
            return None
        
        if self.use_gpu and torch.cuda.is_available():
            self._device = torch.device('cuda')
            logger.info("Using CUDA GPU for deep learning")
        elif self.use_gpu and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self._device = torch.device('mps')
            logger.info("Using Apple MPS for deep learning")
        else:
            self._device = torch.device('cpu')
            logger.info("Using CPU for deep learning")
        
        return self._device
    
    def _load_sam_model(self):
        """Lazily load Segment Anything Model"""
        if self._sam_model is not None:
            return
        
        if not SAM_AVAILABLE:
            logger.error("Cannot load SAM: segment-anything not available")
            return
        
        logger.info(f"Loading SAM model ({self.model_name})...")
        
        try:
            device = self._get_device()
            
            # Try to get model checkpoint
            model_path = None
            
            # First check if auto-download is enabled
            if self.auto_download:
                try:
                    from utils.model_downloader import ensure_model, get_model_path
                    model_path = ensure_model(self.model_name)
                except ImportError:
                    logger.warning("Model downloader not available")
                except Exception as e:
                    logger.warning(f"Auto-download failed: {e}")
            
            # Try to find model in common locations
            if model_path is None:
                from pathlib import Path
                possible_paths = [
                    Path.home() / '.stitch2stitch' / 'models',
                    Path.home() / '.cache' / 'sam',
                    Path(__file__).parent.parent.parent / 'models',
                    Path('.') / 'models',
                ]
                
                model_files = {
                    'mobile_sam': 'mobile_sam.pt',
                    'sam_vit_b': 'sam_vit_b_01ec64.pth',
                    'sam_vit_l': 'sam_vit_l_0b3195.pth',
                    'sam_vit_h': 'sam_vit_h_4b8939.pth',
                }
                
                filename = model_files.get(self.model_name, f'{self.model_name}.pth')
                
                for base_path in possible_paths:
                    candidate = base_path / filename
                    if candidate.exists():
                        model_path = candidate
                        break
            
            if model_path is None:
                logger.error(f"SAM model not found. Run: python -m utils.model_downloader --download {self.model_name}")
                return
            
            logger.info(f"Loading SAM from {model_path}")
            
            # Determine model type
            if 'mobile' in self.model_name:
                model_type = 'vit_t'  # MobileSAM uses tiny ViT
            elif 'vit_b' in self.model_name:
                model_type = 'vit_b'
            elif 'vit_l' in self.model_name:
                model_type = 'vit_l'
            elif 'vit_h' in self.model_name:
                model_type = 'vit_h'
            else:
                model_type = 'vit_b'  # Default
            
            # Load model
            self._sam_model = sam_model_registry[model_type](checkpoint=str(model_path))
            self._sam_model.to(device=device)
            self._sam_model.eval()
            
            # Create mask generator
            self._sam_mask_generator = SamAutomaticMaskGenerator(
                model=self._sam_model,
                points_per_side=self.sam_points_per_side,
                pred_iou_thresh=0.88,
                stability_score_thresh=0.95,
                crop_n_layers=1,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=100,  # Filter small regions
            )
            
            logger.info(f"SAM model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load SAM model: {e}", exc_info=True)
            self._sam_model = None
            self._sam_mask_generator = None
    
    def _run_sam(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run SAM segmentation on an image.
        
        Returns:
            Tuple of (boundary_mask, segment_mask)
        """
        if self._sam_mask_generator is None:
            self._load_sam_model()
        
        if self._sam_mask_generator is None:
            # Fallback
            return self._detect_edges(image), None
        
        try:
            # SAM expects RGB
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize for faster inference on large images
            h, w = image.shape[:2]
            max_size = 1024
            scale = min(max_size / max(h, w), 1.0)
            if scale < 1.0:
                new_h, new_w = int(h * scale), int(w * scale)
                rgb_small = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                rgb_small = rgb
                new_h, new_w = h, w
            
            # Generate masks
            masks = self._sam_mask_generator.generate(rgb_small)
            
            if not masks:
                logger.warning("SAM generated no masks, falling back to edges")
                return self._detect_edges(image), None
            
            # Create segment map (each segment gets unique ID)
            segment_map = np.zeros((new_h, new_w), dtype=np.int32)
            
            # Sort by area (largest first) and assign IDs
            masks_sorted = sorted(masks, key=lambda x: x['area'], reverse=True)
            for idx, mask_data in enumerate(masks_sorted):
                mask = mask_data['segmentation']
                segment_map[mask] = idx + 1
            
            # Find boundaries between segments
            boundary = np.zeros((new_h, new_w), dtype=np.uint8)
            boundary[:-1, :] |= (segment_map[:-1, :] != segment_map[1:, :]).astype(np.uint8) * 255
            boundary[:, :-1] |= (segment_map[:, :-1] != segment_map[:, 1:]).astype(np.uint8) * 255
            
            # Dilate boundaries
            kernel = np.ones((5, 5), np.uint8)
            boundary = cv2.dilate(boundary, kernel, iterations=2)
            
            # Resize back to original size
            if scale < 1.0:
                boundary = cv2.resize(boundary, (w, h), interpolation=cv2.INTER_NEAREST)
                segment_map = cv2.resize(segment_map.astype(np.float32), (w, h), 
                                        interpolation=cv2.INTER_NEAREST).astype(np.int32)
            
            logger.debug(f"SAM found {len(masks)} segments")
            return boundary, segment_map
            
        except Exception as e:
            logger.error(f"SAM inference failed: {e}")
            return self._detect_edges(image), None
    
    def _load_deeplab_model(self):
        """Lazily load DeepLabV3 model"""
        if self._dl_model is not None:
            return
        
        if not TORCHVISION_AVAILABLE:
            logger.error("Cannot load DeepLab: torchvision not available")
            return
        
        logger.info(f"Loading DeepLabV3 model ({self.model_name})...")
        
        try:
            device = self._get_device()
            
            # Load model with pretrained weights
            if self.model_name == 'resnet101':
                weights = DeepLabV3_ResNet101_Weights.DEFAULT
                self._dl_model = deeplabv3_resnet101(weights=weights)
            else:
                weights = DeepLabV3_ResNet50_Weights.DEFAULT
                self._dl_model = deeplabv3_resnet50(weights=weights)
            
            self._dl_model = self._dl_model.to(device)
            self._dl_model.eval()
            
            # Store preprocessing transform
            self._preprocess = weights.transforms()
            
            logger.info(f"DeepLabV3 model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load DeepLabV3 model: {e}")
            self._dl_model = None
    
    def _run_deeplab(self, image: np.ndarray) -> np.ndarray:
        """
        Run DeepLabV3 semantic segmentation on an image.
        
        Returns:
            Semantic mask where each class has a unique ID
        """
        if self._dl_model is None:
            self._load_deeplab_model()
        
        if self._dl_model is None:
            # Fallback to edge detection
            return self._detect_edges(image)
        
        try:
            # Convert BGR to RGB
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Convert to tensor and preprocess
            from PIL import Image
            pil_img = Image.fromarray(rgb)
            
            # Resize for inference (DeepLab works best at certain sizes)
            h, w = image.shape[:2]
            max_size = 520  # DeepLab default
            scale = min(max_size / h, max_size / w, 1.0)
            if scale < 1.0:
                new_h, new_w = int(h * scale), int(w * scale)
                pil_img = pil_img.resize((new_w, new_h), Image.BILINEAR)
            
            # Preprocess
            input_tensor = self._preprocess(pil_img).unsqueeze(0).to(self._device)
            
            # Run inference
            with torch.no_grad():
                output = self._dl_model(input_tensor)['out'][0]
                output = output.argmax(0).cpu().numpy()
            
            # Resize back to original size
            if scale < 1.0:
                output = cv2.resize(output.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            
            return output.astype(np.uint8)
            
        except Exception as e:
            logger.error(f"DeepLab inference failed: {e}")
            return self._detect_edges(image)
    
    def _semantic_to_boundary(self, semantic_mask: np.ndarray) -> np.ndarray:
        """Convert semantic segmentation mask to boundary mask"""
        h, w = semantic_mask.shape[:2]
        boundary = np.zeros((h, w), dtype=np.uint8)
        
        # Find boundaries between different classes
        boundary[:-1, :] |= (semantic_mask[:-1, :] != semantic_mask[1:, :]).astype(np.uint8) * 255
        boundary[:, :-1] |= (semantic_mask[:, :-1] != semantic_mask[:, 1:]).astype(np.uint8) * 255
        
        # Dilate for wider boundaries
        kernel = np.ones((5, 5), np.uint8)
        boundary = cv2.dilate(boundary, kernel, iterations=2)
        
        return boundary
    
    def _semantic_to_foreground(self, semantic_mask: np.ndarray) -> np.ndarray:
        """
        Convert semantic segmentation to foreground mask.
        
        COCO classes considered foreground (people, animals, vehicles, etc.):
        - 0: background
        - 1-20: various foreground classes
        """
        # In COCO/Pascal VOC, 0 is background, 1+ are foreground objects
        foreground = (semantic_mask > 0).astype(np.float32)
        
        # Smooth the mask
        foreground = cv2.GaussianBlur(foreground, (15, 15), 0)
        
        return foreground
    
    def blend_images(
        self,
        aligned_images: List[Dict],
        preserve_foreground: bool = True
    ) -> np.ndarray:
        """
        Blend images with semantic awareness
        
        Args:
            aligned_images: List of aligned image data dictionaries
            preserve_foreground: Boost weight for detected foreground objects
            
        Returns:
            Blended panorama
        """
        if not aligned_images:
            raise ValueError("No images to blend")
        
        if len(aligned_images) == 1:
            return aligned_images[0]['image']
        
        # Use autostitch-style selection if requested
        if self.pixel_selection == 'select':
            return self._blend_images_select(aligned_images, preserve_foreground)
        elif self.pixel_selection == 'pairwise':
            return self._blend_images_pairwise(aligned_images, preserve_foreground)
        
        logger.info(f"Semantic blending {len(aligned_images)} images (mode: {self.mode})")
        
        # Calculate bounding box
        bbox = self._calculate_bbox(aligned_images)
        x_min, y_min, x_max, y_max = bbox
        
        output_h = y_max - y_min
        output_w = x_max - x_min
        
        if output_h <= 0 or output_w <= 0:
            logger.error(f"Invalid bounding box: {bbox}")
            return aligned_images[0]['image'].copy()
        
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
        
        # Extract boundary and foreground masks for all images
        boundary_masks = []
        foreground_masks = []
        
        for idx, img_data in enumerate(aligned_images):
            img = img_data['image']
            logger.debug(f"Processing semantic masks for image {idx + 1}/{len(aligned_images)}")
            
            if self.mode == 'sam':
                # Segment Anything Model - works on any image type
                boundary, segment_map = self._run_sam(img)
                # SAM doesn't distinguish foreground/background, use segment density
                foreground = None
            elif self.mode == 'deeplab':
                # Deep learning segmentation
                semantic = self._run_deeplab(img)
                boundary = self._semantic_to_boundary(semantic)
                foreground = self._semantic_to_foreground(semantic) if preserve_foreground else None
            elif self.mode == 'hybrid':
                boundary = self._detect_hybrid_boundaries(img)
                foreground = None
            elif self.mode == 'texture':
                boundary = self._detect_texture_boundaries(img)
                foreground = None
            elif self.mode == 'superpixel':
                boundary = self._detect_superpixel_boundaries(img)
                foreground = None
            else:  # edge
                boundary = self._detect_edges(img)
                foreground = None
            
            boundary_masks.append(boundary)
            foreground_masks.append(foreground)
        
        # Blend each image
        for idx, img_data in enumerate(aligned_images):
            img = img_data['image'].astype(np.float32)
            h, w = img.shape[:2]
            
            alpha = img_data.get('alpha')
            if alpha is not None:
                alpha = alpha.astype(np.float32) / 255.0
            else:
                alpha = np.ones((h, w), dtype=np.float32)
            
            # Create semantic-aware weight mask
            boundary_mask = boundary_masks[idx] if idx < len(boundary_masks) else None
            foreground_mask = foreground_masks[idx] if idx < len(foreground_masks) else None
            
            weight = self._create_semantic_weight_mask(
                (h, w), alpha, boundary_mask, foreground_mask, preserve_foreground
            )
            
            # Place in panorama
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
            img_region = img[src_y_start:src_y_end, src_x_start:src_x_end]
            weight_region = weight[src_y_start:src_y_end, src_x_start:src_x_end]
            alpha_region = alpha[src_y_start:src_y_end, src_x_start:src_x_end]
            
            combined_weight = weight_region * alpha_region
            
            # Accumulate
            panorama[dst_y_start:dst_y_end, dst_x_start:dst_x_end] += \
                img_region * combined_weight[:, :, np.newaxis]
            weight_sum[dst_y_start:dst_y_end, dst_x_start:dst_x_end] += combined_weight
        
        # Normalize
        weight_sum[weight_sum == 0] = 1
        panorama /= weight_sum[:, :, np.newaxis]
        
        # Cleanup
        del boundary_masks, foreground_masks
        gc.collect()
        
        logger.info("Semantic blending complete")
        return np.clip(panorama, 0, 255).astype(np.uint8)
    
    def _blend_images_select(
        self,
        aligned_images: List[Dict],
        preserve_foreground: bool = True
    ) -> np.ndarray:
        """
        AutoStitch-style blending with semantic seam guidance.
        
        Uses hybrid/semantic boundary detection to compute weights, but instead of
        averaging pixels, selects the pixel from the image with the highest weight
        at each location (winner-take-all). This gives crisp, non-blurred results
        while still placing seams intelligently at texture/edge boundaries.
        
        Args:
            aligned_images: List of aligned image data dictionaries
            preserve_foreground: Boost weight for detected foreground objects
            
        Returns:
            Blended panorama with no averaging in overlap regions
        """
        logger.info(f"Semantic SELECT blending {len(aligned_images)} images (mode: {self.mode})")
        
        # Calculate bounding box
        bbox = self._calculate_bbox(aligned_images)
        x_min, y_min, x_max, y_max = bbox
        
        output_h = y_max - y_min
        output_w = x_max - x_min
        
        if output_h <= 0 or output_w <= 0:
            logger.error(f"Invalid bounding box: {bbox}")
            return aligned_images[0]['image'].copy()
        
        # Scale to target size (500MP default)
        max_pixels = 500_000_000
        scale_factor = 1.0
        total_pixels = output_h * output_w
        # Always scale to target for consistent output size
        scale_factor = np.sqrt(max_pixels / total_pixels)
            output_h = int(output_h * scale_factor)
            output_w = int(output_w * scale_factor)
        logger.info(f"Scaling to target: {output_w}x{output_h} ({output_w*output_h/1e6:.1f}MP)")
        
        # Create output canvas (white background like autostitch)
        panorama = np.ones((output_h, output_w, 3), dtype=np.uint8) * 255
        # Track the best weight at each pixel (for winner selection)
        best_weight = np.zeros((output_h, output_w), dtype=np.float32)
        
        # First pass: compute semantic boundaries for all images
        boundary_masks = []
        for idx, img_data in enumerate(aligned_images):
            img = img_data['image']
            logger.debug(f"Computing semantic boundaries for image {idx + 1}/{len(aligned_images)}")
            
            if self.mode == 'sam':
                boundary, _ = self._run_sam(img)
            elif self.mode == 'deeplab':
                semantic = self._run_deeplab(img)
                boundary = self._semantic_to_boundary(semantic)
            elif self.mode == 'hybrid':
                boundary = self._detect_hybrid_boundaries(img)
            elif self.mode == 'texture':
                boundary = self._detect_texture_boundaries(img)
            elif self.mode == 'superpixel':
                boundary = self._detect_superpixel_boundaries(img)
            else:  # edge
                boundary = self._detect_edges(img)
            
            boundary_masks.append(boundary)
        
        # Sort images by quality (best first, like autostitch)
        sorted_indices = sorted(
            range(len(aligned_images)),
            key=lambda i: aligned_images[i].get('quality', 0.5),
            reverse=True
        )
        
        # Second pass: place pixels using winner-take-all with semantic weights
        for idx in sorted_indices:
            img_data = aligned_images[idx]
            img = img_data['image']
            h, w = img.shape[:2]
            
            # Get alpha mask
            alpha = img_data.get('alpha')
            if alpha is not None:
                alpha_mask = alpha > 127
            else:
                alpha_mask = np.ones((h, w), dtype=bool)
            
            # Filter black borders
            if len(img.shape) == 3:
                min_channel = np.min(img, axis=2)
                black_border = min_channel < 10
                alpha_mask = alpha_mask & (~black_border)
            
            # Compute semantic weight for this image
            boundary_mask = boundary_masks[idx]
            weight = self._create_select_weight_mask((h, w), boundary_mask)
            
            # Calculate position in panorama
            bbox_img = img_data.get('bbox', (0, 0, w, h))
            
            if scale_factor != 1.0:
                x_off = int((bbox_img[0] - x_min) * scale_factor)
                y_off = int((bbox_img[1] - y_min) * scale_factor)
                scaled_w = max(1, int(w * scale_factor))
                scaled_h = max(1, int(h * scale_factor))
                if scaled_w > 0 and scaled_h > 0:
                    img = cv2.resize(img, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
                    alpha_mask = cv2.resize(alpha_mask.astype(np.uint8), (scaled_w, scaled_h),
                                           interpolation=cv2.INTER_NEAREST).astype(bool)
                    weight = cv2.resize(weight, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
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
            
            if src_x_end <= src_x_start or src_y_end <= src_y_start:
                continue
            
            # Extract regions
            img_region = img[src_y_start:src_y_end, src_x_start:src_x_end]
            alpha_region = alpha_mask[src_y_start:src_y_end, src_x_start:src_x_end]
            weight_region = weight[src_y_start:src_y_end, src_x_start:src_x_end]
            
            # Get current best weights in destination
            dst_best = best_weight[dst_y_start:dst_y_end, dst_x_start:dst_x_end]
            
            # Winner-take-all: write pixel if this image has higher weight AND valid alpha
            write_mask = alpha_region & (weight_region > dst_best)
            
            if np.any(write_mask):
                # Write pixels where this image wins
                dst_region = panorama[dst_y_start:dst_y_end, dst_x_start:dst_x_end]
                dst_region[write_mask] = img_region[write_mask]
                
                # Update best weights
                dst_best[write_mask] = weight_region[write_mask]
        
        # Cleanup
        del boundary_masks, best_weight
        gc.collect()
        
        logger.info("Semantic SELECT blending complete")
        return panorama
    
    def _create_select_weight_mask(
        self,
        shape: Tuple[int, int],
        boundary_mask: Optional[np.ndarray]
    ) -> np.ndarray:
        """
        Create weight mask for autostitch-style selection.
        
        Higher weight = more likely to be selected as the winner.
        Uses distance from image center + semantic boundary awareness.
        """
        h, w = shape
        
        # Base weight: distance from center (center pixels preferred)
        y_coords = np.arange(h).reshape(-1, 1)
        x_coords = np.arange(w).reshape(1, -1)
        center_y, center_x = h / 2, w / 2
        
        # Normalized distance from center (0 at edges, 1 at center)
        dist_y = 1.0 - np.abs(y_coords - center_y) / center_y
        dist_x = 1.0 - np.abs(x_coords - center_x) / center_x
        weight = dist_y * dist_x  # Highest at center
        
        # Boost weight at semantic boundaries (prefer placing seams at boundaries)
        # This is inverted from blend mode - we WANT to select pixels AT boundaries
        # because that's where the seam should be, and the current image's boundary
        # detection gives us the best place to cut
        if boundary_mask is not None:
            if boundary_mask.shape[:2] != (h, w):
                boundary_mask = cv2.resize(boundary_mask, (w, h))
            # Reduce weight slightly at boundaries (seams go there)
            # But don't reduce too much - we still want good coverage
            boundary_norm = boundary_mask.astype(np.float32) / 255.0
            weight *= (1.0 - 0.2 * boundary_norm)
        
        return weight.astype(np.float32)
    
    def _blend_images_pairwise(
        self,
        aligned_images: List[Dict],
        preserve_foreground: bool = True
    ) -> np.ndarray:
        """
        Pairwise selection blending - only consider best 2 images per overlap point.
        
        This mode addresses the burst photo problem where many similar frames
        overlap at the same location. Instead of all frames competing, we:
        1. Find which pairs of images actually overlap at each point
        2. For each overlap region, only consider the 2 images with best coverage
        3. Select winner between those 2 using semantic-aware weights
        
        This eliminates the "repeated area" problem from burst photos.
        """
        logger.info(f"Semantic PAIRWISE blending {len(aligned_images)} images (mode: {self.mode})")
        
        # Calculate bounding box
        bbox = self._calculate_bbox(aligned_images)
        x_min, y_min, x_max, y_max = bbox
        
        output_h = y_max - y_min
        output_w = x_max - x_min
        
        if output_h <= 0 or output_w <= 0:
            logger.error(f"Invalid bounding box: {bbox}")
            return aligned_images[0]['image'].copy()
        
        # Scale to target size (500MP default)
        max_pixels = 500_000_000
        scale_factor = 1.0
        total_pixels = output_h * output_w
        scale_factor = np.sqrt(max_pixels / total_pixels)
            output_h = int(output_h * scale_factor)
            output_w = int(output_w * scale_factor)
        logger.info(f"Scaling to target: {output_w}x{output_h} ({output_w*output_h/1e6:.1f}MP)")
        
        # Create output canvas (white background)
        panorama = np.ones((output_h, output_w, 3), dtype=np.uint8) * 255
        
        # Track image coverage for each pixel
        # coverage_count[y,x] = number of images covering this pixel
        # best_idx[y,x] = index of best image at this pixel
        # second_idx[y,x] = index of second-best image (for pairwise comparison)
        coverage_count = np.zeros((output_h, output_w), dtype=np.uint8)
        best_idx = np.full((output_h, output_w), -1, dtype=np.int16)
        best_weight = np.zeros((output_h, output_w), dtype=np.float32)
        second_idx = np.full((output_h, output_w), -1, dtype=np.int16)
        second_weight = np.zeros((output_h, output_w), dtype=np.float32)
        
        # First pass: compute boundaries and find coverage
        boundary_masks = []
        image_regions = []  # Store preprocessed image data
        
        for idx, img_data in enumerate(aligned_images):
            img = img_data['image']
            h, w = img.shape[:2]
            
            # Compute semantic boundaries
            if self.mode == 'sam':
                boundary, _ = self._run_sam(img)
            elif self.mode == 'deeplab':
                semantic = self._run_deeplab(img)
                boundary = self._semantic_to_boundary(semantic)
            elif self.mode == 'hybrid':
                boundary = self._detect_hybrid_boundaries(img)
            elif self.mode == 'texture':
                boundary = self._detect_texture_boundaries(img)
            elif self.mode == 'superpixel':
                boundary = self._detect_superpixel_boundaries(img)
            else:
                boundary = self._detect_edges(img)
            
            boundary_masks.append(boundary)
            
            # Get alpha mask
            alpha = img_data.get('alpha')
            if alpha is not None:
                alpha_mask = alpha > 127
            else:
                alpha_mask = np.ones((h, w), dtype=bool)
            
            # Filter black borders
            if len(img.shape) == 3:
                min_channel = np.min(img, axis=2)
                black_border = min_channel < 10
                alpha_mask = alpha_mask & (~black_border)
            
            # Compute weight
            weight = self._create_select_weight_mask((h, w), boundary)
            
            # Calculate position
            bbox_img = img_data.get('bbox', (0, 0, w, h))
            
            if scale_factor != 1.0:
                x_off = int((bbox_img[0] - x_min) * scale_factor)
                y_off = int((bbox_img[1] - y_min) * scale_factor)
                scaled_w = max(1, int(w * scale_factor))
                scaled_h = max(1, int(h * scale_factor))
                if scaled_w > 0 and scaled_h > 0:
                    img = cv2.resize(img, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
                    alpha_mask = cv2.resize(alpha_mask.astype(np.uint8), (scaled_w, scaled_h),
                                           interpolation=cv2.INTER_NEAREST).astype(bool)
                    weight = cv2.resize(weight, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
                    h, w = scaled_h, scaled_w
            else:
                x_off = bbox_img[0] - x_min
                y_off = bbox_img[1] - y_min
            
            # Store preprocessed data
            image_regions.append({
                'image': img,
                'alpha_mask': alpha_mask,
                'weight': weight,
                'x_off': x_off,
                'y_off': y_off,
                'h': h,
                'w': w
            })
            
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
            
            # Update coverage tracking
            alpha_region = alpha_mask[src_y_start:src_y_end, src_x_start:src_x_end]
            weight_region = weight[src_y_start:src_y_end, src_x_start:src_x_end]
            
            dst_slice = (slice(dst_y_start, dst_y_end), slice(dst_x_start, dst_x_end))
            
            # Increment coverage count where this image is valid
            coverage_count[dst_slice][alpha_region] += 1
            
            # Update best/second-best tracking
            current_best = best_weight[dst_slice]
            current_second = second_weight[dst_slice]
            
            # Where this image beats current best
            beats_best = alpha_region & (weight_region > current_best)
            # Where this image beats second but not best
            beats_second = alpha_region & (~beats_best) & (weight_region > current_second)
            
            # Update second with old best where we beat best
            second_idx[dst_slice][beats_best] = best_idx[dst_slice][beats_best]
            second_weight[dst_slice][beats_best] = current_best[beats_best]
            
            # Update best where we beat it
            best_idx[dst_slice][beats_best] = idx
            best_weight[dst_slice][beats_best] = weight_region[beats_best]
            
            # Update second where we beat only second
            second_idx[dst_slice][beats_second] = idx
            second_weight[dst_slice][beats_second] = weight_region[beats_second]
        
        # Second pass: write pixels using pairwise selection
        # For each pixel, only the top 2 images compete
        for idx, region in enumerate(image_regions):
            img = region['image']
            alpha_mask = region['alpha_mask']
            weight = region['weight']
            x_off = region['x_off']
            y_off = region['y_off']
            h, w = region['h'], region['w']
            
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
            alpha_region = alpha_mask[src_y_start:src_y_end, src_x_start:src_x_end]
            
            dst_slice = (slice(dst_y_start, dst_y_end), slice(dst_x_start, dst_x_end))
            
            # Only write where this image is in the top 2 for that pixel
            is_best = (best_idx[dst_slice] == idx)
            is_second = (second_idx[dst_slice] == idx)
            is_top2 = is_best | is_second
            
            # Write where valid and in top 2 and best weight
            write_mask = alpha_region & is_top2 & is_best
            
            if np.any(write_mask):
                panorama[dst_slice][write_mask] = img_region[write_mask]
        
        # Fill any remaining gaps with second-best images
        for idx, region in enumerate(image_regions):
            img = region['image']
            alpha_mask = region['alpha_mask']
            x_off = region['x_off']
            y_off = region['y_off']
            h, w = region['h'], region['w']
            
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
            
            img_region = img[src_y_start:src_y_end, src_x_start:src_x_end]
            alpha_region = alpha_mask[src_y_start:src_y_end, src_x_start:src_x_end]
            
            dst_slice = (slice(dst_y_start, dst_y_end), slice(dst_x_start, dst_x_end))
            
            # Fill gaps where panorama is still white and this image is second-best
            is_white = np.all(panorama[dst_slice] == 255, axis=2)
            is_second = (second_idx[dst_slice] == idx)
            
            fill_mask = alpha_region & is_white & is_second
            
            if np.any(fill_mask):
                panorama[dst_slice][fill_mask] = img_region[fill_mask]
        
        # Cleanup
        del boundary_masks, image_regions, coverage_count
        del best_idx, best_weight, second_idx, second_weight
        gc.collect()
        
        logger.info("Semantic PAIRWISE blending complete")
        return panorama
    
    def _create_semantic_weight_mask(
        self,
        shape: Tuple[int, int],
        alpha: np.ndarray,
        boundary_mask: Optional[np.ndarray],
        foreground_mask: Optional[np.ndarray],
        preserve_foreground: bool
    ) -> np.ndarray:
        """
        Create weight mask with semantic awareness.
        
        - Reduces weight at boundaries (avoid seams through objects)
        - Increases weight for foreground objects (preserve them)
        """
        h, w = shape
        weight = np.ones((h, w), dtype=np.float32)
        
        # For SAM mode, skip aggressive edge feathering - SAM handles seam placement
        # Only apply very light edge feathering to avoid hard cutoffs
        if self.mode == 'sam':
            # Minimal feathering - just 10 pixels at edges with 0.5 minimum
            dist_y = np.minimum(np.arange(h)[:, None], np.arange(h)[::-1, None])
            dist_x = np.minimum(np.arange(w)[None, :], np.arange(w)[None, ::-1])
            dist = np.minimum(dist_y, dist_x)
            feather = np.clip(dist / 10.0, 0.5, 1.0)  # Only 10px fade, min 50% weight
            weight *= feather
        else:
            # Standard feathering for other modes
            dist_y = np.minimum(np.arange(h)[:, None], np.arange(h)[::-1, None])
            dist_x = np.minimum(np.arange(w)[None, :], np.arange(w)[None, ::-1])
            dist = np.minimum(dist_y, dist_x)
            feather = np.clip(dist / 100.0, 0, 1)
            weight *= feather
        
        # Apply boundary awareness (reduce weight at detected segment boundaries)
        # This guides seams away from object edges
        if boundary_mask is not None:
            if boundary_mask.shape[:2] != (h, w):
                boundary_mask = cv2.resize(boundary_mask, (w, h))
            boundary_norm = boundary_mask.astype(np.float32) / 255.0
            # Gentle penalty - prefer seams at boundaries but don't darken
            weight *= (1.0 - 0.3 * boundary_norm)  # Max 30% reduction (was 50%)
        
        # Apply foreground boost (increase weight for foreground objects)
        if preserve_foreground and foreground_mask is not None:
            if foreground_mask.shape[:2] != (h, w):
                foreground_mask = cv2.resize(foreground_mask, (w, h))
            # Boost foreground pixels
            weight *= (1.0 + (self.foreground_boost - 1.0) * foreground_mask)
        
        # Apply alpha
        weight *= alpha
        
        return weight
    
    # ============= Classical methods (fallbacks) =============
    
    def _detect_edges(self, image: np.ndarray) -> np.ndarray:
        """Multi-scale edge detection"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        edges1 = cv2.Canny(gray, 30, 100)
        edges2 = cv2.Canny(gray, 50, 150)
        edges3 = cv2.Canny(gray, 100, 200)
        edges = np.maximum(edges1, np.maximum(edges2, edges3))
        
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(sobelx**2 + sobely**2)
        if gradient.max() > 0:
            gradient = (gradient / gradient.max() * 255).astype(np.uint8)
        else:
            gradient = np.zeros_like(gray)
        
        combined = np.maximum(edges, gradient)
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(combined, kernel, iterations=2)
        
        return dilated
    
    def _detect_texture_boundaries(self, image: np.ndarray) -> np.ndarray:
        """Texture boundary detection using Gabor filters and LBP"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        h, w = gray.shape
        
        gabor_responses = []
        frequencies = self.options.get('gabor_frequencies', [0.1, 0.2, 0.3])
        
        if SKIMAGE_AVAILABLE:
            for freq in frequencies:
                for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
                    try:
                        filt_real, filt_imag = gabor(gray, frequency=freq, theta=theta)
                        gabor_responses.append(np.sqrt(filt_real**2 + filt_imag**2))
                    except Exception:
                        pass
        
        if gabor_responses:
            texture_map = np.mean(gabor_responses, axis=0)
            sobelx = cv2.Sobel(texture_map.astype(np.float32), cv2.CV_32F, 1, 0)
            sobely = cv2.Sobel(texture_map.astype(np.float32), cv2.CV_32F, 0, 1)
            texture_gradient = np.sqrt(sobelx**2 + sobely**2)
            if texture_gradient.max() > 0:
                texture_gradient = (texture_gradient / texture_gradient.max() * 255).astype(np.uint8)
            else:
                texture_gradient = np.zeros((h, w), dtype=np.uint8)
        else:
            texture_gradient = self._detect_edges(image)
        
        if SKIMAGE_AVAILABLE:
            try:
                lbp = local_binary_pattern((gray * 255).astype(np.uint8), P=8, R=1, method='uniform')
                lbp_gradient = cv2.Sobel(lbp.astype(np.float32), cv2.CV_32F, 1, 0)
                lbp_gradient = np.abs(lbp_gradient)
                if lbp_gradient.max() > 0:
                    lbp_gradient = (lbp_gradient / lbp_gradient.max() * 255).astype(np.uint8)
                else:
                    lbp_gradient = np.zeros((h, w), dtype=np.uint8)
                texture_gradient = np.maximum(texture_gradient, lbp_gradient)
            except Exception:
                pass
        
        return texture_gradient
    
    def _detect_superpixel_boundaries(self, image: np.ndarray) -> np.ndarray:
        """Superpixel-based boundary detection"""
        h, w = image.shape[:2]
        
        if SKIMAGE_AVAILABLE:
            try:
                segments = slic(
                    image,
                    n_segments=self.superpixel_segments,
                    compactness=10,
                    sigma=1,
                    start_label=1
                )
                
                boundary = np.zeros((h, w), dtype=np.uint8)
                boundary[:-1, :] |= (segments[:-1, :] != segments[1:, :]).astype(np.uint8) * 255
                boundary[:, :-1] |= (segments[:, :-1] != segments[:, 1:]).astype(np.uint8) * 255
                
                kernel = np.ones((3, 3), np.uint8)
                boundary = cv2.dilate(boundary, kernel, iterations=1)
                
                return boundary
            except Exception as e:
                logger.warning(f"Superpixel detection failed: {e}")
        
        return self._detect_edges(image)
    
    def _detect_hybrid_boundaries(self, image: np.ndarray) -> np.ndarray:
        """Hybrid approach combining multiple methods"""
        h, w = image.shape[:2]
        
        edges = self._detect_edges(image)
        color_boundaries = self._detect_color_boundaries(image)
        
        if SKIMAGE_AVAILABLE:
            texture = self._detect_texture_boundaries(image)
            superpixel = self._detect_superpixel_boundaries(image)
        else:
            texture = edges
            superpixel = edges
        
        combined = (
            self.edge_weight * edges.astype(np.float32) +
            self.texture_weight * texture.astype(np.float32) +
            0.3 * color_boundaries.astype(np.float32) +
            0.4 * superpixel.astype(np.float32)
        )
        
        if combined.max() > 0:
            combined = (combined / combined.max() * 255).astype(np.uint8)
        else:
            combined = np.zeros((h, w), dtype=np.uint8)
        
        return combined
    
    def _detect_color_boundaries(self, image: np.ndarray) -> np.ndarray:
        """Color-based boundary detection using K-means clustering"""
        h, w = image.shape[:2]
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        pixels = lab.reshape(-1, 3)
        
        n_clusters = 8
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, _ = cv2.kmeans(
            pixels, n_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )
        
        labels = labels.reshape(h, w)
        
        boundary = np.zeros((h, w), dtype=np.uint8)
        boundary[:-1, :] |= (labels[:-1, :] != labels[1:, :]).astype(np.uint8) * 255
        boundary[:, :-1] |= (labels[:, :-1] != labels[:, 1:]).astype(np.uint8) * 255
        
        kernel = np.ones((3, 3), np.uint8)
        boundary = cv2.dilate(boundary, kernel, iterations=1)
        
        return boundary
    
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
                x_min = min(x_min, 0) if x_min != float('inf') else 0
                y_min = min(y_min, 0) if y_min != float('inf') else 0
                x_max = max(x_max, w)
                y_max = max(y_max, h)
        
        if x_min == float('inf'):
            h, w = aligned_images[0]['image'].shape[:2]
            return (0, 0, w, h)
        
        return (int(x_min), int(y_min), int(x_max), int(y_max))
