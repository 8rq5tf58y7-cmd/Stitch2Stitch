"""
Semantic-aware blending algorithms
Based on SemanticStitch: Foreground-Aware Seam Carving
"""

import cv2
import numpy as np
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Semantic blending will use fallback methods.")


class SemanticBlender:
    """
    Semantic-aware blending that preserves foreground objects
    Based on: "SemanticStitch: Foreground-Aware Seam Carving for Image Stitching"
    """
    
    def __init__(self, use_gpu: bool = False):
        """
        Initialize semantic blender
        
        Args:
            use_gpu: Enable GPU acceleration
        """
        self.use_gpu = use_gpu and TORCH_AVAILABLE
        self.semantic_model = None
        self._load_semantic_model()
        logger.info(f"Semantic blender initialized (GPU: {self.use_gpu})")
    
    def _load_semantic_model(self):
        """Load semantic segmentation model (placeholder)"""
        if TORCH_AVAILABLE:
            try:
                # Placeholder: Would load a semantic segmentation model
                # e.g., DeepLabV3, SegFormer, etc.
                # from torchvision.models.segmentation import deeplabv3_resnet50
                # self.semantic_model = deeplabv3_resnet50(pretrained=True)
                # if self.use_gpu:
                #     self.semantic_model = self.semantic_model.cuda()
                logger.info("Semantic model structure ready (weights loading can be added)")
            except Exception as e:
                logger.warning(f"Could not load semantic model: {e}")
    
    def blend_images(
        self,
        aligned_images: List[Dict],
        preserve_foreground: bool = True
    ) -> np.ndarray:
        """
        Blend images with semantic awareness
        
        Args:
            aligned_images: List of aligned image data dictionaries
            preserve_foreground: Whether to preserve foreground objects
            
        Returns:
            Blended panorama
        """
        if not aligned_images:
            raise ValueError("No images to blend")
        
        if len(aligned_images) == 1:
            return aligned_images[0]['image']
        
        logger.info(f"Semantic blending {len(aligned_images)} images")
        
        # Calculate bounding box
        bbox = self._calculate_bbox(aligned_images)
        x_min, y_min, x_max, y_max = bbox
        
        output_h = y_max - y_min
        output_w = x_max - x_min
        
        # Create output canvas
        panorama = np.zeros((output_h, output_w, 3), dtype=np.float32)
        weight_sum = np.zeros((output_h, output_w), dtype=np.float32)
        
        # Get semantic masks if available
        semantic_masks = []
        if preserve_foreground and self.semantic_model is not None:
            semantic_masks = self._extract_semantic_masks(aligned_images)
        
        # Blend each image
        for idx, img_data in enumerate(aligned_images):
            img = img_data['image'].astype(np.float32)
            alpha = img_data.get('alpha')
            
            if alpha is not None:
                alpha = alpha.astype(np.float32) / 255.0
            else:
                alpha = np.ones((img.shape[0], img.shape[1]), dtype=np.float32)
            
            # Create weight mask with semantic awareness
            if preserve_foreground and idx < len(semantic_masks):
                weight = self._create_semantic_weight_mask(
                    img.shape[:2],
                    alpha,
                    semantic_masks[idx]
                )
            else:
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
        weight_sum[weight_sum == 0] = 1
        for c in range(3):
            panorama[:, :, c] /= weight_sum
        
        return panorama.astype(np.uint8)
    
    def _extract_semantic_masks(self, aligned_images: List[Dict]) -> List[np.ndarray]:
        """Extract semantic masks for foreground preservation"""
        masks = []
        
        for img_data in aligned_images:
            img = img_data['image']
            
            if self.semantic_model is not None and TORCH_AVAILABLE:
                try:
                    # Placeholder for semantic segmentation
                    # img_tensor = self._preprocess_for_segmentation(img)
                    # with torch.no_grad():
                    #     output = self.semantic_model(img_tensor)
                    #     mask = self._postprocess_segmentation(output)
                    # masks.append(mask)
                    
                    # Fallback: Use saliency detection
                    mask = self._saliency_mask(img)
                    masks.append(mask)
                except Exception as e:
                    logger.warning(f"Semantic extraction failed: {e}")
                    mask = self._saliency_mask(img)
                    masks.append(mask)
            else:
                # Fallback: Use saliency detection
                mask = self._saliency_mask(img)
                masks.append(mask)
        
        return masks
    
    def _saliency_mask(self, image: np.ndarray) -> np.ndarray:
        """Create saliency mask as fallback for semantic segmentation"""
        # Use OpenCV's saliency detector
        try:
            saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
            success, saliency_map = saliency.computeSaliency(image)
            if success:
                return (saliency_map * 255).astype(np.uint8)
        except:
            pass
        
        # Fallback: Use edge-based saliency
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        return dilated
    
    def _create_semantic_weight_mask(
        self,
        shape: Tuple[int, int],
        alpha: np.ndarray,
        semantic_mask: np.ndarray
    ) -> np.ndarray:
        """Create weight mask that preserves semantic objects"""
        h, w = shape
        weight = np.ones((h, w), dtype=np.float32)
        
        # Enhance weights for foreground objects
        if semantic_mask.shape[:2] == (h, w):
            # Normalize semantic mask
            semantic_norm = semantic_mask.astype(np.float32) / 255.0
            
            # Increase weight for foreground (high saliency)
            weight = weight * (1.0 + 0.5 * semantic_norm)
        
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
    
    def _create_weight_mask(
        self,
        shape: Tuple[int, int],
        alpha: np.ndarray
    ) -> np.ndarray:
        """Create standard weight mask"""
        h, w = shape
        weight = np.ones((h, w), dtype=np.float32) * alpha
        
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

