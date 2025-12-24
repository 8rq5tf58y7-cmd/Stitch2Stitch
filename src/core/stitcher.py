"""
Core image stitching engine
Implements modern algorithms for high-quality panoramic stitching
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import logging

from ..ml.feature_detector import LP_SIFTDetector, ORBDetector, AKAZEDetector
from ..ml.matcher import AdvancedMatcher
from ..ml.advanced_matchers import LoFTRMatcher, SuperGlueMatcher, DISKMatcher
from ..quality.assessor import ImageQualityAssessor
from ..core.alignment import ImageAligner
from ..core.blender import ImageBlender
from ..core.semantic_blender import SemanticBlender
from ..core.pixelstitch_blender import PixelStitchBlender
from ..utils.memory_manager import MemoryManager

logger = logging.getLogger(__name__)


class ImageStitcher:
    """Main stitching engine"""
    
    def __init__(
        self,
        use_gpu: bool = False,
        quality_threshold: float = 0.7,
        max_images: Optional[int] = None,
        memory_limit_gb: float = 16.0,
        feature_detector: str = 'lp_sift',
        feature_matcher: str = 'flann',
        blending_method: str = 'multiband',
        progress_callback: Optional[Callable[[int, str], None]] = None,
        cancel_flag: Optional[Callable[[], bool]] = None
    ):
        """
        Initialize the stitcher
        
        Args:
            use_gpu: Enable GPU acceleration
            quality_threshold: Minimum quality score (0.0-1.0) for image inclusion
            max_images: Maximum number of images to process (None = unlimited)
            memory_limit_gb: Memory limit in GB for processing
            feature_detector: Feature detector algorithm ('lp_sift', 'sift', 'orb', 'akaze')
            feature_matcher: Feature matcher algorithm ('flann', 'loftr', 'superglue', 'disk')
            blending_method: Blending algorithm ('multiband', 'feather', 'linear', 'semantic', 'pixelstitch')
        """
        self.use_gpu = use_gpu
        self.quality_threshold = quality_threshold
        self.max_images = max_images
        self.memory_manager = MemoryManager(memory_limit_gb=memory_limit_gb)
        self.progress_callback = progress_callback
        self.cancel_flag = cancel_flag
        self._cancelled = False
        
        # Initialize feature detector
        self.feature_detector = self._create_feature_detector(feature_detector, use_gpu)
        
        # Initialize feature matcher
        self.matcher = self._create_feature_matcher(feature_matcher, use_gpu)
        
        # Initialize quality assessor
        self.quality_assessor = ImageQualityAssessor(use_gpu=use_gpu)
        
        # Initialize aligner
        self.aligner = ImageAligner(use_gpu=use_gpu)
        
        # Initialize blender
        self.blender = self._create_blender(blending_method, use_gpu)
        
        logger.info(
            f"Stitcher initialized (GPU: {use_gpu}, "
            f"Detector: {feature_detector}, Matcher: {feature_matcher}, "
            f"Blender: {blending_method}, Quality threshold: {quality_threshold})"
        )
    
    def _create_feature_detector(self, method: str, use_gpu: bool):
        """Create feature detector based on method"""
        method = method.lower()
        if method == 'lp_sift' or method == 'sift':
            return LP_SIFTDetector(use_gpu=use_gpu)
        elif method == 'orb':
            return ORBDetector()
        elif method == 'akaze':
            return AKAZEDetector()
        else:
            logger.warning(f"Unknown detector method {method}, using LP-SIFT")
            return LP_SIFTDetector(use_gpu=use_gpu)
    
    def _create_feature_matcher(self, method: str, use_gpu: bool):
        """Create feature matcher based on method"""
        method = method.lower()
        if method == 'flann':
            return AdvancedMatcher(use_gpu=use_gpu, method='flann')
        elif method == 'loftr':
            return LoFTRMatcher(use_gpu=use_gpu)
        elif method == 'superglue':
            return SuperGlueMatcher(use_gpu=use_gpu)
        elif method == 'disk':
            return DISKMatcher(use_gpu=use_gpu)
        else:
            logger.warning(f"Unknown matcher method {method}, using FLANN")
            return AdvancedMatcher(use_gpu=use_gpu, method='flann')
    
    def _create_blender(self, method: str, use_gpu: bool):
        """Create blender based on method"""
        method = method.lower()
        if method == 'multiband':
            return ImageBlender(use_gpu=use_gpu, method='multiband')
        elif method == 'feather':
            return ImageBlender(use_gpu=use_gpu, method='feather')
        elif method == 'linear':
            return ImageBlender(use_gpu=use_gpu, method='linear')
        elif method == 'semantic':
            return SemanticBlender(use_gpu=use_gpu)
        elif method == 'pixelstitch':
            return PixelStitchBlender(use_gpu=use_gpu)
        else:
            logger.warning(f"Unknown blender method {method}, using multiband")
            return ImageBlender(use_gpu=use_gpu, method='multiband')
    
    def stitch(self, image_paths: List[Path]) -> np.ndarray:
        """
        Stitch images into a panorama
        
        Args:
            image_paths: List of paths to input images
            
        Returns:
            Stitched panorama as numpy array
        """
        self._cancelled = False
        
        logger.info(f"Starting stitching process for {len(image_paths)} images")
        self._update_progress(0, "Starting stitching process...")
        
        # Step 1: Load and assess image quality
        self._check_cancel()
        logger.info("Step 1: Loading and assessing image quality...")
        self._update_progress(10, "Loading and assessing image quality...")
        images_data = self._load_and_assess_images(image_paths)
        
        if not images_data:
            raise ValueError("No images passed quality assessment")
        
        logger.info(f"Selected {len(images_data)} high-quality images out of {len(image_paths)}")
        
        # Step 2: Detect features
        self._check_cancel()
        logger.info("Step 2: Detecting features...")
        self._update_progress(30, f"Detecting features in {len(images_data)} images...")
        features_data = self._detect_features(images_data)
        
        # Step 3: Match features
        self._check_cancel()
        logger.info("Step 3: Matching features...")
        self._update_progress(50, "Matching features between images...")
        matches = self._match_features(features_data)
        
        if not matches:
            raise ValueError("No feature matches found between images")
        
        # Step 4: Align images
        self._check_cancel()
        logger.info("Step 4: Aligning images...")
        self._update_progress(70, "Aligning images...")
        aligned_images = self.aligner.align_images(images_data, features_data, matches)
        
        # Step 5: Blend images
        self._check_cancel()
        logger.info("Step 5: Blending images...")
        self._update_progress(85, "Blending images...")
        if isinstance(self.blender, SemanticBlender):
            panorama = self.blender.blend_images(aligned_images, preserve_foreground=True)
        else:
            panorama = self.blender.blend_images(aligned_images)
        
        self._update_progress(100, "Stitching completed successfully!")
        logger.info("Stitching completed successfully")
        return panorama
    
    def cancel(self):
        """Cancel the current stitching operation"""
        self._cancelled = True
        logger.info("Stitching operation cancelled by user")
    
    def _check_cancel(self):
        """Check if operation should be cancelled"""
        if self._cancelled or (self.cancel_flag and self.cancel_flag()):
            raise InterruptedError("Stitching operation cancelled")
    
    def _update_progress(self, percentage: int, message: str = ""):
        """Update progress callback"""
        if self.progress_callback:
            self.progress_callback(percentage, message)
    
    def create_grid_alignment(
        self,
        image_paths: List[Path],
        grid_size: Optional[Tuple[int, int]] = None
    ) -> Dict:
        """
        Create 2D grid alignment without merging
        
        Args:
            image_paths: List of paths to input images
            grid_size: Optional (rows, cols) for grid. If None, auto-calculated
            
        Returns:
            Dictionary with grid layout information
        """
        logger.info(f"Creating grid alignment for {len(image_paths)} images")
        
        # Load and assess images
        images_data = self._load_and_assess_images(image_paths)
        
        # Detect features
        features_data = self._detect_features(images_data)
        
        # Match features to determine spatial relationships
        matches = self._match_features(features_data)
        
        # Create grid layout based on matches
        grid_layout = self.aligner.create_grid_layout(images_data, features_data, matches, grid_size)
        
        return grid_layout
    
    def save_grid(self, grid_layout: Dict, output_path: str):
        """Save grid alignment visualization"""
        from ..core.grid_visualizer import GridVisualizer
        visualizer = GridVisualizer()
        visualizer.save_grid(grid_layout, output_path)
    
    def save_panorama(self, panorama: np.ndarray, output_path: str):
        """Save stitched panorama"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use appropriate format based on extension
        if output_path.suffix.lower() in ['.tif', '.tiff']:
            import tifffile
            tifffile.imwrite(str(output_path), panorama, compression='lzw')
        else:
            cv2.imwrite(str(output_path), panorama)
        
        logger.info(f"Panorama saved to {output_path}")
    
    def _load_and_assess_images(self, image_paths: List[Path]) -> List[Dict]:
        """Load images and assess quality"""
        images_data = []
        total = len(image_paths)
        
        for i, path in enumerate(image_paths):
            self._check_cancel()
            
            if self.max_images and i >= self.max_images:
                break
            
            # Update progress for loading
            if total > 0:
                progress = 10 + int(20 * (i / total))
                self._update_progress(progress, f"Loading image {i+1}/{total}: {path.name}")
            
            try:
                # Load image with transparency support
                img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                if img is None:
                    logger.warning(f"Could not load image: {path}")
                    continue
                
                # Handle transparency
                if len(img.shape) == 3 and img.shape[2] == 4:  # RGBA
                    alpha = img[:, :, 3]
                    img_bgr = img[:, :, :3]
                else:
                    alpha = None
                    img_bgr = img if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                
                # Assess quality
                quality_score = self.quality_assessor.assess(img_bgr)
                
                if quality_score >= self.quality_threshold:
                    images_data.append({
                        'path': path,
                        'image': img_bgr,
                        'alpha': alpha,
                        'quality': quality_score,
                        'shape': img_bgr.shape
                    })
                else:
                    logger.debug(f"Image {path} rejected (quality: {quality_score:.3f})")
            
            except Exception as e:
                logger.error(f"Error processing {path}: {e}")
                continue
        
        # Sort by quality (best first)
        images_data.sort(key=lambda x: x['quality'], reverse=True)
        
        return images_data
    
    def _detect_features(self, images_data: List[Dict]) -> List[Dict]:
        """Detect features in all images"""
        features_data = []
        total = len(images_data)
        
        for idx, img_data in enumerate(images_data):
            self._check_cancel()
            
            # Update progress
            if total > 0:
                progress = 30 + int(20 * (idx / total))
                self._update_progress(progress, f"Detecting features in image {idx+1}/{total}...")
            
            kp_array, descriptors = self.feature_detector.detect_and_compute(img_data['image'])
            
            features_data.append({
                'image_data': img_data,
                'keypoints': kp_array,
                'descriptors': descriptors
            })
        
        return features_data
    
    def _match_features(self, features_data: List[Dict]) -> List[Dict]:
        """Match features between images"""
        matches = []
        
        # Check if matcher is a deep learning matcher that needs images
        from ..ml.advanced_matchers import LoFTRMatcher, SuperGlueMatcher, DISKMatcher
        is_dl_matcher = isinstance(self.matcher, (LoFTRMatcher,))
        is_superglue = isinstance(self.matcher, SuperGlueMatcher)
        
        total_pairs = len(features_data) * (len(features_data) - 1) // 2
        pair_count = 0
        
        for i in range(len(features_data)):
            for j in range(i + 1, len(features_data)):
                self._check_cancel()
                
                # Update progress
                if total_pairs > 0:
                    progress = 50 + int(20 * (pair_count / total_pairs))
                    self._update_progress(progress, f"Matching features: pair {pair_count+1}/{total_pairs}...")
                
                pair_count += 1
                if is_dl_matcher:
                    # LoFTR needs image data
                    img1 = features_data[i]['image_data']['image']
                    img2 = features_data[j]['image_data']['image']
                    match_result = self.matcher.match(img1, img2)
                elif is_superglue:
                    # SuperGlue needs keypoints and descriptors
                    match_result = self.matcher.match(
                        features_data[i]['keypoints'],
                        features_data[i]['descriptors'],
                        features_data[j]['keypoints'],
                        features_data[j]['descriptors']
                    )
                else:
                    # Standard matchers use descriptors
                    match_result = self.matcher.match(
                        features_data[i]['descriptors'],
                        features_data[j]['descriptors']
                    )
                
                if match_result['num_matches'] > 10:  # Minimum matches threshold
                    matches.append({
                        'image_i': i,
                        'image_j': j,
                        'matches': match_result['matches'],
                        'num_matches': match_result['num_matches'],
                        'confidence': match_result['confidence']
                    })
        
        return matches

