"""
Core image stitching engine
Implements modern algorithms for high-quality panoramic stitching
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Callable
import logging
import gc

from ml.feature_detector import LP_SIFTDetector, ORBDetector, AKAZEDetector
from ml.matcher import AdvancedMatcher
from ml.advanced_matchers import LoFTRMatcher, SuperGlueMatcher, DISKMatcher
from ml.efficient_matcher import EfficientMatcher, create_efficient_matcher
from quality.assessor import ImageQualityAssessor
from core.alignment import ImageAligner
from core.blender import ImageBlender
from core.semantic_blender import SemanticBlender
from core.pixelstitch_blender import PixelStitchBlender
from core.control_points import ControlPointManager
from utils.memory_manager import MemoryManager
from utils.lazy_loader import LazyImageLoader, ImageProxy, estimate_memory_for_images
from utils.duplicate_detector import DuplicateDetector, remove_duplicates_from_paths
from utils.image_selector import ImageSelector, select_optimal_images

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
        max_features: int = 5000,
        blending_method: str = 'autostitch',
        blending_options: Optional[Dict] = None,
        progress_callback: Optional[Callable[[int, str], None]] = None,
        cancel_flag: Optional[Callable[[], bool]] = None,
        allow_scale: bool = True,
        max_panorama_pixels: Optional[int] = 100_000_000,
        max_warp_pixels: Optional[int] = 50_000_000,
        memory_efficient: bool = True,
        remove_duplicates: bool = False,
        duplicate_threshold: float = 0.92,
        matching_memory_mode: str = 'balanced',
        smart_select: bool = False,
        max_overlap_percent: float = 25.0
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
            allow_scale: Allow uniform scaling to match images at overlaps
            max_panorama_pixels: Maximum output panorama size in pixels (None/0 = unlimited)
                Memory guidelines for blending:
                - 50MP: ~600MB (uint8) or ~2.4GB (float32 methods)
                - 100MP: ~1.2GB (uint8) or ~4.8GB (float32 methods)
                - 200MP: ~2.4GB (uint8) or ~9.6GB (float32 methods)
            max_warp_pixels: Maximum warped image size in pixels (None/0 = unlimited)
                Memory: ~3 bytes per pixel (RGB uint8)
            memory_efficient: Use lazy loading to reduce memory during feature detection
                Memory savings: ~70-90% during loading/feature detection phase
                - Traditional: All images loaded at once (~3.6GB for 100x 36MB images)
                - Efficient: Thumbnails + compressed cache (~200-400MB)
            remove_duplicates: Pre-scan to remove duplicate/similar images
            duplicate_threshold: Similarity threshold for duplicate detection (0.0-1.0)
                - 0.90 = 90% similar (aggressive, catches more duplicates)
                - 0.95 = 95% similar (moderate, near-identical images)
                - 0.99 = 99% similar (conservative, only exact duplicates)
            matching_memory_mode: Memory optimization level for feature matching
                - 'minimal': Maximum memory savings (PCA + cascade filter + disk cache)
                - 'balanced': Good balance of memory and speed (PCA + cascade filter)
                - 'quality': Full quality, some compression (PCA only)
                - 'standard': No optimization (traditional matching)
            smart_select: Enable smart image selection (skip redundant images)
            max_overlap_percent: Maximum overlap allowed between images (0-100)
                - 25% = Standard (skip highly overlapping photos)
                - 50% = Relaxed (keep more photos)
                - 10% = Aggressive (skip more photos)
        """
        self.use_gpu = use_gpu
        self.quality_threshold = quality_threshold
        self.max_images = max_images
        self.max_features = max_features
        self.allow_scale = allow_scale
        self.max_panorama_pixels = max_panorama_pixels
        self.max_warp_pixels = max_warp_pixels
        self.memory_efficient = memory_efficient
        self.remove_duplicates = remove_duplicates
        self.duplicate_threshold = duplicate_threshold
        self.matching_memory_mode = matching_memory_mode
        self.smart_select = smart_select
        self.max_overlap_percent = max_overlap_percent
        self.memory_manager = MemoryManager(memory_limit_gb=memory_limit_gb)
        self.progress_callback = progress_callback
        self.cancel_flag = cancel_flag
        self._cancelled = False
        
        # Initialize duplicate detector if enabled
        self._duplicate_detector: Optional[DuplicateDetector] = None
        if remove_duplicates:
            self._duplicate_detector = DuplicateDetector(
                similarity_threshold=duplicate_threshold
            )
        
        # Initialize image selector if enabled
        self._image_selector: Optional[ImageSelector] = None
        if smart_select:
            self._image_selector = ImageSelector(
                max_overlap_percent=max_overlap_percent,
                min_overlap_percent=5.0
            )
        
        # Initialize lazy loader if memory efficient mode
        self._lazy_loader: Optional[LazyImageLoader] = None
        if memory_efficient:
            self._lazy_loader = LazyImageLoader(
                thumbnail_size=512,
                cache_compressed=True,
                max_cached_full_images=3
            )
        
        # Initialize feature detector
        self.feature_detector = self._create_feature_detector(feature_detector, use_gpu, max_features)
        
        # Initialize feature matcher (use efficient matcher in memory_efficient mode)
        self.matcher = self._create_feature_matcher(
            feature_matcher, use_gpu, 
            use_efficient=memory_efficient,
            memory_mode=matching_memory_mode
        )
        
        # Initialize quality assessor
        self.quality_assessor = ImageQualityAssessor(use_gpu=use_gpu)
        
        # Initialize aligner
        self.aligner = ImageAligner(
            use_gpu=use_gpu, 
            allow_scale=allow_scale,
            max_warp_pixels=max_warp_pixels
        )
        
        # Initialize control point manager (PTGui-style)
        self.control_point_manager = ControlPointManager()
        
        # Initialize blender
        self.blending_options = blending_options or {}
        self.blender = self._create_blender(blending_method, use_gpu, self.blending_options)
        
        logger.info(
            f"Stitcher initialized (GPU: {use_gpu}, "
            f"Detector: {feature_detector}, Matcher: {feature_matcher}, "
            f"Blender: {blending_method}, Quality threshold: {quality_threshold})"
        )
    
    def _create_feature_detector(self, method: str, use_gpu: bool, max_features: int = 5000):
        """Create feature detector based on method"""
        method = method.lower()
        if method == 'lp_sift' or method == 'sift':
            return LP_SIFTDetector(use_gpu=use_gpu, n_features=max_features)
        elif method == 'orb':
            return ORBDetector(n_features=max_features)
        elif method == 'akaze':
            return AKAZEDetector(n_features=max_features)
        else:
            logger.warning(f"Unknown detector method {method}, using LP-SIFT")
            return LP_SIFTDetector(use_gpu=use_gpu, n_features=max_features)
    
    def _create_feature_matcher(
        self, 
        method: str, 
        use_gpu: bool, 
        use_efficient: bool = False,
        memory_mode: str = 'balanced'
    ):
        """Create feature matcher based on method
        
        Args:
            method: Matching method ('flann', 'loftr', 'superglue', 'disk')
            use_gpu: Enable GPU acceleration
            use_efficient: Use memory-efficient matcher (for FLANN method)
            memory_mode: Memory optimization mode ('minimal', 'balanced', 'quality', 'standard')
        """
        method = method.lower()
        
        # Use efficient matcher for FLANN when memory_efficient is enabled
        if method == 'flann' and use_efficient:
            logger.info(f"Using EfficientMatcher with mode '{memory_mode}'")
            return create_efficient_matcher(
                use_gpu=use_gpu,
                method='flann',
                ratio_threshold=0.75,
                memory_mode=memory_mode
            )
        elif method == 'flann':
            return AdvancedMatcher(use_gpu=use_gpu, method='flann')
        elif method == 'loftr':
            return LoFTRMatcher(use_gpu=use_gpu)
        elif method == 'superglue':
            return SuperGlueMatcher(use_gpu=use_gpu)
        elif method == 'disk':
            return DISKMatcher(use_gpu=use_gpu)
        else:
            logger.warning(f"Unknown matcher method {method}, using FLANN")
            if use_efficient:
                return create_efficient_matcher(use_gpu=use_gpu, method='flann', memory_mode=memory_mode)
            return AdvancedMatcher(use_gpu=use_gpu, method='flann')
    
    def _create_blender(self, method: str, use_gpu: bool, blending_options: Dict = None):
        """Create blender based on method"""
        method = method.lower()
        options = blending_options or {}
        max_pixels = self.max_panorama_pixels
        
        if method == 'autostitch':
            return ImageBlender(use_gpu=use_gpu, method='autostitch', options=options, max_panorama_pixels=max_pixels)
        elif method == 'multiband':
            return ImageBlender(use_gpu=use_gpu, method='multiband', options=options, max_panorama_pixels=max_pixels)
        elif method == 'feather':
            return ImageBlender(use_gpu=use_gpu, method='feather', options=options, max_panorama_pixels=max_pixels)
        elif method == 'linear':
            return ImageBlender(use_gpu=use_gpu, method='linear', options=options, max_panorama_pixels=max_pixels)
        elif method == 'semantic':
            return SemanticBlender(use_gpu=use_gpu, options=options)
        elif method == 'pixelstitch':
            return PixelStitchBlender(use_gpu=use_gpu, options=options)
        else:
            logger.warning(f"Unknown blender method {method}, using multiband")
            return ImageBlender(use_gpu=use_gpu, method='multiband', options=options, max_panorama_pixels=max_pixels)
    
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
        
        # Step 0a (Optional): Remove duplicate/similar images
        if self.remove_duplicates and self._duplicate_detector is not None:
            self._check_cancel()
            logger.info("Step 0a: Detecting and removing duplicate images...")
            self._update_progress(2, f"Scanning {len(image_paths)} images for duplicates...")
            
            image_paths = self._remove_duplicate_images(image_paths)
            
            if not image_paths:
                raise ValueError("No images remaining after duplicate removal")
        
        # Step 0b (Optional): Smart image selection (limit overlap)
        if self.smart_select and self._image_selector is not None:
            self._check_cancel()
            logger.info(f"Step 0b: Smart selection (max {self.max_overlap_percent}% overlap)...")
            self._update_progress(5, f"Analyzing {len(image_paths)} images for optimal selection...")
            
            image_paths, selection_stats = self._select_optimal_images(image_paths)
            
            if not image_paths:
                raise ValueError("No images remaining after smart selection")
            
            self._update_progress(
                8, 
                f"✓ Selected {selection_stats['selected']}/{selection_stats['total']} images "
                f"({selection_stats['reduction_percent']:.0f}% reduction)"
            )
        
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
        gc.collect()  # Free memory after feature detection
        
        # Step 3: Match features
        self._check_cancel()
        logger.info("Step 3: Matching features...")
        self._update_progress(50, "Matching features between images...")
        matches = self._match_features(features_data)
        gc.collect()  # Free memory after matching
        
        if not matches:
            raise ValueError("No feature matches found between images")
        
        # Step 4: Align images (use control points if available)
        self._check_cancel()
        logger.info("Step 4: Aligning images...")
        self._update_progress(70, "Aligning images...")
        
        # If control points exist, use them for alignment
        if len(self.control_point_manager.get_all_control_points()) > 0:
            logger.info("Using control points for alignment")
            aligned_images = self._align_with_control_points(images_data, features_data, matches)
        else:
            aligned_images = self.aligner.align_images(images_data, features_data, matches)
        
        # Free original images if they were replaced by warped versions
        del images_data, features_data, matches
        gc.collect()
        
        # Step 5: Blend images
        self._check_cancel()
        logger.info("Step 5: Blending images...")
        self._update_progress(85, "Blending images...")
        if isinstance(self.blender, SemanticBlender):
            panorama = self.blender.blend_images(aligned_images, preserve_foreground=True)
        else:
            panorama = self.blender.blend_images(aligned_images, options=self.blending_options)
        
        # Cleanup aligned images
        del aligned_images
        gc.collect()
        
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
    
    def _remove_duplicate_images(self, image_paths: List[Path]) -> List[Path]:
        """
        Remove duplicate and similar images from the input list.
        
        Args:
            image_paths: List of image paths
            
        Returns:
            Filtered list with duplicates removed
        """
        if not self._duplicate_detector or len(image_paths) <= 1:
            return image_paths
        
        total = len(image_paths)
        
        def progress_callback(current: int, total_items: int, message: str):
            """Map duplicate detection progress to 2-10% range"""
            self._check_cancel()
            if total_items > 0:
                progress = 2 + int(8 * (current / total_items))
                self._update_progress(progress, f"Duplicate scan: {message}")
        
        # Load thumbnails for comparison
        self._update_progress(2, f"Loading thumbnails for duplicate detection...")
        
        images = []
        for i, path in enumerate(image_paths):
            self._check_cancel()
            progress_callback(i, total, f"Loading {path.name}")
            
            img = cv2.imread(str(path))
            if img is not None:
                # Resize for faster comparison
                h, w = img.shape[:2]
                if max(h, w) > 256:
                    scale = 256 / max(h, w)
                    img = cv2.resize(img, (int(w * scale), int(h * scale)), 
                                   interpolation=cv2.INTER_AREA)
                images.append((path, img))
            else:
                logger.warning(f"Could not load image for duplicate check: {path}")
        
        if len(images) <= 1:
            return image_paths
        
        # Find duplicates
        self._update_progress(5, f"Comparing {len(images)} images for duplicates...")
        
        keep_indices, duplicates = self._duplicate_detector.find_duplicates(
            images, 
            progress_callback=progress_callback
        )
        
        # Get filtered paths
        filtered_paths = [images[i][0] for i in keep_indices]
        num_removed = total - len(filtered_paths)
        
        if num_removed > 0:
            logger.info(f"Duplicate detection: KEEPING {len(filtered_paths)} unique images (removed {num_removed} duplicates)")
            self._update_progress(10, f"✓ Keeping {len(filtered_paths)} unique images (removed {num_removed} duplicate(s))")
            
            # Log which images were removed
            removed_paths = set(image_paths) - set(filtered_paths)
            for path in removed_paths:
                logger.info(f"  Removed: {path.name}")
            logger.info(f"  Kept: {[p.name for p in filtered_paths]}")
        else:
            logger.info(f"Duplicate detection: ALL {total} images are unique (no duplicates found)")
            self._update_progress(10, f"✓ All {total} images are unique (no duplicates)")
        
        # Clean up
        del images
        gc.collect()
        
        return filtered_paths
    
    def _select_optimal_images(self, image_paths: List[Path]) -> Tuple[List[Path], Dict]:
        """
        Select optimal subset of images with limited overlap.
        
        Args:
            image_paths: List of image paths
            
        Returns:
            Tuple of (filtered_paths, stats_dict)
        """
        if not self._image_selector or len(image_paths) <= 2:
            return image_paths, {'selected': len(image_paths), 'total': len(image_paths), 'reduction_percent': 0}
        
        total = len(image_paths)
        
        def progress_callback(current: int, total_items: int, message: str):
            """Map smart selection progress to 5-8% range"""
            self._check_cancel()
            if total_items > 0:
                progress = 5 + int(3 * (current / total_items))
                self._update_progress(progress, f"Smart selection: {message}")
        
        selected_paths, stats = self._image_selector.select_images(
            image_paths,
            progress_callback=progress_callback
        )
        
        if stats['selected'] < total:
            logger.info(
                f"Smart selection: USING {stats['selected']}/{total} images "
                f"({stats['reduction_percent']:.1f}% reduction, "
                f"avg overlap: {stats.get('avg_overlap', 0)*100:.1f}%)"
            )
            
            # Log which images were kept/skipped
            kept_set = set(selected_paths)
            skipped = [p for p in image_paths if p not in kept_set]
            if len(skipped) <= 20:  # Only log if not too many
                for path in skipped:
                    logger.info(f"  Skipped: {path.name}")
        else:
            logger.info(f"Smart selection: All {total} images selected (no redundant overlap)")
        
        return selected_paths, stats
    
    def create_grid_alignment(
        self,
        image_paths: List[Path],
        grid_size: Optional[Tuple[int, int]] = None,
        min_overlap_percent: float = 10.0,
        max_overlap_percent: float = 100.0,
        spacing_factor: float = 1.3
    ) -> Dict:
        """
        Create 2D grid alignment without merging - "exploded view" style
        
        Args:
            image_paths: List of paths to input images
            grid_size: Optional (rows, cols) for grid. If None, auto-calculated
            min_overlap_percent: Minimum overlap percentage (0-100) for images to be included
            max_overlap_percent: Maximum overlap percentage (0-100) to filter near-duplicates
            spacing_factor: Multiplier for spacing between images (1.0 = touching, 1.5 = 50% gap)
            
        Returns:
            Dictionary with grid layout information
        """
        logger.info(f"Creating grid alignment for {len(image_paths)} images (overlap: {min_overlap_percent}%-{max_overlap_percent}%, spacing: {spacing_factor}x)")
        
        # Optional: Remove duplicate images first
        if self.remove_duplicates and self._duplicate_detector is not None:
            self._check_cancel()
            logger.info("Pre-processing: Detecting and removing duplicate images...")
            self._update_progress(2, f"Scanning {len(image_paths)} images for duplicates...")
            image_paths = self._remove_duplicate_images(image_paths)
            
            if not image_paths:
                logger.warning("No images remaining after duplicate removal")
                return {
                    'images': [],
                    'positions': [],
                    'grid_size': (1, 1),
                    'matches': []
                }
        
        # Load and assess images
        images_data = self._load_and_assess_images(image_paths)
        gc.collect()
        
        # Detect features
        features_data = self._detect_features(images_data)
        gc.collect()
        
        # Match features to determine spatial relationships
        matches = self._match_features(features_data)
        gc.collect()
        
        # Filter images and matches based on overlap threshold
        min_overlap_ratio = min_overlap_percent / 100.0
        max_overlap_ratio = max_overlap_percent / 100.0
        filtered_data = self._filter_by_overlap(images_data, features_data, matches, min_overlap_ratio, max_overlap_ratio)
        
        # Clean up unneeded data
        del images_data, features_data, matches
        gc.collect()
        
        if not filtered_data['images']:
            logger.warning(f"No images meet the minimum overlap threshold of {min_overlap_percent}%")
            # Return empty grid layout
            return {
                'images': [],
                'positions': [],
                'grid_size': (1, 1),
                'matches': []
            }
        
        # Create grid layout based on filtered matches
        grid_layout = self.aligner.create_grid_layout(
            filtered_data['images'], 
            filtered_data['features'], 
            filtered_data['matches'], 
            grid_size,
            spacing_factor=spacing_factor
        )
        
        # Cleanup
        del filtered_data
        gc.collect()
        
        return grid_layout
    
    def _filter_by_overlap(
        self,
        images_data: List[Dict],
        features_data: List[Dict],
        matches: List[Dict],
        min_overlap_ratio: float,
        max_overlap_ratio: float = 1.0
    ) -> Dict:
        """
        Filter images and matches based on overlap threshold range
        
        Args:
            images_data: List of image data dictionaries
            features_data: List of feature data dictionaries
            matches: List of match dictionaries
            min_overlap_ratio: Minimum overlap ratio (0.0-1.0)
            max_overlap_ratio: Maximum overlap ratio (0.0-1.0) to filter near-duplicates
            
        Returns:
            Dictionary with filtered images, features, and matches
        """
        if not matches:
            return {
                'images': images_data,
                'features': features_data,
                'matches': []
            }
        
        # Calculate overlap for each match
        overlap_map = {}
        for match in matches:
            i = match['image_i']
            j = match['image_j']
            
            if i >= len(images_data) or j >= len(images_data):
                continue
            
            # Calculate overlap percentage
            confidence = match.get('confidence', 0.0)
            num_matches = match.get('num_matches', 0)
            
            img1_area = images_data[i]['shape'][0] * images_data[i]['shape'][1]
            img2_area = images_data[j]['shape'][0] * images_data[j]['shape'][1]
            avg_area = (img1_area + img2_area) / 2
            
            if avg_area > 0:
                match_density = num_matches / np.sqrt(avg_area) * 100
                overlap_estimate = min(confidence * 0.5 + match_density * 0.01, 1.0)
            else:
                overlap_estimate = confidence
            
            overlap_map[(i, j)] = overlap_estimate
            overlap_map[(j, i)] = overlap_estimate
        
        # Find images that meet the overlap threshold range
        # An image is included if it has at least one match with min <= overlap <= max
        included_indices = set()
        filtered_matches = []
        excluded_high_overlap = 0
        excluded_low_overlap = 0
        
        for match in matches:
            i = match['image_i']
            j = match['image_j']
            overlap_key = (i, j)
            overlap_pct = overlap_map.get(overlap_key, 0.0)
            
            # Check both min and max thresholds
            if overlap_pct < min_overlap_ratio:
                excluded_low_overlap += 1
                continue
            if overlap_pct > max_overlap_ratio:
                excluded_high_overlap += 1
                logger.debug(f"Excluding match {i}-{j}: overlap {overlap_pct*100:.1f}% > max {max_overlap_ratio*100:.1f}%")
                continue
                
                included_indices.add(i)
                included_indices.add(j)
                filtered_matches.append(match)
        
        # Log exclusion stats
        if excluded_high_overlap > 0:
            logger.info(f"Excluded {excluded_high_overlap} matches with overlap > {max_overlap_ratio*100:.1f}% (near-duplicates)")
        if excluded_low_overlap > 0:
            logger.info(f"Excluded {excluded_low_overlap} matches with overlap < {min_overlap_ratio*100:.1f}%")
        
        # If no matches meet threshold, include all images
        if not included_indices:
            logger.warning("No matches meet overlap threshold range, including all images")
            included_indices = set(range(len(images_data)))
            filtered_matches = matches
        
        # Filter images and features
        included_list = sorted(list(included_indices))
        index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(included_list)}
        
        filtered_images = [images_data[i] for i in included_list]
        filtered_features = [features_data[i] for i in included_list]
        
        # Remap match indices
        remapped_matches = []
        for match in filtered_matches:
            i = match['image_i']
            j = match['image_j']
            
            if i in index_map and j in index_map:
                new_match = match.copy()
                new_match['image_i'] = index_map[i]
                new_match['image_j'] = index_map[j]
                remapped_matches.append(new_match)
        
        logger.info(f"Filtered to {len(filtered_images)} images (from {len(images_data)}) with overlap {min_overlap_ratio*100:.1f}%-{max_overlap_ratio*100:.1f}%")
        
        return {
            'images': filtered_images,
            'features': filtered_features,
            'matches': remapped_matches
        }
    
    def save_grid(self, grid_layout: Dict, output_path: str, quality: str = 'high', dpi: int = 300, postproc: Dict = None):
        """Save grid alignment visualization with optional post-processing"""
        from core.grid_visualizer import GridVisualizer
        visualizer = GridVisualizer()
        visualizer.save_grid(grid_layout, output_path, quality=quality, dpi=dpi, postproc=postproc)
    
    def save_panorama(self, panorama: np.ndarray, output_path: str, quality: str = 'high', dpi: int = 300, postproc: Dict = None):
        """
        Save stitched panorama with quality and DPI settings
        
        Args:
            panorama: Image array to save
            output_path: Output file path
            quality: Quality preset ('ultra_high', 'high', 'medium', 'low', 'minimum')
            dpi: Output DPI (dots per inch)
            postproc: Post-processing options dict
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving panorama: shape={panorama.shape}, dtype={panorama.dtype}, quality={quality}, dpi={dpi}")
        
        # Apply post-processing if requested
        if postproc:
            from core.post_processing import apply_post_processing
            panorama = apply_post_processing(panorama, postproc)
            logger.info(f"After post-processing: shape={panorama.shape}")
        
        # Ensure we have a valid image
        if panorama is None or panorama.size == 0:
            raise ValueError("Cannot save empty panorama")
        
        # Ensure uint8 format
        if panorama.dtype != np.uint8:
            if panorama.max() <= 1.0:
                panorama = (panorama * 255).astype(np.uint8)
            else:
                panorama = panorama.astype(np.uint8)
        
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
            self._save_tiff(panorama, output_path, settings['tiff_compression'], dpi)
        elif suffix in ['.png']:
            self._save_png(panorama, output_path, settings['png_level'], dpi)
        elif suffix in ['.jpg', '.jpeg']:
            self._save_jpeg(panorama, output_path, settings['jpeg_quality'], dpi)
        else:
            # Default to PNG for unknown formats
            cv2.imwrite(str(output_path), panorama)
            logger.info(f"Panorama saved to {output_path}")
        
        # Verify file was created
        if output_path.exists():
            file_size = output_path.stat().st_size
            logger.info(f"Output file size: {file_size / 1024 / 1024:.2f} MB")
            if file_size < 1024:
                logger.warning(f"Output file is very small ({file_size} bytes), may be corrupted")
        else:
            logger.error(f"Output file was not created: {output_path}")
    
    def _save_tiff(self, image: np.ndarray, path: Path, compression: str, dpi: int):
        """Save as TIFF with proper fallback"""
        # Convert BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Try tifffile first
        try:
            import tifffile
            
            # Calculate resolution (pixels per cm for TIFF)
            resolution = (dpi / 2.54, dpi / 2.54)  # Convert DPI to pixels per cm
            
            # Try with compression, fall back to uncompressed if it fails
            try:
                if compression:
                    tifffile.imwrite(
                        str(path), 
                        image_rgb,
                        compression=compression,
                        photometric='rgb',
                        resolution=resolution,
                        resolutionunit='CENTIMETER'
                    )
                else:
                    # Uncompressed
                    tifffile.imwrite(
                        str(path), 
                        image_rgb,
                        photometric='rgb',
                        resolution=resolution,
                        resolutionunit='CENTIMETER'
                    )
                logger.info(f"TIFF saved with tifffile (compression={compression}, dpi={dpi})")
                return
            except KeyError as e:
                # Compression codec not available, try uncompressed
                if 'imagecodecs' in str(e):
                    logger.warning(f"Compression codec not available, saving uncompressed TIFF")
                    tifffile.imwrite(
                        str(path), 
                        image_rgb,
                        photometric='rgb',
                        resolution=resolution,
                        resolutionunit='CENTIMETER'
                    )
                    logger.info(f"TIFF saved uncompressed with tifffile (dpi={dpi})")
                    return
                raise
                
        except ImportError:
            logger.warning("tifffile not installed, using OpenCV")
        except Exception as e:
            logger.warning(f"tifffile failed: {e}, using OpenCV")
        
        # Fallback to OpenCV
        cv2.imwrite(str(path), image)
        logger.info(f"TIFF saved with OpenCV (dpi not embedded)")
    
    def _save_png(self, image: np.ndarray, path: Path, compression_level: int, dpi: int):
        """Save as PNG with DPI metadata"""
        # Save with OpenCV
        cv2.imwrite(str(path), image, [cv2.IMWRITE_PNG_COMPRESSION, compression_level])
        
        # Try to add DPI metadata using PIL
        try:
            from PIL import Image
            img = Image.open(str(path))
            img.save(str(path), dpi=(dpi, dpi))
            logger.info(f"PNG saved with compression={compression_level}, dpi={dpi}")
        except ImportError:
            logger.info(f"PNG saved with compression={compression_level} (PIL not available for DPI)")
        except Exception as e:
            logger.warning(f"Could not set PNG DPI: {e}")
    
    def _save_jpeg(self, image: np.ndarray, path: Path, quality: int, dpi: int):
        """Save as JPEG with quality and DPI"""
        # Try PIL first for DPI support
        try:
            from PIL import Image
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            img = Image.fromarray(image_rgb)
            img.save(str(path), quality=quality, dpi=(dpi, dpi))
            logger.info(f"JPEG saved with quality={quality}, dpi={dpi}")
            return
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"PIL save failed: {e}, using OpenCV")
        
        # Fallback to OpenCV
        cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        logger.info(f"JPEG saved with quality={quality} (dpi not embedded)")
    
    def _load_and_assess_images(self, image_paths: List[Path]) -> List[Dict]:
        """Load images and assess quality (with optional memory-efficient mode)"""
        images_data = []
        total = len(image_paths)
        
        # Estimate and log memory requirements
        if total > 0:
            estimates = estimate_memory_for_images(image_paths[:min(3, total)])
            if total > 3:
                # Extrapolate from sample
                scale = total / min(3, total)
                for key in estimates:
                    estimates[key] *= scale
            
            logger.info(f"Estimated memory requirements for {total} images:")
            logger.info(f"  Traditional loading: {estimates['traditional']:.0f} MB")
            logger.info(f"  Memory-efficient (cached): {estimates['lazy_with_cache']:.0f} MB")
            logger.info(f"  Memory-efficient (no cache): {estimates['lazy_no_cache']:.0f} MB")
            
            # Warn if traditional mode would use too much memory
            available_mb = self.memory_manager.get_available_memory() * 1024
            if not self.memory_efficient and estimates['traditional'] > available_mb * 0.7:
                logger.warning(
                    f"Traditional loading may exceed available memory! "
                    f"Consider enabling memory_efficient mode."
                )
        
        # Use memory-efficient loading if enabled
        if self.memory_efficient and self._lazy_loader is not None:
            return self._load_and_assess_images_efficient(image_paths)
        
        # Traditional loading (all images in memory)
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
                try:
                    quality_score = self.quality_assessor.assess(img_bgr)
                    
                    # Handle NaN or invalid scores
                    if np.isnan(quality_score) or quality_score < 0:
                        logger.warning(f"Invalid quality score for {path.name}: {quality_score}, using default 0.5")
                        quality_score = 0.5
                    
                    logger.info(f"Image {path.name}: quality={quality_score:.3f}, threshold={self.quality_threshold:.3f}")
                    
                    if quality_score >= self.quality_threshold:
                        images_data.append({
                            'path': path,
                            'image': img_bgr,
                            'alpha': alpha,
                            'quality': quality_score,
                            'shape': img_bgr.shape
                        })
                    else:
                        logger.info(f"Image {path.name} rejected (quality: {quality_score:.3f} < threshold: {self.quality_threshold:.3f})")
                except Exception as e:
                    logger.error(f"Error assessing quality for {path.name}: {e}", exc_info=True)
                    # Include image anyway with default quality score
                    logger.warning(f"Including {path.name} with default quality score due to assessment error")
                    images_data.append({
                        'path': path,
                        'image': img_bgr,
                        'alpha': alpha,
                        'quality': 0.5,
                        'shape': img_bgr.shape
                    })
            
            except Exception as e:
                logger.error(f"Error processing {path}: {e}")
                continue
        
        # Sort by quality (best first)
        images_data.sort(key=lambda x: x['quality'], reverse=True)
        
        # Log summary
        if images_data:
            logger.info(f"Loaded {len(images_data)} images (quality range: {images_data[-1]['quality']:.3f} - {images_data[0]['quality']:.3f})")
        else:
            logger.warning(f"No images passed quality assessment (threshold: {self.quality_threshold:.3f})")
            logger.warning("Reloading images with lower threshold (0.1) as fallback...")
            
            # Fallback: reload with very low threshold to ensure we get at least some images
            fallback_threshold = 0.1
            images_data = []
            for i, path in enumerate(image_paths):
                if self.max_images and i >= self.max_images:
                    break
                try:
                    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                    if img is None:
                        continue
                    
                    if len(img.shape) == 3 and img.shape[2] == 4:
                        alpha = img[:, :, 3]
                        img_bgr = img[:, :, :3]
                    else:
                        alpha = None
                        img_bgr = img if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    
                    quality_score = self.quality_assessor.assess(img_bgr)
                    if np.isnan(quality_score) or quality_score < 0:
                        quality_score = 0.1
                    
                    if quality_score >= fallback_threshold:
                        images_data.append({
                            'path': path,
                            'image': img_bgr,
                            'alpha': alpha,
                            'quality': quality_score,
                            'shape': img_bgr.shape
                        })
                except Exception as e:
                    logger.error(f"Error in fallback loading for {path}: {e}")
                    continue
            
            if images_data:
                images_data.sort(key=lambda x: x['quality'], reverse=True)
                logger.info(f"Fallback loaded {len(images_data)} images with lower threshold")
            else:
                logger.error("Failed to load any images even with fallback threshold")
        
        return images_data
    
    def _load_and_assess_images_efficient(self, image_paths: List[Path]) -> List[Dict]:
        """
        Memory-efficient image loading using lazy loader.
        
        Key optimizations:
        1. Images stored as compressed JPEG in memory (~10-20x smaller)
        2. Thumbnails used for quality assessment
        3. Full images loaded on-demand for feature detection
        4. Images released after feature detection, reloaded for blending
        
        Memory savings: ~70-90% during loading/feature detection phase
        """
        images_data = []
        total = len(image_paths)
        
        # Limit to max_images if set
        paths_to_load = image_paths[:self.max_images] if self.max_images else image_paths
        total_to_load = len(paths_to_load)
        
        # Phase 1: Create proxies (scan images, create thumbnails, compress)
        # Progress: 10-15%
        self._update_progress(10, f"Phase 1/3: Scanning {total_to_load} images...")
        
        def proxy_progress(current, total_proxies, message):
            """Callback for proxy loading progress"""
            self._check_cancel()
            if total_proxies > 0:
                # Map to progress range 10-15%
                progress = 10 + int(5 * (current / total_proxies))
                self._update_progress(progress, f"Scanning ({current}/{total_proxies}): {message}")
        
        proxies = self._lazy_loader.load_proxies(paths_to_load, progress_callback=proxy_progress)
        
        memory_mb = self._lazy_loader.estimate_total_memory_mb()
        self._update_progress(15, f"Scanned {len(proxies)} images ({memory_mb:.1f} MB in memory)")
        logger.info(f"Created {len(proxies)} image proxies, memory: {memory_mb:.1f} MB")
        
        # Phase 2: Assess quality using thumbnails
        # Progress: 15-22%
        self._update_progress(15, f"Phase 2/3: Assessing image quality...")
        
        accepted_count = 0
        rejected_count = 0
        
        for i, proxy in enumerate(proxies):
            self._check_cancel()
            
            # Progress update with stats
            if total_to_load > 0:
                progress = 15 + int(7 * (i / total_to_load))
                status = f"Quality check ({i+1}/{total_to_load}): {proxy.path.name}"
                if accepted_count > 0 or rejected_count > 0:
                    status += f" [✓{accepted_count} ✗{rejected_count}]"
                self._update_progress(progress, status)
            
            if proxy.shape is None:
                logger.warning(f"Could not load image metadata: {proxy.path}")
                rejected_count += 1
                continue
            
            try:
                # Use thumbnail for quality assessment (much faster)
                thumbnail = proxy.thumbnail
                if thumbnail is not None:
                    quality_score = self.quality_assessor.assess(thumbnail)
                    
                    if np.isnan(quality_score) or quality_score < 0:
                        quality_score = 0.5
                    
                    proxy.quality = quality_score
                    
                    logger.info(f"Image {proxy.path.name}: quality={quality_score:.3f}, threshold={self.quality_threshold:.3f}")
                    
                    if quality_score >= self.quality_threshold:
                        # Create image data dict with proxy reference (image loaded on demand)
                        images_data.append({
                            'path': proxy.path,
                            'image': None,  # Will be loaded on demand
                            'alpha': None,
                            'quality': quality_score,
                            'shape': proxy.shape,
                            '_proxy': proxy  # Store proxy for later loading
                        })
                        accepted_count += 1
                    else:
                        logger.info(f"Image {proxy.path.name} rejected (quality: {quality_score:.3f} < threshold)")
                        rejected_count += 1
                else:
                    logger.warning(f"No thumbnail for {proxy.path.name}")
                    rejected_count += 1
                    
            except Exception as e:
                logger.error(f"Error assessing {proxy.path.name}: {e}", exc_info=True)
                # Include with default quality
                images_data.append({
                    'path': proxy.path,
                    'image': None,
                    'alpha': None,
                    'quality': 0.5,
                    'shape': proxy.shape,
                    '_proxy': proxy
                })
                accepted_count += 1
        
        # Sort by quality
        images_data.sort(key=lambda x: x['quality'], reverse=True)
        
        self._update_progress(22, f"Quality assessment complete: {accepted_count} accepted, {rejected_count} rejected")
        
        # Phase 3: Load full images for accepted images
        # Progress: 22-30%
        total_accepted = len(images_data)
        self._update_progress(22, f"Phase 3/3: Loading {total_accepted} full-resolution images...")
        
        loaded_count = 0
        failed_count = 0
        images_to_remove = []
        
        for i, img_data in enumerate(images_data):
            self._check_cancel()
            
            # Progress with memory info
            if total_accepted > 0:
                progress = 22 + int(8 * (i / total_accepted))
                current_memory = self.memory_manager.get_memory_usage()
                self._update_progress(
                    progress, 
                    f"Loading full image ({i+1}/{total_accepted}): {img_data['path'].name} [{current_memory:.1f}GB used]"
                )
            
            proxy = img_data.get('_proxy')
            if proxy is not None:
                # Load full image from proxy
                img = proxy.load_full()
                if img is not None:
                    img_data['image'] = img
                    # Handle alpha
                    if len(img.shape) == 3 and img.shape[2] == 4:
                        img_data['alpha'] = img[:, :, 3]
                        img_data['image'] = img[:, :, :3]
                    
                    # Update shape with actual loaded shape
                    img_data['shape'] = img_data['image'].shape
                    loaded_count += 1
                else:
                    logger.warning(f"Could not load full image: {proxy.path}")
                    images_to_remove.append(img_data)
                    failed_count += 1
            
            # Periodic GC
            if i > 0 and i % 10 == 0:
                gc.collect()
        
        # Remove failed images
        for img_data in images_to_remove:
            images_data.remove(img_data)
        
        # Final status
        current_memory = self.memory_manager.get_memory_usage()
        self._update_progress(
            30, 
            f"Loaded {loaded_count} images successfully ({current_memory:.1f}GB memory used)"
        )
        logger.info(f"Loaded {len(images_data)} images, current memory usage: {current_memory:.2f} GB")
        
        # Fallback if no images passed
        if not images_data and proxies:
            logger.warning("No images passed quality threshold, reloading with lower threshold...")
            for proxy in proxies:
                if proxy.quality is not None and proxy.quality >= 0.1:
                    img = proxy.load_full()
                    if img is not None:
                        alpha = None
                        if len(img.shape) == 3 and img.shape[2] == 4:
                            alpha = img[:, :, 3]
                            img = img[:, :, :3]
                        
                        images_data.append({
                            'path': proxy.path,
                            'image': img,
                            'alpha': alpha,
                            'quality': proxy.quality,
                            'shape': img.shape,
                            '_proxy': proxy
                        })
            
            images_data.sort(key=lambda x: x['quality'], reverse=True)
        
        return images_data
    
    def _detect_features(self, images_data: List[Dict]) -> List[Dict]:
        """Detect features in all images"""
        features_data = []
        total = len(images_data)
        
        # Log image sizes for debugging
        if images_data:
            sizes = [f"{d['image'].shape[1]}x{d['image'].shape[0]}" for d in images_data[:3]]
            logger.info(f"Detecting features in {total} images (sizes: {', '.join(sizes)}...)")
        
        for idx, img_data in enumerate(images_data):
            self._check_cancel()
            
            # Update progress with ETA for large batches
            if total > 0:
                progress = 30 + int(20 * (idx / total))
                if idx == 0:
                    self._update_progress(progress, f"Initializing feature detector (first image may take longer)...")
                else:
                    self._update_progress(progress, f"Detecting features: {idx+1}/{total}")
            
            try:
                import time
                start_time = time.time()
                kp_array, descriptors = self.feature_detector.detect_and_compute(img_data['image'])
                elapsed = time.time() - start_time
                
                if idx == 0:
                    logger.info(f"First image feature detection took {elapsed:.2f}s (subsequent will be faster)")
                
                # Validate features were detected
                if descriptors is None:
                    logger.warning(f"No descriptors detected for image {idx}")
                    descriptors = np.array([])
                elif len(descriptors) == 0:
                    logger.warning(f"Empty descriptors for image {idx}")
                else:
                    logger.debug(f"Image {idx}: detected {len(descriptors)} features")
                
                features_data.append({
                    'image_data': img_data,
                    'keypoints': kp_array,
                    'descriptors': descriptors
                })
            except Exception as e:
                logger.error(f"Error detecting features in image {idx}: {e}", exc_info=True)
                # Add empty descriptors so processing can continue
                features_data.append({
                    'image_data': img_data,
                    'keypoints': np.array([]),
                    'descriptors': np.array([])
                })
        
        return features_data
    
    def _match_features(self, features_data: List[Dict]) -> List[Dict]:
        """Match features between images
        
        Uses EfficientMatcher with cascade filtering and PCA compression when
        memory_efficient mode is enabled, otherwise falls back to standard matching.
        """
        # Check if we're using the efficient matcher
        if isinstance(self.matcher, EfficientMatcher):
            # Use efficient batch matching with cascade filtering
            self._update_progress(50, "Preparing memory-efficient feature matching...")
            
            # Estimate memory savings
            n_images = len(features_data)
            avg_features = sum(len(fd.get('descriptors', [])) for fd in features_data) // max(n_images, 1)
            savings = self.matcher.estimate_memory_savings(n_images, avg_features)
            logger.info(f"Memory-efficient matching: ~{savings['savings_mb']:.1f}MB savings "
                       f"({savings['savings_percent']:.0f}%), {savings['pair_reduction_percent']:.0f}% fewer pairs")
            
            # Use the efficient matcher's batch method
            matches = self.matcher.match_all(
                features_data,
                progress_callback=self._update_progress,
                cancel_check=lambda: self._cancelled
            )
            return matches
        
        # Standard matching path for other matchers
        matches = []
        
        # Check if matcher is a deep learning matcher that needs images
        from ml.advanced_matchers import LoFTRMatcher, SuperGlueMatcher, DISKMatcher
        is_dl_matcher = isinstance(self.matcher, (LoFTRMatcher,))
        is_superglue = isinstance(self.matcher, SuperGlueMatcher)
        
        total_pairs = len(features_data) * (len(features_data) - 1) // 2
        pair_count = 0
        
        for i in range(len(features_data)):
            for j in range(i + 1, len(features_data)):
                self._check_cancel()
                
                pair_count += 1
                
                # Update progress before matching (so user sees progress immediately)
                if total_pairs > 0:
                    progress = 50 + int(20 * (pair_count / total_pairs))
                    self._update_progress(progress, f"Matching features: pair {pair_count}/{total_pairs}...")
                
                try:
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
                        desc1 = features_data[i]['descriptors']
                        desc2 = features_data[j]['descriptors']
                        
                        # Validate descriptors
                        if desc1 is None or desc2 is None:
                            logger.warning(f"Descriptors are None for pair ({i}, {j})")
                            continue
                        if len(desc1) == 0 or len(desc2) == 0:
                            logger.debug(f"Empty descriptors for pair ({i}, {j})")
                            continue
                        
                        match_result = self.matcher.match(desc1, desc2)
                    
                    num_matches = match_result.get('num_matches', 0) if match_result else 0
                    
                    # Log match results for debugging
                    if num_matches > 0:
                        logger.debug(f"Pair ({i}, {j}): {num_matches} matches, confidence: {match_result.get('confidence', 0.0):.4f}")
                    
                    # Require at least 15 matches for better reliability
                    if match_result and num_matches >= 15:
                        matches.append({
                            'image_i': i,
                            'image_j': j,
                            'matches': match_result.get('matches', []),
                            'num_matches': num_matches,
                            'confidence': match_result.get('confidence', 0.0)
                        })
                    elif num_matches > 0:
                        logger.debug(f"Pair ({i}, {j}) rejected: only {num_matches} matches (need >= 15)")
                except Exception as e:
                    logger.error(f"Error matching features between images {i} and {j}: {e}", exc_info=True)
                    continue
        
        return matches
    
    def _align_with_control_points(
        self,
        images_data: List[Dict],
        features_data: List[Dict],
        matches: List[Dict]
    ) -> List[Dict]:
        """Align images using control points (PTGui-style) with similarity transforms"""
        logger.info("Aligning images using control points")
        
        # Build graph from control points
        graph = {}
        for cp in self.control_point_manager.get_all_control_points():
            i1, i2 = cp.image1_idx, cp.image2_idx
            if i1 not in graph:
                graph[i1] = []
            if i2 not in graph:
                graph[i2] = []
            if i2 not in graph[i1]:
                graph[i1].append(i2)
            if i1 not in graph[i2]:
                graph[i2].append(i1)
        
        # Find reference image (most connected)
        if not graph:
            logger.warning("No control point graph, falling back to standard alignment")
            return self.aligner.align_images(images_data, features_data, matches)
        
        ref_idx = max(graph.keys(), key=lambda k: len(graph.get(k, [])))
        logger.info(f"Using image {ref_idx} as reference (from control points)")
        
        # Calculate transforms using BFS
        transforms = {ref_idx: np.eye(3, dtype=np.float64)}
        queue = [ref_idx]
        visited = {ref_idx}
        
        while queue:
            current = queue.pop(0)
            current_transform = transforms[current]
            
            for neighbor in graph.get(current, []):
                if neighbor in visited:
                    continue
                
                # Estimate transform from neighbor to current
                transform = self.control_point_manager.estimate_homography_from_control_points(
                    neighbor, current, allow_scale=self.allow_scale
                )
                
                if transform is not None:
                    # Compose: neighbor_to_global = current_to_global @ neighbor_to_current
                    neighbor_to_global = current_transform @ transform.astype(np.float64)
                    transforms[neighbor] = neighbor_to_global
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        # Apply transforms to create aligned images (reuse aligner's method)
        return self.aligner._apply_transforms(images_data, transforms, ref_idx)

