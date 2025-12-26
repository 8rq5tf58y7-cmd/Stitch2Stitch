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
from quality.assessor import ImageQualityAssessor
from core.alignment import ImageAligner
from core.blender import ImageBlender
from core.semantic_blender import SemanticBlender
from core.pixelstitch_blender import PixelStitchBlender
from core.control_points import ControlPointManager
from utils.memory_manager import MemoryManager
from utils.lazy_loader import LazyImageLoader, ImageProxy, estimate_memory_for_images
from utils.duplicate_detector import DuplicateDetector

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
        geometric_verify: bool = True,
        select_optimal_coverage: bool = False,
        max_coverage_overlap: float = 0.5,
        remove_duplicates: bool = True,
        duplicate_threshold: float = 0.92,
        optimize_alignment: bool = False,
        alignment_optimization_level: str = 'balanced',
        # AutoPano Giga-inspired features
        use_grid_topology: bool = True,
        use_bundle_adjustment: bool = False,
        use_hierarchical_stitching: bool = False,
        use_enhanced_detection: bool = False
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
            geometric_verify: Enable geometric verification to filter bad feature matches
                Uses RANSAC-based similarity transform estimation to reject outliers
            select_optimal_coverage: Only use images necessary to cover the panorama area
                Avoids redundant images that overlap >max_coverage_overlap with already-selected images
            max_coverage_overlap: Maximum allowed overlap (0.0-1.0) between selected images
                Lower values = fewer images, higher values = more redundancy
            remove_duplicates: Enable duplicate/similar image removal for burst photos
                Removes nearly identical frames to reduce processing and improve alignment
            duplicate_threshold: Similarity threshold (0.0-1.0) for duplicate detection
                0.92 = very strict (only near-identical), 0.85 = moderate, 0.75 = aggressive
            optimize_alignment: Preprocess images to improve feature detection
                Inspired by AutoPano Giga's alignment optimization feature
            alignment_optimization_level: Level of optimization
                'light' = subtle enhancement (fast, preserves original look)
                'balanced' = moderate enhancement (recommended)
                'aggressive' = strong enhancement (best for low-contrast images)
            use_grid_topology: Auto-detect grid structure to reduce matching from O(n²) to O(n)
                Critical for large flat panoramas (microscope, drone, satellite)
            use_bundle_adjustment: Global optimization of all camera poses
                Minimizes reprojection error across all keypoints
            use_hierarchical_stitching: Cluster-based stitching for 1000+ images
                Stitches clusters independently, then merges
            use_enhanced_detection: Use enhanced feature detection for low-texture areas
                Better for skies, walls, water, repetitive patterns
        """
        self.use_gpu = use_gpu
        self.quality_threshold = quality_threshold
        self.max_images = max_images
        self.max_features = max_features
        self.allow_scale = allow_scale
        self.max_panorama_pixels = max_panorama_pixels
        self.max_warp_pixels = max_warp_pixels
        self.memory_efficient = memory_efficient
        self.geometric_verify = geometric_verify
        self.select_optimal_coverage = select_optimal_coverage
        self.max_coverage_overlap = max_coverage_overlap
        self.remove_duplicates = remove_duplicates
        self.duplicate_threshold = duplicate_threshold
        self.optimize_alignment = optimize_alignment
        self.alignment_optimization_level = alignment_optimization_level
        self.use_grid_topology = use_grid_topology
        self.use_bundle_adjustment = use_bundle_adjustment
        self.use_hierarchical_stitching = use_hierarchical_stitching
        self.use_enhanced_detection = use_enhanced_detection
        self.memory_manager = MemoryManager(memory_limit_gb=memory_limit_gb)
        
        # AutoPano-inspired features (lazy loaded)
        self._grid_detector = None
        self._bundle_adjuster = None
        self._hierarchical_stitcher = None
        self._enhanced_detector = None
        self.progress_callback = progress_callback
        self.cancel_flag = cancel_flag
        self._cancelled = False
        
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
        
        # Initialize feature matcher
        self.matcher = self._create_feature_matcher(feature_matcher, use_gpu)
        
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
            f"Blender: {blending_method}, Quality threshold: {quality_threshold}, "
            f"Geometric verify: {geometric_verify}, Optimal coverage: {select_optimal_coverage})"
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
        elif method == 'generative':
            from core.generative_blender import GenerativeBlender
            # Map GUI options to generative blender options
            gen_options = {
                'backend': options.get('gen_backend', 'hybrid'),
                'api_key': options.get('gen_api_key'),
                'strength': options.get('gen_strength', 0.75),
            }
            return GenerativeBlender(use_gpu=use_gpu, options=gen_options)
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
        
        # Step 1: Load and assess image quality
        self._check_cancel()
        logger.info("Step 1: Loading and assessing image quality...")
        self._update_progress(10, "Loading and assessing image quality...")
        images_data = self._load_and_assess_images(image_paths)
        
        if not images_data:
            raise ValueError("No images passed quality assessment")
        
        logger.info(f"Selected {len(images_data)} high-quality images out of {len(image_paths)}")
        
        # Step 1.5: Remove duplicate/similar images (for burst photos)
        if self.remove_duplicates and len(images_data) > 2:
            self._check_cancel()
            logger.info("Step 1.5: Removing duplicate images...")
            self._update_progress(20, "Detecting and removing duplicate images...")
            images_data = self._remove_duplicate_images(images_data)
            gc.collect()
            
            if len(images_data) < 2:
                raise ValueError("Too few unique images after duplicate removal")
            
            logger.info(f"After duplicate removal: {len(images_data)} unique images")
        
        # Step 1.75: Optimize images for alignment (if enabled)
        if self.optimize_alignment:
            self._check_cancel()
            logger.info("Step 1.75: Optimizing images for alignment...")
            self._update_progress(25, "Optimizing images for feature detection...")
            images_data = self._optimize_for_alignment(images_data)
            gc.collect()
        
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
        
        # Step 3.5: Select optimal coverage (remove redundant images)
        if self.select_optimal_coverage:
            self._check_cancel()
            logger.info("Step 3.5: Selecting optimal image coverage...")
            self._update_progress(65, "Selecting optimal image coverage...")
            images_data, features_data, matches = self._select_optimal_images(
                images_data, features_data, matches
            )
            gc.collect()
            logger.info(f"After coverage selection: {len(images_data)} images")
        
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
        
        # Step 4.5: Bundle adjustment (global optimization)
        if self.use_bundle_adjustment and len(aligned_images) >= 3:
            self._check_cancel()
            logger.info("Step 4.5: Bundle adjustment optimization...")
            self._update_progress(78, "Optimizing global alignment (bundle adjustment)...")
            aligned_images = self._run_bundle_adjustment(aligned_images, matches)
        
        # If alignment optimization was used, restore original images for blending
        # (optimized images were only for better feature detection)
        if self.optimize_alignment:
            logger.info("Restoring original images for blending...")
            for aligned_img in aligned_images:
                # Get original from the source data
                for img_data in images_data:
                    if 'original_image' in img_data and img_data.get('path') == aligned_img.get('source_path'):
                        # Re-warp original image with same transform
                        if 'transform' in aligned_img:
                            from core.alignment import ImageAligner
                            original = img_data['original_image']
                            h, w = original.shape[:2]
                            transform = aligned_img['transform']
                            
                            # Apply the same transform to original
                            warped = cv2.warpAffine(
                                original, 
                                transform[:2], 
                                (aligned_img['image'].shape[1], aligned_img['image'].shape[0]),
                                flags=cv2.INTER_LANCZOS4,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(0, 0, 0)
                            )
                            aligned_img['image'] = warped
                        break
        
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
        
        # Step 6: AI Post-processing (optional)
        if self.blending_options.get('ai_post_processing', False):
            self._check_cancel()
            logger.info("Step 6: AI post-processing...")
            self._update_progress(95, "Applying AI enhancement...")
            panorama = self._apply_ai_post_processing(panorama)
        
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
            
            # Passed overlap filters; keep this match and mark images as included
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
        
        # Keep only the largest connected component to avoid stray/redundant tiles
        if remapped_matches and len(filtered_images) > 1:
            adj = {i: set() for i in range(len(filtered_images))}
            for m in remapped_matches:
                a = m['image_i']
                b = m['image_j']
                adj[a].add(b)
                adj[b].add(a)
            
            visited = set()
            components = []
            
            for node in adj:
                if node in visited:
                    continue
                stack = [node]
                comp = []
                while stack:
                    n = stack.pop()
                    if n in visited:
                        continue
                    visited.add(n)
                    comp.append(n)
                    for nei in adj.get(n, []):
                        if nei not in visited:
                            stack.append(nei)
                components.append(comp)
            
            # Select largest component
            largest = max(components, key=len) if components else []
            if len(largest) < len(filtered_images):
                logger.info(f"Pruning to largest connected component: keeping {len(largest)} of {len(filtered_images)} images")
                keep_set = set(largest)
                filtered_images = [img for idx, img in enumerate(filtered_images) if idx in keep_set]
                filtered_features = [feat for idx, feat in enumerate(filtered_features) if idx in keep_set]
                
                # Remap again after pruning
                new_index_map = {old: new for new, old in enumerate(sorted(list(keep_set)))}
                remapped_matches = []
                for m in filtered_matches:
                    if m['image_i'] in keep_set and m['image_j'] in keep_set:
                        new_m = m.copy()
                        new_m['image_i'] = new_index_map[m['image_i']]
                        new_m['image_j'] = new_index_map[m['image_j']]
                        remapped_matches.append(new_m)
        
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
    
    def _remove_duplicate_images(self, images_data: List[Dict]) -> List[Dict]:
        """
        Remove duplicate/similar images from burst photo sets.
        
        Uses perceptual hashing and pixel-level similarity to identify
        near-identical frames that would only add redundancy.
        """
        if len(images_data) <= 2:
            return images_data
        
        # Create detector with progress callback
        detector = DuplicateDetector(
            similarity_threshold=self.duplicate_threshold,
            hash_size=16,
            comparison_window=30,  # Compare with nearby images (efficient for burst)
            progress_callback=self._update_progress
        )
        
        # Extract images and paths for comparison
        images = [d['image'] for d in images_data]
        paths = [d.get('path') for d in images_data]
        
        # Find and remove duplicates
        keep_indices, duplicate_pairs = detector.find_duplicates(images, paths)
        
        # Log duplicate statistics
        if duplicate_pairs:
            logger.info(f"Found {len(duplicate_pairs)} duplicate pairs")
            for idx1, idx2, sim in duplicate_pairs[:5]:  # Show first 5
                name1 = paths[idx1].name if paths[idx1] else f"image_{idx1}"
                name2 = paths[idx2].name if paths[idx2] else f"image_{idx2}"
                logger.debug(f"  {name2} is duplicate of {name1} (similarity={sim:.3f})")
        
        # Filter to keep only unique images
        filtered_data = [images_data[i] for i in keep_indices]
        
        removed_count = len(images_data) - len(filtered_data)
        if removed_count > 0:
            self._update_progress(25, f"Removed {removed_count} duplicate images")
        
        return filtered_data
    
    def _optimize_for_alignment(self, images_data: List[Dict]) -> List[Dict]:
        """
        Optimize images for better feature detection and alignment.
        
        Inspired by AutoPano Giga's alignment optimization feature.
        Applies preprocessing to improve keypoint detection:
        - CLAHE contrast enhancement (adaptive histogram equalization)
        - Edge enhancement for better gradient-based features
        - Noise reduction to avoid false features
        - Color normalization for consistent matching
        
        Note: Original images are preserved; optimized versions are only
        used for feature detection, not blending.
        """
        logger.info(f"Optimizing {len(images_data)} images for alignment (level: {self.alignment_optimization_level})")
        
        # Get optimization parameters based on level
        if self.alignment_optimization_level == 'light':
            clahe_clip = 1.5
            clahe_grid = (4, 4)
            denoise_h = 3
            edge_strength = 0.1
            color_norm = False
        elif self.alignment_optimization_level == 'aggressive':
            clahe_clip = 4.0
            clahe_grid = (16, 16)
            denoise_h = 7
            edge_strength = 0.3
            color_norm = True
        else:  # balanced (default)
            clahe_clip = 2.5
            clahe_grid = (8, 8)
            denoise_h = 5
            edge_strength = 0.2
            color_norm = True
        
        optimized_data = []
        
        for idx, img_data in enumerate(images_data):
            self._check_cancel()
            
            if idx % 10 == 0:
                progress = 25 + int(5 * (idx / len(images_data)))
                self._update_progress(progress, f"Optimizing image {idx+1}/{len(images_data)}...")
            
            # Store original for blending
            original = img_data['image']
            
            try:
                # Create optimized version for feature detection
                optimized = self._apply_alignment_optimization(
                    original,
                    clahe_clip=clahe_clip,
                    clahe_grid=clahe_grid,
                    denoise_h=denoise_h,
                    edge_strength=edge_strength,
                    color_norm=color_norm
                )
                
                # Store both versions
                new_data = img_data.copy()
                new_data['image'] = optimized
                new_data['original_image'] = original  # Keep original for blending
                optimized_data.append(new_data)
                
            except Exception as e:
                logger.warning(f"Optimization failed for image {idx}: {e}, using original")
                optimized_data.append(img_data)
        
        logger.info(f"Alignment optimization complete for {len(optimized_data)} images")
        return optimized_data
    
    def _apply_ai_post_processing(self, panorama: np.ndarray) -> np.ndarray:
        """
        Apply AutoPano Giga-style AI post-processing.
        
        Includes:
        - Color correction (gray world + CLAHE)
        - Denoising
        - Gap inpainting
        - Optional super-resolution
        """
        try:
            from core.autopano_features import AIPostProcessor
            
            processor = AIPostProcessor(use_gpu=self.use_gpu)
            
            # Detect gaps (white/black areas)
            gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
            gap_mask = ((gray < 5) | (gray > 250)).astype(np.uint8) * 255
            
            # Erode to avoid edge artifacts
            kernel = np.ones((3, 3), np.uint8)
            gap_mask = cv2.erode(gap_mask, kernel, iterations=2)
            
            # Inpaint gaps
            if np.any(gap_mask):
                logger.info(f"Inpainting {np.sum(gap_mask > 0)} gap pixels...")
                panorama = processor.inpaint_gaps(panorama, gap_mask)
            
            # Apply enhancements
            panorama = processor.enhance(
                panorama,
                enable_super_res=self.blending_options.get('super_resolution', False),
                enable_denoise=self.blending_options.get('ai_denoise', True),
                enable_color_correct=self.blending_options.get('ai_color_correct', True)
            )
            
            logger.info("AI post-processing complete")
            return panorama
            
        except Exception as e:
            logger.warning(f"AI post-processing failed: {e}")
            return panorama
    
    def _run_bundle_adjustment(
        self,
        aligned_images: List[Dict],
        matches: Dict
    ) -> List[Dict]:
        """
        Run bundle adjustment to globally optimize alignment.
        
        AutoPano Giga-inspired two-stage optimization:
        1. Coarse BA: Fast optimization with reduced parameters
        2. Refine matches: Remove outliers based on current alignment
        3. Fine BA: Full optimization with all parameters
        """
        try:
            from core.autopano_features import TwoStageBundleAdjuster, BundleAdjuster
            
            if self._bundle_adjuster is None:
                # Use two-stage bundle adjustment for better results
                self._bundle_adjuster = TwoStageBundleAdjuster()
            
            # Extract transforms from aligned images
            initial_transforms = []
            for img_data in aligned_images:
                transform = img_data.get('transform')
                if transform is not None:
                    initial_transforms.append(transform)
                else:
                    # Identity transform
                    initial_transforms.append(np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64))
            
            # Convert matches dict to the format expected by bundle adjuster
            matches_dict = {}
            if isinstance(matches, list):
                for match in matches:
                    i, j = match.get('img_i', 0), match.get('img_j', 1)
                    matches_dict[(i, j)] = match
            else:
                matches_dict = matches
            
            # Run optimization
            optimized_transforms, stats = self._bundle_adjuster.optimize(
                aligned_images,
                matches_dict,
                initial_transforms
            )
            
            if stats.get('success', False):
                logger.info(f"Bundle adjustment: {stats.get('improvement_percent', 0):.1f}% improvement")
                
                # Apply optimized transforms
                for i, img_data in enumerate(aligned_images):
                    if i < len(optimized_transforms):
                        old_transform = img_data.get('transform')
                        new_transform = optimized_transforms[i]
                        img_data['transform'] = new_transform
                        
                        # Re-warp image if transform changed significantly
                        if old_transform is not None:
                            diff = np.abs(new_transform - old_transform[:2]).max()
                            if diff > 1.0:  # More than 1 pixel change
                                # Re-apply warp with new transform
                                original = img_data.get('original_image', img_data['image'])
                                h, w = original.shape[:2]
                                
                                # Calculate output size
                                corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
                                transformed = cv2.transform(corners.reshape(1, -1, 2), new_transform)
                                min_x = int(transformed[0, :, 0].min())
                                min_y = int(transformed[0, :, 1].min())
                                max_x = int(transformed[0, :, 0].max())
                                max_y = int(transformed[0, :, 1].max())
                                
                                out_w = max_x - min_x
                                out_h = max_y - min_y
                                
                                # Clamp size
                                max_dim = 10000
                                if out_w > max_dim or out_h > max_dim:
                                    scale = max_dim / max(out_w, out_h)
                                    out_w = int(out_w * scale)
                                    out_h = int(out_h * scale)
                                
                                if out_w > 0 and out_h > 0:
                                    # Adjust transform for output origin
                                    adjusted = new_transform.copy()
                                    adjusted[0, 2] -= min_x
                                    adjusted[1, 2] -= min_y
                                    
                                    warped = cv2.warpAffine(
                                        original, adjusted, (out_w, out_h),
                                        flags=cv2.INTER_LANCZOS4,
                                        borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=(0, 0, 0)
                                    )
                                    img_data['image'] = warped
                                    img_data['bbox'] = (min_x, min_y, max_x, max_y)
            else:
                logger.warning(f"Bundle adjustment failed: {stats.get('reason', 'unknown')}")
        
        except Exception as e:
            logger.warning(f"Bundle adjustment error: {e}")
        
        return aligned_images
    
    def _apply_alignment_optimization(
        self,
        image: np.ndarray,
        clahe_clip: float = 2.5,
        clahe_grid: tuple = (8, 8),
        denoise_h: int = 5,
        edge_strength: float = 0.2,
        color_norm: bool = True
    ) -> np.ndarray:
        """
        Apply alignment optimization to a single image.
        
        Args:
            image: Input BGR image
            clahe_clip: CLAHE clip limit (higher = more contrast)
            clahe_grid: CLAHE tile grid size
            denoise_h: Non-local means denoising strength
            edge_strength: Edge enhancement factor (0-1)
            color_norm: Apply color normalization
            
        Returns:
            Optimized image for feature detection
        """
        result = image.copy()
        
        # Step 1: Convert to LAB color space for better processing
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # Step 2: Apply CLAHE to L channel (contrast enhancement)
        # This is the most important step for feature detection
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
        l_enhanced = clahe.apply(l_channel)
        
        # Step 3: Light denoising to reduce false features
        # Use a small radius to preserve edges
        if denoise_h > 0:
            l_enhanced = cv2.fastNlMeansDenoising(l_enhanced, None, denoise_h, 7, 21)
        
        # Step 4: Edge enhancement using unsharp mask
        if edge_strength > 0:
            # Create a blurred version
            blurred = cv2.GaussianBlur(l_enhanced, (0, 0), 3.0)
            # Unsharp mask: enhanced = original + strength * (original - blurred)
            l_enhanced = cv2.addWeighted(l_enhanced, 1.0 + edge_strength, blurred, -edge_strength, 0)
        
        # Step 5: Color normalization (optional)
        if color_norm:
            # Normalize a and b channels to reduce color variation impact
            a_normalized = cv2.normalize(a_channel, None, 110, 146, cv2.NORM_MINMAX)
            b_normalized = cv2.normalize(b_channel, None, 110, 146, cv2.NORM_MINMAX)
        else:
            a_normalized = a_channel
            b_normalized = b_channel
        
        # Merge back
        lab_enhanced = cv2.merge([l_enhanced, a_normalized, b_normalized])
        result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        # Step 6: Final contrast stretch to use full dynamic range
        # This helps SIFT/ORB detect more features
        for i in range(3):
            channel = result[:, :, i]
            min_val = np.percentile(channel, 1)
            max_val = np.percentile(channel, 99)
            if max_val > min_val:
                result[:, :, i] = np.clip(
                    255 * (channel.astype(float) - min_val) / (max_val - min_val),
                    0, 255
                ).astype(np.uint8)
        
        return result
    
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
        """Detect features in all images with adaptive optimization for large sets"""
        features_data = []
        total = len(images_data)
        
        # Adaptive feature count: reduce for large image sets to maintain performance
        # More images = more total features anyway, so fewer per image is fine
        original_max = self.max_features
        if total > 100:
            # Scale down features for very large sets
            self.max_features = min(self.max_features, 2000)
            logger.info(f"Large set ({total} images): reducing max features to {self.max_features}")
        elif total > 50:
            self.max_features = min(self.max_features, 3000)
            logger.info(f"Medium set ({total} images): reducing max features to {self.max_features}")
        
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

                # If very few descriptors, try a contrast-boosted fallback to rescue low-texture/color-similar images
                if descriptors is None or len(descriptors) < 50:
                    try:
                        gray = cv2.cvtColor(img_data['image'], cv2.COLOR_BGR2GRAY)
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                        boosted = clahe.apply(gray)
                        boosted_bgr = cv2.cvtColor(boosted, cv2.COLOR_GRAY2BGR)
                        kp_array, descriptors = self.feature_detector.detect_and_compute(boosted_bgr)
                        logger.debug(f"Fallback CLAHE detect used for image {idx}, features={0 if descriptors is None else len(descriptors)}")
                    except Exception as e:
                        logger.debug(f"CLAHE fallback failed for image {idx}: {e}")
                
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
        
        # Restore original max features setting
        self.max_features = original_max
        
        return features_data
    
    def _match_features(self, features_data: List[Dict]) -> List[Dict]:
        """
        Match features between images with geometric verification.
        
        For large image sets, uses windowed matching to avoid O(n²) complexity:
        - Small sets (≤20 images): Match all pairs
        - Large sets: Match nearby images + periodic jumps for row detection
        
        When geometric_verify is enabled, uses RANSAC to filter bad matches.
        """
        matches = []
        n_images = len(features_data)
        rejected_count = 0
        
        def compute_inlier_stats(match_list, kp_i, kp_j):
            """Run RANSAC to reject bad links and keep only inlier pairs."""
            if match_list is None or len(match_list) < 4:
                return None, None, None
            
            pts_i = []
            pts_j = []
            for m in match_list:
                qi = m.queryIdx
                tj = m.trainIdx
                if qi < len(kp_i) and tj < len(kp_j):
                    pts_i.append([kp_i[qi][0], kp_i[qi][1]])
                    pts_j.append([kp_j[tj][0], kp_j[tj][1]])
            
            if len(pts_i) < 4:
                return None, None, None
            
            pts_i = np.float32(pts_i)
            pts_j = np.float32(pts_j)
            
            # Robustly estimate similarity transform with RANSAC; reject outliers
            _, inliers = cv2.estimateAffinePartial2D(
                pts_i, pts_j,
                method=cv2.RANSAC,
                ransacReprojThreshold=3.0,
                confidence=0.995,
                maxIters=2000
            )
            
            if inliers is None:
                return None, None, None
            
            inlier_mask = inliers.astype(bool).ravel()
            num_inliers = int(np.sum(inlier_mask))
            inlier_ratio = num_inliers / len(match_list) if len(match_list) > 0 else 0.0
            
            # Relaxed: need at least 6 inliers with 25% inlier ratio
            if num_inliers < 6 or inlier_ratio < 0.25:
                return None, None, None
            
            # Keep only inlier matches to feed alignment
            filtered_matches = [m for m, keep in zip(match_list, inlier_mask) if keep]
            return filtered_matches, num_inliers, inlier_ratio
        
        # Check if matcher is a deep learning matcher that needs images
        from ml.advanced_matchers import LoFTRMatcher, SuperGlueMatcher, DISKMatcher
        is_dl_matcher = isinstance(self.matcher, (LoFTRMatcher,))
        is_superglue = isinstance(self.matcher, SuperGlueMatcher)
        
        # Determine matching strategy based on image count
        # n² growth: 10 imgs = 45 pairs, 20 = 190, 50 = 1225, 100 = 4950
        if n_images <= 10:
            # Small sets: match all pairs (fast enough)
            pairs_to_match = [(i, j) for i in range(n_images) for j in range(i + 1, n_images)]
            logger.info(f"Small image set ({n_images}): matching all {len(pairs_to_match)} pairs")
        else:
            # Larger sets: use smart windowed matching to avoid O(n²)
            pairs_to_match = self._get_smart_matching_pairs(n_images, features_data)
            logger.info(f"Image set ({n_images}): smart matching with {len(pairs_to_match)} pairs (saved {n_images*(n_images-1)//2 - len(pairs_to_match)} comparisons)")
        
        total_pairs = len(pairs_to_match)
        
        # Track connectivity for early termination
        connected_images = set()
        target_connectivity = n_images - 1  # Need n-1 edges for spanning tree
        early_stop_check_interval = max(50, total_pairs // 10)  # Check every ~10%
        
        for pair_idx, (i, j) in enumerate(pairs_to_match):
            self._check_cancel()
            
            # Update progress
            if total_pairs > 0:
                progress = 50 + int(20 * (pair_idx / total_pairs))
                self._update_progress(progress, f"Matching features: pair {pair_idx + 1}/{total_pairs}...")
            
            # Early termination check: if we have a well-connected graph, stop early
            if n_images > 15 and pair_idx > 0 and pair_idx % early_stop_check_interval == 0:
                if len(matches) >= target_connectivity * 1.5:
                    # Check if graph is connected
                    graph_nodes = set()
                    for m in matches:
                        graph_nodes.add(m['image_i'])
                        graph_nodes.add(m['image_j'])
                    if len(graph_nodes) >= n_images * 0.95:  # 95% of images connected
                        logger.info(f"Early termination: {len(graph_nodes)}/{n_images} images connected with {len(matches)} matches")
                        break
            
            try:
                if is_dl_matcher:
                    img1 = features_data[i]['image_data']['image']
                    img2 = features_data[j]['image_data']['image']
                    match_result = self.matcher.match(img1, img2)
                elif is_superglue:
                    match_result = self.matcher.match(
                        features_data[i]['keypoints'],
                        features_data[i]['descriptors'],
                        features_data[j]['keypoints'],
                        features_data[j]['descriptors']
                    )
                else:
                    desc1 = features_data[i]['descriptors']
                    desc2 = features_data[j]['descriptors']
                    kp1 = features_data[i]['keypoints']
                    kp2 = features_data[j]['keypoints']
                    
                    if desc1 is None or desc2 is None:
                        continue
                    if len(desc1) == 0 or len(desc2) == 0:
                        continue
                    
                    # Use geometric verification if enabled
                    if self.geometric_verify and hasattr(self.matcher, 'match_with_keypoints'):
                        match_result = self.matcher.match_with_keypoints(kp1, desc1, kp2, desc2)
                    else:
                        match_result = self.matcher.match(desc1, desc2)
                
                # Check if match was rejected by geometric verification
                if match_result.get('rejected_reason'):
                    logger.debug(f"Pair ({i}, {j}): rejected - {match_result['rejected_reason']}")
                    rejected_count += 1
                    continue
                
                num_matches = match_result.get('num_matches', 0) if match_result else 0
                inlier_ratio = match_result.get('inlier_ratio', 1.0)
                
                if num_matches > 0:
                    logger.debug(f"Pair ({i}, {j}): {num_matches} raw matches, confidence: {match_result.get('confidence', 0.0):.4f}")
                
                # Require at least 10 raw matches before RANSAC (relaxed for better coverage)
                if match_result and num_matches >= 10:
                    kp_i = features_data[i]['keypoints']
                    kp_j = features_data[j]['keypoints']
                    filtered_matches, num_inliers, inlier_ratio = compute_inlier_stats(
                        match_result.get('matches', []),
                        kp_i,
                        kp_j
                    )
                    
                    if filtered_matches is None:
                        logger.debug(f"Pair ({i}, {j}) rejected by RANSAC (inliers too low)")
                        continue
                    
                    matches.append({
                        'image_i': i,
                        'image_j': j,
                        'matches': filtered_matches,
                        'num_matches': len(filtered_matches),
                        'num_inliers': num_inliers,
                        'inlier_ratio': inlier_ratio,
                        'confidence': match_result.get('confidence', 0.0) * max(inlier_ratio, 0.1)
                    })
            except Exception as e:
                logger.error(f"Error matching pair ({i}, {j}): {e}")
                continue
        
        if rejected_count > 0:
            logger.info(f"Geometric verification rejected {rejected_count} bad matches")
        logger.info(f"Found {len(matches)} valid matches from {total_pairs} pairs checked")
        return matches
    
    def _select_optimal_images(
        self,
        images_data: List[Dict],
        features_data: List[Dict],
        matches: List[Dict]
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Select only images necessary to cover the panorama area without redundancy.
        
        Uses a greedy algorithm to build a spanning set of images:
        1. Start with the most connected image (best reference)
        2. Add images that extend coverage without too much overlap
        3. Stop when all connected areas are covered
        
        Args:
            images_data: List of image data dictionaries
            features_data: List of feature data dictionaries
            matches: List of match dictionaries
            
        Returns:
            Filtered (images_data, features_data, matches)
        """
        if len(images_data) <= 3:
            return images_data, features_data, matches
        
        n_images = len(images_data)
        
        # Build overlap graph (how much each pair overlaps)
        overlap_graph = {}
        connection_count = [0] * n_images
        
        for match in matches:
            i = match['image_i']
            j = match['image_j']
            
            # Estimate overlap from match confidence and count
            confidence = match.get('confidence', 0.0)
            inlier_ratio = match.get('inlier_ratio', 0.5)
            num_matches = match.get('num_matches', 0)
            
            # Higher confidence + more matches = more overlap
            overlap_estimate = min(confidence * inlier_ratio * 2, 1.0)
            
            overlap_graph[(i, j)] = overlap_estimate
            overlap_graph[(j, i)] = overlap_estimate
            connection_count[i] += 1
            connection_count[j] += 1
        
        # Start with the most connected image
        selected = set()
        start_idx = max(range(n_images), key=lambda x: connection_count[x])
        selected.add(start_idx)
        
        # Track which images we've already considered for coverage
        coverage_provided = {start_idx}
        
        logger.info(f"Coverage selection starting from image {start_idx} ({connection_count[start_idx]} connections)")
        
        # Iteratively add images that extend coverage
        max_iterations = n_images
        for iteration in range(max_iterations):
            best_candidate = None
            best_score = -1
            
            for idx in range(n_images):
                if idx in selected:
                    continue
                
                # Check if this image connects to the selected set
                connects_to_selected = False
                max_overlap_with_selected = 0.0
                
                for sel_idx in selected:
                    key = (idx, sel_idx)
                    if key in overlap_graph:
                        connects_to_selected = True
                        max_overlap_with_selected = max(max_overlap_with_selected, overlap_graph[key])
                
                if not connects_to_selected:
                    continue
                
                # Skip if too much overlap with existing selection (redundant)
                if max_overlap_with_selected > self.max_coverage_overlap:
                    logger.debug(f"Image {idx}: skipping - {max_overlap_with_selected:.1%} overlap exceeds max {self.max_coverage_overlap:.1%}")
                    continue
                
                # Score based on: connections to non-selected (extends coverage) - overlap with selected (redundancy)
                connections_to_new = sum(1 for k in range(n_images) if k not in selected and (idx, k) in overlap_graph)
                
                # Higher score = better candidate
                score = connections_to_new + (1.0 - max_overlap_with_selected) * 2
                
                if score > best_score:
                    best_score = score
                    best_candidate = idx
            
            if best_candidate is None:
                # No more candidates that connect and don't overlap too much
                break
            
            selected.add(best_candidate)
            logger.debug(f"Added image {best_candidate} (score: {best_score:.2f})")
        
        # Ensure we have at least the minimum connected set
        if len(selected) < 2:
            logger.warning("Coverage selection resulted in too few images, using all connected images")
            selected = set(i for i in range(n_images) if connection_count[i] > 0)
        
        # Build index mapping
        selected_list = sorted(selected)
        index_map = {old: new for new, old in enumerate(selected_list)}
        
        # Filter data
        new_images = [images_data[i] for i in selected_list]
        new_features = [features_data[i] for i in selected_list]
        
        # Filter and remap matches
        new_matches = []
        for match in matches:
            i = match['image_i']
            j = match['image_j']
            if i in index_map and j in index_map:
                new_match = match.copy()
                new_match['image_i'] = index_map[i]
                new_match['image_j'] = index_map[j]
                new_matches.append(new_match)
        
        removed = n_images - len(selected)
        if removed > 0:
            logger.info(f"Coverage selection: using {len(selected)}/{n_images} images (removed {removed} redundant)")
        
        return new_images, new_features, new_matches
    
    def _get_smart_matching_pairs(self, n_images: int, features_data: List[Dict] = None) -> List[Tuple[int, int]]:
        """
        Generate smart matching pairs for large image sets.
        
        Strategy for panoramas/microscopy (AutoPano Giga-inspired):
        1. If grid topology detected: use O(n) neighbor-based matching
        2. Match adjacent images (i, i+1) - these definitely overlap
        3. Match nearby images (adaptive window) - catch overlaps
        4. Add grid-based "jump" connections - helps detect row transitions
        5. Use thumbnail similarity pre-filter when available
        
        This reduces O(n²) to O(n*k) where k is the window size + jumps
        With grid detection, can achieve O(n) for regular grids.
        
        Complexity scaling:
        - 10 images: ~45 pairs (all pairs)
        - 50 images: ~400 pairs (smart) or ~100 (grid)
        - 100 images: ~800 pairs (smart) or ~200 (grid)
        - 500 images: ~4000 pairs (smart) or ~1000 (grid)
        """
        # Check if we have cached grid info from previous detection
        if hasattr(self, '_detected_grid_info') and self._detected_grid_info:
            grid_info = self._detected_grid_info
            from core.autopano_features import create_matching_pairs_from_grid
            pairs = create_matching_pairs_from_grid(n_images, grid_info, 'windowed')
            if pairs:
                logger.info(f"Using grid topology: {len(pairs)} pairs for {n_images} images")
                return pairs
        
        pairs = set()
        
        # Estimate grid dimensions based on typical scanning patterns
        # Assume roughly square-ish grid with some extra columns
        est_cols = int(np.ceil(np.sqrt(n_images * 1.3)))
        est_rows = int(np.ceil(n_images / est_cols))
        
        # Adaptive window size: smaller for very large sets
        if n_images <= 30:
            window = min(8, n_images // 3 + 2)
        elif n_images <= 100:
            window = min(6, n_images // 10 + 3)
        elif n_images <= 500:
            window = 5
        else:
            window = 4  # Very large sets need tight windows
        
        logger.debug(f"Smart matching: n={n_images}, est_grid={est_cols}x{est_rows}, window={window}")
        
        for i in range(n_images):
            # 1. Match nearby images (sliding window) - sequential neighbors
            for offset in range(1, window + 1):
                j = i + offset
                if j < n_images:
                    pairs.add((min(i, j), max(i, j)))
            
            # 2. Add row transition connections (jump by estimated column count)
            # These catch connections between rows in a grid scan
            for jump in [est_cols - 1, est_cols, est_cols + 1]:
                j = i + jump
                if 0 <= j < n_images and j != i:
                    pairs.add((min(i, j), max(i, j)))
            
            # 3. For larger sets, add sparse long-range probes
            # Check every Nth image for possible connections
            if n_images > 50:
                probe_step = max(10, n_images // 20)
                if i % probe_step == 0:
                    # Probe a few distant images
                    for jump in [est_cols * 2, n_images // 3, n_images // 2]:
                        j = i + jump
                        if 0 <= j < n_images:
                            pairs.add((min(i, j), max(i, j)))
        
        # 4. Ensure we have at least some coverage of the full range
        # Add a few strategic pairs at regular intervals
        step = max(1, n_images // 10)
        for i in range(0, n_images, step):
            for offset in [1, 2, est_cols]:
                j = i + offset
                if 0 <= j < n_images:
                    pairs.add((min(i, j), max(i, j)))
        
        result = sorted(list(pairs))
        logger.info(f"Smart matching: {n_images} images → {len(result)} pairs (vs {n_images*(n_images-1)//2} all pairs)")
        return result
    
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

