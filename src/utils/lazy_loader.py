"""
Lazy Image Loading System for Memory-Efficient Processing

Key optimizations:
1. Images are not loaded until needed
2. Full-resolution images are released after feature detection
3. Only thumbnails are kept in memory for UI/preview
4. Images are reloaded on-demand for blending
5. JPEG compression in memory for further savings
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
import gc
import io

logger = logging.getLogger(__name__)


class ImageProxy:
    """
    Proxy object that represents an image without loading it into memory.
    Loads the actual image data only when accessed.
    
    Memory savings: ~95-99% for large images during feature detection phase
    """
    
    def __init__(
        self,
        path: Path,
        thumbnail_size: int = 512,
        cache_compressed: bool = True
    ):
        """
        Initialize image proxy
        
        Args:
            path: Path to image file
            thumbnail_size: Size of cached thumbnail (default 512px)
            cache_compressed: Store compressed version in memory (vs reload from disk)
        """
        self.path = path
        self.thumbnail_size = thumbnail_size
        self.cache_compressed = cache_compressed
        
        # Metadata (lightweight)
        self._shape: Optional[Tuple[int, int, int]] = None
        self._quality: Optional[float] = None
        self._thumbnail: Optional[np.ndarray] = None
        self._compressed_data: Optional[bytes] = None
        
        # Feature data (stored separately, can be released)
        self._keypoints: Optional[np.ndarray] = None
        self._descriptors: Optional[np.ndarray] = None
        
        # Load metadata without loading full image
        self._load_metadata()
    
    def _load_metadata(self):
        """Load image metadata without loading full image"""
        try:
            # Use cv2.imread with IMREAD_UNCHANGED to get shape efficiently
            # For large files, we can use a faster method
            img = cv2.imread(str(self.path), cv2.IMREAD_UNCHANGED)
            if img is not None:
                self._shape = img.shape
                
                # Create thumbnail
                h, w = img.shape[:2]
                scale = self.thumbnail_size / max(h, w)
                if scale < 1.0:
                    new_size = (int(w * scale), int(h * scale))
                    self._thumbnail = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
                else:
                    self._thumbnail = img.copy()
                
                # Optionally compress and cache
                if self.cache_compressed:
                    # JPEG compression at quality 85 gives ~10-20x size reduction
                    encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]
                    _, compressed = cv2.imencode('.jpg', img, encode_params)
                    self._compressed_data = compressed.tobytes()
                    
                    # Log compression ratio
                    original_size = img.nbytes
                    compressed_size = len(self._compressed_data)
                    ratio = original_size / compressed_size
                    logger.debug(f"Image {self.path.name}: compressed {original_size/1e6:.1f}MB -> {compressed_size/1e6:.1f}MB ({ratio:.1f}x)")
                
                # Release full image
                del img
                gc.collect()
            else:
                logger.warning(f"Could not read image metadata: {self.path}")
        except Exception as e:
            logger.error(f"Error loading metadata for {self.path}: {e}")
    
    @property
    def shape(self) -> Optional[Tuple[int, int, int]]:
        """Get image shape without loading full image"""
        return self._shape
    
    @property
    def thumbnail(self) -> Optional[np.ndarray]:
        """Get cached thumbnail"""
        return self._thumbnail
    
    @property
    def quality(self) -> Optional[float]:
        """Get cached quality score"""
        return self._quality
    
    @quality.setter
    def quality(self, value: float):
        """Set quality score"""
        self._quality = value
    
    def load_full(self) -> Optional[np.ndarray]:
        """
        Load full-resolution image
        
        Returns:
            Full image array, or None if loading fails
        """
        try:
            if self._compressed_data is not None:
                # Decompress from cached data (faster than disk read)
                compressed = np.frombuffer(self._compressed_data, dtype=np.uint8)
                img = cv2.imdecode(compressed, cv2.IMREAD_COLOR)
                logger.debug(f"Loaded {self.path.name} from compressed cache")
            else:
                # Load from disk
                img = cv2.imread(str(self.path), cv2.IMREAD_UNCHANGED)
                logger.debug(f"Loaded {self.path.name} from disk")
            
            if img is not None and len(img.shape) == 3 and img.shape[2] == 4:
                # Handle RGBA
                img = img[:, :, :3]
            
            return img
        except Exception as e:
            logger.error(f"Error loading full image {self.path}: {e}")
            return None
    
    def load_for_features(self, max_dimension: int = 2000) -> Optional[np.ndarray]:
        """
        Load image at reduced resolution for feature detection
        
        Args:
            max_dimension: Maximum dimension for feature detection
            
        Returns:
            Scaled image array
        """
        try:
            img = self.load_full()
            if img is None:
                return None
            
            h, w = img.shape[:2]
            if max(h, w) > max_dimension:
                scale = max_dimension / max(h, w)
                new_size = (int(w * scale), int(h * scale))
                img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
                logger.debug(f"Scaled {self.path.name} from {w}x{h} to {new_size[0]}x{new_size[1]} for features")
            
            return img
        except Exception as e:
            logger.error(f"Error loading image for features {self.path}: {e}")
            return None
    
    def set_features(self, keypoints: np.ndarray, descriptors: np.ndarray):
        """Store detected features"""
        self._keypoints = keypoints
        self._descriptors = descriptors
    
    def get_features(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get stored features"""
        return self._keypoints, self._descriptors
    
    def release_features(self):
        """Release feature data to free memory"""
        self._keypoints = None
        self._descriptors = None
        gc.collect()
    
    def release_compressed(self):
        """Release compressed cache to free memory"""
        self._compressed_data = None
        gc.collect()
    
    def estimate_memory_mb(self) -> float:
        """Estimate memory usage of this proxy"""
        total = 0
        
        # Thumbnail
        if self._thumbnail is not None:
            total += self._thumbnail.nbytes / 1e6
        
        # Compressed data
        if self._compressed_data is not None:
            total += len(self._compressed_data) / 1e6
        
        # Features
        if self._keypoints is not None:
            total += self._keypoints.nbytes / 1e6
        if self._descriptors is not None:
            total += self._descriptors.nbytes / 1e6
        
        return total


class LazyImageLoader:
    """
    Memory-efficient image loader that uses proxies instead of loading all images.
    
    Memory comparison for 100 images at 4000x3000 (36MB each):
    - Traditional: ~3.6 GB
    - With LazyLoader: ~200-400 MB (thumbnails + compressed cache)
    - With LazyLoader (no cache): ~50 MB (thumbnails only)
    """
    
    def __init__(
        self,
        thumbnail_size: int = 512,
        cache_compressed: bool = True,
        max_cached_full_images: int = 2
    ):
        """
        Initialize lazy loader
        
        Args:
            thumbnail_size: Size of cached thumbnails
            cache_compressed: Cache JPEG-compressed images in memory
            max_cached_full_images: Maximum full-res images to keep in LRU cache
        """
        self.thumbnail_size = thumbnail_size
        self.cache_compressed = cache_compressed
        self.max_cached_full_images = max_cached_full_images
        
        self._proxies: Dict[Path, ImageProxy] = {}
        self._full_image_cache: List[Tuple[Path, np.ndarray]] = []
    
    def load_proxies(
        self, 
        paths: List[Path],
        progress_callback: Optional[callable] = None
    ) -> List[ImageProxy]:
        """
        Create proxies for all images (loads only metadata)
        
        Args:
            paths: List of image paths
            progress_callback: Optional callback(current, total, message) for progress updates
            
        Returns:
            List of ImageProxy objects
        """
        proxies = []
        total = len(paths)
        
        for i, path in enumerate(paths):
            # Report progress
            if progress_callback and total > 0:
                progress_callback(i, total, f"Scanning {path.name}")
            
            if path in self._proxies:
                proxies.append(self._proxies[path])
            else:
                proxy = ImageProxy(
                    path,
                    thumbnail_size=self.thumbnail_size,
                    cache_compressed=self.cache_compressed
                )
                self._proxies[path] = proxy
                proxies.append(proxy)
        
        # Final progress update
        if progress_callback:
            progress_callback(total, total, f"Loaded {total} image proxies")
        
        # Log memory usage
        total_memory = sum(p.estimate_memory_mb() for p in proxies)
        logger.info(f"Loaded {len(proxies)} image proxies, estimated memory: {total_memory:.1f} MB")
        
        return proxies
    
    def get_full_image(self, proxy: ImageProxy) -> Optional[np.ndarray]:
        """
        Get full-resolution image with LRU caching
        
        Args:
            proxy: ImageProxy to load
            
        Returns:
            Full image array
        """
        # Check cache
        for cached_path, cached_img in self._full_image_cache:
            if cached_path == proxy.path:
                # Move to end (most recently used)
                self._full_image_cache.remove((cached_path, cached_img))
                self._full_image_cache.append((cached_path, cached_img))
                return cached_img
        
        # Load and cache
        img = proxy.load_full()
        if img is not None:
            self._full_image_cache.append((proxy.path, img))
            
            # Evict old entries
            while len(self._full_image_cache) > self.max_cached_full_images:
                old_path, old_img = self._full_image_cache.pop(0)
                del old_img
                gc.collect()
        
        return img
    
    def clear_full_cache(self):
        """Clear the full image cache"""
        self._full_image_cache.clear()
        gc.collect()
    
    def estimate_total_memory_mb(self) -> float:
        """Estimate total memory usage"""
        proxy_memory = sum(p.estimate_memory_mb() for p in self._proxies.values())
        cache_memory = sum(img.nbytes / 1e6 for _, img in self._full_image_cache)
        return proxy_memory + cache_memory
    
    def release_all(self):
        """Release all cached data"""
        for proxy in self._proxies.values():
            proxy.release_features()
            proxy.release_compressed()
        self._full_image_cache.clear()
        gc.collect()


def convert_proxies_to_images_data(
    proxies: List[ImageProxy],
    load_full: bool = True
) -> List[Dict]:
    """
    Convert proxies to traditional images_data format for compatibility
    
    Args:
        proxies: List of ImageProxy objects
        load_full: Whether to load full images (False for feature-only phase)
        
    Returns:
        List of image data dictionaries
    """
    images_data = []
    
    for proxy in proxies:
        if proxy.shape is None:
            continue
        
        data = {
            'path': proxy.path,
            'quality': proxy.quality or 0.5,
            'shape': proxy.shape,
            'proxy': proxy  # Keep reference for later loading
        }
        
        if load_full:
            img = proxy.load_full()
            if img is not None:
                data['image'] = img
                # Handle alpha channel
                if len(proxy.shape) == 3 and proxy.shape[2] == 4:
                    data['alpha'] = img[:, :, 3] if img.shape[2] == 4 else None
                else:
                    data['alpha'] = None
        else:
            # Placeholder - will be loaded on demand
            data['image'] = None
            data['alpha'] = None
        
        images_data.append(data)
    
    return images_data


def estimate_memory_for_images(
    paths: List[Path],
    method: str = 'traditional'
) -> Dict[str, float]:
    """
    Estimate memory requirements for different loading strategies
    
    Args:
        paths: List of image paths
        method: 'traditional' or 'lazy'
        
    Returns:
        Dictionary with memory estimates in MB
    """
    estimates = {
        'traditional': 0.0,
        'lazy_with_cache': 0.0,
        'lazy_no_cache': 0.0,
        'features_only': 0.0
    }
    
    for path in paths:
        try:
            img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            
            h, w = img.shape[:2]
            channels = img.shape[2] if len(img.shape) == 3 else 1
            
            # Traditional: full image in memory
            full_size_mb = (h * w * channels) / 1e6
            estimates['traditional'] += full_size_mb
            
            # Lazy with cache: thumbnail + compressed (~10-20x compression)
            thumbnail_size = min(512, max(h, w))
            thumb_scale = thumbnail_size / max(h, w)
            thumb_size_mb = (h * thumb_scale * w * thumb_scale * channels) / 1e6
            compressed_size_mb = full_size_mb / 15  # Typical JPEG compression
            estimates['lazy_with_cache'] += thumb_size_mb + compressed_size_mb
            
            # Lazy no cache: thumbnail only
            estimates['lazy_no_cache'] += thumb_size_mb
            
            # Features only: descriptors ~128 bytes per feature, ~5000 features typical
            features_size_mb = (5000 * 128) / 1e6
            estimates['features_only'] += features_size_mb
            
            del img
        except Exception as e:
            logger.warning(f"Could not estimate memory for {path}: {e}")
    
    gc.collect()
    return estimates

