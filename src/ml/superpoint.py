"""
SuperPoint: Self-Supervised Interest Point Detection and Description

Based on: "SuperPoint: Self-Supervised Interest Point Detection and Description"
by DeTone, Malisiewicz, and Rabinovich (CVPR 2018)

SuperPoint is a CNN-based feature detector that:
1. Detects repeatable keypoints even in repetitive scenes
2. Produces 256-dim descriptors optimized for matching
3. Works much better than SIFT for large-scale panoramas

This implementation:
- Uses PyTorch if available with pre-trained weights
- Falls back to enhanced SIFT detection if PyTorch unavailable
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict
import logging
import os

logger = logging.getLogger(__name__)

# Check for PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. SuperPoint will use enhanced SIFT fallback.")


class SuperPointNet(nn.Module):
    """
    SuperPoint Convolutional Neural Network architecture.
    
    Encoder-Decoder network that produces:
    - Keypoint heatmap (H/8 x W/8 x 65)
    - Descriptor map (H/8 x W/8 x 256)
    """
    
    def __init__(self):
        super(SuperPointNet, self).__init__()
        
        # Shared encoder
        self.conv1a = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        
        # Keypoint detector head
        self.convPa = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(256, 65, kernel_size=1, stride=1, padding=0)  # 64 cells + 1 dustbin
        
        # Descriptor head
        self.convDa = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input image tensor (B, 1, H, W)
            
        Returns:
            Tuple of (keypoint_heatmap, descriptors)
        """
        # Shared encoder
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        
        # Keypoint detector head
        cPa = self.relu(self.convPa(x))
        semi = self.convPb(cPa)
        
        # Descriptor head
        cDa = self.relu(self.convDa(x))
        desc = self.convDb(cDa)
        
        # Normalize descriptors
        dn = torch.norm(desc, p=2, dim=1, keepdim=True)
        desc = desc.div(torch.clamp(dn, min=1e-8))
        
        return semi, desc


class SuperPointDetector:
    """
    SuperPoint feature detector for large-scale panorama stitching.
    
    Advantages over SIFT:
    - More repeatable in repetitive scenes (windows, grids, tiles)
    - Better descriptors for matching
    - Trained end-to-end for feature matching
    """
    
    def __init__(
        self,
        use_gpu: bool = False,
        n_features: int = 2000,
        confidence_threshold: float = 0.015,
        nms_radius: int = 4,
        use_color: bool = False
    ):
        """
        Initialize SuperPoint detector.
        
        Args:
            use_gpu: Enable GPU acceleration
            n_features: Maximum number of features to detect
            confidence_threshold: Keypoint detection threshold
            nms_radius: Non-maximum suppression radius
            use_color: Extract color descriptors (appended to SuperPoint descriptors)
        """
        self.use_gpu = use_gpu and TORCH_AVAILABLE and torch.cuda.is_available()
        self.n_features = n_features
        self.confidence_threshold = confidence_threshold
        self.nms_radius = nms_radius
        self.use_color = use_color
        
        self.model = None
        self.device = None
        
        if TORCH_AVAILABLE:
            self._load_model()
        
        if self.model is None:
            logger.info("SuperPoint using enhanced SIFT fallback (PyTorch model not available)")
        else:
            logger.info(f"SuperPoint initialized (GPU: {self.use_gpu}, "
                       f"threshold: {confidence_threshold}, nms: {nms_radius})")
    
    def _load_model(self):
        """Load SuperPoint model weights."""
        try:
            self.device = torch.device('cuda' if self.use_gpu else 'cpu')
            self.model = SuperPointNet()
            
            # Try to load pre-trained weights
            weights_path = self._find_weights()
            if weights_path:
                state_dict = torch.load(weights_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                logger.info(f"Loaded SuperPoint weights from {weights_path}")
            else:
                # Initialize with random weights (will use fallback)
                logger.warning("SuperPoint weights not found, using random initialization")
                self.model = None
                return
            
            self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            logger.warning(f"Could not load SuperPoint model: {e}")
            self.model = None
    
    def _find_weights(self) -> Optional[str]:
        """Find SuperPoint weights file."""
        # Check common locations
        possible_paths = [
            'models/superpoint_v1.pth',
            'weights/superpoint_v1.pth',
            os.path.expanduser('~/.cache/superpoint/superpoint_v1.pth'),
            'superpoint_v1.pth'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        return None
    
    def detect_and_compute(
        self,
        image: np.ndarray,
        max_dimension: int = 1600
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect keypoints and compute descriptors.
        
        Args:
            image: Input image (BGR or grayscale)
            max_dimension: Maximum image dimension for detection
            
        Returns:
            Tuple of (keypoints, descriptors)
            - keypoints: Nx4 array (x, y, size, angle)
            - descriptors: Nx256 array (or Nx256+48 with color)
        """
        if self.model is None:
            return self._fallback_detect(image, max_dimension)
        
        return self._superpoint_detect(image, max_dimension)
    
    def _superpoint_detect(
        self,
        image: np.ndarray,
        max_dimension: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Detect using SuperPoint neural network."""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        h, w = gray.shape[:2]
        
        # Scale down if needed
        scale = 1.0
        if max(h, w) > max_dimension:
            scale = max_dimension / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Prepare input tensor
        gray_tensor = torch.from_numpy(gray.astype(np.float32) / 255.0)
        gray_tensor = gray_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        gray_tensor = gray_tensor.to(self.device)
        
        # Run inference
        with torch.no_grad():
            semi, desc = self.model(gray_tensor)
        
        # Process keypoint heatmap
        semi = semi.squeeze(0)  # Remove batch dim
        
        # Softmax over cells
        dense = F.softmax(semi, dim=0)
        nodust = dense[:-1, :, :]  # Remove dustbin channel
        
        # Reshape to full resolution
        Hc, Wc = nodust.shape[1], nodust.shape[2]
        nodust = nodust.permute(1, 2, 0)  # H, W, C
        heatmap = nodust.reshape(Hc, Wc, 8, 8)
        heatmap = heatmap.permute(0, 2, 1, 3)  # H, 8, W, 8
        heatmap = heatmap.reshape(Hc * 8, Wc * 8)
        
        # Extract keypoints
        heatmap_np = heatmap.cpu().numpy()
        
        # Apply NMS
        keypoints = self._nms_fast(heatmap_np, self.confidence_threshold, self.nms_radius)
        
        if len(keypoints) == 0:
            return np.array([]), None
        
        # Limit to n_features
        if len(keypoints) > self.n_features:
            # Sort by score and keep top N
            scores = heatmap_np[keypoints[:, 1].astype(int), keypoints[:, 0].astype(int)]
            top_indices = np.argsort(scores)[::-1][:self.n_features]
            keypoints = keypoints[top_indices]
        
        # Sample descriptors at keypoint locations
        desc = desc.squeeze(0)  # B, C, H, W -> C, H, W
        
        # Normalize keypoint coordinates to [-1, 1] for grid_sample
        kp_norm = keypoints.copy()
        kp_norm[:, 0] = (kp_norm[:, 0] / (Wc * 8 - 1)) * 2 - 1
        kp_norm[:, 1] = (kp_norm[:, 1] / (Hc * 8 - 1)) * 2 - 1
        
        kp_tensor = torch.from_numpy(kp_norm.astype(np.float32))
        kp_tensor = kp_tensor.view(1, 1, -1, 2).to(self.device)
        
        # Sample descriptors
        desc_sampled = F.grid_sample(desc.unsqueeze(0), kp_tensor, mode='bilinear', align_corners=True)
        desc_sampled = desc_sampled.squeeze().t()  # N, 256
        
        # Normalize descriptors
        desc_np = desc_sampled.cpu().numpy()
        desc_np = desc_np / (np.linalg.norm(desc_np, axis=1, keepdims=True) + 1e-8)
        
        # Scale keypoints back to original coordinates
        if scale != 1.0:
            keypoints = keypoints / scale
        
        # Convert to standard format: (x, y, size, angle)
        kp_array = np.zeros((len(keypoints), 4), dtype=np.float32)
        kp_array[:, :2] = keypoints
        kp_array[:, 2] = 8.0  # Default size
        kp_array[:, 3] = 0.0  # No angle
        
        return kp_array, desc_np.astype(np.float32)
    
    def _nms_fast(
        self,
        heatmap: np.ndarray,
        threshold: float,
        radius: int
    ) -> np.ndarray:
        """
        Fast non-maximum suppression.
        
        Args:
            heatmap: 2D array of keypoint scores
            threshold: Minimum score threshold
            radius: NMS radius
            
        Returns:
            Nx2 array of (x, y) keypoint coordinates
        """
        # Threshold
        mask = heatmap > threshold
        
        if not np.any(mask):
            return np.array([])
        
        # Apply max pooling for NMS
        from scipy.ndimage import maximum_filter
        local_max = maximum_filter(heatmap, size=2*radius+1)
        
        # Keep only local maxima above threshold
        keypoints_mask = (heatmap == local_max) & mask
        
        # Get coordinates
        y_coords, x_coords = np.where(keypoints_mask)
        keypoints = np.column_stack([x_coords, y_coords])
        
        return keypoints.astype(np.float32)
    
    def _fallback_detect(
        self,
        image: np.ndarray,
        max_dimension: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fallback to enhanced SIFT when SuperPoint is not available.
        
        Uses edge-prioritized SIFT with additional preprocessing for
        better performance on repetitive scenes.
        """
        # Import the LP-SIFT detector
        from ml.feature_detector import LP_SIFTDetector
        
        detector = LP_SIFTDetector(
            use_gpu=self.use_gpu,
            n_features=self.n_features,
            use_color=self.use_color
        )
        
        return detector.detect_and_compute(image, max_dimension)


def download_superpoint_weights(output_path: str = 'models/superpoint_v1.pth') -> bool:
    """
    Download SuperPoint weights from the official source.
    
    Returns:
        True if successful, False otherwise
    """
    import urllib.request
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # SuperPoint weights URL (you'd need to host these or use the official source)
    # The official weights are at: https://github.com/magicleap/SuperPointPretrainedNetwork
    url = "https://github.com/magicleap/SuperPointPretrainedNetwork/raw/master/superpoint_v1.pth"
    
    try:
        logger.info(f"Downloading SuperPoint weights to {output_path}...")
        urllib.request.urlretrieve(url, output_path)
        logger.info("Download complete!")
        return True
    except Exception as e:
        logger.error(f"Failed to download SuperPoint weights: {e}")
        return False






