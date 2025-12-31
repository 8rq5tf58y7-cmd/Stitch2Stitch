"""
Post-processing filters for image enhancement
"""

import numpy as np
import cv2
import logging
from typing import Optional, Dict, Tuple

logger = logging.getLogger(__name__)


class ImagePostProcessor:
    """Apply post-processing enhancements to images"""
    
    def __init__(self):
        self.interpolation_methods = {
            'lanczos': cv2.INTER_LANCZOS4,
            'cubic': cv2.INTER_CUBIC,
            'linear': cv2.INTER_LINEAR,
            'nearest': cv2.INTER_NEAREST
        }
    
    def process(
        self,
        image: np.ndarray,
        sharpen: bool = False,
        sharpen_amount: float = 1.0,
        denoise: bool = False,
        denoise_strength: int = 5,
        clahe: bool = False,
        clahe_strength: float = 2.0,
        shadow_removal: bool = False,
        shadow_strength: float = 0.5,
        deblur: bool = False,
        deblur_radius: float = 1.5,
        upscale_factor: float = 1.0,
        interpolation: str = 'lanczos'
    ) -> np.ndarray:
        """
        Apply post-processing to an image
        
        Args:
            image: Input image (BGR format)
            sharpen: Apply unsharp mask sharpening
            sharpen_amount: Sharpening strength (0.5-3.0)
            denoise: Apply non-local means denoising
            denoise_strength: Denoising filter strength (1-20)
            clahe: Apply adaptive histogram equalization
            clahe_strength: CLAHE clip limit (1.0-5.0, default 2.0)
            shadow_removal: Apply shadow removal / exposure equalization
            shadow_strength: Shadow removal strength (0.0-1.0)
            deblur: Apply Wiener deconvolution
            deblur_radius: Blur radius estimate for deconvolution
            upscale_factor: Upscaling factor (1.0 = no upscale)
            interpolation: Interpolation method for upscaling
            
        Returns:
            Processed image
        """
        if image is None or image.size == 0:
            return image
        
        result = image.copy()
        
        # Apply in order: shadow removal -> denoise -> deblur -> clahe -> sharpen -> upscale
        
        if shadow_removal:
            logger.info(f"Applying shadow removal (strength={shadow_strength})")
            result = self.apply_shadow_removal(result, shadow_strength)
        
        if denoise:
            logger.info(f"Applying denoising (strength={denoise_strength})")
            result = self.apply_denoise(result, denoise_strength)
        
        if deblur:
            logger.info(f"Applying deblur (radius={deblur_radius})")
            result = self.apply_deblur(result, deblur_radius)
        
        if clahe:
            logger.info(f"Applying CLAHE contrast enhancement (strength={clahe_strength})")
            result = self.apply_clahe(result, clip_limit=clahe_strength)
        
        if sharpen:
            logger.info(f"Applying sharpening (amount={sharpen_amount})")
            result = self.apply_sharpen(result, sharpen_amount)
        
        if upscale_factor > 1.0:
            logger.info(f"Upscaling by {upscale_factor}x using {interpolation}")
            result = self.apply_upscale(result, upscale_factor, interpolation)
        
        return result
    
    def apply_sharpen(self, image: np.ndarray, amount: float = 1.0) -> np.ndarray:
        """
        Apply unsharp mask sharpening
        
        Uses Gaussian blur to create a blurred version, then enhances
        the difference between original and blurred (the "unsharp mask").
        
        Args:
            image: Input image
            amount: Sharpening strength (0.5=subtle, 1.0=normal, 2.0=strong)
            
        Returns:
            Sharpened image
        """
        if amount <= 0:
            return image
        
        # Gaussian blur radius based on image size
        # Smaller radius for sharper results
        blur_radius = max(3, int(min(image.shape[:2]) / 500) * 2 + 1)
        if blur_radius % 2 == 0:
            blur_radius += 1
        
        # Create blurred version
        blurred = cv2.GaussianBlur(image, (blur_radius, blur_radius), 0)
        
        # Unsharp mask: original + amount * (original - blurred)
        # Clip to valid range
        sharpened = cv2.addWeighted(
            image, 1.0 + amount,
            blurred, -amount,
            0
        )
        
        return np.clip(sharpened, 0, 255).astype(np.uint8)
    
    def apply_sharpen_laplacian(self, image: np.ndarray, strength: float = 0.5) -> np.ndarray:
        """
        Apply Laplacian-based sharpening
        
        Enhances edges using the Laplacian operator.
        
        Args:
            image: Input image
            strength: Enhancement strength
            
        Returns:
            Sharpened image
        """
        # Convert to float for processing
        img_float = image.astype(np.float32)
        
        # Apply Laplacian to each channel
        if len(image.shape) == 3:
            laplacian = np.zeros_like(img_float)
            for i in range(3):
                laplacian[:, :, i] = cv2.Laplacian(img_float[:, :, i], cv2.CV_32F)
        else:
            laplacian = cv2.Laplacian(img_float, cv2.CV_32F)
        
        # Add scaled Laplacian to original
        sharpened = img_float - strength * laplacian
        
        return np.clip(sharpened, 0, 255).astype(np.uint8)
    
    def apply_denoise(self, image: np.ndarray, strength: int = 5) -> np.ndarray:
        """
        Apply non-local means denoising
        
        This is a sophisticated denoising algorithm that averages similar
        patches throughout the image, preserving edges while removing noise.
        
        Args:
            image: Input image
            strength: Filter strength h (higher = more denoising, 3-10 typical)
            
        Returns:
            Denoised image
        """
        if len(image.shape) == 3:
            # Color image - use color version for better results
            # Parameters: src, dst, h, hColor, templateWindowSize, searchWindowSize
            return cv2.fastNlMeansDenoisingColored(
                image,
                None,
                strength,      # h - filter strength for luminance
                strength,      # hColor - filter strength for color components
                7,             # templateWindowSize
                21             # searchWindowSize
            )
        else:
            # Grayscale
            return cv2.fastNlMeansDenoising(
                image,
                None,
                strength,      # h
                7,             # templateWindowSize
                21             # searchWindowSize
            )
    
    def apply_clahe(
        self,
        image: np.ndarray,
        clip_limit: float = 2.0,
        tile_size: Tuple[int, int] = (8, 8)
    ) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        
        Improves local contrast while limiting noise amplification.
        Works on the L channel in LAB color space for color images.
        
        Args:
            image: Input image
            clip_limit: Contrast limiting threshold
            tile_size: Size of grid for histogram equalization
            
        Returns:
            Contrast-enhanced image
        """
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        
        if len(image.shape) == 3:
            # Convert to LAB, apply CLAHE to L channel
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            return clahe.apply(image)
    
    def apply_shadow_removal(
        self,
        image: np.ndarray,
        strength: float = 0.5
    ) -> np.ndarray:
        """
        Remove shadows and equalize exposure across the image
        
        Uses a combination of techniques:
        1. Estimates illumination using a large Gaussian blur
        2. Divides out the illumination to normalize lighting
        3. Blends with original based on strength
        
        Args:
            image: Input image
            strength: Effect strength (0.0 = none, 1.0 = full correction)
            
        Returns:
            Shadow-corrected image
        """
        if strength <= 0:
            return image
        
        strength = min(1.0, strength)  # Clamp to max 1.0
        
        if len(image.shape) == 3:
            # Work in LAB color space for better results
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_channel = lab[:, :, 0].astype(np.float32)
            
            # Estimate illumination with large Gaussian blur
            # Kernel size based on image size (larger = more global correction)
            kernel_size = max(31, min(image.shape[:2]) // 8)
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            illumination = cv2.GaussianBlur(l_channel, (kernel_size, kernel_size), 0)
            
            # Avoid division by zero
            illumination = np.maximum(illumination, 1.0)
            
            # Normalize: divide by illumination and rescale
            # Target mean luminance (middle gray)
            target_mean = 128.0
            mean_illum = np.mean(illumination)
            
            # Correct the L channel
            corrected_l = (l_channel / illumination) * target_mean
            
            # Blend with original based on strength
            corrected_l = l_channel * (1 - strength) + corrected_l * strength
            
            # Clip and convert back
            lab[:, :, 0] = np.clip(corrected_l, 0, 255).astype(np.uint8)
            
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            # Grayscale
            img_float = image.astype(np.float32)
            
            kernel_size = max(31, min(image.shape[:2]) // 8)
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            illumination = cv2.GaussianBlur(img_float, (kernel_size, kernel_size), 0)
            illumination = np.maximum(illumination, 1.0)
            
            target_mean = 128.0
            corrected = (img_float / illumination) * target_mean
            corrected = img_float * (1 - strength) + corrected * strength
            
            return np.clip(corrected, 0, 255).astype(np.uint8)
    
    def apply_deblur(self, image: np.ndarray, radius: float = 1.5) -> np.ndarray:
        """
        Apply Wiener deconvolution for deblurring
        
        Estimates a Gaussian blur kernel and applies Wiener filtering
        to reverse the blur effect.
        
        Args:
            image: Input image
            radius: Estimated blur radius
            
        Returns:
            Deblurred image
        """
        # Create Gaussian blur kernel (PSF - Point Spread Function)
        kernel_size = int(radius * 4) | 1  # Ensure odd
        if kernel_size < 3:
            kernel_size = 3
        
        psf = self._create_gaussian_psf(kernel_size, radius)
        
        # Apply Wiener deconvolution per channel
        if len(image.shape) == 3:
            result = np.zeros_like(image, dtype=np.float32)
            for i in range(3):
                result[:, :, i] = self._wiener_filter(
                    image[:, :, i].astype(np.float32),
                    psf
                )
            return np.clip(result, 0, 255).astype(np.uint8)
        else:
            result = self._wiener_filter(image.astype(np.float32), psf)
            return np.clip(result, 0, 255).astype(np.uint8)
    
    def _create_gaussian_psf(self, size: int, sigma: float) -> np.ndarray:
        """Create a Gaussian point spread function"""
        x = np.arange(size) - size // 2
        y = np.arange(size) - size // 2
        X, Y = np.meshgrid(x, y)
        psf = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
        return psf / psf.sum()
    
    def _wiener_filter(
        self,
        image: np.ndarray,
        psf: np.ndarray,
        noise_variance: float = 0.01
    ) -> np.ndarray:
        """
        Apply Wiener filter deconvolution
        
        Args:
            image: Blurred image (single channel, float32)
            psf: Point spread function (blur kernel)
            noise_variance: Estimated noise variance (regularization)
            
        Returns:
            Deblurred image
        """
        # Pad PSF to image size
        psf_padded = np.zeros_like(image)
        kh, kw = psf.shape
        ph, pw = psf_padded.shape
        
        # Center the PSF
        y_start = (ph - kh) // 2
        x_start = (pw - kw) // 2
        psf_padded[y_start:y_start+kh, x_start:x_start+kw] = psf
        
        # Shift to corner for FFT
        psf_padded = np.roll(psf_padded, -y_start, axis=0)
        psf_padded = np.roll(psf_padded, -x_start, axis=1)
        
        # FFT of image and PSF
        image_fft = np.fft.fft2(image)
        psf_fft = np.fft.fft2(psf_padded)
        
        # Wiener filter: H* / (|H|^2 + NSR)
        # where H is PSF in frequency domain, NSR is noise-to-signal ratio
        psf_fft_conj = np.conj(psf_fft)
        psf_power = np.abs(psf_fft) ** 2
        
        # Add small epsilon to avoid division by zero
        wiener_filter = psf_fft_conj / (psf_power + noise_variance + 1e-10)
        
        # Apply filter
        result_fft = image_fft * wiener_filter
        result = np.real(np.fft.ifft2(result_fft))
        
        return result
    
    def apply_upscale(
        self,
        image: np.ndarray,
        factor: float,
        interpolation: str = 'lanczos'
    ) -> np.ndarray:
        """
        Upscale image using high-quality interpolation
        
        Args:
            image: Input image
            factor: Scale factor (e.g., 2.0 for 2x)
            interpolation: Interpolation method
            
        Returns:
            Upscaled image
        """
        if factor <= 1.0:
            return image
        
        h, w = image.shape[:2]
        new_h = int(h * factor)
        new_w = int(w * factor)
        
        interp_method = self.interpolation_methods.get(
            interpolation.lower(),
            cv2.INTER_LANCZOS4
        )
        
        return cv2.resize(image, (new_w, new_h), interpolation=interp_method)
    
    def get_interpolation_flag(self, method: str) -> int:
        """Get OpenCV interpolation flag from method name"""
        return self.interpolation_methods.get(
            method.lower(),
            cv2.INTER_LANCZOS4
        )


def apply_post_processing(
    image: np.ndarray,
    options: Dict
) -> np.ndarray:
    """
    Convenience function to apply post-processing with options dict
    
    Args:
        image: Input image
        options: Dictionary with processing options:
            - sharpen: bool
            - sharpen_amount: float
            - denoise: bool
            - denoise_strength: int
            - clahe: bool
            - clahe_strength: float (default 2.0)
            - shadow_removal: bool
            - shadow_strength: float (default 0.5)
            - deblur: bool
            - deblur_radius: float
            - upscale_factor: float
            - interpolation: str
            
    Returns:
        Processed image
    """
    processor = ImagePostProcessor()
    return processor.process(
        image,
        sharpen=options.get('sharpen', False),
        sharpen_amount=options.get('sharpen_amount', 1.0),
        denoise=options.get('denoise', False),
        denoise_strength=options.get('denoise_strength', 5),
        clahe=options.get('clahe', False),
        clahe_strength=options.get('clahe_strength', 2.0),
        shadow_removal=options.get('shadow_removal', False),
        shadow_strength=options.get('shadow_strength', 0.5),
        deblur=options.get('deblur', False),
        deblur_radius=options.get('deblur_radius', 1.5),
        upscale_factor=options.get('upscale_factor', 1.0),
        interpolation=options.get('interpolation', 'lanczos')
    )

