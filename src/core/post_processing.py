"""
Post-processing filters for image enhancement
"""

import numpy as np
import cv2
import logging
from typing import Optional, Dict, Tuple, Callable

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

    def apply_white_balance(
        self,
        image: np.ndarray,
        method: str = 'grayworld'
    ) -> np.ndarray:
        """
        Apply automatic white balance correction.

        Uses the Gray World assumption: the average color of a scene
        should be neutral gray. Corrects color cast by scaling channels.

        Args:
            image: Input image (BGR format)
            method: White balance method ('grayworld', 'simplest', 'perfect_reflector')

        Returns:
            White-balanced image
        """
        if len(image.shape) != 3:
            return image

        if method == 'simplest':
            # Simplest Color Balance: stretch each channel independently
            result = np.zeros_like(image)
            for i in range(3):
                channel = image[:, :, i]
                low = np.percentile(channel, 1)
                high = np.percentile(channel, 99)
                if high - low > 0:
                    result[:, :, i] = np.clip((channel - low) * 255.0 / (high - low), 0, 255)
                else:
                    result[:, :, i] = channel
            return result.astype(np.uint8)

        elif method == 'perfect_reflector':
            # Perfect Reflector: assume brightest pixel is white
            result = image.astype(np.float32)
            # Find max values per channel
            max_vals = result.max(axis=(0, 1))
            max_vals = np.maximum(max_vals, 1)  # Avoid division by zero
            # Scale to make brightest pixel white
            scale = 255.0 / max_vals
            result *= scale
            return np.clip(result, 0, 255).astype(np.uint8)

        else:  # grayworld (default)
            # Gray World: assume average color should be gray
            result = image.astype(np.float32)
            # Compute average of each channel
            avg = result.mean(axis=(0, 1))
            avg_gray = avg.mean()  # Target gray level
            # Scale each channel
            scale = avg_gray / np.maximum(avg, 1)
            result *= scale
            return np.clip(result, 0, 255).astype(np.uint8)

    def apply_vignette_removal(
        self,
        image: np.ndarray,
        strength: float = 1.0
    ) -> np.ndarray:
        """
        Remove lens vignetting (dark corners).

        Estimates and corrects the radial brightness falloff common in
        microscope lenses and wide-angle photography.

        Args:
            image: Input image
            strength: Correction strength (0.0-1.0, default 1.0)

        Returns:
            Vignette-corrected image
        """
        if strength <= 0:
            return image

        h, w = image.shape[:2]

        # Create radial distance map from center
        y, x = np.ogrid[:h, :w]
        cx, cy = w / 2, h / 2

        # Normalize distance to [0, 1] where 1 is the corner
        max_dist = np.sqrt(cx**2 + cy**2)
        dist = np.sqrt((x - cx)**2 + (y - cy)**2) / max_dist

        # Estimate vignette by analyzing image brightness vs distance
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        gray_float = gray.astype(np.float32)

        # Sample brightness at different radial distances
        # This estimates the vignette profile from the actual image
        n_rings = 20
        ring_radii = np.linspace(0, 1, n_rings)
        ring_brightness = []

        for r in ring_radii:
            # Create ring mask
            r_inner = max(0, r - 0.05)
            r_outer = r + 0.05
            mask = ((dist >= r_inner) & (dist <= r_outer))
            if mask.sum() > 0:
                ring_brightness.append(np.mean(gray_float[mask]))
            else:
                ring_brightness.append(ring_brightness[-1] if ring_brightness else 128)

        ring_brightness = np.array(ring_brightness)

        # Fit a simple vignette model: brightness = a * (1 - b * r^2)
        # Find correction factor at each distance
        center_brightness = ring_brightness[0]
        if center_brightness < 1:
            center_brightness = 128

        correction_profile = center_brightness / np.maximum(ring_brightness, 1)

        # Interpolate correction for all pixels
        correction = np.interp(dist.flatten(), ring_radii, correction_profile)
        correction = correction.reshape(h, w)

        # Blend with strength parameter
        correction = 1.0 + (correction - 1.0) * strength

        # Apply correction
        if len(image.shape) == 3:
            result = image.astype(np.float32)
            for i in range(3):
                result[:, :, i] *= correction
            return np.clip(result, 0, 255).astype(np.uint8)
        else:
            result = gray_float * correction
            return np.clip(result, 0, 255).astype(np.uint8)

    def apply_super_resolution(
        self,
        image: np.ndarray,
        scale: int = 2,
        model_name: str = 'realesrgan-x4plus',
        tile_size: int = 512,
        progress_callback: Optional[Callable] = None
    ) -> np.ndarray:
        """
        Apply AI super-resolution using Real-ESRGAN.

        Uses Real-ESRGAN models for high-quality image upscaling that
        preserves details and reduces artifacts better than traditional
        interpolation methods.

        Args:
            image: Input image (BGR format)
            scale: Upscale factor (2 or 4)
            model_name: Model to use:
                - 'realesrgan-x4plus': General purpose (best quality)
                - 'realesrgan-x4plus-anime': For anime/illustrations
                - 'realesrnet-x4plus': Faster, slightly lower quality
            tile_size: Process in tiles to reduce memory (0 = no tiling)
            progress_callback: Optional callback(percent, message)

        Returns:
            Super-resolved image
        """
        try:
            from realesrgan import RealESRGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet
            import torch
        except ImportError:
            logger.warning("Real-ESRGAN not installed, attempting to install...")
            if progress_callback:
                progress_callback(5, "Installing Real-ESRGAN (this may take a few minutes)...")
            try:
                import subprocess
                import sys

                # Try different install strategies for different environments
                install_commands = [
                    [sys.executable, "-m", "pip", "install", "realesrgan"],
                    [sys.executable, "-m", "pip", "install", "--user", "realesrgan"],
                    [sys.executable, "-m", "pip", "install", "--break-system-packages", "realesrgan"],
                ]

                installed = False
                last_error = ""

                for cmd in install_commands:
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=600  # 10 minute timeout for large package
                    )
                    if result.returncode == 0:
                        installed = True
                        break
                    last_error = result.stderr

                if installed:
                    logger.info("Successfully installed realesrgan")
                    from realesrgan import RealESRGANer
                    from basicsr.archs.rrdbnet_arch import RRDBNet
                    import torch
                else:
                    logger.error(f"Failed to install realesrgan: {last_error}")
                    logger.info("Falling back to Lanczos upscaling")
                    return self.apply_upscale(image, float(scale), 'lanczos')
            except Exception as e:
                logger.error(f"Could not install realesrgan: {e}")
                logger.info("Falling back to Lanczos upscaling")
                return self.apply_upscale(image, float(scale), 'lanczos')

        if progress_callback:
            progress_callback(5, "Loading Real-ESRGAN model...")

        # Determine model parameters based on model_name
        if 'anime' in model_name.lower():
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
            netscale = 4
            model_url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth'
        elif 'realesrnet' in model_name.lower():
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            netscale = 4
            model_url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth'
        else:  # realesrgan-x4plus (default)
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            netscale = 4
            model_url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'

        # Get model path (downloads if needed)
        import os
        from pathlib import Path
        import urllib.request

        # Store models in user's cache directory
        cache_dir = Path.home() / '.cache' / 'stitch2stitch' / 'models'
        cache_dir.mkdir(parents=True, exist_ok=True)

        model_filename = model_url.split('/')[-1]
        model_path = cache_dir / model_filename

        if not model_path.exists():
            if progress_callback:
                progress_callback(10, f"Downloading {model_filename}...")
            logger.info(f"Downloading Real-ESRGAN model to {model_path}")
            try:
                urllib.request.urlretrieve(model_url, str(model_path))
            except Exception as e:
                logger.error(f"Failed to download model: {e}")
                logger.info("Falling back to Lanczos upscaling")
                return self.apply_upscale(image, float(scale), 'lanczos')

        if progress_callback:
            progress_callback(20, "Initializing Real-ESRGAN...")

        # Determine device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        half_precision = device == 'cuda'  # Use FP16 on GPU for speed

        try:
            upsampler = RealESRGANer(
                scale=netscale,
                model_path=str(model_path),
                model=model,
                tile=tile_size,
                tile_pad=10,
                pre_pad=0,
                half=half_precision,
                device=device
            )
        except Exception as e:
            logger.error(f"Failed to initialize Real-ESRGAN: {e}")
            return self.apply_upscale(image, float(scale), 'lanczos')

        if progress_callback:
            progress_callback(30, f"Running Real-ESRGAN on {device.upper()}...")

        # Convert BGR to RGB for Real-ESRGAN
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        try:
            # Run super resolution
            output, _ = upsampler.enhance(image_rgb, outscale=scale)

            if progress_callback:
                progress_callback(90, "Converting result...")

            # Convert back to BGR
            result = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

            if progress_callback:
                progress_callback(100, "Super-resolution complete")

            logger.info(f"Real-ESRGAN complete: {image.shape[:2]} -> {result.shape[:2]}")
            return result

        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                logger.warning(f"GPU out of memory, trying smaller tile size...")
                # Try with smaller tiles
                if tile_size > 128:
                    return self.apply_super_resolution(
                        image, scale, model_name,
                        tile_size=tile_size // 2,
                        progress_callback=progress_callback
                    )
                else:
                    logger.error("Cannot process even with smallest tiles")
                    return self.apply_upscale(image, float(scale), 'lanczos')
            else:
                logger.error(f"Real-ESRGAN error: {e}")
                return self.apply_upscale(image, float(scale), 'lanczos')
        except Exception as e:
            logger.error(f"Real-ESRGAN error: {e}")
            return self.apply_upscale(image, float(scale), 'lanczos')

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


# Convenience wrapper for external callers (e.g., WSL bridge) to run Real-ESRGAN.
def upscale_with_realesrgan(
    image: np.ndarray,
    scale: int = 2,
    model_name: str = "realesrgan-x4plus",
    tile_size: int = 512,
    progress_callback: Optional[Callable] = None,
) -> np.ndarray:
    """
    Upscale an image using Real-ESRGAN. Falls back to Lanczos on failure.

    Args:
        image: Input BGR image.
        scale: Target scale factor (2 or 4).
        model_name: Real-ESRGAN model name.
        tile_size: Tile size for tiled inference (reduces GPU memory).
        progress_callback: Optional callback(percent, message).
    """
    processor = ImagePostProcessor()
    try:
        scale_int = int(scale) if scale else 2
        scale_int = 4 if scale_int >= 4 else 2
    except Exception:
        scale_int = 2

    return processor.apply_super_resolution(
        image,
        scale=scale_int,
        model_name=model_name,
        tile_size=tile_size,
        progress_callback=progress_callback,
    )

# Convenience wrapper for external callers (e.g., WSL bridge) to run Real-ESRGAN.
def upscale_with_realesrgan(
    image: np.ndarray,
    scale: int = 2,
    model_name: str = "realesrgan-x4plus",
    tile_size: int = 512,
    progress_callback: Optional[Callable] = None,
) -> np.ndarray:
    """
    Upscale an image using Real-ESRGAN. Falls back to Lanczos on failure.

    Args:
        image: Input BGR image.
        scale: Target scale factor (2 or 4).
        model_name: Real-ESRGAN model name.
        tile_size: Tile size for tiled inference (reduces GPU memory).
        progress_callback: Optional callback(percent, message).
    """
    processor = ImagePostProcessor()
    try:
        scale_int = int(scale) if scale else 2
        scale_int = 4 if scale_int >= 4 else 2
    except Exception:
        scale_int = 2

    return processor.apply_super_resolution(
        image,
        scale=scale_int,
        model_name=model_name,
        tile_size=tile_size,
        progress_callback=progress_callback,
    )
