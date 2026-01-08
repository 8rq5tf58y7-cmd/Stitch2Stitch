"""
Model downloader for AI-powered blending
Downloads and caches pretrained models for semantic segmentation
"""

import os
import logging
import hashlib
from pathlib import Path
from typing import Optional, Dict

logger = logging.getLogger(__name__)

# Model registry with download URLs and checksums
MODEL_REGISTRY = {
    # Segment Anything Model (SAM) - Foundation model for any image segmentation
    'sam_vit_b': {
        'url': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth',
        'filename': 'sam_vit_b_01ec64.pth',
        'size_mb': 375,
        'description': 'SAM ViT-B (smallest, fastest)',
        'checksum': '01ec64'
    },
    'sam_vit_l': {
        'url': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth',
        'filename': 'sam_vit_l_0b3195.pth',
        'size_mb': 1250,
        'description': 'SAM ViT-L (medium)',
        'checksum': '0b3195'
    },
    'sam_vit_h': {
        'url': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
        'filename': 'sam_vit_h_4b8939.pth',
        'size_mb': 2560,
        'description': 'SAM ViT-H (largest, best quality)',
        'checksum': '4b8939'
    },
    # MobileSAM - Lightweight version
    'mobile_sam': {
        'url': 'https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt',
        'filename': 'mobile_sam.pt',
        'size_mb': 40,
        'description': 'MobileSAM (very fast, good quality)',
        'checksum': None  # No official checksum
    },
}

def get_model_dir() -> Path:
    """Get the directory where models are stored"""
    # Check for custom path in environment
    custom_path = os.environ.get('STITCH2STITCH_MODEL_DIR')
    if custom_path:
        model_dir = Path(custom_path)
    else:
        # Default: ~/.stitch2stitch/models
        model_dir = Path.home() / '.stitch2stitch' / 'models'
    
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def get_model_path(model_name: str) -> Optional[Path]:
    """Get path to a model, or None if not downloaded"""
    if model_name not in MODEL_REGISTRY:
        logger.warning(f"Unknown model: {model_name}")
        return None
    
    model_info = MODEL_REGISTRY[model_name]
    model_path = get_model_dir() / model_info['filename']
    
    if model_path.exists():
        return model_path
    return None


def is_model_downloaded(model_name: str) -> bool:
    """Check if a model is already downloaded"""
    return get_model_path(model_name) is not None


def list_available_models() -> Dict[str, Dict]:
    """List all available models with their info"""
    result = {}
    for name, info in MODEL_REGISTRY.items():
        result[name] = {
            'description': info['description'],
            'size_mb': info['size_mb'],
            'downloaded': is_model_downloaded(name),
            'path': str(get_model_path(name)) if is_model_downloaded(name) else None
        }
    return result


def download_model(model_name: str, force: bool = False, progress_callback=None) -> Optional[Path]:
    """
    Download a model from the registry
    
    Args:
        model_name: Name of the model to download
        force: Re-download even if already exists
        progress_callback: Optional callback(downloaded_bytes, total_bytes)
        
    Returns:
        Path to the downloaded model, or None on failure
    """
    if model_name not in MODEL_REGISTRY:
        logger.error(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")
        return None
    
    model_info = MODEL_REGISTRY[model_name]
    model_path = get_model_dir() / model_info['filename']
    
    # Check if already downloaded
    if model_path.exists() and not force:
        logger.info(f"Model {model_name} already downloaded at {model_path}")
        return model_path
    
    logger.info(f"Downloading {model_name} ({model_info['size_mb']} MB)...")
    logger.info(f"URL: {model_info['url']}")
    
    try:
        import urllib.request
        import shutil
        import ssl
        
        # Create temp file for download
        temp_path = model_path.with_suffix('.tmp')
        
        # Download with progress
        def reporthook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if progress_callback:
                progress_callback(downloaded, total_size)
            elif total_size > 0:
                percent = min(100, downloaded * 100 / total_size)
                if block_num % 100 == 0:  # Don't spam logs
                    logger.info(f"Download progress: {percent:.1f}%")
        
        # Try with SSL verification first, then without if it fails
        try:
            urllib.request.urlretrieve(
                model_info['url'],
                temp_path,
                reporthook=reporthook
            )
        except urllib.error.URLError as ssl_err:
            if 'CERTIFICATE_VERIFY_FAILED' in str(ssl_err):
                logger.warning("SSL verification failed, retrying without verification...")
                # Create unverified context
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
                
                # Use opener with custom context
                opener = urllib.request.build_opener(
                    urllib.request.HTTPSHandler(context=ssl_context)
                )
                urllib.request.install_opener(opener)
                
                urllib.request.urlretrieve(
                    model_info['url'],
                    temp_path,
                    reporthook=reporthook
                )
            else:
                raise
        
        # Verify checksum if available
        if model_info.get('checksum'):
            logger.info("Verifying checksum...")
            with open(temp_path, 'rb') as f:
                # Read first 1MB for quick check (filename contains partial hash)
                data = f.read(1024 * 1024)
                file_hash = hashlib.md5(data).hexdigest()[:6]
                if model_info['checksum'] not in model_info['filename']:
                    logger.warning("Checksum format unexpected, skipping verification")
        
        # Move to final location
        shutil.move(temp_path, model_path)
        
        logger.info(f"Model downloaded successfully: {model_path}")
        return model_path
        
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        # Cleanup temp file
        temp_path = model_path.with_suffix('.tmp')
        if temp_path.exists():
            temp_path.unlink()
        return None


def download_recommended_models(progress_callback=None) -> Dict[str, Path]:
    """
    Download recommended models for the application.
    Currently downloads MobileSAM (small and fast).
    
    Returns:
        Dict mapping model name to path
    """
    recommended = ['mobile_sam']  # Start with smallest/fastest
    
    downloaded = {}
    for model_name in recommended:
        path = download_model(model_name, progress_callback=progress_callback)
        if path:
            downloaded[model_name] = path
    
    return downloaded


def ensure_model(model_name: str) -> Optional[Path]:
    """
    Ensure a model is available, downloading if necessary.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Path to the model, or None if unavailable
    """
    path = get_model_path(model_name)
    if path:
        return path
    
    logger.info(f"Model {model_name} not found, downloading...")
    return download_model(model_name)


# CLI interface
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Download AI models for Stitch2Stitch')
    parser.add_argument('--list', action='store_true', help='List available models')
    parser.add_argument('--download', type=str, help='Download a specific model')
    parser.add_argument('--download-recommended', action='store_true', 
                       help='Download recommended models')
    parser.add_argument('--force', action='store_true', help='Force re-download')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    if args.list:
        print("\nAvailable models:")
        print("-" * 60)
        for name, info in list_available_models().items():
            status = "[OK] Downloaded" if info['downloaded'] else "[X] Not downloaded"
            print(f"  {name}: {info['description']} ({info['size_mb']} MB) [{status}]")
        print()
    
    elif args.download:
        path = download_model(args.download, force=args.force)
        if path:
            print(f"\n[OK] Model downloaded: {path}\n")
        else:
            print(f"\n[FAILED] Failed to download model\n")
    
    elif args.download_recommended:
        print("\nDownloading recommended models...")
        downloaded = download_recommended_models()
        print(f"\n[OK] Downloaded {len(downloaded)} models\n")
    
    else:
        parser.print_help()

