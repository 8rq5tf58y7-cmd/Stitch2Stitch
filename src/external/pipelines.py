"""
External Pipeline Integrations for Large-Scale Stitching

Provides integration with battle-tested external tools:
- COLMAP: Industry-standard SfM with robust matching
- HLOC: SuperPoint + SuperGlue + NetVLAD
- AliceVision/Meshroom: Open-source photogrammetry

These tools are designed for large image sets (500+) and will
outperform custom pipelines in most cases.

Supports detection of tools installed via:
- System PATH
- Conda environments
- pip/PyPI
- Manual installation
"""

import os
import sys
import subprocess
import shutil
import json
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Callable
import logging
import numpy as np
import time

logger = logging.getLogger(__name__)

# #region agent log
DEBUG_LOG_PATH = Path(__file__).parent.parent.parent / ".cursor" / "debug.log"
def _debug_log(location: str, message: str, data: dict = None, hypothesis_id: str = None):
    try:
        # Ensure parent directory exists
        DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        log_entry = {
            "sessionId": "colmap-detection",
            "runId": "run1",
            "hypothesisId": hypothesis_id or "A",
            "location": location,
            "message": message,
            "data": data or {},
            "timestamp": int(time.time() * 1000)
        }
        with open(DEBUG_LOG_PATH, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + '\n')
    except Exception:
        pass
# #endregion


def _find_conda_executable(name: str) -> Optional[str]:
    """
    Find an executable in conda environment.
    
    Checks:
    1. Current conda environment's bin/Scripts directory
    2. Base conda installation
    3. Common conda paths
    """
    # #region agent log
    _debug_log("pipelines.py:_find_conda_executable", "ENTRY", {"name": name, "os_name": os.name}, "A")
    # #endregion
    
    paths_to_check = []
    
    # Current conda environment
    conda_prefix = os.environ.get('CONDA_PREFIX')
    # #region agent log
    _debug_log("pipelines.py:_find_conda_executable", "CONDA_PREFIX check", {
        "CONDA_PREFIX": conda_prefix,
        "CONDA_PREFIX_1": os.environ.get('CONDA_PREFIX_1'),
        "CONDA_EXE": os.environ.get('CONDA_EXE'),
        "has_prefix": conda_prefix is not None
    }, "A")
    # #endregion
    
    if conda_prefix:
        if os.name == 'nt':  # Windows
            paths_to_check.extend([
                Path(conda_prefix) / 'Scripts' / f'{name}.exe',
                Path(conda_prefix) / 'Library' / 'bin' / f'{name}.exe',
                Path(conda_prefix) / f'{name}.exe',
            ])
        else:  # Linux/Mac
            paths_to_check.append(Path(conda_prefix) / 'bin' / name)
    
    # Check conda base
    conda_base = os.environ.get('CONDA_PREFIX_1') or os.environ.get('CONDA_EXE')
    if conda_base:
        base_path = Path(conda_base).parent.parent if 'conda' in str(conda_base).lower() else Path(conda_base)
        if os.name == 'nt':
            paths_to_check.extend([
                base_path / 'Scripts' / f'{name}.exe',
                base_path / 'Library' / 'bin' / f'{name}.exe',
            ])
        else:
            paths_to_check.append(base_path / 'bin' / name)
    
    # Common conda locations
    if os.name == 'nt':
        user_home = Path.home()
        paths_to_check.extend([
            user_home / 'anaconda3' / 'Scripts' / f'{name}.exe',
            user_home / 'miniconda3' / 'Scripts' / f'{name}.exe',
            user_home / 'Anaconda3' / 'Scripts' / f'{name}.exe',
            user_home / 'Miniconda3' / 'Scripts' / f'{name}.exe',
            Path('C:/ProgramData/anaconda3/Scripts') / f'{name}.exe',
            Path('C:/ProgramData/miniconda3/Scripts') / f'{name}.exe',
        ])
    else:
        user_home = Path.home()
        paths_to_check.extend([
            user_home / 'anaconda3' / 'bin' / name,
            user_home / 'miniconda3' / 'bin' / name,
            Path('/opt/conda/bin') / name,
        ])
    
    # #region agent log
    _debug_log("pipelines.py:_find_conda_executable", "Paths to check", {
        "total_paths": len(paths_to_check),
        "paths": [str(p) for p in paths_to_check[:10]]  # First 10 for brevity
    }, "B")
    # #endregion
    
    # Check each path
    for path in paths_to_check:
        exists = path.exists()
        # #region agent log
        _debug_log("pipelines.py:_find_conda_executable", "Path check", {
            "path": str(path),
            "exists": exists
        }, "B")
        # #endregion
        if exists:
            logger.info(f"Found {name} at: {path}")
            # #region agent log
            _debug_log("pipelines.py:_find_conda_executable", "FOUND", {"path": str(path)}, "B")
            # #endregion
            return str(path)
    
    # #region agent log
    _debug_log("pipelines.py:_find_conda_executable", "NOT FOUND", {"checked_paths": len(paths_to_check)}, "B")
    # #endregion
    return None


def _check_python_module(module_name: str) -> Tuple[bool, Optional[str]]:
    """
    Check if a Python module is installed and get its location.
    
    Returns:
        Tuple of (is_available, module_path_or_version)
    """
    try:
        import importlib
        module = importlib.import_module(module_name)
        version = getattr(module, '__version__', 'unknown')
        location = getattr(module, '__file__', 'unknown')
        return True, f"v{version} at {location}"
    except ImportError:
        return False, None


class ExternalPipelineBase:
    """Base class for external pipeline integrations."""
    
    def __init__(self, progress_callback: Optional[Callable] = None):
        self.progress_callback = progress_callback
        self.temp_dir = None
    
    def _update_progress(self, percent: int, message: str):
        if self.progress_callback:
            self.progress_callback(percent, message)
    
    def is_available(self) -> bool:
        """Check if the external tool is installed and available."""
        raise NotImplementedError
    
    def get_install_instructions(self) -> str:
        """Get installation instructions for the tool."""
        raise NotImplementedError
    
    def run(self, image_paths: List[Path], output_dir: Path) -> Optional[np.ndarray]:
        """Run the pipeline and return the result."""
        raise NotImplementedError


class COLMAPPipeline(ExternalPipelineBase):
    """
    COLMAP Integration - Industry Standard SfM/MVS
    
    COLMAP provides:
    - Vocabulary tree-based image retrieval
    - Robust feature matching with geometric verification
    - Track building across images
    - Global bundle adjustment
    - Dense reconstruction (optional)
    
    Best for:
    - Large image sets (100-10000+)
    - Repetitive geometry
    - Wide-baseline views
    
    Install: https://colmap.github.io/install.html
    """
    
    def __init__(self, 
                 colmap_path: str = None,
                 use_gpu: bool = True,
                 matcher_type: str = "exhaustive",  # or "vocab_tree", "sequential"
                 progress_callback: Optional[Callable] = None):
        super().__init__(progress_callback)
        self.use_gpu = use_gpu
        self.matcher_type = matcher_type
        
        # Auto-detect COLMAP path
        if colmap_path:
            self.colmap_path = colmap_path
        else:
            self.colmap_path = self._find_colmap()
        
        self._version = None
        self._location = None
        self._diagnostic_error = None  # Store diagnostic error info
    
    def _find_colmap(self) -> str:
        """Find COLMAP executable."""
        # #region agent log
        _debug_log("pipelines.py:_find_colmap", "ENTRY", {}, "C")
        # #endregion
        
        # 1. Check if 'colmap' is in PATH
        colmap_in_path = shutil.which('colmap')
        # #region agent log
        _debug_log("pipelines.py:_find_colmap", "PATH check", {"found": colmap_in_path}, "C")
        # #endregion
        if colmap_in_path:
            # #region agent log
            _debug_log("pipelines.py:_find_colmap", "RETURN PATH", {"path": colmap_in_path}, "C")
            # #endregion
            return colmap_in_path
        
        # 2. Check conda environment
        conda_colmap = _find_conda_executable('colmap')
        # #region agent log
        _debug_log("pipelines.py:_find_colmap", "CONDA check result", {"found": conda_colmap}, "C")
        # #endregion
        if conda_colmap:
            # #region agent log
            _debug_log("pipelines.py:_find_colmap", "RETURN CONDA", {"path": conda_colmap}, "C")
            # #endregion
            return conda_colmap
        
        # 3. Check common Windows install locations
        if os.name == 'nt':
            common_paths = [
                Path.home() / 'COLMAP' / 'COLMAP.bat',
                Path('C:/COLMAP/COLMAP.bat'),
                Path('C:/Program Files/COLMAP/COLMAP.bat'),
                Path.home() / 'colmap' / 'colmap.exe',
            ]
            for path in common_paths:
                exists = path.exists()
                # #region agent log
                _debug_log("pipelines.py:_find_colmap", "Common path check", {"path": str(path), "exists": exists}, "C")
                # #endregion
                if exists:
                    # #region agent log
                    _debug_log("pipelines.py:_find_colmap", "RETURN COMMON", {"path": str(path)}, "C")
                    # #endregion
                    return str(path)
        
        # Default to 'colmap' and let is_available() handle the check
        # #region agent log
        _debug_log("pipelines.py:_find_colmap", "RETURN DEFAULT", {"path": "colmap"}, "C")
        # #endregion
        return 'colmap'
    
    def is_available(self) -> bool:
        """Check if COLMAP is installed."""
        # #region agent log
        _debug_log("pipelines.py:is_available", "ENTRY", {"colmap_path": self.colmap_path}, "D")
        # #endregion
        
        try:
            cmd = [self.colmap_path, "help"]
            # #region agent log
            _debug_log("pipelines.py:is_available", "BEFORE subprocess.run", {"cmd": cmd}, "D")
            # #endregion
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            # #region agent log
            _debug_log("pipelines.py:is_available", "AFTER subprocess.run", {
                "returncode": result.returncode,
                "stdout_length": len(result.stdout),
                "stderr_length": len(result.stderr),
                "stdout_preview": result.stdout[:200] if result.stdout else "",
                "stderr_preview": result.stderr[:200] if result.stderr else ""
            }, "D")
            # #endregion
            
            if result.returncode == 0:
                self._location = self.colmap_path
                self._diagnostic_error = None
                # Try to extract version
                if 'COLMAP' in result.stdout:
                    lines = result.stdout.split('\n')
                    for line in lines[:5]:
                        if 'version' in line.lower() or 'colmap' in line.lower():
                            self._version = line.strip()
                            break
                # #region agent log
                _debug_log("pipelines.py:is_available", "RETURN TRUE", {
                    "location": self._location,
                    "version": self._version
                }, "D")
                # #endregion
                return True
            
            # Check for Windows DLL error codes
            # 0xC0000135 = STATUS_DLL_NOT_FOUND
            # 0xC0000139 = STATUS_ENTRYPOINT_NOT_FOUND
            if result.returncode in (0xC0000135, 0xC0000139, -1073741515, -1073741511):
                self._diagnostic_error = "DLL_NOT_FOUND"
                # #region agent log
                _debug_log("pipelines.py:is_available", "RETURN FALSE (DLL_NOT_FOUND)", {
                    "returncode": result.returncode,
                    "hex": hex(result.returncode),
                    "colmap_path": self.colmap_path
                }, "E")
                # #endregion
            else:
                # #region agent log
                _debug_log("pipelines.py:is_available", "RETURN FALSE (non-zero)", {"returncode": result.returncode}, "D")
                # #endregion
            
            return False
        except FileNotFoundError as e:
            # #region agent log
            _debug_log("pipelines.py:is_available", "EXCEPTION FileNotFoundError", {
                "error": str(e),
                "colmap_path": self.colmap_path
            }, "E")
            # #endregion
            return False
        except subprocess.SubprocessError as e:
            # #region agent log
            _debug_log("pipelines.py:is_available", "EXCEPTION SubprocessError", {
                "error": str(e),
                "error_type": type(e).__name__
            }, "E")
            # #endregion
            return False
        except OSError as e:
            # #region agent log
            _debug_log("pipelines.py:is_available", "EXCEPTION OSError", {
                "error": str(e),
                "error_code": getattr(e, 'winerror', None)
            }, "E")
            # #endregion
            return False
        except Exception as e:
            # #region agent log
            _debug_log("pipelines.py:is_available", "EXCEPTION Unexpected", {
                "error": str(e),
                "error_type": type(e).__name__
            }, "E")
            # #endregion
            return False
    
    def get_version_info(self) -> str:
        """Get version and location info."""
        # Check availability first (this sets _diagnostic_error)
        is_avail = self.is_available()
        
        if is_avail:
            return f"{self._version or 'COLMAP'} at {self._location}"
        
        # Provide diagnostic info if executable found but can't run
        if self._diagnostic_error == "DLL_NOT_FOUND" and Path(self.colmap_path).exists():
            return f"Found at {self.colmap_path} but missing DLL dependencies. Try: conda install -c conda-forge colmap --force-reinstall"
        
        return "Not installed"
    
    def get_install_instructions(self) -> str:
        return """
COLMAP Installation:

Windows (Recommended - Pre-built binaries):
  1. Download from: https://github.com/colmap/colmap/releases
  2. Get the CUDA version (e.g., COLMAP-3.9-windows-cuda.zip)
  3. Extract to C:\\COLMAP
  4. Add C:\\COLMAP\\bin to your PATH environment variable
  
  OR via conda:
  conda install -c conda-forge colmap

Linux:
  sudo apt install colmap
  
  OR via conda:
  conda install -c conda-forge colmap

macOS:
  brew install colmap

After installation, restart the application.
"""
    
    def run(self, image_paths: List[Path], output_dir: Path) -> Dict:
        """
        Run COLMAP reconstruction pipeline.
        
        Returns:
            Dict with reconstruction results including camera poses
        """
        if not self.is_available():
            raise RuntimeError("COLMAP is not installed. " + self.get_install_instructions())
        
        # Create workspace
        workspace = output_dir / "colmap_workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        
        images_dir = workspace / "images"
        database_path = workspace / "database.db"
        sparse_dir = workspace / "sparse"
        
        images_dir.mkdir(exist_ok=True)
        sparse_dir.mkdir(exist_ok=True)
        
        # Copy/link images to workspace
        self._update_progress(5, "Preparing images for COLMAP...")
        for i, path in enumerate(image_paths):
            dest = images_dir / f"{i:06d}{path.suffix}"
            shutil.copy2(path, dest)
        
        # Feature extraction
        self._update_progress(10, "COLMAP: Extracting features...")
        self._run_colmap_command([
            "feature_extractor",
            "--database_path", str(database_path),
            "--image_path", str(images_dir),
            "--ImageReader.single_camera", "1",
            "--SiftExtraction.use_gpu", "1" if self.use_gpu else "0",
        ])
        
        # Feature matching
        self._update_progress(30, f"COLMAP: Matching features ({self.matcher_type})...")
        if self.matcher_type == "vocab_tree":
            # Requires vocabulary tree file
            self._run_colmap_command([
                "vocab_tree_matcher",
                "--database_path", str(database_path),
                "--SiftMatching.use_gpu", "1" if self.use_gpu else "0",
            ])
        elif self.matcher_type == "sequential":
            self._run_colmap_command([
                "sequential_matcher",
                "--database_path", str(database_path),
                "--SiftMatching.use_gpu", "1" if self.use_gpu else "0",
            ])
        else:  # exhaustive
            self._run_colmap_command([
                "exhaustive_matcher",
                "--database_path", str(database_path),
                "--SiftMatching.use_gpu", "1" if self.use_gpu else "0",
            ])
        
        # Sparse reconstruction (bundle adjustment)
        self._update_progress(60, "COLMAP: Running bundle adjustment...")
        self._run_colmap_command([
            "mapper",
            "--database_path", str(database_path),
            "--image_path", str(images_dir),
            "--output_path", str(sparse_dir),
        ])
        
        # Read reconstruction results
        self._update_progress(90, "COLMAP: Reading results...")
        results = self._read_reconstruction(sparse_dir)
        
        self._update_progress(100, "COLMAP: Complete!")
        return results
    
    def _run_colmap_command(self, args: List[str]):
        """Run a COLMAP command."""
        cmd = [self.colmap_path] + args
        logger.info(f"Running COLMAP: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            logger.error(f"COLMAP error: {result.stderr}")
            raise RuntimeError(f"COLMAP failed: {result.stderr}")
        
        return result
    
    def _read_reconstruction(self, sparse_dir: Path) -> Dict:
        """Read COLMAP reconstruction results."""
        # Find the reconstruction directory (usually 0/)
        recon_dirs = list(sparse_dir.glob("*/"))
        if not recon_dirs:
            return {"success": False, "error": "No reconstruction found"}
        
        recon_dir = recon_dirs[0]
        
        # Read cameras, images, points3D
        results = {
            "success": True,
            "workspace": str(sparse_dir.parent),
            "reconstruction_dir": str(recon_dir),
            "n_images": 0,
            "n_points3d": 0,
            "cameras": {},
            "images": {},
        }
        
        # Parse images.txt for camera poses
        images_file = recon_dir / "images.txt"
        if images_file.exists():
            with open(images_file, 'r') as f:
                lines = f.readlines()
                # Skip comments
                lines = [l for l in lines if not l.startswith('#')]
                results["n_images"] = len(lines) // 2
        
        # Parse points3D.txt for 3D points
        points_file = recon_dir / "points3D.txt"
        if points_file.exists():
            with open(points_file, 'r') as f:
                lines = f.readlines()
                lines = [l for l in lines if not l.startswith('#')]
                results["n_points3d"] = len(lines)
        
        logger.info(f"COLMAP reconstruction: {results['n_images']} images, {results['n_points3d']} points")
        return results


class HLOCPipeline(ExternalPipelineBase):
    """
    HLOC (Hierarchical Localization) Integration
    
    HLOC provides:
    - NetVLAD for global image retrieval
    - SuperPoint for feature detection
    - SuperGlue for feature matching
    - COLMAP for reconstruction
    
    Best for:
    - Large image sets with repetitive geometry
    - Robotics and mapping applications
    - When SIFT fails due to texture repetition
    
    Install: pip install hloc
    GitHub: https://github.com/cvg/Hierarchical-Localization
    """
    
    def __init__(self, progress_callback: Optional[Callable] = None):
        super().__init__(progress_callback)
        self._version = None
        self._location = None
        self._has_superpoint = False
        self._has_superglue = False
        self._has_netvlad = False
    
    def is_available(self) -> bool:
        """Check if HLOC is installed with required components."""
        try:
            import hloc
            self._version = getattr(hloc, '__version__', 'unknown')
            self._location = getattr(hloc, '__file__', 'unknown')
            
            # Check for key components
            try:
                from hloc import extract_features
                self._has_superpoint = 'superpoint' in str(extract_features.confs).lower()
            except:
                pass
            
            try:
                from hloc import match_features
                self._has_superglue = 'superglue' in str(match_features.confs).lower()
            except:
                pass
            
            try:
                from hloc import extract_features
                self._has_netvlad = 'netvlad' in str(extract_features.confs).lower()
            except:
                pass
            
            return True
        except ImportError:
            return False
    
    def get_version_info(self) -> str:
        """Get version and component info."""
        if self.is_available():
            components = []
            if self._has_superpoint:
                components.append("SuperPoint")
            if self._has_superglue:
                components.append("SuperGlue")
            if self._has_netvlad:
                components.append("NetVLAD")
            
            comp_str = ", ".join(components) if components else "base only"
            return f"HLOC v{self._version} ({comp_str})"
        return "Not installed"
    
    def get_install_instructions(self) -> str:
        return """
HLOC Installation:

1. Install PyTorch (with CUDA for GPU support):
   pip install torch torchvision

2. Install HLOC from GitHub (NOT PyPI):
   pip install git+https://github.com/cvg/Hierarchical-Localization.git

   Note: Requires Git to be installed!
   Download Git from: https://git-scm.com/download/win

3. Install COLMAP (required backend):
   - Windows: Download pre-built binaries from:
     https://github.com/colmap/colmap/releases
   - Linux: sudo apt install colmap
   - macOS: brew install colmap

After installation, restart the application.
"""
    
    def run(self, image_paths: List[Path], output_dir: Path) -> Dict:
        """
        Run HLOC pipeline with SuperPoint + SuperGlue.
        """
        if not self.is_available():
            raise RuntimeError("HLOC is not installed. " + self.get_install_instructions())
        
        from hloc import extract_features, match_features, reconstruction
        from hloc import pairs_from_retrieval
        
        # Create workspace
        workspace = output_dir / "hloc_workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        
        images_dir = workspace / "images"
        outputs_dir = workspace / "outputs"
        
        images_dir.mkdir(exist_ok=True)
        outputs_dir.mkdir(exist_ok=True)
        
        # Copy images
        self._update_progress(5, "Preparing images for HLOC...")
        image_list = []
        for i, path in enumerate(image_paths):
            dest = images_dir / f"{i:06d}{path.suffix}"
            shutil.copy2(path, dest)
            image_list.append(dest.relative_to(images_dir))
        
        # Feature extraction with SuperPoint
        self._update_progress(15, "HLOC: Extracting SuperPoint features...")
        feature_conf = extract_features.confs["superpoint_aachen"]
        feature_path = extract_features.main(
            feature_conf, images_dir, outputs_dir
        )
        
        # Global image retrieval with NetVLAD
        self._update_progress(35, "HLOC: Computing global descriptors (NetVLAD)...")
        retrieval_conf = extract_features.confs["netvlad"]
        retrieval_path = extract_features.main(
            retrieval_conf, images_dir, outputs_dir
        )
        
        # Find matching pairs using retrieval
        self._update_progress(45, "HLOC: Finding candidate pairs...")
        pairs_path = outputs_dir / "pairs.txt"
        pairs_from_retrieval.main(
            retrieval_path, pairs_path, num_matched=50
        )
        
        # Match features with SuperGlue
        self._update_progress(55, "HLOC: Matching with SuperGlue...")
        match_conf = match_features.confs["superglue"]
        match_path = match_features.main(
            match_conf, pairs_path, feature_conf["output"], outputs_dir
        )
        
        # Reconstruction with COLMAP
        self._update_progress(75, "HLOC: Running reconstruction...")
        model = reconstruction.main(
            outputs_dir / "sfm",
            images_dir,
            pairs_path,
            feature_path,
            match_path
        )
        
        self._update_progress(100, "HLOC: Complete!")
        
        return {
            "success": True,
            "workspace": str(workspace),
            "model": model,
            "n_images": len(image_paths),
        }


class AliceVisionPipeline(ExternalPipelineBase):
    """
    AliceVision / Meshroom Integration
    
    AliceVision provides:
    - SIFT feature extraction
    - Vocabulary tree matching
    - Incremental SfM
    - Global SfM (optional)
    - MVS dense reconstruction
    
    Best for:
    - Photogrammetry workflows
    - 3D scanning
    - Large outdoor scenes
    
    Install: https://alicevision.org/
    GUI: Meshroom (https://alicevision.org/#meshroom)
    """
    
    def __init__(self, 
                 alicevision_path: str = None,
                 progress_callback: Optional[Callable] = None):
        super().__init__(progress_callback)
        
        # Try to find AliceVision installation
        if alicevision_path:
            self.bin_path = Path(alicevision_path)
        else:
            self.bin_path = self._find_alicevision()
    
    def _find_alicevision(self) -> Optional[Path]:
        """Find AliceVision installation."""
        possible_paths = [
            Path("/usr/local/bin"),
            Path("/opt/AliceVision/bin"),
            Path(os.environ.get("ALICEVISION_ROOT", "")) / "bin",
            Path(os.environ.get("PROGRAMFILES", "")) / "AliceVision" / "bin",
        ]
        
        for path in possible_paths:
            if (path / "aliceVision_featureExtraction").exists():
                return path
        
        return None
    
    def is_available(self) -> bool:
        """Check if AliceVision is installed."""
        if not self.bin_path:
            return False
        
        try:
            result = subprocess.run(
                [str(self.bin_path / "aliceVision_featureExtraction"), "--help"],
                capture_output=True,
                timeout=10
            )
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
    
    def get_install_instructions(self) -> str:
        return """
AliceVision Installation:

Option 1 - Meshroom (includes AliceVision):
  1. Download from https://alicevision.org/#meshroom
  2. Extract and run Meshroom
  3. AliceVision binaries are in the Meshroom folder

Option 2 - Standalone:
  - Linux: Download from https://github.com/alicevision/AliceVision/releases
  - Windows: Use Meshroom or build from source
  - macOS: Build from source

Set ALICEVISION_ROOT environment variable to the installation directory.
"""
    
    def run(self, image_paths: List[Path], output_dir: Path) -> Dict:
        """Run AliceVision SfM pipeline."""
        if not self.is_available():
            raise RuntimeError("AliceVision is not installed. " + self.get_install_instructions())
        
        # Create workspace
        workspace = output_dir / "alicevision_workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        
        # TODO: Implement AliceVision pipeline
        # This requires running multiple AliceVision commands in sequence:
        # 1. aliceVision_cameraInit
        # 2. aliceVision_featureExtraction
        # 3. aliceVision_imageMatching
        # 4. aliceVision_featureMatching
        # 5. aliceVision_structureFromMotion
        
        self._update_progress(100, "AliceVision pipeline not fully implemented")
        
        return {
            "success": False,
            "message": "AliceVision pipeline integration coming soon. Please use Meshroom GUI directly.",
            "workspace": str(workspace),
        }


def check_available_pipelines(detailed: bool = False) -> Dict[str, any]:
    """
    Check which external pipelines are available.
    
    Args:
        detailed: If True, return detailed info including versions and paths
        
    Returns:
        Dict with pipeline availability (and details if requested)
    """
    # #region agent log
    _debug_log("pipelines.py:check_available_pipelines", "ENTRY", {"detailed": detailed}, "C")
    # #endregion
    
    colmap = COLMAPPipeline()
    # #region agent log
    _debug_log("pipelines.py:check_available_pipelines", "COLMAPPipeline created", {
        "colmap_path": colmap.colmap_path
    }, "C")
    # #endregion
    
    hloc = HLOCPipeline()
    alice = AliceVisionPipeline()
    
    colmap_available = colmap.is_available()
    # #region agent log
    _debug_log("pipelines.py:check_available_pipelines", "COLMAP availability check", {
        "available": colmap_available,
        "colmap_path": colmap.colmap_path,
        "location": getattr(colmap, '_location', None),
        "version": getattr(colmap, '_version', None)
    }, "C")
    # #endregion
    
    hloc_available = hloc.is_available()
    alice_available = alice.is_available()
    
    if detailed:
        return {
            "colmap": {
                "available": colmap_available,
                "info": colmap.get_version_info(),  # Always call to get diagnostic info
                "path": colmap.colmap_path if colmap_available or Path(colmap.colmap_path).exists() else None,
            },
            "hloc": {
                "available": hloc_available,
                "info": hloc.get_version_info() if hloc_available else "Not found",
                "has_superpoint": hloc._has_superpoint,
                "has_superglue": hloc._has_superglue,
                "has_netvlad": hloc._has_netvlad,
            },
            "alicevision": {
                "available": alice_available,
                "info": "AliceVision" if alice_available else "Not found",
                "path": str(alice.bin_path) if alice_available else None,
            },
        }
    else:
        return {
            "colmap": colmap_available,
            "hloc": hloc_available,
            "alicevision": alice_available,
        }


def get_recommended_pipeline(n_images: int, has_gpu: bool = True) -> str:
    """
    Get the recommended pipeline for a given image count.
    
    Args:
        n_images: Number of images to process
        has_gpu: Whether GPU is available
        
    Returns:
        Recommended pipeline name
    """
    available = check_available_pipelines()
    
    if n_images >= 500:
        # For very large sets, prefer HLOC (SuperPoint + SuperGlue)
        if available["hloc"]:
            return "hloc"
        elif available["colmap"]:
            return "colmap"
    elif n_images >= 100:
        # For large sets, COLMAP is reliable
        if available["colmap"]:
            return "colmap"
        elif available["hloc"]:
            return "hloc"
    
    # For smaller sets or if nothing is available
    return "internal"


