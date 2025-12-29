"""
External Pipeline Integrations for Large-Scale Stitching

Provides integration with battle-tested external tools:
- PyCOLMAP: Python bindings for COLMAP - Industry-standard SfM with robust matching
- HLOC: SuperPoint + SuperGlue + NetVLAD
- AliceVision/Meshroom: Open-source photogrammetry

These tools are designed for large image sets (500+) and will
outperform custom pipelines in most cases.

Supports detection of tools installed via:
- pip/PyPI (pycolmap)
- System PATH (for other tools)
- Conda environments
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

logger = logging.getLogger(__name__)

# Import ImageBlender for advanced blending
try:
    from core.blender import ImageBlender
except ImportError:
    # If import fails (e.g., in WSL bridge), will be imported locally
    ImageBlender = None


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
    PyCOLMAP Integration - Industry Standard SfM/MVS
    
    PyCOLMAP provides:
    - Python bindings for COLMAP
    - Vocabulary tree-based image retrieval
    - Robust feature matching with geometric verification
    - Track building across images
    - Global bundle adjustment
    - Dense reconstruction (optional)
    
    Best for:
    - Large image sets (100-10000+)
    - Repetitive geometry
    - Wide-baseline views
    
    Install: pip install pycolmap
    
    GPU Support:
    - Windows: Uses WSL with pycolmap-cuda12 for GPU acceleration
    - Linux: Native pycolmap-cuda12
    """
    
    def __init__(self,
                 use_gpu: bool = True,
                 use_affine: bool = False,  # Use affine instead of homography
                 blend_method: str = "multiband",  # Blending method: multiband, feather, autostitch, linear
                 matcher_type: str = "exhaustive",  # or "vocab_tree", "sequential"
                 progress_callback: Optional[Callable] = None):
        super().__init__(progress_callback)
        self.matcher_type = matcher_type
        self.use_affine = use_affine
        self.blend_method = blend_method
        self._version = None
        self._location = None
        self._has_cuda = False
        self._has_wsl_cuda = False
        
        # Check if GPU is available and requested
        if use_gpu and self.is_available():
            import pycolmap
            try:
                # has_cuda is a boolean property, not a method in pycolmap 3.13+
                self._has_cuda = pycolmap.has_cuda if hasattr(pycolmap, 'has_cuda') else False
            except:
                self._has_cuda = False
        
        # On Windows, check for WSL CUDA if native CUDA not available
        if use_gpu and not self._has_cuda and sys.platform == 'win32':
            self._has_wsl_cuda = self._check_wsl_cuda()
            if self._has_wsl_cuda:
                logger.info("WSL pycolmap-cuda12 detected - GPU acceleration available via WSL")
        
        # Use GPU if either native CUDA or WSL CUDA is available
        self.use_gpu = use_gpu and (self._has_cuda or self._has_wsl_cuda)
        
        if use_gpu and not self.use_gpu:
            logger.info("GPU requested but CUDA not available (native or WSL). Using CPU mode.")
    
    def _check_wsl_cuda(self) -> bool:
        """Check if WSL has pycolmap with CUDA support."""
        try:
            result = subprocess.run(
                ['wsl', '-e', 'python3', '-c', 
                 'import pycolmap; print(pycolmap.has_cuda)'],
                capture_output=True,
                text=True,
                timeout=30
            )
            return result.returncode == 0 and 'True' in result.stdout
        except Exception as e:
            logger.debug(f"WSL CUDA check failed: {e}")
            return False
    
    def is_available(self) -> bool:
        """Check if pycolmap is installed."""
        try:
            import pycolmap
            self._version = getattr(pycolmap, '__version__', 'unknown')
            self._location = getattr(pycolmap, '__file__', 'unknown')
            return True
        except ImportError:
            return False
    
    def has_gpu_support(self) -> bool:
        """Check if GPU/CUDA support is available (native or WSL)."""
        return self._has_cuda or self._has_wsl_cuda
    
    def get_version_info(self) -> str:
        """Get version and location info."""
        if self.is_available():
            if self._has_cuda:
                gpu_status = "GPU (native)"
            elif self._has_wsl_cuda:
                gpu_status = "GPU (WSL)"
            else:
                gpu_status = "CPU-only"
            return f"PyCOLMAP v{self._version} ({gpu_status}) at {self._location}"
        return "Not installed"
    
    def _process_wsl_line(self, line: str):
        """Process a single line from WSL stderr and update progress."""
        if '[PROGRESS]' in line:
            msg = line.replace('[PROGRESS]', '').strip()
            
            # Determine progress percentage based on stage
            if 'Copying' in msg and '/' in msg:
                try:
                    for p in msg.split():
                        if '/' in p:
                            current, total = p.split('/')
                            pct = 5 + int(5 * int(current) / int(total))
                            self._update_progress(pct, f"WSL: {msg}")
                            return
                except:
                    pass
                self._update_progress(5, f"WSL: {msg}")
            elif 'Extracting features' in msg:
                self._update_progress(15, f"WSL GPU: {msg}")
            elif 'Feature extraction complete' in msg:
                self._update_progress(30, f"WSL GPU: {msg}")
            elif 'Matching features' in msg:
                self._update_progress(35, f"WSL GPU: {msg}")
            elif 'Feature matching complete' in msg:
                self._update_progress(55, f"WSL GPU: {msg}")
            elif 'Reading matches' in msg:
                self._update_progress(60, f"WSL: {msg}")
            elif 'Found' in msg and 'pairs' in msg:
                self._update_progress(62, f"WSL: {msg}")
            elif 'Loading' in msg and 'images' in msg:
                self._update_progress(65, f"WSL: {msg}")
            elif 'Loaded' in msg and '/' in msg:
                try:
                    for p in msg.split():
                        if '/' in p:
                            current, total = p.split('/')
                            pct = 65 + int(5 * int(current) / int(total))
                            self._update_progress(pct, f"WSL: {msg}")
                            return
                except:
                    pass
                self._update_progress(68, f"WSL: {msg}")
            elif 'Computing homographies' in msg:
                self._update_progress(72, f"WSL: {msg}")
            elif 'Computed' in msg and 'homographies' in msg:
                self._update_progress(75, f"WSL: {msg}")
            elif 'Stitching panorama' in msg:
                self._update_progress(78, f"WSL: {msg}")
            elif 'Stitching complete' in msg:
                self._update_progress(88, f"WSL: {msg}")
            elif 'Panorama size' in msg:
                self._update_progress(90, f"WSL: {msg}")
            elif 'Saving panorama' in msg:
                self._update_progress(92, f"WSL: {msg}")
            elif 'Complete!' in msg:
                self._update_progress(98, f"WSL: {msg}")
            else:
                logger.info(f"WSL: {msg}")
        elif 'SIFT GPU' in line or 'Creating SIFT' in line:
            logger.info(f"WSL COLMAP: {line}")
        else:
            logger.debug(f"WSL: {line}")
    
    def _run_via_wsl(self, image_paths: List[Path], output_dir: Path) -> Dict:
        """Run COLMAP processing via WSL for GPU acceleration."""
        import cv2
        import time
        
        # #region agent log
        debug_log_path = r"c:\Users\ryanf\OneDrive - University of Maryland\Desktop\Stitch2Stitch\Stitch2Stitch-1\.cursor\debug.log"
        # #endregion
        
        self._update_progress(5, "COLMAP (WSL GPU): Preparing...")
        
        # Get the bridge script path
        bridge_script = Path(__file__).parent / "wsl_colmap_bridge.py"
        
        if not bridge_script.exists():
            raise RuntimeError(f"WSL bridge script not found at {bridge_script}")
        
        output_dir_str = str(output_dir)
        
        # Convert bridge script path to WSL path
        bridge_wsl = str(bridge_script).replace('\\', '/')
        if len(bridge_wsl) >= 2 and bridge_wsl[1] == ':':
            drive = bridge_wsl[0].lower()
            bridge_wsl = f'/mnt/{drive}{bridge_wsl[2:]}'
        
        self._update_progress(10, "COLMAP (WSL GPU): Running feature extraction...")
        
        # Write config file with all parameters (avoids shell escaping issues)
        import tempfile
        config = {
            "image_paths": [str(p) for p in image_paths],
            "output_dir": output_dir_str,
            "use_affine": self.use_affine,
            "blend_method": self.blend_method
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f)
            config_path = f.name
        
        # Convert config path to WSL
        config_wsl = config_path.replace('\\', '/')
        if len(config_wsl) >= 2 and config_wsl[1] == ':':
            drive = config_wsl[0].lower()
            config_wsl = f'/mnt/{drive}{config_wsl[2:]}'
        
        # #region agent log
        with open(debug_log_path, 'a') as f: f.write(json.dumps({"hypothesisId":"WSL3","location":"pipelines.py:_run_via_wsl","message":"Config file created","data":{"config_path":config_path,"config_wsl":config_wsl,"bridge_wsl":bridge_wsl,"n_images":len(image_paths)},"timestamp":time.time()}) + '\n')
        # #endregion
        
        # Run the WSL bridge script with config file
        # Use Popen for real-time output streaming
        try:
            cmd = ['wsl', '-e', 'python3', '-u', bridge_wsl, '--config', config_wsl]
            
            # Dynamic timeout: 2 minutes per image, minimum 10 minutes
            n_images = len(image_paths)
            timeout_seconds = max(600, n_images * 120)  # 2 min per image
            
            # #region agent log
            with open(debug_log_path, 'a') as f: f.write(json.dumps({"hypothesisId":"WSL4","location":"pipelines.py:_run_via_wsl","message":"Running WSL command","data":{"cmd":cmd,"timeout":timeout_seconds,"n_images":n_images},"timestamp":time.time()}) + '\n')
            # #endregion
            
            # Use Popen for real-time stderr streaming
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1  # Line buffered
            )
            
            # Stream stderr in a separate thread, queue updates for main thread
            import threading
            import queue
            stderr_lines = []
            progress_queue = queue.Queue()
            
            def read_stderr():
                try:
                    for line in iter(process.stderr.readline, ''):
                        if not line:
                            break
                        line = line.strip()
                        if line:
                            stderr_lines.append(line)
                            progress_queue.put(line)
                except Exception as e:
                    logger.debug(f"Stderr reader ended: {e}")
            
            stderr_thread = threading.Thread(target=read_stderr, daemon=True)
            stderr_thread.start()
            
            # Poll for progress and process queue while waiting
            start_time = time.time()
            while process.poll() is None:
                # Process queued progress messages
                try:
                    while True:
                        line = progress_queue.get_nowait()
                        self._process_wsl_line(line)
                except queue.Empty:
                    pass
                
                # Check timeout
                if time.time() - start_time > timeout_seconds:
                    process.kill()
                    raise RuntimeError(f"WSL COLMAP timed out after {timeout_seconds//60} minutes")
                
                time.sleep(0.05)  # Small sleep to prevent busy-waiting
            
            # Process remaining queued messages
            try:
                while True:
                    line = progress_queue.get_nowait()
                    self._process_wsl_line(line)
            except queue.Empty:
                pass
            
            # Wait for process with timeout
            try:
                stdout, _ = process.communicate(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
                raise RuntimeError(f"WSL COLMAP timed out after {timeout_seconds//60} minutes ({n_images} images)")
            
            stderr_thread.join(timeout=1)
            stderr_output = '\n'.join(stderr_lines)
            
            # #region agent log
            with open(debug_log_path, 'a') as f: f.write(json.dumps({"hypothesisId":"WSL5","location":"pipelines.py:_run_via_wsl","message":"WSL command result","data":{"returncode":process.returncode,"stdout_len":len(stdout),"stderr_lines":len(stderr_lines),"stdout_preview":stdout[:500] if stdout else "","last_stderr_lines":stderr_lines[-5:] if stderr_lines else []},"timestamp":time.time()}) + '\n')
            # #endregion
            
            # Cleanup config file
            try:
                os.unlink(config_path)
            except:
                pass
            
            if process.returncode != 0:
                # #region agent log - Capture full error
                with open(debug_log_path, 'a') as f: f.write(json.dumps({"hypothesisId":"WSL_ERR","location":"pipelines.py:_run_via_wsl","message":"WSL COLMAP failed with non-zero return","data":{"returncode":process.returncode,"stderr_full":stderr_output,"stdout":stdout},"timestamp":time.time()}) + '\n')
                # #endregion
                logger.error(f"WSL COLMAP error (full): {stderr_output}")
                # Extract the actual error from stderr - look for traceback or last few lines
                error_lines = stderr_output.split('\n')
                # Find lines with ERROR, Traceback, or Exception
                error_summary = []
                for i, line in enumerate(error_lines):
                    if any(keyword in line for keyword in ['ERROR', 'Error', 'Traceback', 'Exception', 'failed']):
                        # Include this line and a few lines of context
                        error_summary.extend(error_lines[max(0, i-2):min(len(error_lines), i+5)])
                        break
                if error_summary:
                    error_msg = '\n'.join(error_summary[-20:])  # Last 20 lines of error context
                else:
                    error_msg = stderr_output[-1000:]  # Last 1000 chars if no specific error found
                raise RuntimeError(f"WSL COLMAP failed:\n{error_msg}")
            
            # Parse JSON result
            output = stdout.strip()
            if not output:
                raise RuntimeError(f"No output from WSL COLMAP. stderr: {stderr_output[-500:]}")
            
            wsl_result = json.loads(output)
            
            if not wsl_result.get('success'):
                raise RuntimeError(wsl_result.get('error', 'Unknown error'))
            
            # Load the panorama image
            output_path = wsl_result.get('output_path')
            if output_path:
                panorama = cv2.imread(output_path)
                wsl_result['panorama'] = panorama
            
            self._update_progress(100, "COLMAP (WSL GPU): Complete!")
            return wsl_result
            
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse WSL COLMAP output: {e}. Raw output: {stdout[:200] if stdout else 'empty'}")
    
    def get_install_instructions(self) -> str:
        return """
PyCOLMAP Installation:

Install via pip:
  pip install pycolmap

For GPU support (CUDA), you may need to build from source:
  See: https://colmap.github.io/pycolmap/installation.html

After installation, restart the application.
"""
    
    def run(self, image_paths: List[Path], output_dir: Path) -> Dict:
        """
        Run PyCOLMAP reconstruction pipeline.
        
        Returns:
            Dict with reconstruction results including camera poses
        """
        if not self.is_available():
            raise RuntimeError("PyCOLMAP is not installed. " + self.get_install_instructions())
        
        import pycolmap
        
        # Create workspace
        workspace = output_dir / "colmap_workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        
        images_dir = workspace / "images"
        database_path = workspace / "database.db"
        sparse_dir = workspace / "sparse"
        
        images_dir.mkdir(exist_ok=True)
        sparse_dir.mkdir(exist_ok=True)
        
        # Copy/link images to workspace
        self._update_progress(5, "Preparing images for PyCOLMAP...")
        for i, path in enumerate(image_paths):
            dest = images_dir / f"{i:06d}{path.suffix}"
            shutil.copy2(path, dest)
        
        # Feature extraction
        self._update_progress(10, "PyCOLMAP: Extracting features...")
        try:
            # #region agent log - H1,H2,H3: Check pycolmap API attributes
            import json
            debug_log_path = r"c:\Users\ryanf\OneDrive - University of Maryland\Desktop\Stitch2Stitch\Stitch2Stitch-1\.cursor\debug.log"
            sift_ext_opts = pycolmap.SiftExtractionOptions()
            sift_ext_attrs = [a for a in dir(sift_ext_opts) if not a.startswith('_')]
            with open(debug_log_path, 'a') as f: f.write(json.dumps({"hypothesisId":"H1","location":"pipelines.py:sift_extraction","message":"SiftExtractionOptions attributes","data":{"attrs":sift_ext_attrs,"pycolmap_version":getattr(pycolmap,'__version__','unknown')},"timestamp":__import__('time').time()}) + '\n')
            # Check if use_gpu exists on extraction options
            has_use_gpu_ext = hasattr(sift_ext_opts, 'use_gpu')
            with open(debug_log_path, 'a') as f: f.write(json.dumps({"hypothesisId":"H2","location":"pipelines.py:use_gpu_check","message":"SiftExtractionOptions has use_gpu?","data":{"has_use_gpu":has_use_gpu_ext},"timestamp":__import__('time').time()}) + '\n')
            # Check SiftMatchingOptions for comparison
            sift_match_opts = pycolmap.SiftMatchingOptions()
            sift_match_attrs = [a for a in dir(sift_match_opts) if not a.startswith('_')]
            has_use_gpu_match = hasattr(sift_match_opts, 'use_gpu')
            with open(debug_log_path, 'a') as f: f.write(json.dumps({"hypothesisId":"H3","location":"pipelines.py:sift_matching","message":"SiftMatchingOptions comparison","data":{"attrs":sift_match_attrs,"has_use_gpu":has_use_gpu_match},"timestamp":__import__('time').time()}) + '\n')
            # Check extract_features function signature
            import inspect
            try:
                sig = str(inspect.signature(pycolmap.extract_features))
            except:
                sig = "unknown"
            with open(debug_log_path, 'a') as f: f.write(json.dumps({"hypothesisId":"H5","location":"pipelines.py:extract_features_sig","message":"extract_features signature","data":{"signature":sig},"timestamp":__import__('time').time()}) + '\n')
            # Check has_cuda (it's a boolean property, not a method in pycolmap 3.13+)
            has_cuda = pycolmap.has_cuda if hasattr(pycolmap, 'has_cuda') else 'no has_cuda attr'
            with open(debug_log_path, 'a') as f: f.write(json.dumps({"hypothesisId":"H4","location":"pipelines.py:cuda_check","message":"CUDA availability","data":{"has_cuda":has_cuda},"timestamp":__import__('time').time()}) + '\n')
            # #endregion
            
            # Try calling extract_features without sift_options first (use defaults)
            # Use PINHOLE camera model (no lens distortion)
            pycolmap.extract_features(
                database_path,
                images_dir,
                camera_mode=pycolmap.CameraMode.SINGLE,
                camera_model="PINHOLE"
            )
            with open(debug_log_path, 'a') as f: f.write(json.dumps({"hypothesisId":"H4","location":"pipelines.py:extract_success","message":"extract_features succeeded","data":{"method":"no_sift_options"},"timestamp":__import__('time').time()}) + '\n')
        except Exception as e:
            logger.error(f"PyCOLMAP feature extraction error: {e}")
            raise RuntimeError(f"PyCOLMAP feature extraction failed: {e}")
        
        # Feature matching
        self._update_progress(30, f"PyCOLMAP: Matching features ({self.matcher_type})...")
        try:
            # #region agent log - H3: Check matching options
            import json
            debug_log_path = r"c:\Users\ryanf\OneDrive - University of Maryland\Desktop\Stitch2Stitch\Stitch2Stitch-1\.cursor\debug.log"
            with open(debug_log_path, 'a') as f: f.write(json.dumps({"hypothesisId":"H3","location":"pipelines.py:matching_start","message":"Starting feature matching","data":{"matcher_type":self.matcher_type},"timestamp":__import__('time').time()}) + '\n')
            # #endregion
            
            if self.matcher_type == "vocab_tree":
                # Note: vocab_tree_matcher requires a vocabulary tree file
                # For now, fall back to exhaustive if vocab tree not available
                logger.warning("Vocab tree matcher not implemented, using exhaustive matching")
                pycolmap.match_exhaustive(database_path)
            elif self.matcher_type == "sequential":
                pycolmap.match_sequential(database_path)
            else:  # exhaustive
                pycolmap.match_exhaustive(database_path)
            
            # #region agent log
            with open(debug_log_path, 'a') as f: f.write(json.dumps({"hypothesisId":"H3","location":"pipelines.py:matching_success","message":"Feature matching succeeded","data":{"matcher_type":self.matcher_type},"timestamp":__import__('time').time()}) + '\n')
            # #endregion
        except Exception as e:
            logger.error(f"PyCOLMAP matching error: {e}")
            raise RuntimeError(f"PyCOLMAP matching failed: {e}")
        
        # Sparse reconstruction (bundle adjustment)
        self._update_progress(60, "PyCOLMAP: Running bundle adjustment...")
        try:
            # #region agent log
            import json
            debug_log_path = r"c:\Users\ryanf\OneDrive - University of Maryland\Desktop\Stitch2Stitch\Stitch2Stitch-1\.cursor\debug.log"
            with open(debug_log_path, 'a') as f: f.write(json.dumps({"hypothesisId":"H4","location":"pipelines.py:mapping_start","message":"Starting incremental mapping","data":{},"timestamp":__import__('time').time()}) + '\n')
            # #endregion
            
            reconstructions = pycolmap.incremental_mapping(
                database_path,
                images_dir,
                sparse_dir
            )
            
            # #region agent log
            with open(debug_log_path, 'a') as f: f.write(json.dumps({"hypothesisId":"H4","location":"pipelines.py:mapping_result","message":"Incremental mapping result","data":{"num_reconstructions":len(reconstructions) if reconstructions else 0},"timestamp":__import__('time').time()}) + '\n')
            # #endregion
            
            if not reconstructions:
                return {"success": False, "error": "No reconstruction found"}
            
            # Write the first (best) reconstruction
            reconstructions[0].write(sparse_dir)
            
            # #region agent log
            with open(debug_log_path, 'a') as f: f.write(json.dumps({"hypothesisId":"H4","location":"pipelines.py:write_success","message":"Reconstruction written successfully","data":{},"timestamp":__import__('time').time()}) + '\n')
            # #endregion
        except Exception as e:
            logger.error(f"PyCOLMAP reconstruction error: {e}")
            raise RuntimeError(f"PyCOLMAP reconstruction failed: {e}")
        
        # Read reconstruction results
        self._update_progress(90, "PyCOLMAP: Reading results...")
        results = self._read_reconstruction(sparse_dir)
        
        self._update_progress(100, "PyCOLMAP: Complete!")
        return results
    
    def _read_reconstruction(self, sparse_dir: Path) -> Dict:
        """Read PyCOLMAP reconstruction results."""
        import pycolmap
        import json
        
        # #region agent log - Check Camera attributes in pycolmap 3.13
        debug_log_path = r"c:\Users\ryanf\OneDrive - University of Maryland\Desktop\Stitch2Stitch\Stitch2Stitch-1\.cursor\debug.log"
        # #endregion
        
        try:
            # Load reconstruction from the sparse directory
            reconstruction = pycolmap.Reconstruction(sparse_dir)
            
            # #region agent log - Check Camera object attributes
            if reconstruction.cameras:
                first_cam = next(iter(reconstruction.cameras.values()))
                cam_attrs = [a for a in dir(first_cam) if not a.startswith('_')]
                with open(debug_log_path, 'a') as f: f.write(json.dumps({"hypothesisId":"H6","location":"pipelines.py:camera_attrs","message":"Camera object attributes","data":{"attrs":cam_attrs},"timestamp":__import__('time').time()}) + '\n')
            # #endregion
            
            # Build camera info with compatible attribute names (pycolmap 3.13+)
            cameras_info = {}
            for cam_id, cam in reconstruction.cameras.items():
                # Try different attribute names for model
                model_name = getattr(cam, 'model_name', None) or getattr(cam, 'model', None) or str(type(cam).__name__)
                if hasattr(model_name, 'name'):  # It might be an enum
                    model_name = model_name.name
                cameras_info[cam_id] = {
                    "model": str(model_name),
                    "width": cam.width,
                    "height": cam.height,
                }
            
            # Build image info with compatible attribute names
            images_info = {}
            for img_id, img in reconstruction.images.items():
                # Check for 'registered' attribute or assume True if image exists
                is_registered = getattr(img, 'registered', True)
                images_info[img_id] = {
                    "name": img.name,
                    "camera_id": img.camera_id,
                    "registered": is_registered,
                }
            
            results = {
                "success": True,
                "workspace": str(sparse_dir.parent),
                "reconstruction_dir": str(sparse_dir),
                "n_images": len(reconstruction.images),
                "n_points3d": len(reconstruction.points3D),
                "n_cameras": len(reconstruction.cameras),
                "cameras": cameras_info,
                "images": images_info,
            }
            
            # #region agent log
            with open(debug_log_path, 'a') as f: f.write(json.dumps({"hypothesisId":"H6","location":"pipelines.py:read_success","message":"Reconstruction read successfully","data":{"n_images":results['n_images'],"n_points3d":results['n_points3d']},"timestamp":__import__('time').time()}) + '\n')
            # #endregion
            
            logger.info(f"PyCOLMAP reconstruction: {results['n_images']} images, "
                       f"{results['n_points3d']} 3D points, {results['n_cameras']} cameras")
            return results
        except Exception as e:
            logger.error(f"Error reading PyCOLMAP reconstruction: {e}")
            return {"success": False, "error": f"Failed to read reconstruction: {e}"}
    
    def run_2d_stitch(self, image_paths: List[Path], output_dir: Path) -> Dict:
        """
        Run PyCOLMAP for feature extraction/matching, then create a 2D panorama.
        
        This uses COLMAP's robust SIFT matching but produces a flat 2D composite
        instead of a 3D reconstruction.
        
        Returns:
            Dict with panorama results including the stitched image
        """
        import cv2
        import sqlite3
        import json
        import time
        
        if not self.is_available():
            raise RuntimeError("PyCOLMAP is not installed. " + self.get_install_instructions())
        
        import pycolmap
        
        # #region agent log - CUDA/GPU availability check
        debug_log_path = r"c:\Users\ryanf\OneDrive - University of Maryland\Desktop\Stitch2Stitch\Stitch2Stitch-1\.cursor\debug.log"
        cuda_available = pycolmap.has_cuda if hasattr(pycolmap, 'has_cuda') else False
        pycolmap_version = getattr(pycolmap, '__version__', 'unknown')
        with open(debug_log_path, 'a') as f: f.write(json.dumps({"hypothesisId":"CUDA1","location":"pipelines.py:run_2d_stitch","message":"CUDA availability at start","data":{"has_cuda":cuda_available,"pycolmap_version":pycolmap_version,"use_gpu_requested":self.use_gpu,"_has_cuda_attr":self._has_cuda,"_has_wsl_cuda":self._has_wsl_cuda},"timestamp":time.time()}) + '\n')
        # Check all pycolmap module attributes related to CUDA/device
        cuda_attrs = [a for a in dir(pycolmap) if 'cuda' in a.lower() or 'gpu' in a.lower() or 'device' in a.lower()]
        with open(debug_log_path, 'a') as f: f.write(json.dumps({"hypothesisId":"CUDA2","location":"pipelines.py:run_2d_stitch","message":"CUDA-related attributes in pycolmap","data":{"cuda_attrs":cuda_attrs},"timestamp":time.time()}) + '\n')
        # #endregion
        
        # Use WSL for GPU if native CUDA not available but WSL CUDA is
        if self._has_wsl_cuda and not self._has_cuda:
            with open(debug_log_path, 'a') as f: f.write(json.dumps({"hypothesisId":"WSL1","location":"pipelines.py:run_2d_stitch","message":"Using WSL for GPU acceleration","data":{},"timestamp":time.time()}) + '\n')
            return self._run_via_wsl(image_paths, output_dir)
        
        # Create workspace
        workspace = output_dir / "colmap_workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        
        images_dir = workspace / "images"
        database_path = workspace / "database.db"
        
        images_dir.mkdir(exist_ok=True)
        
        # Build mapping from copied names to original paths
        image_name_to_original = {}
        image_name_to_index = {}
        
        # Copy images to workspace
        self._update_progress(5, "Preparing images...")
        for i, path in enumerate(image_paths):
            dest_name = f"{i:06d}{path.suffix}"
            dest = images_dir / dest_name
            shutil.copy2(path, dest)
            image_name_to_original[dest_name] = path
            image_name_to_index[dest_name] = i
        
        # Feature extraction with COLMAP (PINHOLE = no lens distortion)
        gpu_mode = "GPU" if cuda_available else "CPU"
        n_images = len(image_paths)
        self._update_progress(10, f"COLMAP: Extracting features from {n_images} images using {gpu_mode}...")
        logger.info(f"Starting feature extraction for {n_images} images (this may take a few minutes without progress updates)")

        try:
            # #region agent log - Check extract_features parameters
            import inspect
            try:
                sig = str(inspect.signature(pycolmap.extract_features))
            except:
                sig = "unknown"
            # Check if there's a device parameter available
            sift_ext = pycolmap.SiftExtractionOptions()
            sift_ext_attrs = {a: str(getattr(sift_ext, a, None))[:50] for a in dir(sift_ext) if not a.startswith('_')}
            with open(debug_log_path, 'a') as f: f.write(json.dumps({"hypothesisId":"CUDA3","location":"pipelines.py:extract_features","message":"extract_features signature and options","data":{"signature":sig,"sift_extraction_options":sift_ext_attrs},"timestamp":time.time()}) + '\n')
            # #endregion

            # Log GPU status before extraction
            logger.info(f"PyCOLMAP feature extraction using {gpu_mode} mode (has_cuda={cuda_available})")
            with open(debug_log_path, 'a') as f: f.write(json.dumps({"hypothesisId":"CUDA4","location":"pipelines.py:extract_features","message":"Starting feature extraction","data":{"mode":gpu_mode,"has_cuda":cuda_available,"n_images":n_images},"timestamp":time.time()}) + '\n')

            # This is a blocking call - it will take a while with no intermediate progress
            import time
            start_time = time.time()

            pycolmap.extract_features(
                database_path,
                images_dir,
                camera_mode=pycolmap.CameraMode.SINGLE,
                camera_model="PINHOLE"
            )

            elapsed = time.time() - start_time
            logger.info(f"Feature extraction completed in {elapsed:.1f}s ({elapsed/n_images:.2f}s per image)")
            
            # #region agent log
            with open(debug_log_path, 'a') as f: f.write(json.dumps({"hypothesisId":"CUDA4","location":"pipelines.py:extract_features","message":"Feature extraction completed","data":{"mode":gpu_mode},"timestamp":time.time()}) + '\n')
            # #endregion
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            raise RuntimeError(f"Feature extraction failed: {e}")
        
        # Feature matching
        self._update_progress(30, "COLMAP: Matching features...")
        try:
            # #region agent log - Check match_exhaustive and SiftMatchingOptions
            sift_match = pycolmap.SiftMatchingOptions()
            sift_match_attrs = {a: str(getattr(sift_match, a, None))[:50] for a in dir(sift_match) if not a.startswith('_')}
            with open(debug_log_path, 'a') as f: f.write(json.dumps({"hypothesisId":"CUDA5","location":"pipelines.py:match_exhaustive","message":"SiftMatchingOptions attributes","data":{"sift_matching_options":sift_match_attrs},"timestamp":time.time()}) + '\n')
            
            # Check match_exhaustive signature
            try:
                match_sig = str(inspect.signature(pycolmap.match_exhaustive))
            except:
                match_sig = "unknown"
            with open(debug_log_path, 'a') as f: f.write(json.dumps({"hypothesisId":"CUDA5","location":"pipelines.py:match_exhaustive","message":"match_exhaustive signature","data":{"signature":match_sig},"timestamp":time.time()}) + '\n')
            # #endregion
            
            logger.info(f"PyCOLMAP feature matching using {gpu_mode} mode")
            pycolmap.match_exhaustive(database_path)
            
            # #region agent log
            with open(debug_log_path, 'a') as f: f.write(json.dumps({"hypothesisId":"CUDA5","location":"pipelines.py:match_exhaustive","message":"Feature matching completed","data":{"mode":gpu_mode},"timestamp":time.time()}) + '\n')
            # #endregion
        except Exception as e:
            logger.error(f"Feature matching failed: {e}")
            raise RuntimeError(f"Feature matching failed: {e}")
        
        # Read matches from database and compute transforms
        transform_type = "affine" if self.use_affine else "homography"
        self._update_progress(50, f"Computing {transform_type} transforms from matches...")
        
        try:
            matches_data = self._read_matches_from_db(database_path)
            if not matches_data:
                return {"success": False, "error": "No matches found between images"}
            
            # Load original images
            images = []
            for path in image_paths:
                img = cv2.imread(str(path))
                if img is None:
                    raise RuntimeError(f"Failed to load image: {path}")
                images.append(img)
            
            # Compute pairwise transforms (homography or affine)
            self._update_progress(60, "Computing image transformations...")
            transforms = self._compute_transforms_from_matches(
                matches_data, images, image_name_to_index
            )

            if not transforms:
                return {"success": False, "error": "Could not compute transforms"}

            # Stitch images
            self._update_progress(70, "Warping and blending images...")
            panorama = self._stitch_with_transforms(images, transforms)
            
            if panorama is None:
                return {"success": False, "error": "Failed to create panorama"}
            
            # Save panorama
            self._update_progress(90, "Saving panorama...")
            output_path = output_dir / "colmap_panorama.tiff"
            cv2.imwrite(str(output_path), panorama)
            
            self._update_progress(100, "Complete!")
            
            return {
                "success": True,
                "panorama": panorama,
                "output_path": str(output_path),
                "n_images": len(images),
                "size": panorama.shape[:2],
            }
            
        except Exception as e:
            logger.error(f"2D stitching failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _read_matches_from_db(self, database_path: Path) -> Dict:
        """Read keypoints and matches from COLMAP's SQLite database."""
        import sqlite3
        import json
        
        # #region agent log
        debug_log_path = r"c:\Users\ryanf\OneDrive - University of Maryland\Desktop\Stitch2Stitch\Stitch2Stitch-1\.cursor\debug.log"
        # #endregion
        
        conn = sqlite3.connect(str(database_path))
        cursor = conn.cursor()
        
        # Read images table to get image IDs and names
        cursor.execute("SELECT image_id, name FROM images")
        image_id_to_name = {row[0]: row[1] for row in cursor.fetchall()}
        
        # #region agent log
        with open(debug_log_path, 'a') as f: f.write(json.dumps({"hypothesisId":"DB1","location":"pipelines.py:read_images","message":"Images in database","data":{"image_ids":list(image_id_to_name.keys()),"names":list(image_id_to_name.values())},"timestamp":__import__('time').time()}) + '\n')
        # #endregion
        
        # Read keypoints for each image
        keypoints = {}
        cursor.execute("SELECT image_id, rows, cols, data FROM keypoints")
        for row in cursor.fetchall():
            image_id, rows, cols, data = row
            if data:
                # Keypoints are stored as float32: x, y, scale, orientation (6 values per keypoint)
                arr = np.frombuffer(data, dtype=np.float32).reshape(rows, cols)
                keypoints[image_id] = arr[:, :2]  # Just x, y coordinates
        
        # #region agent log
        kp_info = {str(k): v.shape[0] for k, v in keypoints.items()}
        with open(debug_log_path, 'a') as f: f.write(json.dumps({"hypothesisId":"DB2","location":"pipelines.py:read_keypoints","message":"Keypoints per image","data":{"keypoints_count":kp_info},"timestamp":__import__('time').time()}) + '\n')
        # #endregion
        
        # Read two-view geometry (verified matches) instead of raw matches
        # This contains geometrically verified matches which are more reliable
        matches = {}
        cursor.execute("SELECT pair_id, rows, cols, data FROM two_view_geometries WHERE rows > 0")
        rows_found = cursor.fetchall()
        
        # #region agent log
        with open(debug_log_path, 'a') as f: f.write(json.dumps({"hypothesisId":"DB3","location":"pipelines.py:read_matches","message":"Two-view geometries found","data":{"n_pairs":len(rows_found)},"timestamp":__import__('time').time()}) + '\n')
        # #endregion
        
        for row in rows_found:
            pair_id, n_rows, cols, data = row
            if data and n_rows > 0:
                # Decode pair_id to get image IDs (COLMAP uses image_id1 < image_id2)
                # pair_id = image_id1 * MAX_NUM_IMAGES + image_id2
                # where MAX_NUM_IMAGES = 2147483647
                image_id2 = pair_id % 2147483647
                image_id1 = pair_id // 2147483647
                
                # Ensure id1 < id2 (COLMAP convention)
                if image_id1 > image_id2:
                    image_id1, image_id2 = image_id2, image_id1
                
                # Matches are stored as uint32 pairs
                match_arr = np.frombuffer(data, dtype=np.uint32).reshape(n_rows, cols)
                
                name1 = image_id_to_name.get(image_id1)
                name2 = image_id_to_name.get(image_id2)
                
                # #region agent log
                if image_id1 in keypoints and image_id2 in keypoints:
                    max_idx1 = match_arr[:, 0].max() if len(match_arr) > 0 else -1
                    max_idx2 = match_arr[:, 1].max() if len(match_arr) > 0 else -1
                    kp1_size = keypoints[image_id1].shape[0]
                    kp2_size = keypoints[image_id2].shape[0]
                    with open(debug_log_path, 'a') as f: f.write(json.dumps({"hypothesisId":"DB4","location":"pipelines.py:match_pair","message":"Match pair info","data":{"pair_id":pair_id,"id1":image_id1,"id2":image_id2,"name1":name1,"name2":name2,"n_matches":n_rows,"max_idx1":int(max_idx1),"max_idx2":int(max_idx2),"kp1_size":kp1_size,"kp2_size":kp2_size},"timestamp":__import__('time').time()}) + '\n')
                # #endregion
                
                if name1 and name2 and image_id1 in keypoints and image_id2 in keypoints:
                    kp1 = keypoints[image_id1]
                    kp2 = keypoints[image_id2]
                    
                    # Validate match indices are within bounds
                    valid_mask = (match_arr[:, 0] < kp1.shape[0]) & (match_arr[:, 1] < kp2.shape[0])
                    valid_matches = match_arr[valid_mask]
                    
                    if len(valid_matches) > 0:
                        matches[(name1, name2)] = {
                            'kp1': kp1,
                            'kp2': kp2,
                            'matches': valid_matches
                        }
        
        conn.close()
        
        # #region agent log
        with open(debug_log_path, 'a') as f: f.write(json.dumps({"hypothesisId":"DB5","location":"pipelines.py:matches_result","message":"Final matches loaded","data":{"n_pairs":len(matches),"pairs":list(str(k) for k in matches.keys())},"timestamp":__import__('time').time()}) + '\n')
        # #endregion
        
        logger.info(f"Read {len(matches)} image pairs with matches from COLMAP database")
        return matches
    
    def _compute_transforms_from_matches(
        self,
        matches_data: Dict,
        images: List[np.ndarray],
        image_name_to_index: Dict
    ) -> Dict:
        """Compute transforms (homography or affine) between image pairs from COLMAP matches."""
        import cv2

        transforms = {}

        for (name1, name2), data in matches_data.items():
            idx1 = image_name_to_index.get(name1)
            idx2 = image_name_to_index.get(name2)

            if idx1 is None or idx2 is None:
                continue

            kp1 = data['kp1']
            kp2 = data['kp2']
            match_indices = data['matches']

            if len(match_indices) < 4:
                continue

            # Get matched point coordinates
            pts1 = kp1[match_indices[:, 0]]
            pts2 = kp2[match_indices[:, 1]]

            if self.use_affine:
                # Compute affine transform (2x3 matrix)
                M, mask = cv2.estimateAffine2D(
                    pts1, pts2,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=5.0
                )

                if M is not None:
                    # Convert 2x3 to 3x3 for consistency
                    H = np.vstack([M, [0, 0, 1]])
                    n_inliers = np.sum(mask) if mask is not None else 0
                    if n_inliers >= 10:
                        transforms[(idx1, idx2)] = {
                            'H': H,
                            'n_matches': len(match_indices),
                            'n_inliers': n_inliers,
                            'is_affine': True
                        }
                        logger.debug(f"Affine {idx1}->{idx2}: {n_inliers}/{len(match_indices)} inliers")
            else:
                # Compute homography (3x3 matrix)
                H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)

                if H is not None:
                    n_inliers = np.sum(mask) if mask is not None else 0
                    if n_inliers >= 10:
                        transforms[(idx1, idx2)] = {
                            'H': H,
                            'n_matches': len(match_indices),
                            'n_inliers': n_inliers,
                            'is_affine': False
                        }
                        logger.debug(f"Homography {idx1}->{idx2}: {n_inliers}/{len(match_indices)} inliers")

        return transforms
    
    def _stitch_with_transforms(
        self,
        images: List[np.ndarray],
        transforms: Dict
    ) -> Optional[np.ndarray]:
        """Stitch images using computed transforms (homography or affine) - optimized for ordered sequential captures."""
        import cv2
        
        n_images = len(images)
        if n_images == 0:
            return None
        
        if n_images == 1:
            return images[0]
        
        # Build connectivity graph and find reference image
        connections = {}
        for (i, j) in transforms.keys():
            connections[i] = connections.get(i, 0) + 1
            connections[j] = connections.get(j, 0) + 1

        if not connections:
            # No transforms found, just return first image
            return images[0]
        
        # Use middle image as reference to minimize error accumulation
        ref_idx = n_images // 2
        if ref_idx not in connections:
            # Fallback to most connected if middle not in graph
            ref_idx = max(connections.keys(), key=lambda x: connections[x])
        
        logger.info(f"Using image {ref_idx} as reference (middle of sequence)")
        
        # Build adjacency graph
        # H in transforms[(i,j)] transforms points from image i to image j: p_j = H @ p_i
        import heapq
        adjacency = {}
        for (i, j), data in transforms.items():
            if i not in adjacency:
                adjacency[i] = []
            if j not in adjacency:
                adjacency[j] = []
            
            # Weight = number of inliers (let feature matching determine spatial adjacency)
            # Small sequential bonus for within-row connections, but don't override poor matches
            seq_bonus = 5 if abs(i - j) == 1 else 0
            weight = data['n_inliers'] + seq_bonus
            
            # Store both directions with correct transformations
            # From i to j: use H (transforms points i->j)
            # From j to i: use inv(H) (transforms points j->i)
            adjacency[i].append((j, data['H'], weight))
            adjacency[j].append((i, np.linalg.inv(data['H']), weight))
        
        # Compute cumulative homographies using BFS with priority (Dijkstra-like)
        H_to_ref = {ref_idx: np.eye(3)}
        visited = {ref_idx}
        # Priority queue: (negative_weight, distance_from_ref, current_idx)
        pq = [(0, 0, ref_idx)]
        
        while pq:
            neg_weight, dist, current = heapq.heappop(pq)
            
            if current != ref_idx and current in H_to_ref:
                # Already processed with a better path
                continue
            
            H_current = H_to_ref[current]
            
            for neighbor, H_edge, weight in adjacency.get(current, []):
                if neighbor not in visited:
                    # H_edge transforms from current to neighbor
                    # To get neighbor->ref: first go neighbor->current (inv(H_edge)), then current->ref (H_current)
                    # p_ref = H_current @ inv(H_edge) @ p_neighbor
                    H_neighbor = H_current @ np.linalg.inv(H_edge)
                    H_to_ref[neighbor] = H_neighbor
                    visited.add(neighbor)
                    heapq.heappush(pq, (-weight, dist + 1, neighbor))
        
        logger.info(f"Computed transforms for {len(H_to_ref)}/{n_images} images")
        
        # Compute output canvas size
        corners_all = []
        for idx, H in H_to_ref.items():
            h, w = images[idx].shape[:2]
            corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)

            # Check if we're using affine transforms
            if self.use_affine:
                # For affine, use transform with 2x3 matrix
                M_affine = H[:2, :]
                transformed = cv2.transform(corners, M_affine)
            else:
                # For homography, use perspectiveTransform
                transformed = cv2.perspectiveTransform(corners, H)
            corners_all.append(transformed)
        
        corners_all = np.concatenate(corners_all, axis=0)
        x_min, y_min = corners_all.min(axis=0).ravel()
        x_max, y_max = corners_all.max(axis=0).ravel()
        
        # Add offset to handle negative coordinates
        offset_x = -int(np.floor(x_min))
        offset_y = -int(np.floor(y_min))
        
        output_w = int(np.ceil(x_max - x_min))
        output_h = int(np.ceil(y_max - y_min))
        
        # Limit size to prevent memory issues
        max_dim = 15000
        if output_w > max_dim or output_h > max_dim:
            scale = max_dim / max(output_w, output_h)
            output_w = int(output_w * scale)
            output_h = int(output_h * scale)
            offset_x = int(offset_x * scale)
            offset_y = int(offset_y * scale)
            # Scale homographies
            S = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]])
            H_to_ref = {k: S @ v for k, v in H_to_ref.items()}
        
        # Translation matrix
        T = np.array([[1, 0, offset_x], [0, 1, offset_y], [0, 0, 1]], dtype=np.float64)

        # Prepare aligned images for ImageBlender
        aligned_images = []
        for idx, H in H_to_ref.items():
            img = images[idx]
            H_final = T @ H

            # Compute bbox for this image
            h, w = img.shape[:2]
            corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)

            if self.use_affine:
                M_affine = H_final[:2, :]
                transformed = cv2.transform(corners, M_affine)
            else:
                transformed = cv2.perspectiveTransform(corners, H_final)

            x_min = int(transformed[0, :, 0].min())
            y_min = int(transformed[0, :, 1].min())
            x_max = int(transformed[0, :, 0].max())
            y_max = int(transformed[0, :, 1].max())

            # Warp the image
            if self.use_affine:
                M_affine = H_final[:2, :]
                warped = cv2.warpAffine(img, M_affine, (output_w, output_h))
            else:
                warped = cv2.warpPerspective(img, H_final, (output_w, output_h))

            aligned_images.append({
                'image': warped,
                'bbox': (x_min, y_min, x_max, y_max),
                'transform': H_final
            })

        # Use ImageBlender to blend
        logger.info(f"Blending {len(aligned_images)} images using {self.blend_method} method...")

        # Import ImageBlender if not already imported
        if ImageBlender is None:
            from core.blender import ImageBlender as BlenderClass
        else:
            BlenderClass = ImageBlender

        blender = BlenderClass(
            method=self.blend_method,
            hdr_mode=False,
            anti_ghosting=False
        )

        panorama = blender.blend(aligned_images, padding=0, fit_all=False)

        return panorama


class HLOCPipeline(ExternalPipelineBase):
    """
    HLOC (Hierarchical Localization) Integration
    
    HLOC provides:
    - NetVLAD for global image retrieval
    - SuperPoint for feature detection
    - SuperGlue for feature matching
    - PyCOLMAP for reconstruction
    
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

Click the 'Install' button to automatically install:
1. PyTorch with CUDA 11.8 support (~2GB download)
2. Additional dependencies (kornia, h5py, plotly, gdown)
3. HLOC from GitHub
4. LightGlue from GitHub

Requirements:
- Git must be installed (download from https://git-scm.com/download/win)
- PyCOLMAP must be installed (required by main app)
- ~2.5GB free disk space

Installation may take 15-20 minutes depending on your internet speed.

After installation, restart the application.
"""
    
    def run(self, image_paths: List[Path], output_dir: Path) -> Dict:
        """
        Run HLOC pipeline with SuperPoint + SuperGlue, using PyCOLMAP for reconstruction.
        """
        if not self.is_available():
            raise RuntimeError("HLOC is not installed. " + self.get_install_instructions())
        
        # Check if pycolmap is available
        try:
            import pycolmap
        except ImportError:
            raise RuntimeError("PyCOLMAP is required for HLOC reconstruction. Install with: pip install pycolmap")
        
        from hloc import extract_features, match_features
        from hloc import pairs_from_retrieval
        
        # Create workspace
        workspace = output_dir / "hloc_workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        
        images_dir = workspace / "images"
        outputs_dir = workspace / "outputs"
        sparse_dir = workspace / "sparse"
        
        images_dir.mkdir(exist_ok=True)
        outputs_dir.mkdir(exist_ok=True)
        sparse_dir.mkdir(exist_ok=True)
        
        # Copy images
        self._update_progress(5, "Preparing images for HLOC...")
        logger.info(f"Copying {len(image_paths)} images to {images_dir}")
        image_list = []
        for i, path in enumerate(image_paths):
            dest = images_dir / f"{i:06d}{path.suffix}"
            shutil.copy2(path, dest)
            # Store relative path from images_dir
            image_list.append(dest.name)
            logger.debug(f"Copied {path.name} -> {dest.name}")

        logger.info(f"Created image list with {len(image_list)} images: {image_list[:3]}...")

        # Write image list file (HLOC expects this)
        image_list_path = outputs_dir / "image_list.txt"
        with open(image_list_path, 'w') as f:
            for img_name in image_list:
                f.write(f"{img_name}\n")

        logger.info(f"Images directory: {images_dir}")
        logger.info(f"Images in directory: {list(images_dir.glob('*'))[:3]}...")

        # Feature extraction with SuperPoint
        self._update_progress(15, "HLOC: Extracting SuperPoint features...")
        feature_conf = extract_features.confs["superpoint_aachen"]
        feature_path = extract_features.main(
            feature_conf, images_dir, outputs_dir, image_list=image_list
        )
        
        # Global image retrieval with NetVLAD
        self._update_progress(35, "HLOC: Computing global descriptors (NetVLAD)...")
        retrieval_conf = extract_features.confs["netvlad"]
        retrieval_path = extract_features.main(
            retrieval_conf, images_dir, outputs_dir, image_list=image_list
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
        
        # Reconstruction with PyCOLMAP
        self._update_progress(75, "HLOC: Running reconstruction with PyCOLMAP...")
        try:
            # Use HLOC's triangulation which uses pycolmap internally
            from hloc import triangulation
            
            model = triangulation.main(
                sparse_dir,
                images_dir,
                pairs_path,
                feature_path,
                match_path
            )
        except Exception as e:
            logger.error(f"HLOC reconstruction error: {e}")
            raise RuntimeError(f"HLOC reconstruction failed: {e}")
        
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
    colmap = COLMAPPipeline()
    hloc = HLOCPipeline()
    alice = AliceVisionPipeline()
    
    colmap_available = colmap.is_available()
    hloc_available = hloc.is_available()
    alice_available = alice.is_available()
    
    if detailed:
        return {
            "colmap": {
                "available": colmap_available,
                "info": colmap.get_version_info() if colmap_available else "Not found",
                "version": colmap._version if colmap_available else None,
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
        # For large sets, PyCOLMAP is reliable
        if available["colmap"]:
            return "colmap"
        elif available["hloc"]:
            return "hloc"
    
    # For smaller sets or if nothing is available
    return "internal"
