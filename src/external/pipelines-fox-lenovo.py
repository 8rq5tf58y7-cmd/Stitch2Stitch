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
# NOTE: Debug-mode requires writing to the exact workspace log path (works even if app runs from AppData).
DEBUG_LOG_PATH = Path(r"c:\Users\ryanf\OneDrive - University of Maryland\Desktop\Stitch2Stitch\Stitch2Stitch-1\.cursor\debug.log")
def _debug_log(location: str, message: str, data: dict = None, hypothesis_id: str = None):
    try:
        # Ensure parent directory exists
        DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        run_id = os.environ.get("STITCH2STITCH_RUN_ID", "run1")
        log_entry = {
            "sessionId": "debug-session",
            "runId": run_id,
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
        
        # Search all conda environments (including hlocenv, etc.)
        for conda_root in [user_home / 'miniconda3', user_home / 'anaconda3', 
                           user_home / 'Miniconda3', user_home / 'Anaconda3']:
            envs_dir = conda_root / 'envs'
            if envs_dir.exists():
                for env_dir in envs_dir.iterdir():
                    if env_dir.is_dir():
                        paths_to_check.extend([
                            env_dir / 'Library' / 'bin' / f'{name}.exe',
                            env_dir / 'Scripts' / f'{name}.exe',
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
        self._use_conda_run = False
        
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

    def _infer_conda_env_root_from_colmap(self) -> Optional[Path]:
        """If colmap.exe is inside a conda env, infer the env root path."""
        try:
            p = Path(self.colmap_path)
            if p.exists() and p.parent.name.lower() == "bin" and p.parent.parent.name.lower() == "library":
                env_root = p.parent.parent.parent
                return env_root if env_root.exists() else None
        except Exception:
            return None
        return None

    def _find_conda_bat(self) -> Optional[str]:
        """Find conda.bat (Windows) to enable 'conda run' fallback."""
        candidates: List[Path] = []
        # Environment variable if present
        conda_exe = os.environ.get("CONDA_EXE")
        if conda_exe:
            try:
                p = Path(conda_exe)
                # Often ...\Scripts\conda.exe or ...\condabin\conda.bat
                candidates.extend([
                    p,
                    p.parent / "conda.bat",
                    p.parent.parent / "condabin" / "conda.bat",
                ])
            except Exception:
                pass

        user_home = Path.home()
        for root in [user_home / "miniconda3", user_home / "anaconda3", user_home / "Miniconda3", user_home / "Anaconda3"]:
            candidates.append(root / "condabin" / "conda.bat")
            candidates.append(root / "Scripts" / "conda.exe")

        for c in candidates:
            try:
                if c.exists():
                    return str(c)
            except Exception:
                continue
        return None

    def _run_via_conda(self, env_root: Path, args: List[str], timeout: Optional[int] = None) -> subprocess.CompletedProcess:
        """Run colmap via `conda run -p <env_root>` to get correct DLL environment.

        NOTE: Some COLMAP commands (e.g., feature_extractor for hundreds of images) can take minutes+.
        We therefore allow long timeouts and stream output to avoid pipe buffer stalls.
        """
        conda = self._find_conda_bat()
        # #region agent log
        _debug_log("pipelines.py:_run_via_conda", "ENTRY", {"env_root": str(env_root), "args": args, "conda": conda, "timeout": timeout}, "G")
        # #endregion
        if not conda:
            raise FileNotFoundError("conda executable not found for conda-run fallback")

        if conda.lower().endswith(".bat"):
            cmd = ["cmd", "/c", conda, "run", "-p", str(env_root), "colmap"] + args
        else:
            cmd = [conda, "run", "-p", str(env_root), "colmap"] + args

        # #region agent log
        _debug_log("pipelines.py:_run_via_conda", "BEFORE Popen", {"cmd": cmd}, "G")
        # #endregion

        # Stream stdout/stderr to avoid deadlocks on large output; keep only a small tail in memory.
        output_lines: List[str] = []
        max_tail_lines = 300
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            assert proc.stdout is not None
            start = time.time()
            for line in proc.stdout:
                if line:
                    output_lines.append(line)
                    if len(output_lines) > max_tail_lines:
                        output_lines = output_lines[-max_tail_lines:]
            rc = proc.wait(timeout=timeout)
            elapsed = time.time() - start
            # #region agent log
            _debug_log("pipelines.py:_run_via_conda", "AFTER Popen", {"returncode": rc, "elapsed_s": elapsed}, "G")
            # #endregion
            return subprocess.CompletedProcess(cmd, rc, stdout="".join(output_lines), stderr="")
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except Exception:
                pass
            # #region agent log
            _debug_log("pipelines.py:_run_via_conda", "TIMEOUT", {"timeout": timeout, "tail_lines": output_lines[-20:]}, "G")
            # #endregion
            raise
    
    def _get_colmap_env(self) -> dict:
        """Get environment with conda-style DLL directories in PATH for COLMAP."""
        env = os.environ.copy()
        colmap_path = Path(self.colmap_path)
        colmap_dir = colmap_path.parent

        # If COLMAP comes from a conda env, it's typically:
        #   <env_root>\Library\bin\colmap.exe
        # and many DLLs live in:
        #   <env_root>\Library\bin
        #   <env_root>\Library\usr\bin
        #   <env_root>\Scripts
        env_root = None
        try:
            if colmap_dir.name.lower() == "bin" and colmap_dir.parent.name.lower() == "library":
                env_root = colmap_dir.parent.parent
        except Exception:
            env_root = None

        prepend_paths = []
        if env_root and env_root.exists():
            prepend_paths.extend([
                str(env_root / "Library" / "bin"),
                str(env_root / "Library" / "usr" / "bin"),
                str(env_root / "Scripts"),
                str(env_root),
            ])

            # Help some tooling that expects these to be set when running in a conda env
            env.setdefault("CONDA_PREFIX", str(env_root))
            env.setdefault("CONDA_DEFAULT_ENV", env_root.name)
        else:
            # Fallback: at least ensure COLMAP's own directory is first on PATH
            prepend_paths.append(str(colmap_dir))

        # Prepend only existing directories (avoid bloating PATH with non-existent entries)
        prepend_paths = [p for p in prepend_paths if Path(p).exists()]
        # Windows env var casing: Path vs PATH. If we set the wrong casing, the child
        # process may ignore it depending on how the environment block is built.
        path_keys = [k for k in env.keys() if k.lower() == "path"]
        path_key = path_keys[0] if path_keys else "PATH"

        current_path = env.get(path_key, "")
        new_path = ";".join(prepend_paths + ([current_path] if current_path else []))

        # Remove any other PATH-cased keys to avoid duplicates in the env block
        for k in list(env.keys()):
            if k.lower() == "path" and k != path_key:
                env.pop(k, None)

        env[path_key] = new_path

        # #region agent log
        _debug_log(
            "pipelines.py:_get_colmap_env",
            "Computed env for COLMAP",
            {
                "colmap_path": str(colmap_path),
                "env_root": str(env_root) if env_root else None,
                "prepended_paths": prepend_paths,
                "path_key_used": path_key,
                "path_keys_seen": path_keys,
                "path_head": env[path_key][:200],
            },
            "F",
        )
        # #endregion

        return env
    
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
            
            # Use environment with COLMAP's directory in PATH for DLL loading
            env = self._get_colmap_env()
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10,
                env=env
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

                # Fallback: if this colmap is from a conda env, try `conda run` which sets DLL paths correctly.
                env_root = self._infer_conda_env_root_from_colmap()
                if env_root:
                    try:
                        conda_res = self._run_via_conda(env_root, ["help"])
                        # #region agent log
                        _debug_log("pipelines.py:is_available", "CONDA-RUN result", {
                            "returncode": conda_res.returncode,
                            "stdout_length": len(conda_res.stdout or ""),
                            "stderr_length": len(conda_res.stderr or ""),
                            "stdout_preview": (conda_res.stdout or "")[:200],
                            "stderr_preview": (conda_res.stderr or "")[:200],
                        }, "G")
                        # #endregion
                        if conda_res.returncode == 0 and "COLMAP" in (conda_res.stdout or ""):
                            self._location = self.colmap_path
                            self._diagnostic_error = None
                            self._version = (conda_res.stdout or "").splitlines()[0].strip() if conda_res.stdout else self._version
                            self._use_conda_run = True
                            return True
                    except Exception as e:
                        # #region agent log
                        _debug_log("pipelines.py:is_available", "CONDA-RUN exception", {"error": str(e), "type": type(e).__name__}, "G")
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
        # If detection determined we need conda-run (DLL-safe), use it
        if getattr(self, "_use_conda_run", False):
            env_root = self._infer_conda_env_root_from_colmap()
            if not env_root:
                raise RuntimeError("COLMAP requires conda-run but env root could not be inferred")
            # Use longer timeouts for heavy steps.
            # 563 images can easily take many minutes for feature extraction/matching.
            cmd0 = args[0] if args else ""
            if cmd0 in ("feature_extractor", "exhaustive_matcher", "spatial_matcher", "sequential_matcher", "vocab_tree_matcher"):
                timeout = 60 * 60 * 3  # 3 hours
            elif cmd0 in ("mapper", "bundle_adjuster", "hierarchical_mapper"):
                timeout = 60 * 60 * 3
            else:
                timeout = 60 * 20  # 20 minutes
            result = self._run_via_conda(env_root, args, timeout=timeout)
        else:
            cmd = [self.colmap_path] + args
            logger.info(f"Running COLMAP: {' '.join(cmd)}")
            
            # Use environment with COLMAP's directory in PATH for DLL loading
            env = self._get_colmap_env()
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env
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


