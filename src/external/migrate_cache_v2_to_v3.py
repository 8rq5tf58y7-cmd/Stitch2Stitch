#!/usr/bin/env python3
"""
Cache Migration Script: v2 to v3

Migrates legacy COLMAP caches from v2 format (which included blend_method,
warp_interpolation, use_affine in the key) to v3 format (which only includes
image set + max_features + matcher_type).

Usage:
    # From within WSL:
    python3 migrate_cache_v2_to_v3.py /path/to/original/images

    # Or from Windows, targeting WSL images:
    python migrate_cache_v2_to_v3.py "C:\\path\\to\\images"

The script will:
1. Find all existing v2 caches that match the image set
2. Compute the new v3 cache key
3. Create a symlink or copy to preserve the cache under the new key
"""

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def windows_to_wsl_path(win_path: str) -> str:
    """Convert Windows path to WSL path."""
    path = win_path.replace("\\", "/")
    if len(path) >= 2 and path[1] == ":":
        drive = path[0].lower()
        path = f"/mnt/{drive}{path[2:]}"
    return path


def compute_v2_cache_key(
    image_paths: List[Path],
    max_features: int,
    matcher_type: str = "exhaustive",
    use_affine: bool = False,
    blend_method: str = "multiband",
    warp_interpolation: str = "linear",
) -> str:
    """Compute a v2 format cache key (legacy format)."""
    cache_data: List[str] = []
    for p in sorted(image_paths, key=lambda x: x.name):
        try:
            cache_data.append(f"{p.name}:{p.stat().st_size}")
        except Exception:
            cache_data.append(p.name)

    cache_data.append(f"max_features:{int(max_features)}")
    cache_data.append("global_cache_format:v2")
    cache_data.append(f"matcher_type:{matcher_type}")
    cache_data.append(f"use_affine:{bool(use_affine)}")
    cache_data.append(f"blend_method:{str(blend_method).lower()}")
    cache_data.append(f"warp_interp:{str(warp_interpolation).lower()}")
    cache_data.append(f"count:{len(image_paths)}")
    cache_str = "\n".join(cache_data)
    return hashlib.md5(cache_str.encode("utf-8")).hexdigest()


def compute_v3_cache_key(
    image_paths: List[Path],
    max_features: int,
    matcher_type: str = "exhaustive",
) -> str:
    """Compute a v3 format cache key (new format)."""
    cache_data: List[str] = []
    for p in sorted(image_paths, key=lambda x: x.name):
        try:
            cache_data.append(f"{p.name}:{p.stat().st_size}")
        except Exception:
            cache_data.append(p.name)

    cache_data.append(f"max_features:{int(max_features)}")
    cache_data.append("global_cache_format:v3")
    cache_data.append(f"matcher_type:{matcher_type}")
    cache_data.append(f"count:{len(image_paths)}")
    cache_str = "\n".join(cache_data)
    return hashlib.md5(cache_str.encode("utf-8")).hexdigest()


def get_cache_root() -> Path:
    """Get the cache root directory."""
    cache_base = Path(os.environ.get(
        "STITCH2STITCH_CACHE_DIR",
        str(Path.home() / ".stitch2stitch_cache")
    ))
    return cache_base / "colmap"


def find_v2_caches(
    image_paths: List[Path],
    max_features: int = 8192,
    matcher_type: str = "exhaustive",
) -> List[Tuple[str, Path, Dict]]:
    """
    Find all v2 caches that match the given image set.

    Returns list of (v2_key, cache_path, settings) tuples.
    """
    cache_root = get_cache_root()
    if not cache_root.exists():
        return []

    found_caches = []

    # Try all combinations of v2 settings
    blend_methods = ["multiband", "feather", "autostitch", "linear"]
    warp_interps = ["linear", "cubic", "lanczos", "nearest"]
    use_affines = [False, True]

    for blend in blend_methods:
        for warp in warp_interps:
            for affine in use_affines:
                v2_key = compute_v2_cache_key(
                    image_paths,
                    max_features,
                    matcher_type,
                    use_affine=affine,
                    blend_method=blend,
                    warp_interpolation=warp,
                )

                cache_dir = cache_root / v2_key
                if cache_dir.exists() and (cache_dir / "database.db").exists():
                    settings = {
                        "use_affine": affine,
                        "blend_method": blend,
                        "warp_interpolation": warp,
                    }
                    found_caches.append((v2_key, cache_dir, settings))

    return found_caches


def migrate_cache(
    image_paths: List[Path],
    max_features: int = 8192,
    matcher_type: str = "exhaustive",
    dry_run: bool = False,
) -> Dict:
    """
    Migrate v2 caches to v3 format for the given image set.

    Args:
        image_paths: List of original image paths
        max_features: Max features setting (default 8192)
        matcher_type: Matcher type (default "exhaustive")
        dry_run: If True, only report what would be done

    Returns:
        Dict with migration results
    """
    cache_root = get_cache_root()

    # Find existing v2 caches
    v2_caches = find_v2_caches(image_paths, max_features, matcher_type)

    if not v2_caches:
        return {
            "success": False,
            "message": "No v2 caches found for this image set",
            "v2_caches_found": 0,
        }

    # Compute v3 key
    v3_key = compute_v3_cache_key(image_paths, max_features, matcher_type)
    v3_dir = cache_root / v3_key

    print(f"Found {len(v2_caches)} v2 cache(s) for this image set")
    print(f"New v3 cache key: {v3_key}")

    # Check if v3 cache already exists
    if v3_dir.exists():
        print(f"v3 cache already exists at {v3_dir}")
        return {
            "success": True,
            "message": "v3 cache already exists",
            "v3_key": v3_key,
            "v2_caches_found": len(v2_caches),
            "migrated": False,
        }

    # Pick the best v2 cache (prefer newest, or one with warp cache)
    best_cache = None
    best_score = -1

    for v2_key, cache_path, settings in v2_caches:
        # Score based on modification time and presence of warp cache
        try:
            mtime = (cache_path / "database.db").stat().st_mtime
            has_warp_cache = any(cache_path.glob("warped_*"))
            score = mtime + (1e9 if has_warp_cache else 0)

            if score > best_score:
                best_score = score
                best_cache = (v2_key, cache_path, settings)
        except Exception:
            continue

    if not best_cache:
        return {
            "success": False,
            "message": "No valid v2 cache found",
            "v2_caches_found": len(v2_caches),
        }

    v2_key, source_dir, settings = best_cache
    print(f"Selected v2 cache: {v2_key}")
    print(f"  Settings: {settings}")

    if dry_run:
        print(f"DRY RUN: Would create symlink {v3_dir} -> {source_dir}")
        return {
            "success": True,
            "message": "Dry run - no changes made",
            "v3_key": v3_key,
            "v2_key": v2_key,
            "source_dir": str(source_dir),
            "dry_run": True,
        }

    # Create symlink or copy
    try:
        # Try symlink first (more efficient)
        os.symlink(source_dir, v3_dir)
        print(f"Created symlink: {v3_dir} -> {source_dir}")
        method = "symlink"
    except (OSError, NotImplementedError):
        # Fall back to copying
        print(f"Symlink failed, copying cache directory...")
        shutil.copytree(source_dir, v3_dir)
        print(f"Copied cache to: {v3_dir}")
        method = "copy"

    # Update the cache_key.txt in the new location
    cache_key_file = v3_dir / "cache_key.txt"
    if cache_key_file.exists() or method == "copy":
        try:
            cache_key_file.write_text(v3_key, encoding="utf-8")
            print(f"Updated cache_key.txt with v3 key")
        except Exception as e:
            print(f"Warning: Could not update cache_key.txt: {e}")

    return {
        "success": True,
        "message": f"Cache migrated via {method}",
        "v3_key": v3_key,
        "v2_key": v2_key,
        "source_dir": str(source_dir),
        "target_dir": str(v3_dir),
        "method": method,
        "v2_caches_found": len(v2_caches),
    }


def migrate_all_from_directory(
    image_dir: Path,
    max_features: int = 8192,
    dry_run: bool = False,
) -> Dict:
    """
    Migrate caches for all images in a directory.

    Args:
        image_dir: Directory containing original images
        max_features: Max features setting
        dry_run: If True, only report what would be done

    Returns:
        Dict with migration results
    """
    # Find all image files
    extensions = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
    image_paths = []

    for ext in extensions:
        image_paths.extend(image_dir.glob(f"*{ext}"))
        image_paths.extend(image_dir.glob(f"*{ext.upper()}"))

    image_paths = sorted(set(image_paths))

    if not image_paths:
        return {
            "success": False,
            "message": f"No images found in {image_dir}",
        }

    print(f"Found {len(image_paths)} images in {image_dir}")

    return migrate_cache(image_paths, max_features, dry_run=dry_run)


def list_all_caches() -> List[Dict]:
    """List all existing caches with their metadata."""
    cache_root = get_cache_root()
    if not cache_root.exists():
        return []

    caches = []
    for cache_dir in cache_root.iterdir():
        if not cache_dir.is_dir():
            continue

        db_path = cache_dir / "database.db"
        if not db_path.exists():
            continue

        try:
            import sqlite3
            conn = sqlite3.connect(str(db_path))
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM images")
            n_images = cur.fetchone()[0]
            conn.close()

            # Check for warp caches
            warp_caches = list(cache_dir.glob("warped_*"))

            caches.append({
                "key": cache_dir.name,
                "path": str(cache_dir),
                "n_images": n_images,
                "n_warp_caches": len(warp_caches),
                "is_symlink": cache_dir.is_symlink(),
                "size_mb": sum(f.stat().st_size for f in cache_dir.rglob("*") if f.is_file()) / (1024*1024),
            })
        except Exception as e:
            caches.append({
                "key": cache_dir.name,
                "path": str(cache_dir),
                "error": str(e),
            })

    return caches


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Migrate COLMAP caches from v2 to v3 format"
    )
    parser.add_argument(
        "image_dir",
        nargs="?",
        help="Directory containing original images to migrate cache for"
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=8192,
        help="Max features setting (default: 8192)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show what would be done, don't make changes"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all existing caches"
    )

    args = parser.parse_args()

    if args.list:
        print("Existing caches:")
        print("-" * 60)
        caches = list_all_caches()
        for cache in caches:
            if "error" in cache:
                print(f"  {cache['key']}: ERROR - {cache['error']}")
            else:
                symlink_str = " (symlink)" if cache.get("is_symlink") else ""
                print(f"  {cache['key']}: {cache['n_images']} images, "
                      f"{cache['n_warp_caches']} warp caches, "
                      f"{cache['size_mb']:.1f} MB{symlink_str}")
        print(f"\nTotal: {len(caches)} caches")
        sys.exit(0)

    if not args.image_dir:
        parser.print_help()
        sys.exit(1)

    # Convert Windows path if needed
    image_dir = args.image_dir
    if sys.platform != "win32" and ":\\" in image_dir:
        image_dir = windows_to_wsl_path(image_dir)

    image_dir = Path(image_dir)

    if not image_dir.exists():
        print(f"Error: Directory not found: {image_dir}")
        sys.exit(1)

    result = migrate_all_from_directory(
        image_dir,
        max_features=args.max_features,
        dry_run=args.dry_run,
    )

    print("\nResult:")
    print(json.dumps(result, indent=2))

    sys.exit(0 if result.get("success") else 1)
