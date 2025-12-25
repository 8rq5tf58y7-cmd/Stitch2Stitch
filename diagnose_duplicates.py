#!/usr/bin/env python3
"""
Diagnostic tool to analyze similarity between images.
Run this to see why images are/aren't being marked as duplicates.

Usage:
    python diagnose_duplicates.py /path/to/your/images/folder

This will show the similarity score between all image pairs.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import cv2
import numpy as np
from utils.duplicate_detector import DuplicateDetector


def main():
    if len(sys.argv) < 2:
        print("Usage: python diagnose_duplicates.py /path/to/images/folder [threshold]")
        print("\nExample:")
        print("  python diagnose_duplicates.py ~/Pictures/panorama_photos")
        print("  python diagnose_duplicates.py ~/Pictures/panorama_photos 0.95")
        sys.exit(1)
    
    folder = Path(sys.argv[1])
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.92
    
    if not folder.exists():
        print(f"Error: Folder not found: {folder}")
        sys.exit(1)
    
    # Find all images
    extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
    image_paths = sorted([
        p for p in folder.iterdir() 
        if p.suffix.lower() in extensions
    ])
    
    if len(image_paths) < 2:
        print(f"Found {len(image_paths)} images. Need at least 2 for comparison.")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"DUPLICATE DETECTION DIAGNOSTIC")
    print(f"{'='*60}")
    print(f"Folder: {folder}")
    print(f"Images found: {len(image_paths)}")
    print(f"Threshold: {threshold} ({threshold*100:.0f}% similarity = duplicate)")
    print(f"{'='*60}\n")
    
    # Load thumbnails
    print("Loading images...")
    images = []
    for i, path in enumerate(image_paths):
        print(f"  [{i+1}/{len(image_paths)}] {path.name}")
        img = cv2.imread(str(path))
        if img is not None:
            h, w = img.shape[:2]
            if max(h, w) > 256:
                scale = 256 / max(h, w)
                img = cv2.resize(img, (int(w * scale), int(h * scale)))
            images.append((path, img))
    
    print(f"\nLoaded {len(images)} images successfully.")
    
    # Initialize detector
    detector = DuplicateDetector(similarity_threshold=threshold)
    
    # Compute all pairwise similarities
    print(f"\n{'='*60}")
    print("PAIRWISE SIMILARITY SCORES")
    print(f"{'='*60}")
    print("(Scores >= {:.2f} are marked as DUPLICATES)\n".format(threshold))
    
    n = len(images)
    duplicate_count = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            sim = detector.compute_similarity(images[i][1], images[j][1])
            is_dup = sim >= threshold
            
            if is_dup:
                duplicate_count += 1
                marker = "⚠️  DUPLICATE"
            elif sim >= 0.8:
                marker = "   HIGH"
            elif sim >= 0.6:
                marker = "   medium"
            else:
                marker = "   low"
            
            print(f"{marker}: {images[i][0].name} <-> {images[j][0].name} = {sim:.3f}")
    
    # Run the actual detection
    print(f"\n{'='*60}")
    print("DUPLICATE DETECTION RESULT")
    print(f"{'='*60}")
    
    keep_indices, duplicates = detector.find_duplicates(images)
    
    print(f"\nWith threshold {threshold}:")
    print(f"  - Total images: {len(images)}")
    print(f"  - Unique (KEEP): {len(keep_indices)}")
    print(f"  - Duplicates (REMOVE): {len(images) - len(keep_indices)}")
    
    print(f"\nImages to KEEP:")
    for idx in keep_indices:
        print(f"  ✓ {images[idx][0].name}")
    
    removed = set(range(len(images))) - set(keep_indices)
    if removed:
        print(f"\nImages to REMOVE:")
        for idx in removed:
            print(f"  ✗ {images[idx][0].name}")
    
    # Recommendations
    print(f"\n{'='*60}")
    print("RECOMMENDATIONS")
    print(f"{'='*60}")
    
    if duplicate_count == 0:
        print("✓ No duplicates found - all images are unique!")
        print("   All your images will be used for stitching.")
    elif len(images) - len(keep_indices) == 0:
        print(f"Found {duplicate_count} similar pairs, but smart filtering kept all images.")
        print("   (Adjacent similar images are NOT removed, only true duplicates)")
    else:
        removed_count = len(images) - len(keep_indices)
        print(f"Removing {removed_count} duplicate(s), keeping {len(keep_indices)} unique images.")
        if duplicate_count > removed_count:
            print(f"   ({duplicate_count} pairs detected, but only {removed_count} removed)")
            print("   Note: Sequential burst photos are preserved even if similar.")
        print("\n   If too many are being removed, try:")
        print("   1. Increase threshold: python diagnose_duplicates.py folder 0.98")
        print("   2. Disable duplicate detection in the app settings")


if __name__ == "__main__":
    main()

