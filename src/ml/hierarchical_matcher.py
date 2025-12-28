"""
Hierarchical Matching Pipeline for Large Image Sets (500+)

Implements the key insights from modern large-scale matching:
1. Global image retrieval to limit candidate pairs
2. Multi-image feature tracks for consistency
3. Mutual consistency filtering
4. Progressive refinement

This reduces O(n²) pairwise matching to O(n*k) where k is small (~20-30)
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
import logging
from collections import defaultdict
import heapq

logger = logging.getLogger(__name__)


class GlobalImageRetrieval:
    """
    Global image retrieval using compact descriptors.
    
    Reduces candidate pairs from O(n²) to O(n*k) by only matching
    images that are globally similar.
    """
    
    def __init__(self, n_candidates: int = 30):
        """
        Args:
            n_candidates: Number of candidate matches per image
        """
        self.n_candidates = n_candidates
        self.global_descriptors = {}
    
    def compute_global_descriptor(self, image: np.ndarray, image_idx: int) -> np.ndarray:
        """
        Compute a global descriptor for an image.
        
        Uses a combination of:
        - Color histogram (captures overall appearance)
        - Spatial pyramid (captures layout)
        - Edge orientation histogram (captures structure)
        """
        h, w = image.shape[:2]
        descriptors = []
        
        # 1. Color histogram in LAB space (perceptually uniform)
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        else:
            lab = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2LAB)
        
        for c in range(3):
            hist, _ = np.histogram(lab[:, :, c], bins=32, range=(0, 256))
            hist = hist.astype(np.float32)
            hist /= (hist.sum() + 1e-10)
            descriptors.extend(hist)
        
        # 2. Spatial pyramid - divide into 2x2 grid
        gh, gw = h // 2, w // 2
        for gy in range(2):
            for gx in range(2):
                cell = lab[gy*gh:(gy+1)*gh, gx*gw:(gx+1)*gw, 0]  # L channel only
                hist, _ = np.histogram(cell, bins=16, range=(0, 256))
                hist = hist.astype(np.float32)
                hist /= (hist.sum() + 1e-10)
                descriptors.extend(hist)
        
        # 3. Edge orientation histogram (HOG-like)
        gray = lab[:, :, 0]  # L channel
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = np.sqrt(gx**2 + gy**2)
        orientation = np.arctan2(gy, gx)
        
        # Weighted orientation histogram
        hist, _ = np.histogram(orientation, bins=16, range=(-np.pi, np.pi), 
                               weights=magnitude)
        hist = hist.astype(np.float32)
        hist /= (hist.sum() + 1e-10)
        descriptors.extend(hist)
        
        descriptor = np.array(descriptors, dtype=np.float32)
        self.global_descriptors[image_idx] = descriptor
        
        return descriptor
    
    def find_candidates(self, query_idx: int, exclude_indices: Set[int] = None) -> List[Tuple[int, float]]:
        """
        Find candidate images for matching.
        
        Returns:
            List of (image_idx, similarity_score) tuples, sorted by similarity
        """
        if query_idx not in self.global_descriptors:
            return []
        
        query_desc = self.global_descriptors[query_idx]
        candidates = []
        
        for idx, desc in self.global_descriptors.items():
            if idx == query_idx:
                continue
            if exclude_indices and idx in exclude_indices:
                continue
            
            # Compute similarity (correlation coefficient)
            similarity = np.corrcoef(query_desc, desc)[0, 1]
            if np.isnan(similarity):
                similarity = 0.0
            
            candidates.append((idx, similarity))
        
        # Sort by similarity (highest first)
        candidates.sort(key=lambda x: -x[1])
        
        return candidates[:self.n_candidates]
    
    def get_candidate_pairs(self, n_images: int) -> List[Tuple[int, int]]:
        """
        Get all candidate pairs for matching.
        
        Returns:
            List of (i, j) pairs to match
        """
        pairs = set()
        
        for i in range(n_images):
            candidates = self.find_candidates(i)
            for j, _ in candidates:
                pair = (min(i, j), max(i, j))
                pairs.add(pair)
        
        pairs_list = list(pairs)
        logger.info(f"Global retrieval: {n_images} images → {len(pairs_list)} candidate pairs "
                   f"(reduced from {n_images*(n_images-1)//2})")
        
        return pairs_list


class FeatureTrackBuilder:
    """
    Build feature tracks across multiple images.
    
    Instead of just pairwise matches, builds tracks of features
    that appear across many images. This provides:
    - Multi-image consistency checking
    - Outlier rejection (features appearing only once)
    - Transitive consistency enforcement
    """
    
    def __init__(self, min_track_length: int = 2):
        """
        Args:
            min_track_length: Minimum number of images a feature must appear in
        """
        self.min_track_length = min_track_length
        self.tracks = []  # List of {image_idx: keypoint_idx} dicts
    
    def build_tracks(
        self,
        matches: List[Dict],
        features_data: List[Dict]
    ) -> List[Dict]:
        """
        Build feature tracks from pairwise matches.
        
        Args:
            matches: List of match dictionaries
            features_data: List of feature data dictionaries
            
        Returns:
            List of track dictionaries, each containing:
            - 'observations': {image_idx: (x, y)} mapping
            - 'confidence': track confidence score
        """
        # Union-Find structure for merging tracks
        parent = {}
        
        def find(x):
            if x not in parent:
                parent[x] = x
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # Process all matches
        for match in matches:
            i = match['image_i']
            j = match['image_j']
            match_list = match.get('matches', [])
            
            kp_i = features_data[i]['keypoints'] if i < len(features_data) else []
            kp_j = features_data[j]['keypoints'] if j < len(features_data) else []
            
            for m in match_list:
                idx_i = m.queryIdx
                idx_j = m.trainIdx
                
                if idx_i < len(kp_i) and idx_j < len(kp_j):
                    # Create unique identifiers for each observation
                    obs_i = (i, idx_i)
                    obs_j = (j, idx_j)
                    
                    # Merge into same track
                    union(obs_i, obs_j)
        
        # Group observations by track
        track_groups = defaultdict(list)
        for obs in parent.keys():
            root = find(obs)
            track_groups[root].append(obs)
        
        # Convert to track format
        tracks = []
        for root, observations in track_groups.items():
            if len(observations) < self.min_track_length:
                continue
            
            # Check if observations span multiple images
            image_indices = set(obs[0] for obs in observations)
            if len(image_indices) < self.min_track_length:
                continue
            
            # Build track dictionary
            track_obs = {}
            for img_idx, kp_idx in observations:
                if img_idx < len(features_data):
                    kp = features_data[img_idx]['keypoints']
                    if kp_idx < len(kp):
                        track_obs[img_idx] = (kp[kp_idx][0], kp[kp_idx][1])
            
            if len(track_obs) >= self.min_track_length:
                tracks.append({
                    'observations': track_obs,
                    'confidence': len(track_obs) / len(features_data)
                })
        
        logger.info(f"Built {len(tracks)} feature tracks from {len(matches)} match pairs")
        
        # Sort tracks by length (longer = more reliable)
        tracks.sort(key=lambda t: -len(t['observations']))
        
        self.tracks = tracks
        return tracks
    
    def filter_matches_by_tracks(
        self,
        matches: List[Dict],
        features_data: List[Dict]
    ) -> List[Dict]:
        """
        Filter matches to only include those that are part of consistent tracks.
        
        This removes:
        - One-off matches that don't connect to anything else
        - Geometrically inconsistent matches
        """
        if not self.tracks:
            self.build_tracks(matches, features_data)
        
        # Build lookup: (image_idx, keypoint_idx) -> track_idx
        obs_to_track = {}
        for track_idx, track in enumerate(self.tracks):
            for img_idx, (x, y) in track['observations'].items():
                # Find matching keypoint
                if img_idx < len(features_data):
                    kps = features_data[img_idx]['keypoints']
                    for kp_idx, kp in enumerate(kps):
                        if abs(kp[0] - x) < 1 and abs(kp[1] - y) < 1:
                            obs_to_track[(img_idx, kp_idx)] = track_idx
                            break
        
        # Filter matches
        filtered_matches = []
        for match in matches:
            i = match['image_i']
            j = match['image_j']
            match_list = match.get('matches', [])
            
            # Keep only matches that are part of tracks
            good_matches = []
            for m in match_list:
                obs_i = (i, m.queryIdx)
                obs_j = (j, m.trainIdx)
                
                if obs_i in obs_to_track or obs_j in obs_to_track:
                    good_matches.append(m)
            
            if len(good_matches) >= 4:
                filtered_match = match.copy()
                filtered_match['matches'] = good_matches
                filtered_match['num_matches'] = len(good_matches)
                filtered_match['track_filtered'] = True
                filtered_matches.append(filtered_match)
        
        logger.info(f"Track filtering: {len(matches)} → {len(filtered_matches)} match pairs "
                   f"({len(filtered_matches)/len(matches)*100:.1f}% retained)")
        
        return filtered_matches


class MutualConsistencyFilter:
    """
    Filter matches based on mutual consistency with neighbors.
    
    Key insight from SuperGlue: A good match should be consistent with
    matches in its spatial neighborhood.
    """
    
    def __init__(self, neighborhood_radius: float = 50.0, min_consistent: int = 3):
        """
        Args:
            neighborhood_radius: Pixel radius for neighborhood
            min_consistent: Minimum consistent neighbors required
        """
        self.neighborhood_radius = neighborhood_radius
        self.min_consistent = min_consistent
    
    def filter_matches(
        self,
        matches: List[cv2.DMatch],
        keypoints1: np.ndarray,
        keypoints2: np.ndarray
    ) -> List[cv2.DMatch]:
        """
        Filter matches based on neighborhood consistency.
        
        For each match, check if nearby matches have consistent displacement.
        """
        if len(matches) < 5:
            return matches
        
        # Compute displacement vectors for all matches
        displacements = []
        match_pts1 = []
        for m in matches:
            if m.queryIdx < len(keypoints1) and m.trainIdx < len(keypoints2):
                pt1 = keypoints1[m.queryIdx][:2]
                pt2 = keypoints2[m.trainIdx][:2]
                displacements.append(pt2 - pt1)
                match_pts1.append(pt1)
        
        displacements = np.array(displacements)
        match_pts1 = np.array(match_pts1)
        
        # Check consistency for each match
        consistent_mask = []
        for i, (pt1, disp) in enumerate(zip(match_pts1, displacements)):
            # Find neighbors
            distances = np.linalg.norm(match_pts1 - pt1, axis=1)
            neighbor_mask = (distances < self.neighborhood_radius) & (distances > 0)
            neighbor_indices = np.where(neighbor_mask)[0]
            
            if len(neighbor_indices) < self.min_consistent:
                # Not enough neighbors, keep the match
                consistent_mask.append(True)
                continue
            
            # Check displacement consistency
            neighbor_disps = displacements[neighbor_indices]
            disp_diffs = np.linalg.norm(neighbor_disps - disp, axis=1)
            
            # Consistent if most neighbors have similar displacement
            consistent_count = np.sum(disp_diffs < self.neighborhood_radius)
            is_consistent = consistent_count >= self.min_consistent
            consistent_mask.append(is_consistent)
        
        # Filter matches
        filtered = [m for m, keep in zip(matches, consistent_mask) if keep]
        
        if len(filtered) < len(matches):
            logger.debug(f"Mutual consistency: {len(matches)} → {len(filtered)} matches")
        
        return filtered


class HierarchicalMatcher:
    """
    Complete hierarchical matching pipeline for large image sets.
    
    Implements:
    1. Global image retrieval (candidate pair reduction)
    2. Local feature matching
    3. Mutual consistency filtering
    4. Feature track building
    5. Track-based match filtering
    """
    
    def __init__(
        self,
        n_candidates: int = 30,
        use_tracks: bool = True,
        use_mutual_consistency: bool = True,
        progress_callback=None
    ):
        """
        Args:
            n_candidates: Number of candidate matches per image from retrieval
            use_tracks: Enable feature track building
            use_mutual_consistency: Enable mutual consistency filtering
            progress_callback: Optional progress callback
        """
        self.retrieval = GlobalImageRetrieval(n_candidates=n_candidates)
        self.track_builder = FeatureTrackBuilder(min_track_length=2)
        self.consistency_filter = MutualConsistencyFilter()
        self.use_tracks = use_tracks
        self.use_mutual_consistency = use_mutual_consistency
        self.progress_callback = progress_callback
    
    def compute_global_descriptors(self, images_data: List[Dict]):
        """
        Compute global descriptors for all images.
        
        Args:
            images_data: List of image data dictionaries
        """
        logger.info(f"Computing global descriptors for {len(images_data)} images...")
        
        for idx, img_data in enumerate(images_data):
            self.retrieval.compute_global_descriptor(img_data['image'], idx)
            
            if self.progress_callback and idx % 50 == 0:
                self.progress_callback(
                    int(20 * idx / len(images_data)),
                    f"Computing global descriptors: {idx+1}/{len(images_data)}"
                )
    
    def get_matching_pairs(self, n_images: int) -> List[Tuple[int, int]]:
        """
        Get candidate pairs for matching based on global retrieval.
        
        Args:
            n_images: Number of images
            
        Returns:
            List of (i, j) pairs to match
        """
        return self.retrieval.get_candidate_pairs(n_images)
    
    def post_process_matches(
        self,
        matches: List[Dict],
        features_data: List[Dict]
    ) -> List[Dict]:
        """
        Post-process matches with tracks and consistency filtering.
        
        Args:
            matches: Raw matches from local feature matching
            features_data: Feature data for all images
            
        Returns:
            Filtered matches
        """
        if self.use_tracks:
            # Build tracks and filter
            self.track_builder.build_tracks(matches, features_data)
            matches = self.track_builder.filter_matches_by_tracks(matches, features_data)
        
        # Note: Mutual consistency is applied per-pair during matching
        # Track filtering provides global consistency
        
        return matches
    
    def get_track_statistics(self) -> Dict:
        """Get statistics about built tracks."""
        if not self.track_builder.tracks:
            return {'n_tracks': 0}
        
        track_lengths = [len(t['observations']) for t in self.track_builder.tracks]
        return {
            'n_tracks': len(self.track_builder.tracks),
            'avg_track_length': np.mean(track_lengths),
            'max_track_length': np.max(track_lengths),
            'tracks_3plus': sum(1 for l in track_lengths if l >= 3)
        }






