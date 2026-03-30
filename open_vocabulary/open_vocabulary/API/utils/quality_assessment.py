"""
Quality Assessment Module for Person Tracking

This module provides functions to assess the quality of person detections
based on visibility (keypoints) and distance (depth information).
"""

import numpy as np
from typing import Optional, Tuple, Dict

# COCO keypoint indices
KEYPOINT_NAMES = [
    'nose',           # 0
    'left_eye',       # 1
    'right_eye',      # 2
    'left_ear',       # 3
    'right_ear',      # 4
    'left_shoulder',  # 5
    'right_shoulder', # 6
    'left_elbow',     # 7
    'right_elbow',    # 8
    'left_wrist',     # 9
    'right_wrist',    # 10
    'left_hip',       # 11
    'right_hip',      # 12
    'left_knee',      # 13
    'right_knee',     # 14
    'left_ankle',     # 15
    'right_ankle'     # 16
]

# Keypoint weights for visibility calculation
# Torso keypoints (shoulders, hips) are most important
KEYPOINT_WEIGHTS = {
    0: 0.5,   # nose
    1: 0.3, 2: 0.3,   # eyes
    3: 0.2, 4: 0.2,   # ears
    5: 1.0, 6: 1.0,   # shoulders (CRITICAL)
    7: 0.6, 8: 0.6,   # elbows
    9: 0.4, 10: 0.4,  # wrists
    11: 1.0, 12: 1.0, # hips (CRITICAL)
    13: 0.6, 14: 0.6, # knees
    15: 0.4, 16: 0.4  # ankles
}

# Torso keypoints are most important for person identification
TORSO_KEYPOINTS = [5, 6, 11, 12]  # shoulders and hips


def calculate_visibility_score(keypoints: np.ndarray,
                               confidence_threshold: float = 0.3) -> float:
    """
    Calculate visibility score based on COCO keypoints.

    Args:
        keypoints: Array of shape (17, 3) containing [x, y, confidence] for each keypoint
        confidence_threshold: Minimum confidence to consider a keypoint visible

    Returns:
        Visibility score between 0.0 and 1.0
        - > 0.7: Excellent (torso fully visible)
        - 0.5-0.7: Good (slight occlusion)
        - 0.4-0.5: Fair (partial occlusion)
        - < 0.4: Poor (severe occlusion, should reject)
    """
    if keypoints is None or len(keypoints) == 0:
        return 0.0

    # Ensure keypoints is numpy array
    if not isinstance(keypoints, np.ndarray):
        keypoints = np.array(keypoints)

    # Handle different keypoint formats
    if keypoints.ndim == 1:
        # Flatten format: [x1, y1, conf1, x2, y2, conf2, ...]
        keypoints = keypoints.reshape(-1, 3)

    total_weight = 0.0
    visible_weight = 0.0

    for i in range(min(17, len(keypoints))):
        weight = KEYPOINT_WEIGHTS.get(i, 0.5)
        total_weight += weight

        # Check if keypoint is visible (confidence above threshold)
        if len(keypoints[i]) >= 3 and keypoints[i][2] >= confidence_threshold:
            visible_weight += weight

    if total_weight == 0:
        return 0.0

    visibility_score = visible_weight / total_weight

    # Bonus for torso visibility (critical for person identification)
    torso_visible_count = 0
    for idx in TORSO_KEYPOINTS:
        if idx < len(keypoints) and len(keypoints[idx]) >= 3:
            if keypoints[idx][2] >= confidence_threshold:
                torso_visible_count += 1

    # If all 4 torso keypoints are visible, add bonus
    if torso_visible_count == 4:
        visibility_score = min(1.0, visibility_score * 1.1)
    # If less than 2 torso keypoints visible, apply penalty
    elif torso_visible_count < 2:
        visibility_score *= 0.7

    return float(np.clip(visibility_score, 0.0, 1.0))


def calculate_distance_score(distance_m: float,
                            optimal_dist: float = 1.5,
                            max_dist: float = 3.0) -> float:
    """
    Calculate quality score based on distance from camera.

    Args:
        distance_m: Distance in meters
        optimal_dist: Optimal distance for feature extraction (default 1.5m)
        max_dist: Maximum acceptable distance (default 3.0m)

    Returns:
        Distance score between 0.0 and 1.0
        - 0.5-2.0m: Excellent (1.0)
        - 2.0-2.5m: Good (0.8)
        - 2.5-3.0m: Fair (0.6)
        - > 3.0m: Reject (0.0)
    """
    if distance_m is None or distance_m <= 0:
        return 0.0

    # Too far, reject
    if distance_m > max_dist:
        return 0.0

    # Optimal range: 0.5m to 2.0m
    if 0.5 <= distance_m <= 2.0:
        return 1.0

    # Too close (< 0.5m)
    if distance_m < 0.5:
        return 0.5

    # Gradual decline from 2.0m to 3.0m
    if 2.0 < distance_m <= 2.5:
        return 0.8
    elif 2.5 < distance_m <= max_dist:
        # Linear interpolation from 0.8 to 0.6
        return 0.6 + (max_dist - distance_m) / (max_dist - 2.5) * 0.2

    return 0.0


def calculate_overall_quality(visibility_score: float,
                              distance_score: float,
                              detection_conf: float,
                              visibility_weight: float = 0.4,
                              distance_weight: float = 0.4,
                              conf_weight: float = 0.2) -> float:
    """
    Calculate overall quality score combining multiple factors.

    Args:
        visibility_score: Visibility score from keypoints (0-1)
        distance_score: Distance score (0-1)
        detection_conf: Detection confidence (0-1)
        visibility_weight: Weight for visibility (default 0.4)
        distance_weight: Weight for distance (default 0.4)
        conf_weight: Weight for detection confidence (default 0.2)

    Returns:
        Overall quality score between 0.0 and 1.0
        Minimum acceptable threshold: 0.6
    """
    # Normalize weights
    total_weight = visibility_weight + distance_weight + conf_weight
    visibility_weight /= total_weight
    distance_weight /= total_weight
    conf_weight /= total_weight

    overall_score = (
        visibility_score * visibility_weight +
        distance_score * distance_weight +
        detection_conf * conf_weight
    )

    return float(np.clip(overall_score, 0.0, 1.0))


def should_accept_feature(visibility_score: float,
                         distance_m: Optional[float],
                         quality_score: float,
                         min_visibility: float = 0.4,
                         max_distance: float = 3.0,
                         min_quality: float = 0.6) -> Tuple[bool, str]:
    """
    Determine if a feature should be accepted based on quality criteria.

    Args:
        visibility_score: Visibility score from keypoints
        distance_m: Distance in meters (None if not available)
        quality_score: Overall quality score
        min_visibility: Minimum visibility threshold
        max_distance: Maximum distance threshold
        min_quality: Minimum quality threshold

    Returns:
        Tuple of (should_accept, reason)
    """
    # Check visibility
    if visibility_score < min_visibility:
        return False, f"Low visibility: {visibility_score:.2f} < {min_visibility}"

    # Check distance
    if distance_m is not None and distance_m > max_distance:
        return False, f"Too far: {distance_m:.2f}m > {max_distance}m"

    # Check overall quality
    if quality_score < min_quality:
        return False, f"Low quality: {quality_score:.2f} < {min_quality}"

    return True, "Accepted"


def get_quality_weight(quality_score: float) -> float:
    """
    Get matching weight based on quality score.

    High quality features get higher weight in matching.

    Args:
        quality_score: Quality score (0-1)

    Returns:
        Weight multiplier for matching
        - High quality (>0.8): 1.2x
        - Medium quality (0.6-0.8): 1.0x
        - Low quality (<0.6): 0.5x
    """
    if quality_score >= 0.8:
        return 1.2
    elif quality_score >= 0.6:
        return 1.0
    else:
        return 0.5
