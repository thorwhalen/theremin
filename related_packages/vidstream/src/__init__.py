"""
vidstream - Video feature extraction streams for Python

A framework-agnostic package for extracting features from video streams.
Wraps MediaPipe, DeepFace, and other CV tools with clean dict outputs.

Useful for computer vision applications, pose estimation, emotion detection,
gesture recognition, and more.
"""

from .hand_features import HandFeatureExtractor, extract_hand_landmarks
from .face_features import FaceFeatureExtractor, extract_facial_emotions
from .color_features import ColorFeatureExtractor, extract_dominant_colors
from .pose_features import PoseFeatureExtractor, extract_body_pose

__version__ = "0.1.0"
__all__ = [
    "HandFeatureExtractor",
    "extract_hand_landmarks",
    "FaceFeatureExtractor",
    "extract_facial_emotions",
    "ColorFeatureExtractor",
    "extract_dominant_colors",
    "PoseFeatureExtractor",
    "extract_body_pose",
]
