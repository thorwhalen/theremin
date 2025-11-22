"""
audiostream - Audio input feature extraction for Python

A framework-agnostic package for extracting features from audio input streams.
Wraps aubio, librosa for real-time feature extraction.

Useful for pitch detection, onset detection, rhythm analysis, and more.
"""

from .audio_input import AudioInputStream
from .feature_extraction import (
    extract_audio_features,
    extract_pitch_features,
    extract_rhythm_features,
    extract_onset_features,
)

__version__ = "0.1.0"
__all__ = [
    "AudioInputStream",
    "extract_audio_features",
    "extract_pitch_features",
    "extract_rhythm_features",
    "extract_onset_features",
]
