"""Real-time audio feature extraction."""

import time
from typing import Dict, Optional
import numpy as np


def extract_pitch_features(
    audio_chunk: np.ndarray,
    sample_rate: int = 44100,
    timestamp: Optional[float] = None
) -> Dict:
    """
    Extract pitch-related features from audio chunk.

    Uses aubio for pitch detection.

    Args:
        audio_chunk: Audio samples (NumPy array)
        sample_rate: Sample rate in Hz
        timestamp: Optional timestamp

    Returns:
        Dictionary with pitch features
    """
    if timestamp is None:
        timestamp = time.time()

    try:
        import aubio
    except ImportError:
        return {
            'error': 'aubio not installed',
            'timestamp': timestamp
        }

    # Ensure audio is in correct format
    if audio_chunk.ndim > 1:
        audio_chunk = audio_chunk.flatten()

    # Convert to float32 if needed
    if audio_chunk.dtype != np.float32:
        audio_chunk = audio_chunk.astype(np.float32)

    # Create pitch detector
    hop_size = len(audio_chunk)
    pitch_detector = aubio.pitch("default", hop_size * 2, hop_size, sample_rate)
    pitch_detector.set_unit("Hz")

    # Detect pitch
    pitch = pitch_detector(audio_chunk)[0]
    confidence = pitch_detector.get_confidence()

    return {
        'pitch': float(pitch),
        'pitch_confidence': float(confidence),
        'has_pitch': pitch > 0 and confidence > 0.8,
        'timestamp': timestamp
    }


def extract_onset_features(
    audio_chunk: np.ndarray,
    sample_rate: int = 44100,
    timestamp: Optional[float] = None
) -> Dict:
    """
    Extract onset detection features from audio chunk.

    Uses aubio for onset detection.

    Args:
        audio_chunk: Audio samples (NumPy array)
        sample_rate: Sample rate in Hz
        timestamp: Optional timestamp

    Returns:
        Dictionary with onset features
    """
    if timestamp is None:
        timestamp = time.time()

    try:
        import aubio
    except ImportError:
        return {
            'error': 'aubio not installed',
            'timestamp': timestamp
        }

    # Ensure audio is in correct format
    if audio_chunk.ndim > 1:
        audio_chunk = audio_chunk.flatten()

    if audio_chunk.dtype != np.float32:
        audio_chunk = audio_chunk.astype(np.float32)

    # Create onset detector
    hop_size = len(audio_chunk)
    onset_detector = aubio.onset("default", hop_size * 2, hop_size, sample_rate)

    # Detect onset
    onset = onset_detector(audio_chunk)[0]

    return {
        'onset_detected': bool(onset),
        'timestamp': timestamp
    }


def extract_rhythm_features(
    audio_chunk: np.ndarray,
    sample_rate: int = 44100,
    timestamp: Optional[float] = None
) -> Dict:
    """
    Extract rhythm and tempo features from audio chunk.

    Uses aubio for tempo/beat detection.

    Args:
        audio_chunk: Audio samples (NumPy array)
        sample_rate: Sample rate in Hz
        timestamp: Optional timestamp

    Returns:
        Dictionary with rhythm features
    """
    if timestamp is None:
        timestamp = time.time()

    try:
        import aubio
    except ImportError:
        return {
            'error': 'aubio not installed',
            'timestamp': timestamp
        }

    # Ensure audio is in correct format
    if audio_chunk.ndim > 1:
        audio_chunk = audio_chunk.flatten()

    if audio_chunk.dtype != np.float32:
        audio_chunk = audio_chunk.astype(np.float32)

    # Create tempo detector
    hop_size = len(audio_chunk)
    tempo_detector = aubio.tempo("default", hop_size * 2, hop_size, sample_rate)

    # Detect beat
    is_beat = tempo_detector(audio_chunk)[0]
    bpm = tempo_detector.get_bpm()

    return {
        'beat_detected': bool(is_beat),
        'tempo_bpm': float(bpm),
        'timestamp': timestamp
    }


def extract_audio_features(
    audio_chunk: np.ndarray,
    sample_rate: int = 44100,
    timestamp: Optional[float] = None,
    include_pitch: bool = True,
    include_onset: bool = True,
    include_rhythm: bool = False
) -> Dict:
    """
    Extract comprehensive audio features from audio chunk.

    Args:
        audio_chunk: Audio samples (NumPy array)
        sample_rate: Sample rate in Hz
        timestamp: Optional timestamp
        include_pitch: Extract pitch features
        include_onset: Extract onset features
        include_rhythm: Extract rhythm features

    Returns:
        Dictionary with all requested features
    """
    if timestamp is None:
        timestamp = time.time()

    # Calculate RMS amplitude
    rms = float(np.sqrt(np.mean(audio_chunk ** 2)))

    features = {
        'amplitude': rms,
        'voice_activity': rms > 0.01,
        'timestamp': timestamp
    }

    # Add pitch features
    if include_pitch:
        pitch_features = extract_pitch_features(audio_chunk, sample_rate, timestamp)
        features.update(pitch_features)

    # Add onset features
    if include_onset:
        onset_features = extract_onset_features(audio_chunk, sample_rate, timestamp)
        features.update(onset_features)

    # Add rhythm features
    if include_rhythm:
        rhythm_features = extract_rhythm_features(audio_chunk, sample_rate, timestamp)
        features.update(rhythm_features)

    return features
