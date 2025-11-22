"""Facial expression and emotion detection."""

import time
from typing import Dict, Optional


class FaceFeatureExtractor:
    """
    Extract facial emotions and features from video frames.

    Uses DeepFace for emotion recognition.
    Returns clean dict outputs with emotion classifications and scores.

    Example:
        >>> extractor = FaceFeatureExtractor()
        >>> features = extractor.extract(frame)
        >>> if features.get('face_detected'):
        >>>     print(f"Emotion: {features['dominant_emotion']}")
    """

    def __init__(self, enforce_detection: bool = False):
        """
        Initialize face feature extractor.

        Args:
            enforce_detection: If True, raise error when no face detected
        """
        self.enforce_detection = enforce_detection
        self._deepface_available = False

        # Try to import DeepFace
        try:
            from deepface import DeepFace
            self.DeepFace = DeepFace
            self._deepface_available = True
        except ImportError:
            pass

    def extract(self, frame, timestamp: Optional[float] = None) -> Dict:
        """
        Extract facial emotion features from a video frame.

        Args:
            frame: Video frame (BGR format, numpy array)
            timestamp: Optional timestamp for the frame

        Returns:
            Dictionary with emotion features

        Raises:
            RuntimeError: If DeepFace is not available
        """
        if not self._deepface_available:
            raise RuntimeError(
                "DeepFace is not installed. Install with: pip install deepface"
            )

        if timestamp is None:
            timestamp = time.time()

        try:
            result = self.DeepFace.analyze(
                frame,
                actions=['emotion'],
                enforce_detection=self.enforce_detection,
                silent=True
            )

            # DeepFace returns a list
            if isinstance(result, list):
                result = result[0]

            emotion_scores = result['emotion']
            dominant_emotion = result['dominant_emotion']

            return {
                'face_detected': True,
                'dominant_emotion': dominant_emotion,
                'emotion_scores': emotion_scores,
                'emotion_intensity': emotion_scores[dominant_emotion] / 100.0,
                'timestamp': timestamp
            }

        except Exception as e:
            # No face detected or other error
            return {
                'face_detected': False,
                'dominant_emotion': 'neutral',
                'emotion_intensity': 0.0,
                'timestamp': timestamp,
                'error': str(e)
            }


def extract_facial_emotions(
    frame,
    enforce_detection: bool = False,
    timestamp: Optional[float] = None
) -> Dict:
    """
    Convenience function to extract facial emotions from a frame.

    Args:
        frame: Video frame (BGR format)
        enforce_detection: If True, raise error when no face detected
        timestamp: Optional timestamp

    Returns:
        Dictionary with emotion features
    """
    extractor = FaceFeatureExtractor(enforce_detection=enforce_detection)
    return extractor.extract(frame, timestamp)
