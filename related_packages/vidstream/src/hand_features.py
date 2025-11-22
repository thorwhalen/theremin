"""Hand feature extraction using MediaPipe."""

import time
from typing import Dict, List, Optional, Tuple
import math


class HandFeatureExtractor:
    """
    Extract hand landmarks and features from video frames.

    Uses MediaPipe Hands for real-time hand tracking.
    Returns clean dict outputs with hand positions, gestures, and derived features.

    Example:
        >>> extractor = HandFeatureExtractor()
        >>> features = extractor.extract(frame)
        >>> if features:
        >>>     print(f"Right hand at: {features.get('r_wrist_position')}")
    """

    def __init__(
        self,
        max_hands: int = 2,
        detection_confidence: float = 0.5,
        tracking_confidence: float = 0.5
    ):
        """
        Initialize hand feature extractor.

        Args:
            max_hands: Maximum number of hands to detect
            detection_confidence: Minimum detection confidence
            tracking_confidence: Minimum tracking confidence
        """
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence
        self._mediapipe_available = False
        self.mp_hands = None
        self.hands = None

        # Try to import MediaPipe
        try:
            import mediapipe as mp
            self.mp = mp
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=max_hands,
                min_detection_confidence=detection_confidence,
                min_tracking_confidence=tracking_confidence
            )
            self._mediapipe_available = True
        except ImportError:
            pass

    def extract(self, frame, timestamp: Optional[float] = None) -> Dict:
        """
        Extract hand features from a video frame.

        Args:
            frame: Video frame (BGR format, numpy array)
            timestamp: Optional timestamp for the frame

        Returns:
            Dictionary with hand features (empty if no hands detected)

        Raises:
            RuntimeError: If MediaPipe is not available
        """
        if not self._mediapipe_available:
            raise RuntimeError(
                "MediaPipe is not installed. Install with: pip install mediapipe"
            )

        if timestamp is None:
            timestamp = time.time()

        # Process frame (MediaPipe expects RGB)
        try:
            import cv2
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except ImportError:
            # Assume frame is already RGB
            frame_rgb = frame

        results = self.hands.process(frame_rgb)

        if not results.multi_hand_landmarks:
            return {'timestamp': timestamp, 'has_hands': False}

        features = {'timestamp': timestamp, 'has_hands': True}

        # Extract features for each detected hand
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            handedness = results.multi_handedness[idx].classification[0].label
            prefix = 'l_' if handedness == 'Left' else 'r_'

            # Extract basic landmark positions
            landmarks = hand_landmarks.landmark
            wrist = landmarks[0]

            features[f'{prefix}wrist_position'] = (wrist.x, wrist.y, wrist.z)

            # Palm center
            palm_indices = [0, 1, 5, 9, 13, 17]
            palm_x = sum(landmarks[i].x for i in palm_indices) / len(palm_indices)
            palm_y = sum(landmarks[i].y for i in palm_indices) / len(palm_indices)
            palm_z = sum(landmarks[i].z for i in palm_indices) / len(palm_indices)
            features[f'{prefix}palm_center'] = (palm_x, palm_y, palm_z)

            # Finger tips
            tip_indices = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
            features[f'{prefix}finger_tips'] = [
                (landmarks[i].x, landmarks[i].y, landmarks[i].z)
                for i in tip_indices
            ]

            # Hand openness (average distance from palm to fingertips)
            distances = [
                self._euclidean_distance(
                    (palm_x, palm_y, palm_z),
                    (landmarks[i].x, landmarks[i].y, landmarks[i].z)
                )
                for i in tip_indices
            ]
            features[f'{prefix}openness'] = sum(distances) / len(distances)

        return features

    @staticmethod
    def _euclidean_distance(p1: Tuple, p2: Tuple) -> float:
        """Calculate 3D Euclidean distance."""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

    def __del__(self):
        """Cleanup MediaPipe resources."""
        if self.hands:
            self.hands.close()


def extract_hand_landmarks(
    frame,
    max_hands: int = 2,
    timestamp: Optional[float] = None
) -> Dict:
    """
    Convenience function to extract hand features from a frame.

    Args:
        frame: Video frame (BGR format)
        max_hands: Maximum number of hands to detect
        timestamp: Optional timestamp

    Returns:
        Dictionary with hand features
    """
    extractor = HandFeatureExtractor(max_hands=max_hands)
    return extractor.extract(frame, timestamp)
