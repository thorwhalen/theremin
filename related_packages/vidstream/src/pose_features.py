"""Body pose estimation and keypoint extraction."""

import time
from typing import Dict, List, Optional, Tuple


class PoseFeatureExtractor:
    """
    Extract full-body pose keypoints from video frames.

    Uses MediaPipe Pose for pose estimation.
    Returns clean dict outputs with body keypoints and derived features.

    Example:
        >>> extractor = PoseFeatureExtractor()
        >>> features = extractor.extract(frame)
        >>> if features.get('pose_detected'):
        >>>     print(f"Left wrist: {features['left_wrist']}")
    """

    def __init__(
        self,
        detection_confidence: float = 0.5,
        tracking_confidence: float = 0.5
    ):
        """
        Initialize pose feature extractor.

        Args:
            detection_confidence: Minimum detection confidence
            tracking_confidence: Minimum tracking confidence
        """
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence
        self._mediapipe_available = False
        self.mp_pose = None
        self.pose = None

        # Try to import MediaPipe
        try:
            import mediapipe as mp
            self.mp = mp
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                min_detection_confidence=detection_confidence,
                min_tracking_confidence=tracking_confidence
            )
            self._mediapipe_available = True
        except ImportError:
            pass

    def extract(self, frame, timestamp: Optional[float] = None) -> Dict:
        """
        Extract pose features from a video frame.

        Args:
            frame: Video frame (BGR format, numpy array)
            timestamp: Optional timestamp for the frame

        Returns:
            Dictionary with pose features

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
            frame_rgb = frame

        results = self.pose.process(frame_rgb)

        if not results.pose_landmarks:
            return {'timestamp': timestamp, 'pose_detected': False}

        landmarks = results.pose_landmarks.landmark

        # Extract key body points
        features = {
            'timestamp': timestamp,
            'pose_detected': True,
            'left_wrist': self._landmark_to_dict(landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]),
            'right_wrist': self._landmark_to_dict(landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]),
            'left_elbow': self._landmark_to_dict(landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW]),
            'right_elbow': self._landmark_to_dict(landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW]),
            'left_shoulder': self._landmark_to_dict(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]),
            'right_shoulder': self._landmark_to_dict(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]),
            'nose': self._landmark_to_dict(landmarks[self.mp_pose.PoseLandmark.NOSE]),
        }

        # Add visibility scores
        features['avg_visibility'] = sum(
            lm.visibility for lm in landmarks
        ) / len(landmarks)

        return features

    @staticmethod
    def _landmark_to_dict(landmark) -> Dict:
        """Convert MediaPipe landmark to dictionary."""
        return {
            'x': landmark.x,
            'y': landmark.y,
            'z': landmark.z,
            'visibility': landmark.visibility
        }

    def __del__(self):
        """Cleanup MediaPipe resources."""
        if self.pose:
            self.pose.close()


def extract_body_pose(
    frame,
    timestamp: Optional[float] = None
) -> Dict:
    """
    Convenience function to extract body pose from a frame.

    Args:
        frame: Video frame (BGR format)
        timestamp: Optional timestamp

    Returns:
        Dictionary with pose features
    """
    extractor = PoseFeatureExtractor()
    return extractor.extract(frame, timestamp)
