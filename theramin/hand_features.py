"""Hand gesture recognition and feature extraction for theremin."""

import cv2
import mediapipe as mp
import numpy as np
import math
from typing import Dict, Any, Optional, Set, Tuple, Union, List
from collections import namedtuple
from types import MappingProxyType
from theramin.util import data_files

# Path to the gesture recognizer model
gesture_recognizer_path = str(data_files / 'gesture_recognizer.task')

# -------------------------------------------------------------------------------
# Hand Gesture Recognizer
# -------------------------------------------------------------------------------


class HandGestureRecognizer:
    """
    A class to detect hand gestures using MediaPipe's GestureRecognizer.

    Attributes:
        mode (bool): Mode for the hand detection.
        max_hands (int): Maximum number of hands to detect.
        detection_con (float): Minimum detection confidence threshold.
        track_con (float): Minimum tracking confidence threshold.
    """

    def __init__(
        self,
        mode=False,
        *,
        max_hands=2,
        detection_con=0.5,
        track_con=0.5,
        gesture_recognizer_path=gesture_recognizer_path,
    ):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.track_con = track_con

        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            self.mode,
            self.max_hands,
            min_detection_confidence=self.detection_con,
            min_tracking_confidence=self.track_con,
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Initialize Gesture Recognizer
        self.base_options = mp.tasks.BaseOptions(
            model_asset_path=gesture_recognizer_path
        )
        self.vision_running_mode = mp.tasks.vision.RunningMode

        # Set up Gesture Recognizer options
        self.options = mp.tasks.vision.GestureRecognizerOptions(
            base_options=self.base_options, running_mode=self.vision_running_mode.VIDEO
        )
        self.gesture_recognizer = mp.tasks.vision.GestureRecognizer.create_from_options(
            self.options
        )

    def find_hands(self, img):
        """
        Detects hands in the provided image.

        Args:
            img: The input image.

        Returns:
            tuple: (image, hand_detection results)
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hand_detection = self.hands.process(img_rgb)

        return img, hand_detection


# -------------------------------------------------------------------------------
# Hand feature extraction helpers
# -------------------------------------------------------------------------------


def calculate_euclidean_distance(point1, point2):
    """Calculate the Euclidean distance between two 3D points."""
    return math.sqrt(
        (point1[0] - point2[0]) ** 2
        + (point1[1] - point2[1]) ** 2
        + (point1[2] - point2[2]) ** 2
    )


def calculate_angle(p1, p2, p3):
    """Calculate the angle between three points (p2 is the vertex)."""
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1], p3[2] - p2[2]])

    # Calculate the dot product of v1 and v2
    dot_product = np.dot(v1, v2)

    # Calculate the magnitudes (lengths) of v1 and v2
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)

    # Calculate the cosine of the angle
    cos_angle = dot_product / (magnitude_v1 * magnitude_v2)

    # Ensure cos_angle is within the valid range [-1, 1] to avoid errors from arccos
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    # Calculate the angle in radians using arccos
    angle_rad = np.arccos(cos_angle)

    # Convert the angle to degrees
    angle_deg = np.degrees(angle_rad)

    return angle_deg


# -------------------------------------------------------------------------------
# Hand feature extraction
# -------------------------------------------------------------------------------

ALL_HAND_FEATURES = frozenset(
    {
        "handedness",
        "gesture",
        "gesture_score",
        "wrist_position",
        "palm_center",
        "finger_directions",
        "palm_normal",
        "landmarks",
        "openness",
        "index_finger_extension_angle1",
        "index_finger_extension_angle2",
        "thumb_index_distance",
        "is_pinching",
    }
)

# Full list of available feature keys (sorted alphabetically)
HAND_FEATURES_KEYS = sorted(ALL_HAND_FEATURES)

# Default: all features except raw landmarks
DFLT_HAND_FEATURES_INCLUDE = ALL_HAND_FEATURES - {"landmarks"}


def many_single_hand_features(
    hand_landmarks, include=DFLT_HAND_FEATURES_INCLUDE, exclude=()
):
    """
    Extracts multiple hand features from a MediaPipe hand_landmarks object.

    Args:
        hand_landmarks: MediaPipe hand landmarks
        include: Set of features to include
        exclude: Set of features to exclude

    Returns:
        dict: Dictionary of extracted features
    """
    # Shortcuts
    landmark = hand_landmarks.landmark

    def coords(idx):
        lm = landmark[idx]
        return (lm.x, lm.y, lm.z)

    # Landmark indices
    WRIST = mp.solutions.hands.HandLandmark.WRIST
    THUMB_TIP = mp.solutions.hands.HandLandmark.THUMB_TIP
    INDEX_FINGER_TIP = mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP
    MIDDLE_FINGER_TIP = mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP
    RING_FINGER_TIP = mp.solutions.hands.HandLandmark.RING_FINGER_TIP
    PINKY_TIP = mp.solutions.hands.HandLandmark.PINKY_TIP
    MIDDLE_FINGER_MCP = mp.solutions.hands.HandLandmark.MIDDLE_FINGER_MCP
    THUMB_MCP = mp.solutions.hands.HandLandmark.THUMB_MCP
    INDEX_FINGER_MCP = mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP
    INDEX_FINGER_PIP = mp.solutions.hands.HandLandmark.INDEX_FINGER_PIP
    INDEX_FINGER_DIP = mp.solutions.hands.HandLandmark.INDEX_FINGER_DIP

    # Palm center calculation
    def calculate_palm_center(landmarks):
        palm_indices = [0, 1, 5, 9, 13, 17]
        points = [
            (landmarks[i].x, landmarks[i].y, landmarks[i].z) for i in palm_indices
        ]
        cx = sum(p[0] for p in points) / len(points)
        cy = sum(p[1] for p in points) / len(points)
        cz = sum(p[2] for p in points) / len(points)
        return (cx, cy, cz)

    def calculate_finger_directions(landmarks):
        FINGER_TIPS = [4, 8, 12, 16, 20]
        FINGER_BASES = [2, 5, 9, 13, 17]
        return {
            finger: (
                landmarks[tip].x - landmarks[base].x,
                landmarks[tip].y - landmarks[base].y,
                landmarks[tip].z - landmarks[base].z,
            )
            for finger, (tip, base) in enumerate(zip(FINGER_TIPS, FINGER_BASES))
        }

    def calculate_palm_normal(landmarks):
        p0 = landmarks[0]
        p5 = landmarks[5]
        p17 = landmarks[17]
        v1 = (p5.x - p0.x, p5.y - p0.y, p5.z - p0.z)
        v2 = (p17.x - p0.x, p17.y - p0.y, p17.z - p0.z)
        normal = (
            v1[1] * v2[2] - v1[2] * v2[1],
            v1[2] * v2[0] - v1[0] * v2[2],
            v1[0] * v2[1] - v1[1] * v2[0],
        )
        return normal

    requested_features = include - set(exclude)
    out = {}

    # Precompute if needed
    lm_list = (
        [(lm.x, lm.y, lm.z) for lm in landmark]
        if (
            "landmarks" in requested_features
            or any(
                f in requested_features
                for f in (
                    "wrist_position",
                    "palm_center",
                    "finger_directions",
                    "palm_normal",
                )
            )
        )
        else None
    )

    if "landmarks" in requested_features:
        out["landmarks"] = lm_list

    if "wrist_position" in requested_features:
        out["wrist_position"] = lm_list[0]

    if "palm_center" in requested_features:
        out["palm_center"] = calculate_palm_center(landmark)

    if "finger_directions" in requested_features:
        out["finger_directions"] = calculate_finger_directions(landmark)

    if "palm_normal" in requested_features:
        out["palm_normal"] = calculate_palm_normal(landmark)

    if "handedness" in requested_features:
        out["handedness"] = getattr(hand_landmarks, "handedness", None)

    if "gesture" in requested_features:
        out["gesture"] = getattr(hand_landmarks, "gesture", None)

    if "gesture_score" in requested_features:
        out["gesture_score"] = getattr(hand_landmarks, "gesture_score", None)

    # --------- EXTRA FEATURES ----------------------------------

    if any(
        f in requested_features
        for f in (
            "openness",
            "index_finger_extension_angle1",
            "index_finger_extension_angle2",
            "thumb_index_distance",
            "is_pinching",
        )
    ):
        # Recompute palm_center from wrist and middle_finger_mcp if needed
        palm_center = (
            (landmark[WRIST].x + landmark[MIDDLE_FINGER_MCP].x) / 2,
            (landmark[WRIST].y + landmark[MIDDLE_FINGER_MCP].y) / 2,
            (landmark[WRIST].z + landmark[MIDDLE_FINGER_MCP].z) / 2,
        )

    if "openness" in requested_features:
        palm_to_index = calculate_euclidean_distance(
            palm_center, coords(INDEX_FINGER_TIP)
        )
        palm_to_middle = calculate_euclidean_distance(
            palm_center, coords(MIDDLE_FINGER_TIP)
        )
        palm_to_ring = calculate_euclidean_distance(
            palm_center, coords(RING_FINGER_TIP)
        )
        palm_to_pinky = calculate_euclidean_distance(palm_center, coords(PINKY_TIP))
        out['openness'] = (
            palm_to_index + palm_to_middle + palm_to_ring + palm_to_pinky
        ) / 4

    if "index_finger_extension_angle1" in requested_features:
        out['index_finger_extension_angle1'] = calculate_angle(
            coords(WRIST), coords(INDEX_FINGER_MCP), coords(INDEX_FINGER_PIP)
        )

    if "index_finger_extension_angle2" in requested_features:
        out['index_finger_extension_angle2'] = calculate_angle(
            coords(INDEX_FINGER_MCP), coords(INDEX_FINGER_PIP), coords(INDEX_FINGER_DIP)
        )

    if (
        "thumb_index_distance" in requested_features
        or "is_pinching" in requested_features
    ):
        thumb_index_dist = calculate_euclidean_distance(
            coords(THUMB_TIP), coords(INDEX_FINGER_TIP)
        )
        if "thumb_index_distance" in requested_features:
            out['thumb_index_distance'] = thumb_index_dist
        if "is_pinching" in requested_features:
            out['is_pinching'] = thumb_index_dist < 0.1  # Adjustable threshold

    return out


def many_hand_features(hand_detection, include=DFLT_HAND_FEATURES_INCLUDE, exclude=()):
    """
    Calls many_single_hand_features for each hand in the detection.
    Returns a dictionary corresponding to the many_single_hand_features output,
    but with `l_` and `r_` prefixes for left and right hands.

    Args:
        hand_detection: MediaPipe hand detection results
        include: Set of features to include
        exclude: Set of features to exclude

    Returns:
        dict: Dictionary of extracted features with hand prefixes
    """
    if not hand_detection.multi_hand_landmarks:
        return {}

    hands = {}

    for idx, hand_landmarks in enumerate(hand_detection.multi_hand_landmarks):
        handedness = hand_detection.multi_handedness[idx].classification[0].label
        hand_features = many_single_hand_features(
            hand_landmarks, include=include, exclude=exclude
        )
        if handedness == "Left":
            hands["l_"] = hand_features
        elif handedness == "Right":
            hands["r_"] = hand_features

    # Merge the dictionaries and add prefixes
    merged_hands = {}
    for prefix, features in hands.items():
        for key, value in features.items():
            merged_hands[f"{prefix}{key}"] = value

    return merged_hands


# Dictionary of available hand feature extractors
hand_feature_funcs = {
    "many_hand_features": many_hand_features,
}
