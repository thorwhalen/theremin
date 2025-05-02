import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Union, Any, Callable

from i2 import partialx, Sig as Signature

from theramin.util import data_files, format_dict_values
from hum.pyo_util import Synth

# Path to the gesture recognizer model
gesture_recognizer_path = str(data_files / 'gesture_recognizer.task')


# -------------------------------------------------------------------------------
# Utils
# -------------------------------------------------------------------------------


def obfuscate_args(func, keep_args):

    func_sig = Signature(func)
    if not all([arg in keep_args for arg in func_sig.names[: len(keep_args)]]):
        raise ValueError("keep_args must be in the beginning of Sig(foo).names")
    defaults_of_other_args = {
        k: v for k, v in func_sig.defaults.items() if k not in keep_args
    }
    return partialx(func, **defaults_of_other_args, _rm_partialize=True)


from typing import Callable, Dict, Any, TypeVar

# Define a type variable for the return type to maintain type hinting
T = TypeVar('T')


def resolve_object(
    obj: Union[str, T],
    *,
    object_map: Dict[str, T],
    expected_type: type = None,  # Optional: Enforce a type
    error_message: str = None,
) -> T:
    """
    Resolves an object by either returning it directly if it's of the correct type,
    or looking it up in a mapping if it's a string.

    Args:
        obj: The object to resolve. Can be a string (to be looked up in object_map)
             or the object itself (if it's already of type T).
        object_map: A dictionary mapping strings to objects of type T.
        expected_type: (Optional) The expected type of the resolved object.
                       If provided, raises a TypeError if the resolved object
                       is not of this type.
        error_message: (Optional) A custom error message to use if a ValueError
                       or TypeError is raised. If None, a default message is used.

    Returns:
        The resolved object of type T.

    Raises:
        TypeError: If obj is not a string or of the expected type, or if the
                   resolved object from the map is not of the expected type
                   (when expected_type is provided).
        ValueError: If obj is a string but is not found in object_map.
    """

    if isinstance(obj, str):
        if obj in object_map:
            resolved_obj = object_map[obj]
        else:
            msg = error_message or f"Unknown object identifier: {obj}"
            raise ValueError(msg)
    elif expected_type is None or isinstance(obj, expected_type):
        resolved_obj = obj
    else:
        msg = error_message or f"Expected type {expected_type}, got {type(obj)}"
        raise TypeError(msg)

    if expected_type and not isinstance(resolved_obj, expected_type):
        msg = (
            error_message
            or f"Resolved object should be of type {expected_type}, got {type(resolved_obj)}"
        )
        raise TypeError(msg)

    return resolved_obj


# -------------------------------------------------------------------------------
# Synthesizer functions
# -------------------------------------------------------------------------------

from hum import Synth
from hum.pyo_util import add_default_dials
from pyo import *


# @knob_exclude('waveform')
@add_default_dials('freq volume')
def theremin_synth(
    freq=440,
    volume=0.5,
    attack=0.01,
    release=0.1,
    vibrato_rate=5,
    vibrato_depth=5,
    *,
    waveform='sine',
):
    """
    Emulates a classic theremin sound.

    Parameters:
    - freq (float): Base frequency in Hz.
    - volume (float): Output volume (0 to 1).
    - waveform (str): Waveform type ('sine', 'triangle', 'square').
    - attack (float): Attack time in seconds.
    - release (float): Release time in seconds.
    - vibrato_rate (float): Vibrato frequency in Hz.
    - vibrato_depth (float): Vibrato depth in Hz.

    Returns:
    - PyoObject: The resulting audio signal.
    """
    # Select waveform
    waveforms = {
        'sine': Sine,
        'triangle': lambda freq, mul: LFO(freq=freq, type=3, mul=mul),
        'square': lambda freq, mul: LFO(freq=freq, type=1, mul=mul),
    }
    wave_class = waveforms.get(waveform, Sine)

    # Vibrato modulation
    vibrato = Sine(freq=vibrato_rate, mul=vibrato_depth)

    # Envelope
    env = Adsr(
        attack=attack, decay=0.1, sustain=0.8, release=release, dur=0, mul=volume
    )
    env.play()
    # Oscillator with vibrato
    osc = wave_class(freq=freq + vibrato, mul=env)

    return osc


# Define a basic sine synth


def sine_synth(freq=440, volume=0):
    return Sine(freq=freq, mul=volume)


def fm_synth(freq=440, volume=0, carrier_ratio=1.0, mod_index=2.0, mod_freq_ratio=2.0):
    mod = Sine(freq=freq * mod_freq_ratio, mul=freq * mod_index)
    car = Sine(freq=freq * carrier_ratio + mod, mul=volume)
    return car


def supersaw_synth(freq=440, volume=0, detune=0.01, n_voices=7):
    voices = [
        LFO(
            freq=freq * (1 + detune * (i - n_voices // 2)),
            type=5,
            mul=volume / n_voices,
        )
        for i in range(n_voices)
    ]
    return sum(voices)


def square_synth(freq=440, volume=0):
    return LFO(freq=freq, type=2, mul=volume)


def noise_synth(freq=440, volume=0, noise_level=0.2):
    sine = Sine(freq=freq, mul=volume * (1 - noise_level))
    noise = Noise(mul=volume * noise_level)
    return sine + noise


def ringmod_synth(freq=440, volume=0, mod_freq_ratio=1.5):
    mod = Sine(freq=freq * mod_freq_ratio)
    carrier = Sine(freq=freq)
    return (carrier * mod) * volume


def chorused_sine_synth(freq=440, volume=0, depth=5, speed=0.3):
    lfo = Sine(freq=speed, mul=depth)
    mod_freq = freq + lfo
    return Sine(freq=mod_freq, mul=volume)


def phase_distortion_synth(freq=440, volume=0, distortion=0.5):
    phasor = Phasor(freq=freq)
    distorted = phasor + (Sine(freq=freq * 2, mul=distortion) * phasor)
    return distorted * volume


DFLT_L_SYNTH = sine_synth
DFLT_R_SYNTH = theremin_synth
DFLT_MIN_FREQ = 220
DFLT_MAX_FREQ = DFLT_MIN_FREQ * 8


# Two-voice synth function
def _two_voice_synth_func(
    l_freq=440,
    l_volume=0.0,
    r_freq=440,
    r_volume=0.0,
    *,
    l_synth=DFLT_L_SYNTH,
    r_synth=DFLT_R_SYNTH,
):
    sound1 = l_synth(freq=l_freq, volume=l_volume)
    sound2 = r_synth(freq=r_freq, volume=r_volume)
    return sound1 + sound2


# Note: Using two_voice_synth_func to obfuscate the l_synth and r_synth parameters,
#       which confuse pyo (because no mapping to a knob).
# TODO: Would be nicer to have something that just removes all but the necessary arguments
#   Something like i2 wrappers or partialx...
# Below not working (yet)
# from i2 import Sig, partialx
# _synth_func = (Sig(synth_func) - 'synth0' - 'synth1')(synth_func)
def two_voice_synth_func(l_freq=440, l_volume=0.0, r_freq=440, r_volume=0.0):
    return _two_voice_synth_func(**locals())


two_voice_synth_func = obfuscate_args(
    _two_voice_synth_func, keep_args=['l_freq', 'l_volume', 'r_freq', 'r_volume']
)

# -------------------------------------------------------------------------------
# Knob functions
# -------------------------------------------------------------------------------


def _calculate_freq_and_vol_from_wrist(wrist, min_freq, max_freq):
    """
    Calculate frequency and volume based on wrist position.

    Args:
        wrist: Position of the wrist (tuple or array with x, y coordinates)
        min_freq: Minimum frequency value
        max_freq: Maximum frequency value

    Returns:
        tuple: (frequency, volume)
    """
    freq = float(min_freq + wrist[0] * (max_freq - min_freq))
    vol = float(np.clip(1 - wrist[1], 0, 1))
    return freq, vol


def two_hand_freq_and_volume_knobs(
    hand_features,
    *,
    min_freq: float = DFLT_MIN_FREQ,
    max_freq: float = DFLT_MAX_FREQ,
) -> Dict[str, float]:
    knobs = {}

    # Set default silence
    mid_freq = (min_freq + max_freq) / 2
    knobs['l_freq'] = mid_freq
    knobs['l_volume'] = 0.0
    knobs['r_freq'] = mid_freq
    knobs['r_volume'] = 0.0

    if not hand_features:
        return knobs
    else:
        print(f"{hand_features=}, {type(hand_features)=}")
        if 'l_wrist_position' in hand_features:
            knobs['l_freq'], knobs['l_volume'] = _calculate_freq_and_vol_from_wrist(
                hand_features['l_wrist_position'], min_freq, max_freq
            )
        if 'r_wrist_position' in hand_features:
            knobs['r_freq'], knobs['r_volume'] = _calculate_freq_and_vol_from_wrist(
                hand_features['r_wrist_position'], min_freq, max_freq
            )

    return knobs


def theremin_knobs(
    hand_features,
    *,
    min_freq: float = DFLT_MIN_FREQ,
    max_freq: float = DFLT_MAX_FREQ,
) -> Dict[str, float]:
    """
    Maps right hand to frequency (pitch) and left hand to volume (amplitude),
    mimicking a classic theremin control scheme.

    Args:
        hand_features (dict): Extracted hand feature dictionary.
        min_freq (float): Minimum frequency for pitch control.
        max_freq (float): Maximum frequency for pitch control.

    Returns:
        Dict[str, float]: Dictionary with 'r_freq', 'r_volume', 'l_freq', 'l_volume' keys.
    """
    X, Y = 0, 1
    knobs = {}

    if not hand_features:
        return knobs
    elif 'r_wrist_position' in hand_features and 'l_wrist_position' in hand_features:
        knobs['freq'] = float(
            min_freq + hand_features['r_wrist_position'][X] * (max_freq - min_freq)
        )
        knobs['volume'] = float(np.clip(1 - hand_features['l_wrist_position'][Y], 0, 1))
    else:
        mid_freq = (min_freq + max_freq) / 2
        silent = 0.0
        knobs['freq'] = mid_freq
        knobs['volume'] = silent

    return knobs


# -------------------------------------------------------------------------------
# HandGestureRecognizer class
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
            draw (bool): Whether to draw landmarks on the image.

        Returns:
            img: The image
            hand_detection: The hand detection results.
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hand_detection = self.hands.process(img_rgb)

        return img, hand_detection


# -------------------------------------------------------------------------------
# Hand features calculation
# -------------------------------------------------------------------------------
import mediapipe as mp
import numpy as np
import math  # For angle calculations


def calculate_euclidean_distance(point1, point2):
    return math.sqrt(
        (point1[0] - point2[0]) ** 2
        + (point1[1] - point2[1]) ** 2
        + (point1[2] - point2[2]) ** 2
    )


def calculate_angle(p1, p2, p3):
    """Calculates the angle between three points (p2 is the vertex)."""
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


import math
import numpy as np
from types import MappingProxyType
from collections import namedtuple
import mediapipe as mp

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
    """

    if not hand_detection.multi_hand_landmarks:
        return {}

    hands = {}

    # for hand_landmarks in hand_detection.multi_hand_landmarks:
    #     _hand_features = hand_features(hand_landmarks)
    #     if log_hand_features:
    #         log_hand_features(_hand_features)

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


# -------------------------------------------------------------------------------
# Screen Drawing
# -------------------------------------------------------------------------------

# define the type of a color
from typing import Union, Tuple

Color = Union[Tuple[int, int, int], Tuple[int, int, int, int]]  # BGR or BGRA


def display_sound_features_on_image(
    img: np.ndarray,
    sound_features: dict,
    *,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale: float = 1.0,
    color: Color = (0, 255, 0),
    thickness: float = 2,
    float_format: str = ".2f",
    x_pos=10,
    y_pos=30,
    y_increment=30,
    bg_color: Color = (
        150,
        150,
        150,
        128,
    ),  # Light grey, semi-transparent (BGR + alpha)
):
    """
    Display sound features on the image with a semi-transparent background.

    Args:
        img: The image to draw on
        sound_features: Dictionary of sound features
        font: Font type to use
        font_scale: Size of the font
        color: Text color in BGR format
        thickness: Line thickness of text
        float_format: Format string for float values
        x_pos: Starting x position for text
        y_pos: Starting y position for text
        y_increment: Vertical space between lines
        bg_color: Background color (BGR + alpha) where alpha is 0-255
    """
    # Create an overlay for the background
    overlay = img.copy()

    # Process bg_color to separate BGR and alpha
    if len(bg_color) == 4:
        bg_rgb = bg_color[:3]
        alpha = bg_color[3] / 255.0  # Convert to 0-1 range
    else:
        bg_rgb = bg_color
        alpha = 0.5  # Default alpha

    # Draw background rectangles and text
    for idx, (key, value) in enumerate(sound_features.items()):
        # Format the value based on its type
        if isinstance(value, float):
            formatted_value = f"{value:{float_format}}"
        else:
            formatted_value = str(value)

        text = f"{key}: {formatted_value}"

        # Get text size for background rectangle
        (text_width, text_height), _ = cv2.getTextSize(
            text, font, font_scale, thickness
        )

        # Draw filled rectangle on the overlay
        padding = 5  # Padding around the text
        cv2.rectangle(
            overlay,
            (x_pos - padding, y_pos + idx * y_increment - text_height - padding),
            (x_pos + text_width + padding, y_pos + idx * y_increment + padding),
            bg_rgb,
            -1,  # Filled rectangle
        )

    # Apply the overlay with transparency
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # Draw text on top
    for idx, (key, value) in enumerate(sound_features.items()):
        if isinstance(value, float):
            formatted_value = f"{value:{float_format}}"
        else:
            formatted_value = str(value)

        text = f"{key}: {formatted_value}"
        cv2.putText(
            img,
            text,
            (x_pos, y_pos + idx * y_increment),
            font,
            font_scale,
            color,
            thickness,
        )


def draw_on_screen(
    self,
    img: np.ndarray,
    hand_detection,
    sound_features: Optional[dict] = None,
    *,
    draw_landmarks: bool = True,
    draw_sound_features: Optional[Callable] = display_sound_features_on_image,
):
    """
    Draws vertical and horizontal lines from the edges of the image to the wrist position.

    Args:
        img: The input image.
        hand_detection: The hand detection results.

    Returns:
        img: The image with lines drawn.
    """
    if hand_detection.multi_hand_landmarks:
        for hand_landmarks in hand_detection.multi_hand_landmarks:
            if draw_landmarks:
                self.mp_draw.draw_landmarks(
                    img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )

    if hand_detection.multi_hand_landmarks:
        for hand_landmarks in hand_detection.multi_hand_landmarks:
            wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
            h, w, c = img.shape
            cx, cy = int(wrist.x * w), int(wrist.y * h)

            # Draw vertical line
            cv2.line(img, (cx, 0), (cx, h), (0, 255, 0), 2)
            # Draw horizontal line
            cv2.line(img, (0, cy), (w, cy), (0, 255, 0), 2)

    # Display sound features
    if sound_features:
        display_sound_features_on_image(img, sound_features)

    return img


# -------------------------------------------------------------------------------
# Main function helpers
# -------------------------------------------------------------------------------

from functools import partial

audio_feature_funcs = {
    "two_hand_freq_and_volume_knobs": two_hand_freq_and_volume_knobs,
    "theremin_knobs": theremin_knobs,
}
resolve_audio_features = partial(resolve_object, object_map=audio_feature_funcs)

synth_funcs = {
    "two_voice_synth_func": two_voice_synth_func,
    "theremin_synth": theremin_synth,
}
resolve_synth_func = partial(resolve_object, object_map=synth_funcs)


hand_feature_funcs = {
    "many_hand_features": many_hand_features,
}
resolve_hand_features = partial(resolve_object, object_map=hand_feature_funcs)


def print_plus_newline(x):
    """Prints the input and adds a newline."""
    print(x)
    print()


# -------------------------------------------------------------------------------
# Main function
# -------------------------------------------------------------------------------

DFLT_HAND_FEATURES_NAME = "many_hand_features"

DFLT_AUDIO_FEATURES_NAME = "two_hand_freq_and_volume_knobs"
DFLT_SYNTH_FUNC_NAME = "two_voice_synth_func"

# TODO: Not working yet
DFLT_AUDIO_FEATURES_NAME = "theremin_knobs"
DFLT_SYNTH_FUNC_NAME = "theremin_synth"


def main(
    *,
    hand_features: Union[str, Callable] = DFLT_HAND_FEATURES_NAME,
    audio_features: Union[str, Callable] = DFLT_AUDIO_FEATURES_NAME,
    synth_func: Union[str, Callable] = DFLT_SYNTH_FUNC_NAME,
    draw_on_screen=draw_on_screen,
    log_hand_features=print_plus_newline,
    log_audio_features=print_plus_newline,
    save_recording='theremin_recording.wav',
):
    """Main function to run the hand gesture recognition with pyo theremin."""

    hand_features = resolve_hand_features(hand_features)
    audio_features = resolve_audio_features(audio_features)
    synth_func = resolve_synth_func(synth_func)

    cap = cv2.VideoCapture(0)
    recognizer = HandGestureRecognizer()

    # Initialize the pyo synth
    synth = Synth(synth_func, nchnls=2)
    print(f"\nUsing synth function: {synth_func}: {list(synth.knobs)}\n")

    with synth:
        try:
            while cap.isOpened():
                success, img = cap.read()
                if not success:
                    break

                img = cv2.flip(img, 1)
                img, hand_detection = recognizer.find_hands(img)

                # Compute the hand features
                _hand_features = hand_features(hand_detection)
                if log_hand_features:
                    log_hand_features(_hand_features)

                # Compute sound features from hand landmarks
                _audio_features = audio_features(_hand_features)
                if log_audio_features:
                    log_audio_features(_audio_features)
                if _audio_features is not None:
                    synth.knobs.update(_audio_features)

                # Draw stuff on the screen
                if draw_on_screen:
                    img = draw_on_screen(
                        recognizer, img, hand_detection, _audio_features
                    )

                cv2.imshow('Hand Gesture Recognition with Theremin', img)

                if cv2.waitKey(5) & 0xFF == 27:  # ESC key to exit
                    break
        finally:
            # Save the recording
            synth.stop_recording()

            recording = synth.get_recording()

            # Print recording statistics
            print(f"\n---> Recorded {len(recording)} control events\n")

            # Render the recording to a WAV file?
            if save_recording:
                if isinstance(save_recording, str):
                    output_path = save_recording
                else:
                    assert isinstance(
                        save_recording, bool
                    ), "save_recording should be a string or a boolean"
                    output_path = "theremin_recording.wav"
                synth.render_events(output_filepath=output_path)
                print(f"Saved audio recording to {output_path}")

            # Clean up resources
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
