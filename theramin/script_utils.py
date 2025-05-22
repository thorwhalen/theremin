"""Utility functions for running the theremin scripts."""

import cv2
import time
from typing import Union, Callable, Dict, Optional, Any
from functools import partial
import json

from theremin.video_features import HandGestureRecognizer, hand_feature_funcs
from theremin.audio import synth_funcs, audio_feature_funcs
from theremin.display import draw_on_screen as DFLT_DRAW_ON_SCREEN
from hum.pyo_util import Synth

# -------------------------------------------------------------------------------
# Object resolution
# -------------------------------------------------------------------------------

from typing import TypeVar, Dict, Union, Any

T = TypeVar('T')


def resolve_object(
    obj: Union[str, T],
    *,
    object_map: Dict[str, T],
    expected_type: type = None,
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


# Create partially applied resolution functions for each type of object
resolve_video_features = partial(resolve_object, object_map=hand_feature_funcs)
resolve_audio_features = partial(resolve_object, object_map=audio_feature_funcs)
resolve_synth_func = partial(resolve_object, object_map=synth_funcs)


# -------------------------------------------------------------------------------
# Logging utilities
# -------------------------------------------------------------------------------


def print_plus_newline(x):
    """Prints the input and adds a newline."""
    print(x)
    print()


def print_json_if_possible(x):
    """Prints the input and adds a newline."""
    try:
        x = json.dumps(x)
    except:
        pass
    print(x)
    print()


# -------------------------------------------------------------------------------
# Keyboard handling functions
# -------------------------------------------------------------------------------


class KeyboardBreakSignal(Exception):
    """Exception raised when a break key is pressed."""

    pass


def read_keyboard(wait_time: int = 5) -> int:
    """
    Read keyboard input with the specified wait time.

    Args:
        wait_time: Time to wait for keyboard input in milliseconds

    Returns:
        The key code or 0 if no key was pressed
    """
    return cv2.waitKey(wait_time) & 0xFF


def keyboard_feature_vector(key_code: int) -> Dict[str, Any]:
    """
    Convert a key code into a feature vector with keyboard information.

    Args:
        key_code: The key code from cv2.waitKey

    Returns:
        Dictionary containing keyboard features

    Raises:
        KeyboardBreakSignal: If a key that signals program termination is pressed
    """
    keyboard_fv = {
        'key_code': key_code,
        'key_pressed': key_code > 0,
        'is_escape': key_code == ESCAPE_KEY_ASCII,
        'timestamp': time.time(),
    }

    # Check if this is a break key and raise the exception if so
    if keyboard_fv['key_code'] in BREAK_KEYS:
        raise KeyboardBreakSignal(f"Break key pressed: {key_code}")

    return keyboard_fv


# -------------------------------------------------------------------------------
# Camera handling functions
# -------------------------------------------------------------------------------


class CameraReadError(Exception):
    """Exception raised when camera read fails."""

    pass


def read_camera(cap: cv2.VideoCapture) -> Any:
    """
    Read a frame from the camera and flip it horizontally.

    Args:
        cap: OpenCV video capture object

    Returns:
        The flipped image if successful

    Raises:
        CameraReadError: If the camera read operation fails
    """
    success, img = cap.read()
    if not success:
        raise CameraReadError("Failed to read from camera")

    # Flip image horizontally for a more natural interaction
    return cv2.flip(img, 1)


# -------------------------------------------------------------------------------
# Main run function
# -------------------------------------------------------------------------------

# Default settings
DFLT_VIDEO_FEATURES = "many_video_features"
DFLT_AUDIO_FEATURES = "two_hand_freq_and_volume_knobs"
DFLT_SYNTH_FUNC_NAME = "theremin_synth"

ESCAPE_KEY_ASCII = 27
BREAK_KEYS = set([ESCAPE_KEY_ASCII])


from hum.util import scale_snapper, scale_frequencies, return_none as do_nothing
from theremin.audio import DFLT_MIN_FREQ, DFLT_MAX_FREQ

scale = (0, 2, 4, 5, 7, 9, 11)
freq_trans = scale_snapper(scale=scale)
_scale_frequencies = [
    freq for freq in scale_frequencies() if DFLT_MIN_FREQ <= freq <= DFLT_MAX_FREQ
]

_DFLT_DRAW_ON_SCREEN = partial(DFLT_DRAW_ON_SCREEN, draw_frequencies=_scale_frequencies)


def run_theremin(
    *,
    video_features: Union[str, Callable] = DFLT_VIDEO_FEATURES,
    audio_features: Union[str, Callable] = DFLT_AUDIO_FEATURES,
    synth_func: Union[str, Callable] = DFLT_SYNTH_FUNC_NAME,
    log_video_features: Optional[Callable] = None,
    log_audio_features: Optional[Callable] = None,
    save_recording: Union[str, bool] = 'theremin_recording.wav',
    window_name: str = 'Hand Gesture Recognition with Theremin',
    draw_on_screen: Optional[Callable] = _DFLT_DRAW_ON_SCREEN,
    only_keep_new_freqs: bool = True,
):
    """
    Run the hand gesture theremin application.

    Args:
        video_features: Hand feature extraction function or name
        audio_features: Audio feature mapping function or name
        synth_func: Synthesizer function or name
        log_video_features: Function to log video features (or None to disable)
        log_audio_features: Function to log audio features (or None to disable)
        save_recording: Filename to save recording, True for default name, or False to disable
        window_name: Title for the display window
    """
    # Resolve functions from names if needed
    video_features = resolve_video_features(video_features)
    audio_features = resolve_audio_features(audio_features)
    synth_func = resolve_synth_func(synth_func)

    log_video_features = log_video_features or do_nothing
    log_audio_features = log_audio_features or do_nothing

    # Initialize video capture
    cap = cv2.VideoCapture(0)
    recognizer = HandGestureRecognizer()

    # Initialize the pyo synth
    if not isinstance(synth_func, Synth):
        synth = Synth(synth_func, nchnls=2)
        # synth = Synth(synth_func, nchnls=2, value_trans={'freq': snap_to_c_major})
        # print(f"\n-----> Snapping to C major scale\n\n")
    else:
        synth = synth_func
    print(f"\nUsing synth function: {synth_func.__name__}: {list(synth.knobs)}\n")

    from collections import deque

    last_raw_freqs = deque(maxlen=10)
    last_freq = None
    last_l_freq = None
    last_r_freq = None

    with synth:
        try:
            while cap.isOpened():
                try:

                    keyboard_data = read_keyboard()

                    # Note: Not using keyboard_fv (yet), but keyboard_feature_vector is still called because handles breaking key conditioning
                    keyboard_fv = keyboard_feature_vector(keyboard_data)

                    img = read_camera(cap)

                    # Get hand detection without modifying the image
                    hand_detection = recognizer.find_hands(img)

                    # Compute the hand features
                    _video_features = video_features(hand_detection)
                    if log_video_features:
                        log_video_features(_video_features)

                    # Compute sound features from hand landmarks
                    _audio_features = audio_features(_video_features)

                    # Update synth parameters if we have features
                    if _audio_features:

                        # TODO: Pack into tool:
                        if only_keep_new_freqs:
                            if 'freq' in _audio_features:
                                freq = _audio_features['freq']
                                last_raw_freqs.append(freq)
                                if freq == last_freq:
                                    del _audio_features['freq']
                            if 'l_freq' in _audio_features:
                                l_freq = _audio_features['l_freq']
                                if l_freq == last_l_freq:
                                    del _audio_features['l_freq']
                            if 'r_freq' in _audio_features:
                                r_freq = _audio_features['r_freq']
                                if r_freq == last_r_freq:
                                    del _audio_features['r_freq']

                        synth(**_audio_features)

                    if log_audio_features:
                        log_audio_features(_audio_features)

                    # Draw visualization
                    if draw_on_screen:
                        img = draw_on_screen(
                            recognizer, img, hand_detection, _audio_features
                        )

                    # Display the result
                    cv2.imshow(window_name, img)

                except (CameraReadError, KeyboardBreakSignal):
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
                    output_path = "theremin_recording.wav"
                synth.render_events(output_filepath=output_path)
                print(f"Saved audio recording to {output_path}")

            # Clean up resources
            cap.release()
            cv2.destroyAllWindows()


def theremin_cli(
    # Core components
    video_features: str = "many_video_features",
    audio_features: str = "theremin_knobs",
    synth_func: str = "theremin_synth",
    # Logging options
    log_video_features: bool = False,
    log_audio_features: bool = False,
    # Recording options
    save_recording: str = "theremin_recording.wav",
    no_recording: bool = False,
    # Display options
    window_name: str = "Theremin with Hand Tracking",
    # List available components
    list_synths: bool = False,
    list_audio_features: bool = False,
    list_video_features: bool = False,
):
    """
    Run the theremin application with the specified parameters.

    Args:
        video_features: Name of the hand feature extraction function
        audio_features: Name of the audio feature mapping function
        synth_func: Name of the synthesizer function
        log_video_features: Whether to log hand features
        log_audio_features: Whether to log audio features
        save_recording: Filename to save recording (if no_recording is False)
        no_recording: Disable recording
        window_name: Title for the display window
        list_synths: List available synthesizer functions and exit
        list_audio_features: List available audio feature mapping functions and exit
        list_video_features: List available hand feature extraction functions and exit
    """
    # Import here to avoid loading everything if just listing components
    from theremin.audio import synth_funcs, audio_feature_funcs
    from theremin.video_features import hand_feature_funcs

    # Handle listing available components
    if list_synths:
        print("Available synthesizer functions:")
        for name in sorted(synth_funcs.keys()):
            print(f"  - {name}")
        return

    if list_audio_features:
        print("Available audio feature mapping functions:")
        for name in sorted(audio_feature_funcs.keys()):
            print(f"  - {name}")
        return

    if list_video_features:
        print("Available hand feature extraction functions:")
        for name in sorted(hand_feature_funcs.keys()):
            print(f"  - {name}")
        return

    # Handle recording options
    if no_recording:
        save_recording = False

    # Set up logging callbacks
    log_video_features_callback = print_json_if_possible if log_video_features else None
    log_audio_features_callback = print_json_if_possible if log_audio_features else None

    # Run the theremin application
    run_theremin(
        video_features=video_features,
        audio_features=audio_features,
        synth_func=synth_func,
        log_video_features=log_video_features_callback,
        log_audio_features=log_audio_features_callback,
        save_recording=save_recording,
        window_name=window_name,
    )
