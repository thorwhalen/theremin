"""Utility functions for running the theremin scripts."""

import cv2
import time
from typing import Union, Callable, Dict, Optional, Any
from functools import partial
import json

from i2 import Sig
from theremin.video_features import HandGestureRecognizer, hand_feature_funcs
from theremin.audio import synth_funcs, audio_feature_funcs, audio_pipelines
from theremin.display import draw_on_screen as DFLT_DRAW_ON_SCREEN
from hum.pyo_util import Synth

# -------------------------------------------------------------------------------
# Object resolution
# -------------------------------------------------------------------------------

from typing import TypeVar, Dict, Union, Any


# Create partially applied resolution functions for each type of object
from cw import resolve_to_function
from cw.resolution import parse_ast_spec


resolve_video_features = partial(resolve_to_function, get_func=hand_feature_funcs)

resolve_audio_pipeline = partial(resolve_to_function, get_func=audio_pipelines)
resolve_audio_features = partial(resolve_to_function, get_func=audio_feature_funcs)
resolve_synth_func = partial(resolve_to_function, get_func=synth_funcs)

# resolve_audio_pipeline_func = partial(
#     resolve_to_function,
#     get_func={'audio_pipe': audio_pipe},
# )


# from theremin.audio import audio_pipe
# def resolve_audio_pipeline(pipeline_key):
#     func_spec = audio_pipelines.get(pipeline_key)
#     if func_spec is None:
#         raise ValueError(f"Audio pipeline {pipeline_key} not found in audio_pipelines")
#     func_name, func_kwargs = parse_ast_spec(func_spec)
#     if func_name != "audio_pipe":
#         raise ValueError(f"Should be 'audio_pipe', was {func_name}: In {func_spec}")
#     if not set(func_kwargs.keys()).issubset({'synth', 'knobs'}):
#         raise ValueError(
#             f"Function {func_name} should have only keys in {{'synth', 'knobs'}}: In {func_spec}"
#         )
#     return audio_pipe(**func_kwargs)


# resolve_audio_pipeline = partial(
#     resolve_to_function, get_func=lambda x: resolve_audio_pipeline_func
# )


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
from theremin.audio import (
    DFLT_AUDIO_FEATURES,
    DFLT_SYNTH_FUNC_NAME,
    DFLT_AUDIO_PIPELINE,
)

DFLT_VIDEO_FEATURES = "many_video_features"


ESCAPE_KEY_ASCII = 27
BREAK_KEYS = set([ESCAPE_KEY_ASCII])


from hum.util import scale_snapper, scale_frequencies, return_none as do_nothing
from theremin.audio import DFLT_MIN_FREQ, DFLT_MAX_FREQ, filter_unchanged_frequencies

scale = (0, 2, 4, 5, 7, 9, 11)
freq_trans = scale_snapper(scale=scale)
_scale_frequencies = [
    freq for freq in scale_frequencies() if DFLT_MIN_FREQ <= freq <= DFLT_MAX_FREQ
]

_DFLT_DRAW_ON_SCREEN = partial(DFLT_DRAW_ON_SCREEN, draw_frequencies=_scale_frequencies)


# TODO: Rename audio_features to knobs and synth_func to synth
def run_theremin(
    *,
    video_features: Union[str, Callable] = DFLT_VIDEO_FEATURES,
    audio_pipeline: Union[str, Callable] = DFLT_AUDIO_PIPELINE,  # DFLT_AUDIO_PIPELINE,
    audio_features: Optional[Union[str, Callable]] = None, #DFLT_AUDIO_FEATURES,
    synth_func: Optional[Union[str, Callable]] = None, #DFLT_SYNTH_FUNC_NAME,
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
    if not audio_pipeline:
        audio_features = resolve_audio_features(audio_features)
        synth_func = resolve_synth_func(synth_func)
    else:
        audio_pipeline_getter = resolve_audio_pipeline(audio_pipeline)
        pipeline_components = audio_pipeline_getter()
        audio_features = pipeline_components.get(
            "knobs", audio_features or DFLT_AUDIO_FEATURES
        )
        synth_func = pipeline_components.get(
            "synth", synth_func or DFLT_SYNTH_FUNC_NAME
        )
        audio_features = resolve_audio_features(audio_features)
        synth_func = resolve_synth_func(synth_func)

    print(f"{audio_features=}, {synth_func=}")

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
    from types import SimpleNamespace

    previous_data = SimpleNamespace(
        last_raw_freqs=deque(maxlen=10),
        last_freq=None,
        last_l_freq=None,
        last_r_freq=None,
    )

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
                    log_video_features(_video_features)

                    # Compute sound features from hand landmarks
                    _audio_features = audio_features(_video_features)

                    # Update synth parameters if we have features
                    if _audio_features:

                        # TODO: Pack into tool:
                        if only_keep_new_freqs:
                            _audio_features, previous_data = (
                                filter_unchanged_frequencies(
                                    _audio_features, previous_data
                                )
                            )

                        synth(**_audio_features)

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
                try:
                    if isinstance(save_recording, str):
                        output_path = save_recording
                    else:
                        output_path = "theremin_recording.wav"
                    synth.render_events(output_filepath=output_path)
                    print(f"Saved audio recording to {output_path}")
                except Exception as e:
                    print(f"Warning: Failed to render events: {e}")

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
            func = synth_funcs[name]
            print(f"  - {name}{Sig(func)}")
        return

    if list_audio_features:
        print("Available audio feature mapping functions:")
        for name in sorted(audio_feature_funcs.keys()):
            func = audio_feature_funcs[name]
            print(f"  - {name}{Sig(func)}")
        return

    if list_video_features:
        print("Available hand feature extraction functions:")
        for name in sorted(hand_feature_funcs.keys()):
            func = hand_feature_funcs[name]
            print(f"  - {name}{Sig(func)}")
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
