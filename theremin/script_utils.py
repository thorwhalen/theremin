"""Utility functions for running the theremin scripts."""

import cv2
import time
from typing import Union, Callable, Dict, Optional, Any
from functools import partial
import json

from i2 import Sig
from theremin.video_features import HandGestureRecognizer, hand_feature_funcs
from theremin.audio import synths, knobs, pipelines
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

resolve_pipeline = partial(resolve_to_function, get_func=pipelines)
resolve_knobs = partial(resolve_to_function, get_func=knobs)
resolve_synth_func = partial(resolve_to_function, get_func=synths)

# resolve_pipeline_func = partial(
#     resolve_to_function,
#     get_func={'audio_pipe': audio_pipe},
# )


# from theremin.audio import audio_pipe
# def resolve_pipeline(pipeline_key):
#     func_spec = pipelines.get(pipeline_key)
#     if func_spec is None:
#         raise ValueError(f"Audio pipeline {pipeline_key} not found in pipelines")
#     func_name, func_kwargs = parse_ast_spec(func_spec)
#     if func_name != "audio_pipe":
#         raise ValueError(f"Should be 'audio_pipe', was {func_name}: In {func_spec}")
#     if not set(func_kwargs.keys()).issubset({'synth', 'knobs'}):
#         raise ValueError(
#             f"Function {func_name} should have only keys in {{'synth', 'knobs'}}: In {func_spec}"
#         )
#     return audio_pipe(**func_kwargs)


# resolve_pipeline = partial(
#     resolve_to_function, get_func=lambda x: resolve_pipeline_func
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
    KNOBS,
    DFLT_SYNTH,
    DFLT_PIPELINE,
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
    pipeline: Union[str, Callable] = DFLT_PIPELINE,  # DFLT_PIPELINE,
    knobs: Optional[Union[str, Callable]] = None,  # DFLT_AUDIO_FEATURES,
    synth: Optional[Union[str, Callable]] = None,  # DFLT_SYNTH_FUNC_NAME,
    log_video_features: Optional[Callable] = None,
    log_knobs: Optional[Callable] = None,
    record_to_file: Union[str, bool] = 'theremin_recording.wav',
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
        log_knobs: Function to log audio features (or None to disable)
        record_to_file: Filename to save recording, True for default name, or False to disable
        window_name: Title for the display window
    """
    # Resolve functions from names if needed
    video_features = resolve_video_features(video_features)
    if not pipeline:
        knobs = resolve_knobs(knobs)
        synth = resolve_synth_func(synth)
    else:
        pipeline_getter = resolve_pipeline(pipeline)
        pipeline_components = pipeline_getter()
        knobs = pipeline_components.get("knobs", knobs or KNOBS)
        synth = pipeline_components.get("synth", synth or DFLT_SYNTH)
        knobs = resolve_knobs(knobs)
        synth = resolve_synth_func(synth)

    print(f"{knobs=}, {synth=}")

    log_video_features = log_video_features or do_nothing
    log_knobs = log_knobs or do_nothing

    # Initialize video capture
    cap = cv2.VideoCapture(0)
    recognizer = HandGestureRecognizer()

    # Initialize the pyo synth
    if not isinstance(synth, Synth):
        synth_obj = Synth(synth, nchnls=2)
        synth_obj.__name__ = synth.__name__
    else:
        synth_obj = synth
    print(f"\nUsing synth function: {synth.__name__}: {list(synth_obj.knobs)}\n")

    from collections import deque
    from types import SimpleNamespace

    previous_data = SimpleNamespace(
        last_raw_freqs=deque(maxlen=10),
        last_freq=None,
        last_l_freq=None,
        last_r_freq=None,
    )

    with synth_obj:
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
                    _audio_features = knobs(_video_features)

                    # Update synth_obj parameters if we have features
                    if _audio_features:

                        # TODO: Pack into tool:
                        if only_keep_new_freqs:
                            _audio_features, previous_data = (
                                filter_unchanged_frequencies(
                                    _audio_features, previous_data
                                )
                            )

                        synth_obj(**_audio_features)

                    log_knobs(_audio_features)

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
            synth_obj.stop_recording()

            recording = synth_obj.get_recording()

            # Print recording statistics
            print(f"\n---> Recorded {len(recording)} control events\n")

            # Render the recording to a WAV file?
            if record_to_file:
                try:
                    if isinstance(record_to_file, str):
                        output_path = record_to_file
                    else:
                        output_path = "theremin_recording.wav"
                    synth_obj.render_events(output_filepath=output_path)
                    print(f"Saved audio recording to {output_path}")
                except Exception as e:
                    print(f"Warning: Failed to render events: {e}")

            # Clean up resources
            cap.release()
            cv2.destroyAllWindows()


def list_components(param_value, components_dict, description, component_describer=Sig):
    """
    List available components of a specific type if the parameter value is 'list'.

    Args:
        param_value: The value of the parameter to check
        components_dict: Dictionary of available components
        description: Description of the components being listed

    Returns:
        bool: True if components were listed (and calling function should return),
             False if no listing was performed
    """
    if isinstance(param_value, str) and param_value == 'list':
        print(f"Available {description}:")
        for name in sorted(components_dict.keys()):
            func = components_dict[name]
            print(f"  - {name}{component_describer(func)}")
        return True
    return False


# TODO: get rid of the need of both record_to_file and no_recording
def theremin_cli(
    # Core components
    pipeline: str = "theremin",
    video_features: str = "many_video_features",
    knobs: str = "theremin_knobs",
    synth: str = "theremin_synth",
    # Logging options
    log_video_features: bool = False,
    log_knobs: bool = False,
    # Recording options
    record_to_file: str = "theremin_recording.wav",
    no_recording: bool = False,
    # Display options
    window_name: str = "Theremin with Hand Tracking",
):
    """
    Run the theremin application with the specified parameters.

    Args:
        pipeline: Name of the audio pipeline (if value is list, will list available options)
        video_features: Name of the hand feature extraction function (if value is list, will list available options)
        knobs: Name of the audio feature mapping function (if value is list, will list available options)
        synth: Name of the synthesizer function (if value is list, will list available options)

        log_video_features: Whether to log hand features
        log_knobs: Whether to log audio features
        record_to_file: Filename to save recording (if no_recording is False)
        no_recording: Disable recording
        window_name: Title for the display window
    """
    # Import here to avoid loading everything if just listing components
    from theremin.audio import synths, knobs
    from theremin.video_features import hand_feature_funcs

    # Handle listing available components
    if list_components(pipeline, pipelines, "pipelines"):
        return

    if list_components(synth, synths, "synthesizer functions"):
        return

    if list_components(knobs, knobs, "audio feature mapping functions"):
        return

    if list_components(
        video_features, hand_feature_funcs, "hand feature extraction functions"
    ):
        return

    # Handle recording options
    if no_recording:
        record_to_file = False

    # Set up logging callbacks
    log_video_features_callback = print_json_if_possible if log_video_features else None
    log_knobs_callback = print_json_if_possible if log_knobs else None

    # Run the theremin application
    run_theremin(
        pipeline=pipeline,
        video_features=video_features,
        knobs=knobs,
        synth=synth,
        log_video_features=log_video_features_callback,
        log_knobs=log_knobs_callback,
        record_to_file=record_to_file,
        window_name=window_name,
    )
