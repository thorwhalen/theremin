"""Utility functions for running the theremin scripts."""

import cv2
import time
from typing import Union, Callable, Dict, Optional, Any
from functools import partial

from theramin.hand_features import HandGestureRecognizer, hand_feature_funcs
from theramin.audio import synth_funcs, audio_feature_funcs
from theramin.display import draw_on_screen
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
resolve_hand_features = partial(resolve_object, object_map=hand_feature_funcs)
resolve_audio_features = partial(resolve_object, object_map=audio_feature_funcs)
resolve_synth_func = partial(resolve_object, object_map=synth_funcs)


# -------------------------------------------------------------------------------
# Logging utilities
# -------------------------------------------------------------------------------


def print_plus_newline(x):
    """Prints the input and adds a newline."""
    print(x)
    print()


# -------------------------------------------------------------------------------
# Main run function
# -------------------------------------------------------------------------------

# Default settings
DFLT_HAND_FEATURES_NAME = "many_hand_features"
DFLT_AUDIO_FEATURES_NAME = "two_hand_freq_and_volume_knobs"
DFLT_SYNTH_FUNC_NAME = "theremin_synth"


def run_theremin(
    *,
    hand_features: Union[str, Callable] = DFLT_HAND_FEATURES_NAME,
    audio_features: Union[str, Callable] = DFLT_AUDIO_FEATURES_NAME,
    synth_func: Union[str, Callable] = DFLT_SYNTH_FUNC_NAME,
    log_hand_features: Optional[Callable] = None,
    log_audio_features: Optional[Callable] = None,
    save_recording: Union[str, bool] = 'theremin_recording.wav',
    window_name: str = 'Hand Gesture Recognition with Theremin',
):
    """
    Run the hand gesture theremin application.

    Args:
        hand_features: Hand feature extraction function or name
        audio_features: Audio feature mapping function or name
        synth_func: Synthesizer function or name
        log_hand_features: Function to log hand features (or None to disable)
        log_audio_features: Function to log audio features (or None to disable)
        save_recording: Filename to save recording, True for default name, or False to disable
        window_name: Title for the display window
    """
    # Resolve functions from names if needed
    hand_features = resolve_hand_features(hand_features)
    audio_features = resolve_audio_features(audio_features)
    synth_func = resolve_synth_func(synth_func)

    # Initialize video capture
    cap = cv2.VideoCapture(0)
    recognizer = HandGestureRecognizer()

    # Initialize the pyo synth
    synth = Synth(synth_func, nchnls=2)
    print(f"\nUsing synth function: {synth_func.__name__}: {list(synth.knobs)}\n")

    with synth:
        try:
            while cap.isOpened():
                success, img = cap.read()
                if not success:
                    break

                # Flip image horizontally for a more natural interaction
                img = cv2.flip(img, 1)

                # Detect hands
                img, hand_detection = recognizer.find_hands(img)

                # Compute the hand features
                _hand_features = hand_features(hand_detection)
                if log_hand_features:
                    log_hand_features(_hand_features)

                # Compute sound features from hand landmarks
                _audio_features = audio_features(_hand_features)
                if log_audio_features:
                    log_audio_features(_audio_features)

                # Update synth parameters if we have features
                if _audio_features:
                    synth.knobs.update(_audio_features)

                # Draw visualization
                img = draw_on_screen(recognizer, img, hand_detection, _audio_features)

                # Display the result
                cv2.imshow(window_name, img)

                # Exit on ESC key
                if cv2.waitKey(5) & 0xFF == 27:
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
