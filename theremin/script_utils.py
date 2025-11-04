"""Utility functions for running the theremin scripts.

This module focuses on fast CLI startup by deferring all heavy imports (cv2, mediapipe,
pyo, theremin.audio, etc.) until they are actually needed (lazy import pattern).
Only lightweight stdlib / small packages are imported at module import time so
`theremin -h` remains responsive.
"""

# Only light imports at module import time
import time
import argh
from typing import Union, Dict, Optional, Any
from collections.abc import Callable
from functools import partial
import json
from i2 import Sig

# Heavy modules (cv2, mediapipe, pyo, etc.) and theremin.audio are intentionally NOT imported at top-level
# to make `theremin -h` fast. They will be imported lazily inside runtime functions.

# -------------------------------------------------------------------------------
# Logging utilities
# -------------------------------------------------------------------------------


def print_plus_newline(x):
    """Print value plus an extra blank line (simple pretty printing helper)."""
    print(x)
    print()


def print_json_if_possible(x):
    """Print object as JSON if serializable; fall back to raw `repr`; add blank line."""
    try:
        x = json.dumps(x)
    except Exception:
        pass
    print(x)
    print()


# -------------------------------------------------------------------------------
# Keyboard handling functions (cv2 imported lazily when first used)
# -------------------------------------------------------------------------------


class KeyboardBreakSignal(Exception):
    """Raised internally to break the main loop when a designated key is pressed."""

    pass


def _ensure_cv2():
    """Dynamically import and cache OpenCV (`cv2`) the first time it is needed."""
    global cv2
    if 'cv2' not in globals():  # pragma: no cover - trivial
        import cv2  # type: ignore

        globals()['cv2'] = cv2
    return cv2


def read_keyboard(wait_time: int = 5) -> int:
    """Return last key code (or -1) using cv2.waitKey, importing cv2 lazily."""
    _ensure_cv2()
    return cv2.waitKey(wait_time) & 0xFF


def keyboard_feature_vector(key_code: int) -> dict[str, Any]:
    """Convert a raw key code into a small feature dict; raise if it's a break key."""
    keyboard_fv = {
        'key_code': key_code,
        'key_pressed': key_code > 0,
        'is_escape': key_code == ESCAPE_KEY_ASCII,
        'timestamp': time.time(),
    }
    if keyboard_fv['key_code'] in BREAK_KEYS:
        raise KeyboardBreakSignal(f"Break key pressed: {key_code}")
    return keyboard_fv


# -------------------------------------------------------------------------------
# Camera handling (cv2 imported lazily)
# -------------------------------------------------------------------------------


class CameraReadError(Exception):
    """Raised when a frame cannot be read from the camera."""

    pass


def read_camera(cap):
    """Read and horizontally flip a camera frame; raise on failure."""
    _ensure_cv2()
    success, img = cap.read()
    if not success:
        raise CameraReadError("Failed to read from camera")
    return cv2.flip(img, 1)


# -------------------------------------------------------------------------------
# Defaults & constants (lightweight)
# -------------------------------------------------------------------------------
from hum.util import scale_snapper, scale_frequencies, return_none as do_nothing

# NOTE: We purposely avoid importing theremin.audio here. All audio imports happen lazily.

DFLT_VIDEO_FEATURES = "many_video_features"
ESCAPE_KEY_ASCII = 27
BREAK_KEYS = {ESCAPE_KEY_ASCII}

scale = (0, 2, 4, 5, 7, 9, 11)
freq_trans = scale_snapper(scale=scale)

# Lazy cache for scale frequencies once audio constants are known
_scale_frequencies_cache = None


def _get_scale_frequencies():
    """Compute & cache frequencies within DFLT_MIN/MAX using hum.util + audio limits.

    Does a lazy import of `theremin.audio` to access min/max constants only on first use.
    """
    global _scale_frequencies_cache
    if _scale_frequencies_cache is None:
        from theremin.audio import DFLT_MIN_FREQ, DFLT_MAX_FREQ  # lazy

        _scale_frequencies_cache = [
            freq
            for freq in scale_frequencies()
            if DFLT_MIN_FREQ <= freq <= DFLT_MAX_FREQ
        ]
    return _scale_frequencies_cache


# Lazy default drawing function wrapper


def _get_default_draw_on_screen_for_scale(scale: str | None):
    """Return draw_on_screen partial with scale-dependent frequencies.

    If scale is None: use default cached scale frequencies; if set to disables snapping
    (e.g., 'none'), return no draw_frequencies so existing draw code can handle empty.
    """
    from theremin.display import draw_on_screen as DFLT_DRAW_ON_SCREEN  # lazy
    from theremin.audio import (
        DFLT_MIN_FREQ,
        DFLT_MAX_FREQ,
        DFLT_SCALE as AUDIO_DFLT_SCALE,
        scale_frequencies_for_range,
        resolve_scale_to_freq_trans,
    )

    # Interpret the CLI-provided scale string similar to freq_trans_override
    freq_trans_marker = resolve_scale_to_freq_trans(scale)

    if freq_trans_marker is None:
        draw_freqs = []  # snapping disabled -> no scale guides
    else:
        # Decide which scale string to use for guides
        scale_name = (
            AUDIO_DFLT_SCALE if freq_trans_marker == "__USE_DEFAULT__" else scale
        )
        draw_freqs = scale_frequencies_for_range(
            scale_name, DFLT_MIN_FREQ, DFLT_MAX_FREQ
        )
    return partial(DFLT_DRAW_ON_SCREEN, draw_frequencies=draw_freqs)


# Existing generic factory kept for backward-compat


def _get_default_draw_on_screen():
    from theremin.display import draw_on_screen as DFLT_DRAW_ON_SCREEN  # lazy

    return partial(DFLT_DRAW_ON_SCREEN, draw_frequencies=_get_scale_frequencies())


# -------------------------------------------------------------------------------
# Main run function (heavy imports are done inside)
# -------------------------------------------------------------------------------


def run_theremin(
    *,
    video_features: str | Callable = DFLT_VIDEO_FEATURES,
    pipeline: str | Callable | None = None,
    knobs: str | Callable | None = None,
    synth: str | Callable | None = None,
    log_video_features: Callable | None = None,
    log_knobs: Callable | None = None,
    record_to_file: str | bool = 'theremin_recording.wav',
    window_name: str = 'Hand Gesture Recognition with Theremin',
    draw_on_screen: Callable | None = None,
    only_keep_new_freqs: bool = True,
    freq_trans_override: str | Callable | None = "__USE_DEFAULT__",
):
    """Run the realtime theremin loop.

    Lazy-loads heavy dependencies (cv2, mediapipe, pyo, pipelines) only if actually
    executing the loop (i.e. not when just requesting CLI help). Handles:
      - Resolving named registry entries (video features, knobs/pipelines, synth)
      - Video capture + hand feature extraction
      - Mapping features to audio parameters (knobs)
      - Optional logging & drawing
      - Synth control & (optionally) recording, filtering unchanged freqs
    """
    # Heavy imports here
    _ensure_cv2()
    from theremin.video_features import HandGestureRecognizer, hand_feature_funcs
    from theremin.audio import (
        synths,
        knobs as KNOBS_DICT,
        pipelines,
        filter_unchanged_frequencies,
    )
    from theremin.audio import (
        KNOBS as DFLT_KNOBS,
        DFLT_SYNTH as DFLT_SYNTH_FUNC,
        DFLT_PIPELINE as DFLT_PIPELINE_NAME,
    )
    from theremin.util import ensure_plain_types
    from hum.pyo_util import Synth
    from cw import resolve_to_function

    if draw_on_screen is None:
        draw_on_screen = _get_default_draw_on_screen()

    # Resolution helpers (now that we have the dicts)
    resolve_video_features = partial(resolve_to_function, get_func=hand_feature_funcs)
    resolve_pipeline = partial(resolve_to_function, get_func=pipelines)
    resolve_knobs = partial(resolve_to_function, get_func=KNOBS_DICT)
    resolve_synth_func = partial(resolve_to_function, get_func=synths)

    # Apply defaults aligned with previous behavior
    if pipeline is None:
        pipeline = DFLT_PIPELINE_NAME
    if knobs is None:
        knobs = DFLT_KNOBS
    if synth is None:
        synth = DFLT_SYNTH_FUNC

    video_features = resolve_video_features(video_features)
    if not pipeline:
        knobs = resolve_knobs(knobs)
        synth = resolve_synth_func(synth)
    else:
        pipeline_getter = resolve_pipeline(pipeline)
        pipeline_components = pipeline_getter()
        knobs = pipeline_components.get("knobs", knobs or DFLT_KNOBS)
        synth = pipeline_components.get("synth", synth or DFLT_SYNTH_FUNC)
        knobs = resolve_knobs(knobs)
        synth = resolve_synth_func(synth)

    # If requested, override freq_trans by wrapping the knobs function
    if freq_trans_override != "__USE_DEFAULT__":
        try:
            from inspect import signature

            if 'freq_trans' in signature(knobs).parameters:
                base_knobs = knobs

                def _knobs_with_override(vf):
                    return base_knobs(vf, freq_trans=freq_trans_override)

                knobs = _knobs_with_override
        except Exception:
            pass

    print(f"{knobs=}, {synth=}")

    log_video_features = log_video_features or do_nothing
    log_knobs = log_knobs or do_nothing

    cap = cv2.VideoCapture(0)
    recognizer = HandGestureRecognizer()

    if draw_on_screen is None:
        draw_on_screen = _get_default_draw_on_screen()

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

    try:
        with synth_obj:
            while cap.isOpened():
                try:
                    keyboard_data = read_keyboard()
                    keyboard_feature_vector(keyboard_data)
                    img = read_camera(cap)
                    hand_detection = recognizer.find_hands(img)
                    _video_features = video_features(hand_detection)
                    log_video_features(_video_features)
                    _audio_features = knobs(_video_features)
                    _audio_features = ensure_plain_types(_audio_features)
                    if _audio_features:
                        if only_keep_new_freqs:
                            _audio_features, previous_data = (
                                filter_unchanged_frequencies(
                                    _audio_features, previous_data
                                )
                            )
                        synth_obj(**_audio_features)
                    log_knobs(_audio_features)
                    if draw_on_screen:
                        img = draw_on_screen(
                            recognizer, img, hand_detection, _audio_features
                        )
                    cv2.imshow(window_name, img)
                except (CameraReadError, KeyboardBreakSignal):
                    break
    finally:
        synth_obj.stop_recording()
        recording = synth_obj.get_recording()
        print(f"\n---> Recorded {len(recording)} control events\n")
        if record_to_file:
            try:
                output_path = (
                    record_to_file
                    if isinstance(record_to_file, str)
                    else 'theremin_recording.wav'
                )
                synth_obj.render_events(output_filepath=output_path)
                print(f"Saved audio recording to {output_path}")
            except Exception as e:  # pragma: no cover - safety
                print(f"Warning: Failed to render events: {e}")
        cap.release()
        cv2.destroyAllWindows()


def list_components(param_value, components_dict, description, component_describer=Sig):
    """If the user passed the literal string 'list', print available component names.

    Returns True if listing was performed so caller can early-return.
    """
    if isinstance(param_value, str) and param_value == 'list':
        print(f"Available {description}:")
        for name in sorted(components_dict.keys()):
            func = components_dict[name]
            print(f"  - {name}{component_describer(func)}")
        return True
    return False


@argh.arg(
    '--pipeline',
    '-p',
    nargs='?',
    const='list',
    help='Audio pipeline name (use without argument to list available pipelines)',
)
@argh.arg(
    '--synth',
    '-s',
    nargs='?',
    const='list',
    help='Synthesizer function name (use without argument to list available synths)',
)
@argh.arg(
    '--knobs',
    '-k',
    nargs='?',
    const='list',
    help='Audio knobs function name (use without argument to list available knobs)',
)
@argh.arg(
    '--video-features',
    '-v',
    nargs='?',
    const='list',
    help='Video features function name (use without argument to list available video features)',
)
@argh.arg('--log-video-features', help='Log hand features', default=False)
@argh.arg('--log-knobs', help='Log audio features', default=False)
@argh.arg(
    '-r',
    '--record-to-file',
    help='Filename to save recording',
    default='theremin_recording.wav',
)
@argh.arg('-n', '--no-recording', help='Disable recording', default=False)
@argh.arg(
    '-w', '--window-name', help='Window title', default='Theremin with Hand Tracking'
)
@argh.arg(
    '--scale',
    nargs='?',
    const='list',
    help='Scale for snapping. Use flag without value to list scales; use none/null/off to disable. Default uses audio.DFLT_SCALE.',
    default=None,
)
def theremin_cli(
    pipeline: str | None = 'theremin',
    video_features: str | None = 'many_video_features',
    knobs: str | None = 'theremin_knobs',
    synth: str | None = 'theremin_synth',
    log_video_features: bool = False,
    log_knobs: bool = False,
    record_to_file: str = 'theremin_recording.wav',
    no_recording: bool = False,
    window_name: str = 'Theremin with Hand Tracking',
    scale: str | None = None,
):
    """Run the theremin: map video/keyboard input to synthesized audio.

    Architecture Overview:

    Input: Video/Keyboard
             |
             v
    +----------------------------------------------------------+
    | 1. Sensor Reading (cv2.VideoCapture, cv2.waitKey)       |
    +----------------------------------------------------------+
             |
             v
    +----------------------------------------------------------+
    | 2. Feature Extraction: --video-features                 |
    |    (e.g. many_video_features)                            |
    |    Extracts: hand positions, gestures, openness, etc.    |
    +----------------------------------------------------------+
             |
             v
    +----------------------------------------------------------+
    | 3. Feature Mapping: --knobs (e.g. theremin_knobs)       |
    |    Maps video features → audio params (freq, volume...) |
    +----------------------------------------------------------+
             |
             v
    +----------------------------------------------------------+
    | 4. Synthesis: --synth (e.g. theremin_synth)             |
    |    Generates audio from parameters                       |
    +----------------------------------------------------------+
             |
             v
       [Audio Output + Recording]

    Note: --pipeline combines steps 2-4 in pre-configured packages
          (e.g. "theremin", "two_voice", "simple_sine")

    Parameters:
        pipeline: Pre-configured pipeline name or 'list' to show available
        video_features: Feature extraction function or 'list' to show available
        knobs: Audio parameter mapping function or 'list' to show available
        synth: Synthesizer function or 'list' to show available
        log_video_features: Enable logging of extracted video features
        log_knobs: Enable logging of audio parameters
        record_to_file: Filename for audio recording (default: theremin_recording.wav)
        no_recording: Disable audio recording
        window_name: Title for the video display window
        scale: Musical scale for frequency snapping or 'list' to show available scales
    """
    # Lazy import of component registries ONLY if listing or running
    from theremin.audio import (
        synths,
        knobs as knobs_dict,
        pipelines,
        resolve_scale_to_freq_trans,
        DFLT_SCALE,
    )
    from theremin.video_features import hand_feature_funcs

    # Handle listing of components
    if list_components(pipeline, pipelines, 'pipelines'):
        return
    if list_components(synth, synths, 'synthesizer functions'):
        return
    if list_components(knobs, knobs_dict, 'audio feature mapping functions'):
        return
    if list_components(
        video_features, hand_feature_funcs, 'hand feature extraction functions'
    ):
        return

    # Handle scale listing/help when flag provided without value
    if scale == 'list':
        try:
            from tonal.notes import list_scales_string

            print(list_scales_string())
        except Exception as e:
            print(f"Could not list scales ({e.__class__.__name__}: {e})")
        return

    if no_recording:
        record_to_file = False

    # Validate scale if provided and not a disable marker
    scale_disable_markers = {"none", "null", "off", "false", "no", "0", ""}
    if isinstance(scale, str) and scale.strip().lower() not in scale_disable_markers:
        try:
            from tonal.notes import scale_params, IncorrectScaleSpecification

            # Will raise IncorrectScaleSpecification on failure
            scale_params(scale)
        except IncorrectScaleSpecification as e:
            print(f"{e.__class__.__name__}: {e}")
            return
        except Exception as e:
            # Non-fatal: fall back to runtime behavior, but inform the user
            print(
                "Warning: unexpected issue while validating scale; proceeding anyway. "
                f"{e.__class__.__name__}: {e}"
            )

    # Resolve scale argument to freq transformation override
    freq_trans_override = resolve_scale_to_freq_trans(scale)

    log_video_features_cb = print_json_if_possible if log_video_features else None
    log_knobs_cb = print_json_if_possible if log_knobs else None

    # Pick a draw_on_screen that reflects the chosen scale guides
    draw_on_screen = _get_default_draw_on_screen_for_scale(scale)

    # No wrapping here; let run_theremin handle freq_trans override injection
    knobs_arg = knobs

    run_theremin(
        pipeline=pipeline,
        video_features=video_features,
        knobs=knobs_arg,
        synth=synth,
        log_video_features=log_video_features_cb,
        log_knobs=log_knobs_cb,
        record_to_file=record_to_file,
        window_name=window_name,
        freq_trans_override=freq_trans_override,
        draw_on_screen=draw_on_screen,
    )
