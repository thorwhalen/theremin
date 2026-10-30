# theremin.script_utils

Utility functions for running the theremin scripts.

This module focuses on fast CLI startup by deferring all heavy imports (cv2, mediapipe,
pyo, theremin.audio, etc.) until they are actually needed (lazy import pattern).
Only lightweight stdlib / small packages are imported at module import time so
`theremin -h` remains responsive.

### Functions

| [`keyboard_feature_vector`](#theremin.script_utils.keyboard_feature_vector)(key_code)                 | Convert a raw key code into a small feature dict; raise if it's a break key.   |
|----------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------|
| [`list_components`](#theremin.script_utils.list_components)(param_value, ...[, ...])          | If the user passed the literal string 'list', print available component names. |
| [`print_json_if_possible`](#theremin.script_utils.print_json_if_possible)(x)                         | Print object as JSON if serializable; fall back to raw `repr`; add blank line. |
| [`print_plus_newline`](#theremin.script_utils.print_plus_newline)(x)                             | Print value plus an extra blank line (simple pretty printing helper).          |
| [`read_camera`](#theremin.script_utils.read_camera)(cap)                                  | Read and horizontally flip a camera frame; raise on failure.                   |
| [`read_keyboard`](#theremin.script_utils.read_keyboard)([wait_time])                        | Return last key code (or -1) using cv2.waitKey, importing cv2 lazily.          |
| [`run_theremin`](#theremin.script_utils.run_theremin)(\*[, video_features, pipeline, ...]) | Run the realtime theremin loop.                                                |
| [`theremin_cli`](#theremin.script_utils.theremin_cli)([pipeline, video_features, ...])     | Run the theremin: map video/keyboard input to synthesized audio.               |

### Exceptions

| [`CameraReadError`](#theremin.script_utils.CameraReadError)     | Raised when a frame cannot be read from the camera.                        |
|----------------------------------------------------------------------|----------------------------------------------------------------------------|
| [`KeyboardBreakSignal`](#theremin.script_utils.KeyboardBreakSignal) | Raised internally to break the main loop when a designated key is pressed. |

### *exception* theremin.script_utils.CameraReadError

Bases: [`Exception`](https://docs.python.org/3/builtins/exceptions.html#Exception)

Raised when a frame cannot be read from the camera.

### *exception* theremin.script_utils.KeyboardBreakSignal

Bases: [`Exception`](https://docs.python.org/3/builtins/exceptions.html#Exception)

Raised internally to break the main loop when a designated key is pressed.

### theremin.script_utils.keyboard_feature_vector(key_code)

Convert a raw key code into a small feature dict; raise if it’s a break key.

* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]

### theremin.script_utils.list_components(param_value, components_dict, description, component_describer=<class 'i2.signatures.Sig'>)

If the user passed the literal string ‘list’, print available component names.

Returns True if listing was performed so caller can early-return.

### theremin.script_utils.print_json_if_possible(x)

Print object as JSON if serializable; fall back to raw `repr`; add blank line.

### theremin.script_utils.print_plus_newline(x)

Print value plus an extra blank line (simple pretty printing helper).

### theremin.script_utils.read_camera(cap)

Read and horizontally flip a camera frame; raise on failure.

### theremin.script_utils.read_keyboard(wait_time=5)

Return last key code (or -1) using cv2.waitKey, importing cv2 lazily.

* **Return type:**
  [`int`](https://docs.python.org/3/builtins/functions.html#int)

### theremin.script_utils.run_theremin(, video_features='many_video_features', pipeline=None, knobs=None, synth=None, log_video_features=None, log_knobs=None, record_to_file='theremin_recording.wav', window_name='Hand Gesture Recognition with Theremin', draw_on_screen=None, only_keep_new_freqs=True, freq_trans_override='_\_USE_DEFAULT_\_')

Run the realtime theremin loop.

Lazy-loads heavy dependencies (cv2, mediapipe, pyo, pipelines) only if actually
executing the loop (i.e. not when just requesting CLI help). Handles:

> - Resolving named registry entries (video features, knobs/pipelines, synth)
> - Video capture + hand feature extraction
> - Mapping features to audio parameters (knobs)
> - Optional logging & drawing
> - Synth control & (optionally) recording, filtering unchanged freqs

### theremin.script_utils.theremin_cli(pipeline='theremin', video_features='many_video_features', knobs='theremin_knobs', synth='theremin_synth', log_video_features=False, log_knobs=False, record_to_file='theremin_recording.wav', no_recording=False, window_name='Theremin with Hand Tracking', scale=None)

Run the theremin: map video/keyboard input to synthesized audio.

Architecture Overview:

Input: Video/Keyboard
: <br/>
  <br/>
  v

> <br/>

> v
> <br/>

> v
> <br/>

> v
> > <br/>

> > v

> [Audio Output + Recording]

#### NOTE
–pipeline combines steps 2-4 in pre-configured packages

> (e.g. “theremin”, “two_voice”, “simple_sine”)

* **Parameters:**
  * **pipeline** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Pre-configured pipeline name or ‘list’ to show available
  * **video_features** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Feature extraction function or ‘list’ to show available
  * **knobs** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Audio parameter mapping function or ‘list’ to show available
  * **synth** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Synthesizer function or ‘list’ to show available
  * **log_video_features** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Enable logging of extracted video features
  * **log_knobs** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Enable logging of audio parameters
  * **record_to_file** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Filename for audio recording (default: theremin_recording.wav)
  * **no_recording** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Disable audio recording
  * **window_name** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Title for the video display window
  * **scale** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Musical scale for frequency snapping or ‘list’ to show available scales
