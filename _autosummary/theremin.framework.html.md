# theremin.framework

Theremin framework core - meshed.slabs based architecture.

This module provides the core framework for building modular signal processing
pipelines using functional composition and automatic parameter wiring.

### Classes

| [`SensorReader`](#theremin.framework.SensorReader)()                        | Base class for sensor reading functions.                       |
|----------------------------------------------------------------------------------------|----------------------------------------------------------------|
| [`FeatureExtractor`](#theremin.framework.FeatureExtractor)()                    | Base class for feature extraction functions.                   |
| [`FeatureMapper`](#theremin.framework.FeatureMapper)()                       | Base class for feature mapping functions (knobs).              |
| [`Synthesizer`](#theremin.framework.Synthesizer)()                         | Base class for synthesis functions.                            |
| [`Pipeline`](#theremin.framework.Pipeline)(functions[, name, validate]) | Theremin processing pipeline using meshed DAG for auto-wiring. |
| [`CalibrationStore`](#theremin.framework.CalibrationStore)([base_path])         | Store for sensor calibration data using dict-like interface.   |
| [`PresetStore`](#theremin.framework.PresetStore)([base_path])              | Store for pipeline presets and configurations.                 |
| [`StreamPlayer`](#theremin.framework.StreamPlayer)(feature_file)            | Replay feature streams from JSON for deterministic testing.    |
| [`AudioVerifier`](#theremin.framework.AudioVerifier)()                       | Verify audio output quality and correctness.                   |

### *class* theremin.framework.AudioVerifier

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Verify audio output quality and correctness.

Provides utilities for testing audio generation:

- Frequency content verification
- Silence detection
- Amplitude checks
- Spectral analysis

### Example

```pycon
>>> verifier = AudioVerifier()
>>> audio = generate_audio(frequency=440, amplitude=0.5)
>>> assert verifier.verify_frequency_content(audio, 44100, 440)
>>> assert verifier.verify_not_silent(audio)
```

#### *static* has_no_nan_or_inf(audio)

Check that audio has no NaN or infinite values.

* **Parameters:**
  **audio** (`ndarray`) – Audio samples
* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)
* **Returns:**
  True if audio is valid

#### *static* measure_rms(audio)

Measure RMS amplitude of audio.

* **Parameters:**
  **audio** (`ndarray`) – Audio samples
* **Return type:**
  [`float`](https://docs.python.org/3/builtins/functions.html#float)
* **Returns:**
  RMS amplitude

#### *static* verify_amplitude_range(audio, min_amp=0.0, max_amp=1.0)

Verify audio amplitude is within expected range.

* **Parameters:**
  * **audio** (`ndarray`) – Audio samples
  * **min_amp** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Minimum expected amplitude
  * **max_amp** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Maximum expected amplitude
* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)
* **Returns:**
  True if amplitude is within range

#### *static* verify_frequency_content(audio, sample_rate, expected_freq, tolerance=10.0)

Check if expected frequency is present in audio.

* **Parameters:**
  * **audio** (`ndarray`) – Audio samples (NumPy array)
  * **sample_rate** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Sample rate in Hz
  * **expected_freq** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Expected frequency in Hz
  * **tolerance** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Tolerance in Hz
* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)
* **Returns:**
  True if expected frequency is present

#### *static* verify_not_silent(audio, min_std=0.01)

Ensure audio has variation (not silent).

* **Parameters:**
  * **audio** (`ndarray`) – Audio samples
  * **min_std** ([`float`](https://docs.python.org/3/builtins/functions.html#float)) – Minimum standard deviation
* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)
* **Returns:**
  True if audio has variation

### *class* theremin.framework.CalibrationStore(base_path='./data/calibration')

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Store for sensor calibration data using dict-like interface.

Uses dol for uniform storage abstraction, but falls back to simple
file-based storage if dol is not available.

### Example

```pycon
>>> store = CalibrationStore('./data/calibration')
>>> store['camera_0'] = {'offset_x': 0.1, 'offset_y': 0.2}
>>> calibration = store['camera_0']
```

#### keys()

Get all sensor names with calibration.

### *class* theremin.framework.FeatureExtractor

Bases: [`ABC`](https://docs.python.org/3/library/abc.html#abc.ABC)

Base class for feature extraction functions.

Feature extractors transform raw sensor data into meaningful features.
They consume dicts from sensors and output dicts with extracted features.

#### *abstractmethod* extract(\*\*kwargs)

Extract features from sensor data.

* **Parameters:**
  **\*\*kwargs** – Sensor data (matched by parameter names)
* **Return type:**
  [`Dict`](https://docs.python.org/3/library/typing.html#typing.Dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]
* **Returns:**
  Dictionary with extracted features

### *class* theremin.framework.FeatureMapper

Bases: [`ABC`](https://docs.python.org/3/library/abc.html#abc.ABC)

Base class for feature mapping functions (knobs).

Feature mappers transform extracted features into synthesis parameters.
They define the creative mapping between input features and sound.

#### *abstractmethod* map(\*\*kwargs)

Map features to synthesis parameters.

* **Parameters:**
  **\*\*kwargs** – Extracted features (matched by parameter names)
* **Return type:**
  [`Dict`](https://docs.python.org/3/library/typing.html#typing.Dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]
* **Returns:**
  Dictionary with synthesis parameters

### *class* theremin.framework.Pipeline(functions, name='theremin_pipeline', validate=True)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Theremin processing pipeline using meshed DAG for auto-wiring.

The Pipeline composes sensor reading, feature extraction, feature mapping,
and synthesis into a single callable that automatically wires functions
based on parameter name matching.

### Example

```pycon
>>> from meshed import DAG
>>>
>>> def read_video(device_id: int) -> dict:
>>>     return {'video_frame': frame, 'timestamp': time.time()}
>>>
>>> def extract_hands(video_frame, timestamp) -> dict:
>>>     return {'hand_x': 0.5, 'hand_y': 0.3, 'timestamp': timestamp}
>>>
>>> def map_to_audio(hand_x, hand_y) -> dict:
>>>     return {'frequency': 200 + hand_x * 1800, 'amplitude': 1 - hand_y}
>>>
>>> def synth(frequency, amplitude):
>>>     return generate_audio(frequency, amplitude)
>>>
>>> pipeline = Pipeline([read_video, extract_hands, map_to_audio, synth])
>>> audio = pipeline(device_id=0)
```

#### validate_wiring()

Validate that functions wire together correctly.

* **Return type:**
  [`Dict`](https://docs.python.org/3/library/typing.html#typing.Dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]
* **Returns:**
  Dictionary with validation results

### *class* theremin.framework.PresetStore(base_path='./data/presets')

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Store for pipeline presets and configurations.

### Example

```pycon
>>> store = PresetStore('./data/presets')
>>> store['my_theremin'] = {
>>>     'pipeline': 'theremin',
>>>     'frequency_range': [200, 2000],
>>>     'amplitude_range': [0, 1]
>>> }
>>> preset = store['my_theremin']
```

#### keys()

Get all preset names.

### *class* theremin.framework.SensorReader

Bases: [`ABC`](https://docs.python.org/3/library/abc.html#abc.ABC)

Base class for sensor reading functions.

Sensor readers capture raw input and return dictionaries with sensor data.
All sensor readers should output dicts with descriptive keys.

#### *abstractmethod* read()

Read sensor data.

* **Return type:**
  [`Dict`](https://docs.python.org/3/library/typing.html#typing.Dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]
* **Returns:**
  Dictionary with sensor data and metadata (e.g., timestamp)

### *class* theremin.framework.StreamPlayer(feature_file)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Replay feature streams from JSON for deterministic testing.

This enables testing audio synthesis without camera/video hardware
by using pre-recorded feature streams.

### Example

```pycon
>>> player = StreamPlayer('test_features.json')
>>> for features in player:
>>>     audio_params = map_features_to_audio(features)
>>>     audio = synthesize(audio_params)
```

### *class* theremin.framework.Synthesizer

Bases: [`ABC`](https://docs.python.org/3/library/abc.html#abc.ABC)

Base class for synthesis functions.

Synthesizers generate audio from synthesis parameters.
They consume parameter dicts and produce audio output.

#### *abstractmethod* synthesize(\*\*kwargs)

Generate audio from parameters.

* **Parameters:**
  **\*\*kwargs** – Synthesis parameters (matched by parameter names)
* **Return type:**
  [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)
* **Returns:**
  Audio data (format depends on synthesizer)

### Modules

| [`base`](theremin.framework.base.html.md#module-theremin.framework.base)       | Base abstractions for theremin framework.                   |
|--------------------------------------------------------------------------------------------|-------------------------------------------------------------|
| [`storage`](theremin.framework.storage.html.md#module-theremin.framework.storage) | Storage abstractions using dol for calibration and presets. |
| [`testing`](theremin.framework.testing.html.md#module-theremin.framework.testing) | Testing utilities for pipeline validation.                  |
