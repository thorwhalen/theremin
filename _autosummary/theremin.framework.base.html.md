# theremin.framework.base

Base abstractions for theremin framework.

### Classes

| [`FeatureExtractor`](#theremin.framework.base.FeatureExtractor)()                    | Base class for feature extraction functions.                   |
|----------------------------------------------------------------------------------------|----------------------------------------------------------------|
| [`FeatureMapper`](#theremin.framework.base.FeatureMapper)()                       | Base class for feature mapping functions (knobs).              |
| [`Pipeline`](#theremin.framework.base.Pipeline)(functions[, name, validate]) | Theremin processing pipeline using meshed DAG for auto-wiring. |
| [`SensorReader`](#theremin.framework.base.SensorReader)()                        | Base class for sensor reading functions.                       |
| [`Synthesizer`](#theremin.framework.base.Synthesizer)()                         | Base class for synthesis functions.                            |

### *class* theremin.framework.base.FeatureExtractor

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

### *class* theremin.framework.base.FeatureMapper

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

### *class* theremin.framework.base.Pipeline(functions, name='theremin_pipeline', validate=True)

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

### *class* theremin.framework.base.SensorReader

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

### *class* theremin.framework.base.Synthesizer

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
