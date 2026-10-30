# theremin.framework.testing

Testing utilities for pipeline validation.

### Classes

| [`AudioVerifier`](#theremin.framework.testing.AudioVerifier)()            | Verify audio output quality and correctness.                |
|-----------------------------------------------------------------------------|-------------------------------------------------------------|
| [`StreamPlayer`](#theremin.framework.testing.StreamPlayer)(feature_file) | Replay feature streams from JSON for deterministic testing. |

### *class* theremin.framework.testing.AudioVerifier

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

### *class* theremin.framework.testing.StreamPlayer(feature_file)

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
