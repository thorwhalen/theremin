## audiostream

**Audio input feature extraction for Python**

A framework-agnostic package for extracting features from audio input streams. Wraps aubio and sounddevice for real-time, low-latency audio feature extraction.

Useful for pitch detection, onset detection, rhythm analysis, voice control, and more.

## Installation

```bash
pip install audiostream
# Or for development
pip install -e .
```

## Dependencies

- `sounddevice` - Low-latency audio I/O (installed automatically)
- `aubio` - Real-time audio feature extraction (installed automatically)
- `numpy` - Array operations (installed automatically)

## Quick Start

```python
from audiostream import AudioInputStream, extract_audio_features
import time

# Create and start audio input
stream = AudioInputStream(sample_rate=44100, block_size=2048)
stream.start()

# Process audio in real-time
while True:
    chunk = stream.get_audio_chunk()
    if chunk is not None:
        # Extract features
        features = extract_audio_features(chunk, sample_rate=44100)

        if features.get('has_pitch'):
            print(f"Pitch: {features['pitch']:.1f} Hz")

        if features.get('onset_detected'):
            print("Onset detected!")

        if features.get('voice_activity'):
            print(f"Amplitude: {features['amplitude']:.3f}")

    time.sleep(0.01)
```

## Feature Extraction

### Pitch Detection

```python
from audiostream import extract_pitch_features

features = extract_pitch_features(audio_chunk, sample_rate=44100)
print(f"Pitch: {features['pitch']} Hz")
print(f"Confidence: {features['pitch_confidence']}")
print(f"Has pitch: {features['has_pitch']}")
```

### Onset Detection

```python
from audiostream import extract_onset_features

features = extract_onset_features(audio_chunk, sample_rate=44100)
if features['onset_detected']:
    print("Note onset detected!")
```

### Rhythm/Tempo Detection

```python
from audiostream import extract_rhythm_features

features = extract_rhythm_features(audio_chunk, sample_rate=44100)
if features['beat_detected']:
    print(f"Beat at {features['tempo_bpm']} BPM")
```

### Comprehensive Features

```python
features = extract_audio_features(
    audio_chunk,
    sample_rate=44100,
    include_pitch=True,
    include_onset=True,
    include_rhythm=True
)

# Features include:
# - amplitude (RMS)
# - voice_activity (bool)
# - pitch (Hz)
# - pitch_confidence
# - has_pitch (bool)
# - onset_detected (bool)
# - beat_detected (bool)
# - tempo_bpm
```

## API Reference

### AudioInputStream

**Constructor:**
```python
AudioInputStream(
    sample_rate=44100,
    block_size=2048,
    channels=1,
    dtype='float32'
)
```

**Methods:**
- `start()` - Start audio capture
- `stop()` - Stop audio capture
- `get_audio_chunk()` - Get next audio chunk (NumPy array or None)
- `get_queue_size()` - Number of chunks in queue
- `clear_queue()` - Clear all waiting chunks

### Feature Extraction Functions

All functions return dictionaries with extracted features and timestamp.

**extract_audio_features(audio_chunk, sample_rate=44100, timestamp=None, include_pitch=True, include_onset=True, include_rhythm=False)**
- Comprehensive feature extraction

**extract_pitch_features(audio_chunk, sample_rate=44100, timestamp=None)**
- Pitch and pitch confidence

**extract_onset_features(audio_chunk, sample_rate=44100, timestamp=None)**
- Onset detection

**extract_rhythm_features(audio_chunk, sample_rate=44100, timestamp=None)**
- Beat and tempo detection

## Use Cases

- **Music applications**: Real-time pitch-to-MIDI conversion
- **Voice control**: Speech onset detection for wake words
- **Audio analysis**: Feature extraction for ML pipelines
- **Rhythm games**: Beat detection for gameplay
- **Tuning apps**: Pitch detection for instrument tuning
- **Audio-reactive visuals**: Sound-driven animations

## Performance

- **Latency**: 5-50ms depending on block_size
  - block_size=256: ~6ms at 44.1kHz (lowest latency)
  - block_size=1024: ~23ms (balanced)
  - block_size=2048: ~46ms (higher quality)

- **CPU Usage**: Low (aubio is C-based)
- **Real-time**: Suitable for 30+ FPS applications

## Platform Support

- **All platforms**: Full support via sounddevice
- **Low latency**: Optimized for real-time applications

## License

MIT License
