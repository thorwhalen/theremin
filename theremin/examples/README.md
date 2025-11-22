# Theremin Framework Examples

This directory contains comprehensive examples demonstrating the theremin framework's modular architecture and integration with related packages.

## Examples

### 1. Keyboard Synthesizer (`keyboard_synth.py`)

Piano-like keyboard playing using streamkeys + synthflow.

**Features:**
- QWERTY keyboard mapped to musical notes
- Real-time audio synthesis
- Both simple and meshed DAG versions

**Run:**
```bash
python keyboard_synth.py
```

**Controls:**
- Home row (A-L): Play notes
- ESC: Quit

### 2. Video Theremin (`video_theremin.py`)

Hand-tracking based theremin using vidstream + synthflow.

**Features:**
- MediaPipe hand tracking
- X position → Pitch
- Y position → Volume
- Real-time visual feedback

**Run:**
```bash
python video_theremin.py
```

**Controls:**
- Move hand left/right: Change pitch
- Move hand up/down: Change volume
- Press 'q': Quit

### 3. Multimodal Theremin (`multimodal.py`)

Combines keyboard + trackpad + video for rich expressivecontrol.

**Features:**
- Keyboard: Note selection
- Trackpad: Effect controls (filter cutoff, resonance)
- Video: Pitch bend and modulation
- Effect processing

**Run:**
```bash
python multimodal.py
```

**Controls:**
- Keyboard: Play notes
- Trackpad: Move to control filter
- Hand position: Adds pitch bend
- ESC: Quit

### 4. Audio Reactive (`audio_reactive.py`)

Audio input → feature extraction → responsive synthesis.

**Features:**
- Real-time pitch detection
- Onset detection
- Audio-driven synthesis
- Feedback loop

## Architecture Patterns

All examples demonstrate the theremin framework architecture:

```
Input Sensors → Feature Extraction → Feature Mapping → Synthesis → Audio Output
      ↓                  ↓                   ↓                ↓
  {raw_data}        {features}          {parameters}     audio_bytes
```

### Using meshed DAG

The framework uses meshed for automatic function wiring:

```python
from meshed import DAG

# Define functions with matching parameter names
def read_sensor() -> dict:
    return {'sensor_value': 0.5, 'timestamp': time.time()}

def extract_features(sensor_value, timestamp) -> dict:
    return {'feature_x': sensor_value * 2, 'timestamp': timestamp}

def map_to_params(feature_x) -> dict:
    return {'frequency': 200 + feature_x * 1800}

def synthesize(frequency):
    return generate_audio(frequency)

# Auto-wire into pipeline
pipeline = DAG([read_sensor, extract_features, map_to_params, synthesize])

# Execute
audio = pipeline()
```

## Testing Examples

Examples include both production and testing modes:

### Production Mode
```python
# Use real sensors
keyboard = KeyboardFeatureStream()
keyboard.start()
```

### Testing Mode
```python
# Use pre-recorded features
from theremin.framework import StreamPlayer

player = StreamPlayer('test_features.json')
for features in player:
    audio = pipeline(features)
```

## Requirements

Each example lists its dependencies at the top. Install with:

```bash
# Core framework
pip install meshed dol

# Related packages
pip install -e related_packages/streamkeys
pip install -e related_packages/streamtouch
pip install -e related_packages/vidstream
pip install -e related_packages/audiostream
pip install -e related_packages/synthflow
pip install -e related_packages/accompanist

# External dependencies
pip install opencv-python mediapipe pynput sounddevice aubio numpy scipy
```

## Creating Your Own Examples

1. **Define Your Pipeline Functions:**
   - Each function should return a dict
   - Use descriptive parameter names
   - Keep functions pure (no side effects)

2. **Wire with meshed:**
   - Functions auto-connect by parameter names
   - Output dict keys match input parameter names

3. **Test Separately:**
   - Test each function independently
   - Use StreamPlayer for deterministic tests
   - Use AudioVerifier to validate audio output

4. **Compose:**
   - Combine functions into Pipeline
   - Run and iterate!

## Next Steps

- Explore the related_packages/ directory for package-specific examples
- Check tests/ for comprehensive testing patterns
- Read the framework/ module for architecture details

## License

MIT License
