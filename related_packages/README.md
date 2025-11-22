# Related Packages

This directory contains **independent, framework-agnostic packages** that can be used with or without theremin. Each package outputs clean dictionaries and has zero dependencies on the theremin framework.

## Philosophy

These packages follow the **"related but independent"** principle:

- ✅ Can be used in any Python project
- ✅ Framework agnostic (works with Flask, Django, PyGame, etc.)
- ✅ Output standard dictionaries
- ✅ Well-tested and documented
- ✅ Will be published to PyPI when proven and stable

## Packages

### Input Stream Packages

#### streamkeys - Keyboard Event Streams

Pure keyboard → feature dict conversion.

```python
from streamkeys import KeyboardFeatureStream

stream = KeyboardFeatureStream()
stream.start()

event = stream.get_features()
# {'type': 'keyboard', 'action': 'press', 'key': 'a', ...}
```

**Use cases:** Gaming, music apps, shortcuts, automation

**Install:** `pip install -e streamkeys`

---

#### streamtouch - Trackpad/Mouse Streams

Trackpad gestures → feature dicts.

```python
from streamtouch import TrackpadFeatureStream

stream = TrackpadFeatureStream()
stream.start()

event = stream.get_features()
# {'type': 'trackpad', 'action': 'scroll', 'direction': 'vertical', ...}
```

**Use cases:** UI automation, gesture control, accessibility tools

**Install:** `pip install -e streamtouch`

---

#### vidstream - Video Feature Extraction

Computer vision features (pose, emotion, color, gestures).

```python
from vidstream import HandFeatureExtractor, FaceFeatureExtractor

hand_ex = HandFeatureExtractor()
features = hand_ex.extract(frame)
# {'r_wrist_position': (x, y, z), 'r_openness': 0.7, ...}
```

**Use cases:** Any CV application, pose estimation, emotion detection

**Install:** `pip install -e vidstream`

---

#### audiostream - Audio Input Features

Real-time audio feature extraction (pitch, onset, rhythm).

```python
from audiostream import AudioInputStream, extract_audio_features

stream = AudioInputStream()
stream.start()

chunk = stream.get_audio_chunk()
features = extract_audio_features(chunk)
# {'pitch': 440, 'onset_detected': True, 'amplitude': 0.5, ...}
```

**Use cases:** Audio analysis, music apps, voice control

**Install:** `pip install -e audiostream`

---

### Synthesis and Music Packages

#### synthflow - Dict-Based Synth Control

Standardized wrappers for pyo, SignalFlow, and simple synthesis.

```python
from synthflow import SimpleSynthesizer, PyoSynthesizer

synth = SimpleSynthesizer()
audio = synth.generate({
    'frequency': 440,
    'amplitude': 0.5,
    'waveform': 'sine',
    'duration': 1.0
})
```

**Use cases:** Audio synthesis, sound generation, testing

**Install:** `pip install -e synthflow`

---

#### accompanist - Music Accompaniment

Chord progressions, MIDI processing, harmonic analysis.

```python
from accompanist import generate_progression, chord_to_frequencies

prog = generate_progression('pop', length=4)
freqs = prog.get_frequencies(0)  # Frequencies for first chord
```

**Use cases:** Music generation, improvisation, education

**Install:** `pip install -e accompanist`

---

## Installation

### Install All Packages (Development)

```bash
# From theremin root directory
for pkg in streamkeys streamtouch vidstream audiostream synthflow accompanist; do
    pip install -e related_packages/$pkg
done
```

### Install Individual Package

```bash
cd related_packages/streamkeys
pip install -e .
```

### Install External Dependencies

Some packages have optional external dependencies:

```bash
# For streamkeys and streamtouch
pip install pynput

# For vidstream
pip install opencv-python mediapipe
pip install deepface  # Optional, for emotion detection

# For audiostream
pip install sounddevice aubio numpy

# For synthflow (optional)
pip install pyo scipy

# For accompanist (optional)
pip install mido
```

## Using Packages Outside Theremin

All packages are designed to work standalone:

### Example: Keyboard to MIDI in Flask

```python
from flask import Flask
from streamkeys import KeyboardFeatureStream, keyboard_to_midi_features

app = Flask(__name__)
keyboard = KeyboardFeatureStream()
keyboard.start()

@app.route('/next_note')
def next_note():
    event = keyboard.get_features()
    if event:
        midi = keyboard_to_midi_features(event)
        return midi
    return {}
```

### Example: Video Pose in PyGame

```python
import pygame
from vidstream import PoseFeatureExtractor

extractor = PoseFeatureExtractor()
camera = pygame.camera.Camera(...)

while running:
    frame = camera.get_image()
    features = extractor.extract(frame)
    # Use features to control game...
```

## Package Structure

Each package follows the same structure:

```
package_name/
├── src/
│   ├── __init__.py          # Public API
│   ├── main_module.py       # Core functionality
│   └── utils.py             # Helper functions
├── tests/
│   └── test_*.py            # Tests
├── README.md                # Package documentation
└── setup.py                 # Installation config
```

## Testing

Each package has comprehensive tests:

```bash
cd related_packages/streamkeys
pytest tests/
```

## Publishing to PyPI

When packages are proven and stable, they will be published:

```bash
cd related_packages/streamkeys
python setup.py sdist bdist_wheel
twine upload dist/*
```

## Contributing

To add a new related package:

1. Create directory in `related_packages/`
2. Follow the package structure above
3. Ensure zero dependencies on theremin
4. Output clean dictionaries
5. Write comprehensive tests
6. Document thoroughly
7. Add to this README

## Why Independent Packages?

1. **Reusability**: Useful beyond theremin project
2. **Testing**: Easier to test in isolation
3. **Maintenance**: Independent release cycles
4. **Clarity**: Clear boundaries and responsibilities
5. **Distribution**: Can publish to PyPI separately

## Future Packages

Potential packages for future development:

- **gesturelib**: Advanced gesture recognition
- **beatstream**: Advanced rhythm and beat detection
- **voicecontrol**: Voice command processing
- **midiflow**: Enhanced MIDI utilities
- **audioeffects**: More audio effects processors

## License

Each package is MIT licensed and can be used independently.
