# Modular Live Signal Processing Framework: Theremin Project Specifications

**Real-time sensor-to-audio synthesis reimagined with functional composition patterns**

This specification delivers a comprehensive architecture for building the theremin framework—a Python system that transforms sensor streams (video, keyboard, trackpad, audio) into generated audio using meshed.slabs for self-assembling components, functional programming patterns, and dict-based data flow. The research identifies 50+ production-ready packages across seven project areas, with detailed integration patterns, testing strategies, and migration paths from traditional OOP to functional architectures.

## Background and architecture vision

The theremin framework processes signals through a clean pipeline: **Sensor Reading → Feature Extraction → Feature Mapping (Knobs) → Synthesis → Audio Output**. The core innovation lies in using i2mint ecosystem packages—particularly meshed for automatic function wiring via argument-name matching—to create self-assembling components that connect based on shared parameter names. This approach eliminates boilerplate wiring code while maintaining clear data flow through dict-based interfaces.

Current implementations use MediaPipe for video features and OpenCV for basic keyboard input, but the architecture lacks modularity and testing infrastructure. This specification provides pathways to refactor into slabs-based components while expanding capabilities across multiple input modalities and synthesis backends. Each project area includes package recommendations, architecture patterns, code examples, and testing strategies compatible with the functional programming principles central to the i2mint philosophy.

## A. CORE THEREMIN REFACTORING

### Migrating to meshed.slabs architecture

The fundamental shift moves from object-oriented inheritance hierarchies to composable pure functions automatically wired by argument names. **meshed** (https://github.com/i2mint/meshed, PyPI: https://pypi.org/project/meshed/) provides DAG-based function composition where functions connect when output names match input parameter names.

**Architecture Pattern:**

```
Sensor Functions → Feature Extractors → Mapping Functions → Synthesis Functions
         ↓                  ↓                   ↓                    ↓
    {raw_data}        {features}           {parameters}         audio_bytes
```

Each stage outputs dictionaries consumed by the next stage through automatic parameter matching.

**Refactored Pipeline Example:**

```python
from meshed import DAG

def read_video_sensor(device_id: int) -> dict:
    """Read video frame from camera"""
    frame = cv2.VideoCapture(device_id).read()[1]
    return {'video_frame': frame, 'timestamp': time.time()}

def extract_hand_features(video_frame, timestamp) -> dict:
    """Extract MediaPipe hand landmarks"""
    results = mp_hands.process(video_frame)
    if not results.multi_hand_landmarks:
        return {'hand_x': 0.5, 'hand_y': 0.5, 'hand_z': 0, 'timestamp': timestamp}
    
    landmarks = results.multi_hand_landmarks[0].landmark
    return {
        'hand_x': landmarks[9].x,  # Middle finger base
        'hand_y': landmarks[9].y,
        'hand_z': landmarks[9].z,
        'timestamp': timestamp
    }

def map_hand_to_audio_params(hand_x, hand_y, hand_z) -> dict:
    """Map hand position to frequency and amplitude"""
    frequency = 200 + hand_x * 1800  # 200-2000 Hz range
    amplitude = (1 - hand_y) * 0.8   # Inverted Y for amplitude
    return {'frequency': frequency, 'amplitude': amplitude}

def synthesize_audio(frequency, amplitude, sample_rate=44100) -> bytes:
    """Generate audio chunk"""
    duration = 0.1  # 100ms chunks
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = amplitude * np.sin(2 * np.pi * frequency * t)
    return audio.tobytes()

# Auto-wire into pipeline
theremin_pipeline = DAG([
    read_video_sensor,
    extract_hand_features,
    map_hand_to_audio_params,
    synthesize_audio
])

# Execute
audio_chunk = theremin_pipeline(device_id=0)
```

**Key Benefits:**

The automatic wiring eliminates explicit connection code. Adding new features requires only defining functions with matching parameter names. The pipeline becomes a declarative specification of data transformations rather than imperative object orchestration.

### Storage abstraction with dol

**dol** (https://github.com/i2mint/dol, PyPI: https://pypi.org/project/dol/) provides uniform Mapping interfaces for storing calibration data, presets, and test fixtures. All stores implement dict-like access regardless of backend (files, databases, cloud storage).

```python
from dol import Store
import json

class CalibrationStore(Store):
    """Persist sensor calibration data"""
    
    def _id_of_key(self, sensor_name):
        return f"calibration_{sensor_name}.json"
    
    def _data_of_obj(self, calibration_dict):
        return json.dumps(calibration_dict)
    
    def _obj_of_data(self, json_string):
        return json.loads(json_string)

# Usage in pipeline
calibration_db = CalibrationStore('./data/calibration')

def calibrated_sensor_read(device_id, calibration_data):
    raw_values = read_sensors(device_id)
    return apply_calibration(raw_values, calibration_data)

# Inject calibration from store
pipeline_with_calibration = DAG([
    lambda device_id: calibrated_sensor_read(
        device_id, 
        calibration_db['camera_0']
    ),
    extract_hand_features,
    map_hand_to_audio_params,
    synthesize_audio
])
```

### Testing strategy with precomputed features

The testing architecture separates video processing from audio synthesis testing by recording feature streams as JSON, enabling deterministic testing without camera hardware.

**Test Video Processing:**

```python
import pytest
import json

def test_extract_features_from_video(datadir, tmp_path):
    """Extract features from theremin_test_1.mp4 and save as JSON"""
    video_path = datadir / "theremin_test_1.mp4"
    
    feature_recorder = []
    cap = cv2.VideoCapture(str(video_path))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        features = extract_hand_features(
            video_frame=frame,
            timestamp=cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        )
        feature_recorder.append(features)
    
    cap.release()
    
    # Save features for audio testing
    output_path = tmp_path / "test_1_features.json"
    with open(output_path, 'w') as f:
        json.dump(feature_recorder, f, indent=2)
    
    assert len(feature_recorder) > 0
    assert all('hand_x' in f for f in feature_recorder)
```

**Test Audio Mapping:**

```python
def test_features_to_audio_mapping(fixtures_dir, tmp_path):
    """Test feature → audio using precomputed features"""
    # Load recorded features
    with open(fixtures_dir / "features/test_1_features.json") as f:
        features = json.load(f)
    
    audio_chunks = []
    for frame_features in features:
        # Map features to parameters
        params = map_hand_to_audio_params(**frame_features)
        
        # Generate audio
        audio = synthesize_audio(**params)
        audio_chunks.append(np.frombuffer(audio, dtype=np.float64))
    
    # Concatenate and save
    full_audio = np.concatenate(audio_chunks)
    import soundfile as sf
    sf.write(tmp_path / "test_output.wav", full_audio, 44100)
    
    # Verify audio quality
    assert len(full_audio) > 0
    assert not np.any(np.isnan(full_audio))
    assert np.std(full_audio) > 0.01  # Not silent
```

### Audio recording and verification

Use **librosa** (https://pypi.org/project/librosa/) and **soundfile** (https://pypi.org/project/soundfile/) for audio testing, with **speechmetrics** (https://github.com/aliutkus/speechmetrics) for quality verification when comparing against reference audio.

```python
import librosa
import soundfile as sf

class AudioVerifier:
    @staticmethod
    def verify_frequency_content(audio, sr, expected_freq, tolerance=10):
        """Check if expected frequency is present"""
        fft = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(len(audio), 1/sr)
        peak_freq = freqs[np.argmax(np.abs(fft))]
        return abs(peak_freq - expected_freq) < tolerance
    
    @staticmethod
    def verify_not_silent(audio, min_std=0.01):
        """Ensure audio has variation"""
        return np.std(audio) > min_std

def test_audio_verification(tmp_path):
    """Generate and verify audio output"""
    audio = synthesize_audio(frequency=440, amplitude=0.5)
    audio_array = np.frombuffer(audio, dtype=np.float64)
    
    # Save for inspection
    sf.write(tmp_path / "output_440hz.wav", audio_array, 44100)
    
    verifier = AudioVerifier()
    assert verifier.verify_frequency_content(audio_array, 44100, 440)
    assert verifier.verify_not_silent(audio_array)
```

### Integration with i2mint ecosystem

The refactored architecture leverages **creek** (https://github.com/i2mint/creek, PyPI: https://pypi.org/project/creek/) for stream processing with three hook points: `pre_iter`, `data_to_obj`, and `post_filt`.

```python
from creek import Creek

class SensorStreamProcessor(Creek):
    """Real-time sensor stream processing"""
    
    def pre_iter(self, raw_stream):
        """Buffer incoming data"""
        return buffer_stream(raw_stream, buffer_size=1024)
    
    def data_to_obj(self, buffer):
        """Parse sensor packets"""
        return {
            'pitch_voltage': parse_adc_value(buffer[0:4]),
            'volume_voltage': parse_adc_value(buffer[4:8]),
            'timestamp': time.time()
        }
    
    def post_filt(self, sensor_data):
        """Filter invalid readings"""
        return 0.1 < sensor_data['pitch_voltage'] < 4.9

# Usage
sensor_stream = SensorStreamProcessor(sensor_device)
for reading in sensor_stream:
    audio = theremin_pipeline(**reading)
    audio_output.write(audio)
```

## B. KEYBOARD INPUT COMPONENTS PROJECT

### Package recommendations

**pynput** (https://github.com/moses-palmer/pynput, PyPI: https://pypi.org/project/pynput/) is the recommended cross-platform solution for keyboard event capture. It provides non-blocking threading-based event handling with excellent macOS support (requires accessibility permissions).

**Alternative:** The **keyboard** package (https://github.com/boppreh/keyboard) offers hotkey support and zero dependencies but is currently unmaintained and requires sudo on Linux.

**Installation:** `pip install pynput`

### Architecture: Keyboard events to feature streams

```python
from pynput import keyboard
from queue import Queue
import time

class KeyboardFeatureStream:
    """Convert keyboard events to feature dictionaries"""
    
    def __init__(self):
        self.event_queue = Queue()
        self.active_keys = set()
        self.listener = None
    
    def on_press(self, key):
        """Handle key press"""
        try:
            key_char = key.char
        except AttributeError:
            key_char = str(key)
        
        self.active_keys.add(key_char)
        
        event = {
            'type': 'keyboard',
            'action': 'press',
            'key': key_char,
            'active_keys': list(self.active_keys),
            'timestamp': time.time()
        }
        self.event_queue.put(event)
    
    def on_release(self, key):
        """Handle key release"""
        try:
            key_char = key.char
        except AttributeError:
            key_char = str(key)
        
        self.active_keys.discard(key_char)
        
        event = {
            'type': 'keyboard',
            'action': 'release',
            'key': key_char,
            'active_keys': list(self.active_keys),
            'timestamp': time.time()
        }
        self.event_queue.put(event)
    
    def start(self):
        """Start non-blocking listener"""
        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release
        )
        self.listener.start()
    
    def get_features(self):
        """Get next keyboard feature dict"""
        if not self.event_queue.empty():
            return self.event_queue.get()
        return None
```

### Playing tones and chords from QWERTY

Map keyboard rows to musical scales and columns to octaves, enabling piano-like playing.

```python
# Keyboard to note mapping
KEY_TO_NOTE = {
    'a': 60, 's': 62, 'd': 64, 'f': 65,  # C major scale
    'g': 67, 'h': 69, 'j': 71, 'k': 72,
    'q': 72, 'w': 74, 'e': 76, 'r': 77,  # Octave up
    'z': 48, 'x': 50, 'c': 52, 'v': 53   # Octave down
}

def keyboard_to_midi_features(keyboard_event) -> dict:
    """Convert keyboard event to MIDI-like features"""
    if keyboard_event['action'] != 'press':
        return None
    
    key = keyboard_event['key']
    if key not in KEY_TO_NOTE:
        return None
    
    midi_note = KEY_TO_NOTE[key]
    frequency = 440 * (2 ** ((midi_note - 69) / 12))
    
    return {
        'frequency': frequency,
        'amplitude': 0.5,
        'note_on': True,
        'timestamp': keyboard_event['timestamp']
    }

def synthesize_note(frequency, amplitude, note_on) -> bytes:
    """Generate note audio"""
    if not note_on:
        return b''
    
    duration = 0.2  # 200ms note
    sr = 44100
    t = np.linspace(0, duration, int(sr * duration))
    audio = amplitude * np.sin(2 * np.pi * frequency * t)
    
    # ADSR envelope
    attack = int(0.01 * sr)
    release = int(0.05 * sr)
    envelope = np.ones_like(audio)
    envelope[:attack] = np.linspace(0, 1, attack)
    envelope[-release:] = np.linspace(1, 0, release)
    
    return (audio * envelope).astype(np.float32).tobytes()

# Pipeline
keyboard_pipeline = DAG([
    KeyboardFeatureStream().get_features,
    keyboard_to_midi_features,
    synthesize_note
])
```

### Chord playing with simultaneous keys

```python
def keyboard_to_chord_features(keyboard_event) -> dict:
    """Map active keys to chord"""
    active_notes = [
        KEY_TO_NOTE[k] for k in keyboard_event['active_keys'] 
        if k in KEY_TO_NOTE
    ]
    
    if not active_notes:
        return {'frequencies': [], 'amplitude': 0}
    
    frequencies = [
        440 * (2 ** ((note - 69) / 12)) 
        for note in active_notes
    ]
    
    return {
        'frequencies': frequencies,
        'amplitude': 0.5 / len(frequencies),  # Normalize
        'timestamp': keyboard_event['timestamp']
    }

def synthesize_chord(frequencies, amplitude) -> bytes:
    """Generate polyphonic audio"""
    if not frequencies:
        return b''
    
    duration = 0.2
    sr = 44100
    t = np.linspace(0, duration, int(sr * duration))
    
    audio = sum(np.sin(2 * np.pi * f * t) for f in frequencies)
    audio = audio * amplitude
    
    return audio.astype(np.float32).tobytes()
```

### Combining keyboard and video features

Use keyboard for discrete state changes (mode switching, octave shifts) while video provides continuous control.

```python
def combined_feature_mapping(hand_x, hand_y, keyboard_event) -> dict:
    """Combine video hand tracking with keyboard control"""
    # Base frequency from hand position
    base_freq = 200 + hand_x * 1800
    
    # Keyboard modulates frequency (shift octaves)
    if keyboard_event and 'shift' in keyboard_event.get('active_keys', []):
        base_freq *= 2  # Octave up
    
    # Keyboard triggers effects
    effects = {
        'reverb': 'r' in keyboard_event.get('active_keys', []),
        'vibrato': 'v' in keyboard_event.get('active_keys', []),
    }
    
    return {
        'frequency': base_freq,
        'amplitude': 1 - hand_y,
        'effects': effects,
        'timestamp': time.time()
    }
```

### Testing with precomputed keyboard sequences

```python
def test_keyboard_to_audio(fixtures_dir, tmp_path):
    """Test keyboard → audio with recorded event sequence"""
    # Load recorded keyboard events
    with open(fixtures_dir / "keyboard/scale_up.json") as f:
        keyboard_events = json.load(f)
    
    audio_chunks = []
    for event in keyboard_events:
        features = keyboard_to_midi_features(event)
        if features:
            audio = synthesize_note(**features)
            audio_chunks.append(np.frombuffer(audio, dtype=np.float32))
    
    full_audio = np.concatenate(audio_chunks)
    sf.write(tmp_path / "keyboard_scale.wav", full_audio, 44100)
    
    # Verify expected notes were generated
    assert len(audio_chunks) == 8  # C major scale
```

## C. TRACKPAD/MOUSE INPUT PROJECT

### macOS trackpad challenges and solutions

**Critical Finding:** Full multi-touch gesture support (pinch, rotate, multi-finger swipes) on macOS requires **PyObjC** with Cocoa NSResponder integration, which demands a full application event loop. This is impractical for background scripts.

**PyObjC Installation:**
```bash
pip install pyobjc-framework-Quartz
pip install pyobjc-framework-Cocoa
```

**Practical Solution:** Use **pynput.mouse** for scroll events as gesture proxies. Two-finger scrolling provides directional input suitable for parameter control.

### Recommended approach with pynput

```python
from pynput import mouse
import time

class TrackpadFeatureStream:
    """Convert trackpad/mouse events to feature dicts"""
    
    def __init__(self):
        self.event_queue = Queue()
        self.current_position = (0, 0)
        self.listener = None
    
    def on_move(self, x, y):
        """Track cursor position"""
        self.current_position = (x, y)
        self.event_queue.put({
            'type': 'trackpad',
            'action': 'move',
            'position': {'x': x, 'y': y},
            'timestamp': time.time()
        })
    
    def on_click(self, x, y, button, pressed):
        """Single-finger tap detection"""
        self.event_queue.put({
            'type': 'trackpad',
            'action': 'press' if pressed else 'release',
            'button': str(button),
            'position': {'x': x, 'y': y},
            'timestamp': time.time()
        })
    
    def on_scroll(self, x, y, dx, dy):
        """Two-finger scroll as gesture proxy"""
        # Interpret scroll direction
        if abs(dx) > abs(dy):
            direction = 'horizontal'
            magnitude = abs(dx)
        else:
            direction = 'vertical'
            magnitude = abs(dy)
        
        self.event_queue.put({
            'type': 'trackpad',
            'action': 'scroll',
            'direction': direction,
            'magnitude': magnitude,
            'delta': {'dx': dx, 'dy': dy},
            'position': {'x': x, 'y': y},
            'timestamp': time.time()
        })
    
    def start(self):
        """Start listener"""
        self.listener = mouse.Listener(
            on_move=self.on_move,
            on_click=self.on_click,
            on_scroll=self.on_scroll
        )
        self.listener.start()
```

### Mapping trackpad gestures to knobs

```python
def trackpad_to_audio_knobs(trackpad_event, screen_width=1920, screen_height=1080) -> dict:
    """Map trackpad position to audio parameters"""
    if trackpad_event['action'] == 'move':
        # Normalize position to 0-1 range
        x_norm = trackpad_event['position']['x'] / screen_width
        y_norm = trackpad_event['position']['y'] / screen_height
        
        return {
            'cutoff_frequency': 200 + x_norm * 8000,  # 200-8200 Hz
            'resonance': y_norm * 0.9,  # 0-0.9
            'knob_changed': True
        }
    
    elif trackpad_event['action'] == 'scroll':
        # Vertical scroll controls volume
        # Horizontal scroll controls pan
        return {
            'volume_delta': trackpad_event['delta']['dy'] * 0.01,
            'pan_delta': trackpad_event['delta']['dx'] * 0.01,
            'knob_changed': True
        }
    
    return {'knob_changed': False}

def apply_filter_knobs(audio_chunk, cutoff_frequency, resonance):
    """Apply filter based on knob settings"""
    from scipy.signal import butter, filtfilt
    
    sr = 44100
    nyquist = sr / 2
    cutoff_norm = cutoff_frequency / nyquist
    
    b, a = butter(4, cutoff_norm, btype='low')
    filtered = filtfilt(b, a, audio_chunk)
    
    return filtered
```

### Testing approaches

```python
def test_trackpad_gesture_mapping():
    """Test trackpad event → parameter mapping"""
    # Mock scroll event
    scroll_event = {
        'type': 'trackpad',
        'action': 'scroll',
        'direction': 'vertical',
        'delta': {'dx': 0, 'dy': 5},
        'position': {'x': 960, 'y': 540},
        'timestamp': time.time()
    }
    
    knobs = trackpad_to_audio_knobs(scroll_event)
    
    assert 'volume_delta' in knobs
    assert knobs['volume_delta'] > 0  # Scrolling up increases volume

def test_trackpad_position_mapping():
    """Test position → filter parameters"""
    move_event = {
        'type': 'trackpad',
        'action': 'move',
        'position': {'x': 1920, 'y': 0},  # Top right corner
        'timestamp': time.time()
    }
    
    knobs = trackpad_to_audio_knobs(move_event, 1920, 1080)
    
    assert knobs['cutoff_frequency'] > 8000  # High frequency
    assert knobs['resonance'] < 0.1  # Low resonance
```

### macOS accessibility permissions

Applications using pynput for keyboard/trackpad monitoring require explicit permissions:

1. Open **System Preferences → Security & Privacy → Privacy → Accessibility**
2. Add your Terminal app or Python executable
3. Grant access

For distribution, applications must be code-signed to access accessibility APIs.

## D. FACIAL EXPRESSION & ADVANCED VIDEO FEATURES PROJECT

### Facial expression detection packages

**DeepFace** (https://github.com/serengil/deepface, PyPI: https://pypi.org/project/deepface/) is the recommended solution for emotion recognition. It wraps multiple state-of-the-art models and provides 7 emotion classifications with real-time streaming support.

**Installation:** `pip install deepface`

**Alternative:** **FER** (https://github.com/justinshenk/fer, PyPI: https://pypi.org/project/fer/) offers a lighter-weight CNN-based approach.

```python
from deepface import DeepFace
import cv2

def extract_facial_features(video_frame, timestamp) -> dict:
    """Extract emotion and facial features"""
    try:
        result = DeepFace.analyze(
            video_frame,
            actions=['emotion', 'age', 'gender'],
            enforce_detection=False
        )
        
        emotion_scores = result[0]['emotion']
        dominant_emotion = result[0]['dominant_emotion']
        
        return {
            'emotion': dominant_emotion,
            'emotion_scores': emotion_scores,
            'emotion_intensity': emotion_scores[dominant_emotion] / 100,
            'age': result[0]['age'],
            'timestamp': timestamp
        }
    except:
        return {
            'emotion': 'neutral',
            'emotion_intensity': 0,
            'timestamp': timestamp
        }

def emotion_to_audio_modulation(emotion, emotion_intensity) -> dict:
    """Map emotions to synthesis parameters"""
    emotion_mappings = {
        'happy': {'brightness': 1.5, 'tempo_mult': 1.2},
        'sad': {'brightness': 0.5, 'tempo_mult': 0.8},
        'angry': {'distortion': 0.7, 'tempo_mult': 1.3},
        'fear': {'tremolo': 0.8, 'reverb': 0.9},
        'surprise': {'pitch_bend': 200, 'brightness': 2.0},
        'disgust': {'detuning': 0.3, 'filter_sweep': True},
        'neutral': {}
    }
    
    params = emotion_mappings.get(emotion, {})
    
    # Scale by intensity
    scaled_params = {
        k: v * emotion_intensity 
        for k, v in params.items()
    }
    
    return scaled_params
```

### Color tracking implementations

OpenCV's built-in HSV color space conversion provides the fastest and most flexible color tracking without additional dependencies.

```python
def extract_color_features(video_frame, timestamp) -> dict:
    """Track dominant colors and color-based objects"""
    hsv = cv2.cvtColor(video_frame, cv2.COLOR_BGR2HSV)
    
    # Define color ranges
    color_ranges = {
        'red': ([0, 100, 100], [10, 255, 255]),
        'blue': ([100, 100, 100], [130, 255, 255]),
        'green': ([40, 50, 50], [80, 255, 255]),
        'yellow': ([20, 100, 100], [30, 255, 255])
    }
    
    color_percentages = {}
    for color_name, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        percentage = np.count_nonzero(mask) / mask.size
        color_percentages[color_name] = percentage
    
    # Find dominant color
    dominant_color = max(color_percentages, key=color_percentages.get)
    
    return {
        'dominant_color': dominant_color,
        'color_percentages': color_percentages,
        'timestamp': timestamp
    }

def color_to_timbre(dominant_color, color_percentages) -> dict:
    """Map colors to timbral changes"""
    color_timbres = {
        'red': {'waveform': 'sawtooth', 'harmonics': 8},
        'blue': {'waveform': 'sine', 'harmonics': 1},
        'green': {'waveform': 'triangle', 'harmonics': 4},
        'yellow': {'waveform': 'square', 'harmonics': 6}
    }
    
    return color_timbres.get(dominant_color, {'waveform': 'sine'})
```

### Pose estimation tools

**MMPose** (https://github.com/open-mmlab/mmpose, PyPI: https://pypi.org/project/mmpose/) provides comprehensive pose estimation with 300+ pre-trained models including RTMPose for real-time applications (30-60+ FPS).

**Lightweight Alternative:** **rtmlib** (https://pypi.org/project/rtmlib/) offers RTMPose models without MMPose dependencies, supporting CPU real-time processing.

**Installation:** 
```bash
pip install -U openmim
mim install mmpose
# OR for lightweight
pip install rtmlib
```

```python
from rtmlib import Wholebody, draw_skeleton

def extract_pose_features(video_frame, timestamp) -> dict:
    """Extract full-body pose keypoints"""
    wholebody_estimator = Wholebody(
        mode='balanced', 
        backend='onnxruntime',
        device='cpu'
    )
    
    keypoints, scores = wholebody_estimator(video_frame)
    
    if keypoints is None:
        return {'pose_detected': False, 'timestamp': timestamp}
    
    # Extract specific keypoints
    left_hand = keypoints[0][9]  # Left wrist
    right_hand = keypoints[0][10]  # Right wrist
    
    return {
        'pose_detected': True,
        'left_hand': {'x': left_hand[0], 'y': left_hand[1]},
        'right_hand': {'x': right_hand[0], 'y': right_hand[1]},
        'confidence': float(np.mean(scores)),
        'timestamp': timestamp
    }

def pose_to_polyphonic_control(left_hand, right_hand) -> dict:
    """Map two hands to polyphonic parameters"""
    return {
        'voice_1_frequency': 200 + left_hand['x'] * 1800,
        'voice_1_amplitude': 1 - (left_hand['y'] / 480),
        'voice_2_frequency': 200 + right_hand['x'] * 1800,
        'voice_2_amplitude': 1 - (right_hand['y'] / 480)
    }
```

### Integration with existing video pipeline

```python
from meshed import DAG

def combined_video_features(video_frame, timestamp) -> dict:
    """Extract all video features in parallel"""
    hand_features = extract_hand_features(video_frame, timestamp)
    facial_features = extract_facial_features(video_frame, timestamp)
    color_features = extract_color_features(video_frame, timestamp)
    pose_features = extract_pose_features(video_frame, timestamp)
    
    # Merge all features
    return {
        **hand_features,
        **facial_features,
        **color_features,
        **pose_features
    }

# Extended pipeline
video_pipeline = DAG([
    read_video_sensor,
    combined_video_features,
    map_features_to_comprehensive_params,
    advanced_synthesizer
])
```

## E. AUDIO INPUT STREAMS PROJECT

### Audio capture packages

**sounddevice** (https://github.com/spatialaudio/python-sounddevice, PyPI: https://pypi.org/project/sounddevice/) is the recommended modern solution for low-latency audio I/O with NumPy integration.

**Installation:** `pip install sounddevice`

**Alternative:** **pyaudio** offers similar capabilities but with older API design.

```python
import sounddevice as sd
import numpy as np
from queue import Queue

class AudioInputStream:
    """Capture live audio input"""
    
    def __init__(self, sample_rate=44100, block_size=2048):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.audio_queue = Queue()
    
    def audio_callback(self, indata, frames, time_info, status):
        """Callback for incoming audio"""
        if status:
            print(f"Audio input status: {status}")
        
        # Copy audio data
        self.audio_queue.put(indata.copy())
    
    def start(self):
        """Start audio stream"""
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            channels=1,
            callback=self.audio_callback
        )
        self.stream.start()
    
    def get_audio_chunk(self):
        """Get next audio chunk"""
        if not self.audio_queue.empty():
            return self.audio_queue.get()
        return None
```

### Real-time audio feature extraction

**aubio** (https://github.com/aubio/aubio, PyPI: https://pypi.org/project/aubio/) provides real-time onset detection, pitch tracking, beat detection, and more.

**Installation:** `pip install aubio`

```python
import aubio

def extract_audio_features(audio_chunk, sample_rate=44100) -> dict:
    """Extract pitch, amplitude, and onset features"""
    # Pitch detection
    pitch_detector = aubio.pitch("default", 2048, 2048, sample_rate)
    pitch_detector.set_unit("Hz")
    pitch = pitch_detector(audio_chunk.flatten())[0]
    
    # Onset detection
    onset_detector = aubio.onset("default", 2048, 2048, sample_rate)
    onset = onset_detector(audio_chunk.flatten())[0]
    
    # Amplitude (RMS)
    rms = np.sqrt(np.mean(audio_chunk ** 2))
    
    return {
        'pitch': float(pitch),
        'amplitude': float(rms),
        'onset_detected': bool(onset),
        'timestamp': time.time()
    }

def audio_features_to_parameters(pitch, amplitude, onset_detected) -> dict:
    """Map audio features to synthesis parameters"""
    return {
        'frequency': pitch if pitch > 0 else 440,  # Default to A440
        'amplitude': amplitude * 2,  # Amplify
        'trigger_envelope': onset_detected,
        'voice_activity': amplitude > 0.01
    }
```

### Use cases: Voice control and rhythm detection

**Voice Control:**
```python
import speech_recognition as sr

def extract_voice_commands(audio_chunk, sample_rate) -> dict:
    """Extract voice commands using speech recognition"""
    recognizer = sr.Recognizer()
    
    # Convert numpy array to audio data
    audio_data = sr.AudioData(
        audio_chunk.tobytes(),
        sample_rate,
        2  # sample width
    )
    
    try:
        command = recognizer.recognize_google(audio_data)
        return {
            'command': command.lower(),
            'command_detected': True
        }
    except:
        return {'command_detected': False}

def voice_command_to_mode_change(command, command_detected) -> dict:
    """Map voice commands to mode changes"""
    if not command_detected:
        return {}
    
    mode_mappings = {
        'play': {'mode': 'playing', 'paused': False},
        'stop': {'mode': 'stopped', 'paused': True},
        'record': {'mode': 'recording', 'recording': True},
        'louder': {'volume_mult': 1.5},
        'softer': {'volume_mult': 0.5}
    }
    
    return mode_mappings.get(command, {})
```

**Rhythm Detection:**
```python
def detect_rhythm_features(audio_chunk, sample_rate) -> dict:
    """Detect tempo and beat using aubio"""
    tempo_detector = aubio.tempo("default", 2048, 2048, sample_rate)
    
    is_beat = tempo_detector(audio_chunk.flatten())[0]
    bpm = tempo_detector.get_bpm()
    
    return {
        'beat_detected': bool(is_beat),
        'tempo_bpm': float(bpm),
        'timestamp': time.time()
    }

def rhythm_to_sequencer_control(beat_detected, tempo_bpm) -> dict:
    """Map rhythm to sequencer parameters"""
    return {
        'trigger_step': beat_detected,
        'step_duration': 60 / tempo_bpm if tempo_bpm > 0 else 0.5,
        'sync_to_input': True
    }
```

### Complete audio input pipeline

```python
audio_input_pipeline = DAG([
    AudioInputStream().get_audio_chunk,
    extract_audio_features,
    audio_features_to_parameters,
    synthesize_responsive_audio
])

# Run in real-time loop
audio_input = AudioInputStream()
audio_input.start()

while True:
    audio_chunk = audio_input.get_audio_chunk()
    if audio_chunk is not None:
        output = audio_input_pipeline(audio_chunk=audio_chunk)
        audio_output_stream.write(output)
```

## F. SYNTH BACKEND ALTERNATIVES PROJECT

### Package comparison and recommendations

**Top Recommendation: pyo** (https://github.com/belangeo/pyo, PyPI: https://pypi.org/project/pyo/)

pyo provides the best balance of real-time performance, dict-based parameter control, and ease of use for the theremin framework.

**Capabilities:**
- 1000+ built-in audio objects
- C-based engine for low latency
- Direct parameter updates via `.value` attribute
- Cross-platform (Windows, macOS including M1/M2, Linux)
- MIDI input support
- OSC protocol support

**Installation:** `pip install pyo`

**Alternative #1: SignalFlow** (https://signalflow.dev/, PyPI: https://pypi.org/project/signalflow/)

Modern package with excellent documentation and Jupyter support. C++11 backend provides hardware-accelerated performance.

**Alternative #2: Supriya** (https://github.com/supriya-project/supriya, PyPI: https://pypi.org/project/supriya/)

Full SuperCollider access from Python with asyncio support. Requires separate SuperCollider installation but provides professional-grade synthesis.

### Integration patterns with dict-based parameter streams

**pyo Integration:**

```python
from pyo import *

class PyoSynthesizer:
    """Dict-based synthesizer using pyo"""
    
    def __init__(self, sample_rate=44100):
        self.server = Server(sr=sample_rate).boot()
        self.server.start()
        
        # Create controllable parameters
        self.frequency = Sig(440)
        self.amplitude = Sig(0.5)
        self.waveform_type = 'sine'
        
        # Create oscillator
        self.osc = self._create_oscillator()
        self.osc.out()
    
    def _create_oscillator(self):
        """Create oscillator based on waveform type"""
        if self.waveform_type == 'sine':
            return Sine(freq=self.frequency, mul=self.amplitude)
        elif self.waveform_type == 'saw':
            return Saw(freq=self.frequency, mul=self.amplitude)
        elif self.waveform_type == 'square':
            return Square(freq=self.frequency, mul=self.amplitude)
        else:
            return Sine(freq=self.frequency, mul=self.amplitude)
    
    def update_parameters(self, params: dict):
        """Update synthesis parameters from dict"""
        if 'frequency' in params:
            self.frequency.value = params['frequency']
        
        if 'amplitude' in params:
            self.amplitude.value = params['amplitude']
        
        if 'waveform' in params and params['waveform'] != self.waveform_type:
            self.waveform_type = params['waveform']
            self.osc.stop()
            self.osc = self._create_oscillator()
            self.osc.out()

# Usage in pipeline
def synthesize_with_pyo(frequency, amplitude, waveform='sine'):
    """Synthesizer as pure function"""
    synthesizer = PyoSynthesizer()
    synthesizer.update_parameters({
        'frequency': frequency,
        'amplitude': amplitude,
        'waveform': waveform
    })
    time.sleep(0.1)  # Generate 100ms of audio
    return b''  # pyo handles audio output directly
```

**SignalFlow Integration:**

```python
from signalflow import *

def synthesize_with_signalflow(frequency, amplitude) -> bytes:
    """SignalFlow synthesis"""
    graph = AudioGraph()
    
    # Create synthesis network
    sine = SineOscillator(frequency)
    output = sine * amplitude
    output.play()
    
    # Let it run for chunk duration
    graph.wait(0.1)
    
    return b''  # SignalFlow manages output
```

**Supriya Integration:**

```python
import supriya

def synthesize_with_supriya(frequency, amplitude):
    """SuperCollider synthesis via Supriya"""
    server = supriya.Server().boot()
    
    synth = server.add_synth(
        synthdef='default',
        frequency=frequency,
        amplitude=amplitude
    )
    
    # Update in real-time
    synth.set(frequency=frequency * 1.1)
    
    return synth
```

### Performance considerations

**Latency Comparison:**

| Package | Typical Latency | CPU Usage | Memory |
|---------|----------------|-----------|--------|
| pyo | <5ms | Low | ~50MB |
| Supriya/SuperCollider | <5ms | Medium | ~100MB |
| SignalFlow | <5ms | Low | ~80MB |
| sounddevice (custom) | 5-10ms | Medium | ~20MB |

**Best Practices:**
- Use buffer sizes of 256-1024 samples for real-time (5-23ms latency at 44.1kHz)
- Pre-allocate synthesis objects rather than creating/destroying
- Use C-based packages (pyo, SignalFlow) for lowest CPU usage
- Profile with different buffer sizes to find optimal latency/reliability balance

### Migration guide from current synth

**Current Architecture (Assumed):**
```python
class HumSynthesizer:
    def __init__(self):
        self.frequency = 440
        self.amplitude = 0.5
    
    def set_frequency(self, freq):
        self.frequency = freq
    
    def generate(self):
        return generate_audio(self.frequency, self.amplitude)
```

**Migrated to pyo:**
```python
from meshed import DAG

def read_features(sensor_id) -> dict:
    return {'hand_x': 0.5, 'hand_y': 0.3}

def map_to_synth_params(hand_x, hand_y) -> dict:
    return {
        'frequency': 200 + hand_x * 1800,
        'amplitude': 1 - hand_y
    }

def pyo_synthesize(frequency, amplitude):
    synth = PyoSynthesizer()
    synth.update_parameters({'frequency': frequency, 'amplitude': amplitude})
    return None  # pyo outputs directly

pipeline = DAG([
    read_features,
    map_to_synth_params,
    pyo_synthesize
])
```

**Migration Steps:**
1. Replace synth initialization with pyo Server setup
2. Convert parameter setters to dict-based `update_parameters`
3. Remove manual audio buffer management (pyo handles this)
4. Adjust buffer sizes if latency is problematic
5. Test with same feature inputs to verify equivalent output

## G. MUSIC ACCOMPANIMENT PROJECT

### AI music generation packages (live compatible)

**Magenta RealTime** (https://github.com/magenta/magenta-realtime) - Released June 2025

**Breakthrough for live streaming:** First true real-time AI music generation system capable of generating 2-second audio chunks with 10-second context windows. Real-time factor of 1.6x on TPU (generates 2s audio in 1.25s).

**Installation:** 
```bash
pip install -e magenta-realtime/[tpu]
```

**Requirements:** TPU (free on Colab) or 40GB+ GPU

```python
from magenta_rt import audio, system

def generate_ai_accompaniment_chunk(style_description: str) -> bytes:
    """Generate AI music chunk"""
    mrt = system.MagentaRT()
    style_embedding = system.embed_style(style_description)
    
    state, chunk = mrt.generate_chunk(
        state=None,
        style=style_embedding
    )
    
    return chunk.samples.tobytes(), state

# Usage in pipeline
def accompaniment_pipeline(user_performance_features, style='jazz'):
    """Combine user performance with AI accompaniment"""
    # Generate AI chunk
    ai_audio, state = generate_ai_accompaniment_chunk(style)
    
    # Mix with user's audio
    mixed = mix_audio_streams(user_performance_features, ai_audio)
    
    return mixed
```

**Limitation:** Minimum latency of ~1.25 seconds makes this suitable for accompaniment but not tight interactive control.

### Chord progression generators

**AccoMontage2 / chorderator** (https://github.com/billyblu2000/AccoMontage2, PyPI: `chorderator`)

Most complete solution for generating chord progressions from melodies with style control and accompaniment arrangement.

**Installation:** `pip install chorderator`

```python
import chorderator as cdt

def generate_chord_progression(melody_midi_path: str, style: str) -> dict:
    """Generate styled chord progression"""
    cdt.set_melody(melody_midi_path)
    cdt.set_meta(tonic=cdt.Key.C, mode=cdt.Mode.MAJOR)
    cdt.set_segmentation('A8B8A8B8')  # Phrase structure
    
    # Style options: pop_standard, pop_complex, r&b, dark
    style_map = {
        'pop': cdt.Style.POP_STANDARD,
        'complex': cdt.Style.POP_COMPLEX,
        'rnb': cdt.Style.R_AND_B,
        'dark': cdt.Style.DARK
    }
    
    cdt.set_output_style(style_map.get(style, cdt.Style.POP_STANDARD))
    cdt.set_texture_prefilter((2, 2))  # Rhythmic density
    
    output_dir = '/tmp/chords'
    cdt.generate_save(output_dir)
    
    return {
        'chord_midi': f"{output_dir}/textured_chord_gen.mid",
        'style': style
    }
```

**pychord** (https://pypi.org/project/pychord/) - Real-time chord manipulation

Lightweight library for chord parsing and progression generation suitable for real-time use.

```python
from pychord import Chord, ChordProgression

def generate_simple_progression(key='C', progression_type='jazz') -> dict:
    """Generate algorithmic chord progression"""
    progressions = {
        'jazz': ['Cmaj7', 'Dm7', 'G7', 'Cmaj7'],
        'pop': ['C', 'G', 'Am', 'F'],
        'blues': ['C7', 'F7', 'C7', 'G7'],
        'minor': ['Am', 'F', 'C', 'G']
    }
    
    chords = ChordProgression(progressions.get(progression_type, progressions['pop']))
    
    # Transpose if needed
    if key != 'C':
        semitones = {'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
        chords = chords.transpose(semitones.get(key, 0))
    
    return {
        'chord_names': [str(c) for c in chords],
        'chord_objects': chords
    }
```

### Real-time accompaniment systems

**MMA (Musical MIDI Accompaniment)** (https://www.mellowood.ca/mma/)

Python-scriptable alternative to Band-in-a-Box. Generates MIDI accompaniment from text-based chord charts with extensive groove library.

**Not Real-time:** MMA generates MIDI files rather than live playback, but files can be replayed with precise timing control using mido.

```python
def generate_mma_accompaniment(chords: list, tempo: int, groove: str) -> str:
    """Generate MMA accompaniment file"""
    mma_content = f"""
Tempo {tempo}
Groove {groove}

"""
    for i, chord in enumerate(chords, 1):
        mma_content += f"{i} {chord}\n"
    
    # Write MMA file
    mma_path = '/tmp/accompaniment.mma'
    with open(mma_path, 'w') as f:
        f.write(mma_content)
    
    # Generate MIDI (requires MMA installed)
    import subprocess
    midi_path = '/tmp/accompaniment.mid'
    subprocess.run(['mma', mma_path, '-f', midi_path])
    
    return midi_path
```

### MIDI processing for live accompaniment

**mido** (https://pypi.org/project/mido/) - Primary recommendation

Clean, Pythonic MIDI API with full real-time I/O support.

**Installation:** `pip install mido[ports-rtmidi]`

```python
import mido
from mido import Message, MidiFile
import time

def playback_midi_accompaniment(midi_path: str, output_port_name: str):
    """Play MIDI file with accurate timing"""
    mid = MidiFile(midi_path)
    
    with mido.open_output(output_port_name) as port:
        for msg in mid.play():
            port.send(msg)

def live_chord_accompaniment(chord_progression: list):
    """Generate live MIDI chords"""
    port = mido.open_output()
    
    chord_notes = {
        'C': [60, 64, 67],  # C major
        'Dm': [62, 65, 69],  # D minor
        'G7': [67, 71, 74, 65],  # G7
        'Am': [69, 72, 76]  # A minor
    }
    
    for chord_name in chord_progression:
        notes = chord_notes.get(chord_name, [60, 64, 67])
        
        # Note on
        for note in notes:
            port.send(Message('note_on', note=note, velocity=80))
        
        time.sleep(2)  # Chord duration
        
        # Note off
        for note in notes:
            port.send(Message('note_off', note=note))
        
        time.sleep(0.1)  # Gap between chords
```

### Complete accompaniment pipeline

```python
from meshed import DAG

def detect_user_melody(audio_features) -> dict:
    """Detect melody from user performance"""
    # Simplified: extract pitch sequence
    return {
        'melody_notes': [60, 62, 64, 65, 67],  # C D E F G
        'tempo': 120
    }

def generate_chords_from_melody(melody_notes, tempo) -> dict:
    """Generate harmonizing chords"""
    # Simplified: basic harmonization
    chords = ['C', 'G', 'Am', 'F']
    return {
        'chord_progression': chords,
        'tempo': tempo
    }

def render_accompaniment_midi(chord_progression, tempo) -> str:
    """Render chords to MIDI"""
    midi_path = generate_mma_accompaniment(
        chord_progression,
        tempo,
        groove='Rock'
    )
    return {'midi_path': midi_path}

def playback_with_synthesis(midi_path) -> bytes:
    """Convert MIDI to audio via synthesis"""
    import fluidsynth
    
    fs = fluidsynth.Synth()
    fs.start(driver='alsa')
    sfid = fs.sfload("/usr/share/sounds/sf2/FluidR3_GM.sf2")
    fs.program_select(0, sfid, 0, 0)
    
    mid = mido.MidiFile(midi_path)
    for msg in mid:
        if msg.type == 'note_on':
            fs.noteon(0, msg.note, msg.velocity)
        elif msg.type == 'note_off':
            fs.noteoff(0, msg.note)
    
    return b''  # Audio rendered by FluidSynth

# Complete pipeline
accompaniment_pipeline = DAG([
    detect_user_melody,
    generate_chords_from_melody,
    render_accompaniment_midi,
    playback_with_synthesis
])
```

### Integration with main signal processing pipeline

```python
def combined_performance_and_accompaniment(video_frame, timestamp):
    """User theremin + AI accompaniment"""
    # User's theremin performance
    user_features = extract_hand_features(video_frame, timestamp)
    user_params = map_hand_to_audio_params(**user_features)
    user_audio = synthesize_audio(**user_params)
    
    # Detect melody for accompaniment
    melody = detect_user_melody(user_features)
    
    # Generate accompaniment (slower process, update every 4 bars)
    if should_update_accompaniment(timestamp):
        chords = generate_chords_from_melody(**melody)
        start_accompaniment_playback(chords)
    
    # Mix user audio with accompaniment
    accompaniment_audio = get_current_accompaniment_audio()
    mixed = mix_audio(user_audio, accompaniment_audio, user_level=0.7, acc_level=0.3)
    
    return mixed
```

### Recommended stack for live accompaniment

**For Interactive Performance:**
1. **MIDI Control:** mido or python-rtmidi
2. **Chord Logic:** pychord
3. **Synthesis:** pyo or FluidSynth
4. **NO AI:** Latency too high for tight interaction

**For Studio/Composition:**
1. **Chord Generation:** AccoMontage2
2. **Arrangement:** MMA
3. **Rendering:** FluidSynth or DAW

**For Experimental/Research:**
1. **AI Generation:** Magenta RealTime (requires TPU/GPU)
2. **Real-time Control:** Custom Python logic
3. **Audio Output:** sounddevice

---

## Integration Architecture Summary

### Complete System Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT SOURCES (Parallel)                     │
├─────────────┬──────────────┬────────────────┬──────────────────┤
│   Video     │   Keyboard   │   Trackpad     │   Audio Input    │
│  (MediaPipe,│   (pynput)   │   (pynput)     │  (sounddevice)   │
│  DeepFace,  │              │                │                  │
│  MMPose)    │              │                │                  │
└──────┬──────┴──────┬───────┴────────┬───────┴────────┬─────────┘
       │             │                │                │
       ▼             ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────┐
│              FEATURE EXTRACTION (Dict Outputs)                  │
├─────────────┬──────────────┬────────────────┬──────────────────┤
│ hand_x,     │ key_pressed, │ position_x,    │ pitch,           │
│ hand_y,     │ key_chord,   │ position_y,    │ amplitude,       │
│ emotion,    │ octave_shift │ scroll_delta   │ onset_detected   │
│ pose_left   │              │                │                  │
└──────┬──────┴──────┬───────┴────────┬───────┴────────┬─────────┘
       │             │                │                │
       └─────────────┴────────┬───────┴────────────────┘
                              ▼
            ┌──────────────────────────────────┐
            │   FEATURE MERGING (via meshed)   │
            │    - Argument-name matching      │
            │    - Dict-based data flow        │
            └─────────────┬────────────────────┘
                          ▼
            ┌──────────────────────────────────┐
            │    PARAMETER MAPPING (Knobs)     │
            │  - frequency, amplitude          │
            │  - effects, filters              │
            │  - waveform, modulation          │
            └─────────────┬────────────────────┘
                          ▼
            ┌──────────────────────────────────┐
            │       SYNTHESIS BACKENDS         │
            ├──────────────────────────────────┤
            │  pyo / SignalFlow / Supriya      │
            │  + Accompaniment (mido, MMA)     │
            └─────────────┬────────────────────┘
                          ▼
                  ┌───────────────┐
                  │ AUDIO OUTPUT  │
                  │  (44.1kHz)    │
                  └───────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                  STORAGE LAYER (dol)                            │
├─────────────┬──────────────┬────────────────┬──────────────────┤
│ Calibration │   Presets    │  Test Fixtures │   Recordings     │
│   Store     │    Store     │     Store      │     Store        │
└─────────────┴──────────────┴────────────────┴──────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    TESTING LAYER                                │
├─────────────┬──────────────┬────────────────┬──────────────────┤
│  pytest +   │   librosa    │  StreamPlayer  │   AudioVerifier  │
│ pytest-     │   soundfile  │  (JSON replay) │   (speechmetrics)│
│ datadir     │              │                │                  │
└─────────────┴──────────────┴────────────────┴──────────────────┘
```

### Package Installation Summary

```bash
# Core i2mint ecosystem
pip install meshed dol creek i2

# Audio synthesis (choose primary)
pip install pyo  # Recommended
# OR
pip install signalflow  # Modern alternative
# OR
pip install supriya  # Requires SuperCollider

# Video processing
pip install opencv-python mediapipe
pip install deepface  # Facial expressions
pip install rtmlib  # Lightweight pose estimation
# OR for advanced: mim install mmpose

# Input capture
pip install pynput  # Keyboard + trackpad
pip install pyobjc-framework-Quartz  # macOS (optional)

# Audio I/O and processing
pip install sounddevice aubio librosa soundfile

# Music generation and MIDI
pip install mido[ports-rtmidi] python-rtmidi
pip install pychord chorderator
pip install pyfluidsynth  # MIDI synthesis

# Testing
pip install pytest pytest-cov pytest-xdist pytest-datadir-ng
pip install pesq pystoi  # Audio quality metrics

# Optional: AI music (requires TPU/GPU)
# pip install -e magenta-realtime/[tpu]
```

### Best Practices Recap

**Function Design for Argument Wiring:**
- Use descriptive, consistent parameter names across pipeline stages
- Return dictionaries with clearly named outputs
- Keep functions pure (no side effects) when possible
- Use type hints for better introspection

**Data Flow:**
- Pass dict objects between stages
- Use flat dictionaries over nested structures
- Name parameters identically when data should flow automatically
- Use Mapping/MutableMapping interfaces for storage

**Testing:**
- Record feature streams as JSON for deterministic tests
- Test each pipeline stage independently
- Use pytest fixtures with appropriate scope
- Separate video processing tests from audio tests
- Save generated audio for manual verification

**Performance:**
- Use C-based synthesis packages (pyo, SignalFlow) for low latency
- Buffer sizes: 256-1024 samples (5-23ms at 44.1kHz)
- Profile with different buffer sizes
- Consider threading for parallel input processing

**Modularity:**
- Define small, single-purpose functions
- Compose into pipelines using meshed DAG
- Store configuration in dol stores
- Make backends swappable through consistent interfaces

This specification provides a complete architecture for building a modular, testable, and extensible theremin framework using functional programming patterns, self-assembling components via meshed.slabs, and dict-based data flow throughout. Each component can be developed independently and automatically wires into the larger system through argument-name matching.