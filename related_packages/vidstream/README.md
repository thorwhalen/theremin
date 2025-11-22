# vidstream

**Video feature extraction streams for Python**

A framework-agnostic package for extracting features from video streams. Wraps MediaPipe, DeepFace, and other CV tools with clean dict outputs.

Useful for any computer vision application: pose estimation, emotion detection, gesture recognition, color tracking, and more.

## Installation

```bash
pip install vidstream
# Or for development
pip install -e .
```

## Dependencies

Core dependencies (installed automatically):
- `opencv-python` - Video frame processing
- `mediapipe` - Hand and pose tracking
- `numpy` - Array operations

Optional dependencies:
- `deepface` - For facial emotion detection (install separately)

## Quick Start

### Hand Tracking

```python
from vidstream import HandFeatureExtractor
import cv2

extractor = HandFeatureExtractor(max_hands=2)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    features = extractor.extract(frame)
    if features.get('has_hands'):
        print(f"Right wrist: {features.get('r_wrist_position')}")
        print(f"Hand openness: {features.get('r_openness')}")
```

### Facial Emotion Detection

```python
from vidstream import FaceFeatureExtractor

extractor = FaceFeatureExtractor()

features = extractor.extract(frame)
if features.get('face_detected'):
    print(f"Emotion: {features['dominant_emotion']}")
    print(f"Intensity: {features['emotion_intensity']}")
    print(f"All scores: {features['emotion_scores']}")
```

### Color Tracking

```python
from vidstream import ColorFeatureExtractor

extractor = ColorFeatureExtractor()

features = extractor.extract(frame)
print(f"Dominant color: {features['dominant_color']}")
print(f"Color percentages: {features['color_percentages']}")
```

### Body Pose Estimation

```python
from vidstream import PoseFeatureExtractor

extractor = PoseFeatureExtractor()

features = extractor.extract(frame)
if features.get('pose_detected'):
    print(f"Left wrist: {features['left_wrist']}")
    print(f"Right wrist: {features['right_wrist']}")
    print(f"Visibility: {features['avg_visibility']}")
```

## API Reference

### HandFeatureExtractor

Extract hand landmarks and gestures using MediaPipe Hands.

**Constructor:**
```python
HandFeatureExtractor(
    max_hands=2,
    detection_confidence=0.5,
    tracking_confidence=0.5
)
```

**Returns:**
```python
{
    'timestamp': float,
    'has_hands': bool,
    'r_wrist_position': (x, y, z),
    'r_palm_center': (x, y, z),
    'r_finger_tips': [(x, y, z), ...],
    'r_openness': float,
    'l_wrist_position': (x, y, z),
    # ... similar for left hand
}
```

### FaceFeatureExtractor

Extract facial emotions using DeepFace.

**Constructor:**
```python
FaceFeatureExtractor(enforce_detection=False)
```

**Returns:**
```python
{
    'face_detected': bool,
    'dominant_emotion': str,  # 'happy', 'sad', 'angry', etc.
    'emotion_scores': dict,   # All emotion scores
    'emotion_intensity': float,  # 0-1 normalized score
    'timestamp': float
}
```

### ColorFeatureExtractor

Track dominant colors using HSV color space.

**Constructor:**
```python
ColorFeatureExtractor(color_ranges=None)
```

**Returns:**
```python
{
    'dominant_color': str,
    'dominant_percentage': float,
    'color_percentages': dict,  # All color percentages
    'timestamp': float
}
```

### PoseFeatureExtractor

Extract body pose keypoints using MediaPipe Pose.

**Constructor:**
```python
PoseFeatureExtractor(
    detection_confidence=0.5,
    tracking_confidence=0.5
)
```

**Returns:**
```python
{
    'pose_detected': bool,
    'left_wrist': {'x': float, 'y': float, 'z': float, 'visibility': float},
    'right_wrist': {...},
    'left_elbow': {...},
    'right_elbow': {...},
    # ... other body landmarks
    'avg_visibility': float,
    'timestamp': float
}
```

## Use Cases

- **Music applications**: Hand gesture to sound mapping
- **Emotion-reactive systems**: Content adaptation based on user emotion
- **Gesture control**: Touchless UI interfaces
- **Fitness tracking**: Pose estimation for exercise form
- **Accessibility tools**: Alternative input methods
- **Art installations**: Interactive visual experiences
- **Data analysis**: Video feature extraction for ML pipelines

## Platform Support

- **All platforms**: Full support via MediaPipe and OpenCV
- **GPU Acceleration**: Supported where MediaPipe provides it
- **Real-time**: Optimized for 30+ FPS on modern hardware

## License

MIT License
