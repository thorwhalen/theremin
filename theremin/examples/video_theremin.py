"""
Video Theremin Example.

Demonstrates using vidstream for hand tracking + synthflow for synthesis.
"""

import time
import sys
from pathlib import Path
import numpy as np

# Add related packages to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'related_packages/vidstream/src'))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'related_packages/synthflow/src'))

try:
    from vidstream import HandFeatureExtractor
    from synthflow import SimpleSynthesizer
    import cv2
except ImportError as e:
    print(f"Missing dependencies: {e}")
    print("Install: pip install opencv-python mediapipe")
    sys.exit(1)


def map_hand_to_audio(hand_features: dict) -> dict:
    """
    Map hand position to audio parameters.

    Args:
        hand_features: Hand tracking features

    Returns:
        Audio synthesis parameters
    """
    if not hand_features.get('has_hands'):
        return {'frequency': 0, 'amplitude': 0}

    # Use right hand wrist position
    if 'r_wrist_position' in hand_features:
        x, y, z = hand_features['r_wrist_position']

        # Map X to frequency (200-2000 Hz)
        frequency = 200 + x * 1800

        # Map Y to amplitude (inverted)
        amplitude = max(0, min(1, 1 - y))

        return {
            'frequency': frequency,
            'amplitude': amplitude,
            'duration': 0.1
        }

    return {'frequency': 440, 'amplitude': 0}


def run_video_theremin():
    """Run video theremin example."""
    print("Video Theremin")
    print("=" * 50)
    print("Move your right hand to control pitch and volume")
    print("X position: Pitch (left=low, right=high)")
    print("Y position: Volume (top=loud, bottom=quiet)")
    print("Press 'q' to quit")
    print()

    # Create components
    hand_extractor = HandFeatureExtractor(max_hands=2)
    synth = SimpleSynthesizer(sample_rate=44100)

    # Open camera
    cap = cv2.VideoCapture(0)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Extract hand features
            hand_features = hand_extractor.extract(frame)

            # Map to audio parameters
            audio_params = map_hand_to_audio(hand_features)

            # Generate audio (if hand detected)
            if audio_params['amplitude'] > 0:
                audio = synth.generate(audio_params)
                print(f"🎵 Freq: {audio_params['frequency']:.1f} Hz, "
                      f"Amp: {audio_params['amplitude']:.2f}")

            # Display frame
            cv2.imshow('Video Theremin', frame)

            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nStopped")
    finally:
        cap.release()
        cv2.destroyAllWindows()


def run_video_theremin_with_meshed():
    """Run video theremin using meshed DAG."""
    try:
        from meshed import DAG
    except ImportError:
        print("meshed not installed. Install with: pip install meshed")
        return

    # Define pipeline functions
    def read_video_frame(cap) -> dict:
        """Read frame from camera."""
        ret, frame = cap.read()
        if ret:
            return {'video_frame': frame, 'timestamp': time.time()}
        return {}

    def extract_hand_features(video_frame, timestamp) -> dict:
        """Extract hand features from frame."""
        hand_extractor = HandFeatureExtractor()
        return hand_extractor.extract(video_frame, timestamp)

    def map_to_audio_params(**hand_features) -> dict:
        """Map hand features to audio parameters."""
        return map_hand_to_audio(hand_features)

    def synthesize_audio(synth, **params) -> object:
        """Generate audio from parameters."""
        if params.get('amplitude', 0) > 0:
            return synth.generate(params)
        return None

    # Create components
    cap = cv2.VideoCapture(0)
    synth = SimpleSynthesizer()

    # Create pipeline
    pipeline = DAG([
        lambda: read_video_frame(cap),
        extract_hand_features,
        map_to_audio_params,
        lambda **p: synthesize_audio(synth, **p)
    ])

    print("Video Theremin (meshed DAG version)")
    print("=" * 50)

    try:
        while True:
            audio = pipeline()
            if audio is not None:
                print("🎵 Audio generated")
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopped")
    finally:
        cap.release()


if __name__ == '__main__':
    run_video_theremin()
