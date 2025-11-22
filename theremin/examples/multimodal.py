"""
Multimodal Example.

Combines keyboard + video + trackpad for rich control.
Demonstrates the power of composing multiple input streams.
"""

import time
import sys
from pathlib import Path

# Add related packages to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'related_packages/streamkeys/src'))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'related_packages/streamtouch/src'))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'related_packages/vidstream/src'))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'related_packages/synthflow/src'))

try:
    from streamkeys import KeyboardFeatureStream, keyboard_to_midi_features
    from streamtouch import TrackpadFeatureStream, trackpad_to_audio_knobs
    from vidstream import HandFeatureExtractor
    from synthflow import SimpleSynthesizer, EffectsProcessor
except ImportError as e:
    print(f"Missing dependencies: {e}")
    sys.exit(1)


def combine_multimodal_features(
    keyboard_event: dict = None,
    trackpad_event: dict = None,
    hand_features: dict = None
) -> dict:
    """
    Combine features from multiple input modalities.

    - Keyboard: Discrete note selection
    - Trackpad: Effect controls (filter, etc.)
    - Video: Continuous pitch bend and modulation

    Args:
        keyboard_event: Keyboard event dict
        trackpad_event: Trackpad event dict
        hand_features: Hand tracking features

    Returns:
        Combined synthesis parameters
    """
    params = {
        'frequency': 440,
        'amplitude': 0.5,
        'filter_cutoff': 8000,
        'resonance': 0.5,
        'pitch_bend': 0
    }

    # Keyboard provides base note
    if keyboard_event:
        midi_features = keyboard_to_midi_features(keyboard_event)
        if midi_features:
            params['frequency'] = midi_features['frequency']
            params['amplitude'] = midi_features['amplitude']

    # Trackpad controls effects
    if trackpad_event:
        knobs = trackpad_to_audio_knobs(trackpad_event)
        if knobs.get('knob_changed'):
            if 'cutoff_frequency' in knobs:
                params['filter_cutoff'] = knobs['cutoff_frequency']
                params['resonance'] = knobs['resonance']

    # Video provides pitch bend and modulation
    if hand_features and hand_features.get('has_hands'):
        if 'r_wrist_position' in hand_features:
            x, y, z = hand_features['r_wrist_position']
            # Subtle pitch bend based on hand position
            params['pitch_bend'] = (x - 0.5) * 100  # +/- 50 Hz

    return params


def run_multimodal_theremin():
    """
    Run multimodal theremin example.

    This demonstrates the power of combining multiple input streams:
    - Keyboard for note selection
    - Trackpad for effect control
    - Video for expression
    """
    print("Multimodal Theremin")
    print("=" * 50)
    print("Controls:")
    print("  Keyboard: Play notes (A-L keys)")
    print("  Trackpad: Move to control filter cutoff")
    print("  Video: Hand position adds pitch bend")
    print("  Press ESC to quit")
    print()

    # Create input streams
    keyboard = KeyboardFeatureStream()
    keyboard.start()

    trackpad = TrackpadFeatureStream()
    trackpad.start()

    # Note: Video would require camera setup
    # hand_extractor = HandFeatureExtractor()

    # Create synthesis
    synth = SimpleSynthesizer()
    effects = EffectsProcessor()

    try:
        while True:
            # Gather features from all modalities
            kb_event = keyboard.get_features()
            tp_event = trackpad.get_features()
            # In real app: hand_features = hand_extractor.extract(frame)
            hand_features = None

            # Combine features
            params = combine_multimodal_features(kb_event, tp_event, hand_features)

            # Generate audio if note is playing
            if kb_event and kb_event.get('action') == 'press':
                # Apply pitch bend
                freq_with_bend = params['frequency'] + params.get('pitch_bend', 0)

                # Generate audio
                audio = synth.generate({
                    'frequency': freq_with_bend,
                    'amplitude': params['amplitude'],
                    'duration': 0.2
                })

                # Apply effects
                audio = effects.process(audio, {
                    'filter_cutoff': params['filter_cutoff'],
                    'resonance': params['resonance']
                })

                print(f"🎵 Note: {freq_with_bend:.1f} Hz, "
                      f"Filter: {params['filter_cutoff']:.0f} Hz")

            # Check for quit
            if kb_event and kb_event.get('key') == 'esc':
                break

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nStopped")
    finally:
        keyboard.stop()
        trackpad.stop()


if __name__ == '__main__':
    run_multimodal_theremin()
