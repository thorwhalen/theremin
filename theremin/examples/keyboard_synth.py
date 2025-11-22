"""
Keyboard Synthesizer Example.

Demonstrates using streamkeys + synthflow for piano-like keyboard playing.
"""

import time
import sys
from pathlib import Path

# Add related packages to path (for development)
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'related_packages/streamkeys/src'))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'related_packages/synthflow/src'))

try:
    from streamkeys import KeyboardFeatureStream, keyboard_to_midi_features
    from synthflow import SimpleSynthesizer
except ImportError as e:
    print(f"Missing dependencies: {e}")
    print("Install with: pip install -e related_packages/streamkeys -e related_packages/synthflow")
    sys.exit(1)


def run_keyboard_synth():
    """
    Run keyboard synthesizer example.

    Press keys on home row (ASDFGHJKL) to play notes.
    Press ESC to quit.
    """
    print("Keyboard Synthesizer")
    print("=" * 50)
    print("Press keys on home row (A-L) to play notes")
    print("Press ESC to quit")
    print()

    # Create keyboard stream
    keyboard_stream = KeyboardFeatureStream()
    keyboard_stream.start()

    # Create synthesizer
    synth = SimpleSynthesizer(sample_rate=44100)

    try:
        while True:
            # Get keyboard event
            event = keyboard_stream.get_features()

            if event:
                # Check for ESC key to quit
                if event.get('key') == 'esc':
                    break

                # Convert to MIDI features
                midi_features = keyboard_to_midi_features(event)

                if midi_features:
                    # Generate audio
                    audio = synth.generate({
                        'frequency': midi_features['frequency'],
                        'amplitude': midi_features['amplitude'],
                        'duration': 0.2,
                        'waveform': 'sine'
                    })

                    print(f"🎵 Note: {event['key']} -> {midi_features['frequency']:.1f} Hz")

                    # In a real application, you would play the audio here
                    # For now, we just generate it

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        keyboard_stream.stop()


def run_keyboard_synth_with_meshed():
    """
    Run keyboard synth using meshed DAG for auto-wiring.

    This demonstrates the theremin framework architecture.
    """
    try:
        from meshed import DAG
    except ImportError:
        print("meshed not installed. Install with: pip install meshed")
        return

    # Define pipeline functions
    def get_keyboard_event(keyboard_stream) -> dict:
        """Get next keyboard event."""
        event = keyboard_stream.get_features()
        if event:
            return event
        return {}

    def event_to_midi(keyboard_event) -> dict:
        """Convert keyboard event to MIDI features."""
        if not keyboard_event:
            return {}
        features = keyboard_to_midi_features(keyboard_event)
        return features if features else {}

    def midi_to_synth_params(frequency, amplitude, **kwargs) -> dict:
        """Map MIDI features to synth parameters."""
        return {
            'frequency': frequency,
            'amplitude': amplitude,
            'duration': 0.2,
            'waveform': 'sine'
        }

    def generate_audio(synth, **params) -> object:
        """Generate audio from parameters."""
        if not params:
            return None
        return synth.generate(params)

    # Create components
    keyboard_stream = KeyboardFeatureStream()
    keyboard_stream.start()
    synth = SimpleSynthesizer()

    # Create pipeline
    pipeline = DAG([
        lambda: get_keyboard_event(keyboard_stream),
        event_to_midi,
        midi_to_synth_params,
        lambda **p: generate_audio(synth, **p)
    ])

    print("Keyboard Synthesizer (meshed DAG version)")
    print("=" * 50)
    print("Press keys to play notes, ESC to quit")
    print()

    try:
        while True:
            result = pipeline()
            if result is not None:
                print("🎵 Generated audio")
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nStopped")
    finally:
        keyboard_stream.stop()


if __name__ == '__main__':
    # Run simple version
    run_keyboard_synth()

    # Uncomment to try meshed version
    # run_keyboard_synth_with_meshed()
