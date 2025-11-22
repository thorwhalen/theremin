"""Tests for keyboard stream functionality."""

import pytest
from streamkeys.keyboard_stream import KeyboardFeatureStream, KeyboardEvent
from streamkeys.mappings import (
    keyboard_to_midi_features,
    keyboard_to_chord_features,
    midi_note_to_frequency,
    KEY_TO_NOTE
)


def test_keyboard_event_creation():
    """Test KeyboardEvent dataclass."""
    event = KeyboardEvent(
        action='press',
        key='a',
        active_keys=['a'],
        timestamp=123.456
    )

    assert event.type == 'keyboard'
    assert event.action == 'press'
    assert event.key == 'a'
    assert event.active_keys == ['a']
    assert event.timestamp == 123.456


def test_keyboard_event_to_dict():
    """Test converting event to dictionary."""
    event = KeyboardEvent(
        action='press',
        key='a',
        active_keys=['a', 's'],
        timestamp=123.456
    )

    event_dict = event.to_dict()
    assert event_dict['type'] == 'keyboard'
    assert event_dict['action'] == 'press'
    assert event_dict['key'] == 'a'
    assert 'a' in event_dict['active_keys']
    assert 's' in event_dict['active_keys']


def test_midi_note_to_frequency():
    """Test MIDI note to frequency conversion."""
    # A4 (440 Hz standard)
    assert abs(midi_note_to_frequency(69) - 440.0) < 0.01

    # C4 (middle C)
    c4_freq = midi_note_to_frequency(60)
    assert abs(c4_freq - 261.63) < 0.1

    # C5 (one octave up from middle C)
    c5_freq = midi_note_to_frequency(72)
    assert abs(c5_freq - 523.25) < 0.1


def test_keyboard_to_midi_features():
    """Test keyboard event to MIDI features conversion."""
    # Press event
    event = {
        'action': 'press',
        'key': 'a',
        'active_keys': ['a'],
        'timestamp': 123.456
    }

    features = keyboard_to_midi_features(event)

    assert features is not None
    assert 'frequency' in features
    assert 'amplitude' in features
    assert features['note_on'] is True
    assert features['midi_note'] == 60  # 'a' maps to C4
    assert abs(features['frequency'] - 261.63) < 0.1

    # Release event - should return None
    release_event = {
        'action': 'release',
        'key': 'a',
        'active_keys': [],
        'timestamp': 123.5
    }

    assert keyboard_to_midi_features(release_event) is None

    # Unmapped key - should return None
    unmapped_event = {
        'action': 'press',
        'key': '1',
        'active_keys': ['1'],
        'timestamp': 123.5
    }

    assert keyboard_to_midi_features(unmapped_event) is None


def test_keyboard_to_chord_features():
    """Test keyboard event to chord features conversion."""
    # Single key
    single_key_event = {
        'action': 'press',
        'key': 'a',
        'active_keys': ['a'],
        'timestamp': 123.456
    }

    chord = keyboard_to_chord_features(single_key_event)

    assert len(chord['frequencies']) == 1
    assert len(chord['midi_notes']) == 1
    assert chord['num_voices'] == 1
    assert chord['amplitude'] == 0.5  # 0.5 / 1

    # Multiple keys (chord)
    chord_event = {
        'action': 'press',
        'key': 'a',
        'active_keys': ['a', 'd', 'g'],  # C-E-G (C major chord)
        'timestamp': 123.456
    }

    chord = keyboard_to_chord_features(chord_event)

    assert len(chord['frequencies']) == 3
    assert len(chord['midi_notes']) == 3
    assert chord['num_voices'] == 3
    assert abs(chord['amplitude'] - (0.5 / 3)) < 0.01  # Normalized

    # No mapped keys
    empty_event = {
        'action': 'press',
        'key': '1',
        'active_keys': ['1', '2'],
        'timestamp': 123.456
    }

    chord = keyboard_to_chord_features(empty_event)

    assert chord['frequencies'] == []
    assert chord['amplitude'] == 0
    assert chord['num_voices'] == 0


def test_keyboard_feature_stream_creation():
    """Test KeyboardFeatureStream instantiation."""
    stream = KeyboardFeatureStream()

    assert stream.event_queue is not None
    assert isinstance(stream.active_keys, set)
    assert len(stream.active_keys) == 0


def test_get_features_empty_queue():
    """Test getting features from empty queue."""
    stream = KeyboardFeatureStream()

    features = stream.get_features()
    assert features is None


def test_simulated_key_events():
    """Test simulating keyboard events without actual listener."""
    stream = KeyboardFeatureStream()

    # Simulate a key press
    event = KeyboardEvent(
        action='press',
        key='a',
        active_keys=['a'],
        timestamp=123.456
    )
    stream.event_queue.put(event)

    # Retrieve the event
    features = stream.get_features()

    assert features is not None
    assert features['action'] == 'press'
    assert features['key'] == 'a'
    assert 'a' in features['active_keys']


def test_key_mappings_coverage():
    """Test that all expected keys are mapped."""
    # Test home row
    assert 'a' in KEY_TO_NOTE
    assert 's' in KEY_TO_NOTE
    assert 'd' in KEY_TO_NOTE
    assert 'f' in KEY_TO_NOTE
    assert 'g' in KEY_TO_NOTE
    assert 'h' in KEY_TO_NOTE
    assert 'j' in KEY_TO_NOTE
    assert 'k' in KEY_TO_NOTE

    # Test that different rows map to different octaves
    assert KEY_TO_NOTE['z'] < KEY_TO_NOTE['a']  # Lower octave
    assert KEY_TO_NOTE['a'] < KEY_TO_NOTE['q']  # Higher octave


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
