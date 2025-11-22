"""Keyboard to musical feature mappings."""

from typing import Dict, List, Optional


# Keyboard to MIDI note mapping
# Rows represent different octaves, columns are scale degrees
KEY_TO_NOTE = {
    # Home row - Middle octave (C4-C5)
    'a': 60,  # C4
    's': 62,  # D4
    'd': 64,  # E4
    'f': 65,  # F4
    'g': 67,  # G4
    'h': 69,  # A4
    'j': 71,  # B4
    'k': 72,  # C5
    'l': 74,  # D5

    # Top row - Higher octave (C5-C6)
    'q': 72,  # C5
    'w': 74,  # D5
    'e': 76,  # E5
    'r': 77,  # F5
    't': 79,  # G5
    'y': 81,  # A5
    'u': 83,  # B5
    'i': 84,  # C6
    'o': 86,  # D6
    'p': 88,  # E6

    # Bottom row - Lower octave (C3-C4)
    'z': 48,  # C3
    'x': 50,  # D3
    'c': 52,  # E3
    'v': 53,  # F3
    'b': 55,  # G3
    'n': 57,  # A3
    'm': 59,  # B3
}


def midi_note_to_frequency(midi_note: int) -> float:
    """
    Convert MIDI note number to frequency in Hz.

    Args:
        midi_note: MIDI note number (0-127)

    Returns:
        Frequency in Hz
    """
    return 440.0 * (2 ** ((midi_note - 69) / 12))


def keyboard_to_midi_features(keyboard_event: Dict) -> Optional[Dict]:
    """
    Convert keyboard event to MIDI-like features.

    Args:
        keyboard_event: Dictionary with keyboard event data

    Returns:
        Dictionary with frequency, amplitude, note_on, and timestamp
        Returns None if the event is not a key press or key is not mapped
    """
    if keyboard_event.get('action') != 'press':
        return None

    key = keyboard_event.get('key')
    if key not in KEY_TO_NOTE:
        return None

    midi_note = KEY_TO_NOTE[key]
    frequency = midi_note_to_frequency(midi_note)

    return {
        'frequency': frequency,
        'amplitude': 0.5,
        'note_on': True,
        'midi_note': midi_note,
        'timestamp': keyboard_event.get('timestamp', 0)
    }


def keyboard_to_chord_features(keyboard_event: Dict) -> Dict:
    """
    Map active keys to chord features.

    This function looks at all currently pressed keys and generates
    a polyphonic output with multiple frequencies.

    Args:
        keyboard_event: Dictionary with keyboard event data

    Returns:
        Dictionary with frequencies list and amplitude
    """
    active_keys = keyboard_event.get('active_keys', [])

    # Get MIDI notes for all active keys
    active_notes = [
        KEY_TO_NOTE[k] for k in active_keys
        if k in KEY_TO_NOTE
    ]

    if not active_notes:
        return {
            'frequencies': [],
            'amplitude': 0,
            'midi_notes': [],
            'timestamp': keyboard_event.get('timestamp', 0)
        }

    # Convert to frequencies
    frequencies = [midi_note_to_frequency(note) for note in active_notes]

    # Normalize amplitude by number of notes to avoid clipping
    amplitude = 0.5 / len(frequencies) if frequencies else 0

    return {
        'frequencies': frequencies,
        'amplitude': amplitude,
        'midi_notes': active_notes,
        'num_voices': len(frequencies),
        'timestamp': keyboard_event.get('timestamp', 0)
    }
