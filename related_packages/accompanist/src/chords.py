"""Chord progression generation and manipulation."""

from typing import List, Dict, Tuple, Literal
import random


class ChordProgression:
    """
    Represents and manipulates chord progressions.

    Example:
        >>> prog = ChordProgression(['C', 'Am', 'F', 'G'])
        >>> freqs = prog.get_frequencies(0)  # Get frequencies for first chord
    """

    # Common progressions by style
    PROGRESSIONS = {
        'pop': ['C', 'G', 'Am', 'F'],
        'jazz': ['Cmaj7', 'Dm7', 'G7', 'Cmaj7'],
        'blues': ['C7', 'F7', 'C7', 'G7'],
        'minor': ['Am', 'F', 'C', 'G'],
        'fifties': ['C', 'Am', 'Dm', 'G'],
    }

    # Chord to MIDI notes mapping (root position triads)
    CHORD_NOTES = {
        'C': [60, 64, 67],      # C major
        'Dm': [62, 65, 69],     # D minor
        'Em': [64, 67, 71],     # E minor
        'F': [65, 69, 72],      # F major
        'G': [67, 71, 74],      # G major
        'Am': [69, 72, 76],     # A minor
        'Bdim': [71, 74, 77],   # B diminished
        # Seventh chords
        'Cmaj7': [60, 64, 67, 71],
        'Dm7': [62, 65, 69, 72],
        'G7': [67, 71, 74, 65],
        'C7': [60, 64, 67, 70],
        'F7': [65, 69, 72, 75],
    }

    def __init__(self, chords: List[str]):
        """
        Initialize chord progression.

        Args:
            chords: List of chord names (e.g., ['C', 'Am', 'F', 'G'])
        """
        self.chords = chords

    def get_chord(self, index: int) -> str:
        """Get chord at index (wraps around)."""
        return self.chords[index % len(self.chords)]

    def get_midi_notes(self, index: int) -> List[int]:
        """Get MIDI notes for chord at index."""
        chord = self.get_chord(index)
        return self.CHORD_NOTES.get(chord, [60, 64, 67])  # Default to C major

    def get_frequencies(self, index: int) -> List[float]:
        """Get frequencies for chord at index."""
        midi_notes = self.get_midi_notes(index)
        return [note_to_frequency(note) for note in midi_notes]

    def transpose(self, semitones: int) -> 'ChordProgression':
        """
        Transpose the progression by semitones.

        Args:
            semitones: Number of semitones to transpose

        Returns:
            New transposed ChordProgression
        """
        # Simplified transposition (would need full implementation)
        # For now, return self
        return ChordProgression(self.chords)

    def __len__(self):
        """Length of progression."""
        return len(self.chords)

    def __iter__(self):
        """Iterate over chords."""
        return iter(self.chords)


def generate_progression(
    style: Literal['pop', 'jazz', 'blues', 'minor', 'fifties'] = 'pop',
    key: str = 'C',
    length: int = 4
) -> ChordProgression:
    """
    Generate a chord progression.

    Args:
        style: Style of progression
        key: Key (currently only C supported)
        length: Number of chords

    Returns:
        ChordProgression instance
    """
    if style in ChordProgression.PROGRESSIONS:
        chords = ChordProgression.PROGRESSIONS[style]
    else:
        chords = ChordProgression.PROGRESSIONS['pop']

    # Adjust length if needed
    if len(chords) < length:
        chords = chords * (length // len(chords) + 1)
    chords = chords[:length]

    return ChordProgression(chords)


def chord_to_frequencies(chord_name: str) -> List[float]:
    """
    Convert chord name to list of frequencies.

    Args:
        chord_name: Chord name (e.g., 'C', 'Am', 'G7')

    Returns:
        List of frequencies in Hz
    """
    midi_notes = ChordProgression.CHORD_NOTES.get(chord_name, [60, 64, 67])
    return [note_to_frequency(note) for note in midi_notes]


def note_to_frequency(midi_note: int) -> float:
    """
    Convert MIDI note number to frequency.

    Args:
        midi_note: MIDI note number (0-127)

    Returns:
        Frequency in Hz
    """
    return 440.0 * (2 ** ((midi_note - 69) / 12))
