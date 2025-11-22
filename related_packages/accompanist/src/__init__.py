"""
accompanist - Music accompaniment tools for Python

A framework-agnostic package for generating musical accompaniment.
Includes chord progression generation, MIDI processing, and harmonic analysis.

Useful for music generation, improvisation support, and educational applications.
"""

from .chords import ChordProgression, generate_progression, chord_to_frequencies
from .midi_utils import MIDIPlayer, chord_to_midi_notes, note_to_frequency

__version__ = "0.1.0"
__all__ = [
    "ChordProgression",
    "generate_progression",
    "chord_to_frequencies",
    "MIDIPlayer",
    "chord_to_midi_notes",
    "note_to_frequency",
]
