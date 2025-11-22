"""
streamkeys - Keyboard event streams for Python

A pure, framework-agnostic package for converting keyboard events into feature dictionaries.
Can be used for gaming, shortcuts, automation, music applications, and more.

Zero dependencies on theremin or any specific framework.
"""

from .keyboard_stream import KeyboardFeatureStream, KeyboardEvent
from .mappings import (
    KEY_TO_NOTE,
    keyboard_to_midi_features,
    keyboard_to_chord_features,
)

__version__ = "0.1.0"
__all__ = [
    "KeyboardFeatureStream",
    "KeyboardEvent",
    "KEY_TO_NOTE",
    "keyboard_to_midi_features",
    "keyboard_to_chord_features",
]
