"""
streamtouch - Trackpad and mouse event streams for Python

A pure, framework-agnostic package for converting trackpad/mouse events
into feature dictionaries.

Can be used for UI automation, gesture control, accessibility tools, and more.
Zero dependencies on specific frameworks.
"""

from .trackpad_stream import TrackpadFeatureStream, TrackpadEvent
from .mappings import (
    trackpad_to_audio_knobs,
    trackpad_to_2d_position,
    scroll_to_parameter_delta,
)

__version__ = "0.1.0"
__all__ = [
    "TrackpadFeatureStream",
    "TrackpadEvent",
    "trackpad_to_audio_knobs",
    "trackpad_to_2d_position",
    "scroll_to_parameter_delta",
]
