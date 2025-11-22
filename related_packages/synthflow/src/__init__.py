"""
synthflow - Dict-based synthesizer control for Python

A framework-agnostic package for controlling audio synthesis via parameter dictionaries.
Provides standardized wrappers for pyo, SignalFlow, and other synthesis engines.

Useful for any Python audio synthesis project requiring parameter-based control.
"""

from .pyo_synth import PyoSynthesizer, create_pyo_synth
from .simple_synth import SimpleSynthesizer, synthesize_simple
from .effects import EffectsProcessor

__version__ = "0.1.0"
__all__ = [
    "PyoSynthesizer",
    "create_pyo_synth",
    "SimpleSynthesizer",
    "synthesize_simple",
    "EffectsProcessor",
]
