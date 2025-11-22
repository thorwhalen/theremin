"""
Theremin framework core - meshed.slabs based architecture.

This module provides the core framework for building modular signal processing
pipelines using functional composition and automatic parameter wiring.
"""

from .base import (
    SensorReader,
    FeatureExtractor,
    FeatureMapper,
    Synthesizer,
    Pipeline,
)
from .storage import CalibrationStore, PresetStore
from .testing import StreamPlayer, AudioVerifier

__all__ = [
    "SensorReader",
    "FeatureExtractor",
    "FeatureMapper",
    "Synthesizer",
    "Pipeline",
    "CalibrationStore",
    "PresetStore",
    "StreamPlayer",
    "AudioVerifier",
]
