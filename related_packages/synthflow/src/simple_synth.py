"""Simple NumPy-based synthesizer (no external dependencies)."""

import numpy as np
from typing import Dict, Optional, Literal


class SimpleSynthesizer:
    """
    Simple synthesizer using NumPy (no external synthesis libraries required).

    Generates audio chunks from parameter dictionaries.
    Suitable for basic synthesis and testing.

    Example:
        >>> synth = SimpleSynthesizer(sample_rate=44100)
        >>> audio = synth.generate({'frequency': 440, 'amplitude': 0.5, 'duration': 0.1})
        >>> # audio is a NumPy array of samples
    """

    def __init__(self, sample_rate: int = 44100):
        """
        Initialize simple synthesizer.

        Args:
            sample_rate: Sample rate in Hz
        """
        self.sample_rate = sample_rate

    def generate(
        self,
        params: Dict,
        duration: Optional[float] = None
    ) -> np.ndarray:
        """
        Generate audio from parameters.

        Args:
            params: Dictionary with frequency, amplitude, duration, waveform
            duration: Override duration (seconds), or use params['duration']

        Returns:
            NumPy array of audio samples (float32)
        """
        # Extract parameters with defaults
        frequency = params.get('frequency', 440.0)
        amplitude = params.get('amplitude', 0.5)
        waveform = params.get('waveform', 'sine')

        if duration is None:
            duration = params.get('duration', 0.1)

        # Generate time array
        num_samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, num_samples, dtype=np.float32)

        # Generate waveform
        if waveform == 'sine':
            audio = np.sin(2 * np.pi * frequency * t)
        elif waveform == 'saw':
            audio = 2 * (t * frequency - np.floor(0.5 + t * frequency))
        elif waveform == 'square':
            audio = np.sign(np.sin(2 * np.pi * frequency * t))
        elif waveform == 'triangle':
            audio = 2 * np.abs(2 * (t * frequency - np.floor(0.5 + t * frequency))) - 1
        else:
            audio = np.sin(2 * np.pi * frequency * t)

        # Apply amplitude
        audio = audio * amplitude

        # Apply envelope if specified
        if params.get('envelope', True):
            audio = self._apply_envelope(audio, params)

        return audio.astype(np.float32)

    def _apply_envelope(self, audio: np.ndarray, params: Dict) -> np.ndarray:
        """Apply ADSR envelope to audio."""
        length = len(audio)

        # Get envelope parameters (in samples)
        attack = int(params.get('attack', 0.01) * self.sample_rate)
        release = int(params.get('release', 0.05) * self.sample_rate)

        # Ensure attack and release fit within audio length
        attack = min(attack, length // 4)
        release = min(release, length // 4)

        # Create envelope
        envelope = np.ones_like(audio)

        if attack > 0:
            envelope[:attack] = np.linspace(0, 1, attack)

        if release > 0:
            envelope[-release:] = np.linspace(1, 0, release)

        return audio * envelope


def synthesize_simple(
    frequency: float = 440,
    amplitude: float = 0.5,
    duration: float = 0.1,
    waveform: Literal['sine', 'saw', 'square', 'triangle'] = 'sine',
    sample_rate: int = 44100
) -> np.ndarray:
    """
    Convenience function to generate audio.

    Args:
        frequency: Frequency in Hz
        amplitude: Amplitude (0-1)
        duration: Duration in seconds
        waveform: Waveform type
        sample_rate: Sample rate in Hz

    Returns:
        NumPy array of audio samples
    """
    synth = SimpleSynthesizer(sample_rate=sample_rate)
    params = {
        'frequency': frequency,
        'amplitude': amplitude,
        'duration': duration,
        'waveform': waveform
    }
    return synth.generate(params)
