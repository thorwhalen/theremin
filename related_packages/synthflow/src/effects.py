"""Audio effects processing."""

import numpy as np
from typing import Dict, Optional


class EffectsProcessor:
    """
    Process audio with various effects based on parameter dictionaries.

    Example:
        >>> processor = EffectsProcessor(sample_rate=44100)
        >>> params = {'reverb': 0.5, 'filter_cutoff': 2000}
        >>> processed = processor.process(audio, params)
    """

    def __init__(self, sample_rate: int = 44100):
        """
        Initialize effects processor.

        Args:
            sample_rate: Sample rate in Hz
        """
        self.sample_rate = sample_rate

    def process(self, audio: np.ndarray, params: Dict) -> np.ndarray:
        """
        Process audio with effects from parameters.

        Args:
            audio: Input audio (NumPy array)
            params: Effect parameters

        Returns:
            Processed audio
        """
        processed = audio.copy()

        # Apply filter if specified
        if 'filter_cutoff' in params:
            processed = self.apply_lowpass(
                processed,
                cutoff=params['filter_cutoff'],
                resonance=params.get('resonance', 0.5)
            )

        # Apply distortion if specified
        if 'distortion' in params and params['distortion'] > 0:
            processed = self.apply_distortion(
                processed,
                amount=params['distortion']
            )

        # Apply reverb if specified
        if 'reverb' in params and params['reverb'] > 0:
            processed = self.apply_simple_reverb(
                processed,
                amount=params['reverb']
            )

        return processed

    def apply_lowpass(
        self,
        audio: np.ndarray,
        cutoff: float,
        resonance: float = 0.5
    ) -> np.ndarray:
        """
        Apply simple lowpass filter.

        Args:
            audio: Input audio
            cutoff: Cutoff frequency in Hz
            resonance: Filter resonance (0-1)

        Returns:
            Filtered audio
        """
        try:
            from scipy.signal import butter, filtfilt

            nyquist = self.sample_rate / 2
            cutoff_norm = min(cutoff / nyquist, 0.99)

            # Create butterworth filter
            order = 4
            b, a = butter(order, cutoff_norm, btype='low')

            # Apply filter
            return filtfilt(b, a, audio)

        except ImportError:
            # Fallback: simple RC filter
            alpha = 2 * np.pi * cutoff / self.sample_rate
            alpha = min(alpha, 0.99)

            filtered = np.zeros_like(audio)
            filtered[0] = audio[0]

            for i in range(1, len(audio)):
                filtered[i] = alpha * audio[i] + (1 - alpha) * filtered[i-1]

            return filtered

    def apply_distortion(self, audio: np.ndarray, amount: float) -> np.ndarray:
        """
        Apply soft clipping distortion.

        Args:
            audio: Input audio
            amount: Distortion amount (0-1)

        Returns:
            Distorted audio
        """
        # Soft clip using tanh
        gain = 1 + amount * 10
        return np.tanh(audio * gain) / np.tanh(gain)

    def apply_simple_reverb(self, audio: np.ndarray, amount: float) -> np.ndarray:
        """
        Apply simple reverb (delay-based).

        Args:
            audio: Input audio
            amount: Reverb amount (0-1)

        Returns:
            Audio with reverb
        """
        # Simple delay-based reverb
        delay_samples = int(0.05 * self.sample_rate)  # 50ms delay
        delayed = np.zeros_like(audio)

        if len(audio) > delay_samples:
            delayed[delay_samples:] = audio[:-delay_samples]

        # Mix original with delayed
        return audio + delayed * amount * 0.5
