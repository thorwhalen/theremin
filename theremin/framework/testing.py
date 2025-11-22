"""Testing utilities for pipeline validation."""

import json
from pathlib import Path
from typing import Dict, List, Iterator, Optional
import numpy as np


class StreamPlayer:
    """
    Replay feature streams from JSON for deterministic testing.

    This enables testing audio synthesis without camera/video hardware
    by using pre-recorded feature streams.

    Example:
        >>> player = StreamPlayer('test_features.json')
        >>> for features in player:
        >>>     audio_params = map_features_to_audio(features)
        >>>     audio = synthesize(audio_params)
    """

    def __init__(self, feature_file: str):
        """
        Initialize stream player.

        Args:
            feature_file: Path to JSON file with recorded features
        """
        self.feature_file = Path(feature_file)
        self.features: List[Dict] = []
        self._load_features()

    def _load_features(self):
        """Load features from JSON file."""
        if not self.feature_file.exists():
            raise FileNotFoundError(f"Feature file not found: {self.feature_file}")

        with open(self.feature_file, 'r') as f:
            data = json.load(f)

            # Handle different JSON formats
            if isinstance(data, list):
                self.features = data
            elif isinstance(data, dict) and 'features' in data:
                self.features = data['features']
            else:
                raise ValueError(f"Unknown feature file format: {self.feature_file}")

    def __iter__(self) -> Iterator[Dict]:
        """Iterate over features."""
        return iter(self.features)

    def __len__(self) -> int:
        """Number of feature frames."""
        return len(self.features)

    def __getitem__(self, index: int) -> Dict:
        """Get features at specific index."""
        return self.features[index]


class AudioVerifier:
    """
    Verify audio output quality and correctness.

    Provides utilities for testing audio generation:
    - Frequency content verification
    - Silence detection
    - Amplitude checks
    - Spectral analysis

    Example:
        >>> verifier = AudioVerifier()
        >>> audio = generate_audio(frequency=440, amplitude=0.5)
        >>> assert verifier.verify_frequency_content(audio, 44100, 440)
        >>> assert verifier.verify_not_silent(audio)
    """

    @staticmethod
    def verify_frequency_content(
        audio: np.ndarray,
        sample_rate: int,
        expected_freq: float,
        tolerance: float = 10.0
    ) -> bool:
        """
        Check if expected frequency is present in audio.

        Args:
            audio: Audio samples (NumPy array)
            sample_rate: Sample rate in Hz
            expected_freq: Expected frequency in Hz
            tolerance: Tolerance in Hz

        Returns:
            True if expected frequency is present
        """
        # Compute FFT
        fft = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(len(audio), 1/sample_rate)

        # Find peak frequency
        peak_idx = np.argmax(np.abs(fft))
        peak_freq = freqs[peak_idx]

        # Check if within tolerance
        return abs(peak_freq - expected_freq) < tolerance

    @staticmethod
    def verify_not_silent(audio: np.ndarray, min_std: float = 0.01) -> bool:
        """
        Ensure audio has variation (not silent).

        Args:
            audio: Audio samples
            min_std: Minimum standard deviation

        Returns:
            True if audio has variation
        """
        return np.std(audio) > min_std

    @staticmethod
    def verify_amplitude_range(
        audio: np.ndarray,
        min_amp: float = 0.0,
        max_amp: float = 1.0
    ) -> bool:
        """
        Verify audio amplitude is within expected range.

        Args:
            audio: Audio samples
            min_amp: Minimum expected amplitude
            max_amp: Maximum expected amplitude

        Returns:
            True if amplitude is within range
        """
        peak = np.max(np.abs(audio))
        return min_amp <= peak <= max_amp

    @staticmethod
    def measure_rms(audio: np.ndarray) -> float:
        """
        Measure RMS amplitude of audio.

        Args:
            audio: Audio samples

        Returns:
            RMS amplitude
        """
        return float(np.sqrt(np.mean(audio ** 2)))

    @staticmethod
    def has_no_nan_or_inf(audio: np.ndarray) -> bool:
        """
        Check that audio has no NaN or infinite values.

        Args:
            audio: Audio samples

        Returns:
            True if audio is valid
        """
        return not (np.any(np.isnan(audio)) or np.any(np.isinf(audio)))
