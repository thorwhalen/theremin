"""Pyo-based synthesizer with dict parameter control."""

from typing import Dict, Optional, Literal
import time


class PyoSynthesizer:
    """
    Dict-based synthesizer using pyo.

    Accepts parameter dictionaries and updates synthesis in real-time.
    Provides low-latency audio generation with comprehensive control.

    Example:
        >>> synth = PyoSynthesizer(sample_rate=44100)
        >>> synth.start()
        >>> synth.update_parameters({'frequency': 440, 'amplitude': 0.5})
        >>> time.sleep(1)
        >>> synth.stop()
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        buffer_size: int = 512,
        waveform: Literal['sine', 'saw', 'square', 'triangle'] = 'sine'
    ):
        """
        Initialize pyo synthesizer.

        Args:
            sample_rate: Sample rate in Hz
            buffer_size: Buffer size in samples
            waveform: Initial waveform type
        """
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.waveform_type = waveform
        self._pyo_available = False
        self.server = None
        self.frequency = None
        self.amplitude = None
        self.osc = None

        # Try to import pyo
        try:
            from pyo import Server, Sig, Sine, Saw, Square, Triangle
            self.pyo = {
                'Server': Server,
                'Sig': Sig,
                'Sine': Sine,
                'Saw': Saw,
                'Square': Square,
                'Triangle': Triangle
            }
            self._pyo_available = True
        except ImportError:
            pass

    def start(self):
        """
        Start the pyo server and synthesis.

        Raises:
            RuntimeError: If pyo is not available
        """
        if not self._pyo_available:
            raise RuntimeError(
                "pyo is not installed. Install with: pip install pyo"
            )

        # Initialize server
        self.server = self.pyo['Server'](
            sr=self.sample_rate,
            buffersize=self.buffer_size
        ).boot()
        self.server.start()

        # Create controllable parameters
        self.frequency = self.pyo['Sig'](440)
        self.amplitude = self.pyo['Sig'](0.5)

        # Create oscillator
        self.osc = self._create_oscillator()
        self.osc.out()

    def stop(self):
        """Stop the pyo server."""
        if self.server:
            self.server.stop()
            self.server.shutdown()
            self.server = None

    def _create_oscillator(self):
        """Create oscillator based on waveform type."""
        if self.waveform_type == 'sine':
            return self.pyo['Sine'](freq=self.frequency, mul=self.amplitude)
        elif self.waveform_type == 'saw':
            return self.pyo['Saw'](freq=self.frequency, mul=self.amplitude)
        elif self.waveform_type == 'square':
            return self.pyo['Square'](freq=self.frequency, mul=self.amplitude)
        elif self.waveform_type == 'triangle':
            return self.pyo['Triangle'](freq=self.frequency, mul=self.amplitude)
        else:
            return self.pyo['Sine'](freq=self.frequency, mul=self.amplitude)

    def update_parameters(self, params: Dict):
        """
        Update synthesis parameters from dictionary.

        Args:
            params: Dictionary with parameters (frequency, amplitude, waveform, etc.)
        """
        if not self.server:
            return

        # Update frequency
        if 'frequency' in params:
            self.frequency.value = params['frequency']

        # Update amplitude
        if 'amplitude' in params:
            self.amplitude.value = params['amplitude']

        # Update waveform if changed
        if 'waveform' in params and params['waveform'] != self.waveform_type:
            self.waveform_type = params['waveform']
            self.osc.stop()
            self.osc = self._create_oscillator()
            self.osc.out()

    def __del__(self):
        """Cleanup on deletion."""
        self.stop()


def create_pyo_synth(
    frequency: float = 440,
    amplitude: float = 0.5,
    waveform: str = 'sine',
    sample_rate: int = 44100
) -> PyoSynthesizer:
    """
    Create and start a pyo synthesizer with initial parameters.

    Args:
        frequency: Initial frequency in Hz
        amplitude: Initial amplitude (0-1)
        waveform: Waveform type
        sample_rate: Sample rate in Hz

    Returns:
        Started PyoSynthesizer instance
    """
    synth = PyoSynthesizer(sample_rate=sample_rate, waveform=waveform)
    synth.start()
    synth.update_parameters({'frequency': frequency, 'amplitude': amplitude})
    return synth
