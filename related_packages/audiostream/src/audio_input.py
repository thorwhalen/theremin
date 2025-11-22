"""Audio input stream capture."""

import time
from queue import Queue
from typing import Optional
import numpy as np


class AudioInputStream:
    """
    Capture live audio input streams.

    Uses sounddevice for low-latency audio capture with NumPy integration.

    Example:
        >>> stream = AudioInputStream(sample_rate=44100, block_size=2048)
        >>> stream.start()
        >>>
        >>> while True:
        >>>     chunk = stream.get_audio_chunk()
        >>>     if chunk is not None:
        >>>         # Process audio chunk
        >>>         print(f"Audio level: {np.abs(chunk).mean()}")
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        block_size: int = 2048,
        channels: int = 1,
        dtype: str = 'float32'
    ):
        """
        Initialize audio input stream.

        Args:
            sample_rate: Sample rate in Hz (default 44100)
            block_size: Number of frames per block (default 2048)
            channels: Number of audio channels (default 1 for mono)
            dtype: Data type for audio samples (default 'float32')
        """
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.channels = channels
        self.dtype = dtype
        self.audio_queue = Queue()
        self.stream = None
        self._sounddevice_available = False

        # Try to import sounddevice
        try:
            import sounddevice as sd
            self.sd = sd
            self._sounddevice_available = True
        except ImportError:
            pass

    def audio_callback(self, indata, frames, time_info, status):
        """Callback for incoming audio data."""
        if status:
            print(f"Audio input status: {status}")

        # Copy audio data to avoid buffer issues
        self.audio_queue.put(indata.copy())

    def start(self):
        """
        Start audio input stream.

        Raises:
            RuntimeError: If sounddevice is not available
        """
        if not self._sounddevice_available:
            raise RuntimeError(
                "sounddevice is not installed. Install with: pip install sounddevice"
            )

        self.stream = self.sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            channels=self.channels,
            dtype=self.dtype,
            callback=self.audio_callback
        )
        self.stream.start()

    def stop(self):
        """Stop audio input stream."""
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def get_audio_chunk(self) -> Optional[np.ndarray]:
        """
        Get next audio chunk from the queue.

        Returns:
            NumPy array of audio samples, or None if queue is empty
        """
        if not self.audio_queue.empty():
            return self.audio_queue.get()
        return None

    def get_queue_size(self) -> int:
        """Get the number of chunks waiting in the queue."""
        return self.audio_queue.qsize()

    def clear_queue(self):
        """Clear all waiting audio chunks from the queue."""
        while not self.audio_queue.empty():
            self.audio_queue.get()

    def __del__(self):
        """Cleanup on deletion."""
        self.stop()
