"""MIDI utilities and playback."""

from typing import List, Dict, Optional
import time


class MIDIPlayer:
    """
    Simple MIDI playback using mido (if available).

    Example:
        >>> player = MIDIPlayer()
        >>> player.play_chord([60, 64, 67], duration=1.0)
    """

    def __init__(self):
        """Initialize MIDI player."""
        self._mido_available = False
        self.output = None

        # Try to import mido
        try:
            import mido
            self.mido = mido
            self._mido_available = True
        except ImportError:
            pass

    def open(self, port_name: Optional[str] = None):
        """
        Open MIDI output port.

        Args:
            port_name: Optional MIDI port name. If None, uses default.

        Raises:
            RuntimeError: If mido is not available
        """
        if not self._mido_available:
            raise RuntimeError(
                "mido is not installed. Install with: pip install mido"
            )

        if port_name:
            self.output = self.mido.open_output(port_name)
        else:
            self.output = self.mido.open_output()

    def close(self):
        """Close MIDI output port."""
        if self.output:
            self.output.close()
            self.output = None

    def play_note(
        self,
        note: int,
        velocity: int = 80,
        duration: float = 1.0,
        channel: int = 0
    ):
        """
        Play a single MIDI note.

        Args:
            note: MIDI note number (0-127)
            velocity: Note velocity (0-127)
            duration: Duration in seconds
            channel: MIDI channel (0-15)
        """
        if not self.output:
            return

        # Note on
        self.output.send(self.mido.Message(
            'note_on',
            note=note,
            velocity=velocity,
            channel=channel
        ))

        # Wait
        time.sleep(duration)

        # Note off
        self.output.send(self.mido.Message(
            'note_off',
            note=note,
            velocity=0,
            channel=channel
        ))

    def play_chord(
        self,
        notes: List[int],
        velocity: int = 80,
        duration: float = 1.0,
        channel: int = 0
    ):
        """
        Play a chord (multiple notes simultaneously).

        Args:
            notes: List of MIDI note numbers
            velocity: Note velocity
            duration: Duration in seconds
            channel: MIDI channel
        """
        if not self.output:
            return

        # Notes on
        for note in notes:
            self.output.send(self.mido.Message(
                'note_on',
                note=note,
                velocity=velocity,
                channel=channel
            ))

        # Wait
        time.sleep(duration)

        # Notes off
        for note in notes:
            self.output.send(self.mido.Message(
                'note_off',
                note=note,
                velocity=0,
                channel=channel
            ))

    def __del__(self):
        """Cleanup on deletion."""
        self.close()


def chord_to_midi_notes(chord_name: str) -> List[int]:
    """
    Convert chord name to MIDI note numbers.

    Args:
        chord_name: Chord name (e.g., 'C', 'Am')

    Returns:
        List of MIDI note numbers
    """
    from .chords import ChordProgression
    return ChordProgression.CHORD_NOTES.get(chord_name, [60, 64, 67])


def note_to_frequency(midi_note: int) -> float:
    """
    Convert MIDI note to frequency.

    Args:
        midi_note: MIDI note number

    Returns:
        Frequency in Hz
    """
    return 440.0 * (2 ** ((midi_note - 69) / 12))
