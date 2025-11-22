"""Core keyboard event stream processing."""

from queue import Queue
import time
from typing import Dict, Set, Optional, List
from dataclasses import dataclass, asdict


@dataclass
class KeyboardEvent:
    """Represents a keyboard event."""

    type: str = 'keyboard'
    action: str = 'press'  # 'press' or 'release'
    key: str = ''
    active_keys: List[str] = None
    timestamp: float = 0.0

    def __post_init__(self):
        if self.active_keys is None:
            self.active_keys = []

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


class KeyboardFeatureStream:
    """
    Convert keyboard events to feature dictionaries.

    This class provides a framework-agnostic interface for keyboard event capture.
    It uses pynput for cross-platform keyboard monitoring.

    Example:
        >>> stream = KeyboardFeatureStream()
        >>> stream.start()
        >>>
        >>> while True:
        >>>     event = stream.get_features()
        >>>     if event:
        >>>         print(f"Key {event['key']} was {event['action']}ed")
        >>>     time.sleep(0.01)
    """

    def __init__(self):
        """Initialize the keyboard feature stream."""
        self.event_queue = Queue()
        self.active_keys: Set[str] = set()
        self.listener = None
        self._pynput_available = False

        # Try to import pynput
        try:
            from pynput import keyboard
            self._keyboard_module = keyboard
            self._pynput_available = True
        except ImportError:
            self._keyboard_module = None

    def on_press(self, key):
        """Handle key press event."""
        try:
            key_char = key.char
        except AttributeError:
            key_char = str(key).replace('Key.', '')

        self.active_keys.add(key_char)

        event = KeyboardEvent(
            type='keyboard',
            action='press',
            key=key_char,
            active_keys=list(self.active_keys),
            timestamp=time.time()
        )
        self.event_queue.put(event)

    def on_release(self, key):
        """Handle key release event."""
        try:
            key_char = key.char
        except AttributeError:
            key_char = str(key).replace('Key.', '')

        self.active_keys.discard(key_char)

        event = KeyboardEvent(
            type='keyboard',
            action='release',
            key=key_char,
            active_keys=list(self.active_keys),
            timestamp=time.time()
        )
        self.event_queue.put(event)

    def start(self):
        """
        Start non-blocking keyboard listener.

        Raises:
            RuntimeError: If pynput is not available
        """
        if not self._pynput_available:
            raise RuntimeError(
                "pynput is not installed. Install it with: pip install pynput"
            )

        self.listener = self._keyboard_module.Listener(
            on_press=self.on_press,
            on_release=self.on_release
        )
        self.listener.start()

    def stop(self):
        """Stop the keyboard listener."""
        if self.listener:
            self.listener.stop()
            self.listener = None

    def get_features(self) -> Optional[Dict]:
        """
        Get next keyboard feature dict.

        Returns:
            Dictionary with keyboard event data, or None if queue is empty
        """
        if not self.event_queue.empty():
            event = self.event_queue.get()
            return event.to_dict()
        return None

    def get_active_keys(self) -> Set[str]:
        """Get the set of currently active (pressed) keys."""
        return self.active_keys.copy()

    def is_key_active(self, key: str) -> bool:
        """Check if a specific key is currently pressed."""
        return key in self.active_keys
