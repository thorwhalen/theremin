"""Core trackpad/mouse event stream processing."""

from queue import Queue
import time
from typing import Dict, Optional
from dataclasses import dataclass, asdict, field


@dataclass
class TrackpadEvent:
    """Represents a trackpad/mouse event."""

    type: str = 'trackpad'
    action: str = 'move'  # 'move', 'press', 'release', 'scroll'
    position: Dict[str, float] = field(default_factory=lambda: {'x': 0, 'y': 0})
    button: Optional[str] = None
    delta: Dict[str, float] = field(default_factory=lambda: {'dx': 0, 'dy': 0})
    direction: Optional[str] = None  # 'horizontal' or 'vertical' for scroll
    magnitude: float = 0.0
    timestamp: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        result = asdict(self)
        # Remove None values for cleaner output
        return {k: v for k, v in result.items() if v is not None}


class TrackpadFeatureStream:
    """
    Convert trackpad/mouse events to feature dictionaries.

    This class provides a framework-agnostic interface for trackpad/mouse monitoring.
    Uses pynput for cross-platform support.

    Note: Full multi-touch gestures (pinch, rotate) on macOS require PyObjC and
    are not supported in this lightweight implementation. Scroll events provide
    a practical proxy for directional input.

    Example:
        >>> stream = TrackpadFeatureStream()
        >>> stream.start()
        >>>
        >>> while True:
        >>>     event = stream.get_features()
        >>>     if event:
        >>>         if event['action'] == 'scroll':
        >>>             print(f"Scrolled: {event['direction']}, magnitude: {event['magnitude']}")
        >>>     time.sleep(0.01)
    """

    def __init__(self):
        """Initialize the trackpad feature stream."""
        self.event_queue = Queue()
        self.current_position = {'x': 0, 'y': 0}
        self.listener = None
        self._pynput_available = False

        # Try to import pynput
        try:
            from pynput import mouse
            self._mouse_module = mouse
            self._pynput_available = True
        except ImportError:
            self._mouse_module = None

    def on_move(self, x, y):
        """Track cursor position."""
        self.current_position = {'x': x, 'y': y}

        event = TrackpadEvent(
            type='trackpad',
            action='move',
            position={'x': x, 'y': y},
            timestamp=time.time()
        )
        self.event_queue.put(event)

    def on_click(self, x, y, button, pressed):
        """Single-finger tap/click detection."""
        action = 'press' if pressed else 'release'

        event = TrackpadEvent(
            type='trackpad',
            action=action,
            position={'x': x, 'y': y},
            button=str(button),
            timestamp=time.time()
        )
        self.event_queue.put(event)

    def on_scroll(self, x, y, dx, dy):
        """Two-finger scroll as gesture proxy."""
        # Determine scroll direction based on larger delta
        if abs(dx) > abs(dy):
            direction = 'horizontal'
            magnitude = abs(dx)
        else:
            direction = 'vertical'
            magnitude = abs(dy)

        event = TrackpadEvent(
            type='trackpad',
            action='scroll',
            position={'x': x, 'y': y},
            delta={'dx': dx, 'dy': dy},
            direction=direction,
            magnitude=magnitude,
            timestamp=time.time()
        )
        self.event_queue.put(event)

    def start(self):
        """
        Start non-blocking trackpad listener.

        Raises:
            RuntimeError: If pynput is not available
        """
        if not self._pynput_available:
            raise RuntimeError(
                "pynput is not installed. Install it with: pip install pynput"
            )

        self.listener = self._mouse_module.Listener(
            on_move=self.on_move,
            on_click=self.on_click,
            on_scroll=self.on_scroll
        )
        self.listener.start()

    def stop(self):
        """Stop the trackpad listener."""
        if self.listener:
            self.listener.stop()
            self.listener = None

    def get_features(self) -> Optional[Dict]:
        """
        Get next trackpad feature dict.

        Returns:
            Dictionary with trackpad event data, or None if queue is empty
        """
        if not self.event_queue.empty():
            event = self.event_queue.get()
            return event.to_dict()
        return None

    def get_current_position(self) -> Dict[str, float]:
        """Get the current cursor position."""
        return self.current_position.copy()
