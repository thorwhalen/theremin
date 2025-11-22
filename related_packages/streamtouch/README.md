# streamtouch

**Trackpad and mouse event streams for Python**

A pure, framework-agnostic package for converting trackpad/mouse events into feature dictionaries. Zero dependencies on specific frameworks - can be used for UI automation, gesture control, accessibility tools, audio applications, and more.

## Installation

```bash
pip install streamtouch
# Or for development
pip install -e .
```

## Dependencies

- `pynput` - Cross-platform mouse/trackpad monitoring (installed automatically)

## Quick Start

```python
from streamtouch import TrackpadFeatureStream
import time

# Create and start trackpad stream
stream = TrackpadFeatureStream()
stream.start()

# Process trackpad events
while True:
    event = stream.get_features()
    if event:
        if event['action'] == 'move':
            print(f"Cursor at: {event['position']}")
        elif event['action'] == 'scroll':
            print(f"Scrolled {event['direction']}: {event['magnitude']}")
    time.sleep(0.01)
```

## Audio Control Applications

```python
from streamtouch import trackpad_to_audio_knobs

# Map trackpad to audio parameters
event = stream.get_features()
if event:
    knobs = trackpad_to_audio_knobs(event, screen_width=1920, screen_height=1080)
    if knobs['knob_changed']:
        if 'cutoff_frequency' in knobs:
            print(f"Filter cutoff: {knobs['cutoff_frequency']} Hz")
            print(f"Resonance: {knobs['resonance']}")
        elif 'volume_delta' in knobs:
            print(f"Volume change: {knobs['volume_delta']}")
```

## Normalized Position Mapping

```python
from streamtouch import trackpad_to_2d_position

# Get normalized 0-1 coordinates
event = stream.get_features()
if event and event['action'] == 'move':
    pos = trackpad_to_2d_position(event, screen_width=1920, screen_height=1080)
    print(f"Normalized position: x={pos['x']:.2f}, y={pos['y']:.2f}")
```

## API Reference

### TrackpadFeatureStream

Main class for trackpad/mouse event capture.

**Methods:**
- `start()` - Start non-blocking listener
- `stop()` - Stop the listener
- `get_features()` - Get next trackpad event as dictionary
- `get_current_position()` - Get current cursor position

**Event Dictionary Formats:**

*Move Event:*
```python
{
    'type': 'trackpad',
    'action': 'move',
    'position': {'x': 960, 'y': 540},
    'timestamp': 1234567890.123
}
```

*Scroll Event:*
```python
{
    'type': 'trackpad',
    'action': 'scroll',
    'position': {'x': 960, 'y': 540},
    'delta': {'dx': 0, 'dy': 5},
    'direction': 'vertical',  # or 'horizontal'
    'magnitude': 5.0,
    'timestamp': 1234567890.123
}
```

*Click Event:*
```python
{
    'type': 'trackpad',
    'action': 'press',  # or 'release'
    'position': {'x': 960, 'y': 540},
    'button': 'Button.left',
    'timestamp': 1234567890.123
}
```

### Mapping Functions

**trackpad_to_2d_position(event, screen_width=1920, screen_height=1080, normalize=True)**
- Converts trackpad position to normalized or absolute 2D coordinates
- Returns: `{'x', 'y', 'timestamp'}`

**trackpad_to_audio_knobs(event, screen_width=1920, screen_height=1080)**
- Maps position to filter cutoff/resonance, scroll to volume/pan
- Returns: `{'cutoff_frequency', 'resonance', 'volume_delta', 'pan_delta', 'knob_changed'}`

**scroll_to_parameter_delta(event, param_name='volume', sensitivity=0.01, invert=False)**
- Converts scroll events to parameter changes
- Returns: `{'{param_name}_delta', 'timestamp'}`

## Use Cases

- **Audio applications**: Real-time filter/effect control
- **UI automation**: Mouse movement tracking
- **Gesture control**: Scroll-based parameter adjustment
- **Accessibility**: Custom trackpad interfaces
- **Gaming**: Mouse input processing
- **Data visualization**: Interactive parameter control

## Platform Support

- **macOS**: Full support (requires Accessibility permissions)
  - Note: Advanced gestures (pinch, rotate) require PyObjC and are not supported
  - Scroll events provide a practical alternative for directional input
- **Linux**: Full support (may require permissions)
- **Windows**: Full support

## Limitations

This package focuses on basic mouse/trackpad events (move, click, scroll) that work reliably across platforms. Advanced macOS multi-touch gestures (pinch, rotate, multi-finger swipes) are not supported as they require:

- PyObjC framework integration
- Cocoa NSResponder setup
- Full application event loop

For most use cases, scroll events provide sufficient directional input and work consistently across all platforms.

## License

MIT License
