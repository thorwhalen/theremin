# streamkeys

**Keyboard event streams for Python**

A pure, framework-agnostic package for converting keyboard events into feature dictionaries. Zero dependencies on specific frameworks - can be used for gaming, shortcuts, automation, music applications, and more.

## Installation

```bash
pip install streamkeys
# Or for development
pip install -e .
```

## Dependencies

- `pynput` - Cross-platform keyboard monitoring (installed automatically)

## Quick Start

```python
from streamkeys import KeyboardFeatureStream
import time

# Create and start keyboard stream
stream = KeyboardFeatureStream()
stream.start()

# Process keyboard events
while True:
    event = stream.get_features()
    if event:
        print(f"Key '{event['key']}' was {event['action']}ed")
        print(f"Active keys: {event['active_keys']}")
    time.sleep(0.01)
```

## Musical Applications

```python
from streamkeys import keyboard_to_midi_features, keyboard_to_chord_features

# Single note playing
event = stream.get_features()
if event:
    midi_features = keyboard_to_midi_features(event)
    if midi_features:
        print(f"Play frequency: {midi_features['frequency']} Hz")

# Chord playing (multiple simultaneous keys)
if event:
    chord = keyboard_to_chord_features(event)
    if chord['frequencies']:
        print(f"Play chord with {chord['num_voices']} voices")
        print(f"Frequencies: {chord['frequencies']}")
```

## API Reference

### KeyboardFeatureStream

Main class for keyboard event capture.

**Methods:**
- `start()` - Start non-blocking keyboard listener
- `stop()` - Stop the keyboard listener
- `get_features()` - Get next keyboard event as dictionary
- `get_active_keys()` - Get set of currently pressed keys
- `is_key_active(key)` - Check if specific key is pressed

**Event Dictionary Format:**
```python
{
    'type': 'keyboard',
    'action': 'press',  # or 'release'
    'key': 'a',
    'active_keys': ['a', 'shift'],
    'timestamp': 1234567890.123
}
```

### Mapping Functions

**keyboard_to_midi_features(event)**
- Converts keyboard events to MIDI-like features
- Returns: `{'frequency', 'amplitude', 'note_on', 'midi_note', 'timestamp'}`

**keyboard_to_chord_features(event)**
- Maps all active keys to polyphonic features
- Returns: `{'frequencies', 'amplitude', 'midi_notes', 'num_voices', 'timestamp'}`

### Key Mappings

Default QWERTY to music note mappings:

- **Bottom row (Z-M)**: C3-B3 (lower octave)
- **Home row (A-L)**: C4-D5 (middle octave)
- **Top row (Q-P)**: C5-E6 (higher octave)

## Use Cases

- **Music applications**: Piano-like keyboard playing
- **Gaming**: Keyboard input processing
- **Shortcuts**: Global hotkey detection
- **Automation**: Keyboard-driven workflows
- **Accessibility**: Custom keyboard interfaces

## Platform Support

- **macOS**: Full support (requires Accessibility permissions)
- **Linux**: Full support (may require sudo)
- **Windows**: Full support

## License

MIT License
