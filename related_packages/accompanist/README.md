# accompanist

**Music accompaniment tools for Python**

A framework-agnostic package for generating musical accompaniment. Includes chord progression generation, MIDI processing, and harmonic analysis.

Useful for music generation, improvisation support, educational applications, and live performance.

## Installation

```bash
pip install accompanist
# Or for development
pip install -e .
```

## Dependencies

Core (installed automatically):
- None (pure Python for basic functionality)

Optional:
- `mido` - For MIDI playback (install separately with `pip install mido`)

## Quick Start

### Chord Progressions

```python
from accompanist import ChordProgression, generate_progression

# Create a progression
prog = ChordProgression(['C', 'Am', 'F', 'G'])

# Get frequencies for each chord
for i, chord in enumerate(prog):
    freqs = prog.get_frequencies(i)
    print(f"{chord}: {freqs}")

# Generate common progressions
pop_prog = generate_progression(style='pop', length=4)
jazz_prog = generate_progression(style='jazz', length=4)
```

### MIDI Playback

```python
from accompanist import MIDIPlayer

# Create MIDI player
player = MIDIPlayer()
player.open()

# Play a chord
player.play_chord([60, 64, 67], duration=2.0)  # C major for 2 seconds

# Play a progression
prog = generate_progression('pop')
for i in range(len(prog)):
    notes = prog.get_midi_notes(i)
    player.play_chord(notes, duration=1.0)

player.close()
```

### Convert Chords to Frequencies

```python
from accompanist import chord_to_frequencies

# Get frequencies for a chord
c_major_freqs = chord_to_frequencies('C')
# Returns: [261.63, 329.63, 392.00]  # C, E, G in Hz

a_minor_freqs = chord_to_frequencies('Am')
# Returns frequencies for A minor chord
```

## API Reference

### ChordProgression

**Constructor:**
```python
ChordProgression(chords: List[str])
```

**Methods:**
- `get_chord(index)` - Get chord at index (wraps around)
- `get_midi_notes(index)` - Get MIDI notes for chord
- `get_frequencies(index)` - Get frequencies for chord
- `transpose(semitones)` - Transpose progression

**Supported Chords:**
- Major: C, F, G, etc.
- Minor: Am, Dm, Em
- Seventh: Cmaj7, Dm7, G7, C7, F7
- Diminished: Bdim

### generate_progression()

```python
generate_progression(
    style='pop',  # 'pop', 'jazz', 'blues', 'minor', 'fifties'
    key='C',
    length=4
) -> ChordProgression
```

**Available Progressions:**
- `pop`: C-G-Am-F (I-V-vi-IV)
- `jazz`: Cmaj7-Dm7-G7-Cmaj7 (ii-V-I)
- `blues`: C7-F7-C7-G7 (12-bar blues)
- `minor`: Am-F-C-G (vi-IV-I-V)
- `fifties`: C-Am-Dm-G (I-vi-ii-V)

### MIDIPlayer

**Constructor:**
```python
MIDIPlayer()
```

**Methods:**
- `open(port_name=None)` - Open MIDI port
- `close()` - Close MIDI port
- `play_note(note, velocity=80, duration=1.0, channel=0)` - Play single note
- `play_chord(notes, velocity=80, duration=1.0, channel=0)` - Play chord

### Utility Functions

**chord_to_frequencies(chord_name)**
- Convert chord name to frequencies

**chord_to_midi_notes(chord_name)**
- Convert chord name to MIDI notes

**note_to_frequency(midi_note)**
- Convert MIDI note number to frequency

## Use Cases

- **Live performance**: Real-time chord accompaniment
- **Music education**: Learn chord progressions
- **Improvisation**: Backing tracks for practice
- **Composition**: Generate harmonic ideas
- **Game audio**: Dynamic music generation
- **Interactive installations**: Sound-reactive accompaniment

## Example: Real-time Accompaniment

```python
from accompanist import generate_progression, chord_to_frequencies
from synthflow import SimpleSynthesizer
import time

# Generate progression
prog = generate_progression('pop', length=4)
synth = SimpleSynthesizer()

# Play accompaniment
while True:
    for i in range(len(prog)):
        # Get chord frequencies
        freqs = prog.get_frequencies(i)

        # Generate chord sound
        for freq in freqs:
            audio = synth.generate({
                'frequency': freq,
                'amplitude': 0.3,
                'duration': 1.0
            })
            # Play audio...

        time.sleep(1.0)
```

## License

MIT License
