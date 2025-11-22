# synthflow

**Dict-based synthesizer control for Python**

A framework-agnostic package for controlling audio synthesis via parameter dictionaries. Provides standardized wrappers for pyo, SignalFlow, and simple NumPy-based synthesis.

Useful for any Python audio synthesis project requiring parameter-based control.

## Installation

```bash
pip install synthflow
# Or for development
pip install -e .
```

## Dependencies

- `numpy` - Array operations (installed automatically)
- `pyo` - Optional, for high-quality synthesis (install separately)
- `scipy` - Optional, for better filtering (install separately)

## Quick Start

### Simple Synthesis (No External Dependencies)

```python
from synthflow import synthesize_simple
import soundfile as sf

# Generate audio
audio = synthesize_simple(
    frequency=440,
    amplitude=0.5,
    duration=1.0,
    waveform='sine'
)

# Save to file
sf.write('output.wav', audio, 44100)
```

### Pyo-Based Synthesis (Real-time)

```python
from synthflow import PyoSynthesizer
import time

# Create and start synthesizer
synth = PyoSynthesizer(waveform='sine')
synth.start()

# Update parameters in real-time
synth.update_parameters({
    'frequency': 440,
    'amplitude': 0.5
})

time.sleep(1)

# Change frequency
synth.update_parameters({'frequency': 880})

time.sleep(1)
synth.stop()
```

### With Effects

```python
from synthflow import SimpleSynthesizer, EffectsProcessor

synth = SimpleSynthesizer()
effects = EffectsProcessor()

# Generate audio
audio = synth.generate({
    'frequency': 440,
    'amplitude': 0.5,
    'duration': 1.0
})

# Apply effects
processed = effects.process(audio, {
    'filter_cutoff': 2000,
    'resonance': 0.7,
    'reverb': 0.3
})
```

## API Reference

### SimpleSynthesizer

**Constructor:**
```python
SimpleSynthesizer(sample_rate=44100)
```

**Methods:**
- `generate(params, duration=None)` - Generate audio from parameter dict

**Parameter Dict:**
```python
{
    'frequency': 440,        # Hz
    'amplitude': 0.5,        # 0-1
    'duration': 0.1,         # seconds
    'waveform': 'sine',      # 'sine', 'saw', 'square', 'triangle'
    'attack': 0.01,          # seconds
    'release': 0.05,         # seconds
    'envelope': True         # Apply envelope
}
```

### PyoSynthesizer

**Constructor:**
```python
PyoSynthesizer(
    sample_rate=44100,
    buffer_size=512,
    waveform='sine'
)
```

**Methods:**
- `start()` - Start pyo server
- `stop()` - Stop pyo server
- `update_parameters(params)` - Update synthesis parameters

**Parameter Dict:**
```python
{
    'frequency': 440,        # Hz
    'amplitude': 0.5,        # 0-1
    'waveform': 'sine'       # 'sine', 'saw', 'square', 'triangle'
}
```

### EffectsProcessor

**Constructor:**
```python
EffectsProcessor(sample_rate=44100)
```

**Methods:**
- `process(audio, params)` - Process audio with effects
- `apply_lowpass(audio, cutoff, resonance)` - Apply lowpass filter
- `apply_distortion(audio, amount)` - Apply distortion
- `apply_simple_reverb(audio, amount)` - Apply reverb

**Parameter Dict:**
```python
{
    'filter_cutoff': 2000,   # Hz
    'resonance': 0.5,        # 0-1
    'distortion': 0.3,       # 0-1
    'reverb': 0.2            # 0-1
}
```

## Use Cases

- **Music applications**: Parameter-based synthesis control
- **Audio generation**: Generate tones and sounds programmatically
- **Real-time synthesis**: Low-latency parameter updates
- **Testing**: Simple synthesis for prototyping
- **Education**: Learn synthesis concepts
- **Game audio**: Dynamic sound generation

## Waveforms

All synthesizers support these waveforms:
- **sine**: Pure tone
- **saw**: Sawtooth (bright, buzzy)
- **square**: Square wave (hollow, retro)
- **triangle**: Triangle wave (mellow)

## Performance

**SimpleSynthesizer:**
- CPU: Low (NumPy-based)
- Latency: N/A (offline generation)
- Quality: Good for basic synthesis

**PyoSynthesizer:**
- CPU: Very low (C-based pyo)
- Latency: <5ms (configurable buffer_size)
- Quality: Professional

## License

MIT License
