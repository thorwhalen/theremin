"""Audio and synthesizer functions for theremin."""

import numpy as np
from typing import Dict, Union
from collections.abc import Callable
from functools import lru_cache, partial

from hum import Synth
from hum.pyo_util import add_default_dials, add_default_settings

from pyo import *

# See DFLT_SYNTH_FUNC_NAME and DFLT_KNOBS definitions at the end of this module


# -------------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------------


# TODO: Not working and unfinished
def lr_version(func=None, *, params_to_double=None):
    """
    Decorator that transforms a function into a left-right version by doubling specified parameters.

    Args:
        func: The function to decorate (when used as @lr_version)
        params_to_double: List of parameter names to double (create l_ and r_ versions)

    Returns:
        Decorated function with l_ and r_ versions of the specified parameters
    """
    if params_to_double is None:
        params_to_double = ['freq', 'volume']

    def decorator(func):
        from functools import partial, wraps

        @wraps(func)
        def lr_wrapper(**kwargs):
            # Extract parameters for left and right channel
            l_params = {
                param: kwargs.get(f'l_{param}', 440 if param == 'freq' else 0.0)
                for param in params_to_double
            }
            r_params = {
                param: kwargs.get(f'r_{param}', 440 if param == 'freq' else 0.0)
                for param in params_to_double
            }

            # Get all other parameters
            extra_kwargs = {
                k: v
                for k, v in kwargs.items()
                if not any(k == f'l_{p}' or k == f'r_{p}' for p in params_to_double)
            }

            # Create synth functions
            l_synth = partial(func, **l_params, **extra_kwargs)
            r_synth = partial(func, **r_params, **extra_kwargs)

            # Get values needed for _two_voice_synth_func
            l_freq = l_params.get('freq', 440)
            l_volume = l_params.get('volume', 0.0)
            r_freq = r_params.get('freq', 440)
            r_volume = r_params.get('volume', 0.0)

            return _two_voice_synth_func(
                l_freq, l_volume, r_freq, r_volume, l_synth=l_synth, r_synth=r_synth
            )

        # Add appropriate dials
        dial_str = ' '.join([f'l_{p} r_{p}' for p in params_to_double])
        if hasattr(func, '_default_dials'):
            for dial in func._default_dials.split():
                if dial not in params_to_double:
                    dial_str += f' {dial}'

        return add_default_dials(dial_str)(lr_wrapper)

    # Handle both @lr_version and @lr_version(params_to_double=[...]) usage
    if func is not None:
        return decorator(func)
    return decorator


# -------------------------------------------------------------------------------
# Synthesizer functions
# -------------------------------------------------------------------------------

DFLT_OSC = Sine


@add_default_dials('freq volume')
def theremin_synth(
    freq=440,
    volume=0.5,
    attack=0.01,
    release=0.1,
    vibrato_rate=5,
    vibrato_depth=5,
    *,
    waveform='sine',
):
    """
    Emulates a classic theremin sound.

    Parameters:
    - freq (float): Base frequency in Hz.
    - volume (float): Output volume (0 to 1).
    - waveform (str): Waveform type ('sine', 'triangle', 'square').
    - attack (float): Attack time in seconds.
    - release (float): Release time in seconds.
    - vibrato_rate (float): Vibrato frequency in Hz.
    - vibrato_depth (float): Vibrato depth in Hz.

    Returns:
    - PyoObject: The resulting audio signal.
    """
    # Select waveform
    waveforms = {
        'sine': Sine,
        'triangle': lambda freq, mul: LFO(freq=freq, type=3, mul=mul),
        'square': lambda freq, mul: LFO(freq=freq, type=1, mul=mul),
    }
    wave_class = waveforms.get(waveform, Sine)

    # Vibrato modulation
    vibrato = Sine(freq=vibrato_rate, mul=vibrato_depth)

    # Envelope
    env = Adsr(
        attack=attack, decay=0.1, sustain=0.8, release=release, dur=0, mul=volume
    )
    env.play()
    # Oscillator with vibrato
    osc = wave_class(freq=freq + vibrato, mul=env)

    return osc


def sine_synth(freq=440, volume=0):
    """A basic sine wave synthesizer."""
    return DFLT_OSC(freq=freq, mul=volume)


def fm_synth(freq=440, volume=0, carrier_ratio=1.0, mod_index=2.0, mod_freq_ratio=2.0):
    """Frequency modulation synthesizer."""
    mod = DFLT_OSC(freq=freq * mod_freq_ratio, mul=freq * mod_index)
    car = DFLT_OSC(freq=freq * carrier_ratio + mod, mul=volume)
    return car


def supersaw_synth(freq=440, volume=0, detune=0.01, n_voices=7):
    """Supersaw synthesizer with multiple detuned sawtooth waves."""
    voices = [
        LFO(
            freq=freq * (1 + detune * (i - n_voices // 2)),
            type=5,
            mul=volume / n_voices,
        )
        for i in range(n_voices)
    ]
    return sum(voices)


def square_synth(freq=440, volume=0):
    """Simple square wave synthesizer."""
    return LFO(freq=freq, type=2, mul=volume)


def noise_synth(freq=440, volume=0, noise_level=0.2):
    """Sine wave with noise component."""
    sine = DFLT_OSC(freq=freq, mul=volume * (1 - noise_level))
    noise = Noise(mul=volume * noise_level)
    return sine + noise


def ringmod_synth(freq=440, volume=0, mod_freq_ratio=1.5):
    """Ring modulation synthesizer."""
    mod = DFLT_OSC(freq=freq * mod_freq_ratio)
    carrier = DFLT_OSC(freq=freq)
    return (carrier * mod) * volume


def chorused_sine_synth(freq=440, volume=0, depth=5, speed=0.3):
    """Chorused sine wave with LFO modulation."""
    lfo = Sine(freq=speed, mul=depth)
    mod_freq = freq + lfo
    return Sine(freq=mod_freq, mul=volume)


# -------------------------------------------------------------------------------
# Natural sounding synth (instrument-like timbres)
# -------------------------------------------------------------------------------

from tonal import scale_midi_notes

DFLT_SCALE = "A penta"


def snap_to_scale(freq, scale=DFLT_SCALE):
    """Snap frequency to the nearest note in the C major scale."""
    # Frequencies of C major scale over several octaves (C, D, E, F, G, A, B)
    # semitones_in_scale = np.array([0, 2, 4, 5, 7, 9, 11])
    semitones_in_scale = scale_midi_notes(scale, midi_range=(0, 12))
    scale_freqs = []
    for octave in range(0, 9):
        base_midi = 12 * octave
        for st in semitones_in_scale:
            midi_note = base_midi + st
            hz = 440.0 * 2 ** ((midi_note - 69) / 12)
            scale_freqs.append(hz)
    scale_freqs = np.array(scale_freqs)
    closest = scale_freqs[np.argmin(np.abs(scale_freqs - freq))]
    return float(closest)


# --- New: export frequencies of a given scale in a range ---
from typing import List


def scale_frequencies_for_range(
    scale: str, min_freq: float, max_freq: float
) -> list[float]:
    """Return all frequencies (Hz) for the given scale within [min_freq, max_freq]."""
    semitones_in_scale = scale_midi_notes(scale, midi_range=(0, 12))
    freqs: list[float] = []
    # Iterate through a wide octave range; filter by min/max
    for octave in range(0, 12):
        base_midi = 12 * octave
        for st in semitones_in_scale:
            midi_note = base_midi + st
            hz = 440.0 * 2 ** ((midi_note - 69) / 12)
            if min_freq <= hz <= max_freq:
                freqs.append(hz)
    # Deduplicate and sort
    return sorted(set(freqs))


# --- New: scale resolver for CLI / runtime configuration ---
from typing import Optional


def resolve_scale_to_freq_trans(scale: str | None):
    """Resolve a scale string to a frequency transform function or None.

    - If scale is None: return a sentinel indicating to use the default behavior of
      the target knobs function (i.e., don't override its freq_trans).
    - If scale is in {"none","null","off","false","no","0",""}: return None to disable snapping.
    - Otherwise: return partial(snap_to_scale, scale=<scale>).
    """
    if scale is None:
        # Do not override; let the callee use its default (usually snap_to_scale with DFLT_SCALE)
        return "__USE_DEFAULT__"
    s = str(scale).strip().lower()
    if s in {"none", "null", "off", "false", "no", "0", ""}:
        return None
    # Any other string: try to create a snapper using tonal scale name
    try:
        return partial(snap_to_scale, scale=scale)
    except Exception:
        # Fallback: disable if unknown
        return None


@add_default_dials('freq volume vibrato_rate vibrato_depth reverb_mix ramp_time')
def natural_sounding_synth(
    freq=440,
    volume=0.5,
    *,
    instrument='violin',
    vibrato_rate=5,
    vibrato_depth=5,
    reverb_mix=0.3,
    ramp_time=0.9,
):
    """
    Synthesizer with more harmonically rich, instrument-like timbres.

    Parameters:
    - freq (float): Base frequency in Hz.
    - volume (float): Output volume (0 to 1).
    - instrument (str): Instrument type ('violin', 'organ', 'flute').
    - vibrato_rate (float): Vibrato frequency in Hz.
    - vibrato_depth (float): Vibrato depth in Hz.
    - reverb_mix (float): Reverb mix (0 to 1).
    - ramp_time (float): Portamento time (smooth transition between frequencies).

    Returns:
    - PyoObject: The resulting audio signal.
    """
    instrument_oscillators = {
        'violin': lambda freq, mul: Blit(freq=freq, harms=10, mul=mul),
        'organ': lambda freq, mul: SuperSaw(freq=freq, detune=0.1, bal=0.4, mul=mul),
        'flute': lambda freq, mul: LFO(freq=freq, type=3, mul=mul * 0.5),
    }
    print(f"{instrument=}")
    osc_factory = instrument_oscillators.get(
        instrument, instrument_oscillators['violin']
    )

    # Vibrato modulation
    vibrato = Sine(freq=vibrato_rate, mul=vibrato_depth)

    # Frequency smoothing (ramp)
    smooth_freq = Port(freq + vibrato, risetime=ramp_time, falltime=ramp_time)

    # Envelope
    env = Adsr(attack=0.05, decay=0.1, sustain=0.8, release=0.4, dur=0, mul=volume)
    env.play()

    # Oscillator
    osc = osc_factory(freq=smooth_freq, mul=env)

    # Filter and reverb
    filtered = ButLP(osc, freq=3000)
    rev = Freeverb(filtered, size=0.8, damp=0.5, bal=reverb_mix)

    return rev


# TODO: Make it work
# natural_sounding_synth_lr = lr_version(
#     natural_sounding_synth, params_to_double=['freq', 'volume']
# )
# natural_sounding_synth_lr = add_default_dials(
#     'l_freq l_volume r_freq r_volume vibrato_rate vibrato_depth reverb_mix ramp_time'
# )(natural_sounding_synth_lr)


@add_default_dials(
    'l_freq l_volume r_freq r_volume vibrato_rate vibrato_depth reverb_mix ramp_time'
)
def natural_sounding_synth_lr(
    l_freq=440,
    l_volume=0.0,
    r_freq=440,
    r_volume=0.0,
    *,
    instrument='violin',
    vibrato_rate=5,
    vibrato_depth=5,
    reverb_mix=0.3,
    ramp_time=0.1,
):
    extra_kwargs = {
        k: v
        for k, v in locals().items()
        if k not in ['l_freq', 'l_volume', 'r_freq', 'r_volume']
    }
    l_synth = partial(
        natural_sounding_synth, freq=l_freq, volume=l_volume, **extra_kwargs
    )
    r_synth = partial(
        natural_sounding_synth, freq=r_freq, volume=r_volume, **extra_kwargs
    )
    return _two_voice_synth_func(
        l_freq, l_volume, r_freq, r_volume, l_synth=l_synth, r_synth=r_synth
    )


@add_default_settings('freq_src')
def phase_distortion_synth(freq=440, volume=0, distortion=0.5, *, freq_src=DFLT_OSC):
    """Phase distortion synthesizer."""
    phasor = Phasor(freq=freq)
    distorted = phasor + (freq_src(freq=freq * 2, mul=distortion) * phasor)
    return distorted * volume


# @add_default_dials('carrier_freq_base fm_transpo index_amount beat_density reverb_mix noise_level env_attack env_release feedback pan_pos distortion')
@add_default_dials('carrier_freq_base index_amount carrier_freq_base distortion')
def rhythmic_fm_synth(
    carrier_freq_base=110,
    fm_transpo=1.0,
    index_amount=5.0,
    beat_density=0.1125,
    reverb_mix=0.3,
    noise_level=0.1,
    env_attack=0.01,
    env_release=0.5,
    feedback=0.1,
    pan_pos=0.5,
    distortion=0.0,
):
    env_table = ExpTable([(0, 0), (50, 1), (8191, 0)], exp=12)

    beat = Beat(time=beat_density, taps=16, w1=80, w2=60, w3=50).play()
    amp_env = TrigEnv(beat, table=env_table, dur=beat['dur'], mul=0.2)

    car_freq = Sig(carrier_freq_base * fm_transpo)
    mod_freq = car_freq * 2
    index = Sig(index_amount)

    modulator = Sine(freq=mod_freq, mul=index)
    carrier = Sine(freq=car_freq + modulator, mul=amp_env)

    distorted = (
        Disto(carrier, drive=distortion, slope=0.8) if distortion > 0 else carrier
    )
    rev = Freeverb(distorted, size=0.9, bal=reverb_mix)
    panned = Pan(rev, outs=2, pan=pan_pos)

    return panned


# Defaults
DFLT_L_SYNTH = sine_synth
DFLT_R_SYNTH = theremin_synth
DFLT_MIN_FREQ = 220
DFLT_MAX_FREQ = DFLT_MIN_FREQ * 8


# -------------------------------------------------------------------------------
# Two-voice synth setup
# -------------------------------------------------------------------------------


def _two_voice_synth_func(
    l_freq=440,
    l_volume=0.0,
    r_freq=440,
    r_volume=0.0,
    *,
    l_synth=DFLT_L_SYNTH,
    r_synth=DFLT_R_SYNTH,
):
    """
    Internal two-voice synth function with all parameters.
    Not meant to be used directly.
    """
    sound1 = l_synth(freq=l_freq, volume=l_volume)
    sound2 = r_synth(freq=r_freq, volume=r_volume)
    return sound1 + sound2


from theremin.util import obfuscate_args

# Create a simplified two-voice synth function that only exposes the necessary parameters
two_voice_synth_func = obfuscate_args(
    _two_voice_synth_func, keep_args=['l_freq', 'l_volume', 'r_freq', 'r_volume']
)


# -------------------------------------------------------------------------------
# Knob (control parameter) functions
# -------------------------------------------------------------------------------

from typing import Tuple

Range = tuple[float, float]


# TODO: Make video_features to knobs tools to increase reusability and UX.
#     Namely, easy to specify linear (and more) from hand feature ranges to knob ranges.

# video feature ranges -------------------------------
from theremin.video_features import video_feature_ranges

# audio feature ranges -------------------------------
audio_feature_ranges = {
    # Basic Theremin Parameters
    "freq": (DFLT_MIN_FREQ, DFLT_MAX_FREQ),  # Default: (220, 1760) Hz
    "volume": (0.0, 1.0),  # Standard audio volume range
    # Waveform-based synthesis parameters
    "waveform": ["sine", "triangle", "square"],  # Categorical, not numeric
    # Envelope parameters (in seconds)
    "attack": (0.0, 1.0),  # Envelope attack time in seconds
    "release": (0.0, 1.0),  # Envelope release time in seconds
    # Vibrato parameters
    "vibrato_rate": (1.0, 20.0),  # Vibrato frequency in Hz (typical musical range)
    "vibrato_depth": (
        0.0,
        20.0,
    ),  # Vibrato depth in Hz (typically 0-10Hz for expressiveness)
    # Spatial effects
    "reverb_mix": (0.0, 1.0),  # Reverb dry/wet mix ratio
    # Portamento/Glide
    "ramp_time": (0.0, 1.0),  # Frequency transition time in seconds
    # FM Synthesis parameters
    "carrier_ratio": (0.1, 5.0),  # Carrier frequency multiplier
    "mod_index": (0.0, 10.0),  # Modulation index for FM synthesis
    "mod_freq_ratio": (0.1, 5.0),  # Modulator frequency multiplier
    # Supersaw parameters
    "detune": (0.0, 0.1),  # Detuning amount for supersaw voices
    "n_voices": (3, 11),  # Number of voices (usually odd numbers)
    # Noise synthesis
    "noise_level": (0.0, 1.0),  # Mix ratio of noise to oscillator
    # Ring modulation
    "mod_freq_ratio": (0.5, 3.0),  # Ring modulator frequency ratio
    # Chorus parameters
    "depth": (1.0, 10.0),  # LFO modulation depth in Hz
    "speed": (0.1, 1.0),  # LFO frequency in Hz
    # Phase distortion
    "distortion": (0.0, 1.0),  # Distortion amount
    # Instrument-specific parameters
    "instrument": ["violin", "organ", "flute"],  # Categorical
    # High sines specific parameters
    "base_freq": (2000.0, 6000.0),  # High frequency range for ambient textures
    "mod_freq": (0.1, 1.0),  # Low frequency modulation for ambient sounds
    "mod_mul": (0.0, 0.5),  # Modulation amplitude
    # Two-voice parameters (l_ and r_ prefixes)
    "l_freq": (DFLT_MIN_FREQ, DFLT_MAX_FREQ),  # Left hand frequency
    "l_volume": (0.0, 1.0),  # Left hand volume
    "r_freq": (DFLT_MIN_FREQ, DFLT_MAX_FREQ),  # Right hand frequency
    "r_volume": (0.0, 1.0),  # Right hand volume
}


from typing import Tuple


def identity(x):
    """Identity function."""
    return x


class RangeMapper:
    """
    A callable class that maps values from one range to another.
    Precomputes scaling factors for better performance.

    >>> mapper = RangeMapper((0, 1), (100, 200))
    >>> mapper(0.5)
    150.0
    >>> mapper(-0.1)  # Below range
    100.0
    >>> mapper(1.5)   # Above range
    200.0
    """

    def __init__(
        self,
        value_range: tuple[float, float],
        target_range: tuple[float, float],
        *,
        ingress=identity,
        egress=identity,
    ):
        """
        Initialize the range mapper with source and target ranges.

        Args:
            value_range: The range of the input value (min, max)
            target_range: The range to map to (min, max)
        """
        # Normalize ranges to floats to ensure consistent float outputs
        self.value_min, self.value_max = float(value_range[0]), float(value_range[1])
        self.target_min, self.target_max = float(target_range[0]), float(
            target_range[1]
        )

        # Precompute frequently used values for performance
        self._value_span = float(self.value_max - self.value_min)
        self._target_span = float(self.target_max - self.target_min)
        self._scale_factor = float(self._target_span / self._value_span)
        self.ingress = ingress
        self.egress = egress

    def __call__(self, value: float) -> float:
        """
        Map a value from the source range to the target range.

        Args:
            value: The value to map

        Returns:
            Mapped value in the target range
        """
        value = self.ingress(value)
        if value <= self.value_min:
            output = self.target_min
        elif value >= self.value_max:
            output = self.target_max
        else:
            output = self.target_min + (value - self.value_min) * self._scale_factor

        return self.egress(output)


@lru_cache(maxsize=128)
def range_mapper(
    video_feature: str, audio_feature: str, *, ingress=identity, egress=identity
) -> RangeMapper:
    """
    Create a RangeMapper instance for the given video and audio features.
    """
    # get and validate ranges
    if video_feature in video_feature_ranges:
        video_range = video_feature_ranges[video_feature]
        assert len(video_range) == 2, f"Invalid range for {video_feature}"
    if audio_feature.startswith("l_") or audio_feature.startswith("r_"):
        audio_feature = audio_feature[2:]
    if audio_feature in audio_feature_ranges:
        audio_range = audio_feature_ranges[audio_feature]
        assert len(audio_range) == 2, f"Invalid range for {audio_feature}"
    # return a RangeMapper instance
    return RangeMapper(video_range, audio_range, ingress=ingress, egress=egress)


def _calculate_freq_and_vol_from_wrist(wrist, min_freq, max_freq):
    """
    Calculate frequency and volume based on wrist position.

    Args:
        wrist: Position of the wrist (tuple or array with x, y coordinates)
        min_freq: Minimum frequency value
        max_freq: Maximum frequency value

    Returns:
        tuple: (frequency, volume)
    """
    # freq = float(min_freq + wrist[0] * (max_freq - min_freq))
    # vol = float(np.clip(1 - wrist[1], 0, 1))
    # return freq, vol
    return (
        _calculate_freq_from_wrist(wrist, min_freq, max_freq),
        _calculate_vol_from_wrist(wrist),
    )


def _calculate_freq_from_wrist(wrist, min_freq, max_freq):
    return float(min_freq + wrist[0] * (max_freq - min_freq))


def _calculate_vol_from_wrist(wrist):
    return float(np.clip(1 - wrist, 0, 1))


def two_hand_freq_and_volume_knobs(
    video_features,
    *,
    freq_trans: Callable | None = snap_to_scale,
    min_freq: float = DFLT_MIN_FREQ,
    max_freq: float = DFLT_MAX_FREQ,
    audio_features: list = (
        'l_freq',
        'l_volume',
        'r_freq',
        'r_volume',
        'vibrato_rate',
        'vibrato_depth',
        'reverb_mix',
    ),
) -> dict[str, float]:
    """
    Maps hand positions to frequency and volume for both hands.

    Args:
        video_features (dict): Extracted hand feature dictionary.
        min_freq (float): Minimum frequency for pitch control.
        max_freq (float): Maximum frequency for pitch control.

    Returns:
        Dict[str, float]: Dictionary with 'l_freq', 'l_volume', 'r_freq', 'r_volume' keys.
    """
    knobs = {}

    # Set default silence
    mid_freq = (min_freq + max_freq) / 2
    knobs['l_freq'] = mid_freq
    knobs['l_volume'] = 0.0
    knobs['r_freq'] = mid_freq
    knobs['r_volume'] = 0.0

    if video_features:
        # get range mappers
        freq_mapper = range_mapper('wrist_position_x', 'freq')
        volume_mapper = range_mapper(
            'wrist_position_y', 'volume', ingress=lambda x: 1 - x
        )
        openess_vibrato_rate_mapper = range_mapper('r_openness', 'vibrato_rate')
        tid_vibrato_depth_mapper = range_mapper(
            'r_thumb_index_distance', 'vibrato_depth'
        )

        if 'l_wrist_position' in video_features:
            wrist = video_features['l_wrist_position']
            knobs['l_freq'] = freq_mapper(wrist[0])
            knobs['l_volume'] = volume_mapper(wrist[1])
        if 'r_wrist_position' in video_features:
            wrist = video_features['r_wrist_position']
            knobs['r_freq'] = freq_mapper(wrist[0])
            knobs['r_volume'] = volume_mapper(wrist[1])

        if 'r_openess' in video_features:
            knobs['vibrato_rate'] = np.clip(
                video_features['r_openness'] * audio_feature_ranges['vibrato_rate'][1],
                audio_feature_ranges['vibrato_rate'][0],
                audio_feature_ranges['vibrato_rate'][1],
            )

        if 'r_openness' in video_features and 'vibrato_rate' in audio_features:
            knobs['vibrato_rate'] = openess_vibrato_rate_mapper(
                video_features['r_openness']
            )
        if 'r_thumb_index_distance' in video_features:
            knobs['vibrato_depth'] = tid_vibrato_depth_mapper(
                video_features['r_thumb_index_distance']
            )

    # No frequency transformation needed here; this knobs set doesn't expose l_/r_ freq

    # Filter to include only parameters the synth function can use
    synth_params = {
        'l_freq',
        'l_volume',
        'r_freq',
        'r_volume',
        'vibrato_rate',
        'vibrato_depth',
        'reverb_mix',
    }
    return {k: float(v) for k, v in knobs.items() if k in synth_params}


def two_voice_knobs(
    video_features,
    *,
    freq_trans: Callable | None = snap_to_scale,
    min_freq: float = DFLT_MIN_FREQ,
    max_freq: float = DFLT_MAX_FREQ,
) -> dict[str, float]:
    """
    Maps hand positions to frequency and volume for both hands.
    Specifically designed for two_voice_synth_func which only accepts
    l_freq, l_volume, r_freq, r_volume parameters.

    Args:
        video_features (dict): Extracted hand feature dictionary.
        min_freq (float): Minimum frequency for pitch control.
        max_freq (float): Maximum frequency for pitch control.
        freq_trans: Optional frequency transformation function.

    Returns:
        Dict[str, float]: Dictionary with 'l_freq', 'l_volume', 'r_freq', 'r_volume' keys.
    """
    knobs = {}

    # Set default silence
    mid_freq = (min_freq + max_freq) / 2
    knobs['l_freq'] = mid_freq
    knobs['l_volume'] = 0.0
    knobs['r_freq'] = mid_freq
    knobs['r_volume'] = 0.0

    if video_features:
        # get range mappers
        freq_mapper = range_mapper('wrist_position_x', 'freq')
        volume_mapper = range_mapper(
            'wrist_position_y', 'volume', ingress=lambda x: 1 - x
        )

        if 'l_wrist_position' in video_features:
            wrist = video_features['l_wrist_position']
            knobs['l_freq'] = freq_mapper(wrist[0])
            knobs['l_volume'] = volume_mapper(wrist[1])
        if 'r_wrist_position' in video_features:
            wrist = video_features['r_wrist_position']
            knobs['r_freq'] = freq_mapper(wrist[0])
            knobs['r_volume'] = volume_mapper(wrist[1])

    # No frequency transformation needed here; this knobs set doesn't expose l_/r_ freq

    # Only return parameters that two_voice_synth_func can handle
    synth_params = {'l_freq', 'l_volume', 'r_freq', 'r_volume'}
    return {k: float(v) for k, v in knobs.items() if k in synth_params}


def simple_two_hands_knobs(
    video_features,
    *,
    freq_trans: Callable | None = snap_to_scale,
    min_freq: float = DFLT_MIN_FREQ,
    max_freq: float = DFLT_MAX_FREQ,
) -> dict[str, float]:
    """
    Maps hand positions to simple freq and volume parameters.
    For simple synths that only need freq and volume.

    Args:
        video_features (dict): Extracted hand feature dictionary.
        min_freq (float): Minimum frequency for pitch control.
        max_freq (float): Maximum frequency for pitch control.
        freq_trans: Optional frequency transformation function.

    Returns:
        Dict[str, float]: Dictionary with 'freq', 'volume' keys.
    """
    knobs = {
        'freq': (min_freq + max_freq) / 2,  # Default middle frequency
        'volume': 0.0,  # Default silence
    }

    if video_features:
        # get range mappers
        freq_mapper = range_mapper('wrist_position_x', 'freq')
        volume_mapper = range_mapper(
            'wrist_position_y', 'volume', ingress=lambda x: 1 - x
        )

        # Use left hand for main frequency and volume (fallback to right if no left)
        primary_hand = None
        if 'l_wrist_position' in video_features:
            primary_hand = video_features['l_wrist_position']
        elif 'r_wrist_position' in video_features:
            primary_hand = video_features['r_wrist_position']

        if primary_hand:
            knobs['freq'] = freq_mapper(primary_hand[0])
            knobs['volume'] = volume_mapper(primary_hand[1])

    # Apply frequency transformation if provided
    if freq_trans:
        knobs['freq'] = freq_trans(knobs['freq'])

    return {k: float(v) for k, v in knobs.items()}


def ringmod_two_hands_knobs(
    video_features,
    *,
    freq_trans: Callable | None = snap_to_scale,
    min_freq: float = DFLT_MIN_FREQ,
    max_freq: float = DFLT_MAX_FREQ,
) -> dict[str, float]:
    """
    Maps hand positions to parameters for ringmod_synth.
    Combines both hands into a single frequency and volume, plus modulation ratio.

    Args:
        video_features (dict): Extracted hand feature dictionary.
        min_freq (float): Minimum frequency for pitch control.
        max_freq (float): Maximum frequency for pitch control.
        freq_trans: Optional frequency transformation function.

    Returns:
        Dict[str, float]: Dictionary with 'freq', 'volume', 'mod_freq_ratio' keys.
    """
    knobs = {
        'freq': (min_freq + max_freq) / 2,  # Default middle frequency
        'volume': 0.0,  # Default silence
        'mod_freq_ratio': 1.5,  # Default modulation ratio
    }

    if video_features:
        # get range mappers
        freq_mapper = range_mapper('wrist_position_x', 'freq')
        volume_mapper = range_mapper(
            'wrist_position_y', 'volume', ingress=lambda x: 1 - x
        )

        # Use left hand for main frequency and volume (fallback to right if no left)
        primary_hand = None
        if 'l_wrist_position' in video_features:
            primary_hand = video_features['l_wrist_position']
        elif 'r_wrist_position' in video_features:
            primary_hand = video_features['r_wrist_position']

        if primary_hand:
            knobs['freq'] = freq_mapper(primary_hand[0])
            knobs['volume'] = volume_mapper(primary_hand[1])

        # Use right hand openness for modulation ratio (if available)
        if 'r_openness' in video_features:
            # Map openness (0-1) to modulation ratio range (0.5-5.0)
            knobs['mod_freq_ratio'] = 0.5 + video_features['r_openness'] * 4.5

    # Apply frequency transformation if provided
    if freq_trans:
        knobs['freq'] = freq_trans(knobs['freq'])

    return {k: float(v) for k, v in knobs.items()}


def supersaw_two_hands_knobs(
    video_features,
    *,
    freq_trans: Callable | None = snap_to_scale,
    min_freq: float = DFLT_MIN_FREQ,
    max_freq: float = DFLT_MAX_FREQ,
) -> dict[str, float]:
    """
    Maps hand positions to parameters for supersaw_synth.
    Combines both hands into a single frequency and volume, plus detune and n_voices.

    Args:
        video_features (dict): Extracted hand feature dictionary.
        min_freq (float): Minimum frequency for pitch control.
        max_freq (float): Maximum frequency for pitch control.
        freq_trans: Optional frequency transformation function.

    Returns:
        Dict[str, float]: Dictionary with 'freq', 'volume', 'detune', 'n_voices' keys.
    """
    knobs = {
        'freq': (min_freq + max_freq) / 2,  # Default middle frequency
        'volume': 0.0,  # Default silence
        'detune': 0.01,  # Default detune amount
        'n_voices': 7,  # Default number of voices
    }

    if video_features:
        # get range mappers
        freq_mapper = range_mapper('wrist_position_x', 'freq')
        volume_mapper = range_mapper(
            'wrist_position_y', 'volume', ingress=lambda x: 1 - x
        )

        # Use left hand for main frequency and volume (fallback to right if no left)
        primary_hand = None
        if 'l_wrist_position' in video_features:
            primary_hand = video_features['l_wrist_position']
        elif 'r_wrist_position' in video_features:
            primary_hand = video_features['r_wrist_position']

        if primary_hand:
            knobs['freq'] = freq_mapper(primary_hand[0])
            knobs['volume'] = volume_mapper(primary_hand[1])

        # Use right hand openness for detune amount (if available)
        if 'r_openness' in video_features:
            # Map openness (0-1) to detune range (0.001-0.1)
            knobs['detune'] = 0.001 + video_features['r_openness'] * 0.099

        # Use right thumb-index distance for number of voices (if available)
        if 'r_thumb_index_distance' in video_features:
            # Map distance to n_voices range (3-15)
            normalized_distance = np.clip(
                video_features['r_thumb_index_distance'], 0, 1
            )
            knobs['n_voices'] = int(3 + normalized_distance * 12)

    # Apply frequency transformation if provided
    if freq_trans:
        knobs['freq'] = freq_trans(knobs['freq'])

    return {k: float(v) for k, v in knobs.items()}


def rhythmic_fm_synth_knobs(
    video_features,
    freq_trans: Callable | None = snap_to_scale,
) -> dict[str, float]:
    """
    Maps video features to the parameters of rhythmic_fm_synth.

    Args:
        video_features (dict): Hand tracking data from video.

    Returns:
        Dict[str, float]: Audio knob values for rhythmic_fm_synth.
    """
    knobs = {
        'carrier_freq_base': 220.0,
        'fm_transpo': 1.0,
        'index_amount': 5.0,
        'beat_density': 0.1125,
        'reverb_mix': 0.3,
        'noise_level': 0.1,
        'env_attack': 0.01,
        'env_release': 0.5,
        'feedback': 0.1,
        'pan_pos': 0.5,
        'distortion': 0.0,
    }

    if not video_features:
        return knobs

    # Map right wrist X to carrier frequency base
    if 'r_wrist_position' in video_features:
        x = video_features['r_wrist_position'][0]
        knobs['carrier_freq_base'] = 220 + x * (880 - 220)  # Range 220–880 Hz

    # Map left wrist Y to index amount
    if 'l_wrist_position' in video_features:
        y = video_features['l_wrist_position'][1]
        knobs['index_amount'] = 10 * (1 - y)  # Inverse: higher hand = more index

    # Right openness controls feedback
    if 'r_openness' in video_features:
        knobs['feedback'] = np.clip(video_features['r_openness'], 0.0, 1.0)

    # Thumb-index distance controls distortion
    if 'r_thumb_index_distance' in video_features:
        dist = min(video_features['r_thumb_index_distance'], 0.2) / 0.2
        knobs['distortion'] = dist

    # Left openness for reverb mix
    if 'l_openness' in video_features:
        openness = np.clip(video_features['l_openness'], 0.0, 1.0)
        knobs['reverb_mix'] = float(openness)

    return {k: float(v) for k, v in knobs.items()}


# from hum.util import scale_snapper

# snap_to_scale = scale_snapper(scale=(0, 2, 4, 5, 7, 9, 11))


def theremin_knobs(
    video_features,
    *,
    min_freq: float = DFLT_MIN_FREQ,
    max_freq: float = DFLT_MAX_FREQ,
    freq_trans: Callable | None = snap_to_scale,
) -> dict[str, float]:
    """
    Maps hand positions to frequency (pitch) and volume (amplitude),
    mimicking a classic theremin control scheme.

    When both hands are detected:
    - Right hand X position controls frequency (pitch)
    - Left hand Y position controls volume (amplitude)

    When only one hand is detected:
    - X position controls frequency (pitch)
    - Y position controls volume (amplitude)

    Args:
        video_features (dict): Extracted hand feature dictionary.
        min_freq (float): Minimum frequency for pitch control.
        max_freq (float): Maximum frequency for pitch control.

    Returns:
        Dict[str, float]: Dictionary with 'freq', 'volume' keys.
    """
    X, Y = 0, 1
    knobs = {}

    if not video_features:
        return knobs
    elif 'r_wrist_position' in video_features and 'l_wrist_position' in video_features:
        # Both hands detected - classic theremin control
        knobs['freq'] = float(
            min_freq + video_features['r_wrist_position'][X] * (max_freq - min_freq)
        )
        knobs['volume'] = float(
            np.clip(1 - video_features['l_wrist_position'][Y], 0, 1)
        )
    elif 'r_wrist_position' in video_features:
        # Only right hand detected - use X for frequency, Y for volume
        knobs['freq'] = float(
            min_freq + video_features['r_wrist_position'][X] * (max_freq - min_freq)
        )
        knobs['volume'] = float(
            np.clip(1 - video_features['r_wrist_position'][Y], 0, 1)
        )
    elif 'l_wrist_position' in video_features:
        # Only left hand detected - use X for frequency, Y for volume
        knobs['freq'] = float(
            min_freq + video_features['l_wrist_position'][X] * (max_freq - min_freq)
        )
        knobs['volume'] = float(
            np.clip(1 - video_features['l_wrist_position'][Y], 0, 1)
        )
    else:
        # No hands detected - silent
        mid_freq = (min_freq + max_freq) / 2
        silent = 0.0
        knobs['freq'] = mid_freq
        knobs['volume'] = silent

    if freq_trans:
        knobs['freq'] = freq_trans(knobs['freq'])

    # Ensure plain Python floats (avoid numpy types for pyo compatibility)
    return {k: float(v) for k, v in knobs.items()}


MOD_FREQ = 0.4
MOD_MUL = 0.1


@add_default_settings('freq_src')
def intro_high_sines(
    base_freq=4000, mod_freq=MOD_FREQ, mod_mul=MOD_MUL, *, freq_src=DFLT_OSC
):
    wav = HarmTable([0.1, 0, 0.2, 0, 0.1, 0, 0, 0, 0.04, 0, 0, 0, 0.02])
    fade = Fader(fadein=3, fadeout=20, mul=mod_mul)
    fade.play()
    mod = Osc(table=wav, freq=mod_freq)
    car = freq_src(freq=[base_freq, base_freq + 40, base_freq - 10], mul=mod)
    pan = SPan(car, pan=[0, 0.5, 1], mul=fade)
    return pan


HI_BASE_FREQ_RANGE = (2000, 6000)
MID_BASE_FREQ_RANGE = (1000, 4000)
LO_BASE_FREQ_RANGE = (200, 1000)
MOD_FREQ_RANGE = (0.1, 1.0)
MOD_MUL_RANGE = (0.000, 0.5)

HI_SINES_FREQ_RANGE = MID_BASE_FREQ_RANGE


def high_sines_theremin_knobs(
    video_features,
    *,
    base_freq_range: tuple = HI_SINES_FREQ_RANGE,
    mod_freq_range: tuple = MOD_FREQ_RANGE,
    mod_mul_range: tuple = MOD_MUL_RANGE,
) -> dict[str, float]:
    """
    Maps hand positions to the three parameters of intro_high_sines synth:
    - Right hand X: Controls base_freq
    - Right hand Y: Controls mod_freq
    - Left hand Y: Controls mod_mul (amplitude)

    Args:
        video_features (dict): Extracted hand feature dictionary
        base_freq_range (tuple): Min and max frequency for the base sine wave (Hz)
        mod_freq_range (tuple): Min and max frequency for modulation (Hz)
        mod_mul_range (tuple): Min and max amplitude for modulation

    Returns:
        Dict[str, float]: Dictionary with 'base_freq', 'mod_freq', 'mod_mul' keys
    """
    knobs = {}

    # Default values (middle of ranges)
    knobs['base_freq'] = (base_freq_range[0] + base_freq_range[1]) / 2
    knobs['mod_freq'] = (mod_freq_range[0] + mod_freq_range[1]) / 2
    knobs['mod_mul'] = (mod_mul_range[0] + mod_mul_range[1]) / 2

    # If no hands detected, return default values
    if not video_features:
        return knobs

    # Right hand controls base_freq (X-axis) and mod_freq (Y-axis)
    if 'r_wrist_position' in video_features:
        r_x, r_y = video_features['r_wrist_position'][0:2]

        # X position controls base frequency
        knobs['base_freq'] = float(
            base_freq_range[0] + r_x * (base_freq_range[1] - base_freq_range[0])
        )

        # Y position controls modulation frequency (inverse mapping feels more intuitive)
        knobs['mod_freq'] = float(
            mod_freq_range[0] + (1 - r_y) * (mod_freq_range[1] - mod_freq_range[0])
        )

    # Left hand controls mod_mul (amplitude) with Y-axis
    if 'l_wrist_position' in video_features:
        y = video_features['l_wrist_position'][1]

        # Y position controls modulation amplitude (inverse mapping)
        knobs['mod_mul'] = float(
            mod_mul_range[0] + (1 - y) * (mod_mul_range[1] - mod_mul_range[0])
        )

    return knobs


def high_sines_pinch_theremin_knobs(
    video_features,
    *,
    base_freq_range: tuple = HI_SINES_FREQ_RANGE,
    mod_freq_range: tuple = MOD_FREQ_RANGE,
    mod_mul_range: tuple = MOD_MUL_RANGE,
) -> dict[str, float]:
    """
    Maps hand positions and pinch gesture to intro_high_sines parameters:
    - Right hand X: Controls base_freq
    - Right pinch distance: Controls mod_freq
    - Left pinch distance: Controls mod_mul

    Args:
        video_features (dict): Extracted hand feature dictionary
        base_freq_range (tuple): Min and max frequency for the base sine wave (Hz)
        mod_freq_range (tuple): Min and max frequency for modulation (Hz)
        mod_mul_range (tuple): Min and max amplitude for modulation

    Returns:
        Dict[str, float]: Dictionary with 'base_freq', 'mod_freq', 'mod_mul' keys
    """
    knobs = {}

    # Default values (middle of ranges)
    knobs['base_freq'] = (base_freq_range[0] + base_freq_range[1]) / 2
    knobs['mod_freq'] = (mod_freq_range[0] + mod_freq_range[1]) / 2
    knobs['mod_mul'] = (mod_mul_range[0] + mod_mul_range[1]) / 2

    # If no hands detected, return default values
    if not video_features:
        return knobs

    # Right hand controls base_freq with X position
    if 'r_wrist_position' in video_features:
        r_x = video_features['r_wrist_position'][0]
        knobs['base_freq'] = float(
            base_freq_range[0] + r_x * (base_freq_range[1] - base_freq_range[0])
        )

    # Right hand pinch controls mod_freq (if available)
    if 'r_thumb_index_distance' in video_features:
        # Map thumb-index distance (typically 0-0.2) to modulation frequency
        # Normalize to 0-1 range assuming max distance of 0.2
        pinch_distance = min(video_features['r_thumb_index_distance'], 0.2) / 0.2
        knobs['mod_freq'] = float(
            mod_freq_range[0] + pinch_distance * (mod_freq_range[1] - mod_freq_range[0])
        )

    # Left hand pinch controls mod_mul (if available)
    if 'l_thumb_index_distance' in video_features:
        # Normalize to 0-1 range assuming max distance of 0.2
        pinch_distance = min(video_features['l_thumb_index_distance'], 0.2) / 0.2
        knobs['mod_mul'] = float(
            mod_mul_range[0] + pinch_distance * (mod_mul_range[1] - mod_mul_range[0])
        )

    return knobs


def high_sines_openness_theremin_knobs(
    video_features,
    *,
    base_freq_range: tuple = HI_SINES_FREQ_RANGE,
    mod_freq_range: tuple = MOD_FREQ_RANGE,
    mod_mul_range: tuple = MOD_MUL_RANGE,
) -> dict[str, float]:
    """
    Maps hand positions and hand openness to intro_high_sines parameters:
    - Right hand X: Controls base_freq
    - Right hand openness: Controls mod_freq
    - Left hand openness: Controls mod_mul

    Args:
        video_features (dict): Extracted hand feature dictionary
        base_freq_range (tuple): Min and max frequency for the base sine wave (Hz)
        mod_freq_range (tuple): Min and max frequency for modulation (Hz)
        mod_mul_range (tuple): Min and max amplitude for modulation

    Returns:
        Dict[str, float]: Dictionary with 'base_freq', 'mod_freq', 'mod_mul' keys
    """
    knobs = {}

    # Default values (middle of ranges)
    knobs['base_freq'] = (base_freq_range[0] + base_freq_range[1]) / 2
    knobs['mod_freq'] = (mod_freq_range[0] + mod_freq_range[1]) / 2
    knobs['mod_mul'] = (mod_mul_range[0] + mod_mul_range[1]) / 2

    # If no hands detected, return default values
    if not video_features:
        return knobs

    # Right hand controls base_freq with X position
    if 'r_wrist_position' in video_features:
        r_x = video_features['r_wrist_position'][0]
        knobs['base_freq'] = float(
            base_freq_range[0] + r_x * (base_freq_range[1] - base_freq_range[0])
        )

    # Right hand openness controls mod_freq (if available)
    if 'r_openness' in video_features:
        # Normalize openness to 0-1 range (typically ranges from 0.05 to 0.3)
        openness = np.clip((video_features['r_openness'] - 0.05) / 0.25, 0, 1)
        knobs['mod_freq'] = float(
            mod_freq_range[0] + openness * (mod_freq_range[1] - mod_freq_range[0])
        )

    # Left hand openness controls mod_mul (if available)
    if 'l_openness' in video_features:
        # Normalize openness to 0-1 range
        openness = np.clip((video_features['l_openness'] - 0.05) / 0.25, 0, 1)
        knobs['mod_mul'] = float(
            mod_mul_range[0] + openness * (mod_mul_range[1] - mod_mul_range[0])
        )

    return knobs


# -------------------------------------------------------------------------------
# Misc utils
# -------------------------------------------------------------------------------


def filter_unchanged_frequencies(_audio_features, previous_data):
    """
    Filter out frequency values that haven't changed since the last update.

    Args:
        _audio_features: Dictionary containing audio feature values
        previous_data: Object storing previous frequency values

    Returns:
        Modified _audio_features with unchanged frequencies removed, and the updated previous_data
    """
    if 'freq' in _audio_features:
        freq = _audio_features['freq']
        previous_data.last_raw_freqs.append(freq)
        if freq == previous_data.last_freq:
            del _audio_features['freq']
        else:
            previous_data.last_freq = freq

    if 'l_freq' in _audio_features:
        l_freq = _audio_features['l_freq']
        if l_freq == previous_data.last_l_freq:
            del _audio_features['l_freq']
        else:
            previous_data.last_l_freq = l_freq

    if 'r_freq' in _audio_features:
        r_freq = _audio_features['r_freq']
        if r_freq == previous_data.last_r_freq:
            del _audio_features['r_freq']
        else:
            previous_data.last_r_freq = r_freq

    return _audio_features, previous_data


# -------------------------------------------------------------------------------
# Module exports
# -------------------------------------------------------------------------------

DFLT_SYNTH = "theremin_synth"
KNOBS = "two_hand_freq_and_volume_knobs"
DFLT_PIPELINE = "theremin"

# Dictionary of available synth functions
synths = {
    "default": locals()[DFLT_SYNTH],
    "theremin_synth": theremin_synth,
    "natural_sounding_synth": natural_sounding_synth,
    "natural_sounding_synth_lr": natural_sounding_synth_lr,
    "rhythmic_fm_synth": rhythmic_fm_synth,
    "sine_synth": sine_synth,
    "fm_synth": fm_synth,
    "supersaw_synth": supersaw_synth,
    "square_synth": square_synth,
    "noise_synth": noise_synth,
    "ringmod_synth": ringmod_synth,
    "chorused_sine_synth": chorused_sine_synth,
    "phase_distortion_synth": phase_distortion_synth,
    "two_voice_synth_func": two_voice_synth_func,
    "intro_high_sines": intro_high_sines,
}

# Dictionary of available audio feature extractors
knobs = {
    "default": locals()[KNOBS],
    "two_hand_freq_and_volume_knobs": two_hand_freq_and_volume_knobs,
    "two_voice_knobs": two_voice_knobs,
    "theremin_knobs": theremin_knobs,
    "simple_two_hands_knobs": simple_two_hands_knobs,
    "rhythmic_fm_synth_knobs": rhythmic_fm_synth_knobs,
    "high_sines_theremin_knobs": high_sines_theremin_knobs,
    "high_sines_pinch_theremin_knobs": high_sines_pinch_theremin_knobs,
    "high_sines_openness_theremin_knobs": high_sines_openness_theremin_knobs,
}


# TODO: Make this pipeline definition and handling less of a mess!
_pipelines = {
    "theremin": {
        "synth": "theremin_synth",
        "knobs": "theremin_knobs",
    },
    "rhythmic_fm": {
        "synth": "rhythmic_fm_synth",
        "knobs": "rhythmic_fm_synth_knobs",
    },
    "high_sines": {
        "synth": "intro_high_sines",
        "knobs": "high_sines_theremin_knobs",
    },
    "high_sines_pinch": {
        "synth": "intro_high_sines",
        "knobs": "high_sines_pinch_theremin_knobs",
    },
    "high_sines_openness": {
        "synth": "intro_high_sines",
        "knobs": "high_sines_openness_theremin_knobs",
    },
    "two_voice_and_hands": {
        "synth": "two_voice_synth_func",
        "knobs": "two_voice_knobs",
    },
    # "natural_sounding_synth": {
    #     "synth": "natural_sounding_synth",
    #     "knobs": "two_hand_freq_and_volume_knobs",  # TODO: Needs a knobs that works
    # },
    "natural_sounding_synth_lr": {
        "synth": "natural_sounding_synth_lr",
        "knobs": "two_hand_freq_and_volume_knobs",
    },
    "sine_two_hands": {
        "synth": "sine_synth",
        "knobs": "simple_two_hands_knobs",
    },
    "phase_distortion_synth": {
        "synth": "phase_distortion_synth",
        "knobs": "simple_two_hands_knobs",
    },
    "supersaw_two_hands": {
        "synth": "supersaw_synth",
        "knobs": "two_hand_freq_and_volume_knobs",
    },
    "square_two_hands": {
        "synth": "square_synth",
        "knobs": "simple_two_hands_knobs",
    },
    "noise_two_hands": {
        "synth": "noise_synth",
        "knobs": "simple_two_hands_knobs",
    },
    "ringmod_two_hands": {  # Note: No sound
        "synth": "ringmod_synth",
        "knobs": "two_hand_freq_and_volume_knobs",
    },
    "chorused_two_hands": {
        "synth": "chorused_sine_synth",
        "knobs": "simple_two_hands_knobs",
    },
}

_pipelines["default"] = _pipelines[DFLT_PIPELINE]


def audio_pipe(*, knobs, synth):
    return {
        "knobs": knobs,
        "synth": synth,
    }


def audio_pipe_call_string(*, knobs, synth):
    return f"audio_pipe(knobs='{knobs}', synth='{synth}')"


pipelines = {k: partial(audio_pipe, **v) for k, v in _pipelines.items()}


def two_voice_knobs(
    video_features,
    *,
    freq_trans: Callable | None = snap_to_scale,
    min_freq: float = DFLT_MIN_FREQ,
    max_freq: float = DFLT_MAX_FREQ,
) -> dict[str, float]:
    """
    Maps hand positions to frequency and volume for both hands.
    Specifically designed for two_voice_synth_func which only accepts
    l_freq, l_volume, r_freq, r_volume parameters.

    Args:
        video_features (dict): Extracted hand feature dictionary.
        min_freq (float): Minimum frequency for pitch control.
        max_freq (float): Maximum frequency for pitch control.
        freq_trans: Optional frequency transformation function.

    Returns:
        Dict[str, float]: Dictionary with 'l_freq', 'l_volume', 'r_freq', 'r_volume' keys.
    """
    knobs = {}

    # Set default silence
    mid_freq = (min_freq + max_freq) / 2
    knobs['l_freq'] = mid_freq
    knobs['l_volume'] = 0.0
    knobs['r_freq'] = mid_freq
    knobs['r_volume'] = 0.0

    if video_features:
        # get range mappers
        freq_mapper = range_mapper('wrist_position_x', 'freq')
        volume_mapper = range_mapper(
            'wrist_position_y', 'volume', ingress=lambda x: 1 - x
        )

        if 'l_wrist_position' in video_features:
            wrist = video_features['l_wrist_position']
            knobs['l_freq'] = freq_mapper(wrist[0])
            knobs['l_volume'] = volume_mapper(wrist[1])
        if 'r_wrist_position' in video_features:
            wrist = video_features['r_wrist_position']
            knobs['r_freq'] = freq_mapper(wrist[0])
            knobs['r_volume'] = volume_mapper(wrist[1])

    # Apply frequency transformation if provided
    if freq_trans:
        knobs['l_freq'] = freq_trans(knobs['l_freq'])
        knobs['r_freq'] = freq_trans(knobs['r_freq'])

    # Only return parameters that two_voice_synth_func can handle
    synth_params = {'l_freq', 'l_volume', 'r_freq', 'r_volume'}
    return {k: float(v) for k, v in knobs.items() if k in synth_params}


def simple_two_hands_knobs(
    video_features,
    *,
    freq_trans: Callable | None = snap_to_scale,
    min_freq: float = DFLT_MIN_FREQ,
    max_freq: float = DFLT_MAX_FREQ,
) -> dict[str, float]:
    """
    Maps hand positions to simple freq and volume parameters.
    For simple synths that only need freq and volume.

    Args:
        video_features (dict): Extracted hand feature dictionary.
        min_freq (float): Minimum frequency for pitch control.
        max_freq (float): Maximum frequency for pitch control.
        freq_trans: Optional frequency transformation function.

    Returns:
        Dict[str, float]: Dictionary with 'freq', 'volume' keys.
    """
    knobs = {
        'freq': (min_freq + max_freq) / 2,  # Default middle frequency
        'volume': 0.0,  # Default silence
    }

    if video_features:
        # get range mappers
        freq_mapper = range_mapper('wrist_position_x', 'freq')
        volume_mapper = range_mapper(
            'wrist_position_y', 'volume', ingress=lambda x: 1 - x
        )

        # Use left hand for main frequency and volume (fallback to right if no left)
        primary_hand = None
        if 'l_wrist_position' in video_features:
            primary_hand = video_features['l_wrist_position']
        elif 'r_wrist_position' in video_features:
            primary_hand = video_features['r_wrist_position']

        if primary_hand:
            knobs['freq'] = freq_mapper(primary_hand[0])
            knobs['volume'] = volume_mapper(primary_hand[1])

    # Apply frequency transformation if provided
    if freq_trans:
        knobs['freq'] = freq_trans(knobs['freq'])

    return {k: float(v) for k, v in knobs.items()}


def ringmod_two_hands_knobs(
    video_features,
    *,
    freq_trans: Callable | None = snap_to_scale,
    min_freq: float = DFLT_MIN_FREQ,
    max_freq: float = DFLT_MAX_FREQ,
) -> dict[str, float]:
    """
    Maps hand positions to parameters for ringmod_synth.
    Combines both hands into a single frequency and volume, plus modulation ratio.

    Args:
        video_features (dict): Extracted hand feature dictionary.
        min_freq (float): Minimum frequency for pitch control.
        max_freq (float): Maximum frequency for pitch control.
        freq_trans: Optional frequency transformation function.

    Returns:
        Dict[str, float]: Dictionary with 'freq', 'volume', 'mod_freq_ratio' keys.
    """
    knobs = {
        'freq': (min_freq + max_freq) / 2,  # Default middle frequency
        'volume': 0.0,  # Default silence
        'mod_freq_ratio': 1.5,  # Default modulation ratio
    }

    if video_features:
        # get range mappers
        freq_mapper = range_mapper('wrist_position_x', 'freq')
        volume_mapper = range_mapper(
            'wrist_position_y', 'volume', ingress=lambda x: 1 - x
        )

        # Use left hand for main frequency and volume (fallback to right if no left)
        primary_hand = None
        if 'l_wrist_position' in video_features:
            primary_hand = video_features['l_wrist_position']
        elif 'r_wrist_position' in video_features:
            primary_hand = video_features['r_wrist_position']

        if primary_hand:
            knobs['freq'] = freq_mapper(primary_hand[0])
            knobs['volume'] = volume_mapper(primary_hand[1])

        # Use right hand openness for modulation ratio (if available)
        if 'r_openness' in video_features:
            # Map openness (0-1) to modulation ratio range (0.5-5.0)
            knobs['mod_freq_ratio'] = 0.5 + video_features['r_openness'] * 4.5

    # Apply frequency transformation if provided
    if freq_trans:
        knobs['freq'] = freq_trans(knobs['freq'])

    return {k: float(v) for k, v in knobs.items()}


def supersaw_two_hands_knobs(
    video_features,
    *,
    freq_trans: Callable | None = snap_to_scale,
    min_freq: float = DFLT_MIN_FREQ,
    max_freq: float = DFLT_MAX_FREQ,
) -> dict[str, float]:
    """
    Maps hand positions to parameters for supersaw_synth.
    Combines both hands into a single frequency and volume, plus detune and n_voices.

    Args:
        video_features (dict): Extracted hand feature dictionary.
        min_freq (float): Minimum frequency for pitch control.
        max_freq (float): Maximum frequency for pitch control.
        freq_trans: Optional frequency transformation function.

    Returns:
        Dict[str, float]: Dictionary with 'freq', 'volume', 'detune', 'n_voices' keys.
    """
    knobs = {
        'freq': (min_freq + max_freq) / 2,  # Default middle frequency
        'volume': 0.0,  # Default silence
        'detune': 0.01,  # Default detune amount
        'n_voices': 7,  # Default number of voices
    }

    if video_features:
        # get range mappers
        freq_mapper = range_mapper('wrist_position_x', 'freq')
        volume_mapper = range_mapper(
            'wrist_position_y', 'volume', ingress=lambda x: 1 - x
        )

        # Use left hand for main frequency and volume (fallback to right if no left)
        primary_hand = None
        if 'l_wrist_position' in video_features:
            primary_hand = video_features['l_wrist_position']
        elif 'r_wrist_position' in video_features:
            primary_hand = video_features['r_wrist_position']

        if primary_hand:
            knobs['freq'] = freq_mapper(primary_hand[0])
            knobs['volume'] = volume_mapper(primary_hand[1])

        # Use right hand openness for detune amount (if available)
        if 'r_openness' in video_features:
            # Map openness (0-1) to detune range (0.001-0.1)
            knobs['detune'] = 0.001 + video_features['r_openness'] * 0.099

        # Use right thumb-index distance for number of voices (if available)
        if 'r_thumb_index_distance' in video_features:
            # Map distance to n_voices range (3-15)
            normalized_distance = np.clip(
                video_features['r_thumb_index_distance'], 0, 1
            )
            knobs['n_voices'] = int(3 + normalized_distance * 12)

    # Apply frequency transformation if provided
    if freq_trans:
        knobs['freq'] = freq_trans(knobs['freq'])

    return {k: float(v) for k, v in knobs.items()}


def rhythmic_fm_synth_knobs(
    video_features,
    freq_trans: Callable | None = snap_to_scale,
) -> dict[str, float]:
    """
    Maps video features to the parameters of rhythmic_fm_synth.

    Args:
        video_features (dict): Hand tracking data from video.

    Returns:
        Dict[str, float]: Audio knob values for rhythmic_fm_synth.
    """
    knobs = {
        'carrier_freq_base': 220.0,
        'fm_transpo': 1.0,
        'index_amount': 5.0,
        'beat_density': 0.1125,
        'reverb_mix': 0.3,
        'noise_level': 0.1,
        'env_attack': 0.01,
        'env_release': 0.5,
        'feedback': 0.1,
        'pan_pos': 0.5,
        'distortion': 0.0,
    }

    if not video_features:
        return knobs

    # Map right wrist X to carrier frequency base
    if 'r_wrist_position' in video_features:
        x = video_features['r_wrist_position'][0]
        knobs['carrier_freq_base'] = 220 + x * (880 - 220)  # Range 220–880 Hz

    # Map left wrist Y to index amount
    if 'l_wrist_position' in video_features:
        y = video_features['l_wrist_position'][1]
        knobs['index_amount'] = 10 * (1 - y)  # Inverse: higher hand = more index

    # Right openness controls feedback
    if 'r_openness' in video_features:
        knobs['feedback'] = np.clip(video_features['r_openness'], 0.0, 1.0)

    # Thumb-index distance controls distortion
    if 'r_thumb_index_distance' in video_features:
        dist = min(video_features['r_thumb_index_distance'], 0.2) / 0.2
        knobs['distortion'] = dist

    # Left openness for reverb mix
    if 'l_openness' in video_features:
        openness = np.clip(video_features['l_openness'], 0.0, 1.0)
        knobs['reverb_mix'] = float(openness)

    return {k: float(v) for k, v in knobs.items()}


# from hum.util import scale_snapper

# snap_to_scale = scale_snapper(scale=(0, 2, 4, 5, 7, 9, 11))


def theremin_knobs(
    video_features,
    *,
    min_freq: float = DFLT_MIN_FREQ,
    max_freq: float = DFLT_MAX_FREQ,
    freq_trans: Callable | None = snap_to_scale,
) -> dict[str, float]:
    """
    Maps hand positions to frequency (pitch) and volume (amplitude),
    mimicking a classic theremin control scheme.

    When both hands are detected:
    - Right hand X position controls frequency (pitch)
    - Left hand Y position controls volume (amplitude)

    When only one hand is detected:
    - X position controls frequency (pitch)
    - Y position controls volume (amplitude)

    Args:
        video_features (dict): Extracted hand feature dictionary.
        min_freq (float): Minimum frequency for pitch control.
        max_freq (float): Maximum frequency for pitch control.

    Returns:
        Dict[str, float]: Dictionary with 'freq', 'volume' keys.
    """
    X, Y = 0, 1
    knobs = {}

    if not video_features:
        return knobs
    elif 'r_wrist_position' in video_features and 'l_wrist_position' in video_features:
        # Both hands detected - classic theremin control
        knobs['freq'] = float(
            min_freq + video_features['r_wrist_position'][X] * (max_freq - min_freq)
        )
        knobs['volume'] = float(
            np.clip(1 - video_features['l_wrist_position'][Y], 0, 1)
        )
    elif 'r_wrist_position' in video_features:
        # Only right hand detected - use X for frequency, Y for volume
        knobs['freq'] = float(
            min_freq + video_features['r_wrist_position'][X] * (max_freq - min_freq)
        )
        knobs['volume'] = float(
            np.clip(1 - video_features['r_wrist_position'][Y], 0, 1)
        )
    elif 'l_wrist_position' in video_features:
        # Only left hand detected - use X for frequency, Y for volume
        knobs['freq'] = float(
            min_freq + video_features['l_wrist_position'][X] * (max_freq - min_freq)
        )
        knobs['volume'] = float(
            np.clip(1 - video_features['l_wrist_position'][Y], 0, 1)
        )
    else:
        # No hands detected - silent
        mid_freq = (min_freq + max_freq) / 2
        silent = 0.0
        knobs['freq'] = mid_freq
        knobs['volume'] = silent

    if freq_trans:
        knobs['freq'] = freq_trans(knobs['freq'])

    # Ensure plain Python floats (avoid numpy types for pyo compatibility)
    return {k: float(v) for k, v in knobs.items()}


MOD_FREQ = 0.4
MOD_MUL = 0.1


@add_default_settings('freq_src')
def intro_high_sines(
    base_freq=4000, mod_freq=MOD_FREQ, mod_mul=MOD_MUL, *, freq_src=DFLT_OSC
):
    wav = HarmTable([0.1, 0, 0.2, 0, 0.1, 0, 0, 0, 0.04, 0, 0, 0, 0.02])
    fade = Fader(fadein=3, fadeout=20, mul=mod_mul)
    fade.play()
    mod = Osc(table=wav, freq=mod_freq)
    car = freq_src(freq=[base_freq, base_freq + 40, base_freq - 10], mul=mod)
    pan = SPan(car, pan=[0, 0.5, 1], mul=fade)
    return pan


HI_BASE_FREQ_RANGE = (2000, 6000)
MID_BASE_FREQ_RANGE = (1000, 4000)
LO_BASE_FREQ_RANGE = (200, 1000)
MOD_FREQ_RANGE = (0.1, 1.0)
MOD_MUL_RANGE = (0.000, 0.5)

HI_SINES_FREQ_RANGE = MID_BASE_FREQ_RANGE


def high_sines_theremin_knobs(
    video_features,
    *,
    base_freq_range: tuple = HI_SINES_FREQ_RANGE,
    mod_freq_range: tuple = MOD_FREQ_RANGE,
    mod_mul_range: tuple = MOD_MUL_RANGE,
) -> dict[str, float]:
    """
    Maps hand positions to the three parameters of intro_high_sines synth:
    - Right hand X: Controls base_freq
    - Right hand Y: Controls mod_freq
    - Left hand Y: Controls mod_mul (amplitude)

    Args:
        video_features (dict): Extracted hand feature dictionary
        base_freq_range (tuple): Min and max frequency for the base sine wave (Hz)
        mod_freq_range (tuple): Min and max frequency for modulation (Hz)
        mod_mul_range (tuple): Min and max amplitude for modulation

    Returns:
        Dict[str, float]: Dictionary with 'base_freq', 'mod_freq', 'mod_mul' keys
    """
    knobs = {}

    # Default values (middle of ranges)
    knobs['base_freq'] = (base_freq_range[0] + base_freq_range[1]) / 2
    knobs['mod_freq'] = (mod_freq_range[0] + mod_freq_range[1]) / 2
    knobs['mod_mul'] = (mod_mul_range[0] + mod_mul_range[1]) / 2

    # If no hands detected, return default values
    if not video_features:
        return knobs

    # Right hand controls base_freq (X-axis) and mod_freq (Y-axis)
    if 'r_wrist_position' in video_features:
        r_x, r_y = video_features['r_wrist_position'][0:2]

        # X position controls base frequency
        knobs['base_freq'] = float(
            base_freq_range[0] + r_x * (base_freq_range[1] - base_freq_range[0])
        )

        # Y position controls modulation frequency (inverse mapping feels more intuitive)
        knobs['mod_freq'] = float(
            mod_freq_range[0] + (1 - r_y) * (mod_freq_range[1] - mod_freq_range[0])
        )

    # Left hand controls mod_mul (amplitude) with Y-axis
    if 'l_wrist_position' in video_features:
        y = video_features['l_wrist_position'][1]

        # Y position controls modulation amplitude (inverse mapping)
        knobs['mod_mul'] = float(
            mod_mul_range[0] + (1 - y) * (mod_mul_range[1] - mod_mul_range[0])
        )

    return knobs


def high_sines_pinch_theremin_knobs(
    video_features,
    *,
    base_freq_range: tuple = HI_SINES_FREQ_RANGE,
    mod_freq_range: tuple = MOD_FREQ_RANGE,
    mod_mul_range: tuple = MOD_MUL_RANGE,
) -> dict[str, float]:
    """
    Maps hand positions and pinch gesture to intro_high_sines parameters:
    - Right hand X: Controls base_freq
    - Right pinch distance: Controls mod_freq
    - Left pinch distance: Controls mod_mul

    Args:
        video_features (dict): Extracted hand feature dictionary
        base_freq_range (tuple): Min and max frequency for the base sine wave (Hz)
        mod_freq_range (tuple): Min and max frequency for modulation (Hz)
        mod_mul_range (tuple): Min and max amplitude for modulation

    Returns:
        Dict[str, float]: Dictionary with 'base_freq', 'mod_freq', 'mod_mul' keys
    """
    knobs = {}

    # Default values (middle of ranges)
    knobs['base_freq'] = (base_freq_range[0] + base_freq_range[1]) / 2
    knobs['mod_freq'] = (mod_freq_range[0] + mod_freq_range[1]) / 2
    knobs['mod_mul'] = (mod_mul_range[0] + mod_mul_range[1]) / 2

    # If no hands detected, return default values
    if not video_features:
        return knobs

    # Right hand controls base_freq with X position
    if 'r_wrist_position' in video_features:
        r_x = video_features['r_wrist_position'][0]
        knobs['base_freq'] = float(
            base_freq_range[0] + r_x * (base_freq_range[1] - base_freq_range[0])
        )

    # Right hand pinch controls mod_freq (if available)
    if 'r_thumb_index_distance' in video_features:
        # Map thumb-index distance (typically 0-0.2) to modulation frequency
        # Normalize to 0-1 range assuming max distance of 0.2
        pinch_distance = min(video_features['r_thumb_index_distance'], 0.2) / 0.2
        knobs['mod_freq'] = float(
            mod_freq_range[0] + pinch_distance * (mod_freq_range[1] - mod_freq_range[0])
        )

    # Left hand pinch controls mod_mul (if available)
    if 'l_thumb_index_distance' in video_features:
        # Normalize to 0-1 range assuming max distance of 0.2
        pinch_distance = min(video_features['l_thumb_index_distance'], 0.2) / 0.2
        knobs['mod_mul'] = float(
            mod_mul_range[0] + pinch_distance * (mod_mul_range[1] - mod_mul_range[0])
        )

    return knobs


def high_sines_openness_theremin_knobs(
    video_features,
    *,
    base_freq_range: tuple = HI_SINES_FREQ_RANGE,
    mod_freq_range: tuple = MOD_FREQ_RANGE,
    mod_mul_range: tuple = MOD_MUL_RANGE,
) -> dict[str, float]:
    """
    Maps hand positions and hand openness to intro_high_sines parameters:
    - Right hand X: Controls base_freq
    - Right hand openness: Controls mod_freq
    - Left hand openness: Controls mod_mul

    Args:
        video_features (dict): Extracted hand feature dictionary
        base_freq_range (tuple): Min and max frequency for the base sine wave (Hz)
        mod_freq_range (tuple): Min and max frequency for modulation (Hz)
        mod_mul_range (tuple): Min and max amplitude for modulation

    Returns:
        Dict[str, float]: Dictionary with 'base_freq', 'mod_freq', 'mod_mul' keys
    """
    knobs = {}

    # Default values (middle of ranges)
    knobs['base_freq'] = (base_freq_range[0] + base_freq_range[1]) / 2
    knobs['mod_freq'] = (mod_freq_range[0] + mod_freq_range[1]) / 2
    knobs['mod_mul'] = (mod_mul_range[0] + mod_mul_range[1]) / 2

    # If no hands detected, return default values
    if not video_features:
        return knobs

    # Right hand controls base_freq with X position
    if 'r_wrist_position' in video_features:
        r_x = video_features['r_wrist_position'][0]
        knobs['base_freq'] = float(
            base_freq_range[0] + r_x * (base_freq_range[1] - base_freq_range[0])
        )

    # Right hand openness controls mod_freq (if available)
    if 'r_openness' in video_features:
        # Normalize openness to 0-1 range (typically ranges from 0.05 to 0.3)
        openness = np.clip((video_features['r_openness'] - 0.05) / 0.25, 0, 1)
        knobs['mod_freq'] = float(
            mod_freq_range[0] + openness * (mod_freq_range[1] - mod_freq_range[0])
        )

    # Left hand openness controls mod_mul (if available)
    if 'l_openness' in video_features:
        # Normalize openness to 0-1 range
        openness = np.clip((video_features['l_openness'] - 0.05) / 0.25, 0, 1)
        knobs['mod_mul'] = float(
            mod_mul_range[0] + openness * (mod_mul_range[1] - mod_mul_range[0])
        )

    return knobs


# -------------------------------------------------------------------------------
# Module exports
# -------------------------------------------------------------------------------

DFLT_SYNTH = "theremin_synth"
KNOBS = "two_hand_freq_and_volume_knobs"
DFLT_PIPELINE = "theremin"

# Dictionary of available synth functions
synths = {
    "default": locals()[DFLT_SYNTH],
    "theremin_synth": theremin_synth,
    "natural_sounding_synth": natural_sounding_synth,
    "natural_sounding_synth_lr": natural_sounding_synth_lr,
    "rhythmic_fm_synth": rhythmic_fm_synth,
    "sine_synth": sine_synth,
    "fm_synth": fm_synth,
    "supersaw_synth": supersaw_synth,
    "square_synth": square_synth,
    "noise_synth": noise_synth,
    "ringmod_synth": ringmod_synth,
    "chorused_sine_synth": chorused_sine_synth,
    "phase_distortion_synth": phase_distortion_synth,
    "two_voice_synth_func": two_voice_synth_func,
    "intro_high_sines": intro_high_sines,
}

# Dictionary of available audio feature extractors
knobs = {
    "default": locals()[KNOBS],
    "two_hand_freq_and_volume_knobs": two_hand_freq_and_volume_knobs,
    "two_voice_knobs": two_voice_knobs,
    "theremin_knobs": theremin_knobs,
    "simple_two_hands_knobs": simple_two_hands_knobs,
    "rhythmic_fm_synth_knobs": rhythmic_fm_synth_knobs,
    "high_sines_theremin_knobs": high_sines_theremin_knobs,
    "high_sines_pinch_theremin_knobs": high_sines_pinch_theremin_knobs,
    "high_sines_openness_theremin_knobs": high_sines_openness_theremin_knobs,
}


# TODO: Make this pipeline definition and handling less of a mess!
_pipelines = {
    "theremin": {
        "synth": "theremin_synth",
        "knobs": "theremin_knobs",
    },
    "rhythmic_fm": {
        "synth": "rhythmic_fm_synth",
        "knobs": "rhythmic_fm_synth_knobs",
    },
    "high_sines": {
        "synth": "intro_high_sines",
        "knobs": "high_sines_theremin_knobs",
    },
    "high_sines_pinch": {
        "synth": "intro_high_sines",
        "knobs": "high_sines_pinch_theremin_knobs",
    },
    "high_sines_openness": {
        "synth": "intro_high_sines",
        "knobs": "high_sines_openness_theremin_knobs",
    },
    "two_voice_and_hands": {
        "synth": "two_voice_synth_func",
        "knobs": "two_voice_knobs",
    },
    # "natural_sounding_synth": {
    #     "synth": "natural_sounding_synth",
    #     "knobs": "two_hand_freq_and_volume_knobs",  # TODO: Needs a knobs that works
    # },
    "natural_sounding_synth_lr": {
        "synth": "natural_sounding_synth_lr",
        "knobs": "two_hand_freq_and_volume_knobs",
    },
    "sine_two_hands": {
        "synth": "sine_synth",
        "knobs": "simple_two_hands_knobs",
    },
    "phase_distortion_synth": {
        "synth": "phase_distortion_synth",
        "knobs": "simple_two_hands_knobs",
    },
    "supersaw_two_hands": {
        "synth": "supersaw_synth",
        "knobs": "two_hand_freq_and_volume_knobs",
    },
    "square_two_hands": {
        "synth": "square_synth",
        "knobs": "simple_two_hands_knobs",
    },
    "noise_two_hands": {
        "synth": "noise_synth",
        "knobs": "simple_two_hands_knobs",
    },
    "ringmod_two_hands": {  # Note: No sound
        "synth": "ringmod_synth",
        "knobs": "two_hand_freq_and_volume_knobs",
    },
    "chorused_two_hands": {
        "synth": "chorused_sine_synth",
        "knobs": "simple_two_hands_knobs",
    },
}

_pipelines["default"] = _pipelines[DFLT_PIPELINE]


def audio_pipe(*, knobs, synth):
    return {
        "knobs": knobs,
        "synth": synth,
    }


def audio_pipe_call_string(*, knobs, synth):
    return f"audio_pipe(knobs='{knobs}', synth='{synth}')"


pipelines = {k: partial(audio_pipe, **v) for k, v in _pipelines.items()}
