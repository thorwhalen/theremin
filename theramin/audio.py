"""Audio and synthesizer functions for theremin."""

import numpy as np
from typing import Dict, Union, Callable
from functools import partial

from hum import Synth
from hum.pyo_util import add_default_dials
from pyo import *

# -------------------------------------------------------------------------------
# Synthesizer functions
# -------------------------------------------------------------------------------


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
    return Sine(freq=freq, mul=volume)


def fm_synth(freq=440, volume=0, carrier_ratio=1.0, mod_index=2.0, mod_freq_ratio=2.0):
    """Frequency modulation synthesizer."""
    mod = Sine(freq=freq * mod_freq_ratio, mul=freq * mod_index)
    car = Sine(freq=freq * carrier_ratio + mod, mul=volume)
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
    sine = Sine(freq=freq, mul=volume * (1 - noise_level))
    noise = Noise(mul=volume * noise_level)
    return sine + noise


def ringmod_synth(freq=440, volume=0, mod_freq_ratio=1.5):
    """Ring modulation synthesizer."""
    mod = Sine(freq=freq * mod_freq_ratio)
    carrier = Sine(freq=freq)
    return (carrier * mod) * volume


def chorused_sine_synth(freq=440, volume=0, depth=5, speed=0.3):
    """Chorused sine wave with LFO modulation."""
    lfo = Sine(freq=speed, mul=depth)
    mod_freq = freq + lfo
    return Sine(freq=mod_freq, mul=volume)


def phase_distortion_synth(freq=440, volume=0, distortion=0.5):
    """Phase distortion synthesizer."""
    phasor = Phasor(freq=freq)
    distorted = phasor + (Sine(freq=freq * 2, mul=distortion) * phasor)
    return distorted * volume


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


def obfuscate_args(func, keep_args):
    """
    Creates a new function with only the specified arguments,
    with other arguments fixed to their defaults.
    """
    from i2 import partialx, Sig as Signature

    func_sig = Signature(func)
    if not all([arg in keep_args for arg in func_sig.names[: len(keep_args)]]):
        raise ValueError("keep_args must be in the beginning of Sig(foo).names")
    defaults_of_other_args = {
        k: v for k, v in func_sig.defaults.items() if k not in keep_args
    }
    return partialx(func, **defaults_of_other_args, _rm_partialize=True)


# Create a simplified two-voice synth function that only exposes the necessary parameters
two_voice_synth_func = obfuscate_args(
    _two_voice_synth_func, keep_args=['l_freq', 'l_volume', 'r_freq', 'r_volume']
)


# -------------------------------------------------------------------------------
# Knob (control parameter) functions
# -------------------------------------------------------------------------------


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
    freq = float(min_freq + wrist[0] * (max_freq - min_freq))
    vol = float(np.clip(1 - wrist[1], 0, 1))
    return freq, vol


def two_hand_freq_and_volume_knobs(
    hand_features,
    *,
    min_freq: float = DFLT_MIN_FREQ,
    max_freq: float = DFLT_MAX_FREQ,
) -> Dict[str, float]:
    """
    Maps hand positions to frequency and volume for both hands.

    Args:
        hand_features (dict): Extracted hand feature dictionary.
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

    if not hand_features:
        return knobs
    else:
        if 'l_wrist_position' in hand_features:
            knobs['l_freq'], knobs['l_volume'] = _calculate_freq_and_vol_from_wrist(
                hand_features['l_wrist_position'], min_freq, max_freq
            )
        if 'r_wrist_position' in hand_features:
            knobs['r_freq'], knobs['r_volume'] = _calculate_freq_and_vol_from_wrist(
                hand_features['r_wrist_position'], min_freq, max_freq
            )

    return knobs


def theremin_knobs(
    hand_features,
    *,
    min_freq: float = DFLT_MIN_FREQ,
    max_freq: float = DFLT_MAX_FREQ,
) -> Dict[str, float]:
    """
    Maps right hand to frequency (pitch) and left hand to volume (amplitude),
    mimicking a classic theremin control scheme.

    Args:
        hand_features (dict): Extracted hand feature dictionary.
        min_freq (float): Minimum frequency for pitch control.
        max_freq (float): Maximum frequency for pitch control.

    Returns:
        Dict[str, float]: Dictionary with 'freq', 'volume' keys.
    """
    X, Y = 0, 1
    knobs = {}

    if not hand_features:
        return knobs
    elif 'r_wrist_position' in hand_features and 'l_wrist_position' in hand_features:
        knobs['freq'] = float(
            min_freq + hand_features['r_wrist_position'][X] * (max_freq - min_freq)
        )
        knobs['volume'] = float(np.clip(1 - hand_features['l_wrist_position'][Y], 0, 1))
    else:
        mid_freq = (min_freq + max_freq) / 2
        silent = 0.0
        knobs['freq'] = mid_freq
        knobs['volume'] = silent

    return knobs


# -------------------------------------------------------------------------------
# Module exports
# -------------------------------------------------------------------------------

# Dictionary of available synth functions
synth_funcs = {
    "theremin_synth": theremin_synth,
    "sine_synth": sine_synth,
    "fm_synth": fm_synth,
    "supersaw_synth": supersaw_synth,
    "square_synth": square_synth,
    "noise_synth": noise_synth,
    "ringmod_synth": ringmod_synth,
    "chorused_sine_synth": chorused_sine_synth,
    "phase_distortion_synth": phase_distortion_synth,
    "two_voice_synth_func": two_voice_synth_func,
}

# Dictionary of available audio feature extractors
audio_feature_funcs = {
    "two_hand_freq_and_volume_knobs": two_hand_freq_and_volume_knobs,
    "theremin_knobs": theremin_knobs,
}
