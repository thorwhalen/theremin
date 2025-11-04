import numbers
import numpy as np

from theremin.audio import (
    two_hand_freq_and_volume_knobs,
    two_voice_knobs,
    theremin_knobs,
    simple_two_hands_knobs,
    rhythmic_fm_synth_knobs,
    high_sines_theremin_knobs,
    high_sines_pinch_theremin_knobs,
    high_sines_openness_theremin_knobs,
)
from theremin.dag_audio_features import theremin_dag_knobs
from theremin.audio_features import create_theremin_builder
from theremin.util import ensure_plain_types


def _assert_builtin_numbers(d):
    assert isinstance(d, dict)
    for k, v in d.items():
        assert isinstance(v, numbers.Number), f"{k} not a number: {type(v)}"
        # numpy numbers should have been converted to builtins
        assert not isinstance(
            v, (np.floating, np.integer)
        ), f"{k} is numpy scalar: {type(v)}"


def test_knobs_return_builtin_numbers():
    vf_both = {
        'l_wrist_position': (0.5, 0.5),
        'r_wrist_position': (0.5, 0.5),
        'l_openness': 0.5,
        'r_openness': 0.5,
        'l_thumb_index_distance': 0.1,
        'r_thumb_index_distance': 0.1,
    }
    vf_left = {'l_wrist_position': (0.5, 0.5)}
    vf_right = {'r_wrist_position': (0.5, 0.5)}

    # Call each knobs function with minimal suitable inputs
    _assert_builtin_numbers(theremin_knobs(vf_both))
    _assert_builtin_numbers(two_hand_freq_and_volume_knobs(vf_both))
    _assert_builtin_numbers(two_voice_knobs(vf_both))
    _assert_builtin_numbers(simple_two_hands_knobs(vf_left))
    _assert_builtin_numbers(rhythmic_fm_synth_knobs(vf_both))
    _assert_builtin_numbers(high_sines_theremin_knobs(vf_both))
    _assert_builtin_numbers(
        high_sines_pinch_theremin_knobs(
            {
                'r_wrist_position': (0.5, 0.5),
                'r_thumb_index_distance': 0.1,
                'l_thumb_index_distance': 0.1,
            }
        )
    )
    _assert_builtin_numbers(
        high_sines_openness_theremin_knobs(
            {
                'r_wrist_position': (0.5, 0.5),
                'r_openness': 0.5,
                'l_openness': 0.5,
            }
        )
    )


def test_dag_knobs_return_builtin_numbers():
    vf = {
        'l_wrist_position': (0.5, 0.5),
        'r_wrist_position': (0.5, 0.5),
    }
    out = theremin_dag_knobs(vf)
    _assert_builtin_numbers(out)


def test_audio_feature_builder_returns_builtin_numbers():
    builder = create_theremin_builder()
    vf = {
        'l_wrist_position': (0.5, 0.5),
        'r_wrist_position': (0.5, 0.5),
        'r_openness': 0.5,
        'r_thumb_index_distance': 0.1,
    }
    out = builder(vf)
    _assert_builtin_numbers(out)


def test_ensure_plain_types_converts_numpy_scalars_and_arrays():
    d = {
        'a': np.float64(1.2),
        'b': np.int64(3),
        'c': np.array(4.5),  # 0-d array
        'd': np.array([1.0, 2.0]),  # 1-d array
        'e': {'f': np.float32(0.1)},
    }
    out = ensure_plain_types(d)
    assert isinstance(out['a'], float)
    assert isinstance(out['b'], int)
    assert isinstance(out['c'], float)
    assert isinstance(out['d'], list)
    assert all(isinstance(x, float) for x in out['d'])
    assert isinstance(out['e']['f'], float)
