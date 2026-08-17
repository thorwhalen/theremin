"""Tests for theremin.control_events -- the record-to-render data contract.

Regression tests for issue #4 ("Failed to render events: unhashable type:
'SigTo'"). All tests here are pure: no pyo, no audio server, no camera.

The recording fixtures mirror the exact shape hum's ``Synth`` produces for the
default theremin pipeline: an initial snapshot of *all* parameters (dials as
``{'value', 'time', 'mul', 'add'}`` dicts, settings as raw values -- including
the string ``waveform``), then dial-only update dicts, then a final empty event
marking the end of the recording.
"""

import json

import numpy as np
import pytest

from theremin.control_events import (
    DFLT_RECORDING_FILEPATH,
    looks_like_live_signal,
    plain_control_value,
    render_recording,
    renderable_events,
    synth_dials,
)
from theremin.util import ensure_plain_types


class FakeSigTo:
    """Duck-types pyo.SigTo without importing pyo. Unhashable, like the real one."""

    __hash__ = None  # pyo's SigTo defines __eq__ (signal comparison) sans __hash__

    def __init__(self, value=0.0, time=0.025, mul=1.0, add=0.0):
        self.value = value
        self.time = time
        self.mul = mul
        self.add = add


THEREMIN_DIALS = frozenset({'freq', 'volume'})

WAVEFORMS = {'sine': 'a sine table', 'saw': 'a saw table'}


def theremin_like_synth_func(
    freq=440,
    volume=0.5,
    *,
    attack=0.01,
    release=0.1,
    vibrato_rate=5,
    vibrato_depth=5,
    waveform='sine',
):
    """Stands in for `theremin.audio.theremin_synth`, crash mechanism included.

    Dials arrive as live signals and are simply used as signals; the `waveform`
    *setting* is used as a **dict key**, which is the operation that dies with
    "unhashable type: 'SigTo'" when a settings key is left in the rendered
    stream (issue #4).
    """
    return WAVEFORMS[waveform]


class FakeRenderingSynth:
    """Models hum's ``Synth.render_events`` closely enough to reproduce issue #4.

    The essential behaviour being modelled: the renderer wraps **every key
    present anywhere in the event stream** in a fresh control signal and calls
    the synth function with them. That is why a surviving settings key is fatal,
    and why the fix has to happen before `render_events` is called.
    """

    _dials = THEREMIN_DIALS

    def __init__(self):
        self.rendered_events = None
        self.output_filepath = None

    def render_events(self, control_events, output_filepath):
        self.rendered_events = control_events
        self.output_filepath = output_filepath
        driven_keys = {name for _, knobs in control_events for name in knobs}
        return theremin_like_synth_func(**{k: FakeSigTo() for k in driven_keys})


def issue_4_style_recording():
    """The recording shape that made render_events die in issue #4."""
    return [
        (
            0,
            {
                # dials, serialized at record time by hum's serialize_knobs:
                'freq': {'value': 440, 'time': 0.025, 'mul': 1.0, 'add': 0.0},
                'volume': {'value': 0.5, 'time': 0.025, 'mul': 1.0, 'add': 0.0},
                # settings (non-live), snapshotted alongside the dials:
                'attack': 0.01,
                'release': 0.1,
                'vibrato_rate': 5,
                'vibrato_depth': 5,
                'waveform': 'sine',  # <-- the key that broke rendering
            },
        ),
        (0.5, {'freq': 452.1, 'volume': 0.6}),
        (1.0, {'freq': 460.0}),
        (1.5, {}),  # stop_recording end marker: defines the render duration
    ]


# --------------------------------------------------------------------------------------
# render_recording -- the fix site itself
#
# `renderable_events` is a pure helper; these tests exercise the function that
# `run_theremin` actually calls, against a synth fake that crashes the same way
# hum's renderer did. Which call site invokes it is guarded separately, in
# test_render_call_site.py.


def test_the_synth_fake_really_does_reproduce_issue_4():
    """Guards the guard: if the fake cannot crash, the test below proves nothing."""
    synth = FakeRenderingSynth()
    with pytest.raises(TypeError, match='unhashable'):
        # The *unfiltered* recording -- i.e. what the pre-fix call site handed over.
        synth.render_events(issue_4_style_recording(), output_filepath='out.wav')


def test_render_recording_survives_an_issue_4_style_recording():
    synth = FakeRenderingSynth()
    written_to = render_recording(
        synth, issue_4_style_recording(), output_filepath='out.wav'
    )
    assert written_to == 'out.wav'
    assert synth.output_filepath == 'out.wav'


def test_render_recording_hands_the_synth_dial_keys_only():
    synth = FakeRenderingSynth()
    render_recording(synth, issue_4_style_recording(), output_filepath='out.wav')
    driven_keys = {name for _, knobs in synth.rendered_events for name in knobs}
    # Every key here gets wrapped in a live control signal by the renderer, so
    # settings -- `waveform` above all -- must not be among them.
    assert driven_keys == THEREMIN_DIALS
    assert 'waveform' not in driven_keys
    json.dumps(synth.rendered_events)  # and the stream stays plain data


def test_render_recording_defaults_to_the_shared_recording_filepath():
    synth = FakeRenderingSynth()
    assert render_recording(synth, issue_4_style_recording()) == (
        DFLT_RECORDING_FILEPATH
    )


def test_render_recording_keeps_all_keys_when_dials_cannot_be_determined():
    """No dial information is not a licence to invent one: pass the stream through."""

    class DiallessSynth(FakeRenderingSynth):
        _dials = None

    synth = DiallessSynth()
    with pytest.raises(TypeError, match='unhashable'):
        # Faithful: with no dials to filter by, the settings survive and the
        # renderer chokes exactly as before. Better a loud failure than silently
        # guessing which parameters are live.
        render_recording(synth, issue_4_style_recording(), output_filepath='out.wav')


# --------------------------------------------------------------------------------------
# renderable_events


def test_renderable_events_drops_non_dial_settings():
    events = renderable_events(issue_4_style_recording(), dials=THEREMIN_DIALS)
    all_keys = {k for _, knobs in events for k in knobs}
    # This is exactly the key set hum's render_events will SigTo-wrap and feed
    # back into the synth function: it must contain dials only.
    assert all_keys == THEREMIN_DIALS
    assert 'waveform' not in all_keys


def test_renderable_events_preserves_times_order_and_end_marker():
    events = renderable_events(issue_4_style_recording(), dials=THEREMIN_DIALS)
    assert [t for t, _ in events] == [0.0, 0.5, 1.0, 1.5]
    # The trailing empty event survives: it is what defines the total duration.
    assert events[-1] == (1.5, {})
    # Dial values -- both spec-dicts and raw numbers -- are preserved.
    assert events[0][1]['freq'] == {'value': 440, 'time': 0.025, 'mul': 1.0, 'add': 0.0}
    assert events[1][1] == {'freq': 452.1, 'volume': 0.6}


def test_renderable_events_output_is_json_serializable():
    events = renderable_events(issue_4_style_recording(), dials=THEREMIN_DIALS)
    round_tripped = json.loads(json.dumps(events))
    assert len(round_tripped) == len(events)


def test_renderable_events_without_dials_keeps_all_keys_but_plainifies():
    recording = [(0, {'freq': np.float64(440.0), 'waveform': 'sine'})]
    events = renderable_events(recording)
    assert events == [(0.0, {'freq': 440.0, 'waveform': 'sine'})]
    assert isinstance(events[0][1]['freq'], float)  # numpy scalar converted


def test_renderable_events_converts_live_signal_values_to_plain_data():
    # Belt-and-braces: even if a live SigTo-like object leaks into an event
    # value (directly or nested in a spec dict), the rendered stream stays
    # plain data.
    recording = [
        (0, {'freq': FakeSigTo(value=880.0, time=0.1), 'volume': 0.5}),
        (0.5, {'freq': {'value': FakeSigTo(value=220.0), 'time': 0.05}}),
    ]
    events = renderable_events(recording, dials=THEREMIN_DIALS)
    assert events[0][1]['freq'] == {'value': 880.0, 'time': 0.1, 'mul': 1.0, 'add': 0.0}
    # A nested live value has no plain representation: it is converted to its
    # numeric parameters (none here beyond value/time/mul/add of the fake).
    nested = events[1][1]['freq']
    assert nested['time'] == 0.05
    assert isinstance(nested['value'], dict)  # the fake SigTo, plainified
    json.dumps(events)  # the whole stream must be JSON-serializable


# --------------------------------------------------------------------------------------
# plain_control_value / looks_like_live_signal


def test_plain_control_value_passthrough():
    assert plain_control_value(440.0) == 440.0
    assert plain_control_value(5) == 5
    assert plain_control_value('sine') == 'sine'
    assert plain_control_value({'value': 440, 'time': 0.025}) == {
        'value': 440,
        'time': 0.025,
    }
    assert plain_control_value(np.float64(1.5)) == 1.5


def test_plain_control_value_converts_sigto_like_objects():
    assert plain_control_value(FakeSigTo(value=880.0, time=0.1)) == {
        'value': 880.0,
        'time': 0.1,
        'mul': 1.0,
        'add': 0.0,
    }


def test_looks_like_live_signal():
    assert looks_like_live_signal(FakeSigTo())
    assert not looks_like_live_signal(440.0)
    assert not looks_like_live_signal('sine')
    assert not looks_like_live_signal({'value': 440, 'time': 0.025})


# --------------------------------------------------------------------------------------
# synth_dials


def test_synth_dials_from_synth_object_with_private_dials():
    class FakeSynth:
        _dials = {'freq', 'volume'}

    assert synth_dials(FakeSynth()) == frozenset({'freq', 'volume'})


def test_synth_dials_from_decorated_function():
    def a_synth_func(freq=440, volume=0.5, *, waveform='sine'):
        pass

    a_synth_func._default_dials = {'freq', 'volume'}  # hum's add_default_dials
    assert synth_dials(a_synth_func) == frozenset({'freq', 'volume'})


def test_synth_dials_from_space_separated_string():
    def a_synth_func(freq=440, volume=0.5):
        pass

    a_synth_func._default_dials = 'freq volume'
    assert synth_dials(a_synth_func) == frozenset({'freq', 'volume'})


def test_synth_dials_follows_wrapped_synth_func():
    def a_synth_func(freq=440):
        pass

    a_synth_func._default_dials = {'freq'}

    class FakeSynthWrapper:
        _synth_func = staticmethod(a_synth_func)

    assert synth_dials(FakeSynthWrapper()) == frozenset({'freq'})


def test_synth_dials_returns_none_when_undetermined():
    assert synth_dials(object()) is None


# --------------------------------------------------------------------------------------
# ensure_plain_types (moved from test_type_sanitation.py so it keeps running in
# environments without pyo)


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
