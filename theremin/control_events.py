"""Pure helpers for recorded synth control events (record-to-render data hygiene).

A theremin run records what happened to the synth as *control events*: a list of
``(relative_time_seconds, knobs_dict)`` pairs, where each ``knobs_dict`` maps a
parameter name to a number, a string, or a parameter-spec dict (e.g.
``{'value': 440, 'time': 0.025}``). Control events must always be *plain data*
-- hashable string keys, JSON-serializable values, never live pyo signal
objects -- so they can be saved, inspected, and re-rendered offline.

This module owns that contract. It fixes the root cause of issue #4
("Failed to render events: unhashable type: 'SigTo'"): the recording's initial
snapshot includes the synth's *settings* (non-live parameters, such as the
string ``waveform='sine'``) alongside its *dials* (live parameters such as
``freq`` and ``volume``). The offline renderer (``hum``'s
``Synth.render_events``) drives **every** key present in the events with a
fresh ``pyo.SigTo`` control signal, so a settings key makes it pass a ``SigTo``
where the synth function expects, say, a waveform *name* -- and
``theremin_synth`` then uses that value as a dict key, which dies because
``SigTo`` is unhashable. The fix is data hygiene, not renderer patching:
restrict the rendered event stream to dial parameters (settings are baked into
the synth function's defaults) and guarantee every value is plain data.

`render_recording` is the *only* place in the package allowed to call a synth's
``render_events``: making it a named function is what lets the fix be tested at
its call site instead of only in the helper it delegates to (see
``theremin/tests/test_render_call_site.py``).

Everything here is pure and pyo-free, so it is testable without an audio server.
"""

from collections.abc import Iterable, Mapping
from typing import Any

from theremin.util import ensure_plain_types

# Attributes that define a pyo.SigTo-like control signal's plain-data form.
SIGTO_ATTRS = ('value', 'time', 'mul', 'add')

# Where a run's audio recording lands when the caller asked for one without
# naming a file.
DFLT_RECORDING_FILEPATH = 'theremin_recording.wav'

ControlEvent = tuple[float, dict[str, Any]]


def looks_like_live_signal(obj: Any) -> bool:
    """True for pyo ``SigTo``-like live signal objects (duck-typed, no pyo import).

    A live control signal exposes ``value`` and ``time`` attributes and is not a
    mapping. Numbers, strings, dicts and other plain data all return False.

    >>> looks_like_live_signal(440.0), looks_like_live_signal('sine')
    (False, False)
    >>> looks_like_live_signal({'value': 440, 'time': 0.025})
    False
    >>> class FakeSigTo:
    ...     value, time = 440.0, 0.025
    >>> looks_like_live_signal(FakeSigTo())
    True
    """
    if isinstance(obj, Mapping):
        return False
    return hasattr(obj, 'value') and hasattr(obj, 'time')


def plain_control_value(value: Any) -> Any:
    """Return a plain-data (JSON-serializable) representation of a knob value.

    - numbers and strings: unchanged (numpy scalars converted to builtins)
    - mappings: converted recursively
    - SigTo-like live signals: converted to their parameter dict, keeping only
      numeric parameters (a nested live ``value`` has no plain representation)

    >>> plain_control_value(440.0)
    440.0
    >>> plain_control_value({'value': 440, 'time': 0.025})
    {'value': 440, 'time': 0.025}
    >>> class FakeSigTo:
    ...     def __init__(self):
    ...         self.value, self.time, self.mul, self.add = 880.0, 0.1, 1.0, 0.0
    >>> plain_control_value(FakeSigTo())
    {'value': 880.0, 'time': 0.1, 'mul': 1.0, 'add': 0.0}
    """
    if looks_like_live_signal(value):
        plain = {}
        for attr in SIGTO_ATTRS:
            attr_value = ensure_plain_types(getattr(value, attr, None))
            if isinstance(attr_value, (int, float)) and not isinstance(
                attr_value, bool
            ):
                plain[attr] = attr_value
        return plain
    if isinstance(value, Mapping):
        return {k: plain_control_value(v) for k, v in value.items()}
    return ensure_plain_types(value)


def synth_dials(synth: Any) -> frozenset[str] | None:
    """Best-effort extraction of a synth's dial (live parameter) names.

    Understands, in order: a ``dials``/``_dials`` attribute (e.g.
    ``hum.pyo_util.Synth``), a ``_default_dials`` attribute (synth functions
    decorated with hum's ``add_default_dials`` -- either a collection of names
    or a space-separated string), and finally a wrapped ``_synth_func``.
    Returns None when the dials cannot be determined.

    >>> def a_synth_func(freq=440, volume=0.5):
    ...     pass
    >>> a_synth_func._default_dials = 'freq volume'
    >>> sorted(synth_dials(a_synth_func))
    ['freq', 'volume']
    >>> synth_dials(object()) is None
    True
    """
    for attr in ('dials', '_dials', '_default_dials'):
        dials = getattr(synth, attr, None)
        if dials:
            if isinstance(dials, str):
                dials = dials.split()
            return frozenset(dials)
    wrapped = getattr(synth, '_synth_func', None)
    if wrapped is not None:
        return synth_dials(wrapped)
    return None


def renderable_events(
    control_events: Iterable[ControlEvent], *, dials: Iterable[str] | None = None
) -> list[ControlEvent]:
    """Restrict recorded control events to plain-data dial updates, ready to render.

    Keeps only the keys in ``dials`` (when given): non-dial "settings" (e.g. a
    string waveform name) cannot be driven by a control signal in an offline
    render and are baked into the synth function's defaults instead. Every kept
    value is converted to plain data via `plain_control_value`. Timestamps and
    event order are preserved -- including trailing empty events, which define
    the total render duration.

    >>> events = [
    ...     (0, {'freq': {'value': 440, 'time': 0.025}, 'waveform': 'sine'}),
    ...     (0.5, {'freq': 452.1}),
    ...     (1.0, {}),
    ... ]
    >>> renderable_events(events, dials={'freq', 'volume'})
    [(0.0, {'freq': {'value': 440, 'time': 0.025}}), (0.5, {'freq': 452.1}), (1.0, {})]
    """
    if dials is not None:
        dials = frozenset(dials)
    events = []
    for event_time, knobs in control_events:
        kept = {
            name: plain_control_value(value)
            for name, value in dict(knobs).items()
            if dials is None or name in dials
        }
        events.append((float(event_time), kept))
    return events


def render_recording(
    synth: Any,
    control_events: Iterable[ControlEvent],
    *,
    output_filepath: str = DFLT_RECORDING_FILEPATH,
) -> str:
    """Render a run's recorded control events to an audio file. Dial keys only.

    This is the fix site for issue #4, and the **only** place in the package
    allowed to call a synth's ``render_events``: the offline renderer wraps
    every key it is handed in a fresh live control signal, so letting a settings
    key (e.g. ``waveform='sine'``) through delivers an unhashable ``SigTo``
    where the synth function expects a name. Restricting the stream to the
    synth's dials (`synth_dials`) and plainifying it (`renderable_events`) is
    what prevents that.

    Returns the path written to.

    >>> class FakeSynth:
    ...     _dials = {'freq', 'volume'}
    ...     def render_events(self, events, output_filepath):
    ...         self.rendered = events
    >>> synth = FakeSynth()
    >>> recording = [(0, {'freq': 440, 'waveform': 'sine'}), (1.0, {})]
    >>> render_recording(synth, recording, output_filepath='out.wav')
    'out.wav'
    >>> synth.rendered
    [(0.0, {'freq': 440}), (1.0, {})]
    """
    events = renderable_events(control_events, dials=synth_dials(synth))
    synth.render_events(events, output_filepath=output_filepath)
    return output_filepath
