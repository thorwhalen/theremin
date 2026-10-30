# theremin.control_events

Pure helpers for recorded synth control events (record-to-render data hygiene).

A theremin run records what happened to the synth as *control events*: a list of
`(relative_time_seconds, knobs_dict)` pairs, where each `knobs_dict` maps a
parameter name to a number, a string, or a parameter-spec dict (e.g.
`{'value': 440, 'time': 0.025}`). Control events must always be *plain data*
– hashable string keys, JSON-serializable values, never live pyo signal
objects – so they can be saved, inspected, and re-rendered offline.

This module owns that contract. It fixes the root cause of issue #4
(“Failed to render events: unhashable type: ‘SigTo’”): the recording’s initial
snapshot includes the synth’s *settings* (non-live parameters, such as the
string `waveform='sine'`) alongside its *dials* (live parameters such as
`freq` and `volume`). The offline renderer (`hum`’s
`Synth.render_events`) drives **every** key present in the events with a
fresh `pyo.SigTo` control signal, so a settings key makes it pass a `SigTo`
where the synth function expects, say, a waveform *name* – and
`theremin_synth` then uses that value as a dict key, which dies because
`SigTo` is unhashable. The fix is data hygiene, not renderer patching:
restrict the rendered event stream to dial parameters (settings are baked into
the synth function’s defaults) and guarantee every value is plain data.

`render_recording` is the *only* place in the package allowed to call a synth’s
`render_events`: making it a named function is what lets the fix be tested at
its call site instead of only in the helper it delegates to (see
`theremin/tests/test_render_call_site.py`).

Everything here is pure and pyo-free, so it is testable without an audio server.

### Functions

| [`looks_like_live_signal`](#theremin.control_events.looks_like_live_signal)(obj)                        | True for pyo `SigTo`-like live signal objects (duck-typed, no pyo import).    |
|-----------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------|
| [`plain_control_value`](#theremin.control_events.plain_control_value)(value)                         | Return a plain-data (JSON-serializable) representation of a knob value.       |
| [`render_recording`](#theremin.control_events.render_recording)(synth, control_events, \*[, ...]) | Render a run's recorded control events to an audio file.                      |
| [`renderable_events`](#theremin.control_events.renderable_events)(control_events, \*[, dials])     | Restrict recorded control events to plain-data dial updates, ready to render. |
| [`synth_dials`](#theremin.control_events.synth_dials)(synth)                                 | Best-effort extraction of a synth's dial (live parameter) names.              |

### theremin.control_events.looks_like_live_signal(obj)

True for pyo `SigTo`-like live signal objects (duck-typed, no pyo import).

A live control signal exposes `value` and `time` attributes and is not a
mapping. Numbers, strings, dicts and other plain data all return False.

```pycon
>>> looks_like_live_signal(440.0), looks_like_live_signal('sine')
(False, False)
>>> looks_like_live_signal({'value': 440, 'time': 0.025})
False
>>> class FakeSigTo:
...     value, time = 440.0, 0.025
>>> looks_like_live_signal(FakeSigTo())
True
```

* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

### theremin.control_events.plain_control_value(value)

Return a plain-data (JSON-serializable) representation of a knob value.

- numbers and strings: unchanged (numpy scalars converted to builtins)
- mappings: converted recursively
- SigTo-like live signals: converted to their parameter dict, keeping only
  numeric parameters (a nested live `value` has no plain representation)

```pycon
>>> plain_control_value(440.0)
440.0
>>> plain_control_value({'value': 440, 'time': 0.025})
{'value': 440, 'time': 0.025}
>>> class FakeSigTo:
...     def __init__(self):
...         self.value, self.time, self.mul, self.add = 880.0, 0.1, 1.0, 0.0
>>> plain_control_value(FakeSigTo())
{'value': 880.0, 'time': 0.1, 'mul': 1.0, 'add': 0.0}
```

* **Return type:**
  [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)

### theremin.control_events.render_recording(synth, control_events, , output_filepath='theremin_recording.wav')

Render a run’s recorded control events to an audio file. Dial keys only.

This is the fix site for issue #4, and the **only** place in the package
allowed to call a synth’s `render_events`: the offline renderer wraps
every key it is handed in a fresh live control signal, so letting a settings
key (e.g. `waveform='sine'`) through delivers an unhashable `SigTo`
where the synth function expects a name. Restricting the stream to the
synth’s dials (`synth_dials`) and plainifying it (`renderable_events`) is
what prevents that.

Returns the path written to.

```pycon
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
```

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### theremin.control_events.renderable_events(control_events, , dials=None)

Restrict recorded control events to plain-data dial updates, ready to render.

Keeps only the keys in `dials` (when given): non-dial “settings” (e.g. a
string waveform name) cannot be driven by a control signal in an offline
render and are baked into the synth function’s defaults instead. Every kept
value is converted to plain data via `plain_control_value`. Timestamps and
event order are preserved – including trailing empty events, which define
the total render duration.

```pycon
>>> events = [
...     (0, {'freq': {'value': 440, 'time': 0.025}, 'waveform': 'sine'}),
...     (0.5, {'freq': 452.1}),
...     (1.0, {}),
... ]
>>> renderable_events(events, dials={'freq', 'volume'})
[(0.0, {'freq': {'value': 440, 'time': 0.025}}), (0.5, {'freq': 452.1}), (1.0, {})]
```

* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`float`](https://docs.python.org/3/builtins/functions.html#float), [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]]]

### theremin.control_events.synth_dials(synth)

Best-effort extraction of a synth’s dial (live parameter) names.

Understands, in order: a `dials`/`_dials` attribute (e.g.
`hum.pyo_util.Synth`), a `_default_dials` attribute (synth functions
decorated with hum’s `add_default_dials` – either a collection of names
or a space-separated string), and finally a wrapped `_synth_func`.
Returns None when the dials cannot be determined.

```pycon
>>> def a_synth_func(freq=440, volume=0.5):
...     pass
>>> a_synth_func._default_dials = 'freq volume'
>>> sorted(synth_dials(a_synth_func))
['freq', 'volume']
>>> synth_dials(object()) is None
True
```

* **Return type:**
  [`frozenset`](https://docs.python.org/3/builtins/stdtypes.html#frozenset)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)
