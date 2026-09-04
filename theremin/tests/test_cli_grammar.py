"""Pins the command-line surface the argh -> cw migration had to preserve.

The CLI moved from ``argh.dispatch_command`` + ten ``@argh.arg`` decorators to
``cw.dispatch(theremin_cli, config=THEREMIN_CLI_CONFIG)``. That swap is only safe
because ``cw`` reproduces two argh rules exactly, and **neither is obvious**:

* ``cw`` reads ``config`` and ``func._cw``; it does **not** read an ``@argh.arg``
  decorator. Deleting the decorators without carrying their ``nargs``/``const``/
  ``help``/flag spellings into ``config`` would have changed the grammar silently --
  a bare ``--pipeline`` would stop meaning "list the pipelines".
* declared flags are **appended** to the inferred ones rather than replacing them,
  which is why ``--synth`` renders long-first and every other short-flagged option
  renders short-first. Only ``synth`` genuinely needs its ``-s`` declared: ``synth``
  and ``scale`` share a first character, so the inferred short flag is suppressed for
  both. ``-p -v -k -r -n -w`` would have been inferred anyway.

These tests are the record of what "no diff" meant, so a future cw upgrade that moves
any of it fails here rather than in a user's terminal. They touch only the parser --
never the function body -- so they need neither ``pyo`` nor a camera.
"""

import argparse

import pytest

import cw

from theremin.script_utils import theremin_cli, THEREMIN_CLI_CONFIG

# dest -> (option strings in render order, nargs, const, default)
EXPECTED_OPTIONS = {
    'pipeline': (['-p', '--pipeline'], '?', 'list', 'theremin'),
    'video_features': (['-v', '--video-features'], '?', 'list', 'many_video_features'),
    'knobs': (['-k', '--knobs'], '?', 'list', 'theremin_knobs'),
    # long-first: the inferred `-s` is suppressed by the synth/scale collision, so the
    # declared one is appended after `--synth` instead of merging into an existing pair.
    'synth': (['--synth', '-s'], '?', 'list', 'theremin_synth'),
    # store_true actions report nargs == 0, not None.
    'log_video_features': (['--log-video-features'], 0, True, False),
    'log_knobs': (['--log-knobs'], 0, True, False),
    'record_to_file': (
        ['-r', '--record-to-file'],
        None,
        None,
        'theremin_recording.wav',
    ),
    'no_recording': (['-n', '--no-recording'], 0, True, False),
    'window_name': (
        ['-w', '--window-name'],
        None,
        None,
        'Theremin with Hand Tracking',
    ),
    'scale': (['--scale'], '?', 'list', None),
}

# The five options whose bare form means "list the available components". This is the
# whole reason the decorators could not simply be deleted.
BARE_MEANS_LIST = ('pipeline', 'video_features', 'knobs', 'synth', 'scale')


@pytest.fixture(scope='module')
def parser():
    return cw.mk_parser(theremin_cli, config=THEREMIN_CLI_CONFIG, prog='theremin')


def test_parser_is_a_plain_argument_parser(parser):
    """cw must not hand back a subclass -- argcomplete is argparse-typed."""
    assert type(parser) is argparse.ArgumentParser


def test_every_option_keeps_its_flags_nargs_const_and_default(parser):
    actual = {
        action.dest: (
            list(action.option_strings),
            action.nargs,
            action.const,
            action.default,
        )
        for action in parser._actions
        if action.dest != 'help'
    }
    assert actual == EXPECTED_OPTIONS


def test_option_order_follows_the_signature(parser):
    assert [a.dest for a in parser._actions if a.dest != 'help'] == list(
        EXPECTED_OPTIONS
    )


@pytest.mark.parametrize('dest', BARE_MEANS_LIST)
def test_bare_flag_means_list(parser, dest):
    """`theremin --pipeline` (no value) must arrive as the 'list' sentinel."""
    long_flag = next(f for f in EXPECTED_OPTIONS[dest][0] if f.startswith('--'))
    assert getattr(parser.parse_args([long_flag]), dest) == 'list'


@pytest.mark.parametrize('dest', BARE_MEANS_LIST)
def test_flag_with_a_value_passes_the_value_through(parser, dest):
    long_flag = next(f for f in EXPECTED_OPTIONS[dest][0] if f.startswith('--'))
    assert getattr(parser.parse_args([long_flag, 'whatever']), dest) == 'whatever'


@pytest.mark.parametrize(
    'short,dest,value',
    [
        ('-p', 'pipeline', 'two_voice'),
        ('-s', 'synth', 'chorused_sine_synth'),
        ('-k', 'knobs', 'two_hand_freq_and_volume_knobs'),
        ('-v', 'video_features', 'many_video_features'),
        ('-r', 'record_to_file', 'out.wav'),
        ('-w', 'window_name', 'My Window'),
    ],
)
def test_short_flags_take_values(parser, short, dest, value):
    assert getattr(parser.parse_args([short, value]), dest) == value


@pytest.mark.parametrize(
    'short,dest', [('-p', 'pipeline'), ('-s', 'synth'), ('-k', 'knobs'), ('-v', 'video_features')]
)
def test_bare_short_flags_also_mean_list(parser, short, dest):
    assert getattr(parser.parse_args([short]), dest) == 'list'


def test_store_true_flags(parser):
    namespace = parser.parse_args(['-n', '--log-knobs', '--log-video-features'])
    assert (
        namespace.no_recording,
        namespace.log_knobs,
        namespace.log_video_features,
    ) == (True, True, True)


def test_no_arguments_gives_every_signature_default(parser):
    namespace = vars(parser.parse_args([]))
    assert {
        dest: namespace[dest] for dest in EXPECTED_OPTIONS
    } == {dest: expected[3] for dest, expected in EXPECTED_OPTIONS.items()}


@pytest.mark.parametrize(
    'argv', [['--nope'], ['-x'], ['--record-to-file'], ['--window-name'], ['stray']]
)
def test_usage_errors_still_exit_2(parser, argv, capsys):
    """`cw.dispatch` RETURNS the code argh raised, so main.py re-raises it. A console
    script that starts exiting 0 on a bad command line breaks every caller checking $?."""
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(argv)
    assert excinfo.value.code == 2


def test_help_renders_the_defaults_argh_style(parser):
    """cw.ARGH's formatter renders `%(default)s` as repr, and None as `-`."""
    help_text = parser.format_help()
    assert "(default: 'theremin')" in help_text
    assert '(default: -)' in help_text  # --scale, whose default is None
    assert '--synth [SYNTH], -s [SYNTH]' in help_text
    assert '-p [PIPELINE], --pipeline [PIPELINE]' in help_text


def test_the_config_carries_every_declaration_the_decorators_did():
    """A config key naming no parameter is a startup error in cw, but a *missing* key
    is silent -- so the count is pinned here. Ten @argh.arg decorators, ten entries."""
    from inspect import signature

    assert len(THEREMIN_CLI_CONFIG) == 10
    assert set(THEREMIN_CLI_CONFIG) == set(signature(theremin_cli).parameters)
    assert [
        key for key, leaf in THEREMIN_CLI_CONFIG.items() if leaf.get('const') == 'list'
    ] == ['pipeline', 'synth', 'knobs', 'video_features', 'scale']
