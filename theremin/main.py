#!/usr/bin/env python
"""
Command-line interface for the theremin application.

This script provides a CLI wrapper around the run_theremin function, allowing all
parameters to be controlled via command-line arguments.

Examples:
    # Run with default settings (theremin_knobs + theremin_synth)
    theremin

    # List what is available for any of --pipeline/--synth/--knobs/--video-features/--scale
    # by passing the flag with no value
    theremin --synth

    # Run with two-hand knobs and two-voice synth
    theremin --knobs two_hand_freq_and_volume_knobs --synth two_voice_synth_func

    # Enable logging of extracted video features
    theremin --log-video-features

    # Save recording to a custom file
    theremin --record-to-file my_performance.wav

    # Run with chorused sine synth
    theremin --synth chorused_sine_synth --window-name "Chorused Sine Theremin"
"""

import cw

from theremin.script_utils import theremin_cli, THEREMIN_CLI_CONFIG


def dispatched_theremin_cli():
    """Run the theremin CLI, exiting with the code the command line produced.

    ``cw.dispatch`` *returns* the exit code where ``argh.dispatch_command`` raised
    ``SystemExit`` itself, so the ``raise`` here is what keeps a usage error exiting 2
    instead of 0.
    """
    raise SystemExit(cw.dispatch(theremin_cli, config=THEREMIN_CLI_CONFIG))


if __name__ == "__main__":
    dispatched_theremin_cli()
