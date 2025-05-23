#!/usr/bin/env python
"""
Command-line interface for the theremin application.

This script provides a CLI wrapper around the run_theremin function, allowing all
parameters to be controlled via command-line arguments.

Examples:
    # Run with default settings (theremin_knobs + theremin_synth)
    python theremin_cli.py

    # Run with two-hand knobs and two-voice synth
    python theremin_cli.py --audio-features two_hand_freq_and_volume_knobs --synth-func two_voice_synth_func

    # Enable logging of hand features
    python theremin_cli.py --log-hand-features

    # Save recording to custom file
    python theremin_cli.py --save-recording my_performance.wav

    # Run with chorused sine synth
    python theremin_cli.py --synth-func chorused_sine_synth --window-name "Chorused Sine Theremin"
"""

import argh
from theremin.script_utils import theremin_cli

def dispatched_theremin_cli():
    argh.dispatch_command(theremin_cli)

if __name__ == "__main__":
    dispatched_theremin_cli()

    
