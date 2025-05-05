#!/usr/bin/env python
"""
Command-line interface for the theremin application.

This script provides a CLI wrapper around the run_theramin function, allowing all
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

import sys
import argh
from theramin.script_utils import run_theramin, print_json_if_possible
from typing import Optional


def main(
    # Core components
    video_features: str = "many_video_features",
    audio_features: str = "theremin_knobs",
    synth_func: str = "theremin_synth",
    # Logging options
    log_video_features: bool = False,
    log_audio_features: bool = False,
    # Recording options
    save_recording: str = "theremin_recording.wav",
    no_recording: bool = False,
    # Display options
    window_name: str = "Theremin with Hand Tracking",
    # List available components
    list_synths: bool = False,
    list_audio_features: bool = False,
    list_video_features: bool = False,
):
    """
    Run the theremin application with the specified parameters.

    Args:
        video_features: Name of the hand feature extraction function
        audio_features: Name of the audio feature mapping function
        synth_func: Name of the synthesizer function
        log_video_features: Whether to log hand features
        log_audio_features: Whether to log audio features
        save_recording: Filename to save recording (if no_recording is False)
        no_recording: Disable recording
        window_name: Title for the display window
        list_synths: List available synthesizer functions and exit
        list_audio_features: List available audio feature mapping functions and exit
        list_video_features: List available hand feature extraction functions and exit
    """
    # Import here to avoid loading everything if just listing components
    from theramin.audio import synth_funcs, audio_feature_funcs
    from theramin.video_features import hand_feature_funcs

    # Handle listing available components
    if list_synths:
        print("Available synthesizer functions:")
        for name in sorted(synth_funcs.keys()):
            print(f"  - {name}")
        return

    if list_audio_features:
        print("Available audio feature mapping functions:")
        for name in sorted(audio_feature_funcs.keys()):
            print(f"  - {name}")
        return

    if list_video_features:
        print("Available hand feature extraction functions:")
        for name in sorted(hand_feature_funcs.keys()):
            print(f"  - {name}")
        return

    # Handle recording options
    if no_recording:
        save_recording = False

    # Set up logging callbacks
    log_video_features_callback = print_json_if_possible if log_video_features else None
    log_audio_features_callback = print_json_if_possible if log_audio_features else None

    # Run the theremin application
    run_theramin(
        video_features=video_features,
        audio_features=audio_features,
        synth_func=synth_func,
        log_video_features=log_video_features_callback,
        log_audio_features=log_audio_features_callback,
        save_recording=save_recording,
        window_name=window_name,
    )


if __name__ == "__main__":
    argh.dispatch_command(main)
