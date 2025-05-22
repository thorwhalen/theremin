"""
Theremin application using hand tracking for audio control.

This version uses the refactored modular architecture with:
- video_features.py - Hand gesture recognition and feature extraction
- audio.py - Audio synthesis and knob functions
- display.py - Screen drawing and visualization
- script_utils.py - Main function and script utilities

For a command-line interface, see the theremin_cli.py script.
"""

from theremin.script_utils import run_theremin, print_plus_newline

# -------------------------------------------------------------------------------
# Main script
# -------------------------------------------------------------------------------

if __name__ == "__main__":
    # Run with default settings:
    # - video_features: "many_video_features"
    # - audio_features: "theremin_knobs"
    # - synth_func: "theremin_synth"
    run_theremin(
        # Optional: customize the components
        # video_features="many_video_features",
        audio_features="theremin_knobs",
        synth_func="theremin_synth",
        # Logging options
        log_video_features=None,  # Set to print_plus_newline to see hand features
        log_audio_features=None,  # Set to print_plus_newline to see audio features
        # Recording options
        save_recording="theremin_recording.wav",
        # Display options
        window_name="Theremin with Hand Tracking",
    )
