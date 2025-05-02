"""
Theremin application using hand tracking for audio control.

This version uses the refactored modular architecture with:
- hand_features.py - Hand gesture recognition and feature extraction
- audio.py - Audio synthesis and knob functions
- display.py - Screen drawing and visualization
- script_utils.py - Main function and script utilities
"""

from theramin.script_utils import run_theremin, print_plus_newline

# -------------------------------------------------------------------------------
# Main script
# -------------------------------------------------------------------------------

if __name__ == "__main__":
    # Run with default settings:
    # - hand_features: "many_hand_features"
    # - audio_features: "theremin_knobs"
    # - synth_func: "theremin_synth"
    run_theremin(
        # Optional: customize the components
        # hand_features="many_hand_features",
        audio_features="theremin_knobs",
        synth_func="theremin_synth",
        # Logging options
        log_hand_features=None,  # Set to print_plus_newline to see hand features
        log_audio_features=None,  # Set to print_plus_newline to see audio features
        # Recording options
        save_recording="theremin_recording.wav",
        # Display options
        window_name="Theremin with Hand Tracking",
    )
