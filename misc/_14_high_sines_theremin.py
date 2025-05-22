"""
High Sines Theremin - using the intro_high_sines synth function

This script runs a theremin that uses the intro_high_sines synth function,
which creates ethereal high-frequency sounds with modulation.

Controls:
- Right hand X position: Controls base frequency
- Right hand Y position: Controls modulation frequency
- Left hand Y position: Controls modulation amplitude

Alternative control schemes are available by changing the audio_features parameter.
"""

from theremin.script_utils import run_theremin, print_plus_newline

if __name__ == "__main__":
    run_theremin(
        # Use the intro_high_sines synth with our custom knob function
        synth_func="intro_high_sines",
        audio_features="high_sines_theremin_knobs",  # <-- Default mapping
        # Uncomment one of these to try alternative mappings:
        # audio_features="high_sines_pinch_theremin_knobs",  # Uses pinch gesture
        # audio_features="high_sines_openness_theremin_knobs",  # Uses hand openness
        # Enable to see what features are being extracted:
        # log_video_features=print_plus_newline,
        # log_audio_features=print_plus_newline,
        # Save the recording to a custom file:
        save_recording="high_sines_theremin_recording.wav",
        # Custom window title
        window_name="High Sines Theremin",
    )
