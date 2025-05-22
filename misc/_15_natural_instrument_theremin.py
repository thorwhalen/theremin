"""
Natural Instrument Theremin - using the natural_sounding_synth function

This script runs a theremin that simulates instrument-like sounds using the
natural_sounding_synth function, which creates harmonically rich timbres
resembling real instruments.

Controls:
- Right hand X position: Controls frequency (pitch)
- Left hand Y position: Controls volume
- Use the INSTRUMENT variable below to change the instrument type

Available instruments:
- 'violin': Harmonically rich string sound
- 'organ': Full, sustained organ-like timbre
- 'flute': Softer, airier woodwind-like sound
"""

from theremin.script_utils import (
    run_theremin,
    print_plus_newline,
    print_json_if_possible,
)

# Change this to 'violin', 'organ', or 'flute' to switch instruments
INSTRUMENT = 'violin'

# Dictionary with synth configuration - instrument type and reverb amount
SYNTH_SETTINGS = {
    'instrument': INSTRUMENT,
    'reverb_mix': 0.3,  # Adjust between 0.0 (dry) and 1.0 (wet) for more/less reverb
    'vibrato_rate': 5,  # Speed of vibrato in Hz
    'vibrato_depth': 5,  # Depth of vibrato in Hz
}


def _specific_video_features_log(x):
    """
    Log specific video features to the console.
    """
    print("Hand landmarks:", x['hand_landmarks'])
    print("Hand angles:", x['hand_angles'])
    print("Hand distances:", x['hand_distances'])
    print("Hand velocities:", x['hand_velocities'])


synth_func, audio_features = "natural_sounding_synth", "theremin_knobs"
synth_func, audio_features = (
    "natural_sounding_synth_lr",
    "two_hand_freq_and_volume_knobs",
)
synth_func, audio_features = "complex_fm_synth", "complex_fm_synth_knobs"

# synth_func, audio_features = 'two_hand_freq_and_volume_knobs'

if __name__ == "__main__":
    run_theremin(
        # Use the natural_sounding_synth with our settings
        synth_func=synth_func,
        # Use default theremin control scheme (right hand=pitch, left hand=volume)
        audio_features=audio_features,
        # # Pass our instrument and effect settings
        # synth_settings=SYNTH_SETTINGS,  # TODO: Integrate
        # Enable to see what features are being extracted:
        log_video_features=print_json_if_possible,
        log_audio_features=print_json_if_possible,
        # Save the recording to a custom file:
        # save_recording=f"theremin_{INSTRUMENT}_recording.wav",
        # Custom window title
        window_name=f"Natural Instrument Theremin ({INSTRUMENT.title()})",
    )
