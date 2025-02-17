import contextlib
import time
import numpy as np
import pyaudio
import essentia.standard as es
from pyo import *


# Helper functions to convert between frequency and MIDI note numbers
def freq_to_midi(freq):
    if freq <= 0:
        return None
    return int(round(69 + 12 * np.log2(freq / 440.0)))


def midi_to_freq(midi_note):
    return 440.0 * (2 ** ((midi_note - 69) / 12.0))


# For this basic example, we always generate a major chord:
# Major chord intervals (in semitones): root, +4 (major third), +7 (perfect fifth)
def get_major_chord(midi_root):
    return [midi_root, midi_root + 4, midi_root + 7]


# A simple chord synthesizer that plays a chord using sine oscillators via pyo
class ChordSynth:
    def __init__(self):
        self.oscillators = []  # Sine objects
        self.envs = []  # Envelope objects

    def play_chord(self, frequencies, dur=0.5, amp=0.2):
        # Stop any previous oscillators
        for osc in self.oscillators:
            osc.stop()
        self.oscillators = []
        self.envs = []
        # Create an oscillator and envelope for each note
        for freq in frequencies:
            env = Adsr(
                attack=0.01, decay=0.1, sustain=0.3, release=0.1, dur=dur, mul=amp
            )
            osc = Sine(freq=freq, mul=env).out()
            self.envs.append(env)
            self.oscillators.append(osc)
            env.play()


# Helper function for smoothing pitch estimates over a buffer.
def get_smoothed_pitch(pitch_buffer, conf_buffer, method):
    # pitch_buffer and conf_buffer are lists of floats of length <= smoothing_window.
    if method == "max":
        # Return the pitch corresponding to the maximum confidence in the buffer.
        idx = np.argmax(conf_buffer)
        return pitch_buffer[idx], conf_buffer[idx]
    elif method == "weighted":
        total_conf = np.sum(conf_buffer)
        if total_conf == 0:
            return np.mean(pitch_buffer), 0
        weighted_pitch = (
            np.sum(np.array(pitch_buffer) * np.array(conf_buffer)) / total_conf
        )
        return weighted_pitch, total_conf / len(conf_buffer)
    elif method == "mean":
        return np.mean(pitch_buffer), np.mean(conf_buffer)
    else:
        # Default to max if unknown method.
        idx = np.argmax(conf_buffer)
        return pitch_buffer[idx], conf_buffer[idx]


@contextlib.contextmanager
def live_accompaniment(
    frame_size,
    hop_size,
    *,
    sampling_rate=44100,
    trigger_interval=0.5,
    smoothing_method="max",  # Options: "max", "weighted", "mean"
    smoothing_window=1,
):
    """
    Context manager that sets up live audio acquisition, pitch analysis using Essentia,
    and real-time chord accompaniment synthesis using pyo.

    Parameters:
      frame_size (int): Number of samples per analysis frame.
      hop_size (int): Hop size between frames.

      sampling_rate (int, kw-only): Audio sampling rate in Hz (default 44100).
      trigger_interval (float, kw-only): Minimum time between chord triggers (default 0.5 sec).
      smoothing_method (str, kw-only): Method to smooth pitch estimates over the last N frames:
            "max" (select pitch with max confidence), "weighted" (weighted average), or "mean" (simple average).
      smoothing_window (int, kw-only): Number of frames over which to smooth (default 1 means no extra smoothing).

    Yields:
      run_loop (function): A function that, when called, runs the live processing loop until interrupted.
    """
    # Set up PyAudio
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=sampling_rate,
        input=True,
        frames_per_buffer=hop_size,
    )
    # Set up pyo server
    s = Server().boot()
    s.start()
    # Create Essentia pitch extractor
    pitch_extractor = es.PitchYinProbabilistic(
        frameSize=frame_size, hopSize=hop_size, sampleRate=sampling_rate
    )

    # For some reason can't make PredominantPitchMelodia to work
    # pitch_extractor = es.PredominantPitchMelodia(
    #     frameSize=frame_size, hopSize=hop_size, sampleRate=sampling_rate
    # )
    # pitch_extractor = es.PredominantPitchMelodia(
    #     frameSize=2048, hopSize=512, sampleRate=sampling_rate, minFrequency=40
    # )
    # pitch_extractor = es.PredominantPitchMelodia(
    #     frameSize=2048, hopSize=512, sampleRate=sampling_rate, voicingTolerance=0.1
    # )
    # pitch_extractor = es.PredominantPitchMelodia(
    #     frameSize=2048, hopSize=512, sampleRate=sampling_rate, guessUnvoiced=True
    # )

    # Instantiate chord synthesizer
    chord_synth = ChordSynth()

    # Buffers for smoothing (each iteration, we will append one value)
    pitch_buffer = []
    conf_buffer = []

    def run_loop():
        nonlocal pitch_buffer, conf_buffer
        last_trigger_time = 0
        print("Listening... play your instrument!")
        try:
            while True:
                # Read a chunk from the microphone
                data = stream.read(hop_size, exception_on_overflow=False)
                samples = np.frombuffer(data, dtype=np.float32)
                # Pad if necessary
                if len(samples) < frame_size:
                    samples = np.pad(
                        samples, (0, frame_size - len(samples)), mode='constant'
                    )

                # Get pitch and confidence arrays from Essentia
                pitch_array, confidence_array = pitch_extractor(samples)
                # Use max-confidence approach for this frame:
                idx = np.argmax(confidence_array)
                local_pitch = pitch_array[idx]
                local_conf = confidence_array[idx]
                # Append to smoothing buffers (limit length to smoothing_window)
                pitch_buffer.append(local_pitch)
                conf_buffer.append(local_conf)
                if len(pitch_buffer) > smoothing_window:
                    pitch_buffer.pop(0)
                    conf_buffer.pop(0)
                # Compute smoothed pitch
                smoothed_pitch, smoothed_conf = get_smoothed_pitch(
                    pitch_buffer, conf_buffer, smoothing_method
                )
                current_time = time.time()
                print(
                    f"{current_time:.2f}: "
                    f"rms: {np.sqrt(np.mean(samples**2))} "
                    f"Detected pitch: {smoothed_pitch:.2f} Hz "
                    f"(conf {smoothed_conf:.2f})"
                )
                if (
                    smoothed_conf > 0.7
                    and smoothed_pitch > 50
                    and (current_time - last_trigger_time) > trigger_interval
                ):
                    # Convert the smoothed pitch to a MIDI note
                    midi_note = freq_to_midi(smoothed_pitch)
                    # Generate a major chord based on this root note
                    chord_midi = get_major_chord(midi_note)
                    chord_freqs = [midi_to_freq(n) for n in chord_midi]
                    print(
                        f"-----> Detected pitch: {smoothed_pitch:.2f} Hz (MIDI {midi_note}) with confidence {smoothed_conf:.2f}. Playing chord: {chord_midi}"
                    )
                    print(chord_freqs)
                    chord_synth.play_chord(chord_freqs, dur=trigger_interval, amp=0.2)
                    last_trigger_time = current_time
                time.sleep(0.001)
        except KeyboardInterrupt:
            print("Keyboard interrupt received. Exiting loop.")

    try:
        yield run_loop
    finally:
        # Tear down: stop stream and servers
        stream.stop_stream()
        stream.close()
        p.terminate()
        s.stop()
        s.shutdown()
        print("Cleaned up resources.")


# Example usage:
if __name__ == "__main__":
    with live_accompaniment(
        2048,
        512,
        sampling_rate=44100,
        trigger_interval=0.5,
        smoothing_method="max",
        smoothing_window=3,
    ) as run_loop:
        run_loop()
