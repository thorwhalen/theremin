import contextlib
import time
from collections import deque
import numpy as np
import pyaudio
import essentia.standard as es
from pyo import *

# Helper functions for frequency and MIDI conversion
def freq_to_midi(freq):
    if freq <= 0:
        return None
    return int(round(69 + 12 * np.log2(freq / 440.0)))

def midi_to_freq(midi_note):
    return 440.0 * (2 ** ((midi_note - 69) / 12.0))

# Generate a major chord from a MIDI root note
def get_major_chord(midi_root):
    return [midi_root, midi_root + 4, midi_root + 7]

# Chord synthesizer using pyo sine oscillators
class ChordSynth:
    def __init__(self):
        self.oscillators = []  # Sine oscillator objects
        self.envs = []         # Envelope objects

    def play_chord(self, frequencies, dur=0.5, amp=0.2):
        # Stop any previous oscillators
        for osc in self.oscillators:
            osc.stop()
        self.oscillators = []
        self.envs = []
        # Create an oscillator and envelope for each note
        for freq in frequencies:
            env = Adsr(attack=0.01, decay=0.1, sustain=0.3, release=0.1, dur=dur, mul=amp)
            osc = Sine(freq=freq, mul=env).out()
            self.envs.append(env)
            self.oscillators.append(osc)
            env.play()

# Smoothing function for pitch estimates
def get_smoothed_pitch(pitch_buffer, conf_buffer, method):
    if method == "max":
        idx = np.argmax(conf_buffer)
        return pitch_buffer[idx], conf_buffer[idx]
    elif method == "weighted":
        total_conf = np.sum(conf_buffer)
        if total_conf == 0:
            return np.mean(pitch_buffer), 0
        weighted_pitch = np.sum(np.array(pitch_buffer) * np.array(conf_buffer)) / total_conf
        return weighted_pitch, total_conf / len(conf_buffer)
    elif method == "mean":
        return np.mean(pitch_buffer), np.mean(conf_buffer)
    else:
        idx = np.argmax(conf_buffer)
        return pitch_buffer[idx], conf_buffer[idx]

# chord callback function that generates and plays a major chord
def major_chord_callback(smoothed_pitch, smoothed_conf, chord_synth, trigger_interval):
    midi_note = freq_to_midi(smoothed_pitch)
    chord_midi = get_major_chord(midi_note)
    chord_freqs = [midi_to_freq(n) for n in chord_midi]
    print(
        f"-----> Detected pitch: {smoothed_pitch:.2f} Hz (MIDI {midi_note}) with confidence {smoothed_conf:.2f}. "
        f"Playing chord: {chord_midi}"
    )
    print(chord_freqs)
    chord_synth.play_chord(chord_freqs, dur=trigger_interval, amp=0.2)


def fifths_chord_callback(smoothed_pitch, smoothed_conf, chord_synth, trigger_interval):
    midi_note = freq_to_midi(smoothed_pitch)
    chord_midi = [midi_note, midi_note + 7, midi_note + 14]
    chord_freqs = [midi_to_freq(n) for n in chord_midi]
    print(
        f"-----> Detected pitch: {smoothed_pitch:.2f} Hz (MIDI {midi_note}) with confidence {smoothed_conf:.2f}. "
        f"Playing chord: {chord_midi}"
    )
    print(chord_freqs)
    chord_synth.play_chord(chord_freqs, dur=trigger_interval, amp=0.2)

default_chord_callback = fifths_chord_callback



# Context manager for live accompaniment with configurable parameters and chord callback
@contextlib.contextmanager
def live_accompaniment(
    frame_size,
    hop_size,
    *,
    sampling_rate=44100,
    trigger_interval=0.5,
    smoothing_method="max",  # Options: "max", "weighted", "mean"
    smoothing_window=3,
    conf_threshold=0.7,
    pitch_threshold=50,
    pitch_extractor_cls=es.PitchYinProbabilistic,
    chord_callback=default_chord_callback
):
    # Set up PyAudio for audio input
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=sampling_rate,
        input=True,
        frames_per_buffer=hop_size,
    )
    # Set up and start the pyo server for synthesis
    s = Server().boot()
    s.start()
    # Instantiate the pitch extractor using the provided class
    pitch_extractor = pitch_extractor_cls(frameSize=frame_size, hopSize=hop_size, sampleRate=sampling_rate)
    # Instantiate the chord synthesizer
    chord_synth = ChordSynth()
    
    # Use deques with a fixed maximum length for smoothing buffers
    pitch_buffer = deque(maxlen=smoothing_window)
    conf_buffer = deque(maxlen=smoothing_window)

    def run_loop():
        nonlocal pitch_buffer, conf_buffer
        last_trigger_time = 0
        print("Listening... play your instrument!")
        try:
            while True:
                # Read a chunk from the microphone
                data = stream.read(hop_size, exception_on_overflow=False)
                samples = np.frombuffer(data, dtype=np.float32)
                # Pad samples if necessary
                if len(samples) < frame_size:
                    samples = np.pad(samples, (0, frame_size - len(samples)), mode='constant')
                
                # Extract pitch and confidence arrays
                pitch_array, confidence_array = pitch_extractor(samples)
                idx = np.argmax(confidence_array)
                local_pitch = pitch_array[idx]
                local_conf = confidence_array[idx]
                
                # Append to the smoothing buffers
                pitch_buffer.append(local_pitch)
                conf_buffer.append(local_conf)
                
                # Compute smoothed pitch values
                smoothed_pitch, smoothed_conf = get_smoothed_pitch(list(pitch_buffer), list(conf_buffer), smoothing_method)
                current_time = time.time()
                print(
                    f"{current_time:.2f}: "
                    f"rms: {np.sqrt(np.mean(samples**2)):.4f} "
                    f"Detected pitch: {smoothed_pitch:.2f} Hz "
                    f"(conf {smoothed_conf:.2f})"
                )
                # Check thresholds and trigger the chord callback if conditions are met
                if (
                    smoothed_conf > conf_threshold
                    and smoothed_pitch > pitch_threshold
                    and (current_time - last_trigger_time) > trigger_interval
                ):
                    chord_callback(smoothed_pitch, smoothed_conf, chord_synth, trigger_interval)
                    last_trigger_time = current_time
                time.sleep(0.001)
        except KeyboardInterrupt:
            print("Keyboard interrupt received. Exiting loop.")

    try:
        yield run_loop
    finally:
        # Clean up: stop audio stream and synthesis server
        stream.stop_stream()
        stream.close()
        p.terminate()
        s.stop()
        s.shutdown()
        print("Cleaned up resources.")

# Example usage
if __name__ == "__main__":
    with live_accompaniment(
        frame_size=2048,
        hop_size=512,
        sampling_rate=44100,
        trigger_interval=0.5,
        smoothing_method="max",
        smoothing_window=3,
        conf_threshold=0.7,
        pitch_threshold=50,
    ) as run_loop:
        run_loop()