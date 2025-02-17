import pyaudio
import numpy as np
import essentia.standard as es
import time
from pyo import *

# Parameters for audio capture and processing
FRAME_SIZE = 2048    # Essentia's algorithm expects this many samples per frame
HOP_SIZE   = 512
SAMPLING_RATE = 44100

# Initialize PyAudio for live microphone input
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32,
                channels=1,
                rate=SAMPLING_RATE,
                input=True,
                frames_per_buffer=HOP_SIZE)

# Start the pyo server for audio synthesis
s = Server().boot()
s.start()

# Define a simple chord synthesizer that plays a chord using sine oscillators
class ChordSynth:
    def __init__(self):
        self.oscillators = []
        self.envs = []

    def play_chord(self, frequencies, dur=0.5, amp=0.2):
        # Stop any previous oscillators
        for osc in self.oscillators:
            osc.stop()
        self.oscillators = []
        self.envs = []
        # Create oscillators for each frequency
        for freq in frequencies:
            env = Adsr(attack=0.01, decay=0.1, sustain=0.3, release=0.1, dur=dur, mul=amp)
            osc = Sine(freq=freq, mul=env).out()
            self.envs.append(env)
            self.oscillators.append(osc)
            env.play()

# Instantiate the chord synthesizer
chord_synth = ChordSynth()

# Helper functions to convert between frequency and MIDI note numbers
def freq_to_midi(freq):
    if freq <= 0:
        return None
    return int(round(69 + 12 * np.log2(freq / 440.0)))

def midi_to_freq(midi_note):
    return 440.0 * (2 ** ((midi_note - 69) / 12.0))

# For this example, we always generate a major chord:
# Major chord intervals (in semitones): root, +4 (major third), +7 (perfect fifth)
def get_major_chord(midi_root):
    return [midi_root, midi_root + 4, midi_root + 7]

# Create an Essentia pitch extractor (PredominantPitchMelodia)
pitch_extractor = es.PredominantPitchMelodia(frameSize=FRAME_SIZE, hopSize=HOP_SIZE, sampleRate=SAMPLING_RATE)

# Variables to throttle triggering of the chord generator
last_trigger_time = 0
trigger_interval = 0.5  # seconds

print("Listening with Essentia... play your instrument!")
try:
    while True:
        # Read a short chunk of audio from the microphone
        data = stream.read(HOP_SIZE, exception_on_overflow=False)
        samples = np.frombuffer(data, dtype=np.float32)

        # If we don't have enough samples for one full frame, pad with zeros.
        if len(samples) < FRAME_SIZE:
            samples = np.pad(samples, (0, FRAME_SIZE - len(samples)), mode='constant')

        # Extract pitch and confidence using Essentia's PredominantPitchMelodia
        pitch_array, confidence_array = pitch_extractor(samples)
        
        # Select the pitch with the highest confidence
        idx = np.argmax(confidence_array)
        detected_pitch = pitch_array[idx]
        detected_confidence = confidence_array[idx]
        
        # If the confidence is high and a valid pitch is detected, trigger the chord synth.
        try:
            if detected_confidence > 0.8 and detected_pitch > 50:
                midi_note = freq_to_midi(detected_pitch)
                current_time = time.time()
                if current_time - last_trigger_time > trigger_interval:
                    chord_midi = get_major_chord(midi_note)
                    chord_freqs = [midi_to_freq(n) for n in chord_midi]
                    print(f"Detected pitch: {detected_pitch:.2f} Hz (MIDI {midi_note}). Playing chord: {chord_midi}")
                    chord_synth.play_chord(chord_freqs, dur=trigger_interval, amp=0.2)
                    last_trigger_time = current_time
        except Exception as e:
            print(f"Error: {e}")
            

        time.sleep(0.001)
except KeyboardInterrupt:
    print("Exiting...")
finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
    s.stop()
    s.shutdown()