# import time
# from datetime import datetime
# from pynput import keyboard
# import multiprocessing
# import pygame.midi
# import sounddevice as sd

# # Initialize Pygame MIDI
# pygame.midi.init()
# midi_out = pygame.midi.Output(pygame.midi.get_default_output_id())

# def timestamped_print(message):
#     print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} - {message}")

# def play_note_with_pygame(note, velocity, duration):
#     timestamped_print("Sending note on message")
#     midi_out.note_on(note, velocity)
    
#     time.sleep(duration)
    
#     timestamped_print("Sending note off message")
#     midi_out.note_off(note, velocity)

# def keyboard_listener(queue):
#     def on_press(key):
#         try:
#             timestamped_print(f"Key {key.char} pressed")
#             if key.char == 'q':
#                 return False  # Stop listener
#             queue.put(('note_on', 60, 64))  # Middle C, standard velocity
#         except AttributeError:
#             timestamped_print(f"Special key {key} pressed")

#     def on_release(key):
#         timestamped_print(f"Key {key} released")
#         queue.put(('note_off', 60, 64))  # Middle C, standard velocity
#         if key == keyboard.Key.esc:
#             return False  # Stop listener

#     with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
#         listener.join()

# def midi_processor(queue):
#     while True:
#         try:
#             message = queue.get()
#             if message[0] == 'note_on':
#                 play_note_with_pygame(message[1], message[2], 0.5)  # 0.5 second duration
#             elif message[0] == 'note_off':
#                 play_note_with_pygame(message[1], message[2], 0)
#         except Exception as e:
#             timestamped_print(f"Exception in midi_processor: {e}")

# if __name__ == "__main__":
#     midi_queue = multiprocessing.Queue()

#     listener_process = multiprocessing.Process(target=keyboard_listener, args=(midi_queue,))
#     processor_process = multiprocessing.Process(target=midi_processor, args=(midi_queue,))

#     listener_process.start()
#     processor_process.start()

#     listener_process.join()
#     processor_process.join()

#     # Close the MIDI output and quit Pygame MIDI
#     midi_out.close()
#     pygame.midi.quit()

# import pygame.midi
# import time

# # Initialize Pygame MIDI
# pygame.midi.init()

# # Open the default MIDI output
# midi_out = pygame.midi.Output(pygame.midi.get_default_output_id())

# def play_note_with_pygame(note, velocity, duration):
#     print(f"Sending note on message for note {note} with velocity {velocity}")
#     midi_out.note_on(note, velocity)
    
#     time.sleep(duration)
    
#     print(f"Sending note off message for note {note}")
#     midi_out.note_off(note, velocity)

# # Test playing a note
# play_note_with_pygame(60, 64, 1.0)  # Middle C, velocity 64, duration 1 second

# # Close the MIDI output and quit Pygame MIDI
# midi_out.close()
# pygame.midi.quit()


import time
from datetime import datetime
from pynput import keyboard
import multiprocessing
from midi2audio import FluidSynth
from mido import MidiFile, MidiTrack, Message

# Initialize FluidSynth with soundfont
soundfont_path = "/Users/thorwhalen/Dropbox/Media/soundfonts/Arachno SoundFont - Version 1.0.sf2"
fluidsynth = FluidSynth(soundfont_path)

# Function to add timestamp to prints
def timestamped_print(message):
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} - {message}")

# Function to create and play a MIDI note
def play_note_with_fluidsynth(note, velocity, duration):
    timestamped_print("Creating and playing MIDI note")

    # Create a new MIDI file with a single note
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    track.append(Message('program_change', program=0))
    track.append(Message('note_on', note=note, velocity=velocity, time=0))
    track.append(Message('note_off', note=note, velocity=velocity, time=int(duration * 480)))

    # Save the MIDI file
    midi_file_path = "realtime_midi.mid"
    mid.save(midi_file_path)

    # Convert MIDI to audio and play
    fluidsynth.play_midi(midi_file_path)

# Keyboard listener function
def keyboard_listener(queue):
    def on_press(key):
        try:
            timestamped_print(f"Key {key.char} pressed")
            if key.char == 'q':
                return False  # Stop listener
            queue.put(('note_on', 60, 64, 0.5))  # Middle C, standard velocity, duration 0.5 seconds
        except AttributeError:
            timestamped_print(f"Special key {key} pressed")

    def on_release(key):
        timestamped_print(f"Key {key} released")
        if key == keyboard.Key.esc:
            return False  # Stop listener

    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

# MIDI processor function
def midi_processor(queue):
    while True:
        try:
            message = queue.get()
            if message[0] == 'note_on':
                play_note_with_fluidsynth(message[1], message[2], message[3])
        except Exception as e:
            timestamped_print(f"Exception in midi_processor: {e}")

if __name__ == "__main__":
    midi_queue = multiprocessing.Queue()

    listener_process = multiprocessing.Process(target=keyboard_listener, args=(midi_queue,))
    processor_process = multiprocessing.Process(target=midi_processor, args=(midi_queue,))

    listener_process.start()
    processor_process.start()

    listener_process.join()
    processor_process.join()
