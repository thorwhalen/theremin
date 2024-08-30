import cv2
import mediapipe as mp
import numpy as np
import pyaudio
from collections import deque
import time
from theramin.util import data_files, format_dict_values

# Path to the gesture recognizer model
gesture_recognizer_path = str(data_files / 'gesture_recognizer.task')


class AllZerosDict(dict):
    """A dict that only returns 0.0 for all keys."""

    def __init__(self, *args, **kwargs):
        assert args == () and kwargs == {}

    def __getitem__(self, key):
        return 0.0


all_zeros_dict = AllZerosDict()


def simple_fv_to_wave(fv, time_indices):
    vol = fv['volume']
    freq = fv['frequency']
    return vol * np.sin(2 * np.pi * freq * time_indices).astype(np.float32)


class HandGestureRecognizer:
    """
    A class to detect hand gestures using MediaPipe's GestureRecognizer.

    Attributes:
        mode (bool): Mode for the hand detection.
        max_hands (int): Maximum number of hands to detect.
        detection_con (float): Minimum detection confidence threshold.
        track_con (float): Minimum tracking confidence threshold.
    """

    def __init__(
        self,
        mode=False,
        *,
        max_hands=2,
        detection_con=0.5,
        track_con=0.5,
        gesture_recognizer_path=gesture_recognizer_path,
        fv_to_wave=simple_fv_to_wave,
        fv_fallback=all_zeros_dict,
        min_freq=220,
        max_freq=440 * 4,
        sr=44100,
    ):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.track_con = track_con
        self.fv_to_wave = fv_to_wave
        self.fv_fallback = fv_fallback

        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            self.mode,
            self.max_hands,
            min_detection_confidence=self.detection_con,
            min_tracking_confidence=self.track_con,
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Initialize Gesture Recognizer
        self.base_options = mp.tasks.BaseOptions(
            model_asset_path=gesture_recognizer_path
        )
        self.vision_running_mode = mp.tasks.vision.RunningMode

        # Set up Gesture Recognizer options
        self.options = mp.tasks.vision.GestureRecognizerOptions(
            base_options=self.base_options, running_mode=self.vision_running_mode.VIDEO
        )
        self.gesture_recognizer = mp.tasks.vision.GestureRecognizer.create_from_options(
            self.options
        )

        # Sound parameters
        self.fs = sr  # Sample rate
        self.sr = sr

        self.min_freq = min_freq
        self.max_freq = max_freq
        self.pitch_range = [self.min_freq, self.max_freq]
        self.volume = 0.5  # Default volume

        self.initial_freq = 440
        self.initial_vol = 0

        # PyAudio initialization
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.fs,
            output=True,
            stream_callback=self.audio_callback,
        )
        self.audio_bytes = b""

        self.audio_fvs_buffer = deque(maxlen=5)

    def find_hands(self, img, draw=True):
        """
        Detects hands in the provided image.

        Args:
            img: The input image.
            draw (bool): Whether to draw landmarks on the image.

        Returns:
            img: The image with hand landmarks drawn (if draw=True).
            results: The hand detection results.
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(
                        img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )
        return img, results

    def draw_wrist_lines(self, img, results):
        """
        Draws vertical and horizontal lines from the edges of the image to the wrist position.

        Args:
            img: The input image.
            results: The hand detection results.

        Returns:
            img: The image with lines drawn.
        """
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
                h, w, c = img.shape
                cx, cy = int(wrist.x * w), int(wrist.y * h)

                # Draw vertical line
                cv2.line(img, (cx, 0), (cx, h), (0, 255, 0), 2)
                # Draw horizontal line
                cv2.line(img, (0, cy), (w, cy), (0, 255, 0), 2)
        return img

    def control_sound(self, results, img_shape):
        """
        Method to control sound parameters by pushing them to the buffer.
        """
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
                audio_fvs = dict(
                    frequency=self.min_freq + wrist.x * (self.max_freq - self.min_freq),
                    volume = np.clip(1 - wrist.y, 0, 1),
                )
        else:
            audio_fvs = self.fv_fallback
        self.audio_fvs_buffer.append(audio_fvs)

    def audio_callback(self, in_data, frame_count, time_info, status, *, verbose=1):
        """
        PyAudio callback to continuously generate sound.
        """
        current_time = time_info['current_time']
        ts = (np.arange(frame_count) + self.fs * current_time) / self.fs

        # Read the latest frequency and volume from the buffer, if available
        if len(self.audio_fvs_buffer) > 0:
            fv = self.audio_fvs_buffer[-1]
        else:
            fv = self.fv_fallback

        if verbose:
            if fv:
                print(f"{format_dict_values(fv, 8)}")

        wave = self.fv_to_wave(fv, ts)

        wave_bytes = wave.tobytes()
        self.audio_bytes += wave_bytes
        return (wave_bytes, pyaudio.paContinue)


def main():
    cap = cv2.VideoCapture(0)
    recognizer = HandGestureRecognizer()

    try:
        while cap.isOpened():
            success, img = cap.read()
            if not success:
                break

            img = cv2.flip(img, 1)
            img, results = recognizer.find_hands(img)
            img = recognizer.draw_wrist_lines(img, results)
            recognizer.control_sound(results, img.shape)

            cv2.imshow('Hand Gesture Recognition with Theremin', img)

            if cv2.waitKey(5) & 0xFF == 27:
                break
    finally:
        from pathlib import Path

        Path("theramin_audio.pcm").write_bytes(recognizer.audio_bytes)

        cap.release()
        cv2.destroyAllWindows()
        recognizer.stream.stop_stream()
        recognizer.stream.close()
        recognizer.p.terminate()


if __name__ == "__main__":
    main()
