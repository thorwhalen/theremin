import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Union, Any

from pyo import Sine
from theramin.util import data_files, format_dict_values
from hum.pyo_util import Synth

# Path to the gesture recognizer model
gesture_recognizer_path = str(data_files / 'gesture_recognizer.task')


def simple_sine(freq=440, volume=0.5):
    """
    A simple synth function that returns a sine wave controlled by freq and volume.

    Args:
        freq (float): Frequency of the sine wave in Hz.
        volume (float): Volume of the sine wave, from 0.0 to 1.0.

    Returns:
        pyo.Sine: The sine wave audio object.
    """
    sine = Sine(freq=freq, mul=volume)
    return sine


def compute_knobs_from_results(
    results, img_shape=None, min_freq=220, max_freq=440 * 4
) -> Optional[Dict[str, float]]:
    """
    Compute frequency and volume parameters from hand tracking results.

    Args:
        results: MediaPipe hand tracking results
        img_shape: Shape of the image (height, width, channels)
        min_freq: Minimum frequency in Hz
        max_freq: Maximum frequency in Hz

    Returns:
        Dictionary with 'freq' and 'volume' parameters or None if no hands detected
    """
    if not results.multi_hand_landmarks:
        return None

    # Use the first hand detected for theremin control
    hand_landmarks = results.multi_hand_landmarks[0]
    wrist = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST]

    # Map x position to frequency (logarithmic scale would be even better)
    # Convert to native Python float
    freq = float(min_freq + wrist.x * (max_freq - min_freq))

    # Map y position to volume (inverted so higher hand = louder)
    # Convert to native Python float
    volume = float(np.clip(1 - wrist.y, 0, 1))

    return {'freq': freq, 'volume': volume}


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
        min_freq=220,
        max_freq=440 * 4,
    ):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.track_con = track_con
        self.min_freq = min_freq
        self.max_freq = max_freq

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

                # Display frequency and volume values
                freq_text = f"Freq: {self.min_freq + wrist.x * (self.max_freq - self.min_freq):.1f} Hz"
                vol_text = f"Vol: {np.clip(1 - wrist.y, 0, 1):.2f}"

                cv2.putText(
                    img,
                    freq_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    img,
                    vol_text,
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

        return img


def main():
    """Main function to run the hand gesture recognition with pyo theremin."""
    cap = cv2.VideoCapture(0)
    recognizer = HandGestureRecognizer()

    # Initialize the pyo synth
    synth = Synth(simple_sine, nchnls=2)

    with synth:
        try:
            while cap.isOpened():
                success, img = cap.read()
                if not success:
                    break

                img = cv2.flip(img, 1)
                img, results = recognizer.find_hands(img)
                img = recognizer.draw_wrist_lines(img, results)

                # Update synth parameters based on hand position
                knobs_dict = compute_knobs_from_results(
                    results,
                    img_shape=img.shape,
                    min_freq=recognizer.min_freq,
                    max_freq=recognizer.max_freq,
                )

                if knobs_dict is not None:
                    synth.knobs.update(knobs_dict)

                cv2.imshow('Hand Gesture Recognition with Theremin', img)

                if cv2.waitKey(5) & 0xFF == 27:  # ESC key to exit
                    break
        finally:
            # Save the recording
            synth.stop_recording()
            recording = synth.get_recording()

            # Print recording statistics
            print(f"Recorded {len(recording)} control events")

            # Render the recording to a WAV file
            output_path = "theremin_recording.wav"
            synth.render_recording(output_filepath=output_path)
            print(f"Saved audio recording to {output_path}")

            # Clean up resources
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
