"""
Follows both hands landmarks (21 points) and draws lines to enphasize the wrist location.
Also prints some hand landmarks coordinates.


In this script, I set things up so that one can add several callbacks that will do 
something based on the stream of hand feature vectors created by the streams of 
images from the live video. In this particular instance, I used three callbacks:
* draw_hand_landmarks: The standard 21 points of the hand
* draw_lines_to_wrist: Some "guide" lines to the wrists, which will constitute a reference point to the hand
* log_hand_coordinates: The coordinates of the wrist and the tip of the index

For information on hand landmarks, see:
* https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
* https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/hands.md

"""

from functools import partial
import time
from typing import Callable, Sequence

import cv2
import mediapipe as mp
import numpy as np
from theramin.util import (
    data_files,
    current_time_string_with_milliseconds,
    return_none,
    annotate_with,
)

# Path to the gesture recognizer model
gesture_recognizer_path = str(data_files / 'gesture_recognizer.task')


def log_current_time_string_with_milliseconds(log_func=print):
    log_func(current_time_string_with_milliseconds())


# self, results, img, hand_landmarks, idx

Results = mp.solutions.hands.Hands
Img = np.ndarray
# HandLandmarks = mp.solutions.hands.HandLandmarks
HandLandmarks = "HandLandmarks"

HandLandmarkCallback = Callable[
    ["HandGestureRecognizer", Results, Img, HandLandmarks, int],
    None,
]
NoArgsNoReturnCallback = Callable[[], None]


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
        max_hands=2,
        detection_con=0.5,
        track_con=0.5,
        gesture_recognizer_path=gesture_recognizer_path,
    ):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.track_con = track_con

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

    def find_hands(
        self,
        img,
        callbacks: Sequence[HandLandmarkCallback] = (),
        *,
        result_callback: NoArgsNoReturnCallback = log_current_time_string_with_milliseconds,
        no_result_callback: NoArgsNoReturnCallback = return_none,
    ):
        """
        Detects hands in the provided image.

        Args:
            img: The input image.
            draw (bool): Whether to draw landmarks and wrist lines.

        Returns:
            img: The image with detected hand landmarks.

        For information on hand landmarks, see:
        https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker

        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        if results.multi_hand_landmarks:
            result_callback()

            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                for callback in callbacks:
                    callback(self, results, img, hand_landmarks, idx)
        else:
            no_result_callback()

        return img


# --------------------------------------------------------------------------------------
# Paramerizing a HandGestureRecognizer session

from theramin.util import (
    data_files,
    HandLandmarkIndex,
    format_float,
    current_time_string_with_milliseconds,
    format_label_xyz,
)


label_xyz = partial(format_label_xyz, label_width=15, coord_width=7)


@annotate_with(HandLandmarkCallback)
def draw_hand_landmarks(self, results, img, hand_landmarks, idx):
    mp.solutions.drawing_utils.draw_landmarks(
        img, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS
    )


def _draw_lines_to_coords(img, coords):
    """
    Draws vertical and horizontal lines from the edges of the frame to the coords.

    Args:
        img: The input image.
        coords: (x, y) coordinates (i.e. has x and y attributes).

    Note: Modifies img in place.
    """
    height, width, _ = img.shape
    wrist_x, wrist_y = int(coords.x * width), int(coords.y * height)

    # Draw a vertical line from the top to the wrist location
    cv2.line(img, (wrist_x, 0), (wrist_x, height), (0, 255, 0), 2)

    # Draw a horizontal line from the left to the wrist location
    cv2.line(img, (0, wrist_y), (width, wrist_y), (0, 255, 0), 2)


@annotate_with(HandLandmarkCallback)
def draw_lines_to_wrist(self, results, img, hand_landmarks, idx):
    """
    Draws vertical and horizontal lines from the edges of the frame to the wrist location.

    Args:
        img: The input image.
        wrist_coords: The wrist coordinates (landmark[0]).
    """
    wrist_coords = hand_landmarks.landmark[HandLandmarkIndex.WRIST]
    _draw_lines_to_coords(img, wrist_coords)


@annotate_with(HandLandmarkCallback)
def log_hand_coordinates(
    self, results, img, hand_landmarks, idx, *, ndigits=4, log_func=print
):
    handedness = results.multi_handedness[idx].classification[0].label
    # Create a partial function that formats numbers with 4 decimal places
    r = partial(format_float, ndigits=ndigits)

    # Now use this partial function in your code
    wrist = hand_landmarks.landmark[HandLandmarkIndex.WRIST]
    index_tip = hand_landmarks.landmark[HandLandmarkIndex.INDEX_FINGER_TIP]

    # Use the formatting function in your message
    msg = f"{handedness} " + label_xyz(f"Wrist:", r(wrist.x), r(wrist.y), r(wrist.z))
    msg += "\t" + label_xyz(
        "Index Tip:", r(index_tip.x), r(index_tip.y), r(index_tip.z)
    )

    log_func(msg)


# Assuming there's a main function or similar that captures video frames and calls find_hands
def main():

    handlandmark_callbacks = [
        draw_hand_landmarks,
        draw_lines_to_wrist,
        log_hand_coordinates,
    ]

    recognizer = HandGestureRecognizer()

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame = recognizer.find_hands(frame, handlandmark_callbacks)

        cv2.imshow('Hand Gesture with Wrist Lines', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
