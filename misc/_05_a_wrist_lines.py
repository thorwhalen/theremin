import cv2
import mediapipe as mp
import numpy as np
from theramin.util import data_files

# Path to the gesture recognizer model
gesture_recognizer_path = str(data_files / 'gesture_recognizer.task')

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

    def find_hands(self, img, draw=True):
        """
        Detects hands in the provided image.

        Args:
            img: The input image.
            draw (bool): Whether to draw landmarks and wrist lines.
        
        Returns:
            img: The image with detected hand landmarks.
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    self.draw_lines_to_wrist(img, hand_landmarks.landmark[0])  # Draw lines to the wrist

        return img

    def draw_lines_to_wrist(self, img, wrist_coords):
        """
        Draws vertical and horizontal lines from the edges of the frame to the wrist location.

        Args:
            img: The input image.
            wrist_coords: The wrist coordinates (landmark[0]).
        """
        height, width, _ = img.shape
        wrist_x, wrist_y = int(wrist_coords.x * width), int(wrist_coords.y * height)

        # Draw a vertical line from the top to the wrist location
        cv2.line(img, (wrist_x, 0), (wrist_x, height), (0, 255, 0), 2)

        # Draw a horizontal line from the left to the wrist location
        cv2.line(img, (0, wrist_y), (width, wrist_y), (0, 255, 0), 2)


# Assuming there's a main function or similar that captures video frames and calls find_hands
def main():
    recognizer = HandGestureRecognizer()

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame = recognizer.find_hands(frame, draw=True)

        cv2.imshow('Hand Gesture with Wrist Lines', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()