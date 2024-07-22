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
            draw (bool): Whether to draw landmarks on the image.

        Returns:
            img: The image with hand landmarks drawn (if draw=True).
        """
        # Convert the image to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Process the image and find hands
        self.results = self.hands.process(img_rgb)

        # Draw hand landmarks if found
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(
                        img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )
        return img

    def recognize_gesture(self, img):
        """
        Recognizes gestures in the provided image.

        Args:
            img: The input image.

        Returns:
            results: The gesture recognition results.
        """
        # Convert the image to RGB and create a MediaPipe image object
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(img_rgb)
        )
        # Get the current timestamp in microseconds
        timestamp = int(cv2.getTickCount() / cv2.getTickFrequency() * 1e6)
        # Recognize gestures in the image
        results = self.gesture_recognizer.recognize_for_video(mp_image, timestamp)
        return results


def print_gesture_info(results):
    """
    Callback function to print gesture recognition information.

    Args:
        results: The gesture recognition results.
    """
    if results and results.gestures:
        for gesture_list in results.gestures:
            for gesture in gesture_list:
                print(f"Gesture: {gesture.category_name}, Score: {gesture.score}")


def run(callback=print_gesture_info):
    """
    Captures video from the webcam and detects hand gestures.

    Args:
        callback: A function to handle the gesture recognition results.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture device.")
        return

    detector = HandGestureRecognizer()
    while True:
        success, img = cap.read()
        if not success:
            print("Warning: Failed to capture image.")
            continue

        # Detect hands and recognize gestures
        img = detector.find_hands(img)
        results = detector.recognize_gesture(img)
        callback(results)

        # Display the image with landmarks
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
