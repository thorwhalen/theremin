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

    def hands_info(self, img):
        # Convert the image to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Process the image and find hands
        self.results = self.hands.process(img_rgb)
        return self.results

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


# def _draw_lines_to_wrist(img, wrist_coords):
#     """
#     Draws vertical and horizontal lines from the edges of the frame to the wrist location.

#     Args:
#         img: The input image.
#         wrist_coords: The wrist coordinates (landmark[0]).
#     """
#     height, width, _ = img.shape
#     wrist_x, wrist_y = int(wrist_coords.x * width), int(wrist_coords.y * height)

#     # Draw a vertical line from the top to the wrist location
#     cv2.line(img, (wrist_x, 0), (wrist_x, height), (0, 255, 0), 2)

#     # Draw a horizontal line from the left to the wrist location
#     cv2.line(img, (0, wrist_y), (width, wrist_y), (0, 255, 0), 2)


# def draw_lines_to_wrist(img, wrist_coords):
#         if self.results.multi_hand_landmarks:
#             for hand_landmarks in self.results.multi_hand_landmarks:
#                 if draw:
#                     self.mp_draw.draw_landmarks(
#                         img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
#                     )
#         return img
    


from meshed import Slabs
from meshed.slabs import output_none_if_none_arguments


def create_slabs_instance(cap, detector, callback):
    def video_frame():
        success, img = cap.read()
        if not success:
            print("Warning: Failed to capture image.")
            return None
        img = cv2.flip(img, 1)
        return img

    @output_none_if_none_arguments
    def frame_w_hands(video_frame):
        return detector.find_hands(video_frame)

    @output_none_if_none_arguments
    def gestures(frame_w_hands):
        return detector.recognize_gesture(frame_w_hands)

    def annotated_image(gestures, frame_w_hands):
        if gestures is not None:
            callback(gestures, frame_w_hands)
        return frame_w_hands

    def display_image(annotated_image):
        if annotated_image is not None:
            cv2.imshow("Image", annotated_image)
        return annotated_image

    def check_quit():
        if cv2.waitKey(1) & 0xFF == ord('q'):
            raise KeyboardInterrupt()

    handle_exceptions = {
        KeyboardInterrupt: lambda: print("Keyboard interrupt detected. Exiting...")
    }

    components = [
        video_frame,
        frame_w_hands,
        gestures,
        annotated_image,
        display_image,
        check_quit,
    ]
    return Slabs.from_func_nodes(
        components,
        handle_exceptions=handle_exceptions,
    )


def print_gesture_info(
    gestures_data,
    img,
    *,
    text_template="Gesture: {gesture.category_name}, Score: {gesture.score:.2f}",
    print_to_terminal=True,
    font_scale=1,
    color=(0, 0, 255),
    thickness=2,
    font=cv2.FONT_HERSHEY_SIMPLEX,
):
    """
    Callback function to print gesture recognition information and overlay on image.

    Args:
        results: The gesture recognition results.
        img: The input image to overlay the text.
        text_template (str): Template for the text to overlay on the image.
        print_to_terminal (bool): Whether to print the information to the terminal.
        font_scale (float): Font scale for the overlay text.
        color (tuple): Color of the overlay text (BGR format).
        thickness (int): Thickness of the overlay text.
        font: Font type for the overlay text.
    """
    if gestures_data and gestures_data.gestures:
        for gesture_list in gestures_data.gestures:
            for gesture in gesture_list:
                text = text_template.format(gesture=gesture)
                if print_to_terminal:
                    print(text)
                # Overlay the text on the image
                cv2.putText(
                    img, text, (10, 30), font, font_scale, color, thickness, cv2.LINE_AA
                )


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

    try:
        detector = HandGestureRecognizer()
        slabs = create_slabs_instance(cap, detector, callback)
        slabs.run()
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
