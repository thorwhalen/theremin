"""Display utilities for theremin visualization."""

import cv2
import numpy as np
from typing import Union, Tuple, Dict, Optional, Callable, Iterable

# -------------------------------------------------------------------------------
# Types
# -------------------------------------------------------------------------------

Color = Union[Tuple[int, int, int], Tuple[int, int, int, int]]  # BGR or BGRA

# -------------------------------------------------------------------------------
# Screen drawing functions
# -------------------------------------------------------------------------------


def display_sound_features_on_image(
    img: np.ndarray,
    sound_features: dict,
    *,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale: float = 1.0,
    color: Color = (0, 255, 0),
    thickness: float = 2,
    float_format: str = ".2f",
    x_pos=10,
    y_pos=30,
    y_increment=30,
    bg_color: Color = (
        150,
        150,
        150,
        128,
    ),  # Light grey, semi-transparent (BGR + alpha)
):
    """
    Display sound features on the image with a semi-transparent background.

    Args:
        img: The image to draw on
        sound_features: Dictionary of sound features
        font: Font type to use
        font_scale: Size of the font
        color: Text color in BGR format
        thickness: Line thickness of text
        float_format: Format string for float values
        x_pos: Starting x position for text
        y_pos: Starting y position for text
        y_increment: Vertical space between lines
        bg_color: Background color (BGR + alpha) where alpha is 0-255
    """
    if not sound_features:
        return img

    # Create an overlay for the background
    overlay = img.copy()

    # Process bg_color to separate BGR and alpha
    if len(bg_color) == 4:
        bg_rgb = bg_color[:3]
        alpha = bg_color[3] / 255.0  # Convert to 0-1 range
    else:
        bg_rgb = bg_color
        alpha = 0.5  # Default alpha

    # Draw background rectangles and text
    for idx, (key, value) in enumerate(sound_features.items()):
        # Format the value based on its type
        if isinstance(value, float):
            formatted_value = f"{value:{float_format}}"
        else:
            formatted_value = str(value)

        text = f"{key}: {formatted_value}"

        # Get text size for background rectangle
        (text_width, text_height), _ = cv2.getTextSize(
            text, font, font_scale, thickness
        )

        # Draw filled rectangle on the overlay
        padding = 5  # Padding around the text
        cv2.rectangle(
            overlay,
            (x_pos - padding, y_pos + idx * y_increment - text_height - padding),
            (x_pos + text_width + padding, y_pos + idx * y_increment + padding),
            bg_rgb,
            -1,  # Filled rectangle
        )

    # Apply the overlay with transparency
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # Draw text on top
    for idx, (key, value) in enumerate(sound_features.items()):
        if isinstance(value, float):
            formatted_value = f"{value:{float_format}}"
        else:
            formatted_value = str(value)

        text = f"{key}: {formatted_value}"
        cv2.putText(
            img,
            text,
            (x_pos, y_pos + idx * y_increment),
            font,
            font_scale,
            color,
            thickness,
        )

    return img


def draw_hand_landmarks(img, hand_detection, mp_hands, mp_draw):
    """
    Draw hand landmarks on the image.

    Args:
        img: The image to draw on
        hand_detection: MediaPipe hand detection results
        mp_hands: MediaPipe hands module
        mp_draw: MediaPipe drawing utilities

    Returns:
        img: The image with landmarks drawn
    """
    if hand_detection.multi_hand_landmarks:
        for hand_landmarks in hand_detection.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    return img


def draw_wrist_lines(img, hand_detection, mp_hands):
    """
    Draws vertical and horizontal lines from the edges of the image to the wrist position.

    Args:
        img: The input image.
        hand_detection: The hand detection results.
        mp_hands: MediaPipe hands module

    Returns:
        img: The image with lines drawn.
    """
    if hand_detection.multi_hand_landmarks:
        for hand_landmarks in hand_detection.multi_hand_landmarks:
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            h, w, c = img.shape
            cx, cy = int(wrist.x * w), int(wrist.y * h)

            # Draw vertical line
            cv2.line(img, (cx, 0), (cx, h), (0, 255, 0), 2)
            # Draw horizontal line
            cv2.line(img, (0, cy), (w, cy), (0, 255, 0), 2)
    return img


def draw_on_screen(
    recognizer,
    img: np.ndarray,
    hand_detection,
    sound_features: Optional[dict] = None,
    *,
    draw_landmarks: bool = True,
    draw_sound_features: Optional[Callable] = display_sound_features_on_image,
    draw_frequencies: Optional[Iterable] = None,
):
    """
    Draw hand landmarks, wrist lines, and sound features on the image.

    Args:
        recognizer: HandGestureRecognizer instance
        img: The input image
        hand_detection: Hand detection results
        sound_features: Sound parameter values to display
        draw_landmarks: Whether to draw hand landmarks
        draw_sound_features: Function to draw sound features (or None to skip)
        draw_frequencies: Iterable of frequencies to display as scale points on screen

    Returns:
        img: The image with visualizations added
    """
    # Draw hand landmarks if requested
    if draw_landmarks and hand_detection.multi_hand_landmarks:
        img = draw_hand_landmarks(
            img, hand_detection, recognizer.mp_hands, recognizer.mp_draw
        )

    # Draw wrist lines
    img = draw_wrist_lines(img, hand_detection, recognizer.mp_hands)

    # Display sound features
    if draw_sound_features and sound_features:
        img = draw_sound_features(img, sound_features)

    # Draw frequency scale points
    if draw_frequencies:
        h, w, _ = img.shape
        vertical_center = h // 2

        # Calculate the range for mapping frequencies
        from theremin.audio import DFLT_MIN_FREQ, DFLT_MAX_FREQ

        min_freq = DFLT_MIN_FREQ
        max_freq = DFLT_MAX_FREQ

        # Draw each frequency as a point
        for freq in draw_frequencies:
            # Map frequency to x-coordinate (0 to 1, then scaled to image width)
            # Use the same mapping logic as in the audio functions
            norm_x = (freq - min_freq) / (max_freq - min_freq)
            x_pos = int(norm_x * w)

            # Skip if outside the image bounds
            if 0 <= x_pos < w:
                # Draw a visible point
                point_size = 5
                point_color = (0, 0, 255)  # Red in BGR
                cv2.circle(img, (x_pos, vertical_center), point_size, point_color, -1)

                # Add small frequency label below the point
                label = f"{int(freq)}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.4
                text_color = (0, 200, 200)  # Yellow-green in BGR
                thickness = 1

                # Get text size for better positioning
                (text_width, text_height), _ = cv2.getTextSize(
                    label, font, font_scale, thickness
                )

                # Position text centered below the point
                text_x = x_pos - text_width // 2
                text_y = vertical_center + 20

                cv2.putText(
                    img,
                    label,
                    (text_x, text_y),
                    font,
                    font_scale,
                    text_color,
                    thickness,
                )

    return img
