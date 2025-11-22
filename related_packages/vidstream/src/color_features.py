"""Color tracking and dominant color extraction."""

import time
from typing import Dict, List, Optional, Tuple


class ColorFeatureExtractor:
    """
    Extract dominant colors and color-based features from video frames.

    Uses OpenCV for HSV color space conversion and tracking.
    Returns clean dict outputs with color percentages and dominant colors.

    Example:
        >>> extractor = ColorFeatureExtractor()
        >>> features = extractor.extract(frame)
        >>> print(f"Dominant color: {features['dominant_color']}")
    """

    # Default color ranges in HSV
    DEFAULT_COLOR_RANGES = {
        'red': ([0, 100, 100], [10, 255, 255]),
        'red2': ([170, 100, 100], [180, 255, 255]),  # Red wraps around
        'blue': ([100, 100, 100], [130, 255, 255]),
        'green': ([40, 50, 50], [80, 255, 255]),
        'yellow': ([20, 100, 100], [30, 255, 255]),
        'purple': ([130, 50, 50], ([160, 255, 255])),
        'orange': ([10, 100, 100], [20, 255, 255]),
    }

    def __init__(self, color_ranges: Optional[Dict] = None):
        """
        Initialize color feature extractor.

        Args:
            color_ranges: Optional custom color ranges dict
                         Format: {'color_name': ([h_min, s_min, v_min], [h_max, s_max, v_max])}
        """
        self.color_ranges = color_ranges or self.DEFAULT_COLOR_RANGES
        self._cv2_available = False

        # Try to import OpenCV and numpy
        try:
            import cv2
            import numpy as np
            self.cv2 = cv2
            self.np = np
            self._cv2_available = True
        except ImportError:
            pass

    def extract(self, frame, timestamp: Optional[float] = None) -> Dict:
        """
        Extract color features from a video frame.

        Args:
            frame: Video frame (BGR format, numpy array)
            timestamp: Optional timestamp for the frame

        Returns:
            Dictionary with color features

        Raises:
            RuntimeError: If OpenCV is not available
        """
        if not self._cv2_available:
            raise RuntimeError(
                "OpenCV is not installed. Install with: pip install opencv-python"
            )

        if timestamp is None:
            timestamp = time.time()

        # Convert to HSV
        hsv = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2HSV)

        color_percentages = {}
        total_pixels = frame.shape[0] * frame.shape[1]

        # Calculate percentage for each color
        for color_name, (lower, upper) in self.color_ranges.items():
            mask = self.cv2.inRange(
                hsv,
                self.np.array(lower),
                self.np.array(upper)
            )
            pixel_count = self.np.count_nonzero(mask)
            percentage = pixel_count / total_pixels
            color_percentages[color_name] = percentage

        # Handle red wrapping (combine red and red2 if present)
        if 'red' in color_percentages and 'red2' in color_percentages:
            color_percentages['red'] += color_percentages.pop('red2')

        # Find dominant color
        if color_percentages:
            dominant_color = max(color_percentages, key=color_percentages.get)
            dominant_percentage = color_percentages[dominant_color]
        else:
            dominant_color = 'none'
            dominant_percentage = 0.0

        return {
            'dominant_color': dominant_color,
            'dominant_percentage': dominant_percentage,
            'color_percentages': color_percentages,
            'timestamp': timestamp
        }


def extract_dominant_colors(
    frame,
    color_ranges: Optional[Dict] = None,
    timestamp: Optional[float] = None
) -> Dict:
    """
    Convenience function to extract dominant colors from a frame.

    Args:
        frame: Video frame (BGR format)
        color_ranges: Optional custom color ranges
        timestamp: Optional timestamp

    Returns:
        Dictionary with color features
    """
    extractor = ColorFeatureExtractor(color_ranges=color_ranges)
    return extractor.extract(frame, timestamp)
