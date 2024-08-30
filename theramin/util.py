"""Utils for theramin."""

from functools import partial
import time
from importlib.resources import files

pkg_name = 'theramin'
data_files = files(pkg_name) / 'data'


def return_none(*args, **kwargs):
    """
    An empty function that returns None no matter the arguments.
    Often used as a "do nothing" general callback function.
    """
    return None


# --------------------------------------------------------------------------------------
# Constants


class HandLandmark:
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_FINGER_MCP = 5
    INDEX_FINGER_PIP = 6
    INDEX_FINGER_DIP = 7
    INDEX_FINGER_TIP = 8
    MIDDLE_FINGER_MCP = 9
    MIDDLE_FINGER_PIP = 10
    MIDDLE_FINGER_DIP = 11
    MIDDLE_FINGER_TIP = 12
    RING_FINGER_MCP = 13
    RING_FINGER_PIP = 14
    RING_FINGER_DIP = 15
    RING_FINGER_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


# --------------------------------------------------------------------------------------
# String utils


def format_milliseconds_time(timestamp):
    """Format milliseconds as a string."""
    formatted_time = time.strftime('%H:%M:%S', time.localtime(timestamp))
    milliseconds = int((timestamp % 1) * 1000)
    return f"{formatted_time}.{milliseconds:03d}"


def current_time_string_with_milliseconds():
    """Get the current time with milliseconds, as a string."""
    return format_milliseconds_time(time.time())


def format_float(value, ndigits=4):
    return f"{value:.{ndigits}f}"


def format_label_xyz(label, x, y, z, *, label_width=15, coord_width=8):
    """
    Format the label and coordinates with customizable widths.

    Args:
        label (str): The label for the coordinates (e.g., "Right Wrist:").
        x, y, z (float): The coordinates to format.
        label_width (int): The width of the label field.
        coord_width (int): The width of the coordinate fields.

    Returns:
        str: The formatted string.

    >>> format_label_xyz('Right Wrist:', 1.2, 3.4, 5.6)
    'Right Wrist:    x=     1.2 y=     3.4 z=     5.6'
    >>> format_label_xyz('Right Wrist:', 1.2, 3.4, 5.6, label_width=10, coord_width=4)
    'Right Wrist: x= 1.2 y= 3.4 z= 5.6'
    """
    return f"{label:<{label_width}} x={x:>{coord_width}} y={y:>{coord_width}} z={z:>{coord_width}}"


# --------------------------------------------------------------------------------------
# Misc



import inspect

def annotate_with(annotation_type):
    """
    Decorator to annotate a function with a specified type and store the annotation 
    in the correct scope's `__annotations__` dictionary.

    Args:
        annotation_type (Any): The type to annotate the function with.

    Returns:
        function: The original function with the annotation added.

    Examples:
        >>> @annotate_with('int')
        ... def global_func():
        ...     pass
        ...
        >>> global_func()
        >>> __annotations__['global_func']
        'int'

    """
    def decorator(func):
        try:
            # Get the frame of the caller
            frame = inspect.currentframe().f_back
            # Access the correct scope's `__annotations__`
            annotations = frame.f_globals.setdefault('__annotations__', {})
            
            # Store the annotation
            annotations[func.__name__] = annotation_type
        except Exception as e:
            # Ignore this -- don't have annotations be in the way!
            print(f"Ignoring Error (but not annotating): {e}")
        return func
    
    return decorator
