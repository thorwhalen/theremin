"""Utils for theremin."""

from functools import partial
import time
from importlib.resources import files
from typing import Any
import numbers

try:  # optional numpy import for isinstance checks
    import numpy as _np  # type: ignore
except Exception:  # pragma: no cover - numpy may not be present in some envs
    _np = None

pkg_name = 'theremin'
data_files = files(pkg_name) / 'data'


# --------------------------------------------------------------------------------------
# Placeholders


def return_none(*args, **kwargs):
    """
    An empty function that returns None no matter the arguments.
    Often used as a "do nothing" general callback function.
    """
    return None


class AllZerosDict(dict):
    """A dict that only returns 0.0 for all keys.

    >>> all_zeros_dict = AllZerosDict()
    >>> all_zeros_dict['anything']
    0.0

    """

    def __init__(self, *args, **kwargs):
        assert args == () and kwargs == {}

    def __getitem__(self, key):
        return 0.0


all_zeros_dict = AllZerosDict()

# --------------------------------------------------------------------------------------
# Constants


class HandLandmarkIndex:
    # TODO: Take directly from mp.solutions.hands.HandLandmark
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


def format_dict_values(input_dict, max_digits=8):
    """
    Formats all numerical values in a dictionary to strings with a specified number of significant digits.

    Args:
        input_dict (dict): The dictionary whose numerical values will be formatted.
        max_digits (int): The maximum number of significant digits to use in the formatting.

    Returns:
        dict: A new dictionary with the same keys as input_dict, but with numerical values replaced by formatted strings.

    Example:
        >>> original_dict = {'a': 123456789, 'b': 1234.56789, 'c': 12.345789, 'd': 0.0012345789}
        >>> format_dict_values(original_dict, 5)
        {'a': '1.2346e+08', 'b': '1234.6', 'c': '12.346', 'd': '0.0012346'}

    Note:
        Non-numeric values are left unchanged.
    """

    def format_value(value):
        if isinstance(value, (int, float)):
            return f"{value:.{max_digits}g}"
        return value

    formatted_dict = {k: format_value(v) for k, v in input_dict.items()}
    return formatted_dict


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

from i2 import partialx, Sig as Signature


def obfuscate_args(func, keep_args):
    """
    Creates a new function with only the specified arguments,
    with other arguments fixed to their defaults.
    """

    func_sig = Signature(func)
    if not all([arg in keep_args for arg in func_sig.names[: len(keep_args)]]):
        raise ValueError("keep_args must be in the beginning of Sig(foo).names")
    defaults_of_other_args = {
        k: v for k, v in func_sig.defaults.items() if k not in keep_args
    }
    return partialx(func, **defaults_of_other_args, _rm_partialize=True)


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


import json
from typing import Union, Dict, Any
from collections.abc import Iterator
from pathlib import Path


def json_lines(string_or_path_to_string: str | Path) -> Iterator[dict[str, Any]]:
    """
    Parse a file or string containing JSON-like dictionaries on each line.
    Yields only the lines that can be deserialized to dictionaries.

    Parameters:
        string_or_path_to_string: A file path or the content as a string

    Yields:
        dict: Each successfully parsed dictionary

    """
    # Determine if we're dealing with a file path or string content
    try:
        path = Path(string_or_path_to_string)
        if path.exists():
            content = path.read_text()
        else:
            content = string_or_path_to_string
    except:
        content = string_or_path_to_string

    # Process line by line
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue

        try:
            # Try to parse as JSON
            data = json.loads(line)
            # Only yield if it's a dictionary
            if isinstance(data, dict):
                yield data
        except json.JSONDecodeError:
            # Skip non-JSON lines
            continue


# --------------------------------------------------------------------------------------
# Type sanitation helpers
# --------------------------------------------------------------------------------------


def to_builtin_number(x: Any) -> Any:
    """Convert numpy numeric scalars to builtin numbers; leave others unchanged.

    - numpy floating -> float
    - numpy integer -> int
    - python numbers.Number -> unchanged
    - numpy arrays -> x.tolist() if 0-d array then recurse to scalar
    - other types -> unchanged
    """
    # Handle numpy scalar types when numpy is available
    if _np is not None:
        if isinstance(x, (_np.floating,)):
            return float(x)
        if isinstance(x, (_np.integer,)):
            return int(x)
        # 0-dim arrays: treat as scalars
        if isinstance(x, _np.ndarray):
            if x.ndim == 0:
                return to_builtin_number(x.item())
            # For non-scalar arrays, convert to list (non-audio inputs)
            return x.tolist()
    # For plain python numeric types, return as-is
    if isinstance(x, numbers.Number):
        return x
    return x


def ensure_plain_types(obj: Any) -> Any:
    """Recursively convert numpy numbers/arrays in common containers to builtin types.

    Supports dicts, lists, tuples, sets; leaves other types unchanged.
    """
    if isinstance(obj, dict):
        return {k: ensure_plain_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [ensure_plain_types(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(ensure_plain_types(v) for v in obj)
    if isinstance(obj, set):
        return {ensure_plain_types(v) for v in obj}
    return to_builtin_number(obj)
