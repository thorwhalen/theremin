"""Trackpad/mouse to parameter mappings."""

from typing import Dict, Tuple


def trackpad_to_2d_position(
    trackpad_event: Dict,
    screen_width: int = 1920,
    screen_height: int = 1080,
    normalize: bool = True
) -> Dict:
    """
    Convert trackpad position to normalized 2D coordinates.

    Args:
        trackpad_event: Dictionary with trackpad event data
        screen_width: Screen width for normalization
        screen_height: Screen height for normalization
        normalize: If True, normalize to 0-1 range

    Returns:
        Dictionary with x, y coordinates (normalized or absolute)
    """
    if trackpad_event.get('action') != 'move':
        return {'x': None, 'y': None}

    x = trackpad_event['position']['x']
    y = trackpad_event['position']['y']

    if normalize:
        x_norm = x / screen_width
        y_norm = y / screen_height
        return {
            'x': max(0, min(1, x_norm)),
            'y': max(0, min(1, y_norm)),
            'timestamp': trackpad_event.get('timestamp', 0)
        }
    else:
        return {
            'x': x,
            'y': y,
            'timestamp': trackpad_event.get('timestamp', 0)
        }


def trackpad_to_audio_knobs(
    trackpad_event: Dict,
    screen_width: int = 1920,
    screen_height: int = 1080
) -> Dict:
    """
    Map trackpad position and gestures to audio synthesis parameters.

    Position controls filter cutoff and resonance.
    Scroll controls volume and pan.

    Args:
        trackpad_event: Dictionary with trackpad event data
        screen_width: Screen width for normalization
        screen_height: Screen height for normalization

    Returns:
        Dictionary with audio parameter mappings
    """
    result = {'knob_changed': False}

    if trackpad_event.get('action') == 'move':
        # Normalize position to 0-1 range
        x_norm = trackpad_event['position']['x'] / screen_width
        y_norm = trackpad_event['position']['y'] / screen_height

        # Clamp to valid range
        x_norm = max(0, min(1, x_norm))
        y_norm = max(0, min(1, y_norm))

        result.update({
            'cutoff_frequency': 200 + x_norm * 8000,  # 200-8200 Hz
            'resonance': y_norm * 0.9,  # 0-0.9
            'knob_changed': True,
            'timestamp': trackpad_event.get('timestamp', 0)
        })

    elif trackpad_event.get('action') == 'scroll':
        # Vertical scroll controls volume
        # Horizontal scroll controls pan
        result.update({
            'volume_delta': trackpad_event['delta']['dy'] * 0.01,
            'pan_delta': trackpad_event['delta']['dx'] * 0.01,
            'knob_changed': True,
            'timestamp': trackpad_event.get('timestamp', 0)
        })

    return result


def scroll_to_parameter_delta(
    trackpad_event: Dict,
    param_name: str = 'volume',
    sensitivity: float = 0.01,
    invert: bool = False
) -> Dict:
    """
    Convert scroll events to parameter delta changes.

    Args:
        trackpad_event: Dictionary with trackpad event data
        param_name: Name of parameter to modify
        sensitivity: Scaling factor for delta
        invert: If True, invert the scroll direction

    Returns:
        Dictionary with parameter delta
    """
    if trackpad_event.get('action') != 'scroll':
        return {}

    direction = trackpad_event.get('direction', 'vertical')
    delta_key = 'dy' if direction == 'vertical' else 'dx'
    raw_delta = trackpad_event['delta'][delta_key]

    if invert:
        raw_delta = -raw_delta

    delta = raw_delta * sensitivity

    return {
        f'{param_name}_delta': delta,
        'timestamp': trackpad_event.get('timestamp', 0)
    }
