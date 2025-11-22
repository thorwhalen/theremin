"""Tests for trackpad stream functionality."""

import pytest
from streamtouch.trackpad_stream import TrackpadFeatureStream, TrackpadEvent
from streamtouch.mappings import (
    trackpad_to_audio_knobs,
    trackpad_to_2d_position,
    scroll_to_parameter_delta
)


def test_trackpad_event_creation():
    """Test TrackpadEvent dataclass."""
    event = TrackpadEvent(
        action='move',
        position={'x': 100, 'y': 200},
        timestamp=123.456
    )

    assert event.type == 'trackpad'
    assert event.action == 'move'
    assert event.position['x'] == 100
    assert event.position['y'] == 200
    assert event.timestamp == 123.456


def test_trackpad_event_scroll():
    """Test scroll event creation."""
    event = TrackpadEvent(
        action='scroll',
        position={'x': 500, 'y': 600},
        delta={'dx': 0, 'dy': 5},
        direction='vertical',
        magnitude=5.0,
        timestamp=123.456
    )

    assert event.action == 'scroll'
    assert event.direction == 'vertical'
    assert event.magnitude == 5.0
    assert event.delta['dy'] == 5


def test_trackpad_to_2d_position_normalized():
    """Test trackpad position normalization."""
    event = {
        'action': 'move',
        'position': {'x': 960, 'y': 540},  # Center of 1920x1080
        'timestamp': 123.456
    }

    pos = trackpad_to_2d_position(event, 1920, 1080, normalize=True)

    assert abs(pos['x'] - 0.5) < 0.01  # Center X
    assert abs(pos['y'] - 0.5) < 0.01  # Center Y
    assert pos['timestamp'] == 123.456


def test_trackpad_to_2d_position_absolute():
    """Test trackpad absolute position."""
    event = {
        'action': 'move',
        'position': {'x': 100, 'y': 200},
        'timestamp': 123.456
    }

    pos = trackpad_to_2d_position(event, 1920, 1080, normalize=False)

    assert pos['x'] == 100
    assert pos['y'] == 200


def test_trackpad_to_2d_position_non_move():
    """Test that non-move events return None."""
    event = {
        'action': 'scroll',
        'position': {'x': 100, 'y': 200},
        'timestamp': 123.456
    }

    pos = trackpad_to_2d_position(event)

    assert pos['x'] is None
    assert pos['y'] is None


def test_trackpad_to_audio_knobs_move():
    """Test trackpad position to audio knobs mapping."""
    event = {
        'action': 'move',
        'position': {'x': 1920, 'y': 0},  # Top right corner
        'timestamp': 123.456
    }

    knobs = trackpad_to_audio_knobs(event, 1920, 1080)

    assert knobs['knob_changed'] is True
    assert knobs['cutoff_frequency'] > 8000  # Near max
    assert knobs['resonance'] < 0.1  # Near min


def test_trackpad_to_audio_knobs_scroll():
    """Test trackpad scroll to audio knobs mapping."""
    # Vertical scroll
    scroll_event = {
        'action': 'scroll',
        'position': {'x': 960, 'y': 540},
        'delta': {'dx': 0, 'dy': 5},
        'direction': 'vertical',
        'magnitude': 5.0,
        'timestamp': 123.456
    }

    knobs = trackpad_to_audio_knobs(scroll_event)

    assert knobs['knob_changed'] is True
    assert 'volume_delta' in knobs
    assert knobs['volume_delta'] > 0  # Scrolling up increases volume

    # Horizontal scroll
    h_scroll_event = {
        'action': 'scroll',
        'position': {'x': 960, 'y': 540},
        'delta': {'dx': 3, 'dy': 0},
        'direction': 'horizontal',
        'timestamp': 123.456
    }

    knobs = trackpad_to_audio_knobs(h_scroll_event)

    assert 'pan_delta' in knobs
    assert knobs['pan_delta'] > 0


def test_scroll_to_parameter_delta():
    """Test scroll to parameter delta conversion."""
    event = {
        'action': 'scroll',
        'position': {'x': 500, 'y': 600},
        'delta': {'dx': 0, 'dy': 10},
        'direction': 'vertical',
        'timestamp': 123.456
    }

    # Default (volume)
    delta = scroll_to_parameter_delta(event)
    assert 'volume_delta' in delta
    assert delta['volume_delta'] == 10 * 0.01  # Default sensitivity

    # Custom parameter
    delta = scroll_to_parameter_delta(event, param_name='brightness', sensitivity=0.02)
    assert 'brightness_delta' in delta
    assert delta['brightness_delta'] == 10 * 0.02

    # Inverted
    delta = scroll_to_parameter_delta(event, invert=True)
    assert delta['volume_delta'] < 0  # Inverted sign


def test_scroll_to_parameter_delta_non_scroll():
    """Test that non-scroll events return empty dict."""
    event = {
        'action': 'move',
        'position': {'x': 100, 'y': 200},
        'timestamp': 123.456
    }

    delta = scroll_to_parameter_delta(event)
    assert delta == {}


def test_trackpad_feature_stream_creation():
    """Test TrackpadFeatureStream instantiation."""
    stream = TrackpadFeatureStream()

    assert stream.event_queue is not None
    assert stream.current_position == {'x': 0, 'y': 0}


def test_get_features_empty_queue():
    """Test getting features from empty queue."""
    stream = TrackpadFeatureStream()

    features = stream.get_features()
    assert features is None


def test_simulated_trackpad_events():
    """Test simulating trackpad events without actual listener."""
    stream = TrackpadFeatureStream()

    # Simulate a move event
    event = TrackpadEvent(
        action='move',
        position={'x': 100, 'y': 200},
        timestamp=123.456
    )
    stream.event_queue.put(event)

    # Retrieve the event
    features = stream.get_features()

    assert features is not None
    assert features['action'] == 'move'
    assert features['position']['x'] == 100
    assert features['position']['y'] == 200


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
