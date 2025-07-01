"""
Tests for the restructured theremin audio features system.

This module tests the new audio feature builders, DAG-based systems, and pipelines
using the test video data.
"""

import json
import os
import pytest
from pathlib import Path
from typing import Dict, Any

# Import the new systems
from theremin.audio_features import (
    AudioFeatureBuilder,
    FeatureMapping,
    range_transformer,
    extract_nested_value,
    create_theremin_builder,
    create_two_hand_builder,
    create_enhanced_theremin_builder,
    create_fallback_theremin_builder,
)

from theremin.dag_audio_features import (
    wrist_x_to_freq,
    wrist_y_to_volume,
    theremin_dag_knobs,
    two_voice_dag_knobs,
)

from theremin.pipelines import (
    ALL_PIPELINES,
    validate_all_pipelines,
    get_working_pipelines,
    test_pipeline_with_video_features,
)

# Import video processing to generate test data
from theremin.video_features import HandGestureRecognizer, many_video_features
import cv2

# Test data paths
TEST_DATA_DIR = Path(__file__).parent / "testing_data"
TEST_VIDEO_PATH = TEST_DATA_DIR / "theremin_test_1.mp4"
TEST_FEATURES_PATH = TEST_DATA_DIR / "theremin_test_1__video_features.json"


# --------------------------------------------------------------------------------------
# Test data utilities
# --------------------------------------------------------------------------------------


def extract_features_from_frame(frame, recognizer):
    """Extract video features from a single frame"""
    hand_detection = recognizer.find_hands(frame)
    return many_video_features(hand_detection)


def generate_video_features_from_test_video():
    """Generate video features from the test video and save to JSON"""
    if not TEST_VIDEO_PATH.exists():
        print(f"Test video not found at {TEST_VIDEO_PATH}")
        return None

    print(f"Processing test video: {TEST_VIDEO_PATH}")

    # Initialize hand gesture recognizer
    recognizer = HandGestureRecognizer()

    # Open video
    cap = cv2.VideoCapture(str(TEST_VIDEO_PATH))
    if not cap.isOpened():
        print(f"Could not open video: {TEST_VIDEO_PATH}")
        return None

    all_features = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Extract features from frame
        try:
            features = extract_features_from_frame(frame, recognizer)
            all_features.append({'frame': frame_count, 'features': features})
        except Exception as e:
            print(f"Error processing frame {frame_count}: {e}")
            all_features.append({'frame': frame_count, 'features': {}})

        frame_count += 1

        # Process only first 100 frames for testing
        if frame_count >= 100:
            break

    cap.release()

    # Save features to JSON
    TEST_DATA_DIR.mkdir(exist_ok=True)
    with open(TEST_FEATURES_PATH, 'w') as f:
        json.dump(all_features, f, indent=2)

    print(f"Saved {len(all_features)} frames of features to {TEST_FEATURES_PATH}")
    return all_features


def load_test_video_features():
    """Load video features from JSON file, generating if needed"""
    if not TEST_FEATURES_PATH.exists():
        print("Video features not found, generating from test video...")
        features = generate_video_features_from_test_video()
        if features is None:
            return None

    with open(TEST_FEATURES_PATH, 'r') as f:
        return json.load(f)


def get_sample_video_features() -> Dict[str, Any]:
    """Get a representative sample of video features for testing"""
    all_features = load_test_video_features()
    if all_features:
        # Find a frame with both hands detected
        for frame_data in all_features:
            features = frame_data['features']
            if (
                features
                and 'l_wrist_position' in features
                and 'r_wrist_position' in features
            ):
                return features

        # If no two-hand frame found, return first non-empty frame
        for frame_data in all_features:
            if frame_data['features']:
                return frame_data['features']

    # Return mock features if no video data available
    print("Using mock video features for testing")
    return {
        'l_wrist_position': [0.3, 0.7],  # Left side, high up
        'r_wrist_position': [0.7, 0.3],  # Right side, low down
        'l_openness': 0.8,
        'r_openness': 0.6,
        'l_thumb_index_distance': 0.05,  # Pinching
        'r_thumb_index_distance': 0.2,  # Open
    }


# --------------------------------------------------------------------------------------
# Unit tests for core components
# --------------------------------------------------------------------------------------


def test_range_transformer():
    """Test the range transformation function"""
    # Basic range mapping
    transform = range_transformer(input_range=(0, 1), output_range=(100, 200))

    assert transform(0.0) == 100.0
    assert transform(0.5) == 150.0
    assert transform(1.0) == 200.0

    # With pre-transform (inversion)
    invert_transform = range_transformer(
        input_range=(0, 1), output_range=(100, 200), pre_transform=lambda x: 1 - x
    )

    assert invert_transform(0.0) == 200.0  # 1-0=1 -> 200
    assert invert_transform(1.0) == 100.0  # 1-1=0 -> 100

    # Handle None values
    assert transform(None) == 150.0  # Should return midpoint


def test_extract_nested_value():
    """Test nested value extraction"""
    data = {'r_wrist_position': [0.5, 0.3], 'nested': {'value': 42}}

    assert extract_nested_value(data, 'r_wrist_position.0') == 0.5
    assert extract_nested_value(data, 'r_wrist_position.1') == 0.3
    assert extract_nested_value(data, 'nested.value') == 42


def test_feature_mapping():
    """Test basic feature mapping"""
    transform = range_transformer((0, 1), (440, 880))
    mapping = FeatureMapping("freq", "r_wrist_position.0", transform, 660)

    assert mapping.audio_param == "freq"
    assert mapping.video_feature == "r_wrist_position.0"
    assert mapping.default == 660


def test_audio_feature_builder():
    """Test audio feature builder"""
    freq_transform = range_transformer((0, 1), (440, 880))
    vol_transform = range_transformer((0, 1), (0, 1), pre_transform=lambda x: 1 - x)

    builder = AudioFeatureBuilder(
        [
            FeatureMapping("freq", "r_wrist_position.0", freq_transform, 660),
            FeatureMapping("volume", "l_wrist_position.1", vol_transform, 0.0),
        ]
    )

    video_features = {'r_wrist_position': [0.5, 0.3], 'l_wrist_position': [0.2, 0.7]}

    audio_features = builder(video_features)

    print(f"Debug: audio_features = {audio_features}")

    assert 'freq' in audio_features
    assert 'volume' in audio_features

    # Test the specific transformations
    expected_freq = 440 + 0.5 * (880 - 440)  # 0.5 -> 660
    expected_volume = 1 - 0.7  # 1 - 0.7 = 0.3

    assert abs(audio_features['freq'] - expected_freq) < 0.001
    assert abs(audio_features['volume'] - expected_volume) < 0.001


# --------------------------------------------------------------------------------------
# Tests with real video data
# --------------------------------------------------------------------------------------


def test_with_sample_video_features():
    """Test audio feature extraction with sample video features"""
    sample_features = get_sample_video_features()
    print(f"Using sample features: {sample_features}")

    # Test theremin builder
    theremin_builder = create_theremin_builder()
    audio_features = theremin_builder(sample_features)

    assert 'freq' in audio_features
    assert 'volume' in audio_features
    assert isinstance(audio_features['freq'], (int, float))
    assert isinstance(audio_features['volume'], (int, float))
    assert 0 <= audio_features['volume'] <= 1


def test_dag_functions_with_video_data():
    """Test DAG-based functions with video data"""
    sample_features = get_sample_video_features()
    print(f"Testing DAG functions with: {sample_features}")

    # Test individual DAG functions
    if 'r_wrist_position' in sample_features:
        freq = wrist_x_to_freq(sample_features['r_wrist_position'])
        assert isinstance(freq, (int, float))
        assert 220 <= freq <= 1760

    if 'l_wrist_position' in sample_features:
        volume = wrist_y_to_volume(sample_features['l_wrist_position'])
        assert isinstance(volume, float)
        assert 0 <= volume <= 1

    # Test combined DAG knobs
    theremin_features = theremin_dag_knobs(sample_features)
    assert 'freq' in theremin_features
    assert 'volume' in theremin_features


# --------------------------------------------------------------------------------------
# Pipeline validation tests
# --------------------------------------------------------------------------------------


def test_pipeline_validation():
    """Test that pipelines validate correctly"""
    validation_results = validate_all_pipelines()

    # Print validation results for debugging
    for name, issues in validation_results.items():
        if issues:
            print(f"Pipeline '{name}' has issues: {issues}")

    # At least some pipelines should pass validation
    working_pipelines = get_working_pipelines()
    print(f"Working pipelines: {list(working_pipelines.keys())}")

    # Check that at least simple pipelines work
    simple_working = [
        name
        for name in working_pipelines.keys()
        if name in ['simple_sine', 'square', 'two_voice']
    ]
    assert (
        len(simple_working) > 0
    ), f"No simple pipelines working. Working: {list(working_pipelines.keys())}"


def test_pipeline_execution():
    """Test actual pipeline execution"""
    sample_features = get_sample_video_features()
    print(f"Testing pipeline execution with: {sample_features}")

    # Test a few key pipelines
    pipelines_to_test = ['theremin', 'simple_sine', 'square']

    for pipeline_name in pipelines_to_test:
        if pipeline_name in ALL_PIPELINES:
            result = test_pipeline_with_video_features(pipeline_name, sample_features)

            print(f"Testing pipeline '{pipeline_name}':")
            print(f"  Success: {result['success']}")
            if result['audio_features']:
                print(f"  Audio features: {result['audio_features']}")
            if result.get('error'):
                print(f"  Error: {result['error']}")

            # At minimum, should extract some features
            assert result['audio_features'] is not None


# --------------------------------------------------------------------------------------
# Integration tests
# --------------------------------------------------------------------------------------


def test_fallback_behavior():
    """Test single-hand fallback behavior"""
    # Test with only right hand
    right_hand_only = {'r_wrist_position': [0.6, 0.4]}

    fallback_builder = create_fallback_theremin_builder()
    audio_features = fallback_builder(right_hand_only)

    assert 'freq' in audio_features
    assert 'volume' in audio_features
    assert audio_features['volume'] > 0  # Should use right hand Y for volume

    # Test with only left hand
    left_hand_only = {'l_wrist_position': [0.3, 0.8]}

    audio_features = fallback_builder(left_hand_only)
    assert 'freq' in audio_features
    assert 'volume' in audio_features


def test_two_hand_builder():
    """Test two-hand independent control"""
    both_hands = {'l_wrist_position': [0.2, 0.7], 'r_wrist_position': [0.8, 0.3]}

    two_hand_builder = create_two_hand_builder()
    audio_features = two_hand_builder(both_hands)

    expected_params = ['l_freq', 'l_volume', 'r_freq', 'r_volume']
    for param in expected_params:
        assert param in audio_features
        assert isinstance(audio_features[param], (int, float))


# --------------------------------------------------------------------------------------
# Performance and edge case tests
# --------------------------------------------------------------------------------------


def test_empty_video_features():
    """Test behavior with empty or None video features"""
    builders = [
        create_theremin_builder(),
        create_two_hand_builder(),
        create_enhanced_theremin_builder(),
    ]

    for builder in builders:
        # Empty dict
        audio_features = builder({})
        assert isinstance(audio_features, dict)

        # None values
        none_features = {'l_wrist_position': None, 'r_wrist_position': None}
        audio_features = builder(none_features)
        assert isinstance(audio_features, dict)


def test_malformed_video_features():
    """Test behavior with malformed video features"""
    builder = create_theremin_builder()

    # Missing required fields should use defaults
    malformed = {'some_other_field': 42}
    audio_features = builder(malformed)
    assert isinstance(audio_features, dict)

    # Wrong data types should use defaults
    wrong_types = {'r_wrist_position': 'not_a_list'}
    audio_features = builder(wrong_types)
    assert isinstance(audio_features, dict)


# --------------------------------------------------------------------------------------
# Run tests if script is executed directly
# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing restructured theremin audio features system...")

    # Generate test data if needed
    if not TEST_FEATURES_PATH.exists():
        print("Generating video features from test video...")
        generate_video_features_from_test_video()

    # Run basic tests
    print("\n1. Testing range transformer...")
    test_range_transformer()
    print("✓ Range transformer tests passed")

    print("\n2. Testing feature extraction...")
    test_extract_nested_value()
    test_audio_feature_builder()
    print("✓ Feature extraction tests passed")

    print("\n3. Testing with video data...")
    try:
        test_with_sample_video_features()
        test_dag_functions_with_video_data()
        print("✓ Video data tests passed")
    except Exception as e:
        print(f"✗ Video data tests failed: {e}")

    print("\n4. Testing pipeline validation...")
    test_pipeline_validation()
    print("✓ Pipeline validation tests passed")

    print("\n5. Testing pipeline execution...")
    try:
        test_pipeline_execution()
        print("✓ Pipeline execution tests passed")
    except Exception as e:
        print(f"✗ Pipeline execution tests failed: {e}")

    print("\n6. Testing edge cases...")
    test_fallback_behavior()
    test_two_hand_builder()
    test_empty_video_features()
    test_malformed_video_features()
    print("✓ Edge case tests passed")

    print("\n🎉 All tests completed!")

    # Show pipeline capabilities
    print("\n📊 Pipeline Capabilities Summary:")
    working_pipelines = get_working_pipelines()
    print(f"Working pipelines: {list(working_pipelines.keys())}")

    validation_results = validate_all_pipelines()
    broken_pipelines = {
        name: issues for name, issues in validation_results.items() if issues
    }
    if broken_pipelines:
        print(f"Pipelines with issues: {list(broken_pipelines.keys())}")
        for name, issues in broken_pipelines.items():
            print(f"  {name}: {issues}")
