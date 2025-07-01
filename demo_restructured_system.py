#!/usr/bin/env python3
"""
Demo of the restructured theremin audio features system.

This script demonstrates the key improvements:
- Cleaner audio feature extraction
- Better validation
- More composable pipelines
"""

from theremin.audio_features import (
    AudioFeatureBuilder,
    FeatureMapping,
    range_transformer,
    create_theremin_builder,
    create_enhanced_theremin_builder,
    create_two_hand_builder,
)

from theremin.pipelines import (
    ALL_PIPELINES,
    validate_all_pipelines,
    get_working_pipelines,
    test_pipeline_with_video_features,
)

from theremin.dag_audio_features import theremin_dag_knobs, enhanced_theremin_dag_knobs


def demo_audio_feature_builders():
    print("🎛️  Audio Feature Builders Demo")
    print("=" * 50)

    # Sample video features (what we get from hand detection)
    video_features = {
        'r_wrist_position': [0.7, 0.3],  # Right hand: far right, low
        'l_wrist_position': [0.3, 0.8],  # Left hand: left side, high
        'r_openness': 0.6,  # Right hand partially open
        'l_openness': 0.4,  # Left hand less open
        'r_thumb_index_distance': 0.1,  # Right hand pinching
        'l_thumb_index_distance': 0.3,  # Left hand more open
    }

    print(f"Input video features: {video_features}\n")

    # Test different builders
    builders = {
        "Basic Theremin": create_theremin_builder(),
        "Enhanced Theremin": create_enhanced_theremin_builder(),
        "Two-Hand Control": create_two_hand_builder(),
    }

    for name, builder in builders.items():
        print(f"📡 {name}:")
        audio_features = builder(video_features)
        for param, value in audio_features.items():
            print(f"  {param}: {value:.3f}")
        print()


def demo_pipeline_validation():
    print("🔍 Pipeline Validation Demo")
    print("=" * 50)

    validation_results = validate_all_pipelines()
    working_pipelines = get_working_pipelines()

    print(f"✅ Working pipelines ({len(working_pipelines)}):")
    for name in working_pipelines.keys():
        print(f"  • {name}")

    print(
        f"\n⚠️  Pipelines with issues ({len(validation_results) - len(working_pipelines)}):"
    )
    for name, issues in validation_results.items():
        if issues and name not in working_pipelines:
            print(f"  • {name}: {issues[0]}")  # Show first issue
    print()


def demo_range_transformers():
    print("📏 Range Transformers Demo")
    print("=" * 50)

    # Create various transformers
    freq_transform = range_transformer((0, 1), (220, 1760))
    inverted_volume = range_transformer((0, 1), (0, 1), pre_transform=lambda x: 1 - x)
    vibrato_rate = range_transformer((0, 1), (1, 20))

    test_values = [0.0, 0.25, 0.5, 0.75, 1.0]

    print("Input -> Frequency (220-1760 Hz):")
    for val in test_values:
        print(f"  {val} -> {freq_transform(val):.1f} Hz")

    print("\nInput -> Inverted Volume (high Y = loud):")
    for val in test_values:
        print(f"  {val} -> {inverted_volume(val):.2f}")

    print("\nInput -> Vibrato Rate (1-20 Hz):")
    for val in test_values:
        print(f"  {val} -> {vibrato_rate(val):.1f} Hz")
    print()


def demo_custom_pipeline():
    print("🔧 Custom Pipeline Demo")
    print("=" * 50)

    from theremin.audio import fm_synth
    from theremin.pipelines import AudioPipeline

    # Create a custom FM synthesis pipeline
    fm_builder = AudioFeatureBuilder(
        [
            FeatureMapping(
                "freq", "r_wrist_position.0", range_transformer((0, 1), (220, 880))
            ),
            FeatureMapping(
                "volume",
                "l_wrist_position.1",
                range_transformer((0, 1), (0, 0.8), pre_transform=lambda y: 1 - y),
            ),
            FeatureMapping(
                "mod_index", "r_openness", range_transformer((0, 1), (0, 5))
            ),
            FeatureMapping(
                "carrier_ratio", "l_openness", range_transformer((0, 1), (0.5, 2.0))
            ),
            FeatureMapping(
                "mod_freq_ratio",
                "r_thumb_index_distance",
                range_transformer((0, 1), (0.5, 3.0)),
            ),
        ]
    )

    fm_pipeline = AudioPipeline("custom_fm", fm_builder, fm_synth)

    # Validate the custom pipeline
    issues = fm_pipeline.validate()
    print(f"Custom FM Pipeline validation: {'✅ PASS' if not issues else '❌ FAIL'}")
    if issues:
        for issue in issues:
            print(f"  Issue: {issue}")

    # Test the pipeline
    test_features = {
        'r_wrist_position': [0.6, 0.4],
        'l_wrist_position': [0.3, 0.7],
        'r_openness': 0.8,
        'l_openness': 0.6,
        'r_thumb_index_distance': 0.2,
    }

    if not issues:
        audio_features = fm_pipeline.audio_features(test_features)
        print(f"Audio features generated: {audio_features}")
    print()


def demo_dag_comparison():
    print("🕸️  DAG vs Builder Comparison")
    print("=" * 50)

    test_features = {
        'r_wrist_position': [0.5, 0.6],
        'l_wrist_position': [0.4, 0.3],
        'r_openness': 0.7,
    }

    # AudioFeatureBuilder approach
    builder = create_enhanced_theremin_builder()
    builder_result = builder(test_features)

    # DAG approach
    try:
        dag_result = enhanced_theremin_dag_knobs(test_features)
        if isinstance(dag_result, dict):
            print("AudioFeatureBuilder result:")
            for param, value in sorted(builder_result.items()):
                print(f"  {param}: {value:.3f}")

            print("\nDAG result:")
            for param, value in sorted(dag_result.items()):
                print(f"  {param}: {value:.3f}")

            # Compare
            print(f"\nResults match: {builder_result == dag_result}")
        else:
            print(f"DAG result has unexpected type: {type(dag_result)}")
    except Exception as e:
        print(f"DAG approach failed: {e}")
        print("This is expected since DAG validation is still being refined.")

    print()


if __name__ == "__main__":
    print("🎵 Restructured Theremin Audio Features Demo")
    print("=" * 60)
    print()

    demo_range_transformers()
    demo_audio_feature_builders()
    demo_pipeline_validation()
    demo_custom_pipeline()
    demo_dag_comparison()

    print("🎉 Demo completed! The restructured system provides:")
    print("  • Clearer intent through explicit mappings")
    print("  • Better testability with isolated components")
    print("  • Validation to catch issues early")
    print("  • Easy composition of new pipelines")
    print("  • Both simple (AudioFeatureBuilder) and complex (DAG) approaches")
