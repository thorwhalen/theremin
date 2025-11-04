# Theremin Architecture

## Overview

The **theremin** project is a modular framework for mapping sensor data streams to generated audio streams. While the ultimate vision is to support any sensor-to-audio mapping, the current implementation focuses on video (hand gestures) and keyboard inputs mapped to synthesized audio.

### The Big Picture

At its heart, theremin implements a pipeline architecture with distinct, composable layers:

1. **Sensor Reading**: Capture raw input from sensors (currently: video via MediaPipe, keyboard via OpenCV)
2. **Feature Extraction**: Transform raw sensor data into meaningful features (hand positions, gestures, openness, etc.)
3. **Feature Mapping** ("Knobs"): Map extracted features to audio synthesis parameters
4. **Synthesis**: Generate audio using the mapped parameters

This layered approach provides modularity: you can swap out components at any layer independently. Want a different synthesizer? Change the synth function. Want different gesture-to-sound mappings? Modify the feature mapping. The pipeline composition ties these components together.

### Current Implementation: Synth-Based Architecture

The current "synth-based" implementation follows this flow:

```
Input: Video/Keyboard
         |
         v
+----------------------------------------------------------+
| 1. Sensor Reading (cv2.VideoCapture, cv2.waitKey)       |
+----------------------------------------------------------+
         |
         v
+----------------------------------------------------------+
| 2. Feature Extraction: --video-features                 |
|    (e.g. many_video_features)                            |
|    Extracts: hand positions, gestures, openness, etc.    |
+----------------------------------------------------------+
         |
         v
+----------------------------------------------------------+
| 3. Feature Mapping: --knobs (e.g. theremin_knobs)       |
|    Maps video features → audio params (freq, volume...) |
+----------------------------------------------------------+
         |
         v
+----------------------------------------------------------+
| 4. Synthesis: --synth (e.g. theremin_synth)             |
|    Generates audio from parameters                       |
+----------------------------------------------------------+
         |
         v
   [Audio Output + Recording]

Note: --pipeline combines steps 2-4 in pre-configured packages
(e.g. "theremin", "two_voice", "simple_sine")
```

**Sensor Readers** capture frames from a camera or key presses. The **video_features** module (using MediaPipe) extracts hand landmarks, positions, gestures, and derived features like "openness" (how spread the fingers are) or pinch detection.

These video features feed into **audio feature builders** (also called "knobs functions" historically), which map specific video features to audio synthesis parameters. For example:
- Right wrist X position → frequency
- Left wrist Y position → volume  
- Hand openness → vibrato depth

Finally, the **synthesizer** receives these audio parameters and generates sound in real-time. The `audio.py` module provides various synth functions (sine wave, theremin with vibrato, FM synthesis, etc.).

A **pipeline** is simply the composition of these components: a selector for which features to extract, which mappings to use, and which synth to employ. The `pipelines.py` module defines several pre-built pipelines (e.g., "theremin", "two_voice", "simple_sine") that you can use or extend.

## Core Components

The architecture is implemented through three primary abstractions that make the system flexible and testable:

### 1. FeatureMapping
Defines how a single video feature maps to an audio parameter:

```python
from theremin.audio_features import FeatureMapping, range_transformer

# Map right wrist X position to frequency
freq_mapping = FeatureMapping(
    audio_param="freq",
    video_feature="r_wrist_position.0", 
    transform=range_transformer((0, 1), (220, 1760)),
    default=440.0
)
```

### 2. AudioFeatureBuilder
Combines multiple mappings to extract audio features:

```python
from theremin.audio_features import AudioFeatureBuilder, create_theremin_builder

# Use pre-built builder
builder = create_theremin_builder()

# Or create custom builder
builder = AudioFeatureBuilder([
    FeatureMapping("freq", "r_wrist_position.0", freq_transform),
    FeatureMapping("volume", "l_wrist_position.1", volume_transform),
])

# Extract features
video_features = {"r_wrist_position": [0.5, 0.3], "l_wrist_position": [0.2, 0.7]}
audio_features = builder(video_features)
# Result: {"freq": 660.0, "volume": 0.3}
```

### 3. AudioPipeline
Complete video→audio→synth pipeline with validation:

```python
from theremin.pipelines import AudioPipeline, ALL_PIPELINES
from theremin.audio import sine_synth

# Create custom pipeline
pipeline = AudioPipeline(
    name="my_custom",
    audio_features=create_theremin_builder(),
    synth=sine_synth
)

# Validate pipeline
issues = pipeline.validate()
if not issues:
    print("Pipeline is valid!")

# Use pipeline
result = pipeline(video_features)

# Or use pre-built pipelines
working_pipelines = ALL_PIPELINES
theremin_pipeline = ALL_PIPELINES["theremin"]
```

## Pre-built Components

### Audio Feature Builders

```python
from theremin.audio_features import (
    create_theremin_builder,      # Basic theremin (freq, volume)
    create_two_hand_builder,      # Independent L/R control
    create_enhanced_theremin_builder,  # With vibrato, attack, release
    create_fallback_theremin_builder,  # Single-hand fallback
)

# Basic theremin
basic = create_theremin_builder()

# Enhanced with all theremin_synth parameters
enhanced = create_enhanced_theremin_builder()

# Two independent voices
two_voice = create_two_hand_builder()
```

### Range Transformers

```python
from theremin.audio_features import range_transformer

# Basic range mapping
freq_transform = range_transformer(
    input_range=(0, 1),
    output_range=(220, 1760)
)

# With pre-processing (invert Y axis)
volume_transform = range_transformer(
    input_range=(0, 1),
    output_range=(0, 1),
    pre_transform=lambda y: 1 - y
)

# With post-processing (quantize to scale)
musical_freq = range_transformer(
    input_range=(0, 1), 
    output_range=(220, 1760),
    post_transform=snap_to_c_major
)
```

### Working Pipelines

Based on the test results, these pipelines are working correctly:

- **`theremin`**: Full theremin with vibrato and envelope controls
- **`enhanced_theremin`**: Same as theremin (currently identical)
- **`simple_sine`**: Basic sine wave with freq/volume only
- **`two_voice`**: Independent left/right hand control
- **`square`**: Square wave with freq/volume
- **`default`**: Alias for theremin

## DAG-based Approach (Optional)

For complex interdependent transformations, you can use the DAG approach:

```python
from theremin.dag_audio_features import (
    theremin_dag_knobs,
    enhanced_theremin_dag_knobs,
    two_voice_dag_knobs
)

# Use like any other audio features function
audio_features = theremin_dag_knobs(video_features)
```

## Testing and Validation

### Unit Testing

```python
from theremin.tests.test_audio_features import *

# Test individual components
test_range_transformer()
test_audio_feature_builder()

# Test with real video data
test_with_sample_video_features()

# Test pipelines
test_pipeline_validation()
test_pipeline_execution()
```

### Pipeline Validation

```python
from theremin.pipelines import validate_all_pipelines, get_working_pipelines

# Check all pipelines
results = validate_all_pipelines()
for name, issues in results.items():
    if issues:
        print(f"{name}: {issues}")

# Get only working pipelines  
working = get_working_pipelines()
print(f"Working pipelines: {list(working.keys())}")
```

## Migration from Old System

The new system is backward compatible. To migrate:

### 1. Replace knobs functions with AudioFeatureBuilder

**Old:**
```python
def my_knobs(video_features):
    knobs = {}
    if 'r_wrist_position' in video_features:
        x = video_features['r_wrist_position'][0]
        knobs['freq'] = 220 + x * (1760 - 220)
    return knobs
```

**New:**
```python
my_builder = AudioFeatureBuilder([
    FeatureMapping("freq", "r_wrist_position.0", 
                   range_transformer((0, 1), (220, 1760)))
])
```

### 2. Use AudioPipeline instead of pipeline dicts

**Old:**
```python
pipeline = {"knobs": my_knobs, "synth": sine_synth}
```

**New:**
```python
pipeline = AudioPipeline("my_pipeline", my_builder, sine_synth)
```

### 3. Validate before using

```python
issues = pipeline.validate()
if not issues:
    # Pipeline is good to use
    result = pipeline(video_features)
```

## Benefits

1. **Clearer Intent**: Mappings are explicit data structures
2. **Better Testing**: Each component can be unit tested
3. **Validation**: Parameter mismatches caught at design time
4. **Reusability**: Components can be mixed and matched
5. **Maintainability**: Less duplicate code, clearer separation of concerns
6. **Flexibility**: Easy to create new pipelines and transformations

## Example: Creating a New Pipeline

```python
from theremin.audio_features import AudioFeatureBuilder, FeatureMapping, range_transformer
from theremin.pipelines import AudioPipeline
from theremin.audio import fm_synth

# Create custom audio feature builder
fm_builder = AudioFeatureBuilder([
    FeatureMapping("freq", "r_wrist_position.0", 
                   range_transformer((0, 1), (220, 1760))),
    FeatureMapping("volume", "l_wrist_position.1",
                   range_transformer((0, 1), (0, 1), pre_transform=lambda y: 1-y)),
    FeatureMapping("mod_index", "r_openness",
                   range_transformer((0, 1), (0, 10))),
    FeatureMapping("carrier_ratio", "l_openness", 
                   range_transformer((0, 1), (0.5, 2.0))),
])

# Create pipeline
fm_pipeline = AudioPipeline("fm_theremin", fm_builder, fm_synth)

# Validate
issues = fm_pipeline.validate()
if not issues:
    print("FM pipeline ready to use!")
    
    # Test with sample data
    video_features = {
        "r_wrist_position": [0.6, 0.4],
        "l_wrist_position": [0.3, 0.8], 
        "r_openness": 0.7,
        "l_openness": 0.5
    }
    
    result = fm_pipeline(video_features)
```

The restructured system makes it much easier to understand, test, and extend the theremin's audio features!
