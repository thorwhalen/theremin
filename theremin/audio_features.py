"""
Restructured audio features system for theremin.

This module provides a cleaner, more composable way to map video features to audio parameters.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Union, List, Tuple, Optional
from collections.abc import Callable
import numpy as np
from functools import partial

# Import existing constants
from theremin.audio import (
    DFLT_MIN_FREQ,
    DFLT_MAX_FREQ,
    audio_feature_ranges,
    snap_to_scale,
    identity,
)
from theremin.video_features import video_feature_ranges
from theremin.util import ensure_plain_types


@dataclass
class FeatureMapping:
    """Defines how a video feature maps to an audio parameter"""

    audio_param: str
    video_feature: str
    transform: Callable[[Any], float] = identity
    default: float = 0.0

    def __post_init__(self):
        """Validate that the mapping references valid features and parameters"""
        # Remove l_/r_ prefix for validation if present
        audio_param_key = self.audio_param
        if audio_param_key.startswith(("l_", "r_")):
            audio_param_key = audio_param_key[2:]

        if audio_param_key not in audio_feature_ranges:
            print(f"Warning: Audio parameter '{self.audio_param}' not in known ranges")


def range_transformer(
    input_range: tuple[float, float] = (0, 1),
    output_range: tuple[float, float] = (220, 1760),
    pre_transform: Callable = identity,
    post_transform: Callable = identity,
    clip: bool = True,
) -> Callable[[float], float]:
    """
    Create a function that maps input range to output range with optional transformations.

    Args:
        input_range: Expected input value range (min, max)
        output_range: Desired output range (min, max)
        pre_transform: Function to apply to input before range mapping (e.g., lambda x: 1-x)
        post_transform: Function to apply after range mapping (e.g., snap_to_c_major)
        clip: Whether to clip values to the output range

    Returns:
        A function that transforms input values to the output range
    """
    input_min, input_max = input_range
    output_min, output_max = output_range
    input_span = input_max - input_min
    output_span = output_max - output_min

    def transformer(value):
        if value is None:
            return (output_min + output_max) / 2  # Return midpoint for None values

        # Apply pre-transformation
        value = pre_transform(value)

        # Normalize to 0-1 range
        normalized = (value - input_min) / input_span

        # Clip if requested
        if clip:
            normalized = np.clip(normalized, 0, 1)

        # Map to output range
        result = output_min + normalized * output_span

        # Apply post-transformation and ensure builtin float
        result = post_transform(result)
        try:
            return float(result)
        except Exception:
            return result

    return transformer


def extract_nested_value(data: dict, path: str) -> Any:
    """
    Extract nested values from a dictionary using dot notation.

    Examples:
        extract_nested_value({'r_wrist_position': [0.5, 0.3]}, 'r_wrist_position.0') -> 0.5
        extract_nested_value({'r_wrist_position': [0.5, 0.3]}, 'r_wrist_position.1') -> 0.3
    """
    parts = path.split('.')
    value = data

    for part in parts:
        if isinstance(value, (list, tuple)) and part.isdigit():
            value = value[int(part)]
        elif isinstance(value, dict):
            value = value[part]
        else:
            raise ValueError(f"Cannot access '{part}' on {type(value)}")

    return value


class AudioFeatureBuilder:
    """Builds audio features from video features using mappings and transformations"""

    def __init__(self, mappings: list[FeatureMapping]):
        self.mappings = mappings

    def __call__(self, video_features: dict) -> dict[str, float]:
        """Extract audio features from video features"""
        audio_features = {}

        for mapping in self.mappings:
            try:
                raw_value = extract_nested_value(video_features, mapping.video_feature)
                audio_features[mapping.audio_param] = mapping.transform(raw_value)
            except (KeyError, TypeError, IndexError, ValueError):
                audio_features[mapping.audio_param] = mapping.default

        return ensure_plain_types(audio_features)

    @property
    def output_params(self) -> list[str]:
        """Get list of audio parameters this builder produces"""
        return [mapping.audio_param for mapping in self.mappings]


# --------------------------------------------------------------------------------------
# Pre-defined transformers for common mappings
# --------------------------------------------------------------------------------------

# Wrist position transformers
wrist_x_to_freq = range_transformer(
    input_range=(0, 1),
    output_range=(DFLT_MIN_FREQ, DFLT_MAX_FREQ),
    post_transform=snap_to_scale,
)

wrist_y_to_volume = range_transformer(
    input_range=(0, 1),
    output_range=(0, 1),
    pre_transform=lambda y: 1 - y,  # Invert Y axis (higher position = louder)
)

# Hand openness transformers
openness_to_vibrato_rate = range_transformer(
    input_range=(0, 1), output_range=audio_feature_ranges['vibrato_rate']
)

openness_to_vibrato_depth = range_transformer(
    input_range=(0, 1), output_range=audio_feature_ranges['vibrato_depth']
)

# Distance-based transformers
distance_to_reverb = range_transformer(
    input_range=video_feature_ranges['thumb_index_distance'],
    output_range=audio_feature_ranges['reverb_mix'],
)


# --------------------------------------------------------------------------------------
# Pre-defined audio feature builders for common patterns
# --------------------------------------------------------------------------------------


def create_theremin_builder(
    min_freq: float = DFLT_MIN_FREQ,
    max_freq: float = DFLT_MAX_FREQ,
    freq_trans: Callable = snap_to_scale,
) -> AudioFeatureBuilder:
    """Create an audio feature builder for classic theremin control (right hand freq, left hand volume)"""

    freq_transformer = range_transformer(
        input_range=(0, 1),
        output_range=(min_freq, max_freq),
        post_transform=freq_trans,
    )

    return AudioFeatureBuilder(
        [
            FeatureMapping(
                "freq",
                "r_wrist_position.0",
                freq_transformer,
                (min_freq + max_freq) / 2,
            ),
            FeatureMapping("volume", "l_wrist_position.1", wrist_y_to_volume, 0.0),
        ]
    )


def create_two_hand_builder(
    min_freq: float = DFLT_MIN_FREQ,
    max_freq: float = DFLT_MAX_FREQ,
    freq_trans: Callable = snap_to_scale,
) -> AudioFeatureBuilder:
    """Create builder for independent left/right hand control"""

    freq_transformer = range_transformer(
        input_range=(0, 1),
        output_range=(min_freq, max_freq),
        post_transform=freq_trans,
    )

    return AudioFeatureBuilder(
        [
            FeatureMapping(
                "l_freq",
                "l_wrist_position.0",
                freq_transformer,
                (min_freq + max_freq) / 2,
            ),
            FeatureMapping("l_volume", "l_wrist_position.1", wrist_y_to_volume, 0.0),
            FeatureMapping(
                "r_freq",
                "r_wrist_position.0",
                freq_transformer,
                (min_freq + max_freq) / 2,
            ),
            FeatureMapping("r_volume", "r_wrist_position.1", wrist_y_to_volume, 0.0),
        ]
    )


def create_enhanced_theremin_builder(
    min_freq: float = DFLT_MIN_FREQ,
    max_freq: float = DFLT_MAX_FREQ,
    freq_trans: Callable = snap_to_scale,
) -> AudioFeatureBuilder:
    """Create enhanced theremin with vibrato and reverb controls"""

    freq_transformer = range_transformer(
        input_range=(0, 1),
        output_range=(min_freq, max_freq),
        post_transform=freq_trans,
    )

    return AudioFeatureBuilder(
        [
            FeatureMapping(
                "freq",
                "r_wrist_position.0",
                freq_transformer,
                (min_freq + max_freq) / 2,
            ),
            FeatureMapping("volume", "l_wrist_position.1", wrist_y_to_volume, 0.0),
            FeatureMapping("vibrato_rate", "r_openness", openness_to_vibrato_rate, 5.0),
            FeatureMapping(
                "vibrato_depth",
                "r_thumb_index_distance",
                openness_to_vibrato_depth,
                5.0,
            ),
            FeatureMapping(
                "attack", "l_openness", range_transformer((0, 1), (0.001, 0.1)), 0.01
            ),
            FeatureMapping(
                "release",
                "l_thumb_index_distance",
                range_transformer((0, 1), (0.01, 0.5)),
                0.1,
            ),
            # waveform is a keyword-only argument, so it doesn't need to be provided by the builder
        ]
    )


# --------------------------------------------------------------------------------------
# Fallback behavior for single-hand operation
# --------------------------------------------------------------------------------------


class FallbackAudioFeatureBuilder(AudioFeatureBuilder):
    """Audio feature builder with fallback logic for single-hand operation"""

    def __init__(
        self,
        mappings: list[FeatureMapping],
        fallback_mappings: list[FeatureMapping] = None,
    ):
        super().__init__(mappings)
        self.fallback_mappings = fallback_mappings or []

    def __call__(self, video_features: dict) -> dict[str, float]:
        # Try primary mappings first
        audio_features = super().__call__(video_features)

        # If we have both hands, we're done
        if (
            'r_wrist_position' in video_features
            and 'l_wrist_position' in video_features
        ):
            return audio_features

        # Apply fallback logic for single-hand operation
        for mapping in self.fallback_mappings:
            try:
                raw_value = extract_nested_value(video_features, mapping.video_feature)
                audio_features[mapping.audio_param] = mapping.transform(raw_value)
            except (KeyError, TypeError, IndexError, ValueError):
                audio_features[mapping.audio_param] = mapping.default

        return audio_features


def create_fallback_theremin_builder(
    min_freq: float = DFLT_MIN_FREQ,
    max_freq: float = DFLT_MAX_FREQ,
    freq_trans: Callable = snap_to_scale,
) -> FallbackAudioFeatureBuilder:
    """Create theremin builder with single-hand fallback behavior"""

    freq_transformer = range_transformer(
        input_range=(0, 1),
        output_range=(min_freq, max_freq),
        post_transform=freq_trans,
    )

    # Primary mappings for two-hand operation
    primary_mappings = [
        FeatureMapping(
            "freq", "r_wrist_position.0", freq_transformer, (min_freq + max_freq) / 2
        ),
        FeatureMapping("volume", "l_wrist_position.1", wrist_y_to_volume, 0.0),
    ]

    # Fallback mappings for single-hand operation
    fallback_mappings = [
        # If only right hand, use Y for volume
        FeatureMapping("volume", "r_wrist_position.1", wrist_y_to_volume, 0.0),
        # If only left hand, use X for freq and Y for volume
        FeatureMapping(
            "freq", "l_wrist_position.0", freq_transformer, (min_freq + max_freq) / 2
        ),
    ]

    return FallbackAudioFeatureBuilder(primary_mappings, fallback_mappings)
