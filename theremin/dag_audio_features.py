"""
DAG-based audio feature computation for theremin.

This module provides a way to express audio feature computation as a DAG of functions,
making it easy to compose and test complex video-to-audio mappings.
"""

from typing import Dict, Any, Union
from collections.abc import Callable
import numpy as np
from functools import partial

from meshed import DAG, FuncNode

from theremin.audio import (
    DFLT_MIN_FREQ,
    DFLT_MAX_FREQ,
    snap_to_scale,
    audio_feature_ranges,
)
from theremin.audio_features import range_transformer
from theremin.util import ensure_plain_types


# --------------------------------------------------------------------------------------
# Individual transformation functions for DAG composition
# --------------------------------------------------------------------------------------


def wrist_x_to_freq(
    r_wrist_position=None, min_freq=DFLT_MIN_FREQ, max_freq=DFLT_MAX_FREQ
):
    """Convert right wrist X position to frequency"""
    if r_wrist_position is None:
        return (min_freq + max_freq) / 2
    x = r_wrist_position[0]
    freq = min_freq + x * (max_freq - min_freq)
    return snap_to_scale(freq)


def wrist_y_to_volume(l_wrist_position=None):
    """Convert left wrist Y position to volume"""
    if l_wrist_position is None:
        return 0.0
    y = l_wrist_position[1]
    return float(np.clip(1 - y, 0, 1))


def openness_to_vibrato_rate(
    r_openness=None, max_rate=audio_feature_ranges['vibrato_rate'][1]
):
    """Convert right hand openness to vibrato rate"""
    if r_openness is None:
        return 0.0
    return float(r_openness * max_rate)


def distance_to_vibrato_depth(
    r_thumb_index_distance=None, max_depth=audio_feature_ranges['vibrato_depth'][1]
):
    """Convert right hand thumb-index distance to vibrato depth"""
    if r_thumb_index_distance is None:
        return 0.0
    # Invert: smaller distance = more vibrato
    normalized = 1 - np.clip(r_thumb_index_distance / 0.5, 0, 1)
    return float(normalized * max_depth)


def openness_to_reverb(l_openness=None):
    """Convert left hand openness to reverb mix"""
    if l_openness is None:
        return 0.0
    return float(np.clip(l_openness, 0, 1))


# Two-voice functions
def left_wrist_to_freq(
    l_wrist_position=None, min_freq=DFLT_MIN_FREQ, max_freq=DFLT_MAX_FREQ
):
    """Convert left wrist X position to frequency"""
    if l_wrist_position is None:
        return (min_freq + max_freq) / 2
    x = l_wrist_position[0]
    freq = min_freq + x * (max_freq - min_freq)
    return snap_to_scale(freq)


def left_wrist_to_volume(l_wrist_position=None):
    """Convert left wrist Y position to volume"""
    if l_wrist_position is None:
        return 0.0
    y = l_wrist_position[1]
    return float(np.clip(1 - y, 0, 1))


def right_wrist_to_freq(
    r_wrist_position=None, min_freq=DFLT_MIN_FREQ, max_freq=DFLT_MAX_FREQ
):
    """Convert right wrist X position to frequency"""
    if r_wrist_position is None:
        return (min_freq + max_freq) / 2
    x = r_wrist_position[0]
    freq = min_freq + x * (max_freq - min_freq)
    return snap_to_scale(freq)


def right_wrist_to_volume(r_wrist_position=None):
    """Convert right wrist Y position to volume"""
    if r_wrist_position is None:
        return 0.0
    y = r_wrist_position[1]
    return float(np.clip(1 - y, 0, 1))


# --------------------------------------------------------------------------------------
# DAG-based audio feature extractors
# --------------------------------------------------------------------------------------


def create_theremin_dag():
    """Create DAG for classic theremin audio features"""
    if DAG is None:
        return None

    return DAG(
        [
            FuncNode(wrist_x_to_freq, out='freq'),
            FuncNode(wrist_y_to_volume, out='volume'),
        ]
    )


def create_enhanced_theremin_dag():
    """Create DAG for enhanced theremin with vibrato and reverb"""
    if DAG is None:
        return None

    return DAG(
        [
            FuncNode(wrist_x_to_freq, out='freq'),
            FuncNode(wrist_y_to_volume, out='volume'),
            FuncNode(openness_to_vibrato_rate, out='vibrato_rate'),
            FuncNode(distance_to_vibrato_depth, out='vibrato_depth'),
            # Add simple defaults for missing parameters
            FuncNode(lambda: 0.01, out='attack'),
            FuncNode(lambda: 0.1, out='release'),
        ]
    )


def create_two_voice_dag():
    """Create DAG for two independent voices"""
    if DAG is None:
        return None

    return DAG(
        [
            FuncNode(left_wrist_to_freq, out='l_freq'),
            FuncNode(left_wrist_to_volume, out='l_volume'),
            FuncNode(right_wrist_to_freq, out='r_freq'),
            FuncNode(right_wrist_to_volume, out='r_volume'),
        ]
    )


# --------------------------------------------------------------------------------------
# Fallback DAG wrapper for single-hand operation
# --------------------------------------------------------------------------------------


class FallbackDAG:
    """Wrapper around DAG that provides fallback logic for single-hand operation"""

    def __init__(self, primary_dag, fallback_functions: dict[str, Callable] = None):
        self.primary_dag = primary_dag
        self.fallback_functions = fallback_functions or {}

    def __call__(self, **kwargs) -> dict[str, Any]:
        """Execute DAG with fallback logic"""
        # Try primary DAG first
        try:
            result = self.primary_dag(**kwargs)
        except Exception:
            result = {}

        # Apply fallback logic based on available hands
        has_left = (
            'l_wrist_position' in kwargs and kwargs['l_wrist_position'] is not None
        )
        has_right = (
            'r_wrist_position' in kwargs and kwargs['r_wrist_position'] is not None
        )

        if not has_left and not has_right:
            # No hands - return defaults
            return self._get_defaults()
        elif has_left and has_right:
            # Both hands - primary DAG should work
            return result
        elif has_right and not has_left:
            # Only right hand - use right hand for both freq and volume
            if 'freq' not in result:
                result['freq'] = wrist_x_to_freq(kwargs.get('r_wrist_position'))
            if 'volume' not in result:
                result['volume'] = right_wrist_to_volume(kwargs.get('r_wrist_position'))
        elif has_left and not has_right:
            # Only left hand - use left hand for both freq and volume
            if 'freq' not in result:
                result['freq'] = left_wrist_to_freq(kwargs.get('l_wrist_position'))
            if 'volume' not in result:
                result['volume'] = left_wrist_to_volume(kwargs.get('l_wrist_position'))

        return result

    def _get_defaults(self) -> dict[str, Any]:
        """Get default values when no hands detected"""
        return {
            'freq': (DFLT_MIN_FREQ + DFLT_MAX_FREQ) / 2,
            'volume': 0.0,
            'vibrato_rate': 0.0,
            'vibrato_depth': 0.0,
            'reverb_mix': 0.0,
        }


def create_fallback_theremin_dag():
    """Create theremin DAG with single-hand fallback"""
    primary_dag = create_theremin_dag()
    if primary_dag is None:
        return None
    return FallbackDAG(primary_dag)


# --------------------------------------------------------------------------------------
# Compatibility wrapper to make DAGs work like the old knobs functions
# --------------------------------------------------------------------------------------


def ensure_dag_output_is_dict(dag, out):
    """Normalize DAG output to a dict with meaningful keys when possible.

    - If ``out`` is already a dict, return as-is.
    - If the DAG exposes output names via ``dag.out`` (list/tuple), zip them.
    - Otherwise, fall back to common shapes:
      2-tuple -> ['freq', 'volume']
      4-tuple -> ['l_freq', 'l_volume', 'r_freq', 'r_volume']
    - If no keys can be inferred, return the original ``out``.
    """
    if isinstance(out, dict):
        return out

    keys = None
    if hasattr(dag, 'out') and isinstance(getattr(dag, 'out'), (list, tuple)):
        keys = list(getattr(dag, 'out'))

    if keys is None and isinstance(out, (list, tuple)):
        if len(out) == 2:
            keys = ['freq', 'volume']
        elif len(out) == 4:
            keys = ['l_freq', 'l_volume', 'r_freq', 'r_volume']

    if isinstance(out, (list, tuple)) and keys:
        return dict(zip(keys, out))

    return out


def dag_to_knobs_function(dag) -> Callable:
    """Convert a DAG to a function that matches the old knobs function signature"""

    def knobs_function(video_features: dict) -> dict[str, float]:
        """Extract audio features from video features using DAG"""
        if not video_features:
            if hasattr(dag, '_get_defaults'):
                return dag._get_defaults()
            else:
                return {'freq': (DFLT_MIN_FREQ + DFLT_MAX_FREQ) / 2, 'volume': 0.0}

        # Call the DAG with video features as keyword arguments
        out = dag(**video_features)
        out = ensure_dag_output_is_dict(dag, out)
        return ensure_plain_types(out)

    return knobs_function


# --------------------------------------------------------------------------------------
# Pre-built DAG-based knobs functions for common patterns
# --------------------------------------------------------------------------------------

# Create the DAG-based knobs functions if meshed is available
if DAG is not None:
    theremin_dag_knobs = dag_to_knobs_function(create_fallback_theremin_dag())
    enhanced_theremin_dag_knobs = dag_to_knobs_function(create_enhanced_theremin_dag())
    two_voice_dag_knobs = dag_to_knobs_function(create_two_voice_dag())
else:
    # Fallback implementations without DAG
    def theremin_dag_knobs(video_features: dict) -> dict[str, float]:
        """Fallback theremin knobs without DAG"""
        result = {}
        result['freq'] = wrist_x_to_freq(video_features.get('r_wrist_position'))
        result['volume'] = wrist_y_to_volume(video_features.get('l_wrist_position'))

        # Single hand fallbacks
        if video_features.get('r_wrist_position') and not video_features.get(
            'l_wrist_position'
        ):
            result['volume'] = right_wrist_to_volume(
                video_features.get('r_wrist_position')
            )
        elif video_features.get('l_wrist_position') and not video_features.get(
            'r_wrist_position'
        ):
            result['freq'] = left_wrist_to_freq(video_features.get('l_wrist_position'))

        return result

    def enhanced_theremin_dag_knobs(video_features: dict) -> dict[str, float]:
        """Fallback enhanced theremin knobs without DAG"""
        result = theremin_dag_knobs(video_features)
        result['vibrato_rate'] = openness_to_vibrato_rate(
            video_features.get('r_openness')
        )
        result['vibrato_depth'] = distance_to_vibrato_depth(
            video_features.get('r_thumb_index_distance')
        )
        result['attack'] = 0.01
        result['release'] = 0.1
        return result

    def two_voice_dag_knobs(video_features: dict) -> dict[str, float]:
        """Fallback two-voice knobs without DAG"""
        return {
            'l_freq': left_wrist_to_freq(video_features.get('l_wrist_position')),
            'l_volume': left_wrist_to_volume(video_features.get('l_wrist_position')),
            'r_freq': right_wrist_to_freq(video_features.get('r_wrist_position')),
            'r_volume': right_wrist_to_volume(video_features.get('r_wrist_position')),
        }
