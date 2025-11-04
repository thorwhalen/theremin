"""
Restructured pipeline system for theremin.

This module provides a cleaner way to define and validate pipelines that go from
video features to audio parameters to synthesized sound.
"""

from dataclasses import dataclass
from typing import Union, Dict, Any, Set, List
from collections.abc import Callable
import inspect
from functools import partial

from i2 import Sig

# Import existing synths and new audio feature systems
from theremin.audio import (
    theremin_synth,
    sine_synth,
    square_synth,
    supersaw_synth,
    noise_synth,
    ringmod_synth,
    chorused_sine_synth,
    fm_synth,
    two_voice_synth_func,
    natural_sounding_synth_lr,
)

from theremin.audio_features import (
    AudioFeatureBuilder,
    FallbackAudioFeatureBuilder,
    create_theremin_builder,
    create_two_hand_builder,
    create_enhanced_theremin_builder,
    create_fallback_theremin_builder,
)

from theremin.dag_audio_features import (
    theremin_dag_knobs,
    enhanced_theremin_dag_knobs,
    two_voice_dag_knobs,
)
from theremin.util import ensure_plain_types


@dataclass
class AudioPipeline:
    """Complete specification of video->audio->synth pipeline"""

    name: str
    audio_features: AudioFeatureBuilder | Callable
    synth: Callable
    video_features: Callable = None  # For future extension

    def validate(self) -> list[str]:
        """
        Validate that audio features match synth parameters.

        Returns:
            List of validation warnings/errors
        """
        issues = []

        try:
            # Get expected synth parameters
            synth_sig = Sig(self.synth)
            # Only check non-keyword-only parameters (those without defaults after *)
            import inspect

            sig = inspect.signature(self.synth)
            required_params = set()
            for name, param in sig.parameters.items():
                if param.kind != param.KEYWORD_ONLY:  # Skip keyword-only params
                    required_params.add(name)

            # Get produced audio features
            if isinstance(self.audio_features, AudioFeatureBuilder):
                produced_params = set(self.audio_features.output_params)
            else:
                # Try to discover parameters by calling with mock data
                mock_video_features = self._create_mock_video_features()
                try:
                    result = self.audio_features(mock_video_features)
                    if isinstance(result, dict):
                        produced_params = set(result.keys())
                    else:
                        issues.append(
                            f"Audio features function returned {type(result)}, expected dict"
                        )
                        produced_params = set()
                except Exception as e:
                    issues.append(f"Could not determine output parameters: {e}")
                    produced_params = set()

            # Check compatibility (only for required params)
            missing = required_params - produced_params
            if missing:
                issues.append(
                    f"Synth expects parameters {missing} not produced by audio_features"
                )

            extra = produced_params - required_params
            if extra:
                issues.append(f"Audio features produces unused parameters {extra}")

        except Exception as e:
            issues.append(f"Validation failed: {e}")

        return issues

    def _create_mock_video_features(self) -> dict[str, Any]:
        """Create mock video features for testing"""
        return {
            'l_wrist_position': [0.5, 0.5],
            'r_wrist_position': [0.5, 0.5],
            'l_openness': 0.5,
            'r_openness': 0.5,
            'l_thumb_index_distance': 0.1,
            'r_thumb_index_distance': 0.1,
        }

    def __call__(self, video_features: dict) -> Any:
        """Execute the full pipeline: video_features -> audio_features -> synth"""
        audio_features = self.audio_features(video_features)
        audio_features = ensure_plain_types(audio_features)

        # Filter to only parameters the synth accepts
        synth_sig = Sig(self.synth)
        filtered_features = {
            k: v for k, v in audio_features.items() if k in synth_sig.names
        }

        return self.synth(**filtered_features)


# --------------------------------------------------------------------------------------
# Pipeline definitions using the new system
# --------------------------------------------------------------------------------------

# Basic theremin pipeline - use enhanced builder for full theremin_synth
THEREMIN_PIPELINE = AudioPipeline(
    name="theremin",
    audio_features=create_enhanced_theremin_builder(),
    synth=theremin_synth,
)

# Enhanced theremin with vibrato and effects - need to add missing params
ENHANCED_THEREMIN_PIPELINE = AudioPipeline(
    name="enhanced_theremin",
    audio_features=create_enhanced_theremin_builder(),
    synth=theremin_synth,
)

# Simple sine wave with basic controls
SIMPLE_SINE_PIPELINE = AudioPipeline(
    name="simple_sine",
    audio_features=create_theremin_builder(freq_trans=lambda x: x),  # No quantization
    synth=sine_synth,
)

# Two independent voices
TWO_VOICE_PIPELINE = AudioPipeline(
    name="two_voice",
    audio_features=create_two_hand_builder(),
    synth=two_voice_synth_func,
)

# Square wave
SQUARE_PIPELINE = AudioPipeline(
    name="square", audio_features=create_theremin_builder(), synth=square_synth
)

# DAG-based pipelines (if meshed is available)
THEREMIN_DAG_PIPELINE = AudioPipeline(
    name="theremin_dag", audio_features=theremin_dag_knobs, synth=theremin_synth
)

ENHANCED_DAG_PIPELINE = AudioPipeline(
    name="enhanced_dag",
    audio_features=enhanced_theremin_dag_knobs,
    synth=theremin_synth,
)

TWO_VOICE_DAG_PIPELINE = AudioPipeline(
    name="two_voice_dag", audio_features=two_voice_dag_knobs, synth=two_voice_synth_func
)


# --------------------------------------------------------------------------------------
# Pipeline registry and validation
# --------------------------------------------------------------------------------------

ALL_PIPELINES = {
    "theremin": THEREMIN_PIPELINE,
    "enhanced_theremin": ENHANCED_THEREMIN_PIPELINE,
    "simple_sine": SIMPLE_SINE_PIPELINE,
    "two_voice": TWO_VOICE_PIPELINE,
    "square": SQUARE_PIPELINE,
    "theremin_dag": THEREMIN_DAG_PIPELINE,
    "enhanced_dag": ENHANCED_DAG_PIPELINE,
    "two_voice_dag": TWO_VOICE_DAG_PIPELINE,
}

# Add default
ALL_PIPELINES["default"] = ALL_PIPELINES["theremin"]


def validate_all_pipelines() -> dict[str, list[str]]:
    """Validate all defined pipelines"""
    results = {}
    for name, pipeline in ALL_PIPELINES.items():
        results[name] = pipeline.validate()
    return results


def get_working_pipelines() -> dict[str, AudioPipeline]:
    """Get only pipelines that pass validation"""
    working = {}
    for name, pipeline in ALL_PIPELINES.items():
        issues = pipeline.validate()
        if not any("expects parameters" in issue for issue in issues):
            working[name] = pipeline
    return working


def list_pipeline_capabilities() -> dict[str, dict[str, Any]]:
    """Get summary of what each pipeline can do"""
    capabilities = {}

    for name, pipeline in ALL_PIPELINES.items():
        # Get synth info
        synth_sig = Sig(pipeline.synth)

        # Get audio features info
        if isinstance(pipeline.audio_features, AudioFeatureBuilder):
            audio_params = pipeline.audio_features.output_params
        else:
            try:
                mock_result = pipeline.audio_features(
                    pipeline._create_mock_video_features()
                )
                audio_params = (
                    list(mock_result.keys()) if isinstance(mock_result, dict) else []
                )
            except:
                audio_params = []

        capabilities[name] = {
            'synth_function': pipeline.synth.__name__,
            'synth_parameters': synth_sig.names,
            'audio_parameters': audio_params,
            'validation_issues': pipeline.validate(),
        }

    return capabilities


# --------------------------------------------------------------------------------------
# Backward compatibility helpers
# --------------------------------------------------------------------------------------


def pipeline_to_knobs_and_synth(pipeline: AudioPipeline) -> dict[str, Callable]:
    """Convert new pipeline to old format for backward compatibility"""
    return {'knobs': pipeline.audio_features, 'synth': pipeline.synth}


def create_legacy_pipelines() -> dict[str, dict[str, Callable]]:
    """Create legacy pipeline format for backward compatibility"""
    legacy = {}
    for name, pipeline in ALL_PIPELINES.items():
        legacy[name] = pipeline_to_knobs_and_synth(pipeline)
    return legacy


# Export legacy format for existing code
legacy_pipelines = create_legacy_pipelines()


# --------------------------------------------------------------------------------------
# Testing utilities
# --------------------------------------------------------------------------------------


def test_pipeline_with_video_features(
    pipeline_name: str, video_features: dict
) -> dict[str, Any]:
    """Test a pipeline with specific video features"""
    if pipeline_name not in ALL_PIPELINES:
        raise ValueError(f"Unknown pipeline: {pipeline_name}")

    pipeline = ALL_PIPELINES[pipeline_name]

    try:
        # Extract audio features
        audio_features = pipeline.audio_features(video_features)

        # Try to run synth (but don't actually play audio)
        synth_sig = Sig(pipeline.synth)
        filtered_features = {
            k: v for k, v in audio_features.items() if k in synth_sig.names
        }

        return {
            'success': True,
            'audio_features': audio_features,
            'synth_params': filtered_features,
            'issues': pipeline.validate(),
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'audio_features': None,
            'synth_params': None,
            'issues': pipeline.validate(),
        }
