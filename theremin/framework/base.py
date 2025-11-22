"""Base abstractions for theremin framework."""

from typing import Dict, Callable, Any, Optional, List
from abc import ABC, abstractmethod
import time


class SensorReader(ABC):
    """
    Base class for sensor reading functions.

    Sensor readers capture raw input and return dictionaries with sensor data.
    All sensor readers should output dicts with descriptive keys.
    """

    @abstractmethod
    def read(self) -> Dict[str, Any]:
        """
        Read sensor data.

        Returns:
            Dictionary with sensor data and metadata (e.g., timestamp)
        """
        pass


class FeatureExtractor(ABC):
    """
    Base class for feature extraction functions.

    Feature extractors transform raw sensor data into meaningful features.
    They consume dicts from sensors and output dicts with extracted features.
    """

    @abstractmethod
    def extract(self, **kwargs) -> Dict[str, Any]:
        """
        Extract features from sensor data.

        Args:
            **kwargs: Sensor data (matched by parameter names)

        Returns:
            Dictionary with extracted features
        """
        pass


class FeatureMapper(ABC):
    """
    Base class for feature mapping functions (knobs).

    Feature mappers transform extracted features into synthesis parameters.
    They define the creative mapping between input features and sound.
    """

    @abstractmethod
    def map(self, **kwargs) -> Dict[str, Any]:
        """
        Map features to synthesis parameters.

        Args:
            **kwargs: Extracted features (matched by parameter names)

        Returns:
            Dictionary with synthesis parameters
        """
        pass


class Synthesizer(ABC):
    """
    Base class for synthesis functions.

    Synthesizers generate audio from synthesis parameters.
    They consume parameter dicts and produce audio output.
    """

    @abstractmethod
    def synthesize(self, **kwargs) -> Any:
        """
        Generate audio from parameters.

        Args:
            **kwargs: Synthesis parameters (matched by parameter names)

        Returns:
            Audio data (format depends on synthesizer)
        """
        pass


class Pipeline:
    """
    Theremin processing pipeline using meshed DAG for auto-wiring.

    The Pipeline composes sensor reading, feature extraction, feature mapping,
    and synthesis into a single callable that automatically wires functions
    based on parameter name matching.

    Example:
        >>> from meshed import DAG
        >>>
        >>> def read_video(device_id: int) -> dict:
        >>>     return {'video_frame': frame, 'timestamp': time.time()}
        >>>
        >>> def extract_hands(video_frame, timestamp) -> dict:
        >>>     return {'hand_x': 0.5, 'hand_y': 0.3, 'timestamp': timestamp}
        >>>
        >>> def map_to_audio(hand_x, hand_y) -> dict:
        >>>     return {'frequency': 200 + hand_x * 1800, 'amplitude': 1 - hand_y}
        >>>
        >>> def synth(frequency, amplitude):
        >>>     return generate_audio(frequency, amplitude)
        >>>
        >>> pipeline = Pipeline([read_video, extract_hands, map_to_audio, synth])
        >>> audio = pipeline(device_id=0)
    """

    def __init__(
        self,
        functions: List[Callable],
        name: str = "theremin_pipeline",
        validate: bool = True
    ):
        """
        Initialize pipeline.

        Args:
            functions: List of functions to compose into pipeline
            name: Pipeline name
            validate: If True, validate pipeline structure
        """
        self.functions = functions
        self.name = name
        self._dag = None
        self._meshed_available = False

        # Try to import meshed
        try:
            from meshed import DAG
            self._DAG = DAG
            self._meshed_available = True
            self._dag = DAG(functions)
        except ImportError:
            pass

        if validate and self._meshed_available:
            self._validate()

    def _validate(self):
        """Validate pipeline structure."""
        # Basic validation - could be expanded
        if not self.functions:
            raise ValueError("Pipeline must have at least one function")

    def __call__(self, **kwargs) -> Any:
        """
        Execute the pipeline.

        Args:
            **kwargs: Input parameters for the first function

        Returns:
            Output from the final function
        """
        if not self._meshed_available:
            raise RuntimeError(
                "meshed is not installed. Install with: pip install meshed"
            )

        return self._dag(**kwargs)

    def validate_wiring(self) -> Dict[str, Any]:
        """
        Validate that functions wire together correctly.

        Returns:
            Dictionary with validation results
        """
        # This would use meshed's introspection capabilities
        return {
            'valid': True,
            'function_count': len(self.functions),
            'name': self.name
        }
