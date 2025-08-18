"""Theremin POC using Slabs + DAGs.

This module wires a webcam -> hand features -> audio feature DAG -> synth
using the Slabs pattern where function names determine slab keys and
arguments determine their dependencies.

Key ideas:
- Functions are simple, reusable components. Names map to slab keys.
- Arguments declare dependencies (source keys) in the slab.
- DAG is used to compute audio features (knobs) from video features.
- A synth is updated with the resulting knobs each frame.

Inspired by:
- meshed.slabs.Slabs (see i/meshed/meshed/slabs.py)
- theremin/scrap/example_w_slabs.py
- theremin/dag_audio_features.py
"""

from typing import Dict, Any, Callable, Optional

import cv2

# Video features (MediaPipe-based)
from theremin.video_features import HandGestureRecognizer, many_video_features

# DAG-based audio features
from theremin.dag_audio_features import (
    create_theremin_dag,
    create_enhanced_theremin_dag,
    create_two_voice_dag,
    ensure_dag_output_is_dict,
)

# Synths
from theremin.audio import theremin_synth, two_voice_synth_func
from hum import Synth

# Slabs utilities
from meshed import Slabs
from meshed.slabs import output_none_if_none_arguments
from theremin.util import ensure_plain_types


# --------------------------------------------------------------------------------------
# Component factory
# --------------------------------------------------------------------------------------


def _make_audio_knobs_func(
    variant: str,
) -> Callable[[Dict[str, Any]], Dict[str, float]]:
    """Return a DAG-backed knobs function according to variant.

    Variants:
    - "theremin": freq + volume
    - "enhanced": theremin + vibrato, etc.
    - "two_voice": l_/r_ freq + volume
    """

    if variant == "theremin":
        dag = create_theremin_dag()
    elif variant == "enhanced":
        dag = create_enhanced_theremin_dag()
    elif variant == "two_voice":
        dag = create_two_voice_dag()
    else:
        raise ValueError(f"Unknown audio DAG variant: {variant}")

    # Build a knobs function that filters inputs to what the DAG expects
    from inspect import signature

    try:
        allowed_params = set(signature(dag).parameters.keys())
    except Exception:
        # Fallback: allow all, the DAG will sort itself out
        allowed_params = None

    def knobs_function(video_features: Dict[str, Any]) -> Dict[str, float]:
        # If nothing, provide reasonable defaults
        if not video_features:
            if hasattr(dag, "_get_defaults"):
                return dag._get_defaults()  # type: ignore[attr-defined]
            else:
                from theremin.audio import DFLT_MIN_FREQ, DFLT_MAX_FREQ

                return {"freq": (DFLT_MIN_FREQ + DFLT_MAX_FREQ) / 2, "volume": 0.0}

        if allowed_params is not None:
            kwargs = {k: v for k, v in video_features.items() if k in allowed_params}
        else:
            kwargs = video_features

        out = dag(**kwargs)
        out = ensure_dag_output_is_dict(dag, out)
        return ensure_plain_types(out)

    return knobs_function


def _select_synth(variant: str) -> Callable:
    """Choose a synth function compatible with the selected DAG outputs."""
    if variant in {"theremin", "enhanced"}:
        return theremin_synth
    elif variant == "two_voice":
        return two_voice_synth_func
    else:
        raise ValueError(f"Unknown synth variant: {variant}")


def create_theremin_slabs(cap: cv2.VideoCapture, *, variant: str = "theremin") -> Slabs:
    """Create a Slabs instance that runs the theremin pipeline.

    Args:
            cap: An initialized cv2.VideoCapture
            variant: One of {"theremin", "enhanced", "two_voice"}
    """

    detector = HandGestureRecognizer()
    knobs_func = _make_audio_knobs_func(variant)
    synth_func = _select_synth(variant)

    # Manage Synth lifecycle outside component calls (we'll pass the instance via closure)
    synth_instance: Optional[Synth] = None

    # -------------------------- Components (by slab key name) -------------------------

    def video_frame():
        """Capture and flip a frame. Returns None if capture fails."""
        success, img = cap.read()
        if not success:
            return None
        # Mirror for natural control
        return cv2.flip(img, 1)

    @output_none_if_none_arguments
    def hand_detection(video_frame):
        """Detect hands using MediaPipe."""
        return detector.find_hands(video_frame)

    @output_none_if_none_arguments
    def video_features(hand_detection):
        """Extract left/right-prefixed features from hand detection."""
        return many_video_features(hand_detection)

    def audio_features(video_features: Dict[str, Any]):
        """Compute audio knobs from video features using a DAG-backed function."""
        if not video_features:
            return {}
        return knobs_func(video_features)

    def init_synth():
        """Initialize and return a running Synth instance (once)."""
        nonlocal synth_instance
        if synth_instance is None:
            synth_instance = Synth(synth_func)
            synth_instance.start()
        return synth_instance

    @output_none_if_none_arguments
    def apply_knobs(audio_features, init_synth):
        """Apply computed knobs to the Synth in real-time."""
        if audio_features:
            init_synth(**audio_features)

    def annotated_image(video_frame, audio_features):
        """Draw simple debug text with a couple of knobs."""
        if video_frame is None:
            return None
        if audio_features:
            text = ", ".join(
                f"{k}={v:.2f}"
                for k, v in audio_features.items()
                if isinstance(v, (int, float))
            )
            cv2.putText(
                video_frame,
                text[:120],
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        return video_frame

    def display_image(annotated_image):
        if annotated_image is not None:
            cv2.imshow("Theremin (Slabs+DAG)", annotated_image)

    def check_quit():
        if cv2.waitKey(1) & 0xFF == ord('q'):
            raise KeyboardInterrupt()

    handle_exceptions = {
        KeyboardInterrupt: lambda: print("Quit: Keyboard interrupt (q pressed)."),
    }

    components = [
        video_frame,
        hand_detection,
        video_features,
        audio_features,
        init_synth,
        apply_knobs,
        annotated_image,
        display_image,
        check_quit,
    ]

    slabs = Slabs.from_func_nodes(components, handle_exceptions=handle_exceptions)

    # Attach a cleanup hook to release resources when slabs exits
    # We'll piggyback on slabs.__exit__ via a small wrapper
    original_close = slabs.close

    def _close_with_cleanup(exc_type=None, exc_val=None, exc_tb=None):
        try:
            cv2.destroyAllWindows()
        finally:
            try:
                if synth_instance is not None:
                    synth_instance.stop()
            finally:
                return original_close(exc_type, exc_val, exc_tb)

    slabs.close = _close_with_cleanup  # type: ignore[attr-defined]
    return slabs


# --------------------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------------------


def run(variant: str = "theremin", camera_index: int = 0):
    """Run the Slabs+DAG theremin.

    - Press 'q' to quit.
    - Variants: "theremin", "enhanced", "two_voice".
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Error: Could not open video capture device.")
        return

    try:
        slabs = create_theremin_slabs(cap, variant=variant)
        slabs.run()
    finally:
        cap.release()


if __name__ == "__main__":
    run()
