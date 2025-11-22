"""Extract features from test videos and save as JSON fixtures."""

import cv2
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from theremin.video_features import HandGestureRecognizer, many_video_features


def extract_features_from_video(video_path: str, output_path: str = None):
    """
    Extract features from a video file and save as JSON.

    Args:
        video_path: Path to the video file
        output_path: Path to save the JSON file (optional)

    Returns:
        list: List of feature dictionaries for each frame
    """
    if output_path is None:
        output_path = str(Path(video_path).with_suffix('.json'))

    recognizer = HandGestureRecognizer()
    feature_recorder = []

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Processing video: {video_path}")
    print(f"FPS: {fps}, Total frames: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Extract hand detection
        hand_detection = recognizer.find_hands(frame)

        # Extract features
        features = many_video_features(hand_detection)

        # Add metadata
        features['frame_number'] = frame_count
        features['timestamp'] = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        features['has_hands'] = bool(features)

        feature_recorder.append(features)
        frame_count += 1

        if frame_count % 30 == 0:
            print(f"  Processed {frame_count} frames...")

    cap.release()

    print(f"Extracted features from {frame_count} frames")
    print(f"Frames with hands detected: {sum(1 for f in feature_recorder if f.get('has_hands'))}")

    # Save to JSON
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(feature_recorder, f, indent=2)

    print(f"Saved features to: {output_path}")

    return feature_recorder


def main():
    """Extract features from all test videos."""
    test_data_dir = Path(__file__).parent / 'testing_data'

    # Find all video files
    video_files = list(test_data_dir.glob('*.mp4'))

    if not video_files:
        print(f"No video files found in {test_data_dir}")
        return

    print(f"Found {len(video_files)} video files\n")

    for video_path in sorted(video_files):
        # Skip if JSON already exists
        json_path = video_path.with_name(f"{video_path.stem}__video_features.json")

        if json_path.exists():
            print(f"Skipping {video_path.name} (JSON already exists)")
            continue

        try:
            extract_features_from_video(str(video_path), str(json_path))
            print()
        except Exception as e:
            print(f"Error processing {video_path.name}: {e}")
            print()


if __name__ == '__main__':
    main()
